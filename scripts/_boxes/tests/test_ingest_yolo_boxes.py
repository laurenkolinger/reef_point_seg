#!/usr/bin/env python
"""Generic YOLO box-dataset stager: names, layout, refusals, and the command line.

No-pytest house harness. Every fixture is a synthetic YOLO source tree built in
a temp dir and removed in a finally block: no test here reads a real dataset,
and nothing is written outside the temp dirs.

Covers:
  names       safe_basename over spaces, a .jfif extension, an upper-case
              extension and a non-ascii character
  readers     read_names_csv (header, empty file), read_class_list (mapping,
              list, a gap in the ids, an nc that disagrees)
  layout      images/ labels/ classes.txt data.yaml names.csv split_pins.json
              rename_map.csv LICENSE.md ingest_manifest.json
  manifest    the Coral Intel keys plus source_kind, labels_source, names_csv,
              split_pins, n_pinned, pins_by_split, n_renamed, n_background,
              n_duplicate_lines, subject
  subject     --subject is checked against the vocabulary the Train panel
              reads, recorded in the manifest, refused when it is a word the
              panel does not know, and absent from the run entirely when it
              is not given
  pins        split_pins.json keyed by the staged basename, one pin per image,
              a split tree covering only part of the dataset counted honestly,
              and the three refusals (missing split folder, a pin for an image
              that is not there, an image pinned twice)
  symlinks    a source image that is a symlink is staged as a real file
  hostile     a label file, a data.yaml and a names table holding bytes that
              are not text, a negative record cap and a record cap that is
              not a number
  refusals    a class id outside the class list, a four-field label line, a
              polygon label line, a basename collision, two records sharing a
              staged label stem, a label with no image, a class token with no
              full name, a destination name outside the name rule, and that
              nothing is staged after any of them
  case        two, three and label-only clashes that differ from another name
              only in letter case are disambiguated with a numbered suffix,
              the suffix steps over a name a third file owns, the names are
              the same on a second staging of the same folder, a split pin
              follows the disambiguated name, a re-stage over a name already
              staged in the other case is refused, a dataset folder matching a
              sibling folder only by case is refused, a suffix that would pass
              the 255-byte file-name limit is refused before anything is
              copied while a long name with no twin still stages, and no
              staged tree this suite writes holds a pair differing only by
              case
  leftovers   a destination already holding files this run would not write, a
              destination holding a Coral Intel labels_localized/ folder, and
              a stale names.csv or split_pins.json
  sidecars    a classes.txt beside the label files is passed over, not read as
              a label whose image is missing
  unwritable  a destination that is a file, and a destination folder with no
              write permission, each refused with the house message
  limit       --limit stages a prefix and the manifest records it
  idempotent  a second run over the same destination copies no file
  huge        a 10 MB label file is staged and its boxes counted
  cli         the command line writes the manifest, carries --citation and
              --license into it, falls back to "none recorded" without them,
              and exits 2 with "ingest stopped:" on a refusal
  cleanup     every temp fixture this suite made is gone, and nothing was
              staged into the module's own bbox_datasets folder

Run with:
  env/bin/python scripts/_boxes/tests/test_ingest_yolo_boxes.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _boxes import ingest_yolo_boxes as ing  # noqa: E402

# The stager under test, and the interpreter the module's suites run under.
# The repo's own env is preferred so a CLI test never runs under a foreign
# interpreter; sys.executable covers a checkout whose env is not built yet.
SCRIPT = os.path.join(_SCRIPTS, "_boxes", "ingest_yolo_boxes.py")
_ENV_PYTHON = os.path.join(_REPO, "env", "bin", "python")
PYTHON = _ENV_PYTHON if os.path.isfile(_ENV_PYTHON) else sys.executable

# Roughly 10 MB of distinct label lines (23 bytes each), the adversarial
# "huge input" case the build discipline asks for.
HUGE_LINES = 460000

# Every temp fixture this suite makes carries this prefix, so the closing
# cleanup test can prove each one was removed.
TEMP_PREFIX = "yolo_"

# Two label bodies the case tests plant, one per image of a clashing pair, so
# a staged label file can be traced back to the image it belongs to.
ONE_BOX = "0 0.5 0.5 0.2 0.2\n"
TWO_BOXES = "1 0.3 0.3 0.1 0.1\n1 0.6 0.6 0.1 0.1\n"

# A source stem long enough that adding the two-character case suffix pushes
# the staged name one byte past the 255-byte file-name limit: 250 + "_2.png"
# is 256. The source names themselves fit, so the fixture can be written.
LONG_STEM_CHARS = 250

# Dataset folder names this suite stages. The closing cleanup test proves none
# of them appears in the module's own supporting_data/bbox_datasets.
STAGED_NAMES = ("staged", "small", "second", "plain_set", "Plain_Set",
                "cited_set", "names_only", "no_source", "bad",
                "small_coral", "small_fish", "small_algae")

_pass = 0
_fail = 0

# Absolute paths of the temp directories _mkdtemp handed out this run.
_TEMP_DIRS = []


def check(cond, msg):
    global _pass, _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if cond:
        _pass += 1
    else:
        _fail += 1


def _mkdtemp(tag):
    """A temp directory for one test, recorded so the cleanup test can check it.

    Parameters:
      tag: a short word naming the test, for example "stage".
    Returns:
      the absolute path of the new directory.
    """
    path = tempfile.mkdtemp(prefix=TEMP_PREFIX + tag + "_")
    _TEMP_DIRS.append(path)
    return path


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def write_jpeg(path, size=(32, 32), color=(20, 90, 140)):
    """A real (tiny) JPEG, so a staged copy can be opened by a later reader."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


SOURCE_NAMES = [
    "Fish A_1.jpeg",
    "fish-b_2.png",
    "Screen Shot 2023-07-27 at 10.07.40 AM.png",
    "odd.jfif",
    "plain_5.jpg",
    "unicode_ñ_6.jpg",
]


def make_source(root, n=6, with_split=True):
    """A synthetic YOLO source tree.

    Writes images/ (one name with spaces, one .jfif, one non-ascii name), a
    labels/ folder whose odd-indexed records carry two boxes and whose
    even-indexed records carry an empty file, a two-class data.yaml, a names
    CSV, and (by default) a train/valid/test split tree holding the original
    file names.

    Parameters:
      root: temp directory the tree is built in.
      n: how many of SOURCE_NAMES to write (1 to 6).
      with_split: write the split tree as well.
    Returns:
      (images_dir, labels_dir).
    """
    img = os.path.join(root, "images")
    lab = os.path.join(root, "labels")
    names = SOURCE_NAMES[:n]
    for i, nm in enumerate(names):
        write_jpeg(os.path.join(img, nm))
        write(os.path.join(lab, os.path.splitext(nm)[0] + ".txt"),
              "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n" if i % 2 else "")
    write(os.path.join(root, "data.yaml"),
          "names:\n  0: AAAA_Normal\n  1: BBBB_TP\nnc: 2\n")
    write(os.path.join(root, "names.csv"),
          'token,name\nAAAA_Normal,"Genus a, fish a"\n'
          'BBBB_TP,"Genus b, fish b, terminal phase"\n')
    if with_split:
        subsets = (("train", names[:max(1, n - 2)]),
                   ("valid", names[max(1, n - 2):n - 1]),
                   ("test", names[n - 1:]))
        for split, subset in subsets:
            d = os.path.join(root, "split", split, "images")
            os.makedirs(d, exist_ok=True)
            for nm in subset:
                write(os.path.join(d, nm), "x")
    return img, lab


def fixture_classes():
    """The two class tokens the synthetic data.yaml declares."""
    return ["AAAA_Normal", "BBBB_TP"]


def fixture_names():
    """A names table covering both fixture tokens."""
    return [("AAAA_Normal", "Genus a, fish a"),
            ("BBBB_TP", "Genus b, fish b, terminal phase")]


def quiet(_msg):
    """Swallow the stager's progress lines inside a test."""


# ── names ────────────────────────────────────────────────────────────────

def test_safe_basename():
    check(ing.safe_basename("Screen Shot 2023-07-27 at 10.07.40 AM.png")
          == "Screen_Shot_2023-07-27_at_10.07.40_AM.png",
          "spaces become one underscore")
    check(ing.safe_basename("odd.jfif") == "odd.jpg",
          "jfif is renamed to jpg")
    check(ing.safe_basename("015.JPG") == "015.jpg",
          "extension lower-cased, stem kept")
    check(ing.safe_basename("unicode_ñ_6.jpg") == "unicode_6.jpg",
          "non-ascii dropped to one underscore")
    check(ing.safe_basename("a   b.jpg") == "a_b.jpg",
          "a run of spaces becomes one underscore")
    check(ing.safe_basename("  .jpg") == "image.jpg",
          "a stem with nothing safe in it becomes image")
    check(ing.safe_basename("plain_5.jpg") == "plain_5.jpg",
          "a name already safe is unchanged")


def test_read_names_csv():
    root = _mkdtemp("names")
    try:
        good = os.path.join(root, "good.csv")
        write(good, 'token,name\nA_1,"Genus a, fish a"\nB_2,plain\n')
        rows = ing.read_names_csv(good)
        check(rows == [("A_1", "Genus a, fish a"), ("B_2", "plain")],
              f"read_names_csv keeps the quoted comma (got {rows})")

        bad = os.path.join(root, "bad.csv")
        write(bad, "code,label\nA_1,x\n")
        try:
            ing.read_names_csv(bad)
            check(False, "a wrong header must be refused")
        except ing.IngestError as e:
            check("token" in str(e) and "name" in str(e),
                  f"the refusal names the header it wanted (got {e})")

        empty = os.path.join(root, "empty.csv")
        write(empty, "")
        try:
            ing.read_names_csv(empty)
            check(False, "an empty names CSV must be refused")
        except ing.IngestError as e:
            check("empty" in str(e).lower(),
                  f"the refusal says the file is empty (got {e})")

        try:
            ing.read_names_csv(os.path.join(root, "absent.csv"))
            check(False, "a missing names CSV must be refused")
        except ing.IngestError as e:
            check("absent.csv" in str(e),
                  f"the refusal names the missing path (got {e})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_read_class_list():
    root = _mkdtemp("classes")
    try:
        mapping = os.path.join(root, "map.yaml")
        write(mapping, "names:\n  0: AAAA_Normal\n  1: BBBB_TP\nnc: 2\n")
        check(ing.read_class_list(mapping) == fixture_classes(),
              "a names mapping is read in id order")

        listed = os.path.join(root, "list.yaml")
        write(listed, "names:\n  - AAAA_Normal\n  - BBBB_TP\n")
        check(ing.read_class_list(listed) == fixture_classes(),
              "a names list is read in order")

        gap = os.path.join(root, "gap.yaml")
        write(gap, "names:\n  0: AAAA_Normal\n  2: BBBB_TP\nnc: 2\n")
        try:
            ing.read_class_list(gap)
            check(False, "a gap in the class ids must be refused")
        except ing.IngestError as e:
            check("contiguous" in str(e) or "0" in str(e),
                  f"the refusal says the ids are not contiguous (got {e})")

        wrong_nc = os.path.join(root, "nc.yaml")
        write(wrong_nc, "names:\n  0: AAAA_Normal\n  1: BBBB_TP\nnc: 3\n")
        try:
            ing.read_class_list(wrong_nc)
            check(False, "an nc that disagrees with the names must be refused")
        except ing.IngestError as e:
            check("nc" in str(e) and "3" in str(e),
                  f"the refusal names both counts (got {e})")

        no_names = os.path.join(root, "none.yaml")
        write(no_names, "path: .\nnc: 2\n")
        try:
            ing.read_class_list(no_names)
            check(False, "a data.yaml with no names must be refused")
        except ing.IngestError as e:
            check("names" in str(e),
                  f"the refusal names the missing key (got {e})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── layout, manifest and pins ────────────────────────────────────────────

def test_stage_layout_and_manifest():
    root = _mkdtemp("stage")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root)
        m = ing.ingest_yolo(
            img, lab, dest,
            classes=ing.read_class_list(os.path.join(root, "data.yaml")),
            names=ing.read_names_csv(os.path.join(root, "names.csv")),
            names_csv=os.path.join(root, "names.csv"),
            split_root=os.path.join(root, "split"), log=quiet)
        for sub in ("images", "labels", "classes.txt", "data.yaml", "names.csv",
                    "split_pins.json", "rename_map.csv", "LICENSE.md",
                    "ingest_manifest.json"):
            check(os.path.exists(os.path.join(dest, sub)), f"{sub} written")
        check(m["n_images"] == 6 and m["n_boxes"] == 6 and m["n_classes"] == 2,
              f"counts {m['n_images']}/{m['n_boxes']}/{m['n_classes']}")
        check(m["source_kind"] == "yolo" and m["split_pins"] is True
              and m["n_renamed"] == 4,
              f"manifest facts {m['source_kind']}/{m['split_pins']}/{m['n_renamed']}")
        check(m["n_background"] == 3, "empty label files counted as background")
        check(m["n_duplicate_lines"] == 0,
              f"no duplicate label lines in this fixture (got {m['n_duplicate_lines']})")
        check(m.get("n_case_disambiguated") == 0,
              f"a fixture with no case clash disambiguates nothing "
              f"(got {m.get('n_case_disambiguated')})")
        check(m["labels_source"] == os.path.abspath(lab)
              and m["source_path"] == os.path.abspath(img),
              "the manifest records both source folders")
        check(m["names_csv"] == os.path.abspath(os.path.join(root, "names.csv")),
              f"the manifest records the names table it read (got {m['names_csv']})")
        for key in ("dataset", "n_degenerate", "n_degenerate_exact", "classes",
                    "ingested_at", "limit", "citation", "license_note",
                    "coverage_note"):
            check(key in m, f"the manifest carries the Coral Intel key {key}")
        check(m["dataset"] == "staged", f"dataset is the folder name (got {m['dataset']})")
        check(len(m["ingested_at"]) == 19 and m["ingested_at"][4] == "-",
              f"ingested_at is an AST 24h stamp (got {m['ingested_at']!r})")

        disk = json.load(open(os.path.join(dest, "ingest_manifest.json")))
        check(disk == m, "ingest_yolo returns exactly what it wrote to disk")

        pins_doc = json.load(open(os.path.join(dest, "split_pins.json")))
        check(pins_doc["schema_version"] == 1
              and pins_doc["source"] == os.path.abspath(os.path.join(root, "split")),
              "split_pins.json carries its schema version and source")
        pins = pins_doc["pins"]
        check(pins.get("plain_5.jpg") == "valid"
              and pins.get("unicode_6.jpg") == "test",
              f"pins keyed by staged name {pins}")
        check(len(pins) == 6
              and sum(1 for v in pins.values() if v == "train") == 4,
              "every image pinned")
        check(os.path.isfile(os.path.join(
            dest, "labels", "Screen_Shot_2023-07-27_at_10.07.40_AM.txt")),
            "label renamed with its image")
        check(os.path.isfile(os.path.join(dest, "images", "odd.jpg"))
              and not os.path.exists(os.path.join(dest, "images", "odd.jfif")),
              "the jfif image is staged under the jpg extension")

        classes_txt = open(os.path.join(dest, "classes.txt")).read().splitlines()
        check(classes_txt == fixture_classes(),
              f"classes.txt is the class list in id order (got {classes_txt})")
        yaml_text = open(os.path.join(dest, "data.yaml")).read()
        check("nc: 2" in yaml_text and "path: ." in yaml_text
              and "  0: AAAA_Normal" in yaml_text,
              "data.yaml is written in the Coral Intel shape")
        names_text = open(os.path.join(dest, "names.csv")).read()
        check(names_text.splitlines()[0] == "token,name"
              and '"Genus a, fish a"' in names_text,
              f"names.csv keeps the token,name header and the quoting (got {names_text!r})")
        rename_text = open(os.path.join(dest, "rename_map.csv")).read()
        check(rename_text.splitlines()[0] == "original,staged"
              and "odd.jfif,odd.jpg" in rename_text
              and len(rename_text.splitlines()) == 5,
              f"rename_map.csv holds one row per renamed record (got {rename_text!r})")
        license_text = open(os.path.join(dest, "LICENSE.md")).read()
        check("license recorded" in license_text.lower() or m["license_note"],
              "license text present")
        check(m["citation"] == ing.NO_CITATION
              and m["license_note"] == ing.NO_LICENSE_NOTE,
              "with no terms given the manifest says none is recorded")
        check("inprocess/run_" in m["coverage_note"],
              "the coverage note repeats that this imagery stays out of the runs")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_background_without_a_label_file():
    """An image with no label file at all is background, not a refusal."""
    root = _mkdtemp("bg")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write_jpeg(os.path.join(img, "no_label_7.jpg"))
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_images"] == 7 and m["n_boxes"] == 6,
              f"the unlabelled image is staged and adds no box (got {m['n_images']}/{m['n_boxes']})")
        check(m["n_background"] == 4,
              f"an image with no label file counts as background (got {m['n_background']})")
        check(m["split_pins"] is False,
              "without a split tree the manifest says there are no pins")
        check(not os.path.exists(os.path.join(dest, "split_pins.json")),
              "and no split_pins.json is written")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_duplicate_lines_counted():
    root = _mkdtemp("dupes")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write(os.path.join(lab, "plain_5.txt"),
              "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n0  0.5 0.5 0.2 0.2\n")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_duplicate_lines"] == 2,
              f"repeated label lines are counted, whitespace ignored (got {m['n_duplicate_lines']})")
        check(m["n_boxes"] == 9,
              f"duplicates are still staged and counted as boxes (got {m['n_boxes']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_degenerate_boxes_counted():
    root = _mkdtemp("degen")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write(os.path.join(lab, "plain_5.txt"),
              "0 0.5 0.5 1 1\n1 0.5 0.5 0.9996 1.0\n1 0.4 0.4 0.5 0.5\n")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_degenerate"] == 2,
              f"whole-frame boxes counted by the shared rule (got {m['n_degenerate']})")
        check(m["n_degenerate_exact"] == 1,
              f"only the literal 0.5 0.5 1 1 form is exact (got {m['n_degenerate_exact']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── refusals ─────────────────────────────────────────────────────────────

def test_refusals():
    root = _mkdtemp("refuse")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        classes = fixture_classes()
        names = fixture_names()

        write(os.path.join(lab, "plain_5.txt"), "2 0.5 0.5 0.1 0.1\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names, log=quiet)
            check(False, "class id 2 must be refused")
        except ing.IngestError as e:
            check("class id 2" in str(e), f"refusal names the id: {e}")

        write(os.path.join(lab, "plain_5.txt"), "0 0.5 0.5 0.1\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names, log=quiet)
            check(False, "a four-field label line must be refused")
        except ing.IngestError as e:
            check("class cx cy w h" in str(e) and "plain_5.txt" in str(e),
                  f"refusal names the file and the shape it wanted: {e}")

        write(os.path.join(lab, "plain_5.txt"), "0 0.5 0.5 0.1 0.1\n")
        write_jpeg(os.path.join(img, "Fish A_1.JPEG"))
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names, log=quiet)
            check(False, "collision must be refused")
        except ing.IngestError as e:
            check("Fish_A_1.jpeg" in str(e) and "Fish A_1.JPEG" in str(e),
                  f"collision names both files: {e}")
        os.remove(os.path.join(img, "Fish A_1.JPEG"))

        write(os.path.join(lab, "orphan_9.txt"), "0 0.5 0.5 0.1 0.1\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names, log=quiet)
            check(False, "a label with no image must be refused")
        except ing.IngestError as e:
            check("orphan_9.txt" in str(e), f"refusal names the orphan label: {e}")
        os.remove(os.path.join(lab, "orphan_9.txt"))

        try:
            ing.ingest_yolo(img, lab, dest, classes=classes,
                            names=[("AAAA_Normal", "a")], log=quiet)
            check(False, "missing name must be refused")
        except ing.IngestError as e:
            check("BBBB_TP" in str(e), "names csv must cover every token")

        try:
            ing.ingest_yolo(img, lab, os.path.join(root, "bad name"),
                            classes=classes, names=names, log=quiet)
            check(False, "a destination name outside the name rule must be refused")
        except ing.IngestError as e:
            check("bad name" in str(e), f"refusal names the folder: {e}")

        try:
            ing.ingest_yolo(os.path.join(root, "absent"), lab, dest,
                            classes=classes, names=names, log=quiet)
            check(False, "a missing images folder must be refused")
        except ing.IngestError as e:
            check("absent" in str(e), f"refusal names the missing folder: {e}")

        try:
            ing.ingest_yolo(img, lab, dest, classes=[], names=names, log=quiet)
            check(False, "an empty class list must be refused")
        except ing.IngestError as e:
            check("class" in str(e).lower(), f"refusal says the class list is empty: {e}")

        check(not os.path.exists(dest), "nothing staged after a refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_polygon_label_refused():
    """A YOLO segmentation label (a polygon) is not a box file and is refused.

    Read as boxes, the first four polygon coordinates would become a centre and
    an extent, so every count in the manifest would describe geometry that is
    not in the source.
    """
    root = _mkdtemp("polygon")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write(os.path.join(lab, "plain_5.txt"),
              "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a nine-field polygon label line must be refused")
        except ing.IngestError as e:
            check("plain_5.txt" in str(e) and "9" in str(e)
                  and "polygon" in str(e),
                  f"the refusal names the file, the field count and polygons: {e}")
        check(not os.path.exists(dest),
              "nothing staged after a polygon label line")

        write(os.path.join(lab, "plain_5.txt"), "0 0.5 0.5 0.1 0.1 0.9\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a six-field label line must be refused")
        except ing.IngestError as e:
            check("6" in str(e), f"the refusal names the field count it read: {e}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_label_stem_collision():
    """Two records whose staged label file would be the same file are refused.

    "x y.jpg" and "x_y.png" stage under two different image names but one
    label stem, so a single staged label file would have to describe both
    images and one of them would train on the other's boxes.
    """
    root = _mkdtemp("stems")
    dest = os.path.join(root, "staged")
    try:
        img = os.path.join(root, "images")
        lab = os.path.join(root, "labels")
        write_jpeg(os.path.join(img, "x y.jpg"))
        write_jpeg(os.path.join(img, "x_y.png"))
        write(os.path.join(lab, "x y.txt"), "0 0.5 0.5 0.2 0.2\n")
        write(os.path.join(lab, "x_y.txt"),
              "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "two records sharing a staged label stem must be refused")
        except ing.IngestError as e:
            text = str(e)
            check("x_y.txt" in text and "x y.jpg" in text and "x_y.png" in text,
                  f"the refusal names the staged label and both sources: {e}")
        check(not os.path.exists(dest), "nothing staged after a stem collision")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── case-insensitive destinations ────────────────────────────────────────

def _names_differing_only_by_case(names):
    """The names in one folder that share a lower-case form with another.

    Parameters:
      names: the basenames in one folder.
    Returns:
      a sorted list of the offending names, empty when every name in the
      folder is unique ignoring letter case.
    """
    seen, clashing = {}, set()
    for name in names:
        key = name.lower()
        if key in seen:
            clashing.add(seen[key])
            clashing.add(name)
        seen[key] = name
    return sorted(clashing)


def case_pairs(dest):
    """Every staged name that differs from another staged name only by case.

    Read over the whole staged tree rather than over the two files a test
    planted, so a disambiguation that fixes one pair and makes another is
    caught.

    Parameters:
      dest: a staged dataset folder.
    Returns:
      a sorted list of "images/<name>" and "labels/<name>" entries, empty when
      the staged tree is safe in a case-insensitive destination.
    """
    out = []
    for sub in ("images", "labels"):
        folder = os.path.join(dest, sub)
        if os.path.isdir(folder):
            out += [sub + "/" + name
                    for name in _names_differing_only_by_case(os.listdir(folder))]
    return sorted(out)


def read_staged(dest, *parts):
    """The text of one staged file, or "" when the stager did not write it.

    A case test asserts on the content of a file whose name is the thing under
    test, so a missing file has to report a FAIL rather than stop the suite
    with a traceback.

    Parameters:
      dest: the staged dataset folder.
      parts: the path parts under it, for example "labels", "Nurse_shark.txt".
    Returns:
      the file's text, or "" when it is not there.
    """
    path = os.path.join(dest, *parts)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def add_case_pair(img, lab, first, second):
    """Two source images whose staged stems differ only in letter case.

    Each gets its own label file, with a different number of boxes, so a test
    can prove the staged labels were not swapped or merged.

    Parameters:
      img: the source images folder.
      lab: the source labels folder.
      first: the source basename that sorts first, for example "Nurse shark.JPG".
      second: the other source basename, for example "nurse shark.png".
    Returns:
      None.
    """
    write_jpeg(os.path.join(img, first))
    write_jpeg(os.path.join(img, second))
    write(os.path.join(lab, os.path.splitext(first)[0] + ".txt"),
          ONE_BOX)
    write(os.path.join(lab, os.path.splitext(second)[0] + ".txt"),
          TWO_BOXES)


def test_case_only_collision_is_disambiguated():
    """Two photographs whose names differ only by case both stage, with their labels.

    The fish dataset carried "Nurse shark.JPG" and "nurse shark.png". Staged
    verbatim they became labels/Nurse_shark.txt and labels/nurse_shark.txt, and
    the Synology Drive sync renamed one of them four minutes later, leaving one
    image with no label and every split fingerprint computed from the dataset
    changed.
    """
    root = _mkdtemp("case")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        add_case_pair(img, lab, "Nurse shark.JPG", "nurse shark.png")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        images = sorted(os.listdir(os.path.join(dest, "images")))
        check("Nurse_shark.jpg" in images and "nurse_shark_2.png" in images,
              f"the later source name takes a _2 suffix (got {images})")
        check(m["n_images"] == 8,
              f"both photographs are staged (got {m['n_images']})")
        check(m.get("n_case_disambiguated") == 1,
              f"the manifest counts one disambiguation "
              f"(got {m.get('n_case_disambiguated')})")
        first = read_staged(dest, "labels", "Nurse_shark.txt")
        second = read_staged(dest, "labels", "nurse_shark_2.txt")
        check(first == ONE_BOX,
              f"the first photograph keeps its own label (got {first!r})")
        check(second == TWO_BOXES,
              f"the second photograph keeps its own label (got {second!r})")
        rows = read_staged(dest, "rename_map.csv")
        check("nurse shark.png,nurse_shark_2.png" in rows,
              "rename_map.csv records the disambiguated name the same way as "
              "every other rename")
        check(m["n_renamed"] == len(rows.splitlines()) - 1,
              f"n_renamed still counts every rename_map row (got {m['n_renamed']} "
              f"for {len(rows.splitlines()) - 1} rows)")
        check(case_pairs(dest) == [],
              f"no two staged names differ only by case (got {case_pairs(dest)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_exact_collision_still_refused():
    """Two source names that reduce to one staged name are still refused.

    The regression guard for the case rule: a case-only clash is two real,
    distinct files and is disambiguated, but two names that normalise to one
    name mean the operator has to look at the source, and that refusal stands.
    """
    root = _mkdtemp("exact")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write_jpeg(os.path.join(img, "plain 5.jpg"))
        write(os.path.join(lab, "plain 5.txt"), ONE_BOX)
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "two source names reducing to one must still be refused")
        except ing.IngestError as e:
            check("plain_5.jpg" in str(e) and "plain 5.jpg" in str(e)
                  and "Rename one at the source" in str(e),
                  f"the refusal names the staged name, both sources and the "
                  f"remedy: {e}")
        os.remove(os.path.join(img, "plain 5.jpg"))
        os.remove(os.path.join(lab, "plain 5.txt"))

        write_jpeg(os.path.join(img, "plain_5.png"))
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "two records sharing one staged label stem must still "
                         "be refused")
        except ing.IngestError as e:
            check("plain_5.txt" in str(e) and "plain_5.png" in str(e),
                  f"the stem refusal names the one label file and both "
                  f"sources: {e}")
        check(not os.path.exists(dest), "nothing staged after either refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_case_names_are_deterministic():
    """Staging one source folder twice gives the same staged names both times.

    A staged name that depended on directory order would change a dataset's
    split fingerprint between two stages of the same source, and a fingerprint
    is what decides whether two trained models may be compared at all.
    """
    root = _mkdtemp("determinism")
    first = os.path.join(root, "staged")
    second = os.path.join(root, "second")
    try:
        img, lab = make_source(root, with_split=False)
        add_case_pair(img, lab, "Nurse shark.JPG", "nurse shark.png")
        add_case_pair(img, lab, "REEF.png", "Reef.jpg")
        a = ing.ingest_yolo(img, lab, first, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        b = ing.ingest_yolo(img, lab, second, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        names_a = sorted(os.listdir(os.path.join(first, "images")))
        names_b = sorted(os.listdir(os.path.join(second, "images")))
        check(names_a == names_b,
              f"the two destinations hold the same staged image names "
              f"(first {names_a}, second {names_b})")
        labels_a = sorted(os.listdir(os.path.join(first, "labels")))
        labels_b = sorted(os.listdir(os.path.join(second, "labels")))
        check(labels_a == labels_b,
              "the two destinations hold the same staged label names")
        map_a = read_staged(first, "rename_map.csv")
        map_b = read_staged(second, "rename_map.csv")
        check(map_a == map_b, "both runs write the same rename_map.csv")
        check(a.get("n_case_disambiguated") == b.get("n_case_disambiguated") == 2,
              f"both runs disambiguate the same two names "
              f"(first {a.get('n_case_disambiguated')}, "
              f"second {b.get('n_case_disambiguated')})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_three_way_case_collision():
    """Three source names differing only by case all survive, under distinct names."""
    root = _mkdtemp("three")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        planted = (("CC.webp", "0 0.1 0.1 0.1 0.1\n"),
                   ("Cc.png", "0 0.2 0.2 0.1 0.1\n"),
                   ("cc.jpg", "0 0.3 0.3 0.1 0.1\n"))
        for name, boxes in planted:
            write_jpeg(os.path.join(img, name))
            write(os.path.join(lab, os.path.splitext(name)[0] + ".txt"), boxes)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        images = sorted(os.listdir(os.path.join(dest, "images")))
        check("CC.webp" in images and "Cc_2.png" in images
              and "cc_3.jpg" in images,
              f"all three stage under distinct names (got {images})")
        check(m["n_images"] == 9,
              f"none of the three was dropped (got {m['n_images']})")
        check(m.get("n_case_disambiguated") == 2,
              f"the two later names are counted (got {m.get('n_case_disambiguated')})")
        for staged, (_source, boxes) in zip(("CC.txt", "Cc_2.txt", "cc_3.txt"),
                                            planted):
            text = read_staged(dest, "labels", staged)
            check(text == boxes,
                  f"{staged} carries its own boxes (got {text!r})")
        check(case_pairs(dest) == [],
              f"no two staged names differ only by case (got {case_pairs(dest)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_case_suffix_skips_a_name_a_third_file_owns():
    """A disambiguated name never takes a name another source file already owns.

    "dup.jpg" clashes by case with "DUP.png", and the first suffix a naive
    rule would reach for, dup_2, is the staged name of a third source file.
    """
    root = _mkdtemp("numbered")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        for name in ("DUP.png", "dup.jpg", "dup_2.jpg"):
            write_jpeg(os.path.join(img, name))
            write(os.path.join(lab, os.path.splitext(name)[0] + ".txt"), ONE_BOX)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        images = sorted(os.listdir(os.path.join(dest, "images")))
        check("DUP.png" in images and "dup_2.jpg" in images
              and "dup_3.jpg" in images,
              f"the suffix search steps over the taken name (got {images})")
        check(m["n_images"] == 9,
              f"all three stage (got {m['n_images']})")
        rows = dict(line.split(",", 1) for line
                    in read_staged(dest, "rename_map.csv").splitlines()[1:])
        check(rows.get("dup.jpg") == "dup_3.jpg",
              f"the clashing file took the first free suffix (got {rows.get('dup.jpg')})")
        check("dup_2.jpg" not in rows,
              "the third file keeps its own name and is not recorded as renamed")
        check(m.get("n_case_disambiguated") == 1,
              f"one name was disambiguated (got {m.get('n_case_disambiguated')})")
        check(case_pairs(dest) == [],
              f"no two staged names differ only by case (got {case_pairs(dest)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_case_suffix_that_would_not_fit_on_disk_is_refused():
    """A numbered suffix that would pass the file-name limit is refused, not attempted.

    Every other step of safe_basename shortens a name or keeps its length, so
    a source file that exists on disk can always be copied under its staged
    name. The case suffix is the one step that makes a name longer, so it is
    the one step that can reach the 255-byte limit. It has to refuse while the
    run is still planning: refusing during the copy would stop part way
    through and leave a half-staged folder behind.
    """
    root = _mkdtemp("toolong")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        stem = "L" * LONG_STEM_CHARS
        for name in (stem + ".jpg", stem.lower() + ".png"):
            write_jpeg(os.path.join(img, name))
            write(os.path.join(lab, os.path.splitext(name)[0] + ".txt"), ONE_BOX)
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a suffix that would not fit on disk must be refused")
        except ing.IngestError as e:
            text = str(e)
            check(stem.lower() in text,
                  "the refusal names the source file that could not be named")
            check("'_2'" in text,
                  f"the refusal names the suffix it needed (got {text[-200:]!r})")
            reached = LONG_STEM_CHARS + len("_2.png")
            check(str(reached) in text and str(ing.MAX_STAGED_NAME_BYTES) in text,
                  f"the refusal gives the length reached ({reached}) and the "
                  f"limit ({ing.MAX_STAGED_NAME_BYTES})")
            check("Shorten the name at the source" in text,
                  "the refusal gives the remedy")
        check(not os.path.exists(dest),
              "nothing was staged: the refusal comes before the first copy")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_long_name_without_a_case_twin_still_stages():
    """A name at the limit with no case twin needs no suffix and stages as it is.

    The guard rail above must not turn into a length rule of its own: only a
    name that has to take a suffix is measured.
    """
    root = _mkdtemp("longok")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        stem = "L" * LONG_STEM_CHARS
        write_jpeg(os.path.join(img, stem + ".jpg"))
        write(os.path.join(lab, stem + ".txt"), ONE_BOX)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_images"] == 7,
              f"the long name stages beside the fixture (got {m['n_images']})")
        check(os.path.isfile(os.path.join(dest, "images", stem + ".jpg")),
              "the long image keeps its own name")
        check(m.get("n_case_disambiguated") == 0,
              f"no name took a suffix (got {m.get('n_case_disambiguated')})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_label_case_collision_follows_the_image_stem():
    """Two image names that are safe alone but whose label files would clash.

    "Foo.jpg" and "foo.png" differ by more than letter case, so the two staged
    image names are safe on their own; both label files would be foo.txt
    ignoring case. The label follows the image stem, so the image name is what
    gets fixed.
    """
    root = _mkdtemp("labelcase")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write_jpeg(os.path.join(img, "Foo.jpg"))
        write_jpeg(os.path.join(img, "foo.png"))
        write(os.path.join(lab, "Foo.txt"), ONE_BOX)
        write(os.path.join(lab, "foo.txt"), TWO_BOXES)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        images = sorted(os.listdir(os.path.join(dest, "images")))
        check("Foo.jpg" in images and "foo_2.png" in images,
              f"the second image is renamed although the image names were safe "
              f"(got {images})")
        check(read_staged(dest, "labels", "Foo.txt") == ONE_BOX
              and read_staged(dest, "labels", "foo_2.txt") == TWO_BOXES,
              "each label file follows its own image stem")
        check(m.get("n_case_disambiguated") == 1,
              f"the label clash is counted as one disambiguation "
              f"(got {m.get('n_case_disambiguated')})")
        check(case_pairs(dest) == [],
              f"no two staged names differ only by case (got {case_pairs(dest)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_restage_over_a_case_twin_is_refused():
    """A destination already holding the other case of a name is refused.

    The staged tree has to be safe in a case-insensitive destination as a
    whole, so a name already on disk counts as claimed even when the run's own
    names are unique among themselves.
    """
    root = _mkdtemp("twin")
    dest = os.path.join(root, "staged")
    try:
        lower = os.path.join(root, "lower")
        upper = os.path.join(root, "upper")
        for base, name in ((lower, "tone.jpg"), (upper, "Tone.jpg")):
            write_jpeg(os.path.join(base, "images", name))
            write(os.path.join(base, "labels",
                               os.path.splitext(name)[0] + ".txt"), ONE_BOX)
        ing.ingest_yolo(os.path.join(lower, "images"),
                        os.path.join(lower, "labels"), dest,
                        classes=fixture_classes(), names=fixture_names(),
                        log=quiet)
        try:
            ing.ingest_yolo(os.path.join(upper, "images"),
                            os.path.join(upper, "labels"), dest,
                            classes=fixture_classes(), names=fixture_names(),
                            log=quiet)
            check(False, "a re-stage over a name that differs only by case must "
                         "be refused")
        except ing.IngestError as e:
            check("tone.jpg" in str(e) and "Tone.jpg" in str(e)
                  and "letter case" in str(e),
                  f"the refusal names both names and the reason: {e}")
        check(sorted(os.listdir(os.path.join(dest, "images"))) == ["tone.jpg"],
              "the destination still holds only the name it already had")
        check(case_pairs(dest) == [],
              f"the destination holds no pair differing only by case "
              f"(got {case_pairs(dest)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dataset_folder_case_twin_is_refused():
    """Two staged dataset folders differing only by case cannot sit side by side.

    The rule that keeps the files inside a staged dataset safe in the sync has
    to hold for the dataset folder itself. The sync would rename one of the
    two folders minutes after the stage, and every reader that looks a dataset
    up by its folder name, the Boxes dataset list and the training source
    list, would stop finding it.
    """
    root = _mkdtemp("foldercase")
    parent = os.path.join(root, "bbox_datasets")
    try:
        img, lab = make_source(root, with_split=False)
        first = os.path.join(parent, "plain_set")
        ing.ingest_yolo(img, lab, first, classes=fixture_classes(),
                        names=fixture_names(), log=quiet)
        second = os.path.join(parent, "Plain_Set")
        try:
            ing.ingest_yolo(img, lab, second, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a dataset folder differing from a sibling only by "
                         "case must be refused")
        except ing.IngestError as e:
            check("plain_set" in str(e) and "Plain_Set" in str(e)
                  and "letter case" in str(e),
                  f"the refusal names the folder on disk, the folder this run "
                  f"would create, and the reason: {e}")
        check(not os.path.exists(second),
              "the second dataset folder was not created")
        check(sorted(os.listdir(parent)) == ["plain_set"],
              f"the staged root still holds only the first dataset "
              f"(got {sorted(os.listdir(parent))})")
        again = ing.ingest_yolo(img, lab, first, classes=fixture_classes(),
                                names=fixture_names(), log=quiet)
        check(again["n_images"] == 6,
              f"re-staging a dataset into its own folder is not caught by the "
              f"sibling rule (got {again['n_images']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_split_pins_follow_a_disambiguated_name():
    """A split pin written under the source name follows the disambiguated name.

    The fish dataset pins both nurse shark photographs to the test split under
    their source file names, so a pin read against the un-suffixed name would
    stop the whole run.
    """
    root = _mkdtemp("casepins")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root)
        add_case_pair(img, lab, "Nurse shark.JPG", "nurse shark.png")
        for name in ("Nurse shark.JPG", "nurse shark.png"):
            write(os.path.join(root, "split", "test", "images", name), "x")
        try:
            m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                                names=fixture_names(),
                                split_root=os.path.join(root, "split"),
                                log=quiet)
        except ing.IngestError as e:
            check(False, f"a pinned dataset with a case clash must stage: {e}")
            return
        pins = json.loads(read_staged(dest, "split_pins.json") or "{}").get("pins", {})
        check(pins.get("Nurse_shark.jpg") == "test"
              and pins.get("nurse_shark_2.png") == "test",
              f"both pins are keyed by the staged name (got "
              f"{pins.get('Nurse_shark.jpg')}, {pins.get('nurse_shark_2.png')})")
        check(m["n_pinned"] == m["n_images"] == 8,
              f"every staged image still carries a pin "
              f"(got {m['n_pinned']} of {m['n_images']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_label_folder_sidecars():
    """A classes.txt beside the label files is passed over, not read as a label.

    labelImg and Roboflow write classes.txt into the labels folder, and the
    raw source shares are read only, so the stager cannot ask the operator to
    delete it.
    """
    root = _mkdtemp("sidecar")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write(os.path.join(lab, "classes.txt"), "AAAA_Normal\nBBBB_TP\n")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_images"] == 6 and m["n_boxes"] == 6,
              f"a labels folder with a classes.txt stages cleanly (got {m['n_images']})")
        check(not os.path.isfile(os.path.join(dest, "labels", "classes.txt")),
              "the sidecar is not staged as a label file")

        write(os.path.join(lab, "orphan_9.txt"), "0 0.5 0.5 0.1 0.1\n")
        try:
            ing.ingest_yolo(img, lab, os.path.join(root, "second"),
                            classes=fixture_classes(), names=fixture_names(),
                            log=quiet)
            check(False, "a real orphan label must still be refused")
        except ing.IngestError as e:
            check("orphan_9.txt" in str(e) and "--labels" in str(e),
                  f"the refusal names the orphan and the remedy: {e}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_symlinked_images():
    """A source image that is a symlink is staged as a real file.

    The fish dataset's converted/all_images folder holds symlinks into the
    original export, so a staged copy that re-linked them would break the
    moment the export moved.
    """
    root = _mkdtemp("symlink")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        real = os.path.join(root, "elsewhere", "target_8.jpg")
        write_jpeg(real)
        link = os.path.join(img, "linked_8.jpg")
        os.symlink(real, link)
        write(os.path.join(lab, "linked_8.txt"), "1 0.5 0.5 0.2 0.2\n")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        staged = os.path.join(dest, "images", "linked_8.jpg")
        check(m["n_images"] == 7, f"the linked image is staged (got {m['n_images']})")
        check(os.path.isfile(staged) and not os.path.islink(staged),
              "the staged copy is a real file, not another symlink")
        check(open(staged, "rb").read() == open(real, "rb").read(),
              "the staged copy holds the target's bytes")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_hostile_inputs():
    """Bytes that are not text, and a negative record cap, stop the run."""
    root = _mkdtemp("hostile")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        with open(os.path.join(lab, "plain_5.txt"), "wb") as fh:
            fh.write(b"\x00\xff\xfe 0 0.5 0.5 0.1 0.1\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a label file that is not text must be refused")
        except ing.IngestError as e:
            check("plain_5.txt" in str(e) and "could not be read" in str(e),
                  f"the refusal names the unreadable label file: {e}")
        write(os.path.join(lab, "plain_5.txt"), "0 0.5 0.5 0.1 0.1\n")

        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), limit=-5, log=quiet)
            check(False, "a negative --limit must be refused")
        except ing.IngestError as e:
            check("-5" in str(e), f"the refusal names the cap it was given: {e}")

        bad_yaml = os.path.join(root, "hostile.yaml")
        with open(bad_yaml, "wb") as fh:
            fh.write(b"names:\n  0: \xff\xfe\n")
        try:
            ing.read_class_list(bad_yaml)
            check(False, "a data.yaml that is not text must be refused")
        except ing.IngestError as e:
            check("hostile.yaml" in str(e),
                  f"the refusal names the unreadable data.yaml: {e}")

        bad_names = os.path.join(root, "hostile.csv")
        with open(bad_names, "wb") as fh:
            fh.write(b"token,name\nA_1,\xff\xfe\n")
        try:
            ing.read_names_csv(bad_names)
            check(False, "a names table that is not text must be refused")
        except ing.IngestError as e:
            check("hostile.csv" in str(e),
                  f"the refusal names the unreadable names table: {e}")

        check(not os.path.exists(dest), "nothing staged after a hostile input")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_split_refusals():
    root = _mkdtemp("split")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root)
        classes = fixture_classes()
        names = fixture_names()
        split = os.path.join(root, "split")

        shutil.rmtree(os.path.join(split, "test"))
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            split_root=split, log=quiet)
            check(False, "a split tree missing test/ must be refused")
        except ing.IngestError as e:
            check("test" in str(e), f"refusal names the missing split folder: {e}")
        os.makedirs(os.path.join(split, "test", "images"))
        write(os.path.join(split, "test", "images", SOURCE_NAMES[5]), "x")

        write(os.path.join(split, "test", "images", "ghost_99.jpg"), "x")
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            split_root=split, log=quiet)
            check(False, "a pinned image absent from the inputs must be refused")
        except ing.IngestError as e:
            check("ghost_99.jpg" in str(e), f"refusal names the absent image: {e}")
        os.remove(os.path.join(split, "test", "images", "ghost_99.jpg"))

        write(os.path.join(split, "valid", "images", SOURCE_NAMES[5]), "x")
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            split_root=split, log=quiet)
            check(False, "an image pinned twice must be refused")
        except ing.IngestError as e:
            check("unicode_6.jpg" in str(e) and "valid" in str(e),
                  f"refusal names the image and both splits: {e}")
        os.remove(os.path.join(split, "valid", "images", SOURCE_NAMES[5]))

        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            split_root=os.path.join(root, "no_split"), log=quiet)
            check(False, "a split root that is not a folder must be refused")
        except ing.IngestError as e:
            check("no_split" in str(e), f"refusal names the split root: {e}")

        check(not os.path.exists(dest), "nothing staged after a split refusal")
        m = ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            split_root=split, log=quiet)
        check(m["split_pins"] is True and m["n_images"] == 6,
              "the repaired split tree stages cleanly")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_partial_split_pins():
    """A split tree covering part of the dataset is counted honestly.

    The manifest must say how many staged images carry a pin, because a reader
    of split_pins alone would otherwise believe every image was pinned.
    """
    root = _mkdtemp("partial")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        for split in ("train", "valid", "test"):
            os.makedirs(os.path.join(root, "split", split, "images"))
        for nm in SOURCE_NAMES[:2]:
            write(os.path.join(root, "split", "train", "images", nm), "x")
        write(os.path.join(root, "split", "valid", "images", SOURCE_NAMES[4]), "x")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(),
                            split_root=os.path.join(root, "split"), log=quiet)
        check(m["n_images"] == 6 and m["n_pinned"] == 3,
              f"three of six images pinned (got {m['n_pinned']} of {m['n_images']})")
        check(m["split_pins"] is True,
              "split_pins stays true because split_pins.json was written")
        check(m["pins_by_split"] == {"train": 2, "valid": 1, "test": 0},
              f"pins_by_split carries the three counts (got {m['pins_by_split']})")
        pins = json.load(open(os.path.join(dest, "split_pins.json")))["pins"]
        check(len(pins) == 3, f"the pins file holds only the three pins (got {len(pins)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_pin_counts_on_a_full_split():
    """A split tree covering every image reports n_pinned equal to n_images."""
    root = _mkdtemp("fullpins")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(),
                            split_root=os.path.join(root, "split"), log=quiet)
        check(m["n_pinned"] == m["n_images"] == 6,
              f"every staged image carries a pin (got {m['n_pinned']}/{m['n_images']})")
        check(m["pins_by_split"] == {"train": 4, "valid": 1, "test": 1},
              f"the three counts match the fixture tree (got {m['pins_by_split']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unpinned_dataset_counts_zero():
    """With no split tree the pin counts are zero, not absent."""
    root = _mkdtemp("nopins")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_pinned"] == 0 and m["split_pins"] is False,
              f"no pins means n_pinned 0 (got {m['n_pinned']})")
        check(m["pins_by_split"] == {"train": 0, "valid": 0, "test": 0},
              f"pins_by_split is three zeros (got {m['pins_by_split']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── limit, idempotence, huge input ───────────────────────────────────────

def test_limit_and_idempotence():
    root = _mkdtemp("limit")
    dest = os.path.join(root, "staged")
    small = os.path.join(root, "small")
    try:
        img, lab = make_source(root)
        classes = fixture_classes()
        names = fixture_names()

        m = ing.ingest_yolo(img, lab, small, classes=classes, names=names,
                            split_root=os.path.join(root, "split"), limit=2,
                            log=quiet)
        check(m["n_images"] == 2 and m["limit"] == 2,
              f"--limit stages a prefix and is recorded (got {m['n_images']}/{m['limit']})")
        pins = json.load(open(os.path.join(small, "split_pins.json")))["pins"]
        check(len(pins) == 2,
              f"a limited run pins only what it staged (got {len(pins)})")
        staged_images = sorted(os.listdir(os.path.join(small, "images")))
        check(staged_images == ["Fish_A_1.jpeg", "Screen_Shot_2023-07-27_at_10.07.40_AM.png"],
              f"the prefix is the sorted head of the inputs (got {staged_images})")

        first = ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                                log=quiet)
        staged_image = os.path.join(dest, "images", "plain_5.jpg")
        before = os.stat(staged_image).st_ctime_ns
        second = ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                                 log=quiet)
        check(first["n_images"] == second["n_images"] == 6,
              "a second run over the same folder repeats the same counts")
        check(os.stat(staged_image).st_ctime_ns == before,
              "a second run copies no file that has not changed")
        check(len(os.listdir(os.path.join(dest, "labels"))) == 6,
              "a second run does not duplicate files")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_restage_refuses_leftovers():
    """A destination already holding files this run would not write is refused.

    Staging six records and then staging two into the same folder would leave
    the four older images and labels on disk while the manifest, the Bounding
    Boxes card and the Train panel all read 2, so the trainer would train on
    six images from a dataset described as two.
    """
    root = _mkdtemp("leftover")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        classes = fixture_classes()
        names = fixture_names()
        full = ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                               log=quiet)
        check(full["n_images"] == 6, "the first run stages every record")

        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                            limit=2, log=quiet)
            check(False, "a limited re-stage over a fuller folder must be refused")
        except ing.IngestError as e:
            check("images" in str(e) and "4" in str(e)
                  and "different folder" in str(e),
                  f"the refusal names the folder, the count and the remedy: {e}")
        check(len(os.listdir(os.path.join(dest, "images"))) == 6
              and len(os.listdir(os.path.join(dest, "labels"))) == 6,
              "the six staged images and six label files are untouched")
        manifest = json.load(open(os.path.join(dest, "ingest_manifest.json")))
        check(manifest["n_images"] == 6,
              f"the manifest still describes six images (got {manifest['n_images']})")

        os.remove(os.path.join(lab, os.path.splitext(SOURCE_NAMES[1])[0] + ".txt"))
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=names, log=quiet)
            check(False, "a label file lost at the source must be refused")
        except ing.IngestError as e:
            check("labels" in str(e), f"the refusal names the labels folder: {e}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_refuses_coral_intel_leftovers():
    """A folder holding a Coral Intel labels_localized/ is not staged into.

    The split builder prefers labels_localized/ over labels/, so a generic
    stage into that folder would train on the older dataset's labels.
    """
    root = _mkdtemp("localized")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        os.makedirs(os.path.join(dest, "labels_localized"))
        write(os.path.join(dest, "labels_localized", "old.txt"),
              "0 0.5 0.5 0.2 0.2\n")
        try:
            ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a leftover labels_localized/ must be refused")
        except ing.IngestError as e:
            check("labels_localized" in str(e) and "different folder" in str(e),
                  f"the refusal names the folder and the remedy: {e}")
        check(not os.path.exists(os.path.join(dest, "images")),
              "nothing staged beside the Coral Intel leftovers")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_refuses_stale_sidecars():
    """A names table or a pins file from an earlier run is refused, not left."""
    root = _mkdtemp("stale")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root)
        classes = fixture_classes()
        ing.ingest_yolo(img, lab, dest, classes=classes, names=fixture_names(),
                        split_root=os.path.join(root, "split"), log=quiet)
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=[], log=quiet)
            check(False, "a stale names.csv must be refused")
        except ing.IngestError as e:
            check("names.csv" in str(e), f"the refusal names the stale table: {e}")
        os.remove(os.path.join(dest, "names.csv"))
        try:
            ing.ingest_yolo(img, lab, dest, classes=classes, names=[], log=quiet)
            check(False, "a stale split_pins.json must be refused")
        except ing.IngestError as e:
            check("split_pins.json" in str(e),
                  f"the refusal names the stale pins file: {e}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_unwritable_destination():
    """A destination that cannot be written gets the house refusal, not a traceback."""
    root = _mkdtemp("unwritable")
    try:
        img, lab = make_source(root, with_split=False)
        already_a_file = os.path.join(root, "already_a_file")
        write(already_a_file, "not a folder\n")
        try:
            ing.ingest_yolo(img, lab, already_a_file, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
            check(False, "a destination that is a file must be refused")
        except ing.IngestError as e:
            check("already_a_file" in str(e),
                  f"the refusal names the destination it could not create: {e}")

        try:
            ing.ingest_yolo(img, lab, os.path.join(root, "cap"),
                            classes=fixture_classes(), names=fixture_names(),
                            limit="two", log=quiet)
            check(False, "a record cap that is not a number must be refused")
        except ing.IngestError as e:
            check("two" in str(e), f"the refusal names the cap it was given: {e}")

        if os.geteuid() != 0:
            locked = os.path.join(root, "locked")
            os.makedirs(locked)
            os.chmod(locked, 0o500)
            try:
                try:
                    ing.ingest_yolo(img, lab, os.path.join(locked, "staged"),
                                    classes=fixture_classes(),
                                    names=fixture_names(), log=quiet)
                    check(False, "a destination with no write permission must be refused")
                except ing.IngestError as e:
                    check("staged" in str(e),
                          f"the refusal names the folder it could not write: {e}")
            finally:
                os.chmod(locked, 0o700)
        else:
            check(True, "the permission case is skipped for the root user")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_huge_label_file():
    root = _mkdtemp("huge")
    dest = os.path.join(root, "staged")
    try:
        img, lab = make_source(root, with_split=False)
        write(os.path.join(lab, "plain_5.txt"),
              "".join(f"0 0.{i:06d} 0.5 0.1 0.1\n" for i in range(HUGE_LINES)))
        size = os.path.getsize(os.path.join(lab, "plain_5.txt"))
        check(size > 10 * 1024 * 1024, f"the fixture label file is over 10 MB ({size} bytes)")
        m = ing.ingest_yolo(img, lab, dest, classes=fixture_classes(),
                            names=fixture_names(), log=quiet)
        check(m["n_boxes"] == 6 + HUGE_LINES,
              f"every line of a 10 MB label file is counted (got {m['n_boxes']})")
        check(os.path.getsize(os.path.join(dest, "labels", "plain_5.txt")) == size,
              "the 10 MB label file is staged verbatim")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── command line ─────────────────────────────────────────────────────────

def _run_cli(args):
    """Run the stager through the env interpreter; returns the CompletedProcess."""
    return subprocess.run([PYTHON, SCRIPT] + args, capture_output=True, text=True)


def test_cli_end_to_end():
    root = _mkdtemp("cli")
    dest_root = os.path.join(root, "bbox_datasets")
    try:
        img, lab = make_source(root)
        common = ["--images", img, "--labels", lab,
                  "--data-yaml", os.path.join(root, "data.yaml"),
                  "--names", os.path.join(root, "names.csv"),
                  "--split-from", os.path.join(root, "split"),
                  "--dest-root", dest_root]

        proc = _run_cli(common + ["--name", "plain_set", "--quiet"])
        check(proc.returncode == 0,
              f"the command line exits 0 (rc {proc.returncode}, stderr {proc.stderr.strip()!r})")
        out = os.path.join(dest_root, "plain_set")
        m = json.load(open(os.path.join(out, "ingest_manifest.json")))
        check(m["n_images"] == 6 and m["n_boxes"] == 6 and m["split_pins"] is True,
              f"the command line writes the manifest (got {m['n_images']}/{m['n_boxes']})")
        check(m["citation"] == ing.NO_CITATION
              and m["license_note"] == ing.NO_LICENSE_NOTE,
              "without terms the manifest says none is recorded")
        check(os.path.isfile(os.path.join(out, "names.csv"))
              and os.path.isfile(os.path.join(out, "split_pins.json")),
              "the command line writes the names table and the pins")

        proc = _run_cli(common + ["--name", "cited_set",
                                  "--citation", "Olinger, L. K. (2026). Fish photos.",
                                  "--license", "Local training only, terms pending.",
                                  "--quiet"])
        check(proc.returncode == 0,
              f"the cited run exits 0 (rc {proc.returncode}, stderr {proc.stderr.strip()!r})")
        m2 = json.load(open(os.path.join(dest_root, "cited_set",
                                         "ingest_manifest.json")))
        check(m2["citation"] == "Olinger, L. K. (2026). Fish photos."
              and m2["license_note"] == "Local training only, terms pending.",
              f"--citation and --license reach the manifest (got {m2['citation']!r})")
        check(open(os.path.join(dest_root, "cited_set", "LICENSE.md")).read().strip()
              .endswith("Local training only, terms pending."),
              "the license text is written to LICENSE.md")

        proc = _run_cli(["--images", img, "--labels", lab,
                         "--names", os.path.join(root, "names.csv"),
                         "--dest-root", dest_root, "--name", "names_only", "--quiet"])
        check(proc.returncode == 0,
              f"a run with no data.yaml exits 0 (rc {proc.returncode}, stderr {proc.stderr.strip()!r})")
        classes_txt = open(os.path.join(dest_root, "names_only",
                                        "classes.txt")).read().splitlines()
        check(classes_txt == fixture_classes(),
              f"the class list comes from the names table order (got {classes_txt})")

        proc = _run_cli(["--images", img, "--labels", lab, "--dest-root", dest_root,
                         "--name", "no_source", "--quiet"])
        check(proc.returncode == 2 and "ingest stopped:" in proc.stderr,
              f"a run with neither names nor data.yaml exits 2 (rc {proc.returncode}, "
              f"stderr {proc.stderr.strip()!r})")

        proc = _run_cli(common + ["--name", "bad/name", "--quiet"])
        check(proc.returncode == 2 and "ingest stopped:" in proc.stderr,
              f"a refused name exits 2 with the house message (rc {proc.returncode}, "
              f"stderr {proc.stderr.strip()!r})")
        check(not os.path.exists(os.path.join(dest_root, "bad")),
              "the refused run staged nothing")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fish_names_table_ships():
    """The 34-row fish names table is present and reads cleanly."""
    path = os.path.join(_SCRIPTS, "_boxes", "names", "fish_photo_28feb25.csv")
    check(os.path.isfile(path), "the fish names table ships beside the stager")
    rows = ing.read_names_csv(path)
    check(len(rows) == 34, f"it holds 34 rows (got {len(rows)})")
    tokens = [t for t, _ in rows]
    check(len(set(tokens)) == 34, "every token appears once")
    lookup = dict(rows)
    check(lookup.get("EPST_Dark") == "Epinephelus striatus, Nassau grouper, dark phase",
          f"EPST_Dark carries its full name (got {lookup.get('EPST_Dark')!r})")
    check(all(name.strip() for _, name in rows), "no row carries an empty name")
    text = open(path).read()
    check("—" not in text and "–" not in text,
          "the names table has no em or en dashes")


def test_temp_fixtures_removed():
    """Every temp fixture this suite made is gone, and the module tree is clean.

    Runs last: it reads the list _mkdtemp recorded, so a test that forgot its
    finally block is caught here rather than leaving a folder on the box.
    """
    left = sorted(path for path in _TEMP_DIRS if os.path.exists(path))
    check(not left, f"temp fixtures removed (still present: {left})")
    own_source = open(os.path.abspath(__file__), encoding="utf-8").read()
    # Spelled in two pieces so this line is not itself one of the matches.
    needle = "tempfile." + "mkdtemp("
    check(own_source.count(needle) == 1,
          "every fixture in this suite is made through _mkdtemp, so this check "
          "sees all of them")
    staged_root = os.path.join(_REPO, "supporting_data", "bbox_datasets")
    if os.path.isdir(staged_root):
        present = set(os.listdir(staged_root))
        intruders = sorted(present.intersection(STAGED_NAMES))
        check(not intruders,
              f"nothing this suite staged appears in the module tree "
              f"(found {intruders})")
    else:
        check(True, "the module has no staged bbox_datasets folder to check")


def test_subject_is_checked_and_recorded():
    """What a dataset is about, recorded for the Train panel's Subject control.

    The panel matches that word against what every training recipe declares it
    is for, so a word outside the vocabulary would be carried into every subset
    and then match nothing, with nothing to say why. It is refused here, while
    the operator is still at the command line.
    """
    root = _mkdtemp("subject")
    try:
        img, lab = make_source(root, n=2, with_split=False)
        classes = ing.read_class_list(os.path.join(root, "data.yaml"))
        names = ing.read_names_csv(os.path.join(root, "names.csv"))

        plain = ing.ingest_yolo(img, lab, os.path.join(root, "staged"),
                                classes=classes, names=names, log=quiet)
        check(plain["subject"] == "",
              f"a dataset staged without --subject records none (got {plain['subject']!r})")

        for given, want in ((" Coral ", "coral"), ("FISH", "fish"), ("algae", "algae")):
            dest = os.path.join(root, "small_" + want)
            m = ing.ingest_yolo(img, lab, dest, classes=classes, names=names,
                                subject=given, log=quiet)
            check(m["subject"] == want,
                  f"--subject {given!r} must be recorded as {want!r} (got {m['subject']!r})")
            disk = json.load(open(os.path.join(dest, "ingest_manifest.json")))
            check(disk["subject"] == want,
                  f"and must reach the file on disk (got {disk.get('subject')!r})")
            shutil.rmtree(dest)

        for bad in ("reef", "corals", "coral reef", 7, ["coral"], "  "):
            try:
                ing.check_subject(bad)
                ok = bad in ("  ",)
            except ing.IngestError as exc:
                ok = "not a subject VICARIUS knows" in str(exc)
                check(", ".join(ing.DATASET_SUBJECTS) in str(exc),
                      f"the refusal must name the words that are accepted: {exc}")
            check(ok, f"subject {bad!r} must be refused, or normalised when it is blank")

        check(ing.check_subject(None) == "" and ing.check_subject("") == "",
              "no subject given records none")

        before = sorted(os.listdir(root))
        try:
            ing.ingest_yolo(img, lab, os.path.join(root, "second"), classes=classes,
                            names=names, subject="reef", log=quiet)
            check(False, "a refused subject must stop the run")
        except ing.IngestError as exc:
            check("reef" in str(exc), f"the refusal names the value: {exc}")
        check(sorted(os.listdir(root)) == before,
              "nothing is staged after a refused subject")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_subject_reaches_the_manifest_from_the_command_line():
    root = _mkdtemp("subjcli")
    dest_root = os.path.join(root, "bbox_datasets")
    try:
        img, lab = make_source(root, n=2, with_split=False)
        common = ["--images", img, "--labels", lab,
                  "--data-yaml", os.path.join(root, "data.yaml"),
                  "--dest-root", dest_root, "--quiet"]
        proc = _run_cli(common + ["--name", "plain_set", "--subject", "coral"])
        check(proc.returncode == 0,
              f"--subject coral exits 0 (rc {proc.returncode}, stderr {proc.stderr.strip()!r})")
        m = json.load(open(os.path.join(dest_root, "plain_set", "ingest_manifest.json")))
        check(m["subject"] == "coral",
              f"the command line writes the subject (got {m.get('subject')!r})")

        proc = _run_cli(common + ["--name", "second", "--subject", "reef"])
        check(proc.returncode == 2 and "ingest stopped:" in proc.stderr,
              f"an unknown subject exits 2 with the house message (rc {proc.returncode}, "
              f"stderr {proc.stderr.strip()!r})")
        check(not os.path.exists(os.path.join(dest_root, "second")),
              "and stages nothing")

        proc = _run_cli(common + ["--name", "no_source"])
        check(proc.returncode == 0,
              f"a run without --subject still exits 0 (rc {proc.returncode})")
        m2 = json.load(open(os.path.join(dest_root, "no_source", "ingest_manifest.json")))
        check(m2["subject"] == "",
              f"and records no subject (got {m2.get('subject')!r})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_subject_vocabulary_is_the_train_panels_own():
    """One vocabulary, two trees: the stager writes what the Train panel reads.
    A word in one list and not the other is a dataset whose subject can never
    start the control anywhere."""
    orch = os.path.join(_SCRIPTS, "pipeline_orchestrator")
    text = open(os.path.join(orch, "app.py"), encoding="utf-8").read()
    marker = "STEP6_SUBJECTS = ("
    check(marker in text, "app.py must declare STEP6_SUBJECTS")
    listed = text[text.index(marker) + len(marker):]
    listed = listed[:listed.index(")")]
    panel = tuple(v.strip().strip("'\"") for v in listed.split(",") if v.strip())
    check(panel == tuple(ing.DATASET_SUBJECTS),
          f"the Train panel knows {panel} and this stager writes "
          f"{tuple(ing.DATASET_SUBJECTS)}")


TESTS = [
    test_safe_basename,
    test_read_names_csv,
    test_read_class_list,
    test_stage_layout_and_manifest,
    test_background_without_a_label_file,
    test_duplicate_lines_counted,
    test_degenerate_boxes_counted,
    test_refusals,
    test_polygon_label_refused,
    test_label_stem_collision,
    test_case_only_collision_is_disambiguated,
    test_exact_collision_still_refused,
    test_case_names_are_deterministic,
    test_three_way_case_collision,
    test_case_suffix_skips_a_name_a_third_file_owns,
    test_case_suffix_that_would_not_fit_on_disk_is_refused,
    test_long_name_without_a_case_twin_still_stages,
    test_label_case_collision_follows_the_image_stem,
    test_restage_over_a_case_twin_is_refused,
    test_dataset_folder_case_twin_is_refused,
    test_split_pins_follow_a_disambiguated_name,
    test_label_folder_sidecars,
    test_symlinked_images,
    test_hostile_inputs,
    test_split_refusals,
    test_partial_split_pins,
    test_pin_counts_on_a_full_split,
    test_unpinned_dataset_counts_zero,
    test_limit_and_idempotence,
    test_restage_refuses_leftovers,
    test_refuses_coral_intel_leftovers,
    test_refuses_stale_sidecars,
    test_unwritable_destination,
    test_huge_label_file,
    test_cli_end_to_end,
    test_fish_names_table_ships,
    test_subject_is_checked_and_recorded,
    test_subject_reaches_the_manifest_from_the_command_line,
    test_the_subject_vocabulary_is_the_train_panels_own,
    test_temp_fixtures_removed,
]


def main():
    print("generic YOLO box-dataset stager:")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\n{_pass} passed, {_fail} failed")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
