#!/usr/bin/env python
"""Class-count subset of a staged box dataset: renumbering, counts, refusals.

No-pytest house harness. Every fixture is a synthetic staged parent built in a
temp dir and removed in a finally block: no test here reads a real dataset, and
nothing is written outside the temp dirs.

Covers:
  validation  a minimum that is not a positive whole number, a parent that is
              not a staged dataset (no data.yaml, no manifest), an output name
              outside the plain-name rule, an output folder that already
              exists, a minimum that would keep no class
  layout      images/ labels/ classes.txt data.yaml class_map.csv
              dropped_classes.csv LICENSE.md README.md ingest_manifest.json,
              and names.csv and split_pins.json when the parent has them
  renumber    surviving class ids are contiguous from 0, every kept label line
              is rewritten with its new id, and no dropped class survives in
              any label file
  images      an image whose every box belonged to a dropped class is left
              out, and so is a parent image with no box at all, whose count
              the manifest and the README both carry as its own number
  boundary    a class with exactly the minimum is kept, a class one box below
              it is dropped
  csvs        class_map.csv and dropped_classes.csv agree with the labels
              actually written
  license     LICENSE.md is copied byte for byte, and a parent with no
              LICENSE.md but a license note in its manifest still gets one
  manifest    every parent key carried, the counts recomputed, and the
              eleven subset keys
  sidecars    names.csv narrowed to the surviving classes, split_pins.json
              narrowed to the surviving images
  refusals    a four-field label line, a class id outside the parent class
              list, a classes.txt that disagrees with data.yaml, a label file
              with no image, a parent image pair differing only in letter
              case, a subset folder matching a sibling folder only in letter
              case, a subset folder whose holding folder is not there or is a
              file, an unreadable split_pins.json, and that nothing is written
              after any of them
  repeatable  a parent read twice gives byte-identical output apart from the
              two timestamps
  large       a 2000-image parent carrying 4 MB of labels is subset without
              holding every label in memory at once
  cli         the command line writes the manifest and exits 2 with
              "subset stopped:" on a refusal
  reuse       the three staged-layout writers are the stagers' own
              functions, and write_license reads nothing back off disk
  board       the full test board at scripts/run_all_tests.sh runs this suite
  cleanup     every temp fixture this suite made is gone, and nothing was
              written into the module's own bbox_datasets folder

Run with:
  env/bin/python scripts/_boxes/tests/test_subset_by_class_count.py
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import json
import os
import shutil
import subprocess
import sys
import tempfile
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _boxes import ingest_yolo_boxes as ing  # noqa: E402
from _boxes import subset_by_class_count as sub  # noqa: E402

# The tool under test, and the interpreter the module's suites run under. The
# repo's own env is preferred so a CLI test never runs under a foreign
# interpreter; sys.executable covers a checkout whose env is not built yet.
SCRIPT = os.path.join(_SCRIPTS, "_boxes", "subset_by_class_count.py")
_ENV_PYTHON = os.path.join(_REPO, "env", "bin", "python")
PYTHON = _ENV_PYTHON if os.path.isfile(_ENV_PYTHON) else sys.executable

# Every temp fixture this suite makes carries this prefix, so the closing
# cleanup test can prove each one was removed.
TEMP_PREFIX = "subset_"

# Dataset folder names this suite writes. The closing cleanup test proves none
# of them appears in the module's own supporting_data/bbox_datasets.
WRITTEN_NAMES = ("kept", "kept_again", "boundary_high", "boundary_low",
                 "from_note", "big", "cli_set", "nothing", "Kept")

# The five fixture classes, in parent class-id order. Box counts in the
# fixture labels below: 4, 3, 2, 1 and 0.
FIXTURE_CLASSES = ["Alpha-one", "Beta-two", "Gamma-three", "Delta-four",
                   "Epsilon-five"]

# One label file per fixture image, exactly as the parent carries it. img_e
# deliberately has no trailing newline and img_f is a background image.
FIXTURE_LABELS = {
    "img_a": "0 0.10 0.10 0.20 0.20\n2 0.20 0.20 0.10 0.10\n",
    "img_b": "1 0.30 0.30 0.10 0.10\n1 0.40 0.40 0.10 0.10\n",
    "img_c": "2 0.50 0.50 0.10 0.10\n3 0.60 0.60 0.10 0.10\n",
    "img_d": ("0 0.11 0.11 0.10 0.10\n0 0.12 0.12 0.10 0.10\n"
              "1 0.13 0.13 0.10 0.10\n"),
    "img_e": "0 0.70 0.70 0.10 0.10",
    "img_f": "",
}

# What the subset must hold at --min-boxes 3: Alpha-one (4 boxes) becomes
# class 0 and Beta-two (3 boxes) becomes class 1; the three rarer classes go.
EXPECTED_LABELS = {
    "img_a": "0 0.10 0.10 0.20 0.20\n",
    "img_b": "1 0.30 0.30 0.10 0.10\n1 0.40 0.40 0.10 0.10\n",
    "img_d": ("0 0.11 0.11 0.10 0.10\n0 0.12 0.12 0.10 0.10\n"
              "1 0.13 0.13 0.10 0.10\n"),
    "img_e": "0 0.70 0.70 0.10 0.10\n",
}

MIN_BOXES = 3

FIXTURE_LICENSE = ("# License\n\nAnnotations CC BY 4.0, cite Cordero et al. "
                   "2026. The photographs are third-party and excluded.\n")
FIXTURE_CITATION = "Cordero, E. and others (2026). A fixture citation."

# How many images the large-parent test writes, and how many lines each of its
# label files carries. 2000 x 100 lines of about 22 bytes is roughly 4.4 MB of
# label text, so an implementation that held every label at once would peak
# well above LARGE_PEAK_LIMIT_BYTES.
LARGE_IMAGES = 2000
LARGE_LINES_PER_LABEL = 100
LARGE_PEAK_LIMIT_BYTES = 2 * 1024 * 1024

# The two manifest fields that differ between two runs over the same parent.
TIMESTAMP_KEYS = ("ingested_at", "subset_created_at")

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
      tag: a short word naming the test, for example "layout".
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


def write_jpeg(path, size=(16, 16), color=(20, 90, 140)):
    """A real (tiny) JPEG, so a copied image can be opened by a later reader."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def quiet(_msg):
    """Swallow the tool's progress lines inside a test."""


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_rows(path):
    """Every row of a small CSV, header included, as lists of strings."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def build_parent(root, *, classes=None, labels=None, label_dir="labels_localized",
                 with_names=True, with_pins=True, license_text=FIXTURE_LICENSE,
                 manifest_extra=None, classes_txt=None):
    """A synthetic staged parent dataset in the layout both stagers write.

    Parameters:
      root: the folder the parent is built in.
      classes: class tokens in class-id order (FIXTURE_CLASSES by default).
      labels: {stem: label text} (FIXTURE_LABELS by default).
      label_dir: which label folder the boxes go in, "labels_localized" or
        "labels".
      with_names: write names.csv covering every class.
      with_pins: write split_pins.json.
      license_text: LICENSE.md content, or None to write no LICENSE.md.
      manifest_extra: extra keys merged into ingest_manifest.json.
      classes_txt: the classes.txt content, defaulting to the class list.
    Returns:
      the parent folder path.
    """
    classes = list(FIXTURE_CLASSES if classes is None else classes)
    labels = dict(FIXTURE_LABELS if labels is None else labels)
    for stem, text in labels.items():
        write_jpeg(os.path.join(root, "images", stem + ".jpg"))
        write(os.path.join(root, label_dir, stem + ".txt"), text)
    names_txt = ("".join(token + "\n" for token in classes)
                 if classes_txt is None else classes_txt)
    write(os.path.join(root, "classes.txt"), names_txt)
    lines = ["# YOLO dataset config for a staged bounding-box dataset.",
             "path: .", "names:"]
    lines += [f"  {i}: {token}" for i, token in enumerate(classes)]
    lines.append(f"nc: {len(classes)}")
    write(os.path.join(root, "data.yaml"), "\n".join(lines) + "\n")
    if license_text is not None:
        write(os.path.join(root, "LICENSE.md"), license_text)
    if with_names:
        write(os.path.join(root, "names.csv"),
              "token,name\n" + "".join(f"{t},Full name of {t}\n" for t in classes))
    if with_pins:
        write(os.path.join(root, "split_pins.json"), json.dumps({
            "schema_version": 1,
            "source": "/somewhere/original/split",
            "pins": {"img_a.jpg": "train", "img_c.jpg": "valid",
                     "img_d.jpg": "test", "img_absent.jpg": "train"},
        }, indent=2))
    manifest = {
        "dataset": os.path.basename(root),
        "n_images": len(labels),
        "n_boxes": sum(len([ln for ln in t.splitlines() if ln.split()])
                       for t in labels.values()),
        "n_degenerate": 0,
        "n_degenerate_exact": 0,
        "n_classes": len(classes),
        "classes": classes,
        "source_path": "/somewhere/original/distribution",
        "ingested_at": "2026-09-01 10:00:00",
        "limit": None,
        "citation": FIXTURE_CITATION,
        "license_note": "Annotations CC BY 4.0; photographs excluded.",
        "coverage_note": ing.COVERAGE_NOTE,
    }
    manifest.update(manifest_extra or {})
    write(os.path.join(root, "ingest_manifest.json"),
          json.dumps(manifest, indent=2))
    return root


def make_parent(tag, **kwargs):
    """A temp directory holding one synthetic parent, and the parent's path."""
    root = _mkdtemp(tag)
    parent = os.path.join(root, "parent_set")
    build_parent(parent, **kwargs)
    return root, parent


# ── validation ───────────────────────────────────────────────────────────

def test_normalise_min_boxes():
    check(sub.normalise_min_boxes(1) == 1, "1 is a usable minimum")
    check(sub.normalise_min_boxes("100") == 100, "a numeric string is read")
    for bad in (0, -5, "many", 2.5, None, True):
        try:
            sub.normalise_min_boxes(bad)
            check(False, f"the minimum {bad!r} must be refused")
        except ing.IngestError as exc:
            check("whole number" in str(exc) or "at least 1" in str(exc),
                  f"the refusal of {bad!r} says what a minimum must be "
                  f"(got {exc})")


def test_parent_must_be_a_staged_dataset():
    root, parent = make_parent("noyaml")
    out = os.path.join(root, "kept")
    try:
        os.remove(os.path.join(parent, "data.yaml"))
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "a parent with no data.yaml must be refused")
        except ing.IngestError as exc:
            check("data.yaml" in str(exc) and "staged" in str(exc),
                  f"the refusal names data.yaml (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("nomanifest")
    out2 = os.path.join(root2, "kept")
    try:
        os.remove(os.path.join(parent2, "ingest_manifest.json"))
        try:
            sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
            check(False, "a parent with no manifest must be refused")
        except ing.IngestError as exc:
            check("ingest_manifest.json" in str(exc),
                  f"the refusal names the manifest (got {exc})")
        check(not os.path.exists(out2), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root2, ignore_errors=True)

    missing = os.path.join(tempfile.gettempdir(), "subset_no_such_parent_dir")
    try:
        sub.subset_by_class_count(missing, missing + "_out", MIN_BOXES, log=quiet)
        check(False, "a parent folder that does not exist must be refused")
    except ing.IngestError as exc:
        check("not found" in str(exc) or "not a folder" in str(exc),
              f"the refusal says the folder is not there (got {exc})")


def test_output_name_and_existing_folder():
    root, parent = make_parent("names")
    try:
        try:
            sub.subset_by_class_count(parent, os.path.join(root, "bad name!"),
                                      MIN_BOXES, log=quiet)
            check(False, "an output name outside the name rule must be refused")
        except ing.IngestError as exc:
            check("dataset folder name" in str(exc),
                  f"the refusal quotes the name rule (got {exc})")

        long_name = "x" * 61
        try:
            sub.subset_by_class_count(parent, os.path.join(root, long_name),
                                      MIN_BOXES, log=quiet)
            check(False, "an output name longer than 60 characters is refused")
        except ing.IngestError as exc:
            check("60" in str(exc), f"the refusal names the limit (got {exc})")

        out = os.path.join(root, "kept")
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "an output folder that already exists is refused")
        except ing.IngestError as exc:
            check("already" in str(exc) and "remove" in str(exc).lower(),
                  f"the refusal says how to replace it (got {exc})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_minimum_that_keeps_no_class():
    root, parent = make_parent("nokeep")
    out = os.path.join(root, "nothing")
    try:
        try:
            sub.subset_by_class_count(parent, out, 500, log=quiet)
            check(False, "a minimum that keeps no class must be refused")
        except ing.IngestError as exc:
            check("4" in str(exc) and "no class" in str(exc).lower(),
                  f"the refusal names the largest class count (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── layout, renumbering and counts ───────────────────────────────────────

def test_layout_and_renumbering():
    root, parent = make_parent("layout")
    out = os.path.join(root, "kept")
    try:
        manifest = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        for rel in ("images", "labels", "classes.txt", "data.yaml",
                    "class_map.csv", "dropped_classes.csv", "LICENSE.md",
                    "README.md", "ingest_manifest.json", "names.csv",
                    "split_pins.json"):
            check(os.path.exists(os.path.join(out, rel)),
                  f"the subset carries {rel}")
        check(not os.path.exists(os.path.join(out, "labels_localized")),
              "the subset writes one label folder, labels/, and no other")

        staged = sorted(os.listdir(os.path.join(out, "labels")))
        check(staged == sorted(s + ".txt" for s in EXPECTED_LABELS),
              f"only the images with a surviving box carry a label (got {staged})")
        images = sorted(os.listdir(os.path.join(out, "images")))
        check(images == sorted(s + ".jpg" for s in EXPECTED_LABELS),
              f"only the images with a surviving box are copied (got {images})")

        for stem, text in EXPECTED_LABELS.items():
            got = read(os.path.join(out, "labels", stem + ".txt"))
            check(got == text,
                  f"{stem}.txt is rewritten with the new class ids "
                  f"(got {got!r})")

        classes = read(os.path.join(out, "classes.txt")).split()
        check(classes == ["Alpha-one", "Beta-two"],
              f"classes.txt holds the surviving names in parent order "
              f"(got {classes})")
        doc = read(os.path.join(out, "data.yaml"))
        check("0: Alpha-one" in doc and "1: Beta-two" in doc and "nc: 2" in doc,
              "data.yaml numbers the surviving classes from 0")
        ids = {int(line.split()[0])
               for stem in EXPECTED_LABELS
               for line in read(os.path.join(out, "labels", stem + ".txt")).splitlines()}
        check(ids == {0, 1},
              f"every label line carries a new id inside the class list (got {ids})")
        check(manifest["n_classes"] == 2 and manifest["n_boxes"] == 7
              and manifest["n_images"] == 4,
              f"the returned manifest counts the subset (got {manifest['n_classes']}, "
              f"{manifest['n_boxes']}, {manifest['n_images']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_class_map_and_dropped_csv():
    root, parent = make_parent("maps")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        rows = read_rows(os.path.join(out, "class_map.csv"))
        check(rows[0] == ["old_id", "new_id", "name", "boxes"],
              f"class_map.csv carries the stated header (got {rows[0]})")
        check(rows[1:] == [["0", "0", "Alpha-one", "4"],
                           ["1", "1", "Beta-two", "3"]],
              f"class_map.csv maps every surviving class (got {rows[1:]})")
        dropped = read_rows(os.path.join(out, "dropped_classes.csv"))
        check(dropped[0] == ["old_id", "name", "boxes"],
              f"dropped_classes.csv carries the stated header (got {dropped[0]})")
        check(dropped[1:] == [["2", "Gamma-three", "2"],
                              ["3", "Delta-four", "1"],
                              ["4", "Epsilon-five", "0"]],
              f"dropped_classes.csv lists every dropped class, a class with no "
              f"box included (got {dropped[1:]})")

        # The two CSVs together must describe exactly the parent class list,
        # and the kept box counts must equal the lines actually written.
        old_ids = [int(r[0]) for r in rows[1:]] + [int(r[0]) for r in dropped[1:]]
        check(sorted(old_ids) == list(range(len(FIXTURE_CLASSES))),
              "every parent class appears in exactly one of the two tables")
        written = {0: 0, 1: 0}
        for name in os.listdir(os.path.join(out, "labels")):
            for line in read(os.path.join(out, "labels", name)).splitlines():
                written[int(line.split()[0])] += 1
        check(written == {0: 4, 1: 3},
              f"class_map.csv box counts match the labels on disk (got {written})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_boundary_minimum():
    root, parent = make_parent("boundary")
    try:
        high = os.path.join(root, "boundary_high")
        manifest = sub.subset_by_class_count(parent, high, 3, log=quiet)
        check(manifest["classes"] == ["Alpha-one", "Beta-two"],
              "a class with exactly the minimum is kept")
        low = os.path.join(root, "boundary_low")
        manifest2 = sub.subset_by_class_count(parent, low, 4, log=quiet)
        check(manifest2["classes"] == ["Alpha-one"],
              "a class one box below the minimum is dropped")
        check(manifest2["n_boxes"] == 4 and manifest2["n_images"] == 3,
              f"the boundary run counts only the surviving class "
              f"(got {manifest2['n_boxes']} boxes, {manifest2['n_images']} images)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_manifest_matches_disk():
    root, parent = make_parent("manifest")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        m = json.loads(read(os.path.join(out, "ingest_manifest.json")))
        n_images = len(os.listdir(os.path.join(out, "images")))
        n_labels = len(os.listdir(os.path.join(out, "labels")))
        n_boxes = sum(len([ln for ln in read(os.path.join(out, "labels", f)).splitlines()
                           if ln.split()])
                      for f in os.listdir(os.path.join(out, "labels")))
        check(m["n_images"] == n_images == n_labels,
              f"n_images matches the files on disk (got {m['n_images']})")
        check(m["n_boxes"] == n_boxes,
              f"n_boxes matches the lines on disk (got {m['n_boxes']})")
        check(m["n_classes"] == len(read(os.path.join(out, "classes.txt")).split()),
              "n_classes matches classes.txt")
        check(m["classes"] == ["Alpha-one", "Beta-two"],
              f"classes lists the surviving names (got {m['classes']})")
        check(m["source_kind"] == "subset",
              f"source_kind says how this dataset was made (got {m['source_kind']})")
        check(m["parent_dataset"] == "parent_set"
              and m["parent_source_path"] == "/somewhere/original/distribution",
              "the manifest names its parent and the parent's own source")
        check(m["min_boxes"] == MIN_BOXES and m["n_classes_parent"] == 5
              and m["n_classes_dropped"] == 3 and m["n_images_dropped"] == 2
              and m["n_boxes_dropped"] == 3,
              "the manifest records the rule and what it dropped")
        check(m["label_source"] == "labels_localized",
              f"the manifest says which parent label folder was read "
              f"(got {m['label_source']})")
        check(m["subset_created_at"] == m["ingested_at"]
              and m["subset_created_at"].startswith("20"),
              "the manifest carries an AST timestamp for the subset")
        check(m["citation"] == FIXTURE_CITATION
              and "CC BY 4.0" in m["license_note"],
              "the citation and the license note are carried from the parent")
        check(m["coverage_note"] == ing.COVERAGE_NOTE,
              "every parent key the card reads is carried over")
        check(m["n_background"] == 0,
              f"the subset counts no background image (got {m.get('n_background')})")
        check(m["source_path"] == os.path.abspath(parent),
              f"source_path points at the parent folder this was read from "
              f"(got {m['source_path']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_license_carried_over():
    root, parent = make_parent("license")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        with open(os.path.join(parent, "LICENSE.md"), "rb") as fh:
            want = fh.read()
        with open(os.path.join(out, "LICENSE.md"), "rb") as fh:
            got = fh.read()
        check(got == want, "LICENSE.md is copied byte for byte")
        readme = " ".join(read(os.path.join(out, "README.md")).split())
        check("CC BY 4.0" in readme and FIXTURE_CITATION in readme,
              "the README carries the license and the citation")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_license_written_from_the_manifest_note():
    root, parent = make_parent("nolicense", license_text=None)
    out = os.path.join(root, "from_note")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        text = read(os.path.join(out, "LICENSE.md"))
        check("CC BY 4.0" in text,
              f"a parent with no LICENSE.md still gets one from its manifest "
              f"note (got {text!r})")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("nolicense2", license_text=None,
                                 manifest_extra={"license_note": ""})
    out2 = os.path.join(root2, "kept")
    try:
        try:
            sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
            check(False, "a parent with no license at all must be refused")
        except ing.IngestError as exc:
            check("LICENSE.md" in str(exc) and "license" in str(exc).lower(),
                  f"the refusal names the missing license (got {exc})")
        check(not os.path.exists(out2), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_readme_says_what_the_subset_is():
    root, parent = make_parent("readme")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        text = read(os.path.join(out, "README.md"))
        # Read against the unwrapped text: the README wraps its paragraphs to
        # a fixed width, so where a line breaks depends on the counts in it.
        flat = " ".join(text.split())
        for needle in ("parent_set", "class_map.csv", "AST", "LO",
                       "average precision", "Never quote it beside a score",
                       "at least 3 boxes", "labels_localized", "whole-frame"):
            check(needle in flat, f"the README says {needle!r}")
        check("2" in text and "3" in text and "4" in text,
              "the README carries the kept and dropped counts")
        for banned in ("—", "–"):
            check(banned not in text,
                  f"the README carries no {banned!r} (house prose rule)")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── sidecars ─────────────────────────────────────────────────────────────

def test_names_csv_narrowed():
    root, parent = make_parent("namescsv")
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        rows = read_rows(os.path.join(out, "names.csv"))
        check(rows == [["token", "name"],
                       ["Alpha-one", "Full name of Alpha-one"],
                       ["Beta-two", "Full name of Beta-two"]],
              f"names.csv is narrowed to the surviving classes (got {rows})")
        check(m["names_csv"] == os.path.join(parent, "names.csv"),
              f"the manifest names the parent table it was read from "
              f"(got {m['names_csv']})")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root4, parent4 = make_parent("badnames")
    out4 = os.path.join(root4, "kept")
    try:
        write(os.path.join(parent4, "names.csv"), "code,label\nAlpha-one,x\n")
        try:
            sub.subset_by_class_count(parent4, out4, MIN_BOXES, log=quiet)
            check(False, "a names table with the wrong header must be refused")
        except ing.IngestError as exc:
            check("token" in str(exc) and "name" in str(exc),
                  f"the refusal names the header it wanted (got {exc})")
        check(not os.path.exists(out4),
              "nothing was written after the names-table refusal")
    finally:
        shutil.rmtree(root4, ignore_errors=True)

    root2, parent2 = make_parent("nonames", with_names=False)
    out2 = os.path.join(root2, "kept")
    try:
        sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
        check(not os.path.exists(os.path.join(out2, "names.csv")),
              "a parent with no names table gives a subset with none")
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_split_pins_narrowed():
    root, parent = make_parent("pins")
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        doc = json.loads(read(os.path.join(out, "split_pins.json")))
        check(doc["pins"] == {"img_a.jpg": "train", "img_d.jpg": "test"},
              f"split_pins.json keeps only the images that survive (got {doc['pins']})")
        check(doc["source"] == "/somewhere/original/split",
              "the pins file still names the split tree it came from")
        check(m["split_pins"] is True and m["n_pinned"] == 2
              and m["pins_by_split"] == {"train": 1, "valid": 0, "test": 1},
              f"the manifest counts the narrowed pins (got {m.get('pins_by_split')})")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("nopins", with_pins=False)
    out2 = os.path.join(root2, "kept")
    try:
        m2 = sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
        check(not os.path.exists(os.path.join(out2, "split_pins.json")),
              "a parent with no pins gives a subset with none")
        check(m2.get("split_pins", False) is False,
              "the manifest says the subset pins nothing")
    finally:
        shutil.rmtree(root2, ignore_errors=True)

    root3, parent3 = make_parent("badpins")
    out3 = os.path.join(root3, "kept")
    try:
        write(os.path.join(parent3, "split_pins.json"), "{not json")
        try:
            sub.subset_by_class_count(parent3, out3, MIN_BOXES, log=quiet)
            check(False, "a split_pins.json that cannot be read must be refused")
        except ing.IngestError as exc:
            check("split_pins.json" in str(exc),
                  f"the refusal names the pins file (got {exc})")
        check(not os.path.exists(out3), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root3, ignore_errors=True)


def test_reads_labels_when_there_is_no_localized_folder():
    root, parent = make_parent("plainlabels", label_dir="labels")
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        check(m["label_source"] == "labels",
              f"a parent with only labels/ is read from labels/ (got {m['label_source']})")
        readme = " ".join(read(os.path.join(out, "README.md")).split())
        check("whole-frame" not in readme,
              "the README claims no upstream whole-frame filtering that did not happen")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── refusals over the parent's own files ─────────────────────────────────

def test_bad_label_lines_refused():
    root, parent = make_parent("badline")
    out = os.path.join(root, "kept")
    try:
        write(os.path.join(parent, "labels_localized", "img_a.txt"),
              "0 0.1 0.1 0.2\n")
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "a label line that is not five fields must be refused")
        except ing.IngestError as exc:
            check("img_a.txt" in str(exc) and "5 fields" in str(exc),
                  f"the refusal names the file and the shape (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("badid")
    out2 = os.path.join(root2, "kept")
    try:
        write(os.path.join(parent2, "labels_localized", "img_a.txt"),
              "9 0.1 0.1 0.2 0.2\n")
        try:
            sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
            check(False, "a class id outside the parent class list is refused")
        except ing.IngestError as exc:
            check("img_a.txt" in str(exc) and "9" in str(exc),
                  f"the refusal names the file and the id (got {exc})")
        check(not os.path.exists(out2), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_class_list_disagreement_refused():
    root, parent = make_parent("disagree",
                              classes_txt="Alpha-one\nBeta-two\nOther-name\n")
    out = os.path.join(root, "kept")
    try:
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "a classes.txt that disagrees with data.yaml is refused")
        except ing.IngestError as exc:
            check("classes.txt" in str(exc) and "data.yaml" in str(exc),
                  f"the refusal names both class lists (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_label_without_an_image_refused():
    root, parent = make_parent("noimage")
    out = os.path.join(root, "kept")
    try:
        os.remove(os.path.join(parent, "images", "img_a.jpg"))
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "a surviving label with no image must be refused")
        except ing.IngestError as exc:
            check("img_a" in str(exc),
                  f"the refusal names the record with no image (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_case_twin_names_refused():
    root, parent = make_parent("casetwin")
    out = os.path.join(root, "kept")
    try:
        write_jpeg(os.path.join(parent, "images", "IMG_A.jpg"))
        write(os.path.join(parent, "labels_localized", "IMG_A.txt"),
              "0 0.9 0.9 0.1 0.1\n")
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "two names differing only in letter case are refused")
        except ing.IngestError as exc:
            check("letter case" in str(exc),
                  f"the refusal explains the sync rule (got {exc})")
        check(not os.path.exists(out), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_output_folder_case_twin_refused():
    root, parent = make_parent("foldertwin")
    try:
        os.makedirs(os.path.join(root, "Kept"))
        try:
            sub.subset_by_class_count(parent, os.path.join(root, "kept"),
                                      MIN_BOXES, log=quiet)
            check(False, "a folder matching a sibling only by case is refused")
        except ing.IngestError as exc:
            check("letter case" in str(exc),
                  f"the refusal explains the sync rule (got {exc})")
        check(not os.path.exists(os.path.join(root, "kept")),
              "nothing was written after the refusal")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_empty_and_hostile_parent_parts():
    """A parent whose folders or files are empty, hostile, or the wrong shape."""
    root, parent = make_parent("emptylabels")
    out = os.path.join(root, "kept")
    try:
        for name in os.listdir(os.path.join(parent, "labels_localized")):
            os.remove(os.path.join(parent, "labels_localized", name))
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, "a parent whose label folders hold no label is refused")
        except ing.IngestError as exc:
            check("no label file" in str(exc),
                  f"the refusal says there is nothing to subset (got {exc})")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("noimages")
    out2 = os.path.join(root2, "kept")
    try:
        for name in os.listdir(os.path.join(parent2, "images")):
            os.remove(os.path.join(parent2, "images", name))
        try:
            sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
            check(False, "a parent whose images folder is empty is refused")
        except ing.IngestError as exc:
            check("no image" in str(exc),
                  f"the refusal says the images folder is empty (got {exc})")
    finally:
        shutil.rmtree(root2, ignore_errors=True)

    root3, parent3 = make_parent("hostile")
    out3 = os.path.join(root3, "kept")
    try:
        with open(os.path.join(parent3, "labels_localized", "img_a.txt"),
                  "wb") as fh:
            fh.write(b"\xff\xfe\x00 not text at all")
        try:
            sub.subset_by_class_count(parent3, out3, MIN_BOXES, log=quiet)
            check(False, "a label file that is not text must be refused")
        except ing.IngestError as exc:
            check("img_a.txt" in str(exc) and "could not be read" in str(exc),
                  f"the refusal names the unreadable label (got {exc})")
        check(not os.path.exists(out3), "nothing was written after the refusal")
    finally:
        shutil.rmtree(root3, ignore_errors=True)

    root4, parent4 = make_parent("badmanifest")
    out4 = os.path.join(root4, "kept")
    try:
        write(os.path.join(parent4, "ingest_manifest.json"), "[1, 2, 3]")
        try:
            sub.subset_by_class_count(parent4, out4, MIN_BOXES, log=quiet)
            check(False, "a manifest that is not a JSON object is refused")
        except ing.IngestError as exc:
            check("not a JSON object" in str(exc),
                  f"the refusal says what the manifest must be (got {exc})")
    finally:
        shutil.rmtree(root4, ignore_errors=True)


def test_counting_and_writing_passes_must_agree():
    """The guard that catches a parent changing between the two passes."""
    kept = [(0, 0, "Alpha-one", 4), (1, 1, "Beta-two", 3)]
    try:
        sub._verify_counts({"n_boxes": 6}, kept, "/tmp/somewhere")
        check(False, "a writing pass that wrote fewer boxes must be refused")
    except ing.IngestError as exc:
        check("7" in str(exc) and "6" in str(exc),
              f"the refusal names both counts (got {exc})")
    sub._verify_counts({"n_boxes": 7}, kept, "/tmp/somewhere")
    check(True, "two passes that agree pass the guard")


# ── repeatability and size ───────────────────────────────────────────────

def test_second_read_is_identical():
    root, parent = make_parent("repeat")
    try:
        first = os.path.join(root, "kept")
        second = os.path.join(root, "kept_again")
        sub.subset_by_class_count(parent, first, MIN_BOXES, log=quiet)
        sub.subset_by_class_count(parent, second, MIN_BOXES, log=quiet)
        differ = []
        for base, _dirs, files in os.walk(first):
            for name in files:
                a = os.path.join(base, name)
                b = os.path.join(second, os.path.relpath(a, first))
                if not os.path.isfile(b):
                    differ.append(os.path.relpath(a, first))
                    continue
                if name in ("ingest_manifest.json", "README.md"):
                    continue
                with open(a, "rb") as fa, open(b, "rb") as fb:
                    if fa.read() != fb.read():
                        differ.append(os.path.relpath(a, first))
        check(not differ, f"a parent read twice gives the same files (differ: {differ})")
        m1 = json.loads(read(os.path.join(first, "ingest_manifest.json")))
        m2 = json.loads(read(os.path.join(second, "ingest_manifest.json")))
        for key in TIMESTAMP_KEYS:
            m1.pop(key, None)
            m2.pop(key, None)
        m1.pop("dataset", None)
        m2.pop("dataset", None)
        check(m1 == m2,
              "the two manifests agree apart from the name and the timestamps")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_large_parent_is_streamed():
    root = _mkdtemp("large")
    parent = os.path.join(root, "parent_set")
    out = os.path.join(root, "big")
    try:
        from PIL import Image  # imported before the measurement starts
        Image.new("RGB", (8, 8)).save(os.path.join(root, "seed.jpg"), "JPEG")
        labels = {}
        for i in range(LARGE_IMAGES):
            cls = 0 if i % 2 else 1
            labels[f"rec_{i:05d}"] = "".join(
                f"{cls} 0.5 0.5 0.10 0.10\n" for _ in range(LARGE_LINES_PER_LABEL))
        build_parent(parent, classes=["Alpha-one", "Beta-two"], labels=labels,
                     with_pins=False)
        label_bytes = sum(
            os.path.getsize(os.path.join(parent, "labels_localized", f))
            for f in os.listdir(os.path.join(parent, "labels_localized")))
        tracemalloc.start()
        manifest = sub.subset_by_class_count(parent, out, 100, log=quiet)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        check(manifest["n_images"] == LARGE_IMAGES,
              f"every image of the large parent survives (got {manifest['n_images']})")
        check(manifest["n_boxes"] == LARGE_IMAGES * LARGE_LINES_PER_LABEL,
              f"every box of the large parent survives (got {manifest['n_boxes']})")
        check(peak < LARGE_PEAK_LIMIT_BYTES,
              f"the run never held every label at once: peak {peak} bytes "
              f"against {label_bytes} bytes of labels")
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        shutil.rmtree(root, ignore_errors=True)


# ── command line ─────────────────────────────────────────────────────────

def test_cli_end_to_end():
    root, parent = make_parent("cli")
    try:
        dest_root = os.path.join(root, "staged")
        proc = subprocess.run(
            [PYTHON, SCRIPT, "--parent", parent, "--name", "cli_set",
             "--min-boxes", str(MIN_BOXES), "--dest-root", dest_root],
            capture_output=True, text=True)
        check(proc.returncode == 0,
              f"the command line exits 0 (stderr: {proc.stderr[-400:]})")
        out = os.path.join(dest_root, "cli_set")
        check(os.path.isfile(os.path.join(out, "ingest_manifest.json")),
              "the command line writes the manifest")
        check("4 image(s)" in proc.stdout or "4 images" in proc.stdout,
              f"the run says how many images it wrote (got {proc.stdout!r})")

        bad = subprocess.run(
            [PYTHON, SCRIPT, "--parent", parent, "--name", "cli_set",
             "--min-boxes", "0", "--dest-root", dest_root],
            capture_output=True, text=True)
        check(bad.returncode == 2,
              f"a refusal exits 2 (got {bad.returncode})")
        check("subset stopped:" in bad.stderr,
              f"a refusal says so on stderr (got {bad.stderr!r})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_temp_fixtures_removed():
    """Every temp fixture this suite made is gone, and the module tree is clean.

    Runs last: it reads the list _mkdtemp recorded, so a test that forgot its
    finally block is caught here rather than leaving a folder on the box.
    """
    left = sorted(path for path in _TEMP_DIRS if os.path.exists(path))
    check(not left, f"temp fixtures removed (still present: {left})")
    own_source = read(os.path.abspath(__file__))
    # Spelled in two pieces so this line is not itself one of the matches.
    needle = "tempfile." + "mkdtemp("
    check(own_source.count(needle) == 1,
          "every fixture in this suite is made through _mkdtemp, so this check "
          "sees all of them")
    staged_root = os.path.join(_REPO, "supporting_data", "bbox_datasets")
    if os.path.isdir(staged_root):
        intruders = sorted(set(os.listdir(staged_root)).intersection(WRITTEN_NAMES))
        check(not intruders,
              f"nothing this suite wrote appears in the module tree (found {intruders})")
    else:
        check(True, "the module has no staged bbox_datasets folder to check")


# -- review fixes ---------------------------------------------------------

def test_writers_are_the_stagers_own():
    """The three staged-layout writers are imported from the stager, not copied.

    A subset writes the same text, CSV and JSON files the stagers write. Two
    copies of a writer drift apart the next time one of them is fixed, so the
    subset tool holds the stager's own functions under its own names.
    """
    for name in ("_write_text", "_write_csv", "_write_json"):
        check(getattr(sub, name) is getattr(ing, name),
              f"{name} is the stager's own writer rather than a second copy")
    root, parent = make_parent("writers")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        check(os.path.isfile(os.path.join(out, "class_map.csv"))
              and os.path.isfile(os.path.join(out, "ingest_manifest.json"))
              and os.path.isfile(os.path.join(out, "README.md")),
              "the imported writers still write the CSV, the JSON and the text")
        try:
            sub._write_text(os.path.join(out, "no_such_folder", "deeper.txt"), "x")
            check(False, "a write into a folder that is not there must be refused")
        except ing.IngestError as exc:
            check("could not be written" in str(exc) and "deeper.txt" in str(exc),
                  f"the refusal names the file that could not be written "
                  f"(got {exc})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_license_write_reads_nothing_back():
    """write_license copies or writes the licence and reads no file back off disk.

    The read-back was wasted work on every run, and its one failure message
    called LICENSE.md a label file.
    """
    root, parent = make_parent("licret")
    try:
        source, text = sub.resolve_license(parent, {})
        check(source.endswith("LICENSE.md") and text == "",
              f"a parent carrying LICENSE.md is copied rather than rewritten "
              f"(got {source!r}, {text!r})")
        copied = os.path.join(root, "copied")
        os.makedirs(copied)
        check(sub.write_license(copied, source, text) is None,
              "write_license reads nothing back off disk after the copy")
        check(read(os.path.join(copied, "LICENSE.md")) == FIXTURE_LICENSE,
              "the copied licence is the parent's, byte for byte")
        written = os.path.join(root, "written")
        os.makedirs(written)
        check(sub.write_license(written, "", "# License\n\nterms\n") is None,
              "write_license reads nothing back when it wrote the text itself")
        check(read(os.path.join(written, "LICENSE.md")) == "# License\n\nterms\n",
              "the written licence carries the text it was given")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_background_images_are_counted_on_the_record():
    """A parent image carrying no box is left out, and the count is in the manifest.

    The fixture's img_f carries no box at all, so it was never a candidate for
    the class-count rule. It is still one of the parent images this subset does
    not hold, and the manifest says how many of those there were.
    """
    root, parent = make_parent("background")
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        check(m.get("n_background_dropped") == 1,
              f"the manifest counts the parent image that carried no box at "
              f"all (got {m.get('n_background_dropped')})")
        check(m["n_images_dropped"] == 2,
              f"n_images_dropped still counts every parent image left out "
              f"(got {m['n_images_dropped']})")
        check(m["n_background"] == 0,
              f"the subset itself holds no background image "
              f"(got {m['n_background']})")
        staged = os.listdir(os.path.join(out, "images"))
        check("img_f.jpg" not in staged,
              f"the background image is not copied (got {sorted(staged)})")
        flat = " ".join(read(os.path.join(out, "README.md")).split())
        check("1 of the 6 parent images" in flat,
              "the README gives the background count in words")
        check("n_background_dropped" in flat,
              "the README says which manifest field carries that count")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # A parent with no background image says nothing about one.
    labels = {stem: text for stem, text in FIXTURE_LABELS.items() if text.split()}
    root2, parent2 = make_parent("nobackground", labels=labels)
    out2 = os.path.join(root2, "kept")
    try:
        m2 = sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
        check(m2.get("n_background_dropped") == 0,
              f"a parent with no empty label file counts none "
              f"(got {m2.get('n_background_dropped')})")
        flat2 = " ".join(read(os.path.join(out2, "README.md")).split())
        check("n_background_dropped" not in flat2,
              "the README of a subset with no background image says nothing "
              "about one")
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_readme_table_names_the_label_folder():
    """The README's box total says which parent label folder it counted.

    The parent's own card on the Bounding Boxes page counts labels/, which on a
    Coral Intel parent is larger than labels_localized/ by the whole-frame
    boxes. Two documents giving two totals for one parent need the folder named
    on the number people quote.
    """
    root, parent = make_parent("table")
    out = os.path.join(root, "kept")
    try:
        sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        text = read(os.path.join(out, "README.md"))
        check("| boxes in labels_localized/ |" in text,
              "the boxes row of the README table names the parent label "
              "folder the totals were counted in")
        flat = " ".join(text.split())
        check("Bounding Boxes page" in flat and "labels/" in flat,
              "the README says why the parent's card can show a larger box "
              "total than this table")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    root2, parent2 = make_parent("table_plain", label_dir="labels")
    out2 = os.path.join(root2, "kept")
    try:
        sub.subset_by_class_count(parent2, out2, MIN_BOXES, log=quiet)
        text2 = read(os.path.join(out2, "README.md"))
        check("| boxes in labels/ |" in text2,
              "a parent read from labels/ names that folder instead")
        check("Bounding Boxes page" not in " ".join(text2.split()),
              "a parent with one label folder is promised no second total")
    finally:
        shutil.rmtree(root2, ignore_errors=True)


def test_dest_must_sit_in_a_folder_that_exists():
    """A subset folder is refused when the folder meant to hold it is not there.

    The command line joins a checked name to --dest-root, so this only reaches
    an in-process caller, for example a route that builds a subset from the
    page. Without the check, os.makedirs creates whatever intermediate folders
    the path named.
    """
    root, parent = make_parent("destparent")
    holder = os.path.join(root, "not_made_yet")
    try:
        try:
            sub.subset_by_class_count(parent, os.path.join(holder, "kept"),
                                      MIN_BOXES, log=quiet)
            check(False,
                  "a subset folder whose holding folder is not there must be "
                  "refused")
        except ing.IngestError as exc:
            check(holder in str(exc) and "is not there" in str(exc),
                  f"the refusal names the folder that is missing (got {exc})")
        check(not os.path.exists(holder),
              "no intermediate folder was created by the refused run")

        blocker = os.path.join(root, "a_file")
        write(blocker, "not a folder\n")
        try:
            sub.subset_by_class_count(parent, os.path.join(blocker, "kept"),
                                      MIN_BOXES, log=quiet)
            check(False, "a subset folder held by a file must be refused")
        except ing.IngestError as exc:
            check(blocker in str(exc) and "is not there" in str(exc),
                  f"the refusal names the path that is not a folder "
                  f"(got {exc})")
        check(read(blocker) == "not a folder\n",
              "the refused run left the blocking file alone")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_suite_is_registered_in_the_test_board():
    """The full test board runs this suite, so it cannot rot unnoticed.

    The board's suite list is hardcoded. A suite missing from it never runs
    again unless someone invokes the file by hand, and the board still reports
    every suite green.
    """
    board = os.path.join(_SCRIPTS, "run_all_tests.sh")
    check(os.path.isfile(board), f"the test board is at {board}")
    if not os.path.isfile(board):
        return
    text = read(board)
    check('"py _boxes/tests/test_subset_by_class_count.py"' in text,
          "the full test board runs this suite beside the other _boxes suites")


def test_subject_follows_the_data_into_the_subset():
    """A subset is its parent's images with some classes removed, so it is
    about what the parent was about. The Train panel reads the same word off
    both, and starts its Subject control there."""
    root, parent = make_parent("subject", manifest_extra={"subject": "coral"})
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        check(m["subject"] == "coral",
              f"the parent's subject must reach the subset (got {m.get('subject')!r})")
        disk = json.loads(read(os.path.join(out, "ingest_manifest.json")))
        check(disk["subject"] == "coral",
              f"and must be on disk (got {disk.get('subject')!r})")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_parent_with_no_subject_gives_a_subset_with_none():
    """Every dataset staged before 2026-09-15 records no subject, and a subset
    of one must still be written rather than refused."""
    root, parent = make_parent("nosubject")
    out = os.path.join(root, "kept")
    try:
        m = sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
        check(m["subject"] == "",
              f"a parent with no subject gives a subset with none (got {m.get('subject')!r})")
        check(os.path.isdir(os.path.join(out, "images")), "and the subset is still written")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_parent_subject_outside_the_vocabulary_stops_the_run():
    """Carrying a hand-edited word forward would give the subset a subject no
    training recipe can ever match. The refusal names the file to fix, and
    nothing is written."""
    for bad in ("reef", "corals", 7):
        root, parent = make_parent("badsubject", manifest_extra={"subject": bad})
        out = os.path.join(root, "kept")
        try:
            sub.subset_by_class_count(parent, out, MIN_BOXES, log=quiet)
            check(False, f"subject {bad!r} must stop the run")
        except sub.SubsetError as exc:
            check(str(bad) in str(exc), f"the refusal names the value: {exc}")
            check("ingest_manifest.json" in str(exc),
                  f"the refusal names the file to fix: {exc}")
        finally:
            check(not os.path.exists(out), "nothing is written after the refusal")
            shutil.rmtree(root, ignore_errors=True)


TESTS = [
    test_normalise_min_boxes,
    test_parent_must_be_a_staged_dataset,
    test_output_name_and_existing_folder,
    test_minimum_that_keeps_no_class,
    test_layout_and_renumbering,
    test_class_map_and_dropped_csv,
    test_boundary_minimum,
    test_manifest_matches_disk,
    test_license_carried_over,
    test_license_written_from_the_manifest_note,
    test_readme_says_what_the_subset_is,
    test_names_csv_narrowed,
    test_split_pins_narrowed,
    test_reads_labels_when_there_is_no_localized_folder,
    test_bad_label_lines_refused,
    test_class_list_disagreement_refused,
    test_label_without_an_image_refused,
    test_case_twin_names_refused,
    test_output_folder_case_twin_refused,
    test_empty_and_hostile_parent_parts,
    test_counting_and_writing_passes_must_agree,
    test_second_read_is_identical,
    test_large_parent_is_streamed,
    test_cli_end_to_end,
    test_writers_are_the_stagers_own,
    test_license_write_reads_nothing_back,
    test_background_images_are_counted_on_the_record,
    test_readme_table_names_the_label_folder,
    test_dest_must_sit_in_a_folder_that_exists,
    test_subject_follows_the_data_into_the_subset,
    test_a_parent_with_no_subject_gives_a_subset_with_none,
    test_a_parent_subject_outside_the_vocabulary_stops_the_run,
    test_suite_is_registered_in_the_test_board,
    test_temp_fixtures_removed,
]


def main():
    print("class-count subset of a staged box dataset:")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\n{_pass} passed, {_fail} failed")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
