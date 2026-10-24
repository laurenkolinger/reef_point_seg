"""The VEGA tutorial box dataset: the invariants a student's run depends on.

No pytest. Run with the unified env python:
    env/bin/python scripts/_tutorial/tests/test_tutorial_set.py

Two halves, both of them run by __main__:

  BUILT SET   the dataset on disk at
              supporting_data/bbox_datasets/tutorial_boxes is read and
              checked: the staged files are all there, every picture has a
              label and every label a picture, the class list agrees with
              data.yaml and with names.csv, every box names a class the
              dataset declares, the per-class instance counts are what the
              manifest says, split_pins.json pins every picture and only
              pictures that are there, the split sizes are right, and every
              class appears in all three splits. A class absent from the test
              split leaves the evaluation report without a row for it, which
              is the number the tutorial's evaluation lesson reads, so that
              one is checked class by class rather than in total.

  BUILDER     the tool itself, over fixtures in a temp folder: the happy path
              end to end, the split is reproducible from the seed, a picture
              carrying a class this build drops is passed over rather than
              copied with half its annotation, and every refusal the tool
              promises. The adversarial cases are a class the parent does not
              declare, one class named twice, one class only, split counts
              that do not add up, a per-class count of zero, a fractional
              count, a boolean count, a class the parent has too few pictures
              of, a destination that already exists, a destination whose
              holding folder is not there, a name outside the plain-name rule,
              a parent that is not a staged dataset, a parent whose data.yaml
              cannot be read, an empty label file, a label line with four
              fields, a label line of polygon points, a label naming a class
              id the parent does not declare, a label file with no picture,
              and a parent with no licence terms at all.

Nothing here trains, reaches a GPU, opens a port, or writes anywhere but the
one temp folder, which is removed in a finally block and whose removal is
verified. The built set is read and never written.
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
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUTORIAL = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_TUTORIAL)
_REPO = os.path.dirname(_SCRIPTS)
for _path in (_SCRIPTS, _TUTORIAL):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_tutorial_set as bts  # noqa: E402
from _boxes import ingest_yolo_boxes as ing  # noqa: E402

# The dataset this suite checks, and the staged tree it lives in.
BBOX_ROOT = os.path.join(_REPO, "supporting_data", "bbox_datasets")
BUILT = os.path.join(BBOX_ROOT, "tutorial_boxes")

# What the built set is expected to hold. These are the tool's own defaults,
# read from the tool rather than retyped, so a change to the defaults that is
# not carried into a rebuild shows up here as a failure instead of passing
# quietly.
EXPECT_CLASSES = list(bts.TUTORIAL_CLASSES)
EXPECT_PER_CLASS = bts.IMAGES_PER_CLASS
EXPECT_QUOTAS = {"train": bts.TRAIN_PER_CLASS,
                 "valid": bts.VALID_PER_CLASS,
                 "test": bts.TEST_PER_CLASS}
EXPECT_IMAGES = EXPECT_PER_CLASS * len(EXPECT_CLASSES)

# Every file a staged dataset owes a reader, plus the two this tool adds.
REQUIRED_FILES = ("classes.txt", "data.yaml", "names.csv", "class_map.csv",
                  "dropped_classes.csv", "split_pins.json", "LICENSE.md",
                  "README.md", "ingest_manifest.json")

# Fixture sizes for the builder half. Small enough to write in a moment, large
# enough that a quota of three splits into one of each.
FIXTURE_PER_CLASS = 9
FIXTURE_QUOTAS = {"train": 5, "valid": 2, "test": 2}
FIXTURE_CLASSES = ("Alpha-one", "Beta-two", "Gamma-three")
FIXTURE_EXTRA_CLASS = "Delta-four"
FIXTURE_IMAGES_EACH = 12
# The quota that takes every eligible picture when one of a class's pictures
# has been made ineligible, so a test can prove which picture was left out.
FIXTURE_ALL_BUT_ONE = FIXTURE_IMAGES_EACH - 1
FIXTURE_ALL_BUT_ONE_QUOTAS = {"train": 7, "valid": 2, "test": 2}

# A one pixel JPEG, so a fixture picture is a real image file without Pillow
# being asked to draw anything.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffc20011"
    "08000100010101110000ffc40014000100000000000000000000"
    "00000000000009ffda0008010100000000013fffc40014010100"
    "0000000000000000000000000000ffda0008010100010502ffc4"
    "0014100100000000000000000000000000000000ffda00080101"
    "00010500013fffc40014110100000000000000000000000000000"
    "000ffda0008010100063f02ffc40014100100000000000000000"
    "00000000000000000ffda0008010100013f21ffda0008010100"
    "00000010ffc4001410010000000000000000000000000000000"
    "0ffda0008010100013f10ffd9")

_RESULTS = []
_FIXTURE_ROOTS = []


def check(cond, msg):
    """Raise AssertionError with ``msg`` when ``cond`` is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, record its outcome, print a per-test line."""
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as exc:
        _RESULTS.append((fn.__name__, False, f"{exc}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {exc}")


# ── Readers for the built set ────────────────────────────────────────────

def read_json(path):
    """One JSON document from the built set."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_lines(path):
    """The non-blank lines of one text file, stripped."""
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def built_classes():
    """The class tokens the built set declares, in class-id order."""
    return read_lines(os.path.join(BUILT, "classes.txt"))


def built_images():
    """The image basenames in the built set, sorted."""
    folder = os.path.join(BUILT, "images")
    return sorted(name for name in os.listdir(folder)
                  if os.path.splitext(name)[1].lower() in ing.IMAGE_EXTS)


def built_labels():
    """The label basenames in the built set, sorted."""
    folder = os.path.join(BUILT, "labels")
    return sorted(name for name in os.listdir(folder) if name.endswith(".txt"))


def built_boxes():
    """(by_class, by_class_split, n_boxes) read from the built labels and pins.

    by_class counts boxes; by_class_split counts pictures, so a picture with
    two boxes of one class counts once for that class in its split.
    """
    classes = built_classes()
    pins = read_json(os.path.join(BUILT, "split_pins.json"))["pins"]
    stem_split = {os.path.splitext(k)[0]: v for k, v in pins.items()}
    by_class, by_class_split, n_boxes = {}, {}, 0
    for name in built_labels():
        stem = name[:-4]
        with open(os.path.join(BUILT, "labels", name), encoding="utf-8") as fh:
            boxes = ing.parse_boxes(fh.read(), len(classes), name)
        n_boxes += len(boxes)
        split = stem_split[stem]
        for class_id in {c for c, _coords in boxes}:
            key = (class_id, split)
            by_class_split[key] = by_class_split.get(key, 0) + 1
        for class_id, _coords in boxes:
            by_class[class_id] = by_class.get(class_id, 0) + 1
    return by_class, by_class_split, n_boxes


# ── The built set ────────────────────────────────────────────────────────

def test_the_built_set_is_there_with_every_staged_file():
    check(os.path.isdir(BUILT), f"{BUILT} is not there")
    for sub_dir in ("images", "labels"):
        check(os.path.isdir(os.path.join(BUILT, sub_dir)),
              f"{BUILT} has no {sub_dir}/ folder")
    missing = [name for name in REQUIRED_FILES
               if not os.path.isfile(os.path.join(BUILT, name))]
    check(not missing, f"{BUILT} is missing {', '.join(missing)}")


def test_image_and_label_counts_match_the_manifest():
    manifest = read_json(os.path.join(BUILT, "ingest_manifest.json"))
    images, labels = built_images(), built_labels()
    check(len(images) == EXPECT_IMAGES,
          f"{len(images)} picture(s) on disk, {EXPECT_IMAGES} expected")
    check(len(labels) == EXPECT_IMAGES,
          f"{len(labels)} label file(s) on disk, {EXPECT_IMAGES} expected")
    check(manifest["n_images"] == EXPECT_IMAGES,
          f"the manifest says {manifest['n_images']} pictures")


def test_every_picture_has_a_label_and_every_label_a_picture():
    image_stems = {os.path.splitext(name)[0] for name in built_images()}
    label_stems = {name[:-4] for name in built_labels()}
    lonely = image_stems ^ label_stems
    check(not lonely,
          f"{len(lonely)} name(s) unpaired, first: {sorted(lonely)[:5]}")


def test_no_label_file_is_empty():
    empty = [name for name in built_labels()
             if not read_lines(os.path.join(BUILT, "labels", name))]
    check(not empty,
          f"{len(empty)} label file(s) hold no box, first: {empty[:5]}")


def test_the_class_lists_agree_with_one_another():
    classes = built_classes()
    check(classes == EXPECT_CLASSES,
          f"classes.txt holds {classes}, expected {EXPECT_CLASSES}")
    declared = ing.read_class_list(os.path.join(BUILT, "data.yaml"))
    check(declared == classes,
          f"data.yaml holds {declared} and classes.txt holds {classes}")
    named = ing.read_names_csv(os.path.join(BUILT, "names.csv"))
    check([token for token, _name in named] == classes,
          f"names.csv names {[t for t, _n in named]}")
    for token, full in named:
        check(full == token.replace("-", " "),
              f"names.csv gives {token!r} the full name {full!r}")


def test_every_box_names_a_class_the_dataset_declares():
    classes = built_classes()
    for name in built_labels():
        with open(os.path.join(BUILT, "labels", name), encoding="utf-8") as fh:
            ing.parse_boxes(fh.read(), len(classes), name)


def test_instances_per_class_match_the_class_map_and_the_manifest():
    by_class, _by_split, n_boxes = built_boxes()
    manifest = read_json(os.path.join(BUILT, "ingest_manifest.json"))
    check(n_boxes == manifest["n_boxes"],
          f"{n_boxes} box(es) on disk, the manifest says {manifest['n_boxes']}")
    rows = read_lines(os.path.join(BUILT, "class_map.csv"))[1:]
    for row in rows:
        _old, new_id, token, boxes = row.split(",")
        new_id, boxes = int(new_id), int(boxes)
        check(built_classes()[new_id] == token,
              f"class_map.csv row {row!r} names the wrong class")
        check(by_class.get(new_id, 0) == boxes,
              f"class_map.csv says {token} has {boxes} box(es), the labels "
              f"hold {by_class.get(new_id, 0)}")
    check(len(rows) == len(EXPECT_CLASSES),
          f"class_map.csv holds {len(rows)} row(s)")
    check(all(by_class.get(i, 0) > 0 for i in range(len(EXPECT_CLASSES))),
          f"a class carries no box at all: {by_class}")


def test_split_pins_cover_every_picture_and_nothing_else():
    doc = read_json(os.path.join(BUILT, "split_pins.json"))
    check(isinstance(doc, dict) and isinstance(doc.get("pins"), dict),
          "split_pins.json is not an object carrying a pins object")
    pins, images = doc["pins"], set(built_images())
    check(set(pins) == images,
          f"{len(images - set(pins))} picture(s) unpinned and "
          f"{len(set(pins) - images)} pin(s) name no picture")
    bad = {k: v for k, v in pins.items() if v not in ing.SPLIT_DIRS}
    check(not bad, f"{len(bad)} pin(s) name an unknown split: {list(bad)[:5]}")


def test_the_split_builder_reads_the_pins_this_file_writes():
    """The one reader that matters agrees with this file, keys and all."""
    sys.path.insert(0, os.path.join(_REPO, "scripts", "TCRMPtrain_oceankindCV",
                                    "src"))
    import detect_dataset as dd
    pins = dd.read_source_pins(BUILT)
    check(set(pins) == set(built_images()),
          "the split builder reads a different set of pins than this dataset "
          "holds, so the holdout would move")
    pairs, unlabelled = dd.bbox_pairs(BUILT)
    check(unlabelled == 0, f"{unlabelled} picture(s) have no label")
    names = [p[0] for p in pairs]
    replayed = dd.assign_split(names, "tutorial-seed", pins)
    counts = {s: sum(1 for v in replayed.values() if v == s)
              for s in ing.SPLIT_DIRS}
    wanted = {s: EXPECT_QUOTAS[s] * len(EXPECT_CLASSES) for s in ing.SPLIT_DIRS}
    check(counts == wanted,
          f"the split builder replays {counts}, expected {wanted}")


def test_the_split_sizes_are_what_was_asked_for():
    manifest = read_json(os.path.join(BUILT, "ingest_manifest.json"))
    by_split = manifest["pins_by_split"]
    for split in ing.SPLIT_DIRS:
        wanted = EXPECT_QUOTAS[split] * len(EXPECT_CLASSES)
        check(by_split[split] == wanted,
              f"the {split} split holds {by_split[split]}, expected {wanted}")
    check(sum(by_split.values()) == EXPECT_IMAGES,
          f"the three splits hold {sum(by_split.values())} picture(s)")


def test_every_class_appears_in_all_three_splits():
    _by_class, by_class_split, _n = built_boxes()
    classes = built_classes()
    missing = [f"{classes[i]} in {split}"
               for i in range(len(classes)) for split in ing.SPLIT_DIRS
               if by_class_split.get((i, split), 0) == 0]
    check(not missing,
          "a class is absent from a split, so the evaluation report would "
          f"have no row for it: {', '.join(missing)}")
    wrong = [f"{classes[i]} has {by_class_split.get((i, split), 0)} in {split}"
             for i in range(len(classes)) for split in ing.SPLIT_DIRS
             if by_class_split.get((i, split), 0) != EXPECT_QUOTAS[split]]
    check(not wrong, f"the split is uneven: {', '.join(wrong)}")


def test_the_manifest_records_what_the_page_and_the_builder_read():
    manifest = read_json(os.path.join(BUILT, "ingest_manifest.json"))
    check(manifest["split_pins"] is True and manifest["n_pinned"] == EXPECT_IMAGES,
          f"the manifest records {manifest['n_pinned']} pin(s)")
    check(manifest["subject"] in ing.DATASET_SUBJECTS,
          f"subject {manifest['subject']!r} is outside the Train panel's "
          f"vocabulary {ing.DATASET_SUBJECTS}")
    check(manifest["classes"] == EXPECT_CLASSES,
          f"the manifest names {manifest['classes']}")
    check(manifest["n_classes"] == len(EXPECT_CLASSES), "n_classes is wrong")
    check(manifest["read_only"] is True and manifest["purpose"] == "vega_tutorial",
          "the manifest does not record this as the read-only tutorial set")
    check(manifest["images_per_class"] == EXPECT_PER_CLASS
          and manifest["images_per_class_by_split"] == EXPECT_QUOTAS,
          "the manifest does not record the per-class counts that were used")
    check(str(manifest.get("split_seed") or ""),
          "the manifest records no seed, so the split cannot be reproduced")
    check(manifest["n_background"] == 0,
          f"{manifest['n_background']} background picture(s) recorded")


def test_the_boxes_page_lists_the_built_set_with_its_pins():
    """The listing the Bounding Boxes page reads sees this dataset whole."""
    from _boxes import blueprint
    entries = {d["name"]: d for d in blueprint.list_datasets(BBOX_ROOT)}
    check("tutorial_boxes" in entries,
          f"the page lists {sorted(entries)} and not tutorial_boxes")
    entry = entries["tutorial_boxes"]
    check(entry["n_images"] == EXPECT_IMAGES, "the card would show the wrong count")
    check(entry["has_split_pins"] is True and entry["split_pins_unreadable"] is False,
          "the page cannot read this dataset's split pins")
    wanted = {s: EXPECT_QUOTAS[s] * len(EXPECT_CLASSES) for s in ing.SPLIT_DIRS}
    check(entry["split_counts"] == wanted,
          f"the card would show {entry['split_counts']}, expected {wanted}")
    check(entry["n_names"] == len(EXPECT_CLASSES),
          f"the card reads {entry['n_names']} full name(s)")
    check(entry["has_license_file"] is True, "the card finds no LICENSE.md")


def test_the_readme_follows_the_prose_rules():
    text = open(os.path.join(BUILT, "README.md"), encoding="utf-8").read()
    for dash in ("—", "–"):
        check(dash not in text, f"the README carries a {dash!r}")
    lowered = text.lower()
    for word in (" land ", " lands ", " landed ", " landing "):
        check(word not in lowered, f"the README carries{word.rstrip()}")
    check(max(ord(ch) for ch in text) < 0x2100,
          "the README carries a character outside plain text and italics")
    check("read-only" in lowered or "read only" in lowered,
          "the README does not say the folder is read only")
    check("tutorial" in lowered, "the README does not say what it is for")
    check("Lauren Olinger" in text, "the README carries no attribution")


# ── Builder fixtures ─────────────────────────────────────────────────────

def make_root():
    """One temp folder, remembered so the final sweep removes it."""
    root = tempfile.mkdtemp(prefix="tutorial_set_test_")
    _FIXTURE_ROOTS.append(root)
    return root


def write(path, text):
    """Write one small text file, making its folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def make_parent(root, classes=FIXTURE_CLASSES, per_class=FIXTURE_IMAGES_EACH,
                name="parent", license_note="Fixture terms.",
                with_license_file=True, label_text=None, extra_lines=None):
    """A staged bounding-box dataset to subset, written under ``root``.

    Every picture carries one box of its own class, named
    "<class token>_<n>.jpg", which is the shape of the real parent.
    """
    parent = os.path.join(root, name)
    os.makedirs(os.path.join(parent, "images"))
    os.makedirs(os.path.join(parent, "labels"))
    for class_id, token in enumerate(classes):
        for n in range(per_class):
            stem = f"{token}_{n}"
            with open(os.path.join(parent, "images", stem + ".jpg"), "wb") as fh:
                fh.write(_TINY_JPEG)
            body = (label_text if label_text is not None
                    else f"{class_id} 0.5 0.5 0.4 0.4\n")
            if extra_lines:
                body += extra_lines
            write(os.path.join(parent, "labels", stem + ".txt"), body)
    write(os.path.join(parent, "classes.txt"),
          "".join(token + "\n" for token in classes))
    write(os.path.join(parent, "data.yaml"),
          "path: .\nnames:\n"
          + "".join(f"  {i}: {t}\n" for i, t in enumerate(classes))
          + f"nc: {len(classes)}\n")
    if with_license_file:
        write(os.path.join(parent, "LICENSE.md"), "# License\n\nFixture.\n")
    write(os.path.join(parent, "ingest_manifest.json"), json.dumps({
        "dataset": name, "n_images": len(classes) * per_class,
        "n_boxes": len(classes) * per_class, "n_classes": len(classes),
        "classes": list(classes), "subject": "coral",
        "citation": "Fixture citation.", "license_note": license_note,
        "source_path": os.path.join(root, "upstream"),
    }, indent=2))
    return parent


def build(parent, root, name="tutorial_boxes", classes=FIXTURE_CLASSES[:2],
          per_class=FIXTURE_PER_CLASS, quotas=None, seed="fixture-seed"):
    """Run the builder over a fixture, quietly."""
    quotas = quotas or FIXTURE_QUOTAS
    return bts.build_tutorial_set(
        parent, os.path.join(root, name), classes=list(classes),
        per_class=per_class, train=quotas["train"], valid=quotas["valid"],
        test=quotas["test"], seed=seed, log=lambda *a, **k: None)


def refused(fn, what):
    """Run ``fn`` and return the refusal message it raised.

    Raises AssertionError when it wrote something instead of refusing.
    """
    try:
        fn()
    except ing.IngestError as exc:
        return str(exc)
    raise AssertionError(f"{what} was accepted and should have been refused")


# ── The builder ──────────────────────────────────────────────────────────

def test_builder_writes_a_whole_staged_dataset():
    root = make_root()
    parent = make_parent(root)
    manifest = build(parent, root)
    dest = os.path.join(root, "tutorial_boxes")
    check(manifest["n_images"] == FIXTURE_PER_CLASS * 2,
          f"built {manifest['n_images']} picture(s)")
    missing = [n for n in REQUIRED_FILES
               if not os.path.isfile(os.path.join(dest, n))]
    check(not missing, f"the build is missing {', '.join(missing)}")
    images = os.listdir(os.path.join(dest, "images"))
    labels = os.listdir(os.path.join(dest, "labels"))
    check(len(images) == len(labels) == FIXTURE_PER_CLASS * 2,
          f"{len(images)} picture(s) beside {len(labels)} label file(s)")
    declared = ing.read_class_list(os.path.join(dest, "data.yaml"))
    check(declared == list(FIXTURE_CLASSES[:2]), f"data.yaml declares {declared}")
    for name in labels:
        with open(os.path.join(dest, "labels", name), encoding="utf-8") as fh:
            for line in fh.read().splitlines():
                check(0 <= int(line.split()[0]) < 2,
                      f"{name} carries a class id outside the new list: {line}")


def test_builder_pins_every_class_into_every_split():
    root = make_root()
    parent = make_parent(root)
    build(parent, root)
    pins = read_json(os.path.join(root, "tutorial_boxes",
                                  "split_pins.json"))["pins"]
    seen = {}
    for image, split in pins.items():
        token = image.rsplit("_", 1)[0]
        seen.setdefault(token, {}).setdefault(split, 0)
        seen[token][split] += 1
    for token in FIXTURE_CLASSES[:2]:
        for split, wanted in FIXTURE_QUOTAS.items():
            got = seen.get(token, {}).get(split, 0)
            check(got == wanted,
                  f"{token} has {got} picture(s) in {split}, expected {wanted}")


def test_the_same_seed_chooses_the_same_pictures_and_the_same_holdout():
    root = make_root()
    parent = make_parent(root)
    build(parent, root, name="first")
    build(parent, root, name="second")
    first = read_json(os.path.join(root, "first", "split_pins.json"))["pins"]
    second = read_json(os.path.join(root, "second", "split_pins.json"))["pins"]
    check(first == second,
          "two builds with one seed pinned different pictures, so two "
          "students would evaluate on different holdouts")


def test_a_different_seed_chooses_a_different_holdout():
    root = make_root()
    parent = make_parent(root)
    build(parent, root, name="one", seed="fixture-seed")
    build(parent, root, name="two", seed="another-seed")
    one = read_json(os.path.join(root, "one", "split_pins.json"))["pins"]
    two = read_json(os.path.join(root, "two", "split_pins.json"))["pins"]
    check(one != two, "a different seed pinned exactly the same holdout")


def test_a_picture_carrying_a_dropped_class_is_passed_over():
    root = make_root()
    parent = make_parent(root, classes=FIXTURE_CLASSES + (FIXTURE_EXTRA_CLASS,))
    # One picture of Alpha-one also carries a box of the class this build drops.
    mixed = os.path.join(parent, "labels", f"{FIXTURE_CLASSES[0]}_0.txt")
    write(mixed, "0 0.5 0.5 0.4 0.4\n3 0.2 0.2 0.1 0.1\n")
    manifest = build(parent, root, per_class=FIXTURE_ALL_BUT_ONE,
                     quotas=FIXTURE_ALL_BUT_ONE_QUOTAS)
    check(manifest["n_images_mixed_class_skipped"] == 1,
          f"{manifest['n_images_mixed_class_skipped']} mixed picture(s) "
          "counted, expected 1")
    copied = os.listdir(os.path.join(root, "tutorial_boxes", "images"))
    check(f"{FIXTURE_CLASSES[0]}_0.jpg" not in copied,
          "a picture carrying a dropped class was copied, so the detector "
          "would be taught that species is background")


def test_the_verify_step_reads_the_folder_back():
    root = make_root()
    parent = make_parent(root)
    build(parent, root)
    counts = bts.verify(os.path.join(root, "tutorial_boxes"), FIXTURE_QUOTAS,
                        lambda *a, **k: None)
    check(counts["by_split"] == {k: v * 2 for k, v in FIXTURE_QUOTAS.items()},
          f"verify counted {counts['by_split']}")


def test_verify_refuses_a_folder_whose_label_was_removed():
    root = make_root()
    parent = make_parent(root)
    build(parent, root)
    dest = os.path.join(root, "tutorial_boxes")
    victim = sorted(os.listdir(os.path.join(dest, "labels")))[0]
    os.remove(os.path.join(dest, "labels", victim))
    msg = refused(lambda: bts.verify(dest, FIXTURE_QUOTAS, lambda *a, **k: None),
                  "a folder with a missing label")
    check("label" in msg, f"the refusal does not name the fault: {msg}")


# ── Refusals ─────────────────────────────────────────────────────────────

def test_a_class_the_parent_does_not_declare_is_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(lambda: build(parent, root, classes=("Alpha-one", "Nope-nine")),
                  "an unknown class")
    check("Nope-nine" in msg, f"the refusal does not name the class: {msg}")
    check(not os.path.exists(os.path.join(root, "tutorial_boxes")),
          "a refused build left a folder behind")


def test_one_class_named_twice_is_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(lambda: build(parent, root, classes=("Alpha-one", "Alpha-one")),
                  "a class named twice")
    check("twice" in msg, f"the refusal does not say what is wrong: {msg}")


def test_a_single_class_is_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(lambda: build(parent, root, classes=("Alpha-one",)),
                  "a one-class set")
    check("two classes" in msg, f"the refusal does not say why: {msg}")


def test_split_counts_that_do_not_add_up_are_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(
        lambda: build(parent, root,
                      quotas={"train": 5, "valid": 2, "test": 1}),
        "split counts that miss a picture")
    check("add up" in msg, f"the refusal does not say why: {msg}")


def test_a_zero_or_fractional_or_boolean_count_is_refused():
    root = make_root()
    parent = make_parent(root)
    for quotas, what in (({"train": 5, "valid": 2, "test": 0}, "a zero split"),
                         ({"train": 5.5, "valid": 2, "test": 2}, "a fraction"),
                         ({"train": True, "valid": 2, "test": 2}, "a boolean")):
        msg = refused(lambda q=quotas: build(parent, root, quotas=q), what)
        check("whole number" in msg, f"{what}: {msg}")


def test_a_class_the_parent_has_too_few_pictures_of_is_refused():
    root = make_root()
    parent = make_parent(root, per_class=4)
    msg = refused(lambda: build(parent, root, per_class=FIXTURE_PER_CLASS),
                  "a quota the parent cannot fill")
    check("has 4" in msg, f"the refusal does not say how many: {msg}")
    check("--per-class" in msg, f"the refusal offers no remedy: {msg}")


def test_a_destination_that_already_exists_is_refused():
    root = make_root()
    parent = make_parent(root)
    os.makedirs(os.path.join(root, "tutorial_boxes"))
    msg = refused(lambda: build(parent, root), "an occupied destination")
    check("already exists" in msg, f"the refusal does not say why: {msg}")


def test_a_destination_whose_holder_is_not_there_is_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(
        lambda: bts.build_tutorial_set(
            parent, os.path.join(root, "typo", "tutorial_boxes"),
            classes=list(FIXTURE_CLASSES[:2]), per_class=FIXTURE_PER_CLASS,
            train=FIXTURE_QUOTAS["train"], valid=FIXTURE_QUOTAS["valid"],
            test=FIXTURE_QUOTAS["test"], log=lambda *a, **k: None),
        "a destination inside a folder that is not there")
    check("is not there" in msg, f"the refusal does not say why: {msg}")


def test_a_name_outside_the_plain_name_rule_is_refused():
    root = make_root()
    parent = make_parent(root)
    msg = refused(lambda: build(parent, root, name="tutorial boxes!"),
                  "a name with a space and a bang")
    check("name" in msg.lower(), f"the refusal does not name the rule: {msg}")


def test_a_parent_that_is_not_a_staged_dataset_is_refused():
    root = make_root()
    empty = os.path.join(root, "not_staged")
    os.makedirs(empty)
    msg = refused(lambda: build(empty, root), "a folder that is not staged")
    check("ingest_manifest.json" in msg, f"the refusal is unclear: {msg}")


def test_a_parent_whose_data_yaml_is_broken_is_refused():
    root = make_root()
    parent = make_parent(root)
    write(os.path.join(parent, "data.yaml"), "names: not a mapping at all\n")
    msg = refused(lambda: build(parent, root), "an unreadable data.yaml")
    check("data.yaml" in msg or "names" in msg, f"the refusal is unclear: {msg}")


def test_a_parent_whose_class_lists_disagree_is_refused():
    root = make_root()
    parent = make_parent(root)
    write(os.path.join(parent, "classes.txt"), "Alpha-one\nSomething-else\n")
    msg = refused(lambda: build(parent, root), "class lists that disagree")
    check("classes.txt" in msg, f"the refusal does not name the file: {msg}")


def test_an_empty_label_file_leaves_its_picture_out():
    root = make_root()
    parent = make_parent(root)
    write(os.path.join(parent, "labels", f"{FIXTURE_CLASSES[0]}_0.txt"), "")
    manifest = build(parent, root, per_class=FIXTURE_ALL_BUT_ONE,
                     quotas=FIXTURE_ALL_BUT_ONE_QUOTAS)
    copied = os.listdir(os.path.join(root, "tutorial_boxes", "images"))
    check(f"{FIXTURE_CLASSES[0]}_0.jpg" not in copied,
          "a picture whose label holds no box was copied")
    check(manifest["n_background"] == 0, "a background picture was staged")


def test_a_label_line_with_four_fields_is_refused():
    root = make_root()
    parent = make_parent(root, label_text="0 0.5 0.5 0.4\n")
    msg = refused(lambda: build(parent, root), "a truncated label line")
    check("5 fields" in msg or "fields" in msg, f"the refusal is unclear: {msg}")


def test_a_polygon_label_is_refused():
    root = make_root()
    parent = make_parent(
        root, label_text="0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n")
    msg = refused(lambda: build(parent, root), "a polygon label")
    check("segmentation" in msg or "fields" in msg, f"the refusal is unclear: {msg}")


def test_a_label_naming_an_undeclared_class_id_is_refused():
    root = make_root()
    parent = make_parent(root, label_text="99 0.5 0.5 0.4 0.4\n")
    msg = refused(lambda: build(parent, root), "a class id past the class list")
    check("outside" in msg, f"the refusal is unclear: {msg}")


def test_a_label_with_no_picture_is_refused():
    root = make_root()
    parent = make_parent(root)
    write(os.path.join(parent, "labels", "orphan.txt"), "0 0.5 0.5 0.4 0.4\n")
    msg = refused(lambda: build(parent, root), "a label with no picture")
    check("no image" in msg, f"the refusal is unclear: {msg}")


def test_a_parent_with_no_licence_terms_at_all_is_refused():
    root = make_root()
    parent = make_parent(root, license_note="", with_license_file=False)
    msg = refused(lambda: build(parent, root), "a parent recording no terms")
    check("licen" in msg.lower(), f"the refusal does not name the fault: {msg}")


def test_a_hostile_label_file_of_nulls_is_refused_not_crashed():
    root = make_root()
    parent = make_parent(root)
    with open(os.path.join(parent, "labels", f"{FIXTURE_CLASSES[0]}_1.txt"),
              "wb") as fh:
        fh.write(b"\x00\xff\xfe" * 100)
    msg = refused(lambda: build(parent, root), "a label file of raw bytes")
    check(msg, "the refusal carried no message")


def test_the_builder_command_line_refuses_and_says_so():
    root = make_root()
    parent = make_parent(root)
    code = bts.main(["--parent", parent, "--dest-root", root,
                     "--name", "cli_set", "--classes", "Alpha-one",
                     "Nope-nine", "--per-class", str(FIXTURE_PER_CLASS),
                     "--train-per-class", "5", "--valid-per-class", "2",
                     "--test-per-class", "2", "--quiet"])
    check(code == 1, f"the command line answered {code} on a refusal")
    check(not os.path.exists(os.path.join(root, "cli_set")),
          "a refused command line left a folder behind")


def test_the_builder_command_line_builds_and_answers_zero():
    root = make_root()
    parent = make_parent(root)
    code = bts.main(["--parent", parent, "--dest-root", root,
                     "--name", "cli_ok", "--classes", "Alpha-one", "Beta-two",
                     "--per-class", str(FIXTURE_PER_CLASS),
                     "--train-per-class", "5", "--valid-per-class", "2",
                     "--test-per-class", "2", "--quiet"])
    check(code == 0, f"the command line answered {code} on a good build")
    check(os.path.isfile(os.path.join(root, "cli_ok", "split_pins.json")),
          "the command line wrote no pins file")


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print("THE BUILT SET:")
        run(test_the_built_set_is_there_with_every_staged_file)
        run(test_image_and_label_counts_match_the_manifest)
        run(test_every_picture_has_a_label_and_every_label_a_picture)
        run(test_no_label_file_is_empty)
        run(test_the_class_lists_agree_with_one_another)
        run(test_every_box_names_a_class_the_dataset_declares)
        run(test_instances_per_class_match_the_class_map_and_the_manifest)
        run(test_split_pins_cover_every_picture_and_nothing_else)
        run(test_the_split_builder_reads_the_pins_this_file_writes)
        run(test_the_split_sizes_are_what_was_asked_for)
        run(test_every_class_appears_in_all_three_splits)
        run(test_the_manifest_records_what_the_page_and_the_builder_read)
        run(test_the_boxes_page_lists_the_built_set_with_its_pins)
        run(test_the_readme_follows_the_prose_rules)

        print("THE BUILDER:")
        run(test_builder_writes_a_whole_staged_dataset)
        run(test_builder_pins_every_class_into_every_split)
        run(test_the_same_seed_chooses_the_same_pictures_and_the_same_holdout)
        run(test_a_different_seed_chooses_a_different_holdout)
        run(test_a_picture_carrying_a_dropped_class_is_passed_over)
        run(test_the_verify_step_reads_the_folder_back)
        run(test_verify_refuses_a_folder_whose_label_was_removed)

        print("REFUSALS:")
        run(test_a_class_the_parent_does_not_declare_is_refused)
        run(test_one_class_named_twice_is_refused)
        run(test_a_single_class_is_refused)
        run(test_split_counts_that_do_not_add_up_are_refused)
        run(test_a_zero_or_fractional_or_boolean_count_is_refused)
        run(test_a_class_the_parent_has_too_few_pictures_of_is_refused)
        run(test_a_destination_that_already_exists_is_refused)
        run(test_a_destination_whose_holder_is_not_there_is_refused)
        run(test_a_name_outside_the_plain_name_rule_is_refused)
        run(test_a_parent_that_is_not_a_staged_dataset_is_refused)
        run(test_a_parent_whose_data_yaml_is_broken_is_refused)
        run(test_a_parent_whose_class_lists_disagree_is_refused)
        run(test_an_empty_label_file_leaves_its_picture_out)
        run(test_a_label_line_with_four_fields_is_refused)
        run(test_a_polygon_label_is_refused)
        run(test_a_label_naming_an_undeclared_class_id_is_refused)
        run(test_a_label_with_no_picture_is_refused)
        run(test_a_parent_with_no_licence_terms_at_all_is_refused)
        run(test_a_hostile_label_file_of_nulls_is_refused_not_crashed)

        print("COMMAND LINE:")
        run(test_the_builder_command_line_refuses_and_says_so)
        run(test_the_builder_command_line_builds_and_answers_zero)
    finally:
        left = []
        for root in _FIXTURE_ROOTS:
            shutil.rmtree(root, ignore_errors=True)
            if os.path.exists(root):
                left.append(root)
        print(f"\ncleaned up {len(_FIXTURE_ROOTS)} fixture folder(s); "
              f"{len(left)} left on disk"
              + (f": {left}" if left else ""))

    failed = [r for r in _RESULTS if not r[1]]
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _ok, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
