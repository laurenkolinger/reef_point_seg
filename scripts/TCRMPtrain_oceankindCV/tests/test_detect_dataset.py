#!/usr/bin/env python
"""Shared detection-dataset builder (src/detect_dataset.py).

No-pytest house harness. Nothing here trains, spawns a service or binds a
port: every path is a temp fixture removed in the finally block.

Covers:
  bbox_pairs                  images paired with labels, unlabelled counted,
                              labels_localized preferred over labels
  build_detect_dataset        seeded 80/10/10, pins replayed on a rebuild,
                              data.yaml with val: valid/images, split_manifest
                              honest about the holdout it recorded
  seg_labels_to_boxes         polygon -> normalized box, clamped to [0,1],
                              degenerate lines dropped, negatives preserved
  dataset_label_kind          a polygon dataset reads as segment, a box one
                              as detect
  build_detect_from_seg_dataset  dataset_detect/ beside dataset/ with
                              symlinked images and converted labels
  prepare                     picks the data.yaml a detect run trains on
  CLI                         `prepare` prints that path on stdout
  split record                split_manifest.json records the seed, the
                              fractions and a split fingerprint; pins are
                              replayed when fractions or seed change;
                              read_split_record reads them back or reports
                              an older manifest as not recorded; two seeds
                              over the same images give two splits (the
                              measured defect); adversarial cases for one
                              image, no labels, corrupt JSON, missing keys,
                              unicode basenames, 60000 pins, a read-only
                              destination, and a bare-interpreter import
  read_source_pins            a staged dataset's split_pins.json is replayed
                              before any shuffle, pins_source records which
                              of the three sources of pins was used, two
                              datasets built from one staged file share a
                              fingerprint; a malformed file is refused, a
                              pin for an absent image is refused (so a
                              held-out photo is never re-drawn), 60000
                              pins stay fast
  read_names_csv              a staged names.csv puts names_full beside
                              names in data.yaml (ultralytics' own loader
                              reads the file unchanged), a token with no row
                              or a blank name cell is left out, a byte-order
                              mark is read through, a wrong header and one
                              token named twice two ways are refused
  holdout_mode                the recorded policy follows the assignment:
                              "staged" when the staged split_pins.json
                              supplied the pins, "lab-random" for a seeded
                              draw, and whatever an existing manifest already
                              recorded on a rebuild that replays its pins; a
                              build that replays no pin records its own draw,
                              including over the segmentation splitter's
                              manifest in the same folder and over a manifest
                              left by a source since switched away; a value
                              that is not a string is refused; every
                              fingerprint is unchanged by the decision; a
                              staged holdout gates as pinned in evaluate_run
  no-training-image refusal   names the staged pins, the dataset manifest or
                              the fractions, whichever actually held every
                              image out

Run with:
  env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_detect_dataset.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import detect_dataset as DD  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def build_bbox_source(root):
    """A staged detection dataset: images/ + labels/ + classes.txt + data.yaml,
    20 labelled images and one image nobody labelled."""
    ds = os.path.join(root, "bbox_datasets", "reefset")
    for i in range(20):
        write(os.path.join(ds, "images", f"img_{i:03d}.jpg"), "x")
        write(os.path.join(ds, "labels", f"img_{i:03d}.txt"),
              f"{i % 2} 0.5 0.5 0.2 0.3\n")
    write(os.path.join(ds, "images", "img_999.jpg"), "x")
    write(os.path.join(ds, "classes.txt"), "AAGA\nACER\n")
    write(os.path.join(ds, "data.yaml"), "names:\n  0: AAGA\n  1: ACER\n")
    # A source with images and no labels at all: refused, never trained on.
    os.makedirs(os.path.join(root, "bbox_datasets", "partial", "images"), exist_ok=True)
    return ds


def build_seg_dataset(root):
    """What pinned_split.py writes: train/valid/test with images/ + labels/
    holding YOLO-seg polygon lines, plus data.yaml, test.yaml and a
    split_manifest.json recording a frozen by-transect holdout."""
    ds = os.path.join(root, "step6", "dataset")
    polys = {
        "train": "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n",
        "valid": "1 0.2 0.2 0.4 0.2 0.4 0.4\n",
        "test": "0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n",
    }
    for split in ("train", "valid", "test"):
        for i in range(2):
            stem = f"TCRMP20260101_clip_SITE_T{5 if split == 'valid' else 6}0{i}"
            write(os.path.join(ds, split, "images", stem + ".jpeg"), "img")
            write(os.path.join(ds, split, "labels", stem + ".txt"), polys[split])
    # One negative frame: an empty label file must survive the conversion.
    write(os.path.join(ds, "train", "images", "TCRMP20260101_clip_SITE_T199.jpeg"), "img")
    write(os.path.join(ds, "train", "labels", "TCRMP20260101_clip_SITE_T199.txt"), "")
    write(os.path.join(ds, "data.yaml"),
          "path: %s\ntrain: train/images\nval: valid/images\ntest: test/images\n"
          "names:\n  0: PA\n  1: MC\n" % os.path.abspath(ds))
    write(os.path.join(ds, "test.yaml"),
          "path: %s\ntrain: train/images\nval: test/images\ntest: test/images\n"
          "names:\n  0: PA\n  1: MC\n" % os.path.abspath(ds))
    write(os.path.join(ds, "split_manifest.json"),
          json.dumps({"holdout_mode": "transect", "pins": {}}))
    return ds


# ---- Split record: seed, fractions, fingerprint -------------------------
#
# Named cases for the 2026-09-04 change. Each takes the shared temp ``root``,
# builds its own fixture under it and reports through check().

def _synthetic_names(n, prefix="img"):
    """``n`` distinct image basenames."""
    return [f"{prefix}_{i:05d}.jpg" for i in range(n)]


def build_small_bbox_source(root, name, n_images, with_labels=True):
    """A staged detection source called ``name`` holding ``n_images`` images,
    each with a label file unless ``with_labels`` is False (then images/
    exists and no label folder does)."""
    ds = os.path.join(root, "sources", name)
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    for i in range(n_images):
        write(os.path.join(ds, "images", f"{name}_{i:03d}.jpg"), "x")
        if with_labels:
            write(os.path.join(ds, "labels", f"{name}_{i:03d}.txt"), "0 0.5 0.5 0.2 0.3\n")
    write(os.path.join(ds, "classes.txt"), "AAGA\n")
    return ds


def load_manifest(dataset_dir):
    """The split_manifest.json a build wrote under ``dataset_dir``."""
    with open(os.path.join(dataset_dir, "split_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _leftover_build_dirs(dataset_dir):
    """Temp build folders left beside ``dataset_dir`` by an aborted build."""
    parent = os.path.dirname(dataset_dir)
    if not os.path.isdir(parent):
        return []
    return [d for d in os.listdir(parent) if d.startswith(".dataset_build_")]


def test_split_manifest_records_seed_fractions_and_fingerprint(root):
    ds = build_small_bbox_source(root, "rec", 30)
    out = os.path.join(root, "rec_out", "dataset")
    summary = DD.build_detect_dataset(ds, out, "rec-seed", valid_frac=0.2, test_frac=0.1)
    m = load_manifest(out)
    check(m.get("seed") == "rec-seed",
          f"the manifest records the seed string used (got {m.get('seed')!r})")
    check(m.get("valid_frac") == 0.2 and m.get("test_frac") == 0.1,
          f"the manifest records the fractions applied (got {m.get('valid_frac')}, {m.get('test_frac')})")
    check(m["counts"] == {"train": 21, "valid": 6, "test": 3},
          f"and those fractions are the ones applied: 30 images split 21/6/3 (got {m['counts']})")
    check(m.get("split_fingerprint") == DD.split_fingerprint(m["pins"]),
          "the manifest fingerprint is the fingerprint of its own pins")
    check(m.get("split_fingerprint_scheme") == DD.SPLIT_FINGERPRINT_SCHEME,
          f"the manifest names the fingerprint scheme (got {m.get('split_fingerprint_scheme')!r})")
    check(m.get("schema_version") == DD.SPLIT_MANIFEST_SCHEMA_VERSION
          and DD.SPLIT_MANIFEST_SCHEMA_VERSION >= 2,
          f"the manifest schema version moved past 1 (got {m.get('schema_version')})")
    check(m.get("n_fresh") == 30 and m.get("n_replayed") == 0,
          f"a first build reports every image as freshly assigned "
          f"(got fresh={m.get('n_fresh')}, replayed={m.get('n_replayed')})")
    check(summary.get("split_fingerprint") == m["split_fingerprint"] and summary.get("valid_frac") == 0.2,
          "the returned summary is the manifest that was written")
    rec = DD.read_split_record(os.path.join(out, "split_manifest.json"))
    check(rec.recorded and rec.missing == (),
          f"the reader reports the split as recorded (missing={rec.missing})")
    check(rec.seed == "rec-seed" and rec.valid_frac == 0.2 and rec.test_frac == 0.1
          and rec.split_fingerprint == m["split_fingerprint"] and rec.n_pins == 30,
          f"and hands back seed, fractions, fingerprint and pin count (got {rec})")
    check(rec.holdout_mode == "lab-random" and rec.schema_version == DD.SPLIT_MANIFEST_SCHEMA_VERSION
          and rec.split_fingerprint_scheme == DD.SPLIT_FINGERPRINT_SCHEME,
          "with the holdout mode, schema version and scheme beside them")
    import pathlib
    rec2 = DD.read_split_record(pathlib.Path(out) / "split_manifest.json")
    check(rec2 == rec, "the reader accepts a pathlib.Path and returns the same record")
    check(rec.same_split_as(rec2), "same_split_as is true for two records of the same fingerprint")


def test_fingerprint_is_stable_across_runs(root):
    names = _synthetic_names(300)
    pins = {n: DD.SPLIT_DIRS[i % 3] for i, n in enumerate(names)}
    fp1 = DD.split_fingerprint(pins)
    fp2 = DD.split_fingerprint(pins)
    check(fp1 == fp2, "the same assignment fingerprints the same twice in one process")
    reordered = dict(reversed(list(pins.items())))
    check(DD.split_fingerprint(reordered) == fp1,
          "insertion order does not matter: the input is sorted before hashing")
    check(len(fp1) == DD.SPLIT_FINGERPRINT_HEX_CHARS and all(c in "0123456789abcdef" for c in fp1),
          f"a fingerprint is {DD.SPLIT_FINGERPRINT_HEX_CHARS} lowercase hex characters (got {fp1!r})")
    # Known-answer vectors computed by hand from the documented scheme: SHA-256
    # over the netstrings b"5:a.jpg,5:train,5:b.jpg,4:test," and over nothing,
    # first 16 hex characters. If the encoding ever changes, these fail and
    # the scheme string must change with it.
    check(DD.split_fingerprint({"b.jpg": "test", "a.jpg": "train"}) == "15b781a844dcfe7e",
          "the known-answer vector pins the scheme byte for byte")
    check(DD.split_fingerprint({}) == "e3b0c44298fc1c14",
          "an empty assignment has the SHA-256 empty digest")
    # A fresh interpreter must agree, so a fingerprint written on one machine
    # can be compared on another.
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import detect_dataset as DD; "
            "print(DD.split_fingerprint(json.load(open(sys.argv[2], encoding='utf-8'))))")
    pins_path = os.path.join(root, "pins_for_subprocess.json")
    write(pins_path, json.dumps(pins))
    proc = subprocess.run([sys.executable, "-c", code, _SRC, pins_path],
                          capture_output=True, text=True)
    check(proc.returncode == 0 and proc.stdout.strip() == fp1,
          f"a fresh interpreter computes the same fingerprint (got {proc.stdout.strip()!r})")
    ds = build_small_bbox_source(root, "stable", 25)
    out_a = os.path.join(root, "stable_a", "dataset")
    out_b = os.path.join(root, "stable_b", "dataset")
    DD.build_detect_dataset(ds, out_a, "stable")
    DD.build_detect_dataset(ds, out_b, "stable")
    check(load_manifest(out_a)["split_fingerprint"] == load_manifest(out_b)["split_fingerprint"],
          "two builds from the same source and seed write the same fingerprint")


def test_fingerprint_differs_when_an_image_moves_split(root):
    names = _synthetic_names(100)
    pins = {n: "train" for n in names}
    for n in names[:10]:
        pins[n] = "test"
    base = DD.split_fingerprint(pins)
    moved = dict(pins)
    moved[names[50]] = "test"
    check(DD.split_fingerprint(moved) != base,
          "moving one image from train to test changes the fingerprint")
    swapped = dict(pins)
    swapped[names[0]], swapped[names[50]] = "train", "test"
    check(DD.split_fingerprint(swapped) != base,
          "swapping two images keeps every count and still changes the fingerprint")
    grown = dict(pins)
    grown["extra.jpg"] = "train"
    check(DD.split_fingerprint(grown) != base, "adding an image changes the fingerprint")
    renamed = {("x" + n if n == names[3] else n): s for n, s in pins.items()}
    check(DD.split_fingerprint(renamed) != base, "renaming an image changes the fingerprint")
    shrunk = dict(pins)
    del shrunk[names[99]]
    check(DD.split_fingerprint(shrunk) != base, "removing an image changes the fingerprint")


def test_existing_pins_are_replayed_when_fractions_change(root):
    ds = build_small_bbox_source(root, "replay", 40)
    out = os.path.join(root, "replay_out", "dataset")
    DD.build_detect_dataset(ds, out, "replay", valid_frac=0.1, test_frac=0.1)
    m1 = load_manifest(out)
    check(m1["counts"] == {"train": 32, "valid": 4, "test": 4},
          f"the first build is 32/4/4 (got {m1['counts']})")
    DD.build_detect_dataset(ds, out, "replay", valid_frac=0.3, test_frac=0.3)
    m2 = load_manifest(out)
    check(m2["pins"] == m1["pins"],
          "rebuilding with different fractions moves no image between splits")
    check(m2["counts"] == m1["counts"], f"and the on-disk counts are unchanged (got {m2['counts']})")
    check(m2["split_fingerprint"] == m1["split_fingerprint"], "so the fingerprint is unchanged")
    check(m2["valid_frac"] == 0.3 and m2["test_frac"] == 0.3,
          "the manifest records the fractions requested on the rebuild")
    check(m2["n_replayed"] == 40 and m2["n_fresh"] == 0,
          f"and says every pin was replayed and none freshly assigned "
          f"(got replayed={m2['n_replayed']}, fresh={m2['n_fresh']})")
    for split in DD.SPLIT_DIRS:
        on_disk = sorted(os.listdir(os.path.join(out, split, "images")))
        pinned = sorted(n for n, s in m2["pins"].items() if s == split)
        check(on_disk == pinned, f"{split}/ on disk holds exactly the pinned images")
    # A different seed on a rebuild moves nothing either: the seed only ever
    # reaches images that have no pin yet.
    DD.build_detect_dataset(ds, out, "a-different-seed")
    m3 = load_manifest(out)
    check(m3["pins"] == m1["pins"] and m3["split_fingerprint"] == m1["split_fingerprint"],
          "rebuilding under a different seed moves no pinned image")
    check(m3["seed"] == "a-different-seed" and m3["n_fresh"] == 0,
          "and the manifest says which seed was offered and that it reached zero images")
    # A grown source keeps every old pin and assigns only the new image.
    write(os.path.join(ds, "images", "replay_new.jpg"), "x")
    write(os.path.join(ds, "labels", "replay_new.txt"), "0 0.5 0.5 0.2 0.3\n")
    DD.build_detect_dataset(ds, out, "replay", valid_frac=0.3, test_frac=0.3)
    m4 = load_manifest(out)
    check(all(m4["pins"][k] == v for k, v in m1["pins"].items()) and "replay_new.jpg" in m4["pins"],
          "a grown source keeps every old pin and assigns only the new image")
    check(m4["n_replayed"] == 40 and m4["n_fresh"] == 1,
          f"counted as 40 replayed and 1 fresh (got {m4['n_replayed']}, {m4['n_fresh']})")
    check(m4["split_fingerprint"] != m1["split_fingerprint"],
          "and the fingerprint changes because the image set grew")


def test_reader_reports_not_recorded_for_an_old_manifest(root):
    old_path = os.path.join(root, "old_manifest", "split_manifest.json")
    old = {
        "schema_version": 1, "holdout_mode": "lab-random", "seed": "coral_intel",
        "created_at": "2026-09-02 16:38:25", "source": "/nowhere/coral_intel",
        "label_dir": "/nowhere/coral_intel/labels_localized",
        "n_pairs": 3, "n_unlabelled": 0,
        "counts": {"train": 1, "valid": 1, "test": 1},
        "pins": {"a.jpg": "train", "b.jpg": "valid", "c.jpg": "test"},
    }
    write(old_path, json.dumps(old, indent=2))
    rec = DD.read_split_record(old_path)
    check(rec.recorded is False, "a manifest written before this change reads as not recorded")
    check(set(rec.missing) == {"valid_frac", "test_frac", "split_fingerprint"},
          f"and names exactly what is missing (got {rec.missing})")
    check(rec.seed == "coral_intel" and rec.valid_frac is None and rec.test_frac is None
          and rec.split_fingerprint is None and rec.split_fingerprint_scheme is None,
          "the seed it did record comes back and the rest is None")
    check(rec.n_pins == 3 and rec.holdout_mode == "lab-random" and rec.schema_version == 1,
          f"pin count, holdout mode and schema version are still reported (got {rec})")
    check(rec.same_split_as(rec) is False,
          "an unrecorded manifest is never the same split as anything, not even itself")
    check(DD.split_fingerprint(old["pins"]) == DD.split_fingerprint(load_manifest(os.path.dirname(old_path))["pins"]),
          "the fingerprint is still computable from an old manifest's pins, so it can be backfilled")
    # pinned_split.py keeps its pins under "pinned": the count is reported and
    # nothing else is claimed.
    seg_path = os.path.join(root, "seg_manifest", "split_manifest.json")
    write(seg_path, json.dumps({"schema_version": 1, "holdout_mode": "transect",
                                "pinned": {"x.jpeg": "train", "y.jpeg": "test"}}))
    seg = DD.read_split_record(seg_path)
    check(not seg.recorded and seg.n_pins == 2 and seg.holdout_mode == "transect"
          and set(seg.missing) == {"seed", "valid_frac", "test_frac", "split_fingerprint"},
          f"a transect manifest from pinned_split reads as not recorded with its pin count (got {seg})")


def test_two_seeds_produce_different_splits(root):
    # The measured defect: the model lab seeded on the lab name and the
    # pipeline on the dataset name, so the same images got two holdouts.
    names = _synthetic_names(1000)
    lab = DD.assign_split(names, "coral_intel_det_260902")
    pipeline = DD.assign_split(names, "coral_intel")
    check(lab != pipeline, "two seed strings over the same images produce two different splits")
    n_test = sum(1 for n in names if pipeline[n] == "test")
    leaked = sum(1 for n in names if pipeline[n] == "test" and lab[n] == "train")
    check(leaked > 0,
          f"{leaked} of the {n_test} test images under one seed sit in the other seed's training set")
    check(DD.split_fingerprint(lab) != DD.split_fingerprint(pipeline),
          "and their fingerprints differ, which is how a later task tells them apart")
    again = DD.assign_split(names, "coral_intel")
    check(again == pipeline and DD.split_fingerprint(again) == DD.split_fingerprint(pipeline),
          "the same seed reproduces the same split and the same fingerprint")
    check(DD.DEFAULT_VALID_FRAC == 0.1 and DD.DEFAULT_TEST_FRAC == 0.1,
          "the defaults both callers rely on are 0.1 and 0.1")
    counts = {s: sum(1 for v in pipeline.values() if v == s) for s in DD.SPLIT_DIRS}
    check(counts == {"train": 800, "valid": 100, "test": 100},
          f"and they split 1000 images 800/100/100 (got {counts})")


def test_adversarial_single_image_source(root):
    ds = build_small_bbox_source(root, "single", 1)
    out = os.path.join(root, "single_out", "dataset")
    summary = DD.build_detect_dataset(ds, out, "single")
    check(summary["counts"] == {"train": 1, "valid": 0, "test": 0},
          f"one image goes to train and the holdouts are empty rather than failing (got {summary['counts']})")
    check(summary["split_fingerprint"] == DD.split_fingerprint({"single_000.jpg": "train"}),
          "and the one-image assignment is fingerprinted")
    check(os.path.isdir(os.path.join(out, "valid", "images"))
          and os.path.isdir(os.path.join(out, "test", "images")),
          "empty split folders still exist so data.yaml resolves")


def test_adversarial_source_with_no_label_files(root):
    no_labels = build_small_bbox_source(root, "nolabels", 5, with_labels=False)
    out = os.path.join(root, "nolabels_out", "dataset")
    try:
        DD.build_detect_dataset(no_labels, out, "nolabels")
        check(False, "a source with images and no label folder is refused")
    except ValueError as e:
        check(no_labels in str(e),
              f"a source with images and no label folder is refused, naming the source ({e})")
    check(not os.path.exists(out), "and nothing is written for it")
    empty_labels = build_small_bbox_source(root, "emptylabels", 5, with_labels=False)
    os.makedirs(os.path.join(empty_labels, "labels"), exist_ok=True)
    try:
        DD.build_detect_dataset(empty_labels, out, "emptylabels")
        check(False, "a source whose label folder is empty is refused")
    except ValueError as e:
        check("pair" in str(e), f"a source whose label folder is empty is refused ({e})")
    try:
        DD.build_detect_dataset(os.path.join(root, "no_such_source"), out, "missing")
        check(False, "a source folder that does not exist is refused")
    except ValueError as e:
        check("no_such_source" in str(e), f"a source folder that does not exist is refused ({e})")
    check(_leftover_build_dirs(out) == [], "no half-built temp folder is left behind")


def test_adversarial_corrupt_manifest_json(root):
    corrupt = os.path.join(root, "corrupt", "split_manifest.json")
    write(corrupt, "{\"pins\": {\"a.jpg\": \"train\",")
    try:
        DD.read_split_record(corrupt)
        check(False, "the reader refuses corrupt JSON")
    except ValueError as e:
        check(corrupt in str(e), f"the reader refuses corrupt JSON and names the file ({e})")
    # A build over a dataset whose manifest is corrupt must not silently
    # re-split: that would move images between train, valid and test.
    ds = build_small_bbox_source(root, "corruptds", 12)
    out = os.path.join(root, "corruptds_out", "dataset")
    DD.build_detect_dataset(ds, out, "corruptds")
    before = {s: sorted(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS}
    write(os.path.join(out, "split_manifest.json"), "not json at all")
    try:
        DD.build_detect_dataset(ds, out, "corruptds")
        check(False, "a corrupt existing manifest stops the rebuild instead of re-splitting")
    except ValueError as e:
        check("split_manifest.json" in str(e) and "remove" in str(e),
              f"a corrupt existing manifest stops the rebuild with advice ({e})")
    after = {s: sorted(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS}
    check(after == before, "and the existing dataset on disk is untouched")
    check(_leftover_build_dirs(out) == [], "and no temp build folder is left behind")
    write(os.path.join(out, "split_manifest.json"), "[1, 2, 3]")
    try:
        DD.build_detect_dataset(ds, out, "corruptds")
        check(False, "a manifest that is a JSON list is refused by the build")
    except ValueError as e:
        check("object" in str(e), f"a manifest that is a JSON list is refused by the build ({e})")
    try:
        DD.read_split_record(os.path.join(out, "split_manifest.json"))
        check(False, "the reader refuses a manifest that is a JSON list")
    except ValueError as e:
        check("object" in str(e), f"the reader refuses a manifest that is a JSON list ({e})")
    write(os.path.join(out, "split_manifest.json"), json.dumps({"pins": ["a.jpg"]}))
    try:
        DD.build_detect_dataset(ds, out, "corruptds")
        check(False, "a manifest whose pins are not an object is refused")
    except ValueError as e:
        check("pins" in str(e), f"a manifest whose pins are not an object is refused ({e})")
    check(after == {s: sorted(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS},
          "the dataset on disk is still untouched after every refusal")


def test_adversarial_manifest_missing_key(root):
    base = os.path.join(root, "missing_key")
    empty = os.path.join(base, "empty", "split_manifest.json")
    write(empty, "{}")
    rec = DD.read_split_record(empty)
    check(not rec.recorded and set(rec.missing) == {"seed", "valid_frac", "test_frac", "split_fingerprint"}
          and rec.n_pins == 0 and rec.holdout_mode is None and rec.schema_version is None,
          f"an empty object reads as not recorded with every field missing (got {rec})")
    partial = os.path.join(base, "partial", "split_manifest.json")
    write(partial, json.dumps({"seed": "s", "valid_frac": 0.1, "split_fingerprint": "abc"}))
    rec = DD.read_split_record(partial)
    check(not rec.recorded and rec.missing == ("test_frac",),
          f"one missing field is reported by name (got {rec.missing})")
    for bad_doc, key in (({"valid_frac": "0.1"}, "valid_frac"),
                         ({"test_frac": True}, "test_frac"),
                         ({"seed": 7}, "seed"),
                         ({"split_fingerprint": 12}, "split_fingerprint"),
                         ({"split_fingerprint_scheme": []}, "split_fingerprint_scheme"),
                         ({"holdout_mode": 3}, "holdout_mode"),
                         ({"pins": "nope"}, "pins"),
                         ({"schema_version": "two"}, "schema_version")):
        p = os.path.join(base, "bad_" + key, "split_manifest.json")
        write(p, json.dumps(bad_doc))
        try:
            DD.read_split_record(p)
            check(False, f"a wrongly typed {key} is refused")
        except ValueError as e:
            check(key in str(e) and p in str(e),
                  f"a wrongly typed {key} is refused, naming key and file ({e})")
    try:
        DD.read_split_record(os.path.join(base, "nowhere", "split_manifest.json"))
        check(False, "a missing manifest raises")
    except FileNotFoundError as e:
        check("split_manifest.json" in str(e),
              f"a missing manifest raises FileNotFoundError naming the path ({e})")
    # A manifest with a null seed reads as missing, the same as an absent key.
    nulls = os.path.join(base, "nulls", "split_manifest.json")
    write(nulls, json.dumps({"seed": None, "valid_frac": None, "pins": None}))
    rec = DD.read_split_record(nulls)
    check(not rec.recorded and "seed" in rec.missing and rec.n_pins == 0,
          f"null values read as missing rather than crashing (got {rec})")


def test_adversarial_unicode_and_space_basenames(root):
    odd = ["Acropora cervicornis (1).jpg", "珊瑚 礁.png", "café_ñ_Ø.jpeg",
           "with\ttab.jpg", "a'quote\"and#hash.jpg", "  leading and trailing  .jpg"]
    ds = os.path.join(root, "sources", "odd")
    for n in odd:
        write(os.path.join(ds, "images", n), "x")
        write(os.path.join(ds, "labels", os.path.splitext(n)[0] + ".txt"), "0 0.5 0.5 0.2 0.3\n")
    write(os.path.join(ds, "classes.txt"), "AAGA\n")
    out = os.path.join(root, "odd_out", "dataset")
    DD.build_detect_dataset(ds, out, "odd seed ñ")
    m = load_manifest(out)
    check(sorted(m["pins"]) == sorted(odd),
          f"every odd basename is pinned as written (got {sorted(m['pins'])})")
    check(m["split_fingerprint"] == DD.split_fingerprint(m["pins"]),
          "the fingerprint survives the JSON round trip of unicode, spaces and quotes")
    placed = sorted(f for s in DD.SPLIT_DIRS for f in os.listdir(os.path.join(out, s, "images")))
    check(placed == sorted(odd), "every odd basename reaches a split folder as written")
    check(DD.read_split_record(os.path.join(out, "split_manifest.json")).seed == "odd seed ñ",
          "a unicode seed is recorded and read back")
    # Netstring framing: a name holding a tab and a newline cannot masquerade
    # as two names. Under a naive tab-and-newline join these two assignments
    # would hash the same.
    tricky = {"x\ttrain\ny": "test"}
    plain = {"x": "train", "y": "test"}
    check(DD.split_fingerprint(tricky) != DD.split_fingerprint(plain),
          "a tab or newline inside a basename cannot collide with two separate basenames")
    check(DD.split_fingerprint({"\u00e9.jpg": "train"}) != DD.split_fingerprint({"e\u0301.jpg": "train"}),
          "composed and decomposed forms of the same letter are different filenames and hash apart")
    undecodable = DD.split_fingerprint({"bad\udcff.jpg": "train"})
    check(len(undecodable) == DD.SPLIT_FINGERPRINT_HEX_CHARS,
          "a filename with undecodable bytes (surrogateescape) still fingerprints instead of raising")
    for bad, kind in ((["a.jpg"], TypeError), ("a.jpg", TypeError), ({1: "train"}, TypeError),
                      ({"a.jpg": "holdout"}, ValueError), ({"a.jpg": None}, ValueError),
                      ({"a.jpg": 0}, ValueError)):
        try:
            DD.split_fingerprint(bad)
            check(False, f"split_fingerprint refuses {bad!r}")
        except Exception as e:
            check(isinstance(e, kind), f"split_fingerprint refuses {bad!r} with {kind.__name__} ({e})")


def test_adversarial_large_basename_list_is_linear(root):
    import time
    small = {n: DD.SPLIT_DIRS[i % 3] for i, n in enumerate(_synthetic_names(5000))}
    big = {n: DD.SPLIT_DIRS[i % 3] for i, n in enumerate(_synthetic_names(60000))}

    def fastest(pins):
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            fp = DD.split_fingerprint(pins)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return fp, best

    fp_small, t_small = fastest(small)
    fp_big, t_big = fastest(big)
    check(len(fp_small) == DD.SPLIT_FINGERPRINT_HEX_CHARS and fp_small != fp_big,
          "5000 and 60000 entries both fingerprint")
    check(t_big < 5.0, f"60000 entries fingerprint in under five seconds (took {t_big:.3f}s)")
    ratio = t_big / max(t_small, 1e-6)
    check(ratio < 60,
          f"twelve times the input costs far less than 144 times the time, so the work is not "
          f"quadratic (ratio {ratio:.1f}, {t_small * 1000:.1f} ms vs {t_big * 1000:.1f} ms)")
    big_path = os.path.join(root, "big_manifest", "split_manifest.json")
    write(big_path, json.dumps({"schema_version": DD.SPLIT_MANIFEST_SCHEMA_VERSION, "seed": "big",
                                "valid_frac": 0.1, "test_frac": 0.1,
                                "split_fingerprint": fp_big,
                                "split_fingerprint_scheme": DD.SPLIT_FINGERPRINT_SCHEME,
                                "pins": big}))
    t0 = time.perf_counter()
    rec = DD.read_split_record(big_path)
    t_read = time.perf_counter() - t0
    check(rec.recorded and rec.n_pins == 60000 and rec.split_fingerprint == fp_big and t_read < 5.0,
          f"the reader handles a 60000-pin manifest (took {t_read:.3f}s, got n_pins={rec.n_pins})")


def test_adversarial_read_only_destination(root):
    if os.geteuid() == 0:
        print("  SKIP read-only destination checks (running as root)")
        return
    ds = build_small_bbox_source(root, "ro", 4)
    parent = os.path.join(root, "ro_parent")
    os.makedirs(parent)
    os.chmod(parent, 0o500)
    out = os.path.join(parent, "dataset")
    try:
        try:
            DD.build_detect_dataset(ds, out, "ro")
            check(False, "a read-only destination is refused")
        except OSError as e:
            check(parent in str(e), f"a read-only destination raises OSError naming the folder ({e})")
        check(not os.path.exists(out), "and nothing is written")
    finally:
        os.chmod(parent, 0o700)
    # A previous dataset whose folders cannot be deleted is moved aside whole,
    # the new dataset takes its place, and the failure to remove the old copy
    # is reported with both paths. Nothing is ever half-deleted.
    out2 = os.path.join(root, "ro_existing", "dataset")
    DD.build_detect_dataset(ds, out2, "ro")
    for split in DD.SPLIT_DIRS:
        os.chmod(os.path.join(out2, split, "images"), 0o500)
    try:
        DD.build_detect_dataset(ds, out2, "ro-again")
        check(False, "a previous dataset that cannot be removed is reported")
    except OSError as e:
        check(out2 in str(e) and ".dataset_old_" in str(e),
              f"a previous dataset that cannot be removed raises OSError naming the new dataset "
              f"and the leftover ({e})")
    check(os.path.isfile(os.path.join(out2, "split_manifest.json"))
          and load_manifest(out2)["seed"] == "ro-again"
          and all(os.access(os.path.join(out2, s, "images"), os.W_OK) for s in DD.SPLIT_DIRS),
          "and the new dataset is whole and in place")
    parent2 = os.path.dirname(out2)
    leftovers = [d for d in os.listdir(parent2) if d.startswith(".dataset_old_")]
    check(len(leftovers) == 1, f"the old copy is left aside under one .dataset_old_ folder (got {leftovers})")
    for d in leftovers:
        for dirpath, _dirs, _files in os.walk(os.path.join(parent2, d)):
            os.chmod(dirpath, 0o700)
        shutil.rmtree(os.path.join(parent2, d))
    check(_leftover_build_dirs(out2) == [], "and no staged build folder is left behind")
    # With the leftover gone, the next rebuild replays the pins and succeeds.
    m_after = DD.build_detect_dataset(ds, out2, "ro")
    check(m_after["n_replayed"] == 4 and m_after["n_fresh"] == 0,
          f"the next rebuild replays every pin (got replayed={m_after['n_replayed']})")


def test_adversarial_fraction_and_seed_validation(root):
    ds = build_small_bbox_source(root, "frac", 6)
    out = os.path.join(root, "frac_out", "dataset")
    for kwargs, why in (({"valid_frac": 0.6, "test_frac": 0.6}, "fractions that leave no training image"),
                        ({"valid_frac": -0.1}, "a negative fraction"),
                        ({"valid_frac": "0.1"}, "a string fraction"),
                        ({"test_frac": True}, "a boolean fraction"),
                        ({"test_frac": float("nan")}, "a NaN fraction")):
        try:
            DD.build_detect_dataset(ds, out, "frac", **kwargs)
            check(False, f"{why} is refused")
        except ValueError as e:
            check("frac" in str(e), f"{why} is refused ({e})")
    for seed in ("", "   ", None, 12):
        try:
            DD.build_detect_dataset(ds, out, seed)
            check(False, f"seed {seed!r} is refused")
        except ValueError as e:
            check("seed" in str(e), f"seed {seed!r} is refused ({e})")
    check(not os.path.exists(out), "a refused build writes nothing")
    try:
        DD.assign_split(["a.jpg"], "s", valid_frac=0.5, test_frac=0.5)
        check(False, "assign_split refuses fractions that leave no training image")
    except ValueError as e:
        check("frac" in str(e), f"assign_split refuses fractions that leave no training image ({e})")


def test_reader_imports_without_flask_or_ultralytics(root):
    heavy = ("flask", "ultralytics", "torch", "yaml", "cv2", "numpy")
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import detect_dataset; "
            "detect_dataset.read_split_record; detect_dataset.split_fingerprint; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in %r); "
            "print(','.join(bad) or 'clean')" % (heavy,))
    proc = subprocess.run([sys.executable, "-c", code, _SRC], capture_output=True, text=True)
    check(proc.returncode == 0 and proc.stdout.strip() == "clean",
          f"importing detect_dataset pulls in none of {heavy} "
          f"(got {proc.stdout.strip()!r} {proc.stderr.strip()[-200:]!r})")


def test_adversarial_rebuild_with_many_pins_is_linear(root):
    import time

    def timed(n):
        names = _synthetic_names(n)
        existing = {nm: DD.SPLIT_DIRS[i % 3] for i, nm in enumerate(names)}
        best = None
        for _ in range(3):
            t0 = time.perf_counter()
            pins = DD.assign_split(names, "seed", existing)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        return pins, best

    _small, t_small = timed(2000)
    big, t_big = timed(20000)
    check(len(big) == 20000
          and all(big[n] == DD.SPLIT_DIRS[i % 3] for i, n in enumerate(_synthetic_names(20000))),
          "every one of 20000 existing pins is replayed unchanged")
    check(t_big < 2.0, f"replaying 20000 pins takes under two seconds (took {t_big:.3f}s)")
    ratio = t_big / max(t_small, 1e-6)
    check(ratio < 50,
          f"ten times the pins costs far less than 100 times the time, so the replay is not "
          f"quadratic (ratio {ratio:.1f}, {t_small * 1000:.1f} ms vs {t_big * 1000:.1f} ms)")
    pins = DD.assign_split((n for n in ("a.jpg", "b.jpg", "c.jpg")), "seed", {"a.jpg": "test"})
    check(sorted(pins) == ["a.jpg", "b.jpg", "c.jpg"] and pins["a.jpg"] == "test",
          f"a generator of names is read once and every name is assigned (got {pins})")


def test_adversarial_bad_pin_value_in_manifest(root):
    ds = build_small_bbox_source(root, "badpin", 6)
    out = os.path.join(root, "badpin_out", "dataset")
    DD.build_detect_dataset(ds, out, "badpin")
    good = load_manifest(out)
    before = {s: sorted(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS}
    for bad_value in ("Train", ["train"], None, 1):
        doc = dict(good)
        doc["pins"] = dict(good["pins"])
        doc["pins"]["badpin_000.jpg"] = bad_value
        write(os.path.join(out, "split_manifest.json"), json.dumps(doc))
        try:
            DD.build_detect_dataset(ds, out, "badpin")
            check(False, f"a pin whose split is {bad_value!r} stops the rebuild")
        except ValueError as e:
            check("badpin_000.jpg" in str(e) and "split_manifest.json" in str(e),
                  f"a pin whose split is {bad_value!r} stops the rebuild, naming the image and "
                  f"the file ({e})")
    check(before == {s: sorted(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS},
          "and the dataset on disk is untouched")
    check(_leftover_build_dirs(out) == [], "and no temp build folder is left behind")


def test_adversarial_symlinked_dataset_dir(root):
    ds = build_small_bbox_source(root, "sym", 3)
    real = os.path.join(root, "sym_real", "dataset")
    DD.build_detect_dataset(ds, real, "sym")
    link = os.path.join(root, "sym_link", "dataset")
    os.makedirs(os.path.dirname(link))
    os.symlink(real, link)
    try:
        DD.build_detect_dataset(ds, link, "sym")
        check(False, "a symbolic link as dataset_dir is refused")
    except ValueError as e:
        check(link in str(e) and "link" in str(e),
              f"a symbolic link as dataset_dir is refused, naming it ({e})")
    check(os.path.islink(link) and os.path.isfile(os.path.join(real, "split_manifest.json")),
          "the link and its target are untouched")
    try:
        DD.build_detect_from_seg_dataset(real, link)
        check(False, "a symbolic link as detect_dir is refused")
    except ValueError as e:
        check("link" in str(e), f"a symbolic link as detect_dir is refused ({e})")


def test_adversarial_zero_train_split(root):
    ds = build_small_bbox_source(root, "tiny", 2)
    out = os.path.join(root, "tiny_out", "dataset")
    try:
        DD.build_detect_dataset(ds, out, "tiny", valid_frac=0.45, test_frac=0.45)
        check(False, "a split that leaves no training image is refused")
    except ValueError as e:
        check("training" in str(e), f"a split that leaves no training image is refused ({e})")
    check(not os.path.exists(out) and _leftover_build_dirs(out) == [], "and nothing is written")
    summary = DD.build_detect_dataset(ds, out, "tiny", valid_frac=0.0, test_frac=0.0)
    check(summary["counts"] == {"train": 2, "valid": 0, "test": 0},
          f"zero fractions put everything in train (got {summary['counts']})")


def test_adversarial_folder_named_like_an_image(root):
    ds = build_small_bbox_source(root, "trap", 3)
    os.makedirs(os.path.join(ds, "images", "trap.jpg"))
    write(os.path.join(ds, "labels", "trap.txt"), "0 0.5 0.5 0.2 0.3\n")
    pairs, _unlabelled = DD.bbox_pairs(ds)
    check(len(pairs) == 3 and all(p[0] != "trap.jpg" for p in pairs),
          f"a folder named like an image is not a pair (got {[p[0] for p in pairs]})")
    out = os.path.join(root, "trap_out", "dataset")
    summary = DD.build_detect_dataset(ds, out, "trap")
    check(summary["n_pairs"] == 3 and "trap.jpg" not in summary["pins"],
          "and the build skips it instead of failing on the copy")


def test_adversarial_corrupt_seg_manifest_blocks_detect_view(root):
    seg = build_seg_dataset(os.path.join(root, "segcorrupt"))
    write(os.path.join(seg, "split_manifest.json"), "{not json")
    det = os.path.join(os.path.dirname(seg), "dataset_detect")
    try:
        DD.build_detect_from_seg_dataset(seg, det)
        check(False, "a corrupt seg split manifest stops the detection view build")
    except ValueError as e:
        check("split_manifest.json" in str(e),
              f"a corrupt seg split manifest stops the detection view build ({e})")
    check(not os.path.exists(det), "and no dataset_detect is written")
    check(not [d for d in os.listdir(os.path.dirname(seg)) if d.startswith(".dataset_")],
          "and no temp build folder is left behind")


def test_adversarial_failed_swap_puts_the_old_dataset_back(root):
    ds = build_small_bbox_source(root, "swap", 3)
    out = os.path.join(root, "swap_out", "dataset")
    parent = os.path.dirname(out)
    DD.build_detect_dataset(ds, out, "swap")
    before = load_manifest(out)
    real_rename = os.rename

    def failing_rename(src, dst, *a, **k):
        if os.path.abspath(dst) == os.path.abspath(out) and ".dataset_build_" in src:
            raise PermissionError(13, "Operation not permitted")
        return real_rename(src, dst, *a, **k)

    os.rename = failing_rename
    try:
        DD.build_detect_dataset(ds, out, "swap-2")
        check(False, "a failed move into place is reported")
    except OSError as e:
        check(out in str(e) and "put back" in str(e),
              f"a failed move into place is reported and the old dataset put back ({e})")
    finally:
        os.rename = real_rename
    check(load_manifest(out) == before, "the previous dataset is back in place, byte for byte")
    check(_leftover_build_dirs(out) == []
          and not [d for d in os.listdir(parent) if d.startswith(".dataset_old_")],
          "and no temp folders remain")

    def failing_twice(src, dst, *a, **k):
        if os.path.abspath(dst) == os.path.abspath(out):
            raise PermissionError(13, "Operation not permitted")
        return real_rename(src, dst, *a, **k)

    os.rename = failing_twice
    try:
        DD.build_detect_dataset(ds, out, "swap-3")
        check(False, "a failed rollback is reported")
    except OSError as e:
        check(".dataset_old_" in str(e) and "by hand" in str(e),
              f"a failed rollback names where the previous dataset sits ({e})")
    finally:
        os.rename = real_rename
    leftovers = [d for d in os.listdir(parent) if d.startswith(".dataset_old_")]
    kept = os.path.join(parent, leftovers[0], "dataset") if leftovers else ""
    check(len(leftovers) == 1 and not os.path.exists(out)
          and json.load(open(os.path.join(kept, "split_manifest.json"), encoding="utf-8")) == before,
          "the previous dataset is whole under the .dataset_old_ folder")
    check(_leftover_build_dirs(out) == [], "and the staged build folder is cleaned up")
    real_rename(kept, out)
    os.rmdir(os.path.join(parent, leftovers[0]))
    check(load_manifest(out) == before, "and it can be moved back by hand")


def test_adversarial_rollback_leaves_only_an_empty_folder_when_rmdir_fails(root):
    ds = build_small_bbox_source(root, "rmdir", 3)
    out = os.path.join(root, "rmdir_out", "dataset")
    parent = os.path.dirname(out)
    DD.build_detect_dataset(ds, out, "rmdir")
    before = load_manifest(out)
    real_rename, real_rmdir = os.rename, os.rmdir

    def failing_rename(src, dst, *a, **k):
        if os.path.abspath(dst) == os.path.abspath(out) and ".dataset_build_" in src:
            raise PermissionError(13, "Operation not permitted")
        return real_rename(src, dst, *a, **k)

    def failing_rmdir(path, *a, **k):
        if ".dataset_old_" in os.fspath(path):
            raise PermissionError(13, "Operation not permitted")
        return real_rmdir(path, *a, **k)

    os.rename, os.rmdir = failing_rename, failing_rmdir
    try:
        DD.build_detect_dataset(ds, out, "rmdir-2")
        check(False, "a failed move into place followed by a failed rmdir is reported")
    except OSError as e:
        named = [d for d in os.listdir(parent) if d.startswith(".dataset_old_")]
        check("put back" in str(e) and "left behind" in str(e)
              and any(os.path.join(parent, d) in str(e) for d in named),
              f"the message says the previous dataset was put back and names the empty folder "
              f"left behind ({e})")
    finally:
        os.rename, os.rmdir = real_rename, real_rmdir
    check(load_manifest(out) == before, "the previous dataset is back in place, byte for byte")
    leftovers = [d for d in os.listdir(parent) if d.startswith(".dataset_old_")]
    check(len(leftovers) == 1 and os.listdir(os.path.join(parent, leftovers[0])) == [],
          f"exactly one empty .dataset_old_ folder is left behind (got {leftovers})")
    check(_leftover_build_dirs(out) == [], "and the staged build folder is cleaned up")
    for d in leftovers:
        real_rmdir(os.path.join(parent, d))


# ---- Staged split pins and the staged names table -----------------------
#
# A staged detection dataset may carry two optional files beside its
# images/ and labels/: split_pins.json, the split its original experiment
# used, and names.csv, the full species name for every class token.
# build_detect_dataset replays the staged pins before any seeded shuffle,
# so an imported model and a VEGA retrain are scored on the same held-out
# images and their fingerprints match, and records in pins_source which of
# the three sources of pins it actually used.

def _write_source_pins(source_dir, pins, source="the original experiment"):
    """Write a staged dataset's split_pins.json holding ``pins``."""
    write(os.path.join(source_dir, DD.SOURCE_PINS_NAME),
          json.dumps({"schema_version": 1, "source": source, "pins": pins}))


def test_source_pins_replayed_and_recorded(root):
    ds = build_small_bbox_source(root, "pinned", 8)
    pins = {f"pinned_{i:03d}.jpg": ("test" if i < 2 else "valid" if i < 4 else "train")
            for i in range(8)}
    _write_source_pins(ds, pins)
    out = os.path.join(root, "pinned_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "seedA", valid_frac=0.2, test_frac=0.1)
    check(m.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"the manifest records the staged file as the source of the pins "
          f"(got {m.get('pins_source')!r})")
    check(m["counts"] == {"train": 4, "valid": 2, "test": 2},
          f"the staged pins decide every image and the fractions reach none (got {m['counts']})")
    check(m["n_replayed"] == 8 and m["n_fresh"] == 0,
          f"every image is replayed and none is freshly assigned "
          f"(got replayed={m['n_replayed']}, fresh={m['n_fresh']})")
    check(m["pins"] == pins, "and the assignment written is exactly the staged one")
    for split in DD.SPLIT_DIRS:
        on_disk = sorted(os.listdir(os.path.join(out, split, "images")))
        check(on_disk == sorted(n for n, s in pins.items() if s == split),
              f"{split}/ on disk holds exactly the images the staged file pinned there")
    m2 = DD.build_detect_dataset(ds, out, "seedB", valid_frac=0.5, test_frac=0.4)
    check(m2["split_fingerprint"] == m["split_fingerprint"],
          "a rebuild under another seed and other fractions keeps the fingerprint")
    check(m2.get("pins_source") == DD.PINS_SOURCE_DATASET,
          f"and takes its pins from the dataset manifest the first build wrote "
          f"(got {m2.get('pins_source')!r})")
    out2 = os.path.join(root, "pinned_other_project", "dataset")
    m3 = DD.build_detect_dataset(ds, out2, "seedC")
    check(m3["split_fingerprint"] == m["split_fingerprint"],
          "two datasets built from the same staged pins carry one fingerprint, which is what "
          "makes an imported model and a retrain comparable")
    check(m3.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"and the second dataset reads the staged file too (got {m3.get('pins_source')!r})")
    check(DD.read_source_pins(ds) == pins, "read_source_pins hands back the staged table itself")
    check(DD.read_source_pins(os.path.join(root, "sources", "no_such_source_at_all")) == {},
          "and a source with no split_pins.json reads as no pins at all")


def test_source_pins_refused_when_malformed(root):
    ds = build_small_bbox_source(root, "badpins", 3)
    cases = (
        ("[1, 2]", "a JSON list where an object belongs", "JSON object"),
        ('{"pins": [1]}', "pins that are not an object", "object of basename to split"),
        ('{"pins": {"badpins_000.jpg": "validation"}}', "a split spelled validation",
         "validation"),
        ('{"schema_version": 1}', "a file carrying no pins key at all", "carries no pins"),
        ("not json at all", "a file that is not JSON", "cannot be read as JSON"),
    )
    for i, (bad, why, must_say) in enumerate(cases):
        write(os.path.join(ds, DD.SOURCE_PINS_NAME), bad)
        out = os.path.join(root, f"badpins_out_{i}", "dataset")
        try:
            DD.build_detect_dataset(ds, out, "badpins")
            check(False, f"{why} must stop the build instead of being ignored")
        except ValueError as e:
            check(DD.SOURCE_PINS_NAME in str(e) and must_say in str(e),
                  f"{why} stops the build and the refusal names {DD.SOURCE_PINS_NAME} and "
                  f"says {must_say!r} ({e})")
            check("NoneType" not in str(e),
                  f"and says it in plain words, not a Python type name, for {why} ({e})")
        check(not os.path.exists(out), f"and no dataset is written for {why}")
    os.remove(os.path.join(ds, DD.SOURCE_PINS_NAME))
    ok = DD.build_detect_dataset(ds, os.path.join(root, "badpins_ok", "dataset"), "badpins")
    check(ok.get("pins_source") == DD.PINS_SOURCE_FRESH,
          f"removing the bad file lets the same source build a fresh split "
          f"(got {ok.get('pins_source')!r})")


def test_fresh_split_recorded_when_no_pins(root):
    ds = build_small_bbox_source(root, "nopins", 5)
    out = os.path.join(root, "nopins_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "nopins")
    check(m.get("pins_source") == DD.PINS_SOURCE_FRESH,
          f"a source with no staged pins and no earlier manifest reads as fresh "
          f"(got {m.get('pins_source')!r})")
    check(m["n_fresh"] == 5 and m["n_replayed"] == 0,
          f"and every image is assigned by the seeded shuffle "
          f"(got fresh={m['n_fresh']}, replayed={m['n_replayed']})")
    check(load_manifest(out).get("pins_source") == DD.PINS_SOURCE_FRESH,
          "the manifest on disk carries pins_source as well as the returned summary")


def test_names_full_in_data_yaml(root):
    import yaml
    ds = build_small_bbox_source(root, "named", 10)
    write(os.path.join(ds, "classes.txt"), "KYSE_Normal\nEPST_Dark\n")
    write(os.path.join(ds, DD.NAMES_CSV_NAME),
          "token,name\n"
          'KYSE_Normal,"Kyphosus sectatrix, Bermuda chub, normal phase"\n'
          'EPST_Dark,"Epinephelus striatus, Nassau grouper, dark phase"\n')
    out = os.path.join(root, "named_out", "dataset")
    DD.build_detect_dataset(ds, out, "named")
    with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    check(set(doc.get("names_full") or {}) == set(doc["names"]),
          f"names_full covers every class id names carries (got {doc.get('names_full')})")
    check(doc["names_full"][0] == "Kyphosus sectatrix, Bermuda chub, normal phase",
          f"the full name is copied verbatim, comma and phase and all "
          f"(got {doc['names_full'].get(0)!r})")
    check(doc["names"][0] == "KYSE_Normal" and doc["names"][1] == "EPST_Dark",
          f"and names still carries the tokens the label files use (got {doc['names']})")
    check(DD.read_names_csv(ds)["EPST_Dark"] == "Epinephelus striatus, Nassau grouper, dark phase",
          "read_names_csv reads the same table on its own")
    # The extra key must be inert for training: ultralytics' own loader reads
    # the file and resolves the class list unchanged.
    from ultralytics.data.utils import check_det_dataset
    resolved = check_det_dataset(os.path.join(out, "data.yaml"))
    check(resolved["nc"] == 2 and resolved["names"][1] == "EPST_Dark",
          f"ultralytics accepts a data.yaml carrying names_full and resolves two classes "
          f"(got nc={resolved.get('nc')}, names={resolved.get('names')})")
    # A token with no row in names.csv is left out rather than guessed.
    partial = build_small_bbox_source(root, "partialnames", 6)
    write(os.path.join(partial, "classes.txt"), "KYSE_Normal\nEPST_Dark\n")
    write(os.path.join(partial, DD.NAMES_CSV_NAME),
          "token,name\nEPST_Dark,Epinephelus striatus\n")
    out2 = os.path.join(root, "partialnames_out", "dataset")
    DD.build_detect_dataset(partial, out2, "partialnames")
    with open(os.path.join(out2, "data.yaml"), encoding="utf-8") as fh:
        doc2 = yaml.safe_load(fh)
    check(doc2.get("names_full") == {1: "Epinephelus striatus"},
          f"a token with no row in names.csv is left out of names_full, never guessed "
          f"(got {doc2.get('names_full')})")
    # A row whose name cell is blank is left out for the same reason: a class
    # the table names with nothing at all would show a reader an empty species
    # column, which is worse than falling back to the token.
    blank = build_small_bbox_source(root, "blanknames", 6)
    write(os.path.join(blank, "classes.txt"), "AAGA\nACER\n")
    write(os.path.join(blank, DD.NAMES_CSV_NAME),
          "token,name\nAAGA,\nACER,Agaricia cervicornis\n")
    out_blank = os.path.join(root, "blanknames_out", "dataset")
    DD.build_detect_dataset(blank, out_blank, "blanknames")
    with open(os.path.join(out_blank, "data.yaml"), encoding="utf-8") as fh:
        doc_blank = yaml.safe_load(fh)
    check(doc_blank.get("names_full") == {1: "Agaricia cervicornis"},
          f"a token whose name cell is blank is absent from names_full rather than named by "
          f"an empty string (got {doc_blank.get('names_full')})")
    check("AAGA" not in DD.read_names_csv(blank),
          f"and read_names_csv leaves it out of the table too (got {DD.read_names_csv(blank)})")

    # Coral Intel carries no names.csv: its data.yaml must be the one it always was.
    plain = build_small_bbox_source(root, "plainnames", 6)
    out3 = os.path.join(root, "plainnames_out", "dataset")
    DD.build_detect_dataset(plain, out3, "plainnames")
    with open(os.path.join(out3, "data.yaml"), encoding="utf-8") as fh:
        doc3 = yaml.safe_load(fh)
    check("names_full" not in doc3,
          f"a source with no names.csv writes the data.yaml it always wrote (got {sorted(doc3)})")


def test_names_csv_wrong_header_refused(root):
    ds = build_small_bbox_source(root, "badnames", 4)
    write(os.path.join(ds, DD.NAMES_CSV_NAME), "code,label\nAAGA,Agaricia agaricites\n")
    try:
        DD.read_names_csv(ds)
        check(False, "a names.csv whose header is not token,name must be refused")
    except ValueError as e:
        check("token" in str(e) and "name" in str(e) and DD.NAMES_CSV_NAME in str(e),
              f"the refusal names the missing columns and the file ({e})")
    out = os.path.join(root, "badnames_out", "dataset")
    try:
        DD.build_detect_dataset(ds, out, "badnames")
        check(False, "and the build stops rather than writing a data.yaml with no full names")
    except ValueError as e:
        check(DD.NAMES_CSV_NAME in str(e), f"the build's refusal names the file too ({e})")
    check(not os.path.exists(out) and _leftover_build_dirs(out) == [],
          "and leaves neither a dataset nor a temp build folder behind")
    check(DD.read_names_csv(os.path.join(root, "sources", "no_such_source_at_all")) == {},
          "a source with no names.csv reads as no full names at all")
    # Hostile file contents: a table nobody can decode is refused by name,
    # never skipped into an empty table that would show bare tokens.
    hostile = build_small_bbox_source(root, "hostilenames", 3)
    with open(os.path.join(hostile, DD.NAMES_CSV_NAME), "wb") as fh:
        fh.write(b"token,name\nAAGA,Agaricia \xff\xfe agaricites\n")
    try:
        DD.read_names_csv(hostile)
        check(False, "a names.csv that is not valid UTF-8 must be refused")
    except ValueError as e:
        check(DD.NAMES_CSV_NAME in str(e) and "UTF-8" in str(e),
              f"the refusal names the file and says it is not valid UTF-8 ({e})")


def test_names_csv_duplicate_token_refused(root):
    ds = build_small_bbox_source(root, "dupnames", 4)
    write(os.path.join(ds, "classes.txt"), "AAGA\nACER\n")
    write(os.path.join(ds, DD.NAMES_CSV_NAME),
          "token,name\nAAGA,Agaricia agaricites\nACER,Agaricia cervicornis\n"
          "AAGA,Agaricia lamarcki\n")
    try:
        DD.read_names_csv(ds)
        check(False, "one token named twice with two different species must be refused")
    except ValueError as e:
        check("AAGA" in str(e) and DD.NAMES_CSV_NAME in str(e),
              f"the refusal names the repeated token and the file ({e})")
    out = os.path.join(root, "dupnames_out", "dataset")
    try:
        DD.build_detect_dataset(ds, out, "dupnames")
        check(False, "and the build stops rather than picking whichever row is read last")
    except ValueError as e:
        check("AAGA" in str(e), f"the build's refusal names the repeated token too ({e})")
    check(not os.path.exists(out) and _leftover_build_dirs(out) == [],
          "and leaves neither a dataset nor a temp build folder behind")
    # The same row twice names the same species, so nothing is ambiguous.
    write(os.path.join(ds, DD.NAMES_CSV_NAME),
          "token,name\nAAGA,Agaricia agaricites\nAAGA,Agaricia agaricites\n")
    check(DD.read_names_csv(ds) == {"AAGA": "Agaricia agaricites"},
          f"a row repeated verbatim is read rather than refused "
          f"(got {DD.read_names_csv(ds)})")


def test_names_csv_with_a_byte_order_mark_is_read(root):
    import yaml
    ds = build_small_bbox_source(root, "bomnames", 4)
    write(os.path.join(ds, "classes.txt"), "AAGA\nACER\n")
    with open(os.path.join(ds, DD.NAMES_CSV_NAME), "w", encoding="utf-8-sig") as fh:
        fh.write("token,name\nAAGA,Agaricia agaricites\nACER,Agaricia cervicornis\n")
    check(DD.read_names_csv(ds) == {"AAGA": "Agaricia agaricites",
                                    "ACER": "Agaricia cervicornis"},
          f"a names.csv a spreadsheet saved with a byte-order mark reads as itself "
          f"(got {DD.read_names_csv(ds)})")
    out = os.path.join(root, "bomnames_out", "dataset")
    DD.build_detect_dataset(ds, out, "bomnames")
    with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    check(doc.get("names_full") == {0: "Agaricia agaricites", 1: "Agaricia cervicornis"},
          f"and both full names reach the split dataset's data.yaml (got {doc.get('names_full')})")


def test_zero_train_split_names_whatever_held_the_images_out(root):
    # Staged pins that hold every image out: the fractions reached nothing, so
    # the refusal must point at the staged file, never at two numbers that
    # cannot change the outcome.
    staged = build_small_bbox_source(root, "allheld", 4)
    _write_source_pins(staged, {f"allheld_{i:03d}.jpg": ("test" if i < 2 else "valid")
                                for i in range(4)})
    out = os.path.join(root, "allheld_out", "dataset")
    try:
        DD.build_detect_dataset(staged, out, "allheld")
        check(False, "a staged file that holds every image out must stop the build")
    except ValueError as e:
        msg = str(e)
        check(DD.SOURCE_PINS_NAME in msg,
              f"the refusal names the staged split pins that made the decision ({msg})")
        check("lower the fractions" not in msg,
              f"and offers no fraction change, because the fractions reached no image ({msg})")
    check(not os.path.exists(out) and _leftover_build_dirs(out) == [], "and nothing is written")

    # A dataset manifest that holds every image out: the refusal names that
    # manifest instead, because that is the file a rebuild replays.
    again = build_small_bbox_source(root, "heldagain", 4)
    out2 = os.path.join(root, "heldagain_out", "dataset")
    DD.build_detect_dataset(again, out2, "heldagain", valid_frac=0.0, test_frac=0.0)
    manifest = load_manifest(out2)
    manifest["pins"] = {name: "valid" for name in manifest["pins"]}
    write(os.path.join(out2, "split_manifest.json"), json.dumps(manifest))
    try:
        DD.build_detect_dataset(again, out2, "heldagain")
        check(False, "an earlier split that holds every image out must stop the rebuild")
    except ValueError as e:
        msg = str(e)
        check("split_manifest.json" in msg,
              f"the refusal names the manifest whose pins were replayed ({msg})")
        check("lower the fractions" not in msg,
              f"and offers no fraction change there either ({msg})")

    # Staged pins on some images and a seeded draw on the rest, with nothing
    # left for train: both the file and the fractions are named, because both
    # can change the outcome.
    mixed = build_small_bbox_source(root, "mixedheld", 4)
    _write_source_pins(mixed, {f"mixedheld_{i:03d}.jpg": "test" for i in range(2)})
    out_mixed = os.path.join(root, "mixedheld_out", "dataset")
    try:
        DD.build_detect_dataset(mixed, out_mixed, "mixedheld", valid_frac=0.45, test_frac=0.45)
        check(False, "pins plus fractions that together leave no training image must stop")
    except ValueError as e:
        msg = str(e)
        check(DD.SOURCE_PINS_NAME in msg and "lower the fractions" in msg,
              f"the refusal names the staged file and the fractions, both of which reached "
              f"images ({msg})")
        check("2 of them" in msg,
              f"and says how many images the staged file decided ({msg})")

    # With no pin anywhere the fractions really are the remedy.
    fresh = build_small_bbox_source(root, "freshheld", 2)
    out3 = os.path.join(root, "freshheld_out", "dataset")
    try:
        DD.build_detect_dataset(fresh, out3, "freshheld", valid_frac=0.45, test_frac=0.45)
        check(False, "fractions that leave no training image must stop the build")
    except ValueError as e:
        msg = str(e)
        check("lower the fractions" in msg,
              f"a fresh split's refusal still offers the fractions ({msg})")
        check(DD.SOURCE_PINS_NAME not in msg,
              f"and names no staged file, because none of them decided anything ({msg})")


def _refused(fn):
    """The ValueError message fn() raised, or '' when it raised nothing."""
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return ""


def test_adversarial_source_pins_for_absent_images(root):
    ds = build_small_bbox_source(root, "ghostpins", 6)
    pins = {f"ghostpins_{i:03d}.jpg": "test" for i in range(3)}
    pins["a_photo_that_was_never_staged.jpg"] = "valid"
    pins["another_absent_photo.jpg"] = "train"
    _write_source_pins(ds, pins)
    out = os.path.join(root, "ghostpins_out", "dataset")
    msg = _refused(lambda: DD.build_detect_dataset(ds, out, "ghostpins"))
    check("2 of them cannot be trained on" in msg and "2 are not in" in msg
          and "a_photo_that_was_never_staged.jpg" in msg,
          f"a pin for a photo the dataset does not hold stops the build and names it: {msg!r}")
    check(not os.path.exists(out), "and nothing is written")


def test_stem_keyed_pins_are_refused_not_redrawn(root):
    """Regression, 2026-09-26: a pin file keyed by stem matched nothing and
    VEGA drew a random split while recording the pin file as its source."""
    ds = build_small_bbox_source(root, "stems", 6)
    _write_source_pins(ds, {f"stems_{i:03d}": ("test" if i < 2 else "train") for i in range(6)})
    out = os.path.join(root, "stems_out", "dataset")
    msg = _refused(lambda: DD.build_detect_dataset(ds, out, "stems"))
    check("6 of them cannot be trained on" in msg and "extension included" in msg,
          f"stem-keyed pins are refused with the fix spelled out: {msg!r}")


def test_a_project_split_naming_a_vanished_photo_is_refused(root):
    ds = build_small_bbox_source(root, "vanish", 6)
    out = os.path.join(root, "vanish_out", "dataset")
    first = DD.build_detect_dataset(ds, out, "vanish")
    gone = sorted(first["pins"])[0]
    os.remove(os.path.join(ds, "images", gone))
    msg = _refused(lambda: DD.build_detect_dataset(ds, out, "vanish"))
    check(gone in msg and "split_manifest.json" in msg and "start a new project" in msg
          and "Do not edit this file by hand" in msg,
          f"a rebuild whose saved split names a photo no longer staged stops: {msg!r}")
    check(DD.split_fingerprint(json.load(open(os.path.join(out, "split_manifest.json")))["pins"])
          == first["split_fingerprint"], "and the previous split stays on disk untouched")


def test_an_unlabelled_photo_is_named_as_unlabelled(root):
    ds = build_small_bbox_source(root, "nolabel", 6)
    out = os.path.join(root, "nolabel_out", "dataset")
    first = DD.build_detect_dataset(ds, out, "nolabel")
    victim = sorted(first["pins"])[1]
    os.remove(os.path.join(ds, "labels", victim.replace(".jpg", ".txt")))
    msg = _refused(lambda: DD.build_detect_dataset(ds, out, "nolabel"))
    check("1 are in images/ but have no label file" in msg and victim in msg
          and "are not in" not in msg,
          f"a photo that only lost its label is reported as unlabelled, not missing: {msg!r}")


def test_the_same_photos_through_another_path_keep_their_split(root):
    ds = build_small_bbox_source(root, "moved", 40)
    out = os.path.join(root, "moved_out", "dataset")
    link = os.path.join(root, "moved_link")
    os.symlink(ds, link)
    first = DD.build_detect_dataset(link, out, "moved", valid_frac=0.2, test_frac=0.1)
    via_real = DD.build_detect_dataset(ds, out, "moved", valid_frac=0.3, test_frac=0.3)
    check(via_real["pins"] == first["pins"] and via_real["pins_source"] == DD.PINS_SOURCE_DATASET,
          "a split built through a linked path is replayed through the real one, no photo moved")
    check(first["source"] == os.path.realpath(ds), "the split records the resolved folder")
    copy = os.path.join(root, "moved_copy")
    shutil.copytree(ds, copy)
    via_copy = DD.build_detect_dataset(copy, out, "moved_copy", valid_frac=0.3, test_frac=0.3)
    check(via_copy["pins"] == first["pins"],
          "the same photos in a renamed folder keep their split rather than being re-drawn")


def test_many_unlabelled_photos_are_refused_quickly(root):
    import time
    ds = build_small_bbox_source(root, "manyunlab", 3)
    names = [f"u_{i:05d}.jpg" for i in range(20000)]
    for n in names:
        write(os.path.join(ds, "images", n), "x")
    _write_source_pins(ds, {n: "train" for n in names} | {"../escape.jpg": "test"})
    t0 = time.perf_counter()
    msg = _refused(lambda: DD.build_detect_dataset(ds, os.path.join(root, "mu_out", "dataset"), "mu"))
    took = time.perf_counter() - t0
    check(took < 5.0 and "20000 are in images/ but have no label file" in msg,
          f"20000 unlabelled photos are refused in under five seconds (took {took:.2f}s)")
    check("1 are not in" in msg and "../escape.jpg" in msg,
          "an entry carrying a folder part is filed as missing, never as unlabelled")


def test_a_grown_dataset_keeps_every_pinned_photo(root):
    ds = build_small_bbox_source(root, "grow", 6)
    out = os.path.join(root, "grow_out", "dataset")
    first = DD.build_detect_dataset(ds, out, "grow")
    extra = build_small_bbox_source(os.path.join(root, "grow_extra"), "grownew", 2)
    for sub in ("images", "labels"):
        for fn in os.listdir(os.path.join(extra, sub)):
            shutil.copy(os.path.join(extra, sub, fn), os.path.join(ds, sub, fn))
    second = DD.build_detect_dataset(ds, out, "grow")
    check(all(second["pins"][k] == v for k, v in first["pins"].items())
          and second["n_fresh"] == 2,
          "photos the dataset gained are placed and no pinned photo moves")


def test_adversarial_sixty_thousand_source_pins(root):
    import time
    ds = build_small_bbox_source(root, "manypins", 5)
    pins = {f"manypins_{i:03d}.jpg": "valid" for i in range(2)}
    pins.update({n: DD.SPLIT_DIRS[i % 3]
                 for i, n in enumerate(_synthetic_names(60000, "absent"))})
    _write_source_pins(ds, pins)
    t0 = time.perf_counter()
    read = DD.read_source_pins(ds)
    t_read = time.perf_counter() - t0
    check(len(read) == 60002, f"read_source_pins reads all 60002 pins (got {len(read)})")
    check(t_read < 5.0, f"and reads them in under five seconds (took {t_read:.3f}s)")
    out = os.path.join(root, "manypins_out", "dataset")
    t1 = time.perf_counter()
    msg = _refused(lambda: DD.build_detect_dataset(ds, out, "manypins"))
    t_build = time.perf_counter() - t1
    check(t_build < 10.0,
          f"a build against 60002 pins answers in under ten seconds (took {t_build:.3f}s)")
    check("60000 of them cannot be trained on" in msg and msg.count("'") <= 2 * DD.ABSENT_PIN_EXAMPLES,
          f"60000 pins for absent photos are refused, quoting only a few: {msg[:160]!r}")


def _staged_pins_for(name, n_images):
    """A staged split table over ``n_images`` basenames of source ``name``:
    the first two to test, the next two to valid, the rest to train."""
    return {f"{name}_{i:03d}.jpg": ("test" if i < 2 else "valid" if i < 4 else "train")
            for i in range(n_images)}


def test_holdout_mode_staged_when_the_staged_file_supplied_the_pins(root):
    ds = build_small_bbox_source(root, "hmstaged", 12)
    _write_source_pins(ds, _staged_pins_for("hmstaged", 12))
    out = os.path.join(root, "hmstaged_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "hmstaged", valid_frac=0.2, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED,
          f"a build whose pins came from the staged split_pins.json records the holdout as "
          f"staged, because nothing was drawn here (got {m.get('holdout_mode')!r})")
    check(m.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"and pins_source still names the file the pins were read from "
          f"(got {m.get('pins_source')!r})")
    check(load_manifest(out).get("holdout_mode") == DD.HOLDOUT_MODE_STAGED,
          "the manifest on disk carries the same value as the returned summary")
    check(m["n_replayed"] == 12 and m["n_fresh"] == 0,
          f"every image was replayed and none drawn (got replayed={m['n_replayed']}, "
          f"fresh={m['n_fresh']})")
    check(m["split_fingerprint"] == DD.split_fingerprint(m["pins"]),
          "and the fingerprint is still the fingerprint of the assignment written")
    rec = DD.read_split_record(os.path.join(out, "split_manifest.json"))
    check(rec.recorded and rec.holdout_mode == DD.HOLDOUT_MODE_STAGED,
          f"the reader hands the new value back beside the rest of the record (got {rec})")


def test_holdout_mode_lab_random_when_nothing_pinned_the_split(root):
    ds = build_small_bbox_source(root, "hmfresh", 12)
    out = os.path.join(root, "hmfresh_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "hmfresh", valid_frac=0.2, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a source carrying no pins still records the seeded draw as lab-random "
          f"(got {m.get('holdout_mode')!r})")
    check(m.get("pins_source") == DD.PINS_SOURCE_FRESH,
          f"with pins_source fresh beside it (got {m.get('pins_source')!r})")
    check(m["n_fresh"] == 12 and m["n_replayed"] == 0,
          f"every image was drawn by the seeded shuffle (got fresh={m['n_fresh']}, "
          f"replayed={m['n_replayed']})")
    check(load_manifest(out).get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          "and the manifest on disk says the same")


def test_rebuild_keeps_the_staged_holdout_mode(root):
    ds = build_small_bbox_source(root, "hmrestaged", 12)
    _write_source_pins(ds, _staged_pins_for("hmrestaged", 12))
    out = os.path.join(root, "hmrestaged_out", "dataset")
    first = DD.build_detect_dataset(ds, out, "hmrestaged", valid_frac=0.2, test_frac=0.1)
    again = DD.build_detect_dataset(ds, out, "a-different-seed", valid_frac=0.5, test_frac=0.4)
    check(again.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED,
          f"a rebuild over a staged holdout keeps it staged: the images kept the splits that "
          f"manifest pinned, so the policy that chose them is still the earlier one "
          f"(got {again.get('holdout_mode')!r})")
    check(again.get("pins_source") == DD.PINS_SOURCE_DATASET,
          f"while pins_source moves to the dataset manifest this build actually read "
          f"(got {again.get('pins_source')!r})")
    check(again["split_fingerprint"] == first["split_fingerprint"],
          f"and no image moved, so the fingerprint is unchanged "
          f"(got {again['split_fingerprint']}, was {first['split_fingerprint']})")
    check(again["pins"] == first["pins"], "the assignment written is the same table")
    # A build of the same staged source into a folder that has no manifest yet
    # reads the staged file again and records the staged policy again, on the
    # same fingerprint: this is what makes an imported model and a retrain
    # comparable.
    other = os.path.join(root, "hmrestaged_other", "dataset")
    fresh_dest = DD.build_detect_dataset(ds, other, "yet-another-seed")
    check(fresh_dest.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED
          and fresh_dest.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"a first build into a new folder reads the staged file and records staged "
          f"(got {fresh_dest.get('holdout_mode')!r}, {fresh_dest.get('pins_source')!r})")
    check(fresh_dest["split_fingerprint"] == first["split_fingerprint"],
          "on the same fingerprint as the first dataset, which is what makes the two runs "
          "comparable")


def test_rebuild_keeps_the_lab_random_holdout_mode(root):
    ds = build_small_bbox_source(root, "hmrefresh", 12)
    out = os.path.join(root, "hmrefresh_out", "dataset")
    first = DD.build_detect_dataset(ds, out, "hmrefresh", valid_frac=0.2, test_frac=0.1)
    again = DD.build_detect_dataset(ds, out, "a-different-seed", valid_frac=0.5, test_frac=0.4)
    check(again.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a rebuild over a lab-random holdout keeps it lab-random "
          f"(got {again.get('holdout_mode')!r})")
    check(again.get("pins_source") == DD.PINS_SOURCE_DATASET,
          f"with pins_source naming the dataset manifest (got {again.get('pins_source')!r})")
    check(again["split_fingerprint"] == first["split_fingerprint"],
          "and the fingerprint is unchanged")
    # A staged file appearing after the first build changes nothing: the
    # dataset manifest wins, so the recorded policy stays the one that chose
    # these images.
    _write_source_pins(ds, {f"hmrefresh_{i:03d}.jpg": "test" for i in range(12)})
    third = DD.build_detect_dataset(ds, out, "hmrefresh")
    check(third.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM
          and third.get("pins_source") == DD.PINS_SOURCE_DATASET,
          f"a staged file that arrives after the first build relabels nothing and moves nothing "
          f"(got {third.get('holdout_mode')!r}, {third.get('pins_source')!r})")
    check(third["split_fingerprint"] == first["split_fingerprint"],
          "and the fingerprint is still the first build's")


def test_rebuild_over_a_manifest_with_no_holdout_mode(root):
    # The shape on disk before 2026-09-15 from a hand-written or truncated
    # manifest: pins and no holdout_mode at all. The build must not crash, and
    # must not invent a policy it cannot know. It writes lab-random, which is
    # what every detection build wrote before this change, so the value is no
    # stronger a claim than the file already carried.
    ds = build_small_bbox_source(root, "hmnomode", 12)
    out = os.path.join(root, "hmnomode_out", "dataset")
    old_pins = {f"hmnomode_{i:03d}.jpg": ("test" if i < 3 else "train") for i in range(12)}
    write(os.path.join(out, "split_manifest.json"),
          json.dumps({"schema_version": 1, "seed": "hmnomode", "pins": old_pins}))
    m = DD.build_detect_dataset(ds, out, "hmnomode", valid_frac=0.2, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a rebuild over a manifest that records no holdout_mode writes lab-random "
          f"(got {m.get('holdout_mode')!r})")
    check(m.get("pins_source") == DD.PINS_SOURCE_DATASET,
          f"taking its pins from that manifest (got {m.get('pins_source')!r})")
    check(m["pins"] == old_pins and m["n_replayed"] == 12,
          f"every old pin is replayed unchanged (got replayed={m['n_replayed']})")
    check(m["split_fingerprint"] == DD.split_fingerprint(old_pins),
          "and the fingerprint is the fingerprint of those very pins, so nothing moved")
    # A null holdout_mode is the same case as an absent one.
    out2 = os.path.join(root, "hmnullmode_out", "dataset")
    write(os.path.join(out2, "split_manifest.json"),
          json.dumps({"schema_version": 2, "holdout_mode": None, "pins": old_pins}))
    m2 = DD.build_detect_dataset(ds, out2, "hmnomode")
    check(m2.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a manifest whose holdout_mode is null reads the same way (got "
          f"{m2.get('holdout_mode')!r})")


def test_rebuild_refuses_a_holdout_mode_that_is_not_a_string(root):
    ds = build_small_bbox_source(root, "hmbadmode", 6)
    pins = {f"hmbadmode_{i:03d}.jpg": "train" for i in range(6)}
    for i, bad in enumerate((7, ["staged"], {"mode": "staged"}, "", "   ")):
        out = os.path.join(root, f"hmbadmode_out_{i}", "dataset")
        write(os.path.join(out, "split_manifest.json"),
              json.dumps({"schema_version": 2, "holdout_mode": bad, "pins": pins}))
        try:
            DD.build_detect_dataset(ds, out, "hmbadmode")
            check(False, f"a holdout_mode of {bad!r} must stop the build, not be carried forward")
        except ValueError as e:
            check("holdout_mode" in str(e) and "split_manifest.json" in str(e),
                  f"a holdout_mode of {bad!r} stops the build and the refusal names the field "
                  f"and the file ({e})")


def test_the_holdout_decision_is_a_table(root):
    # The decision itself, without a filesystem: what the destination manifest
    # recorded, which file this build read its pins from, and how many of this
    # build's images those pins actually decided. The third number is what ties
    # a carried-forward policy to its premise: a policy describes a holdout
    # only while the images that holdout pinned are still the ones on disk.
    check(DD.holdout_mode_for_build("transect", DD.PINS_SOURCE_DATASET, 3) == "transect",
          "a recorded policy is carried forward when this build replayed the pins that "
          "manifest holds")
    check(DD.holdout_mode_for_build(DD.HOLDOUT_MODE_STAGED, DD.PINS_SOURCE_DATASET, 3)
          == DD.HOLDOUT_MODE_STAGED,
          "including the staged policy, which is the rebuild the fish models depend on")
    check(DD.holdout_mode_for_build("transect", DD.PINS_SOURCE_DATASET, 0)
          == DD.HOLDOUT_MODE_LAB_RANDOM,
          "a manifest whose pins reached none of this build's images carries nothing "
          "forward: the seeded shuffle drew every image, so lab-random is what happened")
    check(DD.holdout_mode_for_build("transect", DD.PINS_SOURCE_FRESH, 0)
          == DD.HOLDOUT_MODE_LAB_RANDOM,
          "a policy read from a manifest this build replayed nothing from (the segmentation "
          "splitter's, whose pin table is pinned) never reaches a freshly drawn split")
    check(DD.holdout_mode_for_build("transect", DD.PINS_SOURCE_STAGED, 4)
          == DD.HOLDOUT_MODE_STAGED,
          "and a staged pin file that supplied the pins records the staged policy, whatever "
          "an unrelated manifest in the destination folder named")
    check(DD.holdout_mode_for_build(None, DD.PINS_SOURCE_STAGED, 4) == DD.HOLDOUT_MODE_STAGED,
          "a first build off the staged pin file records staged")
    check(DD.holdout_mode_for_build(None, DD.PINS_SOURCE_STAGED, 0)
          == DD.HOLDOUT_MODE_LAB_RANDOM,
          "a staged pin file naming no image this build holds decided nothing, so the seeded "
          "draw is recorded instead")
    check(DD.holdout_mode_for_build(None, DD.PINS_SOURCE_FRESH, 0) == DD.HOLDOUT_MODE_LAB_RANDOM,
          "a first build with no pins anywhere records lab-random")
    check(DD.holdout_mode_for_build(None, DD.PINS_SOURCE_DATASET, 5) == DD.HOLDOUT_MODE_LAB_RANDOM,
          "and a manifest that pinned images without naming a policy records lab-random, the "
          "value every detection build wrote before this change")
    check(set(DD.HOLDOUT_MODES) == {DD.HOLDOUT_MODE_LAB_RANDOM, DD.HOLDOUT_MODE_STAGED},
          f"the two values a first build can choose between are named (got {DD.HOLDOUT_MODES})")
    try:
        DD.holdout_mode_for_build(None, "some other source", 0)
        check(False, "a pins source outside the three constants must be refused")
    except ValueError as e:
        check("pins_source" in str(e) and "holdout_mode" in str(e),
              f"a pins source outside the three constants is refused by name ({e})")
    for bad in (-1, 1.5, "3", None, True):
        try:
            DD.holdout_mode_for_build("staged", DD.PINS_SOURCE_DATASET, bad)
            check(False, f"an n_replayed of {bad!r} must be refused, not counted")
        except (TypeError, ValueError) as e:
            check("n_replayed" in str(e),
                  f"an n_replayed of {bad!r} is refused by name ({e})")


def test_a_seg_manifest_in_the_folder_does_not_relabel_a_fresh_draw(root):
    # Both splitters write split_manifest.json into the same <step6>/dataset
    # folder, and a project that trained a segmentation model and then picks a
    # staged box dataset rebuilds over the segmentation splitter's manifest.
    # That manifest records the frozen transect holdout and keeps its pin table
    # under "pinned", which this builder never reads, so it replays nothing
    # from it: the draw is this module's own and has to be recorded as its own.
    # Stamping "transect" on it would make the strongest claim in the module's
    # vocabulary about a seeded shuffle.
    seg_manifest = {
        "schema_version": 2,
        "holdout_mode": "transect",
        "holdout_transects": {"val": [5], "test": [6]},
        "pinned": {f"TCRMP20260101_clip_SITE_T60{i}.jpeg": "test" for i in range(3)},
    }
    ds = build_small_bbox_source(root, "segfolder", 12)
    out = os.path.join(root, "segfolder_out", "dataset")
    write(os.path.join(out, "split_manifest.json"), json.dumps(seg_manifest))
    m = DD.build_detect_dataset(ds, out, "segfolder", valid_frac=0.2, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a build over the segmentation splitter's manifest records the seeded draw it "
          f"actually made (got {m.get('holdout_mode')!r})")
    check(m.get("pins_source") == DD.PINS_SOURCE_FRESH,
          f"with pins_source fresh, because that manifest's pin table is one this builder "
          f"never reads (got {m.get('pins_source')!r})")
    check(m["n_replayed"] == 0 and m["n_fresh"] == 12,
          f"and nothing was replayed (got replayed={m['n_replayed']}, fresh={m['n_fresh']})")
    check(load_manifest(out).get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          "the manifest on disk says the same")
    clean = os.path.join(root, "segfolder_clean", "dataset")
    c = DD.build_detect_dataset(ds, clean, "segfolder", valid_frac=0.2, test_frac=0.1)
    check(c["pins"] == m["pins"] and c["split_fingerprint"] == m["split_fingerprint"],
          f"and the same source into an empty folder gives the same assignment, so the "
          f"segmentation manifest moved no image either (got {c['split_fingerprint']} "
          f"against {m['split_fingerprint']})")
    # The staged pin file still wins over that manifest, and still records its
    # own policy rather than the transect one.
    staged_src = build_small_bbox_source(root, "segfolder_staged", 12)
    _write_source_pins(staged_src, _staged_pins_for("segfolder_staged", 12))
    out2 = os.path.join(root, "segfolder_staged_out", "dataset")
    write(os.path.join(out2, "split_manifest.json"), json.dumps(seg_manifest))
    m2 = DD.build_detect_dataset(staged_src, out2, "segfolder_staged")
    check(m2.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED
          and m2.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"a staged pin file over the same manifest records staged, not transect "
          f"(got {m2.get('holdout_mode')!r}, {m2.get('pins_source')!r})")
    check(m2["n_replayed"] == 12 and m2["split_fingerprint"] == DD.split_fingerprint(m2["pins"]),
          f"replaying all twelve of its pins (got replayed={m2['n_replayed']})")


def test_a_rebuild_that_replays_nothing_records_its_own_draw(root):
    # The same defect reached the other way: the destination holds this
    # builder's own manifest, staged holdout and all, but the Step 5 source has
    # been switched to a dataset whose basenames that manifest never named. Not
    # one image keeps a split the staged experiment chose, so the staged policy
    # describes nothing on disk.
    first_src = build_small_bbox_source(root, "switchone", 12)
    _write_source_pins(first_src, _staged_pins_for("switchone", 12))
    out = os.path.join(root, "switch_out", "dataset")
    first = DD.build_detect_dataset(first_src, out, "switchone", valid_frac=0.2, test_frac=0.1)
    check(first.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED,
          f"the folder starts out recording the staged holdout (got "
          f"{first.get('holdout_mode')!r})")
    other_src = build_small_bbox_source(root, "switchtwo", 12)
    second = DD.build_detect_dataset(other_src, out, "switchtwo", valid_frac=0.2, test_frac=0.1)
    check(second.get("holdout_mode") == DD.HOLDOUT_MODE_LAB_RANDOM,
          f"a rebuild from a source that manifest pinned none of records the seeded draw "
          f"(got {second.get('holdout_mode')!r})")
    check(second.get("pins_source") == DD.PINS_SOURCE_FRESH and second["n_replayed"] == 0,
          f"the old source's manifest is set aside, so the record says the split is freshly "
          f"drawn (got {second.get('pins_source')!r}, replayed={second['n_replayed']})")
    clean = os.path.join(root, "switch_clean", "dataset")
    c = DD.build_detect_dataset(other_src, clean, "switchtwo", valid_frac=0.2, test_frac=0.1)
    check(c["pins"] == second["pins"] and c["split_fingerprint"] == second["split_fingerprint"],
          f"and the old manifest moved no image of the new source (got "
          f"{second['split_fingerprint']} against {c['split_fingerprint']})")


def test_holdout_mode_staged_when_only_part_of_the_source_is_pinned(root):
    # A staged pin file that names part of the source: the staged experiment
    # chose those images, the seeded shuffle drew the rest, and holdout_mode
    # says staged for the whole build. n_replayed against n_pairs is what tells
    # a reader how much of the holdout the staged file really chose, which is
    # why both numbers sit beside the value in every manifest.
    ds = build_small_bbox_source(root, "partpins", 100)
    _write_source_pins(ds, {f"partpins_{i:03d}.jpg": "test" for i in range(5)})
    out = os.path.join(root, "partpins_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "partpins", valid_frac=0.1, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED
          and m.get("pins_source") == DD.PINS_SOURCE_STAGED,
          f"a partly pinned staged source records the staged policy (got "
          f"{m.get('holdout_mode')!r}, {m.get('pins_source')!r})")
    check(m["n_replayed"] == 5 and m["n_fresh"] == 95 and m["n_pairs"] == 100,
          f"beside the two counts that say how far the staged file reached "
          f"(got replayed={m['n_replayed']}, fresh={m['n_fresh']}, pairs={m['n_pairs']})")
    check(m["n_replayed"] + m["n_fresh"] == m["n_pairs"],
          "and those two counts account for every pair in the source, so a reader can read "
          "the share the staged experiment chose straight off the manifest")
    check(all(m["pins"][f"partpins_{i:03d}.jpg"] == "test" for i in range(5)),
          "every staged pin is replayed exactly as the staged file names it")
    check(m["split_fingerprint"] == DD.split_fingerprint(m["pins"]),
          "and the fingerprint is the fingerprint of the assignment written")
    # A staged file that names none of this source's images is refused: until
    # 2026-09-26 it was dropped in silence and the whole source re-drawn.
    ghost = build_small_bbox_source(root, "ghostonly", 8)
    _write_source_pins(ghost, {"a_photo_that_was_never_staged.jpg": "test"})
    out2 = os.path.join(root, "ghostonly_out", "dataset")
    msg = _refused(lambda: DD.build_detect_dataset(ghost, out2, "ghostonly",
                                                   valid_frac=0.2, test_frac=0.1))
    check("a_photo_that_was_never_staged.jpg" in msg,
          f"a staged file naming no image this source holds stops the build: {msg!r}")


def test_holdout_mode_change_moved_no_image(root):
    # Known-answer fingerprints measured on this box on 2026-09-15 against the
    # code as it stood BEFORE holdout_mode was decided from the assignment.
    # holdout_mode is a label on the split, never an input to it, so every one
    # of these must survive the change: a moved fingerprint would make every
    # run built on one of these datasets read as not comparable to the runs
    # already scored on it.
    before_staged = "ea6750a4942f747b"
    before_fresh = "cc8b774b153632c8"
    before_old = "59273b036e429767"

    staged_src = build_small_bbox_source(root, "mode_staged", 12)
    _write_source_pins(staged_src, _staged_pins_for("mode_staged", 12))
    staged_out = os.path.join(root, "fp_staged", "dataset")
    a = DD.build_detect_dataset(staged_src, staged_out, "mode-seed",
                                valid_frac=0.2, test_frac=0.1)
    check(a["split_fingerprint"] == before_staged,
          f"a staged-pin build fingerprints exactly as it did before the change "
          f"(got {a['split_fingerprint']}, was {before_staged})")
    a2 = DD.build_detect_dataset(staged_src, staged_out, "other-seed",
                                 valid_frac=0.5, test_frac=0.4)
    check(a2["split_fingerprint"] == before_staged,
          f"and so does a rebuild over it (got {a2['split_fingerprint']})")

    fresh_src = build_small_bbox_source(root, "mode_fresh", 12)
    fresh_out = os.path.join(root, "fp_fresh", "dataset")
    b = DD.build_detect_dataset(fresh_src, fresh_out, "mode-seed",
                                valid_frac=0.2, test_frac=0.1)
    check(b["split_fingerprint"] == before_fresh,
          f"a seeded draw fingerprints exactly as it did before the change "
          f"(got {b['split_fingerprint']}, was {before_fresh})")
    b2 = DD.build_detect_dataset(fresh_src, fresh_out, "other-seed",
                                 valid_frac=0.5, test_frac=0.4)
    check(b2["split_fingerprint"] == before_fresh,
          f"and so does a rebuild over it (got {b2['split_fingerprint']})")

    old_src = build_small_bbox_source(root, "mode_old", 12)
    old_out = os.path.join(root, "fp_old", "dataset")
    old_pins = {f"mode_old_{i:03d}.jpg": ("test" if i < 3 else "train") for i in range(12)}
    write(os.path.join(old_out, "split_manifest.json"),
          json.dumps({"schema_version": 1, "seed": "mode-seed", "pins": old_pins}))
    c = DD.build_detect_dataset(old_src, old_out, "mode-seed", valid_frac=0.2, test_frac=0.1)
    check(c["split_fingerprint"] == before_old,
          f"and a rebuild over a manifest with no holdout_mode fingerprints as it did "
          f"(got {c['split_fingerprint']}, was {before_old})")


def test_a_staged_holdout_is_gated_as_pinned(root):
    # evaluate_run gates a run on whether its split is frozen and reproducible.
    # A staged holdout is 7572 pins on the real fish dataset, so it must gate
    # as pinned: reading unpinned would silently drop every promotion gate for
    # the fish models. Proven by running the gate's own reader over a dataset
    # this builder wrote, not by reasoning about it.
    import evaluate_run as ER
    ds = build_small_bbox_source(root, "hmgate", 10)
    _write_source_pins(ds, _staged_pins_for("hmgate", 10))
    out = os.path.join(root, "hmgate_out", "dataset")
    m = DD.build_detect_dataset(ds, out, "hmgate", valid_frac=0.2, test_frac=0.1)
    check(m.get("holdout_mode") == DD.HOLDOUT_MODE_STAGED,
          f"the dataset under test really does record the staged holdout "
          f"(got {m.get('holdout_mode')!r})")
    check(ER._split_pinned(out) is True,
          "and evaluate_run reads it as a pinned split, so the promotion gates still run")
    check(ER._split_pinned(os.path.join(root, "hmgate_no_such_dataset")) is False,
          "while a folder with no manifest at all still reads as unpinned")
    # The value itself must not be what decides it: a manifest carrying the new
    # value and an empty pin table has nothing to prove a frozen split with.
    empty = os.path.join(root, "hmgate_empty")
    write(os.path.join(empty, "split_manifest.json"),
          json.dumps({"schema_version": 2, "holdout_mode": DD.HOLDOUT_MODE_STAGED, "pins": {}}))
    check(ER._split_pinned(empty) is False,
          "and a staged manifest with an empty pin table is still unpinned, because the pins "
          "are what freeze a split")

SPLIT_RECORD_CASES = (
    test_split_manifest_records_seed_fractions_and_fingerprint,
    test_fingerprint_is_stable_across_runs,
    test_fingerprint_differs_when_an_image_moves_split,
    test_existing_pins_are_replayed_when_fractions_change,
    test_reader_reports_not_recorded_for_an_old_manifest,
    test_two_seeds_produce_different_splits,
    test_adversarial_single_image_source,
    test_adversarial_source_with_no_label_files,
    test_adversarial_corrupt_manifest_json,
    test_adversarial_manifest_missing_key,
    test_adversarial_unicode_and_space_basenames,
    test_adversarial_large_basename_list_is_linear,
    test_adversarial_read_only_destination,
    test_adversarial_fraction_and_seed_validation,
    test_reader_imports_without_flask_or_ultralytics,
    test_adversarial_rebuild_with_many_pins_is_linear,
    test_adversarial_bad_pin_value_in_manifest,
    test_adversarial_symlinked_dataset_dir,
    test_adversarial_zero_train_split,
    test_adversarial_folder_named_like_an_image,
    test_adversarial_corrupt_seg_manifest_blocks_detect_view,
    test_adversarial_failed_swap_puts_the_old_dataset_back,
    test_adversarial_rollback_leaves_only_an_empty_folder_when_rmdir_fails,
    # Staged split pins and the staged names table (2026-09-15).
    test_source_pins_replayed_and_recorded,
    test_source_pins_refused_when_malformed,
    test_fresh_split_recorded_when_no_pins,
    test_names_full_in_data_yaml,
    test_names_csv_wrong_header_refused,
    test_names_csv_duplicate_token_refused,
    test_names_csv_with_a_byte_order_mark_is_read,
    test_zero_train_split_names_whatever_held_the_images_out,
    test_adversarial_source_pins_for_absent_images,
    test_stem_keyed_pins_are_refused_not_redrawn,
    test_a_project_split_naming_a_vanished_photo_is_refused,
    test_a_grown_dataset_keeps_every_pinned_photo,
    test_an_unlabelled_photo_is_named_as_unlabelled,
    test_the_same_photos_through_another_path_keep_their_split,
    test_many_unlabelled_photos_are_refused_quickly,
    test_adversarial_sixty_thousand_source_pins,
    # holdout_mode says how the holdout was really chosen (2026-09-15).
    test_holdout_mode_staged_when_the_staged_file_supplied_the_pins,
    test_holdout_mode_lab_random_when_nothing_pinned_the_split,
    test_rebuild_keeps_the_staged_holdout_mode,
    test_rebuild_keeps_the_lab_random_holdout_mode,
    test_rebuild_over_a_manifest_with_no_holdout_mode,
    test_rebuild_refuses_a_holdout_mode_that_is_not_a_string,
    test_the_holdout_decision_is_a_table,
    test_a_seg_manifest_in_the_folder_does_not_relabel_a_fresh_draw,
    test_a_rebuild_that_replays_nothing_records_its_own_draw,
    test_holdout_mode_staged_when_only_part_of_the_source_is_pinned,
    test_holdout_mode_change_moved_no_image,
    test_a_staged_holdout_is_gated_as_pinned,
)


def run_split_record_cases(root):
    """Run every named case; a case that raises is one failure, not a crash
    of the whole suite."""
    import traceback
    for case in SPLIT_RECORD_CASES:
        print(f" {case.__name__}")
        try:
            case(root)
        except Exception as e:
            traceback.print_exc()
            check(False, f"{case.__name__} raised {e!r}")


root = tempfile.mkdtemp(prefix="detect_dataset_test_")
try:
    # ---- bbox_pairs -----------------------------------------------------
    print("\n-- bbox_pairs --")
    ds = build_bbox_source(root)
    pairs, unlabelled = DD.bbox_pairs(ds)
    check(len(pairs) == 20 and unlabelled == 1,
          f"every labelled image is a pair, the unlabelled one is counted (got {len(pairs)}, {unlabelled})")
    check(DD.bbox_label_dir(ds) == os.path.join(ds, "labels"), "bbox_label_dir finds labels/")
    os.makedirs(os.path.join(ds, "labels_localized"), exist_ok=True)
    write(os.path.join(ds, "labels_localized", "img_000.txt"), "0 0.5 0.5 0.2 0.3\n")
    check(DD.bbox_label_dir(ds) == os.path.join(ds, "labels_localized"),
          "labels_localized wins over labels when it is present")
    pairs_loc, unlabelled_loc = DD.bbox_pairs(ds)
    check(len(pairs_loc) == 1 and unlabelled_loc == 20,
          f"bbox_pairs reads the preferred label dir (got {len(pairs_loc)}, {unlabelled_loc})")
    shutil.rmtree(os.path.join(ds, "labels_localized"))
    check(DD.read_classes_txt(os.path.join(ds, "classes.txt")) == ["AAGA", "ACER"],
          "read_classes_txt reads the pinned class list")

    # ---- build_detect_dataset -------------------------------------------
    print("\n-- build_detect_dataset --")
    out = os.path.join(root, "_models", "boxlab", "dataset")
    summary = DD.build_detect_dataset(ds, out, "boxlab")
    check(summary["n_pairs"] == 20 and summary["n_unlabelled"] == 1,
          f"summary counts pairs and unlabelled images (got {summary['n_pairs']}, {summary['n_unlabelled']})")
    counts = {s: len(os.listdir(os.path.join(out, s, "images"))) for s in DD.SPLIT_DIRS}
    check(sum(counts.values()) == 20 and counts["valid"] == 2 and counts["test"] == 2,
          f"80/10/10 across train, valid and test (got {counts})")
    for s in DD.SPLIT_DIRS:
        imgs = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, s, "images")))
        lbls = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(out, s, "labels")))
        check(imgs == lbls, f"{s}/ carries one label per image")
    import yaml as _yaml
    doc = _yaml.safe_load(open(os.path.join(out, "data.yaml")))
    check(doc["train"] == "train/images" and doc["val"] == "valid/images"
          and doc["test"] == "test/images" and doc["path"] == os.path.abspath(out),
          f"data.yaml uses the split paths pinned_split writes (got {doc})")
    check(doc["names"] == {0: "AAGA", 1: "ACER"}, f"data.yaml names come from classes.txt (got {doc['names']})")
    check(os.path.isfile(os.path.join(out, "test.yaml")), "test.yaml written for parity with pinned_split")
    manifest = json.load(open(os.path.join(out, "split_manifest.json")))
    check(manifest["holdout_mode"] == "lab-random" and len(manifest["pins"]) == 20,
          f"split_manifest is honest about the holdout (got {manifest.get('holdout_mode')})")

    out2 = os.path.join(root, "_models", "boxlab2", "dataset")
    DD.build_detect_dataset(ds, out2, "boxlab")
    m2 = json.load(open(os.path.join(out2, "split_manifest.json")))
    check(m2["pins"] == manifest["pins"], "the same seed name splits the same images the same way")
    out3 = os.path.join(root, "_models", "boxlab3", "dataset")
    DD.build_detect_dataset(ds, out3, "otherlab")
    m3 = json.load(open(os.path.join(out3, "split_manifest.json")))
    check(m3["pins"] != manifest["pins"], "a different seed name gets its own split")
    write(os.path.join(ds, "images", "img_777.jpg"), "x")
    write(os.path.join(ds, "labels", "img_777.txt"), "1 0.5 0.5 0.2 0.3\n")
    DD.build_detect_dataset(ds, out, "boxlab")
    m4 = json.load(open(os.path.join(out, "split_manifest.json")))
    check(all(m4["pins"][k] == v for k, v in manifest["pins"].items()) and "img_777.jpg" in m4["pins"],
          "a rebuild replays every existing pin and only assigns the new image")
    os.remove(os.path.join(ds, "images", "img_777.jpg"))
    os.remove(os.path.join(ds, "labels", "img_777.txt"))
    try:
        DD.build_detect_dataset(os.path.join(root, "bbox_datasets", "partial"),
                                os.path.join(root, "_models", "nope", "dataset"), "nope")
        check(False, "a source with no image and label pairs is refused")
    except ValueError as e:
        check("pair" in str(e).lower() or "image" in str(e).lower(),
              f"a source with no image and label pairs is refused ({e})")

    # ---- seg_labels_to_boxes --------------------------------------------
    print("\n-- seg_labels_to_boxes --")
    lab_in = os.path.join(root, "conv", "in")
    lab_out = os.path.join(root, "conv", "out")
    write(os.path.join(lab_in, "a.txt"), "0 0.1 0.2 0.5 0.2 0.5 0.6 0.1 0.6\n")
    write(os.path.join(lab_in, "b.txt"), "1 -0.5 0.5 1.5 0.5 1.5 1.5\n")
    write(os.path.join(lab_in, "c.txt"),
          "0 0.5 0.5 0.5 0.5 0.5 0.5\n"      # zero area
          "0 0.5 0.5\n"                       # one point
          "0 0.1 0.1 0.2 0.2\n"               # two points
          "x 0.1 0.1 0.2 0.2 0.3 0.3\n"       # class id is not a number
          "0 0.1 0.1 0.9 0.1 0.9 0.9\n")      # the one real polygon
    write(os.path.join(lab_in, "neg.txt"), "")
    conv = DD.seg_labels_to_boxes(lab_in, lab_out)
    check(open(os.path.join(lab_out, "a.txt")).read() == "0 0.300000 0.400000 0.400000 0.400000\n",
          "a polygon becomes the normalized box of its extent")
    check(open(os.path.join(lab_out, "b.txt")).read() == "1 0.500000 0.750000 1.000000 0.500000\n",
          "coordinates outside the frame are clamped to [0,1]")
    check(open(os.path.join(lab_out, "c.txt")).read() == "0 0.500000 0.500000 0.800000 0.800000\n",
          f"degenerate and unparsable lines are dropped (got {open(os.path.join(lab_out, 'c.txt')).read()!r})")
    check(os.path.isfile(os.path.join(lab_out, "neg.txt"))
          and open(os.path.join(lab_out, "neg.txt")).read() == "",
          "an empty label file stays empty: a negative frame is still a negative")
    check(conv["n_files"] == 4 and conv["n_boxes"] == 3 and conv["n_dropped"] == 4,
          f"the conversion counts what it wrote and what it dropped (got {conv})")
    # The three ways a file ends with no boxes mean different things and are
    # counted apart: a real negative, a frame whose every line was dropped, and
    # a file that could not be read at all.
    check(conv["n_negative"] == 1 and conv["n_emptied"] == 0 and conv["n_unreadable"] == 0,
          f"a file that was already empty counts as a negative (got {conv})")
    lab_in2 = os.path.join(root, "conv2", "in")
    lab_out2 = os.path.join(root, "conv2", "out")
    write(os.path.join(lab_in2, "lost.txt"), "0 0.5 0.5\n0 0.1 0.1 0.2 0.2\n")
    write(os.path.join(lab_in2, "neg.txt"), "")
    conv2 = DD.seg_labels_to_boxes(lab_in2, lab_out2)
    check(conv2["n_emptied"] == 1 and conv2["n_negative"] == 1 and conv2["n_dropped"] == 2,
          f"a frame whose every line was dropped is counted apart from a true negative "
          f"(got {conv2})")
    # chmod 000 does not stop root, so the unreadable-file checks only mean
    # something for an ordinary user.
    CAN_TEST_UNREADABLE = os.geteuid() != 0
    unreadable = os.path.join(root, "conv3", "in")
    write(os.path.join(unreadable, "bad.txt"), "0 0.1 0.1 0.9 0.1 0.9 0.9\n")
    os.chmod(os.path.join(unreadable, "bad.txt"), 0o000)
    if not CAN_TEST_UNREADABLE:
        print("  SKIP unreadable-label checks (running as root)")
    try:
        conv3 = DD.seg_labels_to_boxes(unreadable, os.path.join(root, "conv3", "out"))
        if CAN_TEST_UNREADABLE:
            check(conv3["n_unreadable"] == 1 and conv3["n_negative"] == 0,
                  f"an unreadable label file is counted as unreadable, never as a negative "
                  f"(got {conv3})")
            check(not os.path.isfile(os.path.join(root, "conv3", "out", "bad.txt")),
                  "no output label is written for a file that could not be read")
    finally:
        os.chmod(os.path.join(unreadable, "bad.txt"), 0o644)

    # ---- dataset_label_kind ---------------------------------------------
    print("\n-- dataset_label_kind --")
    seg_ds = build_seg_dataset(root)
    check(DD.dataset_label_kind(seg_ds) == "segment", "a polygon dataset reads as segment")
    check(DD.dataset_label_kind(out) == "detect", "a box dataset reads as detect")
    # Reef clips are negative-heavy. A scan that gave up early would call a
    # segmentation dataset "empty" and skip building dataset_detect entirely.
    heavy = os.path.join(root, "negative_heavy", "dataset")
    for i in range(600):
        write(os.path.join(heavy, "train", "images", f"neg_{i:04d}.jpeg"), "img")
        write(os.path.join(heavy, "train", "labels", f"neg_{i:04d}.txt"), "")
    write(os.path.join(heavy, "train", "images", "zz_has_a_mask.jpeg"), "img")
    write(os.path.join(heavy, "train", "labels", "zz_has_a_mask.txt"),
          "0 0.1 0.2 0.5 0.2 0.5 0.6\n")
    write(os.path.join(heavy, "data.yaml"),
          "path: %s\ntrain: train/images\nval: valid/images\nnames:\n  0: PA\n"
          % os.path.abspath(heavy))
    check(DD.dataset_label_kind(heavy) == "segment",
          "600 negatives before the first polygon still reads as segment")
    all_neg = os.path.join(root, "all_negative", "dataset")
    for i in range(3):
        write(os.path.join(all_neg, "train", "images", f"neg_{i}.jpeg"), "img")
        write(os.path.join(all_neg, "train", "labels", f"neg_{i}.txt"), "")
    write(os.path.join(all_neg, "data.yaml"),
          "path: %s\ntrain: train/images\nval: valid/images\nnames:\n  0: PA\n"
          % os.path.abspath(all_neg))
    check(DD.dataset_label_kind(all_neg) == "empty",
          "only a dataset with no label line at all reads as empty")

    # ---- build_detect_from_seg_dataset ----------------------------------
    print("\n-- build_detect_from_seg_dataset --")
    det = os.path.join(os.path.dirname(seg_ds), "dataset_detect")
    built = DD.build_detect_from_seg_dataset(seg_ds, det)
    check(os.path.isdir(det), "dataset_detect is written beside dataset")
    check(os.path.isdir(seg_ds) and os.path.isfile(os.path.join(seg_ds, "data.yaml")),
          "the seg dataset is left intact for later seg runs")
    seg_train_label = open(os.path.join(seg_ds, "train", "labels",
                                        "TCRMP20260101_clip_SITE_T600.txt")).read()
    check(seg_train_label.startswith("0 0.1 0.2 0.5"),
          "the seg dataset's own labels are still polygons")
    link = os.path.join(det, "train", "images", "TCRMP20260101_clip_SITE_T600.jpeg")
    check(os.path.islink(link), "every image in dataset_detect is a symlink, never a copy")
    check(not os.path.isabs(os.readlink(link)),
          f"the symlink is relative so the pair survives a move (got {os.readlink(link)})")
    check(os.path.realpath(link) == os.path.realpath(
        os.path.join(seg_ds, "train", "images", "TCRMP20260101_clip_SITE_T600.jpeg")),
        "the symlink resolves to the seg dataset's image")
    check(open(os.path.join(det, "train", "labels", "TCRMP20260101_clip_SITE_T600.txt")).read()
          == "0 0.300000 0.400000 0.400000 0.400000\n",
          "the converted label is the polygon's box")
    check(open(os.path.join(det, "test", "labels", "TCRMP20260101_clip_SITE_T600.txt")).read()
          == "0 0.500000 0.500000 1.000000 1.000000\n",
          "a full-frame polygon converts to a full-frame box")
    ddoc = _yaml.safe_load(open(os.path.join(det, "data.yaml")))
    check(ddoc["path"] == os.path.abspath(det) and ddoc["val"] == "valid/images"
          and ddoc["train"] == "train/images" and ddoc["test"] == "test/images",
          f"dataset_detect/data.yaml points at itself (got {ddoc})")
    check(ddoc["names"] == {0: "PA", 1: "MC"}, f"class names carry over unchanged (got {ddoc['names']})")
    tdoc = _yaml.safe_load(open(os.path.join(det, "test.yaml")))
    check(tdoc["val"] == "test/images", "test.yaml is written for parity with pinned_split")
    dman = json.load(open(os.path.join(det, "split_manifest.json")))
    check(dman["holdout_mode"] == "transect",
          "the pinned holdout is carried across so an evaluation still reads it as pinned")
    check(os.path.abspath(seg_ds) == dman.get("derived_from"),
          f"the manifest records the seg dataset it was derived from (got {dman.get('derived_from')})")
    check(built["counts"]["train"] == 3 and built["counts"]["valid"] == 2
          and built["counts"]["test"] == 2,
          f"every image of every split is carried across (got {built['counts']})")
    # Rebuilding replaces the tree in place rather than stacking onto it.
    built2 = DD.build_detect_from_seg_dataset(seg_ds, det)
    check(built2["counts"] == built["counts"], "a rebuild is idempotent")
    check(len(os.listdir(os.path.join(det, "train", "images"))) == 3,
          "a rebuild leaves no duplicate images behind")

    # ---- prepare ---------------------------------------------------------
    print("\n-- prepare --")
    yaml_path, built3 = DD.prepare(seg_ds)
    check(yaml_path == os.path.join(os.path.dirname(seg_ds), "dataset_detect", "data.yaml"),
          f"a seg dataset trains detect from dataset_detect (got {yaml_path})")
    check(built3 is not None, "prepare reports the conversion summary for a seg source")
    yaml_path2, built4 = DD.prepare(out)
    check(yaml_path2 == os.path.join(out, "data.yaml"),
          f"a prebuilt box dataset trains from its own data.yaml (got {yaml_path2})")
    check(built4 is None and not os.path.isdir(os.path.join(os.path.dirname(out), "dataset_detect")),
          "a prebuilt box dataset is never converted and nothing is written beside it")
    # A negative-heavy seg dataset must still be converted.
    yaml_path3, built5 = DD.prepare(heavy)
    check(built5 is not None and built5["conversions"]["train"]["n_boxes"] == 1
          and built5["conversions"]["train"]["n_negative"] == 600,
          f"a negative-heavy seg dataset is converted, negatives and all (got "
          f"{built5['conversions']['train'] if built5 else None})")
    check(os.path.isfile(yaml_path3), "and its data.yaml is written")
    # An all-negative dataset cannot be told from a seg dataset with no masks
    # yet, so it is built rather than passed through as if it held boxes.
    yaml_path4, built6 = DD.prepare(all_neg)
    check(built6 is not None
          and yaml_path4 == os.path.join(os.path.dirname(all_neg), "dataset_detect", "data.yaml"),
          f"an all-negative dataset is built, not passed through (got {yaml_path4})")
    check(built6["conversions"]["train"]["n_negative"] == 3
          and built6["conversions"]["train"]["n_boxes"] == 0,
          f"and every negative survives the conversion (got {built6['conversions']['train']})")
    # An unreadable label file stops the build instead of turning an annotated
    # frame into a background image.
    bad_ds = os.path.join(root, "unreadable", "dataset")
    write(os.path.join(bad_ds, "train", "images", "a.jpeg"), "img")
    write(os.path.join(bad_ds, "train", "labels", "a.txt"), "0 0.1 0.2 0.5 0.2 0.5 0.6\n")
    write(os.path.join(bad_ds, "data.yaml"),
          "path: %s\ntrain: train/images\nval: valid/images\nnames:\n  0: PA\n"
          % os.path.abspath(bad_ds))
    os.chmod(os.path.join(bad_ds, "train", "labels", "a.txt"), 0o000)
    try:
        DD.prepare(bad_ds)
        check(not CAN_TEST_UNREADABLE, "an unreadable label file stops the build")
    except ValueError as e:
        check("could not be read" in str(e), f"an unreadable label file stops the build ({e})")
    finally:
        os.chmod(os.path.join(bad_ds, "train", "labels", "a.txt"), 0o644)
    if CAN_TEST_UNREADABLE:
        check(not os.path.isdir(os.path.join(os.path.dirname(bad_ds), "dataset_detect")),
              "and leaves no half-built dataset_detect behind")

    # ---- CLI -------------------------------------------------------------
    print("\n-- CLI --")
    proc = subprocess.run([sys.executable, os.path.join(_SRC, "detect_dataset.py"),
                           "prepare", "--dataset", seg_ds],
                          capture_output=True, text=True)
    check(proc.returncode == 0, f"prepare exits 0 (stderr tail: {proc.stderr.strip()[-200:]!r})")
    check(proc.stdout.strip() == os.path.join(os.path.dirname(seg_ds), "dataset_detect", "data.yaml"),
          f"prepare prints only the data.yaml path on stdout (got {proc.stdout.strip()!r})")
    proc_bad = subprocess.run([sys.executable, os.path.join(_SRC, "detect_dataset.py"),
                               "prepare", "--dataset", os.path.join(root, "no_such_dataset")],
                              capture_output=True, text=True)
    check(proc_bad.returncode != 0, "prepare fails loudly when the dataset is missing")

    # ---- Split record: seed, fractions, fingerprint ---------------------
    print("\n-- split record: seed, fractions, fingerprint --")
    run_split_record_cases(root)
finally:
    shutil.rmtree(root, ignore_errors=True)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
