"""Tests for pinned_split.py (frozen-holdout, by-transect split).

Numbered runs 1 to 6 cover the split rules (transect holdout, replay,
degrade and auto-upgrade, legacy adoption, determinism). The named split
record cases after them (2026-09-04) cover the seed, fractions, holdout
mode and fingerprint every manifest records, the shared reader in
detect_dataset.py, and the adversarial inputs.

Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_pinned_split.py"""
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pinned_split as PS

_fail = 0


def check(cond, msg):
    """Record a failed check: count it and print FAIL with ``msg``."""
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)


# ---------------------------------------------------------------------------
# parse_transect unit checks (real filenames, WITH extensions)
# ---------------------------------------------------------------------------
check(PS.parse_transect("TCRMP20140930_clip_SCP_T623.jpeg") == 6, "3-digit T623 -> transect 6")
check(PS.parse_transect("TCRMP20140930_clip_SCP_T102.jpeg") == 1, "3-digit T102 -> transect 1")
check(PS.parse_transect("TCRMP20140930_clip_SCP_T6.jpeg") is None, "1-digit T6 -> None")
check(PS.parse_transect("TCRMP20140930_clip_SCP_T61.jpeg") is None, "2-digit T61 -> None")
check(PS.parse_transect("TCRMP20140930_clip_SCP_T6012.jpeg") is None, "4-digit T6012 -> None")


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "SS", 1: "MCAV"}

# Minimal 1x1-byte placeholder "image" content: pinned_split.py never opens
# image bytes, it only copies files by name, so content is irrelevant.
IMG_BYTES = b"\xff\xd8\xff\xd9"


def _label_text(classes):
    """YOLO-seg label lines, one polygon per class id in ``classes``."""
    return "".join(f"{c} 0.5 0.5 0.1 0.1 0.5 0.4 0.4 0.5\n" for c in classes)


def make_export(root, frames):
    """frames: list of (basename_without_ext, [class_ids]) tuples.
    Writes all_images/<name>.jpeg + all_labels/<name>.txt + data.yaml."""
    img_dir = os.path.join(root, "all_images")
    lbl_dir = os.path.join(root, "all_labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for name, classes in frames:
        with open(os.path.join(img_dir, name + ".jpeg"), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write(_label_text(classes))
    with open(os.path.join(root, "data.yaml"), "w") as f:
        import yaml
        yaml.dump({"path": root, "train": "all_images", "val": "all_images",
                   "names": CLASS_NAMES}, f)


def frame_name(site, transect_3digit, idx):
    """A TCRMP frame stem for ``site`` whose T-number starts with ``transect_3digit``."""
    # e.g. TCRMP20140101_clip_AAA_T102 -> transect 1, site AAA
    return f"TCRMP20140101_clip_{site}_T{transect_3digit}{idx}"


def build_full_fixture(root):
    """2 sites (AAA, BBB), transects 1-6, a few frames each, 2 classes
    distributed so every transect has both classes represented and
    min_samples (default 10) is satisfied for train-split classes."""
    frames = []
    for site in ("AAA", "BBB"):
        for transect in range(1, 7):
            for idx in range(3):
                # 3-digit T-number: first digit = transect, so e.g. T1_00 -> "100"
                tnum = f"{transect}{idx:02d}"
                name = f"TCRMP20140101_clip_{site}_T{tnum}"
                classes = [0, 1] if idx % 2 == 0 else [0]
                frames.append((name, classes))
    make_export(root, frames)
    return frames


def read_manifest(out_dir):
    """The split_manifest.json under ``out_dir`` as a dict."""
    with open(os.path.join(out_dir, "split_manifest.json")) as f:
        return json.load(f)


def split_basenames(out_dir, split):
    """Image basenames under <out_dir>/<split>/images; empty set when absent."""
    img_dir = os.path.join(out_dir, split, "images")
    if not os.path.isdir(img_dir):
        return set()
    return set(os.listdir(img_dir))


def transects_in(names):
    """Transect numbers parsed from ``names``, unparseable names ignored."""
    result = set()
    for n in names:
        t = PS.parse_transect(n)
        if t is not None:
            result.add(t)
    return result


# ---------------------------------------------------------------------------
# Run 1: fresh, default transect mode
# ---------------------------------------------------------------------------
tmp1 = tempfile.mkdtemp(prefix="pinned_split_run1_")
try:
    src1 = os.path.join(tmp1, "export")
    out1 = os.path.join(tmp1, "dataset")
    build_full_fixture(src1)
    rc = PS.main(["--src", src1, "--out", out1, "--min_samples", "1"])
    check(rc == 0 or rc is None, "run1: exits cleanly")

    manifest1 = read_manifest(out1)
    check(manifest1["holdout_mode"] == "transect", "run1: holdout_mode=transect")

    test_names = split_basenames(out1, "test")
    valid_names = split_basenames(out1, "valid")
    train_names = split_basenames(out1, "train")

    check(transects_in(test_names) == {6}, f"run1: test is exactly transect 6, got {transects_in(test_names)}")
    check(transects_in(valid_names) == {5}, f"run1: valid is exactly transect 5, got {transects_in(valid_names)}")
    check(transects_in(train_names) <= {1, 2, 3, 4}, f"run1: train only has transects 1-4, got {transects_in(train_names)}")

    # No transect straddles splits.
    check(test_names.isdisjoint(valid_names), "run1: test/valid disjoint")
    check(test_names.isdisjoint(train_names), "run1: test/train disjoint")
    check(valid_names.isdisjoint(train_names), "run1: valid/train disjoint")

    with open(os.path.join(out1, "data.yaml")) as f:
        import yaml
        data_yaml1 = yaml.safe_load(f)
    check(data_yaml1.get("test") == "test/images", "run1: data.yaml has test: test/images")
    check(data_yaml1.get("train") == "train/images", "run1: data.yaml train key")
    check(data_yaml1.get("val") == "valid/images", "run1: data.yaml val key")
    check(os.path.isabs(data_yaml1.get("path", "")), "run1: data.yaml path is absolute")
    check(data_yaml1.get("path") == os.path.abspath(out1),
          f"run1: data.yaml path is the final out_dir, not a tmp build dir (got {data_yaml1.get('path')})")
except Exception as e:
    _fail += 1
    print("FAIL: run1 raised", repr(e))

# ---------------------------------------------------------------------------
# Run 2: add 2 new T6 frames + 2 new T2 frames to site AAA, re-run
# ---------------------------------------------------------------------------
try:
    prior_test = split_basenames(out1, "test")
    prior_valid = split_basenames(out1, "valid")

    new_frames = [
        ("TCRMP20140101_clip_AAA_T690", [0]),
        ("TCRMP20140101_clip_AAA_T691", [0]),
        ("TCRMP20140101_clip_AAA_T290", [0]),
        ("TCRMP20140101_clip_AAA_T291", [0]),
    ]
    img_dir = os.path.join(src1, "all_images")
    lbl_dir = os.path.join(src1, "all_labels")
    for name, classes in new_frames:
        with open(os.path.join(img_dir, name + ".jpeg"), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write(_label_text(classes))

    rc2 = PS.main(["--src", src1, "--out", out1, "--min_samples", "1"])
    check(rc2 == 0 or rc2 is None, "run2: exits cleanly")

    test_names2 = split_basenames(out1, "test")
    valid_names2 = split_basenames(out1, "valid")
    train_names2 = split_basenames(out1, "train")

    check("TCRMP20140101_clip_AAA_T690.jpeg" in test_names2, "run2: new T6 frame joins test")
    check("TCRMP20140101_clip_AAA_T691.jpeg" in test_names2, "run2: new T6 frame joins test (2)")
    check("TCRMP20140101_clip_AAA_T290.jpeg" in train_names2, "run2: new T2 frame joins train")
    check("TCRMP20140101_clip_AAA_T291.jpeg" in train_names2, "run2: new T2 frame joins train (2)")

    check(prior_test <= test_names2, "run2: prior test membership unchanged (subset of new test)")
    check(prior_valid <= valid_names2, "run2: prior valid membership unchanged (subset of new valid)")
    # Only the 2 new T6 frames should have been added to test.
    check(test_names2 - prior_test == {"TCRMP20140101_clip_AAA_T690.jpeg", "TCRMP20140101_clip_AAA_T691.jpeg"},
          "run2: test grew by exactly the 2 new T6 frames")
    check(valid_names2 == prior_valid, "run2: valid membership exactly unchanged")
except Exception as e:
    _fail += 1
    print("FAIL: run2 raised", repr(e))
finally:
    shutil.rmtree(tmp1, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run 3: DEGRADE - no T6 frames at all
# ---------------------------------------------------------------------------
tmp3 = tempfile.mkdtemp(prefix="pinned_split_run3_")
try:
    src3 = os.path.join(tmp3, "export")
    out3 = os.path.join(tmp3, "dataset")
    frames3 = []
    for site in ("AAA", "BBB"):
        for transect in range(1, 6):  # no transect 6
            for idx in range(3):
                tnum = f"{transect}{idx:02d}"
                name = f"TCRMP20140101_clip_{site}_T{tnum}"
                classes = [0, 1] if idx % 2 == 0 else [0]
                frames3.append((name, classes))
    make_export(src3, frames3)

    rc3 = PS.main(["--src", src3, "--out", out3, "--min_samples", "1"])
    check(rc3 == 0 or rc3 is None, "run3: does NOT exit 2 on empty T6 (degrades instead)")

    manifest3 = read_manifest(out3)
    check(manifest3["holdout_mode"] == "transect-degraded", "run3: holdout_mode=transect-degraded")
    check("degrade_warning" in manifest3, "run3: degrade_warning present")
    check("6" in manifest3.get("degrade_warning", "") or "T6" in manifest3.get("degrade_warning", ""),
          "run3: degrade_warning names T6")

    test_names3 = split_basenames(out3, "test")
    train_names3 = split_basenames(out3, "train")
    check(len(test_names3) > 0, "run3: test is non-empty (random holdout drawn)")
    check(len(train_names3) > 0, "run3: train is non-empty")
except Exception as e:
    _fail += 1
    print("FAIL: run3 raised", repr(e))


# ---------------------------------------------------------------------------
# Run 3c: empty TRAIN hard-fails with exit 2. Forced deterministically via a
# manifest that pins every frame to valid/test (holdout_mode=adopted-random,
# so replay never sends anything to train).
# ---------------------------------------------------------------------------
tmp3c = tempfile.mkdtemp(prefix="pinned_split_run3c_")
try:
    src3c = os.path.join(tmp3c, "export")
    out3c = os.path.join(tmp3c, "dataset")
    frames3c = [(f"TCRMP20140101_clip_AAA_T1{i:02d}", [0]) for i in range(4)]
    make_export(src3c, frames3c)
    os.makedirs(out3c, exist_ok=True)
    pins3c = {name + ".jpeg": ("valid" if i % 2 == 0 else "test")
              for i, (name, _) in enumerate(frames3c)}
    manifest3c = PS.build_manifest("adopted-random", {5}, {6}, pins3c,
                                   seed="1", valid_frac=0.2, test_frac=0.1)
    with open(os.path.join(out3c, "split_manifest.json"), "w") as f:
        json.dump(manifest3c, f)
    try:
        PS.main(["--src", src3c, "--out", out3c, "--min_samples", "1"])
        check(False, "run3c: expected SystemExit(2) on empty train")
    except SystemExit as se:
        check(se.code == 2, f"run3c: exits with code 2 on empty train, got {se.code}")
except Exception as e:
    _fail += 1
    print("FAIL: run3c raised", repr(e))
finally:
    shutil.rmtree(tmp3c, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run 4: AUTO-UPGRADE - take the degraded export from run3, add T6 frames
# ---------------------------------------------------------------------------
try:
    new_t6 = [
        ("TCRMP20140101_clip_AAA_T600", [0]),
        ("TCRMP20140101_clip_AAA_T601", [0]),
        ("TCRMP20140101_clip_BBB_T600", [0]),
    ]
    img_dir = os.path.join(src3, "all_images")
    lbl_dir = os.path.join(src3, "all_labels")
    for name, classes in new_t6:
        with open(os.path.join(img_dir, name + ".jpeg"), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write(_label_text(classes))

    rc4 = PS.main(["--src", src3, "--out", out3, "--min_samples", "1"])
    check(rc4 == 0 or rc4 is None, "run4: exits cleanly")

    manifest4 = read_manifest(out3)
    check(manifest4["holdout_mode"] == "transect", "run4: manifest flips to holdout_mode=transect")

    test_names4 = split_basenames(out3, "test")
    check(transects_in(test_names4) == {6}, f"run4: test == all-T6 after upgrade, got transects {transects_in(test_names4)}")
    for name, _ in new_t6:
        check(name + ".jpeg" in test_names4, f"run4: {name} is in test")
except Exception as e:
    _fail += 1
    print("FAIL: run4 raised", repr(e))
finally:
    shutil.rmtree(tmp3, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run 4b: AUTO-UPGRADE must require BOTH val and test holdouts labeled.
# T6 (test) has frames from the start; T5 (valid) starts empty. A fresh
# split must degrade (never a clean 'transect' with an empty valid split),
# and a re-run while T5 is still unlabeled must NOT auto-upgrade. Only once
# T5 frames are added does it upgrade, with both val and test non-empty.
# ---------------------------------------------------------------------------
tmp4b = tempfile.mkdtemp(prefix="pinned_split_run4b_")
try:
    src4b = os.path.join(tmp4b, "export")
    out4b = os.path.join(tmp4b, "dataset")
    frames4b = []
    for site in ("AAA", "BBB"):
        # transects 1-4 and 6 only; no transect 5 (valid) at all.
        for transect in (1, 2, 3, 4, 6):
            for idx in range(3):
                tnum = f"{transect}{idx:02d}"
                name = f"TCRMP20140101_clip_{site}_T{tnum}"
                classes = [0, 1] if idx % 2 == 0 else [0]
                frames4b.append((name, classes))
    make_export(src4b, frames4b)

    rc4b = PS.main(["--src", src4b, "--out", out4b, "--min_samples", "1"])
    check(rc4b == 0 or rc4b is None, "run4b: fresh split with T6 present but T5 absent exits cleanly")

    manifest4b = read_manifest(out4b)
    check(manifest4b["holdout_mode"] == "transect-degraded",
          f"run4b: T6-only fresh split degrades (holdout_mode={manifest4b.get('holdout_mode')}), never a clean 'transect' with empty valid")
    check("degrade_warning" in manifest4b, "run4b: degrade_warning present")

    valid_names4b = split_basenames(out4b, "valid")
    test_names4b = split_basenames(out4b, "test")
    check(len(valid_names4b) > 0, "run4b: degrade path still produces a non-empty valid split")
    check(len(test_names4b) > 0, "run4b: degrade path still produces a non-empty test split")

    # Re-run with the SAME source (T5 still unlabeled): must stay degraded,
    # never auto-upgrade to 'transect' just because T6 has frames.
    rc4b2 = PS.main(["--src", src4b, "--out", out4b, "--min_samples", "1"])
    check(rc4b2 == 0 or rc4b2 is None, "run4b: re-run (T5 still unlabeled) exits cleanly")
    manifest4b2 = read_manifest(out4b)
    check(manifest4b2["holdout_mode"] == "transect-degraded",
          f"run4b: re-run with T5 still unlabeled stays degraded (got {manifest4b2.get('holdout_mode')}), "
          f"does not auto-upgrade on T6 presence alone")

    # Now add T5 (valid) frames and re-run: should upgrade to a clean
    # 'transect' split with BOTH val and test non-empty.
    img_dir4b = os.path.join(src4b, "all_images")
    lbl_dir4b = os.path.join(src4b, "all_labels")
    new_t5 = [
        ("TCRMP20140101_clip_AAA_T500", [0]),
        ("TCRMP20140101_clip_AAA_T501", [0]),
        ("TCRMP20140101_clip_BBB_T500", [0]),
    ]
    for name, classes in new_t5:
        with open(os.path.join(img_dir4b, name + ".jpeg"), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(lbl_dir4b, name + ".txt"), "w") as f:
            f.write(_label_text(classes))

    rc4b3 = PS.main(["--src", src4b, "--out", out4b, "--min_samples", "1"])
    check(rc4b3 == 0 or rc4b3 is None, "run4b: upgrade run (T5 now labeled) exits cleanly")
    manifest4b3 = read_manifest(out4b)
    check(manifest4b3["holdout_mode"] == "transect",
          f"run4b: upgrades to holdout_mode=transect once T5 is labeled (got {manifest4b3.get('holdout_mode')})")
    check("degrade_warning" not in manifest4b3, "run4b: clean upgrade carries no degrade_warning")

    valid_names4b3 = split_basenames(out4b, "valid")
    test_names4b3 = split_basenames(out4b, "test")
    check(len(valid_names4b3) > 0, "run4b: upgraded split has non-empty valid")
    check(len(test_names4b3) > 0, "run4b: upgraded split has non-empty test")
    check(transects_in(valid_names4b3) == {5}, f"run4b: upgraded valid is exactly transect 5, got {transects_in(valid_names4b3)}")
    check(transects_in(test_names4b3) == {6}, f"run4b: upgraded test is exactly transect 6, got {transects_in(test_names4b3)}")
except Exception as e:
    _fail += 1
    print("FAIL: run4b raised", repr(e))
finally:
    shutil.rmtree(tmp4b, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run 4c: tiny export in the degrade path must still produce a non-empty
# test split (target_n floored to >= 1 instead of rounding down to 0).
# 5 frames, no transect info (all --test 0.1 -> round(0.5) == 0 pre-fix).
# ---------------------------------------------------------------------------
tmp4c = tempfile.mkdtemp(prefix="pinned_split_run4c_")
try:
    src4c = os.path.join(tmp4c, "export")
    out4c = os.path.join(tmp4c, "dataset")
    # No transect suffixes at all -> parse_transect() is None for every
    # frame -> Rule 3b degrade path (assign_degrade_random) on a fresh run.
    frames4c = [(f"TCRMP20140101_clip_AAA_frame{i}", [0]) for i in range(5)]
    make_export(src4c, frames4c)

    rc4c = PS.main(["--src", src4c, "--out", out4c, "--min_samples", "1", "--test", "0.1", "--valid", "0.1"])
    check(rc4c == 0 or rc4c is None, "run4c: tiny degrade export exits cleanly")

    manifest4c = read_manifest(out4c)
    check(manifest4c["holdout_mode"] == "transect-degraded", "run4c: tiny export with no transect info degrades")

    test_names4c = split_basenames(out4c, "test")
    valid_names4c = split_basenames(out4c, "valid")
    check(len(test_names4c) >= 1, f"run4c: tiny degrade test split is non-empty (target_n floored to >=1), got {len(test_names4c)}")
    check(len(valid_names4c) >= 1, f"run4c: tiny degrade valid split is non-empty (target_n floored to >=1), got {len(valid_names4c)}")
except Exception as e:
    _fail += 1
    print("FAIL: run4c raised", repr(e))
finally:
    shutil.rmtree(tmp4c, ignore_errors=True)


# ---------------------------------------------------------------------------
# Run 5: legacy adoption - delete manifest, keep hand-built random split
# ---------------------------------------------------------------------------
tmp5 = tempfile.mkdtemp(prefix="pinned_split_run5_")
try:
    src5 = os.path.join(tmp5, "export")
    out5 = os.path.join(tmp5, "dataset")
    frames5 = []
    for site in ("AAA", "BBB"):
        for transect in range(1, 7):
            for idx in range(3):
                tnum = f"{transect}{idx:02d}"
                name = f"TCRMP20140101_clip_{site}_T{tnum}"
                classes = [0, 1] if idx % 2 == 0 else [0]
                frames5.append((name, classes))
    make_export(src5, frames5)

    # Hand-build a random split: pick a mixed-transect valid/test set that is
    # NOT a clean by-transect split, to prove adoption preserves it exactly.
    import random as _random
    rng = _random.Random(42)
    shuffled = [n for n, _ in frames5]
    rng.shuffle(shuffled)
    hand_test = set(shuffled[:5])
    hand_valid = set(shuffled[5:10])
    hand_train = set(shuffled[10:])

    for split, names in (("train", hand_train), ("valid", hand_valid), ("test", hand_test)):
        img_out = os.path.join(out5, split, "images")
        lbl_out = os.path.join(out5, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for name in names:
            shutil.copy(os.path.join(src5, "all_images", name + ".jpeg"), os.path.join(img_out, name + ".jpeg"))
            shutil.copy(os.path.join(src5, "all_labels", name + ".txt"), os.path.join(lbl_out, name + ".txt"))
    # No split_manifest.json written -> triggers legacy adoption.

    hand_test_basenames = {n + ".jpeg" for n in hand_test}
    hand_valid_basenames = {n + ".jpeg" for n in hand_valid}

    rc5 = PS.main(["--src", src5, "--out", out5, "--min_samples", "1"])
    check(rc5 == 0 or rc5 is None, "run5: exits cleanly")

    manifest5 = read_manifest(out5)
    check(manifest5["holdout_mode"] == "adopted-random", "run5: holdout_mode=adopted-random")

    test_names5 = split_basenames(out5, "test")
    valid_names5 = split_basenames(out5, "valid")
    check(test_names5 == hand_test_basenames, "run5: adopted test set preserved byte-for-byte")
    check(valid_names5 == hand_valid_basenames, "run5: adopted valid set preserved byte-for-byte")

    # New frames (not in the hand-built dirs originally) go to train: verify
    # every non-adopted frame ended up in train.
    train_names5 = split_basenames(out5, "train")
    all_basenames = {n + ".jpeg" for n, _ in frames5}
    expected_train = all_basenames - hand_test_basenames - hand_valid_basenames
    check(train_names5 == expected_train, "run5: all non-adopted frames go to train")
except Exception as e:
    _fail += 1
    print("FAIL: run5 raised", repr(e))
finally:
    shutil.rmtree(tmp5, ignore_errors=True)


# ---------------------------------------------------------------------------
# Determinism: same inputs -> byte-identical dataset dirs
# ---------------------------------------------------------------------------
tmp6 = tempfile.mkdtemp(prefix="pinned_split_run6_")
try:
    src6 = os.path.join(tmp6, "export")
    out6a = os.path.join(tmp6, "dataset_a")
    out6b = os.path.join(tmp6, "dataset_b")
    build_full_fixture(src6)
    PS.main(["--src", src6, "--out", out6a, "--min_samples", "1"])
    PS.main(["--src", src6, "--out", out6b, "--min_samples", "1"])

    for split in ("train", "valid", "test"):
        a = split_basenames(out6a, split)
        b = split_basenames(out6b, split)
        check(a == b, f"determinism: {split} split identical across independent fresh runs")
except Exception as e:
    _fail += 1
    print("FAIL: determinism check raised", repr(e))
finally:
    shutil.rmtree(tmp6, ignore_errors=True)


# ---------------------------------------------------------------------------
# Split record (2026-09-04): every manifest records the seed, the fractions,
# the holdout mode and a split fingerprint computed with the same function
# the detection splitter (detect_dataset.py) uses, so a reader can tell two
# runs scored on the same held-out images from two runs that were not.
# Named cases: each builds its own fixture under a fresh temp root, and the
# runner at the end prints PASS or FAIL per case.
# ---------------------------------------------------------------------------
import contextlib
import io
import pathlib

import detect_dataset as DD  # sibling of pinned_split.py in src/, same sys.path entry

RECORD_KEYS = ("seed", "valid_frac", "test_frac", "split_fingerprint",
               "split_fingerprint_scheme", "counts")
# The keys pinned_split.py wrote before 2026-09-04. Readers in evaluate_run.py
# (_split_pinned) and seed_from_predictions.py depend on them, so every one
# must survive unchanged.
LEGACY_KEYS = ("schema_version", "created_at", "updated_at", "holdout_mode",
               "transect_rule", "holdout_transects", "pinned")
LEGACY_CREATED_AT = "2026-07-09T10:00:00-04:00"
DEFAULT_ARGS = ["--min_samples", "1"]


def assignment_on_disk(out_dir):
    """{basename: split} from the images actually placed under out_dir."""
    pins = {}
    for split in ("train", "valid", "test"):
        for name in split_basenames(out_dir, split):
            pins[name] = split
    return pins


def run_split(argv):
    """Run pinned_split.main(argv) with stdout captured. Returns
    (exit_code, stdout): 0 for a clean return, else the SystemExit code."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = PS.main(argv)
        except SystemExit as se:
            return se.code, buf.getvalue()
    return (rc or 0), buf.getvalue()


def write_manifest(out_dir, doc):
    """Write ``doc`` as <out_dir>/split_manifest.json and return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "split_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return path


def legacy_manifest(pins, holdout_mode="transect"):
    """The version 1 manifest shape pinned_split.py wrote until 2026-09-04."""
    return {"schema_version": 1,
            "created_at": LEGACY_CREATED_AT,
            "updated_at": LEGACY_CREATED_AT,
            "holdout_mode": holdout_mode,
            "transect_rule": "stem _T + 3 digits, first digit = transect",
            "holdout_transects": {"val": [5], "test": [6]},
            "pinned": dict(sorted(pins.items()))}


def make_detect_source(root, basenames):
    """A staged detection dataset (images/ + labels/ + classes.txt) holding
    exactly ``basenames``: the shape detect_dataset.build_detect_dataset reads."""
    ds = os.path.join(root, "detect_source")
    os.makedirs(os.path.join(ds, "images"), exist_ok=True)
    os.makedirs(os.path.join(ds, "labels"), exist_ok=True)
    for name in basenames:
        with open(os.path.join(ds, "images", name), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(ds, "labels", os.path.splitext(name)[0] + ".txt"), "w") as f:
            f.write("0 0.5 0.5 0.2 0.3\n")
    with open(os.path.join(ds, "classes.txt"), "w") as f:
        f.write("SS\nMCAV\n")
    return ds


def add_frames(src, frames):
    """Append (stem, [class_ids]) frames to an export built by make_export."""
    for name, classes in frames:
        with open(os.path.join(src, "all_images", name + ".jpeg"), "wb") as f:
            f.write(IMG_BYTES)
        with open(os.path.join(src, "all_labels", name + ".txt"), "w") as f:
            f.write(_label_text(classes))


def test_manifest_records_seed_fractions_mode_and_fingerprint(root):
    """A fresh split writes every earlier key plus seed, fractions, holdout mode,
    counts and the shared fingerprint, and the log line names the fingerprint."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    rc, log = run_split(["--src", src, "--out", out, "--valid", "0.25", "--test", "0.15",
                         "--rand", "7"] + DEFAULT_ARGS)
    check(rc == 0, f"a fresh transect split exits cleanly (rc={rc})")
    m = read_manifest(out)
    check(all(k in m for k in LEGACY_KEYS), f"every earlier key is still written (got {sorted(m)})")
    check(all(k in m for k in RECORD_KEYS), f"the split record keys are written (got {sorted(m)})")
    check(m.get("schema_version") == 2 and PS.SCHEMA_VERSION == 2,
          f"the manifest schema version is 2 (got {m.get('schema_version')})")
    check(m.get("seed") == "7", f"the seed is recorded as the string given (got {m.get('seed')!r})")
    check(m.get("valid_frac") == 0.25 and m.get("test_frac") == 0.15,
          f"the fractions the call carried are recorded (got {m.get('valid_frac')}, {m.get('test_frac')})")
    check(m.get("holdout_mode") == "transect", f"the holdout mode is recorded (got {m.get('holdout_mode')})")
    pins = m["pinned"]
    check(pins == assignment_on_disk(out), "the pinned table is exactly the assignment on disk")
    check(m.get("split_fingerprint") == DD.split_fingerprint(pins),
          f"the fingerprint is detect_dataset.split_fingerprint over the pinned table (got {m.get('split_fingerprint')})")
    check(m.get("split_fingerprint_scheme") == DD.SPLIT_FINGERPRINT_SCHEME,
          f"the scheme is the shared one (got {m.get('split_fingerprint_scheme')})")
    counts = {s: sum(1 for v in pins.values() if v == s) for s in ("train", "valid", "test")}
    check(m.get("counts") == counts, f"counts per split are recorded (got {m.get('counts')}, want {counts})")
    check(str(m.get("split_fingerprint")) in log, "the run log line names the fingerprint")


def test_fingerprint_matches_detect_dataset_for_the_same_assignment(root):
    """The same basename-to-split table written through both splitters gives one
    fingerprint, and the shared reader calls the two splits the same."""
    src = os.path.join(root, "export")
    out_seg = os.path.join(root, "dataset_seg")
    build_full_fixture(src)
    rc, _ = run_split(["--src", src, "--out", out_seg] + DEFAULT_ARGS)
    check(rc == 0, f"the segmentation split exits cleanly (rc={rc})")
    seg = read_manifest(out_seg)
    pins = seg["pinned"]
    # The detection splitter replays every pin its manifest already holds, so
    # seeding its manifest with this assignment makes it write the very same
    # assignment through its own code path, whatever seed and fractions it gets.
    det_src = make_detect_source(root, sorted(pins))
    out_det = os.path.join(root, "dataset_det")
    write_manifest(out_det, {"pins": pins})
    det = DD.build_detect_dataset(det_src, out_det, "parity-seed", 0.3, 0.3)
    check(det["pins"] == pins and det["n_replayed"] == len(pins),
          f"the detection splitter replayed every pin (replayed {det['n_replayed']} of {len(pins)})")
    check(det["split_fingerprint"] == seg.get("split_fingerprint"),
          f"both splitters write the same fingerprint for the same assignment "
          f"(detect {det['split_fingerprint']}, pinned {seg.get('split_fingerprint')})")
    check(det["split_fingerprint_scheme"] == seg.get("split_fingerprint_scheme"),
          "both splitters write the same scheme")
    check(seg.get("split_fingerprint") == DD.split_fingerprint(pins),
          "and it equals the function applied directly to the table")
    rec_seg = DD.read_split_record(os.path.join(out_seg, "split_manifest.json"))
    rec_det = DD.read_split_record(os.path.join(out_det, "split_manifest.json"))
    check(rec_seg.same_split_as(rec_det) and rec_det.same_split_as(rec_seg),
          "the shared reader calls the two splits the same in both directions")
    moved = dict(pins)
    first_train = next(n for n, s in sorted(pins.items()) if s == "train")
    moved[first_train] = "valid"
    check(DD.split_fingerprint(moved) != seg.get("split_fingerprint"),
          "moving one image to another split changes the fingerprint")


def test_existing_pins_are_kept_on_rebuild(root):
    """A rebuild with every knob changed moves no image; new frames follow the
    transect rule and only then does the fingerprint change."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    first = read_manifest(out)
    # Every knob changed, no new image: nothing moves, the fingerprint holds.
    rc, _ = run_split(["--src", src, "--out", out, "--valid", "0.4", "--test", "0.3",
                       "--rand", "99", "--holdout-mode", "pinned-random"] + DEFAULT_ARGS)
    second = read_manifest(out)
    check(rc == 0, f"a rebuild with different knobs exits cleanly (rc={rc})")
    check(second["pinned"] == first["pinned"], "no image changed split on the rebuild")
    check(second.get("split_fingerprint") == first.get("split_fingerprint"),
          "the fingerprint is unchanged when no image moved")
    check(second.get("seed") == "99" and second.get("valid_frac") == 0.4 and second.get("test_frac") == 0.3,
          f"the record says what the rebuild was given (got seed {second.get('seed')!r}, "
          f"{second.get('valid_frac')}, {second.get('test_frac')})")
    check(second.get("holdout_mode") == "transect",
          f"the holdout mode comes from the manifest, not the flag (got {second.get('holdout_mode')})")
    check(second.get("created_at") == first.get("created_at"), "created_at survives the rebuild")
    # Two new frames join: the prior pins stay, the new ones follow the transect rule.
    add_frames(src, [("TCRMP20140101_clip_AAA_T690", [0]), ("TCRMP20140101_clip_AAA_T290", [0])])
    rc, _ = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    third = read_manifest(out)
    check(rc == 0, f"a rebuild with new frames exits cleanly (rc={rc})")
    check(all(third["pinned"].get(n) == s for n, s in first["pinned"].items()),
          "every prior pin is kept when new frames arrive")
    check(third["pinned"].get("TCRMP20140101_clip_AAA_T690.jpeg") == "test"
          and third["pinned"].get("TCRMP20140101_clip_AAA_T290.jpeg") == "train",
          "the new frames are assigned by transect")
    check(third.get("split_fingerprint") != first.get("split_fingerprint")
          and third.get("split_fingerprint") == DD.split_fingerprint(third["pinned"]),
          "the fingerprint changed because the image set changed, and matches the new table")
    check(third["pinned"] == assignment_on_disk(out), "the table still matches the files on disk")


def test_read_split_record_reads_a_pinned_split_manifest(root):
    """detect_dataset.read_split_record reads this tool's manifest unchanged:
    recorded, with seed, fractions, mode, version, fingerprint and pin count."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    m = read_manifest(out)
    rec = DD.read_split_record(pathlib.Path(out) / "split_manifest.json")
    check(rec.recorded and rec.missing == (), f"the shared reader reports the split as recorded (missing={rec.missing})")
    check(rec.seed == "1" and rec.valid_frac == 0.2 and rec.test_frac == 0.1,
          f"the defaults read back as seed '1', valid 0.2, test 0.1 (got {rec.seed!r}, {rec.valid_frac}, {rec.test_frac})")
    check(rec.holdout_mode == "transect" and rec.schema_version == 2,
          f"holdout mode and schema version read back (got {rec.holdout_mode}, {rec.schema_version})")
    check(rec.split_fingerprint == m.get("split_fingerprint")
          and rec.split_fingerprint_scheme == DD.SPLIT_FINGERPRINT_SCHEME,
          "the fingerprint and scheme read back unchanged")
    check(rec.n_pins == len(m["pinned"]) == 36, f"the pin count comes from the pinned table (got {rec.n_pins})")
    check(rec.same_split_as(rec), "a recorded manifest is the same split as itself")
    out2 = os.path.join(root, "dataset_again")
    run_split(["--src", src, "--out", out2] + DEFAULT_ARGS)
    rec2 = DD.read_split_record(os.path.join(out2, "split_manifest.json"))
    check(rec.same_split_as(rec2), "an independent fresh split over the same export is the same split")


def test_old_manifest_without_fingerprint_reads_as_not_recorded(root):
    """A version 1 manifest (no record keys) reads as not recorded and is never
    the same split as anything, itself included."""
    pins = {"TCRMP20140101_clip_AAA_T100.jpeg": "train",
            "TCRMP20140101_clip_AAA_T500.jpeg": "valid",
            "TCRMP20140101_clip_AAA_T600.jpeg": "test"}
    path = write_manifest(os.path.join(root, "old"), legacy_manifest(pins))
    rec = DD.read_split_record(path)
    check(rec.recorded is False and rec.missing == DD.SPLIT_RECORD_KEYS,
          f"a version 1 manifest reads as not recorded with every record key missing (got {rec.missing})")
    check(rec.seed is None and rec.valid_frac is None and rec.test_frac is None
          and rec.split_fingerprint is None and rec.split_fingerprint_scheme is None,
          "every absent value is None")
    check(rec.holdout_mode == "transect" and rec.schema_version == 1 and rec.n_pins == 3,
          f"the holdout mode, version and pin count still read (got {rec.holdout_mode}, {rec.schema_version}, {rec.n_pins})")
    check(rec.same_split_as(rec) is False, "an unrecorded manifest is never the same split as anything")


def test_adversarial_manifest_with_pins_and_no_fingerprint_is_replayed(root):
    """The shape on disk today, a pinned table with no record, is replayed pin
    for pin and rewritten as a recorded version 2 manifest."""
    # The shape on disk today: a version 1 manifest with a pinned table and
    # none of the record keys. A rebuild must keep every pin and add the record.
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    m1 = read_manifest(out)
    write_manifest(out, legacy_manifest(m1["pinned"]))
    add_frames(src, [("TCRMP20140101_clip_BBB_T691", [0])])
    rc, _ = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 0, f"a rebuild over a version 1 manifest exits cleanly (rc={rc})")
    m2 = read_manifest(out)
    check(all(m2["pinned"].get(n) == s for n, s in m1["pinned"].items()), "every version 1 pin is kept")
    check(m2["pinned"].get("TCRMP20140101_clip_BBB_T691.jpeg") == "test", "the new frame is assigned by transect")
    check(m2.get("schema_version") == 2 and m2.get("created_at") == LEGACY_CREATED_AT,
          f"the manifest moves to version 2 and keeps its created_at (got {m2.get('schema_version')}, {m2.get('created_at')})")
    rec = DD.read_split_record(os.path.join(out, "split_manifest.json"))
    check(rec.recorded and rec.split_fingerprint == DD.split_fingerprint(m2["pinned"]),
          "the rebuilt manifest is recorded with a fingerprint over the kept pins")


def test_adversarial_corrupt_manifest_json(root):
    """Corrupt JSON, a non-object, a pin outside train/valid/test and a
    list-shaped pin table each exit 2 naming the file, with nothing written."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    os.makedirs(out)
    path = os.path.join(out, "split_manifest.json")
    with open(path, "w") as f:
        f.write("{not json")
    rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 2 and path in log and "JSON" in log,
          f"corrupt JSON exits 2 and names the manifest (rc={rc}, log={log.strip()[-160:]!r})")
    check(not os.path.isdir(os.path.join(out, "train")), "nothing is written over a corrupt manifest")
    with open(path, "w") as f:
        f.write("[]")
    rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 2 and path in log, f"a manifest that is not a JSON object exits 2 (rc={rc}, log={log.strip()[-160:]!r})")
    write_manifest(out, legacy_manifest({"TCRMP20140101_clip_AAA_T100.jpeg": "val"}))
    rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 2 and "T100" in log and "'val'" in log,
          f"a pin outside train, valid, test is refused by name rather than re-split (rc={rc}, log={log.strip()[-160:]!r})")
    write_manifest(out, {"schema_version": 1, "holdout_mode": "transect", "pinned": ["a.jpeg"]})
    rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 2 and path in log, f"a pinned table that is not an object exits 2 (rc={rc}, log={log.strip()[-160:]!r})")
    check(not os.path.isdir(os.path.join(out, "train")), "still nothing written")


def test_adversarial_single_transect_source(root):
    """A source on one transect degrades to a seeded random holdout that is
    recorded, reproducible from its seed, and different under another seed."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    frames = [(f"TCRMP20140101_clip_AAA_T1{i:02d}", [0, 1] if i % 2 == 0 else [0]) for i in range(40)]
    make_export(src, frames)
    rc, _ = run_split(["--src", src, "--out", out, "--rand", "3"] + DEFAULT_ARGS)
    check(rc == 0, f"a single-transect source exits cleanly (rc={rc})")
    m = read_manifest(out)
    check(m.get("holdout_mode") == "transect-degraded" and "degrade_warning" in m,
          f"one transect degrades to a random holdout (got {m.get('holdout_mode')})")
    counts = m.get("counts") or {}
    check(counts.get("train", 0) >= 1 and counts.get("valid", 0) >= 1 and counts.get("test", 0) >= 1,
          f"all three splits are non-empty (got {counts})")
    check(m.get("seed") == "3" and m.get("valid_frac") == 0.2 and m.get("test_frac") == 0.1,
          "the seed and fractions that drew the random holdout are recorded")
    rec = DD.read_split_record(os.path.join(out, "split_manifest.json"))
    check(rec.recorded and rec.split_fingerprint == DD.split_fingerprint(assignment_on_disk(out)),
          "the degraded split is recorded and its fingerprint matches the files on disk")
    out2 = os.path.join(root, "dataset_same_seed")
    run_split(["--src", src, "--out", out2, "--rand", "3"] + DEFAULT_ARGS)
    check(read_manifest(out2).get("split_fingerprint") == m.get("split_fingerprint"),
          "the recorded seed reproduces the same random holdout in a fresh folder")
    out3 = os.path.join(root, "dataset_other_seed")
    run_split(["--src", src, "--out", out3, "--rand", "4"] + DEFAULT_ARGS)
    check(read_manifest(out3).get("split_fingerprint") != m.get("split_fingerprint"),
          "a different seed draws a different holdout and a different fingerprint")


def test_adversarial_non_integer_transect_id(root):
    """A non-integer transect or class id exits 2 naming the flag and the value,
    with nothing written."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    for flag, value in (("--val-transects", "five"), ("--test-transects", "6,x"),
                        ("--include-classes", "a,1")):
        rc, log = run_split(["--src", src, "--out", out, flag, value] + DEFAULT_ARGS)
        check(rc == 2 and flag in log and value in log,
              f"{flag} {value} is refused with exit 2 naming the flag and value (rc={rc}, log={log.strip()[-160:]!r})")
    check(not os.path.exists(out), "nothing is written for a refused transect list")


def test_adversarial_basename_with_spaces_and_unicode(root):
    """Basenames with spaces and non-ASCII characters are pinned, placed, hashed
    and read back exactly, and a rebuild keeps the fingerprint."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    frames = [("reef ñ 東 frame 1", [0]), ("reef ñ 東 frame 2", [0]),
              ("clip ü_T500", [0]), ("clip ü_T600", [0]), ("clip ü_T100", [0])]
    make_export(src, frames)
    rc, _ = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 0, f"names with spaces and unicode exit cleanly (rc={rc})")
    expected = {"reef ñ 東 frame 1.jpeg": "train", "reef ñ 東 frame 2.jpeg": "train",
                "clip ü_T500.jpeg": "valid", "clip ü_T600.jpeg": "test", "clip ü_T100.jpeg": "train"}
    m = read_manifest(out)
    check(m["pinned"] == expected, f"the pinned table holds the exact names (got {m['pinned']})")
    check(assignment_on_disk(out) == expected, "the files are placed under their exact names")
    check(m.get("split_fingerprint") == DD.split_fingerprint(expected),
          "the fingerprint over unicode names matches the shared function")
    rec = DD.read_split_record(os.path.join(out, "split_manifest.json"))
    check(rec.recorded and rec.n_pins == 5, f"the reader reads the unicode manifest (got {rec})")
    rc, _ = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 0 and read_manifest(out).get("split_fingerprint") == m.get("split_fingerprint"),
          "a rebuild keeps the same fingerprint")


def test_adversarial_read_only_destination(root):
    """A destination that cannot be created raises OSError naming the folder,
    with nothing written."""
    if os.geteuid() == 0:
        print("  SKIP read-only destination checks (running as root)")
        return
    src = os.path.join(root, "export")
    build_full_fixture(src)
    parent = os.path.join(root, "ro_parent")
    os.makedirs(parent)
    os.chmod(parent, 0o500)
    out = os.path.join(parent, "dataset")
    try:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                PS.main(["--src", src, "--out", out] + DEFAULT_ARGS)
            check(False, "a read-only destination is refused")
        except OSError as e:
            check(out in str(e), f"a read-only destination raises OSError naming the folder ({e})")
        check(not os.path.exists(out), "and nothing is written")
    finally:
        os.chmod(parent, 0o700)


def test_adversarial_fraction_validation(root):
    """A fraction outside [0, 1), a pair summing to 1 or more, or NaN exits 2;
    zero is accepted; the validator agrees with detect_dataset's over a grid."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    for extra in (["--valid", "1.5"], ["--test", "-0.1"], ["--valid", "0.6", "--test", "0.5"],
                  ["--valid", "nan"]):
        rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS + extra)
        check(rc == 2 and ("valid" in log or "test" in log),
              f"{' '.join(extra)} is refused with exit 2 (rc={rc}, log={log.strip()[-160:]!r})")
    check(not os.path.exists(out), "nothing is written for a refused fraction")
    rc, _ = run_split(["--src", src, "--out", out, "--valid", "0", "--test", "0"] + DEFAULT_ARGS)
    check(rc == 0 and read_manifest(out).get("valid_frac") == 0.0 and read_manifest(out).get("test_frac") == 0.0,
          "a zero fraction is accepted and recorded as 0.0")
    # validate_fractions is a keep-in-sync copy of detect_dataset._validate_fractions
    # (private there). Both must give the same verdict for every pair, or the two
    # splitters accept different fractions for the same request.
    grid = (0.0, 0.1, 0.2, 0.5, 0.9, 1.0, -0.1, 1.5, float("nan"), True)
    disagreements = []
    for v in grid:
        for t in grid:
            verdicts = []
            for validator in (PS.validate_fractions, DD._validate_fractions):
                try:
                    validator(v, t)
                    verdicts.append("accepted")
                except ValueError:
                    verdicts.append("refused")
            if verdicts[0] != verdicts[1]:
                disagreements.append((v, t, verdicts))
    check(disagreements == [],
          f"validate_fractions and detect_dataset._validate_fractions agree over {len(grid) ** 2} pairs "
          f"(disagreements: {disagreements})")


def test_adversarial_bad_holdout_mode_in_manifest(root):
    """A manifest whose holdout_mode is not a string, or names a policy this
    tool never writes, exits 2 naming the file and the value, with nothing
    written; a manifest with no holdout_mode still replays as transect."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    path = os.path.join(out, "split_manifest.json")
    for mode in (5, "lab-random", ["transect"]):
        doc = legacy_manifest({"TCRMP20140101_clip_AAA_T100.jpeg": "train"})
        doc["holdout_mode"] = mode
        write_manifest(out, doc)
        rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
        check(rc == 2 and path in log and "holdout_mode" in log and repr(mode) in log,
              f"holdout_mode {mode!r} is refused with exit 2 naming the file and value "
              f"(rc={rc}, log={log.strip()[-200:]!r})")
    check(not os.path.isdir(os.path.join(out, "train")), "nothing is written over a bad holdout_mode")
    write_manifest(out, {"schema_version": 1, "pinned": {"TCRMP20140101_clip_AAA_T100.jpeg": "valid"}})
    rc, _ = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    m = read_manifest(out) if rc == 0 else {}
    check(rc == 0 and m.get("holdout_mode") == "transect"
          and m.get("pinned", {}).get("TCRMP20140101_clip_AAA_T100.jpeg") == "valid",
          f"a manifest with no holdout_mode replays as transect and keeps its pin (rc={rc})")


def test_adversarial_overlapping_transect_lists(root):
    """A transect named in both --val-transects and --test-transects exits 2
    naming the transect, instead of sending it to test, leaving valid empty
    and degrading the split under a misleading warning."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    for val, test, named in (("5", "5", "[5]"), ("4,5", "5,6", "[5]"), ("3,4", "4,3", "[3, 4]")):
        rc, log = run_split(["--src", src, "--out", out, "--val-transects", val,
                             "--test-transects", test] + DEFAULT_ARGS)
        check(rc == 2 and "--val-transects" in log and "--test-transects" in log and named in log,
              f"val {val} / test {test} is refused with exit 2 naming transect(s) {named} "
              f"(rc={rc}, log={log.strip()[-200:]!r})")
    check(not os.path.exists(out), "nothing is written for overlapping transect lists")


def test_label_line_with_non_integer_class_is_reported(root):
    """A label line whose first field is not a class id is skipped, and the
    skip names the file and line number: once when classes are counted and
    once when the label is copied, never once per lookup."""
    src = os.path.join(root, "export")
    out = os.path.join(root, "dataset")
    build_full_fixture(src)
    bad_label = os.path.join(src, "all_labels", "TCRMP20140101_clip_AAA_T100.txt")
    with open(bad_label, "a") as f:
        f.write("abc 0.1 0.1 0.2 0.2 0.3 0.3\n")
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    rc, log = run_split(["--src", src, "--out", out] + DEFAULT_ARGS)
    check(rc == 0, f"a bad label line does not stop the split (rc={rc})")
    notice = f"{bad_label} line 3"
    check(notice in log and "'abc'" in log,
          f"the skip names the label file, line 3 and the field (log={log.strip()[-300:]!r})")
    check(log.count(notice) == 2,
          f"the skip is reported once per read of the file, counted then copied (got {log.count(notice)})")
    copied = os.path.join(out, "train", "labels", "TCRMP20140101_clip_AAA_T100.txt")
    with open(copied) as f:
        firsts = [ln.split()[0] for ln in f if ln.strip()]
    check("abc" not in firsts and firsts.count("0") == 2,
          f"the copied label keeps the class 0 lines and drops the bad one (got {firsts})")


SPLIT_RECORD_CASES = (
    test_manifest_records_seed_fractions_mode_and_fingerprint,
    test_fingerprint_matches_detect_dataset_for_the_same_assignment,
    test_existing_pins_are_kept_on_rebuild,
    test_read_split_record_reads_a_pinned_split_manifest,
    test_old_manifest_without_fingerprint_reads_as_not_recorded,
    test_adversarial_manifest_with_pins_and_no_fingerprint_is_replayed,
    test_adversarial_corrupt_manifest_json,
    test_adversarial_single_transect_source,
    test_adversarial_non_integer_transect_id,
    test_adversarial_basename_with_spaces_and_unicode,
    test_adversarial_read_only_destination,
    test_adversarial_fraction_validation,
    test_adversarial_bad_holdout_mode_in_manifest,
    test_adversarial_overlapping_transect_lists,
    test_label_line_with_non_integer_class_is_reported,
)

print("\n-- split record cases --")
for case in SPLIT_RECORD_CASES:
    root = tempfile.mkdtemp(prefix="pinned_split_record_")
    failures_before = _fail
    try:
        case(root)
    except Exception as e:
        _fail += 1
        print(f"FAIL: {case.__name__} raised {e!r}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(("PASS " if _fail == failures_before else "FAIL ") + case.__name__)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
