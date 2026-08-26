"""Tests for pinned_split.py (frozen-holdout, by-transect split).
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_pinned_split.py"""
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
    with open(os.path.join(out_dir, "split_manifest.json")) as f:
        return json.load(f)


def split_basenames(out_dir, split):
    img_dir = os.path.join(out_dir, split, "images")
    if not os.path.isdir(img_dir):
        return set()
    return set(os.listdir(img_dir))


def transects_in(names):
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
    manifest3c = PS.build_manifest("adopted-random", {5}, {6}, pins3c)
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
    check(train_names5 == expected_train, "run5: all non-adopted frames land in train")
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


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
