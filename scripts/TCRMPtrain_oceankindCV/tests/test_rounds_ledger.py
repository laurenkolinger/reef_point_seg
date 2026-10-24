"""Tests for evaluate_run.py's append_round_row (rounds ledger, gates,
champion-baselined lineage, the image counts in the n_train, n_valid and
n_test columns, and the conf, iou and imgsz columns that say what a score was
measured with). No model, no GPU and no network: the only thing the counting
tests borrow from ultralytics is its list of readable image types, and where
ultralytics is not installed the detection stager's IMAGE_EXTS stands in.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_rounds_ledger.py"""
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

import csv
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import evaluate_run as ER

_fail = 0


def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def make_dataset_dir(root, holdout_mode='transect', classes=('SS', 'MCAV', 'PAST'),
                      trained_classes=('SS', 'MCAV', 'PAST'),
                      n_train=40, n_valid=10, n_test=8):
    """Build a minimal dataset dir: data.yaml with class names, train/valid/test
    images/ dirs with N placeholder files, and train/labels/ populated only for
    `trained_classes` (so PAST can be "never trained" when desired)."""
    os.makedirs(root, exist_ok=True)
    import yaml
    names = {i: c for i, c in enumerate(classes)}
    with open(os.path.join(root, 'data.yaml'), 'w') as f:
        yaml.dump({'path': root, 'train': 'train/images', 'val': 'valid/images',
                   'test': 'test/images', 'names': names}, f)

    for split, n in (('train', n_train), ('valid', n_valid), ('test', n_test)):
        img_dir = os.path.join(root, split, 'images')
        os.makedirs(img_dir, exist_ok=True)
        for i in range(n):
            with open(os.path.join(img_dir, f'{split}_{i}.jpeg'), 'wb') as f:
                f.write(b'\xff\xd8\xff\xd9')

    lbl_dir = os.path.join(root, 'train', 'labels')
    os.makedirs(lbl_dir, exist_ok=True)
    trained_ids = [cid for cid, nm in names.items() if nm in trained_classes]
    for i in range(n_train):
        cid = trained_ids[i % len(trained_ids)] if trained_ids else 0
        with open(os.path.join(lbl_dir, f'train_{i}.txt'), 'w') as f:
            f.write(f'{cid} 0.5 0.5 0.1 0.1 0.5 0.4 0.4 0.5\n')

    if holdout_mode is not None:
        with open(os.path.join(root, 'split_manifest.json'), 'w') as f:
            json.dump({'schema_version': 1, 'holdout_mode': holdout_mode,
                       'pinned': {}}, f)
    return root


def make_run_dir(root, run_name, base_model='yolov8n-seg.pt'):
    run_dir = os.path.join(root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'args.yaml'), 'w') as f:
        f.write(f'model: {base_model}\n')
    return run_dir


def fabricate_metrics(map50_95_m, recall_m, per_class_ap):
    """per_class_ap: dict of class_name -> mask_map50_95 (AP)."""
    per_class = {nm: {'mask_map50_95': ap, 'mask_map50': min(1.0, ap + 0.1),
                       'mask_precision': 0.7, 'mask_recall': recall_m}
                 for nm, ap in per_class_ap.items()}
    return {
        'overall': {
            'mask_map50': min(1.0, map50_95_m + 0.1),
            'mask_map50_95': map50_95_m,
            'mask_precision': 0.72,
            'mask_recall': recall_m,
            'box_map50': min(1.0, map50_95_m + 0.12),
            'box_map50_95': map50_95_m + 0.02,
        },
        'per_class': per_class,
    }


def read_rows(rounds_dir):
    path = os.path.join(rounds_dir, 'rounds.csv')
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Test 1: two rounds, round 2 regresses mask mAP by 0.02 and class PAST by 0.08
# ---------------------------------------------------------------------------
tmp1 = tempfile.mkdtemp(prefix='rounds_ledger_t1_')
try:
    rounds_dir = os.path.join(tmp1, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp1, 'dataset'), holdout_mode='transect')

    run1_dir = make_run_dir(tmp1, 'run1')
    m1 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.45})
    row1 = ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)

    run2_dir = make_run_dir(tmp1, 'run2')
    # round 2: mask mAP drops by 0.02 (0.500 -> 0.480); PAST class drops by 0.08 (0.45 -> 0.37)
    m2 = fabricate_metrics(map50_95_m=0.480, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.37})
    row2 = ER.append_round_row(rounds_dir, run2_dir, m2, dataset_dir)

    rows = read_rows(rounds_dir)
    check(len(rows) == 2, f"expected 2 rows after 2 distinct runs, got {len(rows)}")
    check(row1['gate_map'] == 'pass', f"round 1 (no baseline) gate_map should be pass, got {row1['gate_map']}")
    check(row2['gate_map'] == 'fail',
          f"round 2 mask mAP50-95 dropped 0.02 > eps 0.005, expected gate_map=fail, got {row2['gate_map']}")
    check('flag' in row2['gate_class'] and 'PA' in row2['gate_class'],
          f"round 2 class PAST dropped 0.08 > 0.05, expected gate_class flag containing PA code, got {row2['gate_class']}")
    check(row2['gate_recall'] == 'pass',
          f"round 2 recall unchanged, expected gate_recall=pass, got {row2['gate_recall']}")

    # per-class sidecars exist
    sidecar1 = os.path.join(rounds_dir, row1['per_class_json'])
    sidecar2 = os.path.join(rounds_dir, row2['per_class_json'])
    check(os.path.isfile(sidecar1), f"round 1 per-class sidecar missing: {sidecar1}")
    check(os.path.isfile(sidecar2), f"round 2 per-class sidecar missing: {sidecar2}")
    check(row1['per_class_json'] == 'rounds/round_1_per_class.json',
          f"round 1 sidecar path unexpected: {row1['per_class_json']}")
    check(row2['per_class_json'] == 'rounds/round_2_per_class.json',
          f"round 2 sidecar path unexpected: {row2['per_class_json']}")

    # Idempotency: re-evaluating run2 (same run_name) updates in place, no dup row.
    m2b = fabricate_metrics(map50_95_m=0.481, recall_m=0.80,
                             per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.37})
    row2b = ER.append_round_row(rounds_dir, run2_dir, m2b, dataset_dir)
    rows_after_reeval = read_rows(rounds_dir)
    check(len(rows_after_reeval) == 2,
          f"re-evaluating an existing run_name must not add a row, got {len(rows_after_reeval)} rows")
    check(abs(float(row2b['map50_95_M']) - 0.481) < 1e-9,
          f"re-eval should update the row's map50_95_M in place, got {row2b['map50_95_M']}")
    check(row2b['round'] == row2['round'],
          f"re-eval should keep the same round number, got {row2b['round']} vs {row2['round']}")
finally:
    shutil.rmtree(tmp1, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: never-trained class must not trip gate_class even with a huge AP swing
# ---------------------------------------------------------------------------
tmp2 = tempfile.mkdtemp(prefix='rounds_ledger_t2_')
try:
    rounds_dir = os.path.join(tmp2, 'step6')
    # PAST has zero labels in train/labels -> "never trained"
    dataset_dir = make_dataset_dir(os.path.join(tmp2, 'dataset'), holdout_mode='transect',
                                    trained_classes=('SS', 'MCAV'))

    run1_dir = make_run_dir(tmp2, 'run1')
    m1 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.01})
    ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)

    run2_dir = make_run_dir(tmp2, 'run2')
    # PAST swings wildly (0.01 -> 0.60, an increase) and SS/MCAV unchanged;
    # also test PAST dropping hard (0.01 -> 0.0) should still be ignored.
    m2 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.0})
    row2 = ER.append_round_row(rounds_dir, run2_dir, m2, dataset_dir)
    check(row2['gate_class'] == 'pass',
          f"never-trained class PAST must be ignored by gate_class, got {row2['gate_class']}")
finally:
    shutil.rmtree(tmp2, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: out-of-order eval baselines on champion.json, not the last row
# ---------------------------------------------------------------------------
tmp3 = tempfile.mkdtemp(prefix='rounds_ledger_t3_')
try:
    rounds_dir = os.path.join(tmp3, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp3, 'dataset'), holdout_mode='transect')

    run1_dir = make_run_dir(tmp3, 'run1')
    m1 = fabricate_metrics(map50_95_m=0.600, recall_m=0.85,
                            per_class_ap={'SS': 0.6, 'MCAV': 0.6, 'PAST': 0.6})
    ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)

    run2_dir = make_run_dir(tmp3, 'run2')
    # run2 is a weak/regressed round, appended last -> becomes "last row"
    m2 = fabricate_metrics(map50_95_m=0.300, recall_m=0.50,
                            per_class_ap={'SS': 0.3, 'MCAV': 0.3, 'PAST': 0.3})
    ER.append_round_row(rounds_dir, run2_dir, m2, dataset_dir)

    # champion.json points at run1 (the good round), NOT the last-appended run2
    os.makedirs(rounds_dir, exist_ok=True)
    with open(os.path.join(rounds_dir, 'champion.json'), 'w') as f:
        json.dump({'run_dir': run1_dir, 'promoted_at': '2026-07-09T08:00:00-04:00',
                   'map50_95_M': 0.600, 'note': ''}, f)

    run3_dir = make_run_dir(tmp3, 'run3')
    # run3 sits between champion (0.600) and last row (0.300): 0.550.
    # If baselined on the LAST ROW (0.300) this would pass; baselined on the
    # CHAMPION (0.600) it must fail (0.550 < 0.600 - 0.005).
    m3 = fabricate_metrics(map50_95_m=0.550, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.55, 'PAST': 0.55})
    row3 = ER.append_round_row(rounds_dir, run3_dir, m3, dataset_dir)
    check(row3['gate_map'] == 'fail',
          f"out-of-order eval must baseline gate_map on champion (0.600), not last row (0.300); got {row3['gate_map']}")
finally:
    shutil.rmtree(tmp3, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: unpinned dataset (no manifest, or holdout_mode != transect) -> gates 'unpinned'
# ---------------------------------------------------------------------------
tmp4 = tempfile.mkdtemp(prefix='rounds_ledger_t4_')
try:
    rounds_dir = os.path.join(tmp4, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp4, 'dataset'), holdout_mode=None)  # no manifest at all

    run1_dir = make_run_dir(tmp4, 'run1')
    m1 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.45})
    row1 = ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)
    check(row1['split_pinned'] == 'false', f"missing manifest should yield split_pinned=false, got {row1['split_pinned']}")
    check(row1['gate_map'] == 'unpinned', f"missing manifest should yield gate_map=unpinned, got {row1['gate_map']}")
    check(row1['gate_class'] == 'unpinned', f"missing manifest should yield gate_class=unpinned, got {row1['gate_class']}")
    check(row1['gate_recall'] == 'unpinned', f"missing manifest should yield gate_recall=unpinned, got {row1['gate_recall']}")

    run2_dir = make_run_dir(tmp4, 'run2')
    m2 = fabricate_metrics(map50_95_m=0.100, recall_m=0.10,
                            per_class_ap={'SS': 0.1, 'MCAV': 0.1, 'PAST': 0.1})
    row2 = ER.append_round_row(rounds_dir, run2_dir, m2, dataset_dir)
    check(row2['gate_map'] == 'unpinned',
          f"even a large regression stays 'unpinned' (informational only) on a non-frozen holdout, got {row2['gate_map']}")

    # adopted-random explicitly -> also unpinned
    dataset_dir_ar = make_dataset_dir(os.path.join(tmp4, 'dataset_ar'), holdout_mode='adopted-random')
    run3_dir = make_run_dir(tmp4, 'run3')
    m3 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.45})
    row3 = ER.append_round_row(os.path.join(tmp4, 'step6_ar'), run3_dir, m3, dataset_dir_ar)
    check(row3['split_pinned'] == 'false',
          f"holdout_mode=adopted-random must yield split_pinned=false, got {row3['split_pinned']}")
    check(row3['gate_map'] == 'unpinned',
          f"holdout_mode=adopted-random must yield gate_map=unpinned, got {row3['gate_map']}")

    # transect-degraded -> also unpinned
    dataset_dir_td = make_dataset_dir(os.path.join(tmp4, 'dataset_td'), holdout_mode='transect-degraded')
    run4_dir = make_run_dir(tmp4, 'run4')
    m4 = fabricate_metrics(map50_95_m=0.500, recall_m=0.80,
                            per_class_ap={'SS': 0.55, 'MCAV': 0.50, 'PAST': 0.45})
    row4 = ER.append_round_row(os.path.join(tmp4, 'step6_td'), run4_dir, m4, dataset_dir_td)
    check(row4['split_pinned'] == 'false',
          f"holdout_mode=transect-degraded must yield split_pinned=false, got {row4['split_pinned']}")
finally:
    shutil.rmtree(tmp4, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: base_model + n_train/n_valid/n_test sanity
# ---------------------------------------------------------------------------
tmp5 = tempfile.mkdtemp(prefix='rounds_ledger_t5_')
try:
    rounds_dir = os.path.join(tmp5, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp5, 'dataset'), holdout_mode='transect',
                                    n_train=37, n_valid=9, n_test=6)
    run1_dir = make_run_dir(tmp5, 'run1', base_model='yolov8s-seg.pt')
    m1 = fabricate_metrics(map50_95_m=0.5, recall_m=0.8, per_class_ap={'SS': 0.5})
    row1 = ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)
    check(row1['base_model'] == 'yolov8s-seg.pt', f"base_model should read from args.yaml, got {row1['base_model']}")
    check(row1['n_train'] == '37' or row1['n_train'] == 37, f"n_train mismatch: {row1['n_train']}")
    check(row1['n_valid'] == '9' or row1['n_valid'] == 9, f"n_valid mismatch: {row1['n_valid']}")
    check(row1['n_test'] == '6' or row1['n_test'] == 6, f"n_test mismatch: {row1['n_test']}")
finally:
    shutil.rmtree(tmp5, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: the image count covers every type the trainer reads.
# Before 2026-09-15 _count_images held its own six-entry list, so the fish
# detector's ledger row said 5296 / 1512 / 757 for a dataset of 5300 / 1514 /
# 758: short by exactly the 7 WebP images in it.
# ---------------------------------------------------------------------------
tmp6 = tempfile.mkdtemp(prefix='rounds_ledger_t6_')
try:
    import detect_dataset

    suffixes = ER._image_suffixes()
    check('.webp' in suffixes, f"WebP is an image the trainer reads; suffixes were {suffixes}")
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
        check(ext in suffixes, f"{ext} counted before the fix and must still count; got {suffixes}")
    unlisted = [e for e in detect_dataset.IMAGE_EXTS if e.lower() not in suffixes]
    check(not unlisted,
          f"every type the detection stager writes must be countable, missing {unlisted}")

    # One image of each supported type, all counted.
    all_types_dir = os.path.join(tmp6, 'all_types')
    img_dir = os.path.join(all_types_dir, 'train', 'images')
    os.makedirs(img_dir)
    for i, ext in enumerate(suffixes):
        with open(os.path.join(img_dir, f'img_{i}{ext}'), 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')
    n_all = ER._count_images(all_types_dir, 'train')
    check(n_all == len(suffixes),
          f"one image of each of the {len(suffixes)} supported types should count "
          f"{len(suffixes)}, got {n_all}")

    # A missing split folder is still 0, not an error.
    check(ER._count_images(all_types_dir, 'valid') == 0,
          "a split folder that does not exist must count 0")

    # Test 6b: nothing that is not a readable image inflates the count.
    mixed_dir = os.path.join(tmp6, 'mixed')
    mixed_imgs = os.path.join(mixed_dir, 'train', 'images')
    os.makedirs(mixed_imgs)
    for name in ('a.jpg', 'b.png', 'c.webp', 'D.JPEG'):
        with open(os.path.join(mixed_imgs, name), 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')
    for name in ('labels.txt', 'notes.md', 'checksums.sha256', 'README', 'a.jpg.bak', '.hidden'):
        with open(os.path.join(mixed_imgs, name), 'w') as f:
            f.write('not an image\n')
    os.makedirs(os.path.join(mixed_imgs, 'archive.png'))          # a folder named like an image
    os.symlink(os.path.join(mixed_imgs, 'gone.jpg'),
               os.path.join(mixed_imgs, 'dangling.jpg'))          # a symlink to nothing
    n_mixed = ER._count_images(mixed_dir, 'train')
    check(n_mixed == 4,
          f"4 images beside 6 non-images, a folder named archive.png and a dangling "
          f"symlink should count 4, got {n_mixed}")

    # Test 6c: the six types counted before the fix still count, on their own.
    legacy_dir = os.path.join(tmp6, 'legacy')
    legacy_imgs = os.path.join(legacy_dir, 'test', 'images')
    os.makedirs(legacy_imgs)
    for i, ext in enumerate(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        with open(os.path.join(legacy_imgs, f'old_{i}{ext}'), 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')
    n_legacy = ER._count_images(legacy_dir, 'test')
    check(n_legacy == 6, f"the six pre-existing image types should count 6, got {n_legacy}")
finally:
    shutil.rmtree(tmp6, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: the ledger row itself carries the WebP images (the reported defect)
# ---------------------------------------------------------------------------
tmp7 = tempfile.mkdtemp(prefix='rounds_ledger_t7_')
try:
    rounds_dir = os.path.join(tmp7, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp7, 'dataset'), holdout_mode='transect',
                                    n_train=40, n_valid=10, n_test=8)
    extra_webp = {'train': 4, 'valid': 2, 'test': 1}
    for split, n_extra in extra_webp.items():
        img_dir = os.path.join(dataset_dir, split, 'images')
        for i in range(n_extra):
            with open(os.path.join(img_dir, f'{split}_webp_{i}.webp'), 'wb') as f:
                f.write(b'RIFF0000WEBPVP8 ')

    run1_dir = make_run_dir(tmp7, 'run1', base_model='yolo11l.pt')
    m1 = fabricate_metrics(map50_95_m=0.5, recall_m=0.8, per_class_ap={'SS': 0.5})
    row1 = ER.append_round_row(rounds_dir, run1_dir, m1, dataset_dir)
    check(int(row1['n_train']) == 44,
          f"40 JPEG + 4 WebP training images should be recorded as 44, got {row1['n_train']}")
    check(int(row1['n_valid']) == 12,
          f"10 JPEG + 2 WebP validation images should be recorded as 12, got {row1['n_valid']}")
    check(int(row1['n_test']) == 9,
          f"8 JPEG + 1 WebP test images should be recorded as 9, got {row1['n_test']}")
finally:
    shutil.rmtree(tmp7, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: with ultralytics unavailable the count falls back to the detection
# stager's own list, and no third list hides in evaluate_run.py
# ---------------------------------------------------------------------------
tmp8 = tempfile.mkdtemp(prefix='rounds_ledger_t8_')
blocked = {name: mod for name, mod in sys.modules.items()
           if name == 'ultralytics' or name.startswith('ultralytics.')}
try:
    import detect_dataset

    for name in blocked:
        sys.modules[name] = None          # any ultralytics import now raises ImportError
    try:
        from ultralytics.data.utils import IMG_FORMATS  # noqa: F401
        check(False, "the fixture failed to hide ultralytics, so test 8 proves nothing")
    except ImportError:
        pass

    fallback = ER._image_suffixes()
    expected = tuple(sorted(e.lower() for e in detect_dataset.IMAGE_EXTS))
    check(fallback == expected,
          f"without ultralytics the suffixes must be the stager's IMAGE_EXTS {expected}, got {fallback}")

    img_dir = os.path.join(tmp8, 'train', 'images')
    os.makedirs(img_dir)
    for name in ('a.jpg', 'b.webp', 'c.avif', 'labels.txt'):
        with open(os.path.join(img_dir, name), 'wb') as f:
            f.write(b'\xff\xd8\xff\xd9')
    n_fallback = ER._count_images(tmp8, 'train')
    check(n_fallback == 2,
          f"the stager's list covers JPEG and WebP but not AVIF, so 2 of the 4 files count, "
          f"got {n_fallback}")
finally:
    for name, mod in blocked.items():
        sys.modules[name] = mod
    shutil.rmtree(tmp8, ignore_errors=True)

try:
    from ultralytics.data.utils import IMG_FORMATS as restored_formats
except ImportError:
    restored_formats = None          # the box has no ultralytics; the rest of the suite stands
if restored_formats is not None:
    check(ER._image_suffixes() == tuple(sorted('.' + f.lower() for f in restored_formats)),
          "putting ultralytics back must restore the trainer's own list of image types")


# ---------------------------------------------------------------------------
# Test 9: the row records what the score was measured with (conf, iou, imgsz)
# ---------------------------------------------------------------------------
tmp9 = tempfile.mkdtemp(prefix='rounds_ledger_t9_')
try:
    rounds_dir = os.path.join(tmp9, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp9, 'dataset'), holdout_mode='transect')
    m = fabricate_metrics(map50_95_m=0.5, recall_m=0.8, per_class_ap={'SS': 0.5})

    check(ER.ROUNDS_CSV_HEADER[-3:] == ['conf', 'iou', 'imgsz'],
          f"the settings columns go at the END so an older ledger still reads, "
          f"got {ER.ROUNDS_CSV_HEADER[-4:]}")

    run1_dir = make_run_dir(tmp9, 'run1')
    row1 = ER.append_round_row(rounds_dir, run1_dir, m, dataset_dir,
                               conf=0.001, iou=0.6, imgsz=512)
    check((row1['conf'], row1['iou'], row1['imgsz']) == (0.001, 0.6, 512),
          f"the row carries the settings the evaluation used, got "
          f"{(row1['conf'], row1['iou'], row1['imgsz'])}")
    on_disk = read_rows(rounds_dir)[0]
    check((on_disk['conf'], on_disk['iou'], on_disk['imgsz']) == ('0.001', '0.6', '512'),
          f"and writes them to rounds.csv, got "
          f"{(on_disk['conf'], on_disk['iou'], on_disk['imgsz'])}")

    # A caller that does not say what it measured with leaves the cells empty
    # rather than inventing a number.
    run2_dir = make_run_dir(tmp9, 'run2')
    row2 = ER.append_round_row(rounds_dir, run2_dir, m, dataset_dir)
    check((row2['conf'], row2['iou'], row2['imgsz']) == ('', '', ''),
          f"unstated settings stay blank, never a guessed number, got "
          f"{(row2['conf'], row2['iou'], row2['imgsz'])}")

    # A deliberate 0 is a real threshold and must survive as one.
    zeroed = ER._eval_settings(0, 0.0, 0)
    check(zeroed == {'conf': 0.0, 'iou': 0.0, 'imgsz': 0},
          f"a typed 0 is kept, never read as 'not supplied', got {zeroed}")

    # A setting that is not a number is refused by name, not written as text,
    # and nothing is written before the refusal.
    for bad in ({'conf': 'very low'}, {'iou': [0.6]}, {'imgsz': '512px'}):
        name = list(bad)[0]
        before = read_rows(rounds_dir)
        try:
            ER.append_round_row(rounds_dir, make_run_dir(tmp9, 'run_' + name), m,
                                dataset_dir, **bad)
            check(False, f"a non-numeric {name} must raise ValueError")
        except ValueError as exc:
            check(name in str(exc), f"the refusal names the setting, got {exc}")
        check(read_rows(rounds_dir) == before,
              f"a refused {name} leaves the ledger exactly as it was")
finally:
    shutil.rmtree(tmp9, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: a rounds.csv written before the settings columns still reads back
# ---------------------------------------------------------------------------
tmp10 = tempfile.mkdtemp(prefix='rounds_ledger_t10_')
try:
    rounds_dir = os.path.join(tmp10, 'step6')
    dataset_dir = make_dataset_dir(os.path.join(tmp10, 'dataset'), holdout_mode='transect')
    os.makedirs(rounds_dir, exist_ok=True)

    old_header = [c for c in ER.ROUNDS_CSV_HEADER if c not in ('conf', 'iou', 'imgsz')]
    old_row = {c: '' for c in old_header}
    old_row.update({'round': '1', 'at': '2026-09-05T10:00:00-04:00', 'run_name': 'old_run',
                    'run_dir': os.path.join(tmp10, 'old_run'), 'base_model': 'yolo11l.pt',
                    'split_pinned': 'true', 'n_train': '40', 'n_valid': '10', 'n_test': '8',
                    'map50_M': '0.600', 'map50_95_M': '0.500', 'recall_M': '0.80',
                    'precision_M': '0.72', 'gate_map': 'pass', 'gate_class': 'pass',
                    'gate_recall': 'pass', 'per_class_json': 'rounds/round_1_per_class.json',
                    'metric_family': 'mask', 'metrics_schema': '2'})
    with open(os.path.join(rounds_dir, 'rounds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=old_header)
        w.writeheader()
        w.writerow(old_row)

    before = ER._read_rounds_csv(os.path.join(rounds_dir, 'rounds.csv'))
    check(len(before) == 1 and 'conf' not in before[0],
          f"the fixture really is a pre-settings ledger, got {sorted(before[0])[:4]}")

    m = fabricate_metrics(map50_95_m=0.520, recall_m=0.81, per_class_ap={'SS': 0.52})
    new_run_dir = make_run_dir(tmp10, 'new_run')
    row = ER.append_round_row(rounds_dir, new_run_dir, m, dataset_dir,
                              conf=0.001, iou=0.6, imgsz=512)
    rows = read_rows(rounds_dir)
    check(len(rows) == 2, f"the older row survives beside the new one, got {len(rows)} rows")
    check(rows[0]['map50_95_M'] == '0.500' and rows[0]['run_name'] == 'old_run',
          f"the older row's score is never rewritten, got {rows[0]['map50_95_M']}")
    check(rows[0]['conf'] == '' and rows[0]['iou'] == '' and rows[0]['imgsz'] == '',
          f"the older row reads back with empty settings cells, got {rows[0]['conf']!r}")
    check(rows[1]['conf'] == '0.001' and rows[1]['imgsz'] == '512',
          f"the new row carries what it was measured with, got {rows[1]['conf']!r}")
    check(row['gate_map'] == ER.GATE_INCOMPARABLE,
          f"a baseline that records no confidence cannot be ranked against a measured "
          f"round, so the gate reads incomparable, got {row['gate_map']}")
    check((row['gate_class'], row['gate_recall'])
          == (ER.GATE_INCOMPARABLE, ER.GATE_INCOMPARABLE),
          f"and so do the other two gates, got {(row['gate_class'], row['gate_recall'])}")
finally:
    shutil.rmtree(tmp10, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 11: two rows measured at different confidences are never ranked
# ---------------------------------------------------------------------------
tmp11 = tempfile.mkdtemp(prefix='rounds_ledger_t11_')
try:
    dataset_dir = make_dataset_dir(os.path.join(tmp11, 'dataset'), holdout_mode='transect')
    strong = fabricate_metrics(map50_95_m=0.600, recall_m=0.85, per_class_ap={'SS': 0.60})
    weak = fabricate_metrics(map50_95_m=0.400, recall_m=0.60, per_class_ap={'SS': 0.40})

    def ledger_with(baseline_settings, round_settings, baseline_metrics=strong,
                    round_metrics=weak, folder='case'):
        """Two rounds in a fresh ledger; return the second round's row."""
        rounds_dir = os.path.join(tmp11, folder)
        ER.append_round_row(rounds_dir, make_run_dir(tmp11, folder + '_base'),
                            baseline_metrics, dataset_dir, **baseline_settings)
        return ER.append_round_row(rounds_dir, make_run_dir(tmp11, folder + '_new'),
                                   round_metrics, dataset_dir, **round_settings)

    measured = {'conf': 0.001, 'iou': 0.6, 'imgsz': 512}
    same = ledger_with(measured, measured, folder='same_conf')
    check(same['gate_map'] == ER.GATE_FAIL,
          f"two rounds measured the same way are still ranked normally, got {same['gate_map']}")

    moved = ledger_with(dict(measured, conf=0.25), measured, folder='moved_conf')
    check(moved['gate_map'] == ER.GATE_INCOMPARABLE,
          f"a baseline measured at another confidence yields no verdict, got {moved['gate_map']}")
    check(moved['gate_recall'] == ER.GATE_INCOMPARABLE and 'flag' not in moved['gate_class'],
          f"the per-class and recall gates say the same, got {moved['gate_class']}")

    silent = ledger_with({}, {}, folder='neither_records')
    check(silent['gate_map'] == ER.GATE_FAIL,
          f"a ledger where neither row records a confidence behaves as it did before the "
          f"columns existed, got {silent['gate_map']}")

    # A hand-edited ledger must never crash an evaluation: a confidence cell
    # that is not a number reads as 'not recorded', which is incomparable.
    hostile_dir = os.path.join(tmp11, 'hostile')
    ER.append_round_row(hostile_dir, make_run_dir(tmp11, 'hostile_base'), strong,
                        dataset_dir, **measured)
    rows = ER._read_rounds_csv(os.path.join(hostile_dir, 'rounds.csv'))
    rows[0]['conf'] = 'very low'
    ER._write_rounds_csv(os.path.join(hostile_dir, 'rounds.csv'), rows)
    hostile = ER.append_round_row(hostile_dir, make_run_dir(tmp11, 'hostile_new'), weak,
                                  dataset_dir, **measured)
    check(hostile['gate_map'] == ER.GATE_INCOMPARABLE,
          f"an unreadable confidence cell is treated as unrecorded, got {hostile['gate_map']}")

    # An unpinned split still outranks every other reason: the verdict names
    # the loose split, which is the thing to fix first.
    loose_dataset = make_dataset_dir(os.path.join(tmp11, 'loose'), holdout_mode=None)
    loose_dir = os.path.join(tmp11, 'loose_rounds')
    ER.append_round_row(loose_dir, make_run_dir(tmp11, 'loose_base'), strong,
                        loose_dataset, **dict(measured, conf=0.25))
    loose = ER.append_round_row(loose_dir, make_run_dir(tmp11, 'loose_new'), weak,
                                loose_dataset, **measured)
    check(loose['gate_map'] == ER.GATE_UNPINNED,
          f"a loose split still reads unpinned, not incomparable, got {loose['gate_map']}")
finally:
    shutil.rmtree(tmp11, ignore_errors=True)


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
