"""Tests for evaluate_run.py's append_round_row (rounds ledger, gates,
champion-baselined lineage). Pure function: no ultralytics import needed.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_rounds_ledger.py"""
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


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
