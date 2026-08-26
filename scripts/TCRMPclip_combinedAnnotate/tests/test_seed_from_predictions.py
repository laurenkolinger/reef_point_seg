"""Tests for seed_from_predictions.py. Run: env/bin/python .../tests/test_seed_from_predictions.py

House harness style: standalone script, invokes the seeder as a subprocess
(exactly as the orchestrator would), asserts on stdout summary JSON + the
resulting segmentations.json / round manifest / symlinks on disk.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PY = os.path.join(REPO, 'env', 'bin', 'python')
SEEDER = os.path.join(HERE, '..', 'src', 'seed_from_predictions.py')

_fail = 0
def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)
    else:
        print("PASS:", msg)


def make_triangle_image(path, w=200, h=100):
    im = Image.new('RGB', (w, h), (10, 20, 30))
    draw = ImageDraw.Draw(im)
    draw.polygon([(10, 10), (190, 10), (100, 90)], fill=(200, 200, 200))
    im.save(path, 'JPEG', quality=90)


def build_fixture(root):
    """Build a temp export dir + raw source dir + predictions.json + codes csv
    + split manifest. Returns paths dict."""
    export_dir = os.path.join(root, 'export')
    seg_dir = os.path.join(export_dir, 'segmentations')
    os.makedirs(seg_dir, exist_ok=True)

    raw_src_dir = os.path.join(root, 'raw_src')
    os.makedirs(raw_src_dir, exist_ok=True)

    # --- fixture frames (real .jpeg/.jpg extensions everywhere) ---
    existing_name = 'TCRMPX_clip_AAA_T118_existing.jpeg'
    train_name = 'TCRMPX_clip_AAA_T118.jpeg'
    empty_name = 'TCRMPX_clip_AAA_T218.jpeg'
    holdout_name = 'TCRMPX_clip_AAA_T601.jpeg'

    for name in (existing_name, train_name, empty_name, holdout_name):
        make_triangle_image(os.path.join(raw_src_dir, name))

    # --- pre-existing exported frame in segmentations.json ---
    seg_dict = {
        existing_name: {
            'image_path': f'raw/{existing_name}',
            'image_path_abs': os.path.join(raw_src_dir, existing_name),
            'image_width': 200,
            'image_height': 100,
            'masks': [],
            'reference_points': [],
            'processed_at': '2026-07-01T10:00:00-04:00',
            'reviewed': True,
            'exported': True,
            'label_outcomes': {},
        }
    }
    seg_path = os.path.join(seg_dir, 'segmentations.json')
    with open(seg_path, 'w') as f:
        json.dump(seg_dict, f)
    existing_mtime = os.path.getmtime(seg_path)

    # --- codes csv (code,category,name) ---
    codes_csv = os.path.join(root, 'master_codes.csv')
    with open(codes_csv, 'w') as f:
        f.write('code,category,name\n')
        f.write('AA,Coral,Agaricia agaricites\n')

    # --- split manifest: holdout_transects val=[5] test=[6], transect mode ---
    split_manifest = os.path.join(root, 'split_manifest.json')
    with open(split_manifest, 'w') as f:
        json.dump({
            'schema_version': 1,
            'created_at': '2026-07-01T10:00:00-04:00',
            'updated_at': '2026-07-01T10:00:00-04:00',
            'holdout_mode': 'transect',
            'transect_rule': 'stem _T + 3 digits, first digit = transect',
            'holdout_transects': {'val': [5], 'test': [6]},
            'pinned': {},
        }, f)

    # --- predictions.json ---
    # (b) train frame, one conf 0.9 triangle detection + one conf 0.1 (filtered)
    triangle_xyn_a = [0.05, 0.1, 0.95, 0.1, 0.5, 0.9]
    triangle_xyn_b = [0.1, 0.15, 0.6, 0.15, 0.35, 0.5]
    predictions = {
        'schema_version': 1,
        'generated_at': '2026-07-09T14:00:00-04:00',
        'run_dir': '/abs/path/step6_trainModel/runs/exemplar',
        'imgsz': 512, 'conf': 0.1, 'iou': 0.7,
        'class_names': {'0': 'AA'},
        'items': [
            {
                'filename': existing_name,
                'raw': os.path.join(raw_src_dir, existing_name),
                'width': 200, 'height': 100,
                'detections': [
                    {'class_id': 0, 'class': 'AA', 'confidence': 0.9, 'polygon_xyn': triangle_xyn_a},
                ],
            },
            {
                'filename': train_name,
                'raw': os.path.join(raw_src_dir, train_name),
                'width': 200, 'height': 100,
                'detections': [
                    {'class_id': 0, 'class': 'AA', 'confidence': 0.9, 'polygon_xyn': triangle_xyn_a},
                    {'class_id': 0, 'class': 'AA', 'confidence': 0.1, 'polygon_xyn': triangle_xyn_b},
                ],
            },
            {
                'filename': empty_name,
                'raw': os.path.join(raw_src_dir, empty_name),
                'width': 200, 'height': 100,
                'detections': [],
            },
            {
                'filename': holdout_name,
                'raw': os.path.join(raw_src_dir, holdout_name),
                'width': 200, 'height': 100,
                'detections': [
                    {'class_id': 0, 'class': 'AA', 'confidence': 0.9, 'polygon_xyn': triangle_xyn_a},
                ],
            },
        ],
    }
    predictions_path = os.path.join(root, 'predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f)

    return {
        'export_dir': export_dir,
        'seg_path': seg_path,
        'codes_csv': codes_csv,
        'split_manifest': split_manifest,
        'predictions_path': predictions_path,
        'existing_name': existing_name,
        'train_name': train_name,
        'empty_name': empty_name,
        'holdout_name': holdout_name,
        'existing_mtime': existing_mtime,
    }


def run_seeder(fx, extra_args=None, use_split_manifest=True):
    cmd = [
        PY, SEEDER,
        '--predictions', fx['predictions_path'],
        '--export_dir', fx['export_dir'],
        '--codes_csv', fx['codes_csv'],
    ]
    if use_split_manifest:
        cmd += ['--split_manifest', fx['split_manifest']]
    if extra_args:
        cmd += extra_args
    return subprocess.run(cmd, capture_output=True, text=True)


# ── Test 1: main happy-path run ──────────────────────────────────────
with tempfile.TemporaryDirectory() as root:
    fx = build_fixture(root)
    result = run_seeder(fx)
    check(result.returncode == 0, f"seeder exits 0 (stderr: {result.stderr})")

    try:
        summary = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        summary = None
        check(False, f"summary JSON parses: {e}; stdout={result.stdout!r}")

    if summary is not None:
        check('seeded_frames' in summary, "summary has seeded_frames")
        check(summary['skipped_existing'] == 1, f"existing frame skipped (got {summary['skipped_existing']})")
        check(summary['skipped_holdout'] == 1, f"T601 test-transect frame skipped_holdout (got {summary['skipped_holdout']})")
        check(summary['skipped_below_conf'] == 1, f"conf 0.1 detection filtered (got {summary['skipped_below_conf']})")
        check(summary['seeded_frames'] == 2, f"train frame + empty frame seeded (got {summary['seeded_frames']})")
        check(summary['seeded_masks'] == 1, f"exactly 1 mask seeded (got {summary['seeded_masks']})")
        check(summary.get('round_manifest'), "round_manifest path present in summary")

    with open(fx['seg_path']) as f:
        seg_dict = json.load(f)

    check(fx['holdout_name'] not in seg_dict, "holdout frame never seeded into segmentations.json")
    check(fx['train_name'] in seg_dict, "train frame present in segmentations.json")

    train_entry = seg_dict.get(fx['train_name'], {})
    check(len(train_entry.get('masks', [])) == 1, "train frame has exactly 1 mask")
    if train_entry.get('masks'):
        m = train_entry['masks'][0]
        check(m.get('status') == 'pending', "mask status == pending")
        check(m.get('source_type') == 'model', "mask source_type == model")
        check(m.get('label') == 'M', "mask label == M")
        check(bool(m.get('rle')), "mask has non-empty rle")
        check(bool(m.get('polygon_norm')), "mask has non-empty polygon_norm")
        check(bool(m.get('bbox')), "mask has non-empty bbox")
        check(abs(m.get('score', 0) - 0.9) < 1e-6, f"mask score == 0.9 (got {m.get('score')})")

    empty_entry = seg_dict.get(fx['empty_name'], {})
    check(empty_entry.get('masks', None) == [], "zero-detection frame present with masks==[]")
    check(empty_entry.get('exported') is False, "zero-detection frame exported==False")

    raw_link = os.path.join(fx['export_dir'], 'routed_input', 'raw', fx['train_name'])
    check(os.path.islink(raw_link) or os.path.exists(raw_link), "symlink exists in routed_input/raw/ for train frame")

    round_manifest_rel = summary['round_manifest'] if summary else None
    round_ok = False
    if round_manifest_rel:
        round_manifest_path = os.path.join(fx['export_dir'], round_manifest_rel)
        round_ok = os.path.exists(round_manifest_path)
        check(round_ok, f"round manifest file exists at {round_manifest_path}")
        if round_ok:
            with open(round_manifest_path) as f:
                rm = json.load(f)
            check(fx['train_name'] in rm.get('seeded', {}), "round manifest maps seeded filename to mask ids")
            check(rm['seeded'].get(fx['train_name']) == [m['id'] for m in train_entry['masks']],
                  "round manifest mask-id list matches seeded masks")

    # ── Test: second identical run seeds nothing ──
    result2 = run_seeder(fx)
    check(result2.returncode == 0, "second run exits 0")
    try:
        summary2 = json.loads(result2.stdout.strip().splitlines()[-1])
        check(summary2['seeded_frames'] == 0, f"second run seeds nothing (got {summary2['seeded_frames']})")
        check(summary2['skipped_existing'] >= 2, f"second run: all previously-seeded + existing frames skipped (got {summary2['skipped_existing']})")
    except Exception as e:
        check(False, f"second-run summary JSON parses: {e}")


# ── Test 2: fail-closed without --split_manifest ────────────────────
with tempfile.TemporaryDirectory() as root:
    fx = build_fixture(root)
    result = run_seeder(fx, use_split_manifest=False)
    check(result.returncode != 0, f"run WITHOUT --split_manifest hard-errors (got exit {result.returncode})")


# ── Test 3: --dry_run leaves mtime unchanged and writes nothing ─────
with tempfile.TemporaryDirectory() as root:
    fx = build_fixture(root)
    before_mtime = os.path.getmtime(fx['seg_path'])
    time.sleep(0.05)
    result = run_seeder(fx, extra_args=['--dry_run'])
    check(result.returncode == 0, f"dry_run exits 0 (stderr: {result.stderr})")
    after_mtime = os.path.getmtime(fx['seg_path'])
    check(before_mtime == after_mtime, "dry_run leaves segmentations.json mtime unchanged")
    raw_dir = os.path.join(fx['export_dir'], 'routed_input', 'raw')
    check(not os.path.isdir(raw_dir) or len(os.listdir(raw_dir)) == 0,
          "dry_run creates no symlinks in routed_input/raw/")
    rounds_dir = os.path.join(fx['export_dir'], 'loop_rounds')
    check(not os.path.isdir(rounds_dir) or len(os.listdir(rounds_dir)) == 0,
          "dry_run writes no round manifest")
    try:
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        check(summary['seeded_frames'] == 2, "dry_run still reports the summary it WOULD have seeded")
    except Exception as e:
        check(False, f"dry_run summary JSON parses: {e}")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
