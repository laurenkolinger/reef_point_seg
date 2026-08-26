"""seed_from_predictions.py: seed model predictions.json into a step4test
combined-annotator export as PENDING, editable masks for human review.

Part of the active-learning loop: Step 8 inference writes predictions.json
(filename, raw abs path, per-detection normalized polygon + confidence).
This script turns those detections into ordinary seg_dict/mask entries in
segmentations/segmentations.json (source_type='model', status='pending'),
symlinks the raw frame into routed_input/raw/, and writes a round manifest
so a later "reset this round" tool (Task 5) can find exactly what it added.

No prompts; fully flag-driven. Requires cv2/numpy/PIL (present in REPO/env).
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mask_ops
import provenance


# ── Transect helper (DUPLICATED, kept byte-identical with the sibling copy) ──
# keep in sync with TR/src/pinned_split.py:parse_transect (byte-identical, like the provenance.py copies)
def parse_transect(basename):
    """Transect number from a TCRMP frame basename, or None.
    Real names end in .jpeg/.jpg; strip the extension, then the trailing
    _T<digits>. Only a 3-digit T-number yields a transect (its first digit);
    1-/2-/4-digit T-numbers (and unparseable names) yield None -> train, never holdout."""
    stem = os.path.splitext(basename)[0]
    m = re.search(r'_T(\d+)$', stem)
    if m and len(m.group(1)) == 3:
        return int(m.group(1)[0])
    return None


def _load_codes_csv(path):
    """basename-agnostic species code -> {name, category} lookup."""
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get('code') or row.get('species_code') or '').strip()
            if not code:
                continue
            lookup[code] = {
                'name': (row.get('name') or row.get('species_name') or '').strip(),
                'category': (row.get('category') or '').strip(),
            }
    return lookup


def _load_lores_manifest(path):
    """basename (no ext) -> row dict, from a make_lores_variants.py manifest."""
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = (row.get('basename') or '').strip()
            if b:
                lookup[b] = row
    return lookup


def _load_split_manifest(path):
    if not path or not os.path.exists(path):
        print(f"ERROR: --split_manifest is required and was not found: {path}", file=sys.stderr)
        sys.exit(2)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def _is_holdout(basename, split_manifest):
    """Rule 1b: exclude if (a) pinned to valid/test, OR (b) holdout_mode ==
    'transect' AND parse_transect(basename) is in holdout_transects.val+test."""
    pinned = split_manifest.get('pinned', {}) or {}
    assignment = pinned.get(basename)
    if assignment in ('valid', 'test'):
        return True
    if split_manifest.get('holdout_mode') == 'transect':
        holdout = split_manifest.get('holdout_transects', {}) or {}
        val_transects = holdout.get('val', []) or []
        test_transects = holdout.get('test', []) or []
        t = parse_transect(basename)
        if t is not None and (t in val_transects or t in test_transects):
            return True
    return False


def _polygon_xyn_to_px(polygon_xyn, w, h):
    flat = []
    for i in range(0, len(polygon_xyn), 2):
        flat.append(polygon_xyn[i] * w)
        flat.append(polygon_xyn[i + 1] * h)
    return flat


def _centroid(polygon_px):
    xs = polygon_px[0::2]
    ys = polygon_px[1::2]
    if not xs:
        return 0, 0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Seed step8 predictions.json into a step4test export as pending review masks.")
    p.add_argument('--predictions', required=True, help="Path to predictions.json (Step 8 output).")
    p.add_argument('--export_dir', required=True, help="step4test export dir (contains/receives segmentations/).")
    p.add_argument('--codes_csv', required=True, help="Master codes CSV (code,category,name) for species -> name/category.")
    p.add_argument('--split_manifest', required=True,
                   help="REQUIRED (fail-closed, D8): pinned_split.py's split_manifest.json. "
                        "Frames pinned/derived to valid or test are never seeded.")
    p.add_argument('--conf_min', type=float, default=0.25, help="Minimum detection confidence to seed (default 0.25).")
    p.add_argument('--max_frames', type=int, default=0, help="Cap on frames seeded this round (0 = no cap).")
    p.add_argument('--skip_empty', action='store_true', help="Skip zero-detection frames instead of seeding masks:[].")
    p.add_argument('--val_transects', default='5', help="Unused placeholder for CLI symmetry with pinned_split (holdout comes from --split_manifest).")
    p.add_argument('--test_transects', default='6', help="Unused placeholder for CLI symmetry with pinned_split (holdout comes from --split_manifest).")
    p.add_argument('--lores_manifest', default='', help="Optional make_lores_variants.py manifest CSV; prefer lores file when original width > 1920.")
    p.add_argument('--round_note', default='', help="Free-text note stored in the round manifest.")
    p.add_argument('--dry_run', action='store_true', help="Print the summary only; write nothing.")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    split_manifest = _load_split_manifest(args.split_manifest)
    codes_lookup = _load_codes_csv(args.codes_csv)
    lores_lookup = _load_lores_manifest(args.lores_manifest) if args.lores_manifest else {}

    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    export_dir = args.export_dir
    seg_dir = os.path.join(export_dir, 'segmentations')
    raw_dir = os.path.join(export_dir, 'routed_input', 'raw')
    rounds_dir = os.path.join(export_dir, 'loop_rounds')
    seg_path = os.path.join(seg_dir, 'segmentations.json')

    if not args.dry_run:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(rounds_dir, exist_ok=True)

    seg_dict = {}
    if os.path.exists(seg_path):
        with open(seg_path, 'r') as f:
            seg_dict = json.load(f)

    seeded_frames = 0
    seeded_masks = 0
    skipped_existing = []
    skipped_below_conf = 0
    skipped_missing_raw = 0
    skipped_holdout = 0
    round_seeded = {}

    items = predictions.get('items', [])
    for item in items:
        if args.max_frames and seeded_frames >= args.max_frames:
            break

        filename = item.get('filename')
        if not filename:
            continue

        if filename in seg_dict:
            skipped_existing.append(filename)
            continue

        if _is_holdout(filename, split_manifest):
            skipped_holdout += 1
            continue

        raw_abs = item.get('raw')
        if not raw_abs or not os.path.exists(raw_abs):
            skipped_missing_raw += 1
            continue

        width = item.get('width')
        height = item.get('height')

        # Lores preference: only when the original is wider than 1920 and the
        # manifest maps this basename (basename column has no extension).
        seed_src = raw_abs
        stem = os.path.splitext(filename)[0]
        lores_row = lores_lookup.get(stem)
        if lores_row and width and width > 1920:
            lores_path = lores_row.get('lores_path')
            if lores_path and os.path.exists(lores_path):
                seed_src = lores_path
                with Image.open(lores_path) as im:
                    width, height = im.size

        detections = item.get('detections', []) or []

        masks = []
        next_mask_id = 0
        for det in detections:
            confidence = det.get('confidence', 0.0)
            if confidence < args.conf_min:
                skipped_below_conf += 1
                continue

            polygon_xyn = det.get('polygon_xyn') or []
            if len(polygon_xyn) < 6:
                continue

            polygon_px = _polygon_xyn_to_px(polygon_xyn, width, height)
            binary_mask = mask_ops.polygon_to_mask(
                [polygon_px[i:i + 2] for i in range(0, len(polygon_px), 2)],
                height, width)
            cx, cy = _centroid(polygon_px)

            code = det.get('class', '')
            info = codes_lookup.get(code, {})
            point_info = {
                'label': 'M',
                'species': code,
                'name': info.get('name', code),
                'category': info.get('category', ''),
                'x': cx,
                'y': cy,
            }

            mask_dict = mask_ops.build_mask_dict(
                mask_id=next_mask_id,
                binary_mask=binary_mask,
                score=confidence,
                point_info=point_info,
                source_type='model',
            )
            if mask_dict is None:
                continue
            mask_dict['status'] = 'pending'
            masks.append(mask_dict)
            next_mask_id += 1

        if args.skip_empty and not masks:
            continue

        # Symlink raw frame into routed_input/raw/; skip on name collision
        # with a different target (never overwrite).
        link_path = os.path.join(raw_dir, filename)
        if not args.dry_run:
            if os.path.islink(link_path) or os.path.exists(link_path):
                existing_target = os.path.realpath(link_path) if os.path.islink(link_path) else None
                if existing_target != os.path.realpath(seed_src):
                    skipped_existing.append(filename)
                    continue
            else:
                os.symlink(seed_src, link_path)

        seg_dict[filename] = {
            'image_path': f'raw/{filename}',
            'image_path_abs': seed_src,
            'image_width': width,
            'image_height': height,
            'masks': masks,
            'reference_points': [],
            'processed_at': provenance.ast_now(),
            'reviewed': False,
            'exported': False,
            'label_outcomes': {},
        }

        round_seeded[filename] = [m['id'] for m in masks]
        seeded_frames += 1
        seeded_masks += len(masks)

    round_manifest = {
        'schema_version': 1,
        'at': provenance.ast_now(),
        'predictions': os.path.abspath(args.predictions),
        'params': {
            'conf_min': args.conf_min,
            'max_frames': args.max_frames,
            'skip_empty': args.skip_empty,
        },
        'seeded': round_seeded,
        'skipped_existing': skipped_existing,
        'note': args.round_note,
    }

    round_manifest_rel = None
    if not args.dry_run and seeded_frames > 0:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=seg_dir, suffix='.tmp')
        os.close(tmp_fd)
        with open(tmp_path, 'w') as f:
            json.dump(seg_dict, f, indent=2)
        os.replace(tmp_path, seg_path)

        # Build the round filename timestamp from ast_now's own clock so it
        # always matches the AST timestamp recorded 'at' in the manifest.
        ast_dt = datetime.fromisoformat(round_manifest['at'])
        stamp = ast_dt.strftime('%Y%m%d_%H%M%S')
        round_manifest_name = f'round_{stamp}.json'
        round_manifest_path = os.path.join(rounds_dir, round_manifest_name)
        tmp_fd2, tmp_path2 = tempfile.mkstemp(dir=rounds_dir, suffix='.tmp')
        os.close(tmp_fd2)
        with open(tmp_path2, 'w') as f:
            json.dump(round_manifest, f, indent=2)
        os.replace(tmp_path2, round_manifest_path)
        round_manifest_rel = os.path.join('loop_rounds', round_manifest_name)
    elif seeded_frames > 0:
        # dry_run: still report where the manifest WOULD go.
        ast_dt = datetime.fromisoformat(round_manifest['at'])
        stamp = ast_dt.strftime('%Y%m%d_%H%M%S')
        round_manifest_rel = os.path.join('loop_rounds', f'round_{stamp}.json')

    summary = {
        'seeded_frames': seeded_frames,
        'seeded_masks': seeded_masks,
        'skipped_existing': len(skipped_existing),
        'skipped_below_conf': skipped_below_conf,
        'skipped_missing_raw': skipped_missing_raw,
        'skipped_holdout': skipped_holdout,
        'round_manifest': round_manifest_rel,
    }
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
