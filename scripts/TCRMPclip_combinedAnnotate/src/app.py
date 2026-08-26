"""
TCRMP Segment Images — SAM3 Segmentation Review App

Takes sam_click_prompts.json + raw images from TCRMPclip_placePoints
and runs SAM3 segmentation on each point. Provides a review UI for
accepting/rejecting/refining masks, then exports to YOLO Segmentation format.

Usage:
    python app.py [--port 5070]
"""

import os
import sys
import json
import argparse
import subprocess
import threading
import numpy as np
import cv2
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from mask_ops import (
    rle_encode, rle_decode, mask_to_polygon, mask_bbox, mask_centroid,
    resolve_overlaps, merge_close_masks, check_no_overlap, union_masks,
    build_mask_dict, update_mask_geometry, merge_overlapping_same_id,
)
from export_yolo import load_class_map, save_class_map, export_batch, update_data_yaml
from render_overlay import render_segmentation_overlays
from provenance import compute_label_outcomes, write_provenance_csv
import custom_imports as ci

app = Flask(__name__, template_folder='templates')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)         # this is actually the scripts/ dir

# Shared expert-review helpers live in scripts/_reefreview/. Import is guarded
# so a problem there can never stop the core segmentation app from running.
sys.path.insert(0, REPO_DIR)
try:
    from _reefreview import review_export as _review_export
    _REVIEW_OK = True
except Exception as _re_err:  # pragma: no cover
    _review_export = None
    _REVIEW_OK = False
    print(f"[review] _reefreview unavailable, expert-review export disabled: {_re_err}")

# Load target species from TCRMPcvr_chooseImages config
def _load_target_species():
    import re
    # Pipeline orchestrator override
    env_sp = os.environ.get('TCRMP_TARGET_SPECIES', '')
    if env_sp:
        return [s.strip() for s in env_sp.split(',') if s.strip()]
    config_path = os.path.join(REPO_DIR, 'TCRMPcvr_chooseImages', 'src', 'config.py')
    try:
        with open(config_path) as f:
            for line in f:
                if line.strip().startswith('TARGET_SPECIES'):
                    match = re.search(r'\[(.+)\]', line)
                    if match:
                        return [s.strip().strip('"').strip("'")
                                for s in match.group(1).split(',')]
    except FileNotFoundError:
        pass
    return ['OFRA', 'PA', 'OA', 'OFAV', 'AL', 'MC', 'AA']

TARGET_SPECIES = _load_target_species()


def _is_review(p):
    """A REVIEW-flagged prompt/mask: segmented as its OWN positive mask (unlike
    a non-target '?' point, which is a negative refinement click), shipped to the
    expert-review site, and never trained."""
    return bool(p.get('review')) or p.get('species', p.get('species_code', '')) == 'REVIEW'


def _load_target_label_info():
    """Return [{code, name, category}, ...] for each target label.

    Parallels the step 4 helper: prefers master_codes_recoded.csv from the
    project dir, falls back to scanning the recoded all_points.csv. Always
    emits one entry per code so the UI has a stable index for 1..N keys.
    """
    import csv as _csv
    lookup = {}

    input_dir = os.environ.get('TCRMP_INPUT_DIR', '') or cfg.INPUT_DIR
    project_dir = os.path.dirname(input_dir) if input_dir else ''
    candidate_dirs = []
    if project_dir:
        candidate_dirs.append(project_dir)
        candidate_dirs.append(os.path.join(project_dir, 'step2_recodeSpecies'))
    candidate_dirs.append(os.path.join(REPO_DIR, 'TCRMPcvr_recodeSpecies', 'output'))
    candidate_dirs.append(os.path.join(REPO_DIR, 'output'))

    master_candidates = []
    for d in candidate_dirs:
        master_candidates.append(os.path.join(d, 'master_codes_recoded.csv'))
        master_candidates.append(os.path.join(d, 'master_codes.csv'))

    for mc in master_candidates:
        if not os.path.exists(mc):
            continue
        try:
            with open(mc, 'r') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    code = (row.get('species_code') or row.get('code') or '').strip()
                    if not code:
                        continue
                    lookup[code] = {
                        'name': (row.get('species_name') or row.get('name') or '').strip(),
                        'category': (row.get('category') or '').strip(),
                    }
            if lookup:
                print(f"[target-labels] Loaded {len(lookup)} codes from {os.path.basename(mc)}")
                break
        except Exception as e:
            print(f"[target-labels] Failed to read {mc}: {e}")

    return [{'code': c,
             'name': lookup.get(c, {}).get('name', ''),
             'category': lookup.get(c, {}).get('category', '')}
            for c in TARGET_SPECIES]


ALL_TARGET_LABELS = _load_target_label_info()

# ── Session state ────────────────────────────────────────────────

session = {
    'input_dir': '',
    'export_dir': '',
    'categories': [],
    'configured': False,
    'phase': 'startup',          # startup | processing | review
    'sam_engine': None,

    # Per-image segmentation data, flat dict
    'segmentations': {},  # flat {image_filename: seg_dict}

    # Processing queue
    'processing_queue': [],      # list of {image_path, filename, prompts}
    'processing_idx': 0,

    # Review state
    'review_files': [],
    'review_offset': 0,
    'review_batch_size': cfg.REVIEW_BATCH_SIZE,

    # Class registry for YOLO export
    'class_map': {},

    # Flush
    'flush_counter': 0,
    'flush_interval': cfg.FLUSH_INTERVAL,

    # Currently-embedded image for refinement
    'current_review_image': None,

    # Operator initials for provenance attribution (best-effort).
    'reviewer': cfg.REVIEWER,
}


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)


# ── Persistence (flat JSON) ──────────────────────────────────────

def _seg_path(export_dir):
    return os.path.join(export_dir, 'segmentations', 'segmentations.json')


def _manifest_path(export_dir):
    return os.path.join(export_dir, 'segmentations', 'processed_manifest.json')


def load_segmentations(export_dir):
    path = _seg_path(export_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_segmentations(export_dir, data):
    path = _seg_path(export_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)


def save_manifest(export_dir, filenames):
    path = _manifest_path(export_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'processed': sorted(filenames),
            'last_updated': datetime.now().isoformat(),
            'count': len(filenames),
        }, f, indent=2)


EXPORT_MANIFEST_FIELDS = ['basename', 'n_masks', 'label_file', 'image_file',
                          'outcome', 'preview_file', 'batch_name', 'image_path']


def _read_export_manifest(export_dir):
    """Read export_manifest.csv into {basename: full_row_dict}, preserving ALL
    columns (base fields + dynamic per-species counts). {} if absent."""
    import csv as _csv
    path = os.path.join(export_dir, 'export_manifest.csv')
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {r['basename']: dict(r) for r in _csv.DictReader(f)}


def _next_batch_name(export_dir):
    """Sequential per-export label batch_<NNN>_<YYYYMMDD-HHMMSS>; <NNN> persists
    across batches in .export_batch_seq."""
    seq_path = os.path.join(export_dir, '.export_batch_seq')
    n = 0
    if os.path.exists(seq_path):
        try:
            n = int((open(seq_path).read().strip() or '0'))
        except ValueError:
            n = 0
    n += 1
    with open(seq_path, 'w') as f:
        f.write(str(n))
    return f"batch_{n:03d}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _publish_preview(export_dir, stem):
    """Copy the reviewed overlay for one frame into the flat previews/ folder.
    Returns the export-root-relative path, or "" when no source overlay exists.
    """
    import shutil as _shutil
    src = os.path.join(export_dir, 'segmentations', 'overlays_reviewed', f'{stem}_seg.jpg')
    if not os.path.exists(src):
        return ''
    previews_dir = os.path.join(export_dir, 'previews')
    os.makedirs(previews_dir, exist_ok=True)
    dest_name = f'{stem}_seg.jpg'
    _shutil.copyfile(src, os.path.join(previews_dir, dest_name))
    return os.path.join('previews', dest_name)


def write_export_manifest(export_dir, entries):
    """Cumulative manifest with dynamic per-species columns. Base columns are
    EXPORT_MANIFEST_FIELDS; every later column is a per-image accepted-mask count
    for one species (bare code header). Merge by basename (re-export overwrites),
    union all species columns, backfill missing species with 0, sort by basename.
    """
    import csv as _csv
    os.makedirs(export_dir, exist_ok=True)
    merged = _read_export_manifest(export_dir)
    for e in entries:
        merged[e['basename']] = dict(e)
    species = set()
    for row in merged.values():
        species.update(k for k in row if k not in EXPORT_MANIFEST_FIELDS)
    species_cols = sorted(species)
    fieldnames = EXPORT_MANIFEST_FIELDS + species_cols
    path = os.path.join(export_dir, 'export_manifest.csv')
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for basename in sorted(merged):
            row = merged[basename]
            out = {k: row.get(k, '') for k in EXPORT_MANIFEST_FIELDS}
            for sc in species_cols:
                out[sc] = row.get(sc, 0)
            w.writerow(out)


def reconcile_export(export_dir, manifest_entries):
    """Reconcile the on-disk export against the CUMULATIVE manifest.

    Because all_images/ accumulates across batches, this-batch entries alone
    will never match the on-disk image count from batch 2 on. Reconcile against
    the full export_manifest.csv on disk (written cumulatively just before this
    call). If no manifest exists on disk (e.g. a direct reconcile_export call in
    a unit test), fall back to the passed-in this-batch entries — a single
    batch's cumulative is exactly that batch.
    """
    on_disk = _read_export_manifest(export_dir)
    rows = list(on_disk.values()) if on_disk else list(manifest_entries)

    imgs = os.listdir(os.path.join(export_dir, 'all_images')) \
        if os.path.isdir(os.path.join(export_dir, 'all_images')) else []
    exported = [e for e in rows
                if e['outcome'] in ('exported_with_masks', 'exported_empty')]
    for e in exported:
        lp = os.path.join(export_dir, 'all_labels', e['label_file'])
        assert os.path.exists(lp), f"routed frame {e['basename']} missing label file {lp}"
    assert len(imgs) == len(exported), (
        f"all_images={len(imgs)} != exported={len(exported)} "
        f"(every routed non-review/non-scrap image must be on disk)")
    return {"on_disk_images": len(imgs), "exported": len(exported)}


def flush_all():
    export_dir = session['export_dir']
    if not export_dir:
        return
    save_segmentations(export_dir, session['segmentations'])
    save_manifest(export_dir, list(session['segmentations'].keys()))
    if session['class_map']:
        save_class_map(export_dir, session['class_map'])


# ── Review helpers ───────────────────────────────────────────────

def _build_review_list():
    """Build the review queue.

    Mirrors the step 4 policy: once a frame is exported or scrapped it is
    considered done and does not re-enter the queue. Done-is-done.

    If `session['shuffle']` is set, the unexported list is shuffled so the
    user sees a random sample instead of alphabetical ordering.
    """
    import random as _random
    target_only = getattr(cfg, 'TARGET_SPECIES_ONLY', 0)
    unexported = []
    for fn in sorted(session['segmentations']):
        seg = session['segmentations'][fn]
        if seg.get('scrapped', False):
            continue
        if seg.get('exported', False):
            continue
        if target_only and not getattr(cfg, 'MANUAL_ANNOTATE', False):
            # Custom imports start maskless by definition (no routed prompts),
            # so the no-target-mask skip must not hide them from review.
            if not seg.get('custom_import', False):
                masks = seg.get('masks', [])
                if not any(m.get('species', '') in TARGET_SPECIES for m in masks):
                    continue
        unexported.append(fn)
    if session.get('shuffle'):
        _random.shuffle(unexported)
    session['review_files'] = unexported
    session['review_offset'] = 0


def _merge_custom_imports():
    """Graft previously imported custom images into the live session.

    Reads custom_imports/imports_manifest.json under the session export dir
    and adds a seg_dict for every entry not already present (the normal case:
    segmentations.json already carries them from the flush at import time, so
    this only backfills after a segmentations store reset). Returns the number
    added. Never raises: a missing delivered file is logged and skipped."""
    export_dir = session.get('export_dir')
    if not export_dir:
        return 0
    added = 0
    for entry in ci.load_manifest(export_dir):
        fn = entry.get('filename')
        if not fn or fn in session['segmentations']:
            continue
        seg = ci.seg_dict_for(export_dir, entry)
        if not os.path.exists(seg['image_path_abs']):
            log(f"[import] WARNING: delivered file missing for {fn}, skipping")
            continue
        session['segmentations'][fn] = seg
        added += 1
    if added:
        log(f"[import] restored {added} custom import(s) from the manifest")
    return added


def _find_segmentation(filename):
    """Return the seg_dict for a filename, or None."""
    return session['segmentations'].get(filename)


def _ensure_image_embedded(filename):
    """Ensure the SAM3 engine has the given image embedded for refinement."""
    engine = session.get('sam_engine')
    if engine is None:
        return False

    seg = _find_segmentation(filename)
    if seg is None:
        return False

    image_path = seg.get('image_path_abs')
    if not image_path or not os.path.exists(image_path):
        return False

    # Always verify engine has the right image loaded (don't trust session cache)
    if session.get('current_review_image') != filename or engine._current_image_path != image_path:
        engine._current_image_path = None  # force reload
        engine.set_image(image_path)
        session['current_review_image'] = filename

    return True


# ── Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           default_input=cfg.INPUT_DIR,
                           default_export=cfg.EXPORT_DIR,
                           default_categories=cfg.DEFAULT_CATEGORIES,
                           all_categories=[
                               "Target species only",
                               "Coral", "Sponge", "Macroalgae", "Gorgonian",
                               "Dca", "Turf", "Non-living",
                           ],
                           target_species=TARGET_SPECIES,
                           target_labels=ALL_TARGET_LABELS,
                           default_review_dir=cfg.REVIEW_DIR,
                           default_contacts=cfg.REVIEW_CONTACTS,
                           target_species_only=bool(getattr(cfg, 'TARGET_SPECIES_ONLY', 1)),
                           reference_default=bool(getattr(cfg, 'REFERENCE_DEFAULT', 1)),
                           default_confidence=cfg.CONFIDENCE_THRESHOLD,
                           default_min_area=cfg.MIN_MASK_AREA_PX,
                           default_merge_dist=cfg.MERGE_DISTANCE_PX,
                           default_overlap=cfg.OVERLAP_STRATEGY,
                           default_mask_size=cfg.SAM3_MASK_SIZE,
                           default_thin_ratio=cfg.THIN_MASK_RATIO,
                           default_simplify=cfg.POLYGON_SIMPLIFY_EPSILON,
                           default_exemplar=cfg.EXEMPLAR_THRESHOLD,
                           default_dev_tracker=cfg.SAM3_DEVICE_TRACKER,
                           default_dev_exemplar=cfg.SAM3_DEVICE_EXEMPLAR,
                           orchestrator_url=_orchestrator_url(),
                           orchestrated=(bool(os.environ.get('TCRMP_INPUT_DIR'))
                                         or request.args.get('orchestrated') == '1'),
                           session_mode=getattr(cfg, 'SESSION_MODE', 'configure'))


@app.route('/api/configure', methods=['POST'])
def configure():
    """Load sam_click_prompts.json for selected categories, build queue."""
    data = request.json or {}
    input_dir = data.get('input_dir', cfg.INPUT_DIR)
    export_dir = data.get('export_dir', cfg.EXPORT_DIR)
    categories = data.get('categories', cfg.DEFAULT_CATEGORIES)
    batch_size = data.get('review_batch_size', cfg.REVIEW_BATCH_SIZE)

    # Honor the target-labels-only flag EXPLICITLY (cfg.TARGET_SPECIES_ONLY, set
    # from env by the orchestrator panel), not only via the "Target species only"
    # Categories string. The orchestrated launch POSTs an empty body, so the
    # default must come from cfg/env here. A request may still override it.
    if data.get('target_species_only') is not None:
        target_only_flag = bool(data.get('target_species_only'))
    else:
        target_only_flag = bool(getattr(cfg, 'TARGET_SPECIES_ONLY', 1))
    if target_only_flag and 'Target species only' not in categories:
        categories = ['Target species only'] + list(categories)
    elif not target_only_flag and 'Target species only' in categories:
        categories = [c for c in categories if c != 'Target species only']

    session['input_dir'] = os.path.abspath(input_dir)
    session['export_dir'] = os.path.abspath(export_dir)
    session['categories'] = categories
    session['review_batch_size'] = batch_size
    session['shuffle'] = bool(data.get('shuffle', False))
    session['reviewer'] = (data.get('reviewer') or cfg.REVIEWER)

    # Expert-review settings: where to push the queue, and who gets the CSV.
    session['review_dir'] = (data.get('review_dir') or cfg.REVIEW_DIR)
    _raw_contacts = data.get('contacts', cfg.REVIEW_CONTACTS)
    if isinstance(_raw_contacts, str):
        _raw_contacts = _raw_contacts.split(',')
    session['contacts'] = [c.strip() for c in (_raw_contacts or []) if c and c.strip()]

    os.makedirs(session['export_dir'], exist_ok=True)

    # Load existing class map
    session['class_map'] = load_class_map(session['export_dir'])

    # Flat input: routed_input/ids/sam_click_prompts.json + routed_input/raw/
    input_dir_abs = session['input_dir']
    ids_path = os.path.join(input_dir_abs, 'ids', 'sam_click_prompts.json')
    raw_dir = os.path.join(input_dir_abs, 'raw')
    if not os.path.exists(ids_path):
        return jsonify({'error': (
            'No Step 4 output found for this project (no '
            'ids/sam_click_prompts.json under ' + input_dir_abs + '). Run Step 4 '
            '(routing) first, or open a project that has completed Step 4.'
        )}), 404

    # Load existing segmentations (single flat read)
    existing = load_segmentations(session['export_dir'])
    session['segmentations'] = {}
    session['segmentations'].update(existing)
    already_processed = set(existing.keys())

    with open(ids_path) as f:
        prompts = json.load(f)

    # Build processing queue
    queue = []
    skipped = 0

    for img_filename, img_data in prompts.items():
        if img_filename in already_processed:
            skipped += 1
            continue

        # Filter points by selected categories / target species
        target_sp_only = 'Target species only' in categories
        real_categories = [c for c in categories if c != 'Target species only']

        # Non-target prompt points (right-click from step 4): always
        # included in segmentation regardless of the target-species filter.
        # These are points the reviewer flagged as "segment this too" without
        # assigning a target label; SAM3 still produces a mask for each.
        def _is_non_target(p):
            return p.get('non_target') is True or \
                p.get('species', p.get('species_code', '')) == '?'

        if target_sp_only:
            # Filter directly by target species codes
            # sam_click_prompts uses 'species' field, not 'species_code'
            selected_points = [
                p for p in img_data.get('points', [])
                if p.get('species', p.get('species_code', '')) in TARGET_SPECIES
                or _is_non_target(p) or _is_review(p)
            ]
        elif real_categories:
            selected_points = [
                p for p in img_data.get('points', [])
                if p.get('category', '') in real_categories
                or _is_non_target(p) or _is_review(p)
            ]
            # If TARGET_SPECIES_ONLY config flag, further filter (but keep non-targets + review)
            if getattr(cfg, 'TARGET_SPECIES_ONLY', 0):
                selected_points = [
                    p for p in selected_points
                    if p.get('species', p.get('species_code', '')) in TARGET_SPECIES
                    or _is_non_target(p) or _is_review(p)
                ]
        else:
            selected_points = list(img_data.get('points', []))

        if not selected_points:
            continue

        # Resolve raw image path under the flat raw dir
        raw_rel = img_data.get('raw', img_data.get('raw_image', ''))
        raw_path = os.path.join(raw_dir, os.path.basename(raw_rel)) if raw_rel else None
        if not raw_path or not os.path.exists(raw_path):
            raw_path = os.path.join(raw_dir, img_filename)
        if not os.path.exists(raw_path):
            log(f"WARNING: Raw image not found for {img_filename}")
            continue

        if getattr(cfg, 'MANUAL_ANNOTATE', False):
            # Combined annotator: no SAM3 pass. Build an empty-mask seg_dict
            # carrying the FULL prompt-point set as read-only reference. The
            # `already_processed` skip at the top of this loop handles resume
            # (existing seg_dicts with hand-drawn masks are never overwritten).
            img_h, img_w = 0, 0
            _img = cv2.imread(raw_path)
            if _img is not None:
                img_h, img_w = _img.shape[:2]
            seg_dict = {
                'image_path': raw_rel or img_filename,
                'image_path_abs': raw_path,
                'image_width': img_w,
                'image_height': img_h,
                'masks': [],
                'reference_points': img_data.get('points', []),
                'processed_at': datetime.now().isoformat(),
                'reviewed': False,
                'exported': False,
            }
            session['segmentations'][img_filename] = seg_dict
            continue

        queue.append({
            'filename': img_filename,
            'image_path': raw_path,
            'points': selected_points,
        })

    session['processing_queue'] = queue
    session['processing_idx'] = 0

    # Custom imports (operator-supplied images) ride alongside routed frames:
    # backfill any manifest entry the segmentations store does not carry yet.
    _merge_custom_imports()

    if getattr(cfg, 'MANUAL_ANNOTATE', False) or not queue:
        session['phase'] = 'review'
        _build_review_list()
    else:
        session['phase'] = 'processing'

    # Log config
    config_log = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': input_dir_abs,
        'export_dir': session['export_dir'],
        'categories': categories,
        'confidence_threshold': cfg.CONFIDENCE_THRESHOLD,
        'min_mask_area': cfg.MIN_MASK_AREA_PX,
        'merge_distance': cfg.MERGE_DISTANCE_PX,
        'overlap_strategy': cfg.OVERLAP_STRATEGY,
    }
    config_path = os.path.join(session['export_dir'], 'config_log.json')
    with open(config_path, 'w') as f:
        json.dump(config_log, f, indent=2)

    session['configured'] = True

    log(f"Configured: {len(prompts)} images total, {len(queue)} to process, "
        f"{skipped} already done, categories={categories}")

    return jsonify({
        'images': len(prompts),
        'to_process': len(queue),
        'already_processed': skipped,
        'phase': session['phase'],
        'review_count': len(session.get('review_files', [])),
        'categories': categories,
    })


@app.route('/api/resume', methods=['POST'])
def resume():
    """Resume from existing segmentation output."""
    data = request.json
    session['export_dir'] = os.path.abspath(data['export_dir'])
    session['review_batch_size'] = data.get('review_batch_size', cfg.REVIEW_BATCH_SIZE)
    session['shuffle'] = bool(data.get('shuffle', False))
    session['reviewer'] = (data.get('reviewer') or cfg.REVIEWER)
    session['review_dir'] = (data.get('review_dir') or cfg.REVIEW_DIR)
    _rc = data.get('contacts', cfg.REVIEW_CONTACTS)
    if isinstance(_rc, str):
        _rc = _rc.split(',')
    session['contacts'] = [c.strip() for c in (_rc or []) if c and c.strip()]

    export_dir = session['export_dir']
    if not os.path.isdir(export_dir):
        return jsonify({'error': 'Export directory not found'}), 404

    # Load class map
    session['class_map'] = load_class_map(export_dir)

    # Load flat segmentations (Task 2 will extend this to handle year-nested
    # legacy exports; for now load from the flat path only)
    if not os.path.exists(_seg_path(export_dir)):
        return jsonify({'error': 'No segmentation data found'}), 404

    session['segmentations'] = load_segmentations(export_dir)
    total = len(session['segmentations'])

    # Backfill custom imports missing from the loaded store (normally none:
    # imports are flushed into segmentations.json the moment they land).
    _merge_custom_imports()

    session['phase'] = 'review'
    session['processing_queue'] = []
    _build_review_list()
    session['configured'] = True

    return jsonify({
        'total_processed': total,
        'phase': 'review',
        'review_count': len(session['review_files']),
        'total_classes': len(session['class_map']),
    })


@app.route('/api/import_images', methods=['POST'])
def import_images():
    """Bring ANY custom image(s) into the active run's annotation queue.

    Accepts multipart uploads (field 'images', repeatable) or JSON
    {'paths': [server paths]}. Each image is copied verbatim into
    custom_imports/originals/, delivered (lores-downscaled when its long edge
    exceeds 1920px, same rule as pipeline frames) into custom_imports/raw/,
    manifested, added to the segmentations store + review queue, and recorded
    in the run's project.json (access_log + custom_imports). See
    custom_imports.py for the full contract."""
    export_dir = session.get('export_dir')
    if not export_dir:
        return jsonify({'error': 'No active session yet: configure or resume '
                                 'a project before importing images.'}), 400

    items = []
    if request.files:
        for fs in request.files.getlist('images'):
            if fs and fs.filename:
                items.append((fs.filename, fs.read()))
    else:
        data = request.get_json(silent=True) or {}
        for p in data.get('paths', []):
            p = str(p)
            items.append((os.path.basename(p), p))
    if not items:
        return jsonify({'error': 'No images supplied. Send multipart field '
                                 '"images" or JSON {"paths": [...]}.'}), 400

    initials = (session.get('reviewer') or cfg.REVIEWER or 'LO')
    entries, skipped = ci.ingest_files(
        export_dir, items, taken_names=set(session['segmentations'].keys()),
        reviewer=initials, log=log)

    recorded = False
    if entries:
        for e in entries:
            session['segmentations'][e['filename']] = ci.seg_dict_for(export_dir, e)
        flush_all()
        _build_review_list()
        recorded = ci.record_in_project(export_dir, entries, initials=initials)
        log(f"[import] {len(entries)} custom image(s) added to the review "
            f"queue ({len(skipped)} skipped); project.json "
            f"{'updated' if recorded else 'not found (standalone session)'}")

    return jsonify({
        'ok': bool(entries),
        'imported': [e['filename'] for e in entries],
        'skipped': [{'name': n, 'reason': r} for n, r in skipped],
        'project_recorded': recorded,
        'review_count': len(session.get('review_files', [])),
    })


def _process_single_image(engine, item, export_dir):
    """Segment one image. Returns (filename, n_masks, n_points).

    Prompt policy:
      - Each target point (species ∈ TARGET_SPECIES) -> positive SAM3 click
        that anchors a mask for that species.
      - Non-target points (right-click reviewer markers from step 4 —
        species='?' or non_target=True) -> negative SAM3 clicks that refine
        the target masks by telling SAM3 "don't include these regions".
        They DO NOT produce their own masks.
    """
    filename = item['filename']
    image_path = item['image_path']
    points = item['points']

    img_w, img_h = engine.set_image(image_path)

    def _is_negative(p):
        # REVIEW always wins: a point flagged for expert review must produce its
        # OWN positive mask, even if it also carries non_target/'?' (precedence).
        if _is_review(p):
            return False
        return p.get('non_target') is True \
            or p.get('species', p.get('species_code', '')) == '?'

    negatives = [p for p in points if _is_negative(p)]
    targets = [p for p in points if not _is_negative(p)]
    neg_clicks = [(p['x'], p['y']) for p in negatives]

    raw_masks = []
    for pt in targets:
        if neg_clicks:
            # Multi-point prompt: [positive, *negatives] with labels [1, 0…0].
            clicks = [(pt['x'], pt['y'])] + neg_clicks
            labels = [1] + [0] * len(neg_clicks)
            result = engine.refine_mask(clicks, labels)
        else:
            result = engine.segment_point(pt['x'], pt['y'], label=1)
        if result is not None:
            mask_dict = build_mask_dict(
                mask_id=len(raw_masks),
                binary_mask=result['mask'],
                score=result['score'],
                point_info=pt,
                source_type='auto',
                simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON,
            )
            if mask_dict is not None:
                # REVIEW prompts produce a real mask, but it is flagged so YOLO
                # export skips it and the batch export ships it to the expert.
                if _is_review(pt):
                    mask_dict['review'] = True
                    mask_dict['species'] = 'REVIEW'
                    mask_dict['category'] = 'Review'
                mask_dict['mask'] = result['mask']
                raw_masks.append(mask_dict)

    if raw_masks:
        raw_masks = resolve_overlaps(
            raw_masks, strategy=cfg.OVERLAP_STRATEGY,
            min_area=cfg.MIN_MASK_AREA_PX, thin_ratio=cfg.THIN_MASK_RATIO,
        )
        raw_masks = merge_close_masks(
            raw_masks, distance_px=cfg.MERGE_DISTANCE_PX, same_species_only=True,
        )

    final_masks = []
    for i, m in enumerate(raw_masks):
        binary = m.pop('mask', None)
        if binary is not None:
            m = update_mask_geometry(m, binary, cfg.POLYGON_SIMPLIFY_EPSILON)
        if m is not None:
            m['id'] = i
            final_masks.append(m)

    seg_dict = {
        'image_path': item.get('raw_rel', filename),
        'image_path_abs': image_path,
        'image_width': img_w,
        'image_height': img_h,
        'masks': final_masks,
        'processed_at': datetime.now().isoformat(),
        'reviewed': False,
        'exported': False,
    }

    session['segmentations'][filename] = seg_dict
    render_segmentation_overlays(seg_dict, filename, export_dir, stage="auto")

    return filename, len(final_masks), len(points)


# Background processing thread
_processing_lock = threading.Lock()

def _background_process_all():
    """Process entire queue in background. Browser can disconnect and reconnect."""
    engine = session.get('sam_engine')
    if engine is None:
        log("ERROR: SAM3 engine not loaded, cannot process")
        return

    queue = session['processing_queue']
    export_dir = session['export_dir']
    total = len(queue)

    log(f"Background processing started: {total} images")

    for idx in range(session['processing_idx'], total):
        if session.get('_stop_processing'):
            log(f"Processing stopped at {idx}/{total}")
            break

        item = queue[idx]
        with _processing_lock:
            try:
                fn, n_masks, n_pts = _process_single_image(engine, item, export_dir)
                log(f"SAM3 {idx + 1}/{total}: {fn} ({n_pts} pts -> {n_masks} masks)")
                # Persist segmentations after EVERY image so an orchestrator
                # restart (or crash, or ctrl+C) can't rewind progress.
                # Cost: one ~KB JSON rewrite per image — cheap.
                try:
                    save_segmentations(export_dir, session['segmentations'])
                except Exception as flush_err:
                    log(f"flush failed after {fn}: {flush_err}")
            except Exception as e:
                log(f"ERROR processing {item['filename']}: {e}")

        session['processing_idx'] = idx + 1

    flush_all()

    # Clear engine's cached image so review forces a fresh set_image
    if engine is not None:
        engine._current_image_path = None
        engine._current_image = None
    session['current_review_image'] = None

    session['phase'] = 'review'
    _build_review_list()
    log(f"Background processing complete: {session['processing_idx']}/{total}")


@app.route('/api/process', methods=['POST'])
def process_start():
    """Start background processing of all images. Returns immediately."""
    if session.get('_processing_thread') and session['_processing_thread'].is_alive():
        return jsonify({
            'already_running': True,
            'processed': session['processing_idx'],
            'total': len(session['processing_queue']),
        })

    session['_stop_processing'] = False
    t = threading.Thread(target=_background_process_all, daemon=True)
    session['_processing_thread'] = t
    t.start()

    return jsonify({
        'started': True,
        'total': len(session['processing_queue']),
    })


@app.route('/api/process_status')
def process_status():
    """Poll processing progress."""
    queue = session['processing_queue']
    total = len(queue)
    done = session['processing_idx']
    is_running = (session.get('_processing_thread') and
                  session['_processing_thread'].is_alive())

    # Allow review even while processing is ongoing
    if done > 0 and not session.get('review_files'):
        _build_review_list()

    return jsonify({
        'processed': done,
        'total': total,
        'running': is_running,
        'done': done >= total and not is_running,
        'phase': session['phase'],
        'review_count': len(session.get('review_files', [])),
    })


@app.route('/api/flush', methods=['POST'])
def force_flush():
    flush_all()
    return jsonify({'ok': True})


@app.route('/api/review_batch')
def review_batch():
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']

    batch_files = files[offset:offset + batch_size]
    images = []
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            n_masks = len(seg.get('masks', []))
            n_pending = sum(1 for m in seg.get('masks', []) if m.get('status') == 'pending')
            n_accepted = sum(1 for m in seg.get('masks', []) if m.get('status') == 'accepted')
            images.append({
                'filename': fn,
                'n_masks': n_masks,
                'n_pending': n_pending,
                'n_accepted': n_accepted,
                'reviewed': seg.get('reviewed', False),
                'exported': seg.get('exported', False),
                'custom_import': seg.get('custom_import', False),
            })

    total_batches = (len(files) + batch_size - 1) // batch_size if files else 0
    current_batch = (offset // batch_size) + 1 if files else 0

    return jsonify({
        'images': images,
        'batch_number': current_batch,
        'total_batches': total_batches,
        'total_images': len(files),
        'offset': offset,
        'batch_size': batch_size,
    })


@app.route('/api/unlabeled')
def unlabeled_frames():
    """List every frame in the review queue that has a mask with no species
    assigned yet (species=='', accepted or pending, see _needs_species).

    Scoped to session['review_files'], the same frame set the review UI's
    Prev/Next navigation walks. A frame already exported or scrapped is not
    reachable from that navigation, so it is out of scope here too. This
    powers the client-side "Unlabeled only" filter and "next unlabeled" jump,
    which only ever move within the current review queue's frame list.
    """
    frames = []
    for fn in session.get('review_files', []):
        seg = _find_segmentation(fn)
        if not seg:
            continue
        if any(_needs_species(m) for m in seg.get('masks', [])):
            frames.append(fn)
    return jsonify({'frames': frames, 'count': len(frames)})


@app.route('/api/next_review_batch', methods=['POST'])
def next_review_batch():
    # Removed-frames model: do_export() drops exported frames from review_files
    # and resets review_offset to 0 (see _build_review_list). There is no
    # "advance past the current batch" any more; the rebuilt list IS the next
    # batch at offset 0. This route now just reports whether any frames remain;
    # `done` is true only when review_files is empty. The frontend's
    # nextReviewBatch() no longer relies on it (it reloads /api/review_batch
    # directly), but the route is kept as a non-contradictory status probe.
    files = session['review_files']
    return jsonify({
        'offset': 0,
        'total_images': len(files),
        'done': len(files) == 0,
    })


@app.route('/api/image/<path:filename>')
def get_image_data(filename):
    """Get all mask data for a specific image."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    # Pre-embed this image for fast refinement
    _ensure_image_embedded(filename)

    return jsonify({
        'masks': seg.get('masks', []),
        'image_width': seg.get('image_width', 0),
        'image_height': seg.get('image_height', 0),
        'reviewed': seg.get('reviewed', False),
        'exported': seg.get('exported', False),
        'custom_import': seg.get('custom_import', False),
        'reference_points': [
            {'x': p['x'], 'y': p['y'], 'label': p.get('label', ''),
             'species_code': p.get('species', '')}
            for p in seg.get('reference_points', [])
        ],
    })


def _needs_species(mask):
    """True when a mask has no real species classification yet.

    An empty species ('' or missing) means the operator hasn't labeled the
    mask. A REVIEW mask (species=='REVIEW') is an intentional label, not a
    missing one, so it is never caught here.
    """
    return (mask.get('species', '') or '') == ''


@app.route('/api/image/<path:filename>/masks', methods=['PUT'])
def update_masks(filename):
    """Bulk update mask statuses (accept/reject).

    Guard: a mask cannot be set to 'accepted' while it has no species
    assigned (empty string). REVIEW masks (species=='REVIEW') are exempt,
    since that's an intentional label, not a missing one. A refused mask stays at
    its prior status (or 'pending' if it had none) and its id is reported
    back in `needs_species` so the client can prompt for a label instead of
    silently dropping the accept.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json
    updates = data.get('updates', {})  # {mask_id: status}
    relabel = data.get('relabel', {})  # {mask_id: {species, review, category, name}} (optional)

    needs_species = []
    for mask in seg.get('masks', []):
        mid = str(mask['id'])
        if mid in relabel:
            info = relabel[mid] or {}
            if 'species' in info:
                mask['species'] = info['species']
            if 'review' in info:
                mask['review'] = bool(info['review'])
            if 'category' in info:
                mask['category'] = info['category']
            if 'name' in info:
                mask['name'] = info['name']
        if mid in updates:
            new_status = updates[mid]
            if new_status == 'accepted' and _needs_species(mask):
                needs_species.append(mask['id'])
                # Refused: leave status as-is (do not accept an unlabeled mask).
            else:
                mask['status'] = new_status

    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    out = {'ok': True}
    if needs_species:
        out['needs_species'] = needs_species
    return jsonify(out)


@app.route('/api/image/<path:filename>/accept_all', methods=['POST'])
def accept_all_masks(filename):
    """Accept all pending masks for an image.

    Guard: pending masks with no species assigned are skipped (left
    'pending') instead of being accepted unlabeled. Their ids are reported
    in `needs_species`. REVIEW masks are exempt (intentional label).
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    count = 0
    needs_species = []
    for mask in seg.get('masks', []):
        if mask.get('status') == 'pending':
            if _needs_species(mask):
                needs_species.append(mask['id'])
                continue
            mask['status'] = 'accepted'
            count += 1
    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    out = {'ok': True, 'accepted': count}
    if needs_species:
        out['needs_species'] = needs_species
    return jsonify(out)


def _commit_candidate_to_seg(seg, cand):
    """Promote an ephemeral scan candidate into seg['masks'] as an accepted mask.

    Scan candidates (Task 12) live only on the client until the operator keeps
    one (Task 13 Enter). Keeping it commits a COPY here: a fresh id (max existing
    + 1) so it never collides with a live mask, status='accepted' so it exports
    like any reviewed mask. Rejected candidates are dropped client-side and never
    reach this helper, so nothing dropped is ever persisted.
    """
    masks = seg.setdefault('masks', [])
    next_id = max((m['id'] for m in masks), default=-1) + 1
    mask = dict(cand)
    mask['id'] = next_id
    mask['status'] = 'accepted'
    masks.append(mask)
    return mask


@app.route('/api/image/<path:filename>/commit_mask', methods=['POST'])
def commit_mask(filename):
    """Keep a scan candidate: append it to seg['masks'] as an accepted mask.

    Guard: same as update_masks / accept_all_masks. A candidate with no
    species assigned (e.g. seeded from an unlabeled source mask via
    exemplar_scan(mode='this')) is refused here too, so this third accept
    path cannot bypass the label guard. The candidate is not committed and
    the response reports needs_species instead of a mask, so the client can
    prompt for a label rather than silently accepting it unlabeled.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    cand = (request.json or {}).get('mask')
    if not cand:
        return jsonify({'error': 'no mask'}), 400
    if _needs_species(cand):
        return jsonify({'ok': False, 'needs_species': True})
    mask = _commit_candidate_to_seg(seg, cand)
    seg['reviewed'] = True
    _stamp_provenance(seg, filename)
    return jsonify({'ok': True, 'mask': mask})


@app.route('/api/image/<path:filename>/refine', methods=['POST'])
def refine_mask(filename):
    """Refine a mask with an additional positive/negative click."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']
    click_x = data['click_x']
    click_y = data['click_y']
    click_label = data.get('click_label', 1)  # 1=positive, 0=negative

    # Find the mask
    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    # Ensure image is embedded
    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Build accumulated clicks: original source click + all refinement clicks + new
    clicks = [(target_mask['source_x'], target_mask['source_y'])]
    labels = [1]

    for rc in target_mask.get('refinement_clicks', []):
        clicks.append((rc['x'], rc['y']))
        labels.append(rc['label'])

    clicks.append((click_x, click_y))
    labels.append(click_label)

    # Run SAM3 refinement
    result = engine.refine_mask(clicks, labels)
    if result is None:
        return jsonify({'error': 'refinement produced no mask'}), 400

    # REFINE keeps the strict clip-to-neighbors rule on purpose — a positive
    # click on a mask shouldn't cause it to eat an accepted neighbor just
    # because SAM3 regrew into that region. User asked for this explicitly.
    # Rejected masks are always ignored (they're "deleted" in the UI), including
    # ones the client just deleted whose status PUT may not have landed yet.
    import numpy as np
    from mask_ops import rle_decode
    new_binary = np.asarray(result['mask'], dtype=bool)
    h, w = new_binary.shape
    rejected_ids = data.get('rejected_ids') or []
    other_masks = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                   if m.get('id') != mask_id]
    if other_masks:
        forbidden = np.zeros((h, w), dtype=bool)
        for m in other_masks:
            rle = m.get('rle')
            if not rle:
                continue
            try:
                forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
            except Exception:
                continue
        new_binary = new_binary & ~forbidden
        if not new_binary.any():
            blockers = _overlapping_blockers(np.asarray(result['mask'], dtype=bool), other_masks)
            names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'other masks'
            return jsonify({'error': f'refinement fully overlapped {names}', 'blocked_by': blockers}), 400

    # Update mask geometry
    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after refinement'}), 400
    _overlap_delta = {'updated': [], 'removed': []}

    # Record the refinement click
    target_mask['refinement_clicks'].append({
        'x': click_x, 'y': click_y, 'label': click_label,
    })
    target_mask['score'] = round(result['score'], 4)

    return jsonify({
        'ok': True,
        'mask': target_mask,
        'affected': _overlap_delta,
    })


@app.route('/api/image/<path:filename>/rebox', methods=['POST'])
def rebox_mask(filename):
    """Redraw a mask from a bounding box prompt."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']
    box = data['box']  # [x_min, y_min, x_max, y_max]

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    result = engine.segment_box(*box)
    if result is None:
        return jsonify({'error': 'box prompt produced no mask'}), 400

    # Strict clip-to-neighbors (same rule as /refine and /add). Rejected
    # masks are ignored so you can redraw cleanly over a deleted one, including
    # ones the client just deleted whose status PUT may not have landed yet.
    import numpy as np
    from mask_ops import rle_decode
    new_binary = np.asarray(result['mask'], dtype=bool)
    h, w = new_binary.shape
    rejected_ids = data.get('rejected_ids') or []
    other_masks = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                   if m.get('id') != mask_id]
    if other_masks:
        forbidden = np.zeros((h, w), dtype=bool)
        for m in other_masks:
            rle = m.get('rle')
            if not rle:
                continue
            try:
                forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
            except Exception:
                continue
        new_binary = new_binary & ~forbidden
        if not new_binary.any():
            blockers = _overlapping_blockers(np.asarray(result['mask'], dtype=bool), other_masks)
            names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'other masks'
            return jsonify({'error': f'rebox fully overlapped {names}', 'blocked_by': blockers}), 400

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after rebox'}), 400

    # Reset refinement history since we redrew from scratch
    target_mask['refinement_clicks'] = []
    target_mask['score'] = round(result['score'], 4)
    target_mask['source_type'] = 'manual_box'

    return jsonify({'ok': True, 'mask': target_mask})


@app.route('/api/image/<path:filename>/draw_edit', methods=['POST'])
def draw_edit(filename):
    """TagLab-style freehand edit on the selected mask.

    The client sends a path as [[x,y], ...] in image coordinates. We close
    the polygon (connecting end to start if the gesture was open), rasterize
    it, and decide add-vs-subtract by testing whether the polygon centroid
    falls inside the current mask:
      - centroid inside  -> subtract the polygon from the mask (carve chunk)
      - centroid outside -> union the polygon into the mask (add chunk)

    In both cases the result is then clipped against every live (non-rejected)
    neighbor via _clip_to_neighbors, so a draw edit can never overlap an
    already-made mask (existing masks win; same strict rule as /add and
    /refine). Neighbors are never modified by the edit.
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return jsonify({'error': 'cv2 not available — draw edit needs opencv'}), 500
    from mask_ops import rle_decode, update_mask_geometry

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json or {}
    mask_id = data.get('mask_id')
    raw_path = data.get('path') or []
    if not isinstance(raw_path, list) or len(raw_path) < 3:
        return jsonify({'error': 'path needs at least 3 points'}), 400

    target_mask = None
    for m in seg.get('masks', []):
        if m.get('id') == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    h = int(seg.get('image_height') or 0)
    w = int(seg.get('image_width') or 0)
    if not h or not w:
        return jsonify({'error': 'missing image dimensions'}), 400

    # Current mask as a binary array (preserve existing RLE if available).
    rle = target_mask.get('rle')
    if rle:
        current = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
    else:
        current = np.zeros((h, w), dtype=bool)

    # Rasterize the user-drawn polygon. If the gesture was open, OpenCV's
    # fillPoly auto-closes by connecting the last vertex to the first.
    poly_pts = np.asarray([[int(round(p[0])), int(round(p[1]))] for p in raw_path],
                          dtype=np.int32)
    poly_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_img, [poly_pts], 255)
    poly_binary = poly_img > 0
    if not poly_binary.any():
        return jsonify({'error': 'drawn polygon has no interior'}), 400

    # Centroid of the drawn polygon, clipped to image bounds.
    cx = int(np.clip(np.mean(poly_pts[:, 0]), 0, w - 1))
    cy = int(np.clip(np.mean(poly_pts[:, 1]), 0, h - 1))
    is_subtract = bool(current[cy, cx])

    if is_subtract:
        new_binary = current & ~poly_binary
    else:
        new_binary = current | poly_binary

    if not new_binary.any():
        return jsonify({'error': 'edit left an empty mask — try a different stroke'}), 400

    # Strict clip-to-neighbors: the edited mask is trimmed to stay off every
    # other live (non-rejected) mask, so a draw edit can never overlap an
    # already-made mask. Same rule as /add and /refine.
    rejected_ids = data.get('rejected_ids') or []
    blocking = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                if m.get('id') != mask_id]
    new_binary = _clip_to_neighbors(new_binary, blocking)
    if not new_binary.any():
        return jsonify({'error': 'edit fully overlapped other masks — try a different stroke'}), 400

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after draw edit'}), 400

    overlap_delta = {'updated': [], 'removed': []}  # neighbors are never modified now

    # Mark as manual and record for audit. Kept out of refinement_clicks so
    # the existing SAM3 undo path isn't confused by a non-point edit.
    target_mask['score'] = 1.0
    target_mask.setdefault('draw_edits', []).append({
        'op': 'subtract' if is_subtract else 'add',
        'n_points': len(raw_path),
    })

    return jsonify({
        'ok': True,
        'mask': target_mask,
        'op': 'subtract' if is_subtract else 'add',
        'affected': overlap_delta,
    })


@app.route('/api/image/<path:filename>/undo_refine', methods=['POST'])
def undo_refine(filename):
    """Undo the last refinement click on a mask."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']
    rejected_ids = data.get('rejected_ids') or []

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    clicks = target_mask.get('refinement_clicks', [])
    if not clicks:
        return jsonify({'error': 'nothing to undo'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Pop last click
    clicks.pop()
    target_mask['refinement_clicks'] = clicks

    # Re-run SAM3 with original point + remaining clicks
    all_clicks = [(target_mask['source_x'], target_mask['source_y'])]
    all_labels = [1]
    for rc in clicks:
        all_clicks.append((rc['x'], rc['y']))
        all_labels.append(rc['label'])

    result = engine.refine_mask(all_clicks, all_labels)
    if result is None:
        return jsonify({'error': 'undo produced no mask'}), 400

    # Same subtraction rule as /refine — the re-run can still re-grow into
    # neighboring masks.
    import numpy as np
    from mask_ops import rle_decode
    new_binary = np.asarray(result['mask'], dtype=bool)
    h, w = new_binary.shape
    # Exclude the mask being edited AND any rejected masks — rejected masks
    # are "deleted" in the user's mental model, so they shouldn't block a
    # refined neighbor from spilling into that region. Reuse _blocking_masks so
    # client-side rejected_ids (a just-deleted mask whose 'rejected' PUT has not
    # landed yet) are honored too, closing the same race the /add fix closes.
    other_masks = [m for m in _blocking_masks(seg.get('masks', []), rejected_ids)
                   if m.get('id') != mask_id]
    if other_masks:
        forbidden = np.zeros((h, w), dtype=bool)
        for m in other_masks:
            rle = m.get('rle')
            if not rle:
                continue
            try:
                forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
            except Exception:
                continue
        new_binary = new_binary & ~forbidden

    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after undo'}), 400

    target_mask['score'] = round(result['score'], 4)
    return jsonify({'ok': True, 'mask': target_mask, 'remaining_undos': len(clicks)})


@app.route('/api/image/<path:filename>/reset_mask', methods=['POST'])
def reset_mask(filename):
    """Reset a mask to its original SAM3 segmentation (discard all refinements)."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mask_id = data['mask_id']

    target_mask = None
    for m in seg.get('masks', []):
        if m['id'] == mask_id:
            target_mask = m
            break
    if target_mask is None:
        return jsonify({'error': 'mask not found'}), 404

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    # Re-run SAM3 with just the original source point
    result = engine.segment_point(target_mask['source_x'], target_mask['source_y'], label=1)
    if result is None:
        return jsonify({'error': 'reset produced no mask'}), 400

    updated = update_mask_geometry(
        target_mask, result['mask'], cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after reset'}), 400

    target_mask['refinement_clicks'] = []
    target_mask['score'] = round(result['score'], 4)
    target_mask['source_type'] = 'auto'
    return jsonify({'ok': True, 'mask': target_mask})


@app.route('/api/image/<path:filename>/merge', methods=['POST'])
def merge_masks(filename):
    """Merge two masks into one."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json
    id_a = data['mask_id_a']
    id_b = data['mask_id_b']

    mask_a = mask_b = None
    for m in seg.get('masks', []):
        if m['id'] == id_a:
            mask_a = m
        elif m['id'] == id_b:
            mask_b = m

    if mask_a is None or mask_b is None:
        return jsonify({'error': 'one or both masks not found'}), 404

    # Reconstruct binary masks from RLE
    h, w = seg['image_height'], seg['image_width']
    bin_a = rle_decode(mask_a['rle'], (h, w))
    bin_b = rle_decode(mask_b['rle'], (h, w))

    # Union
    merged = union_masks(bin_a, bin_b)

    # Update mask_a with merged geometry
    updated = update_mask_geometry(mask_a, merged, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'merge produced empty mask'}), 400

    # Remove mask_b from the list
    seg['masks'] = [m for m in seg['masks'] if m['id'] != id_b]

    return jsonify({'ok': True, 'mask': mask_a, 'removed_id': id_b})


@app.route('/api/image/<path:filename>/merge_same_id', methods=['POST'])
def merge_same_id(filename):
    """Union OVERLAPPING masks that share the same species/label into one.

    On-demand, because segmentation-time overlap resolution uses
    OVERLAP_STRATEGY='larger_wins', which DROPS the smaller of two overlapping
    same-species masks rather than unioning them — so two SAM3 clicks on one
    colony can leave the colony split across masks (one trimmed away). This
    endpoint rebuilds the current frame's masks by unioning every overlapping
    same-species group via mask_ops.merge_overlapping_same_id (pure RLE math,
    no SAM3 engine needed) and reports how many masks were merged away.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    h = int(seg.get('image_height') or 0)
    w = int(seg.get('image_width') or 0)
    if not h or not w:
        return jsonify({'error': 'missing image dimensions'}), 400

    masks = seg.get('masks', [])
    new_masks, merged, refused = merge_overlapping_same_id(
        masks, h, w, simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON)
    seg['masks'] = new_masks
    return jsonify({
        'ok': True,
        'merged': merged,
        'refused': refused,
        'masks': new_masks,
    })


# Mask fields the operator owns locally — a disk reload must NOT clobber these
# (they hold unsaved accept/reject decisions and geometry edits).
_OPERATOR_OWNED_FIELDS = (
    'status', 'rle', 'polygon_px', 'polygon_norm', 'bbox', 'area',
    'refinement_clicks', 'draw_edits', 'source_type', 'source_x', 'source_y',
)
# Imported review fields a disk reload MAY merge in (expert pipeline output).
_IMPORTED_REVIEW_FIELDS = ('reviews', 'expert_id', 'review_uid', 'review')


def _merge_imported_review_fields(current_masks, disk_masks):
    """Merge expert-review fields from a freshly-read disk copy onto the
    operator's in-memory masks WITHOUT clobbering unsaved accept/reject or
    geometry edits.

    Matching key: review_uid first (stable across the expert round-trip), then
    id. For each matched mask we copy the imported review fields
    (reviews[], expert_id, review_uid, review) and — only when an expert
    actually accepted an ID (expert_id.mode == 'EXPERT') — the accepted species
    /name/category. Operator-owned fields (status, geometry, refinement) are
    left exactly as the operator has them in memory.

    Returns (merged_count, expert_count): masks that gained imported fields,
    and of those how many carry an expert-accepted ID.
    """
    def _key_index(masks):
        by_uid, by_id = {}, {}
        for m in masks:
            uid = m.get('review_uid')
            if uid:
                by_uid[uid] = m
            if m.get('id') is not None:
                by_id[m.get('id')] = m
        return by_uid, by_id

    disk_by_uid, disk_by_id = _key_index(disk_masks)

    merged_count = 0
    expert_count = 0
    for cm in current_masks:
        dm = None
        uid = cm.get('review_uid')
        if uid and uid in disk_by_uid:
            dm = disk_by_uid[uid]
        elif cm.get('id') in disk_by_id:
            dm = disk_by_id[cm.get('id')]
        if dm is None:
            continue

        touched = False
        for f in _IMPORTED_REVIEW_FIELDS:
            if f in dm and dm.get(f) != cm.get(f):
                cm[f] = dm[f]
                touched = True

        eid = dm.get('expert_id')
        if eid and isinstance(eid, dict) and eid.get('mode') == 'EXPERT':
            # An expert accepted an ID — promote the accepted species/name/cat
            # so step-5 colours it expert-green. This is review metadata, not an
            # operator geometry edit, so it's safe to merge.
            for f in ('species', 'name', 'category'):
                if f in dm and dm.get(f) != cm.get(f):
                    cm[f] = dm[f]
                    touched = True
            expert_count += 1
        if touched:
            merged_count += 1

    return merged_count, expert_count


@app.route('/api/reload_segmentations', methods=['POST'])
def reload_segmentations():
    """Re-read the current image's segmentations.json from disk and MERGE IN the
    imported review/expert fields (reviews[], expert_id, species-when-accepted)
    onto the in-memory masks by review_uid/id — WITHOUT clobbering the
    operator's unsaved accept/reject or geometry edits.

    Lets imported expert/pending colours appear without restarting the app:
    after the importer (or a teammate) writes accepted IDs into the on-disk
    segmentations.json, the operator hits "Reload IDs" and the freshly-confirmed
    masks light up green / pending-amber in place.
    """
    data = request.json or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'filename required'}), 400

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    export_dir = session.get('export_dir')
    if not export_dir:
        return jsonify({'error': 'no export dir configured'}), 400

    disk_segs = load_segmentations(export_dir)
    disk_seg = disk_segs.get(filename)
    if not disk_seg:
        return jsonify({'error': 'no on-disk segmentation for this frame'}), 404

    merged_count, expert_count = _merge_imported_review_fields(
        seg.get('masks', []), disk_seg.get('masks', []))

    return jsonify({
        'ok': True,
        'merged': merged_count,
        'expert': expert_count,
        'masks': seg.get('masks', []),
    })


def _orchestrator_url():
    """The orchestrator base URL the 'Done' button can return to, if launched
    from the pipeline orchestrator (else '')."""
    return os.environ.get('TCRMP_ORCHESTRATOR_URL', '').rstrip('/')


def _build_exclusion_mask(seg, margin_px=20):
    """Build a binary mask of all accepted/pending regions + a margin buffer.
    This prevents exemplar candidates from being thin borders around existing masks."""
    h, w = seg['image_height'], seg['image_width']
    occupied = np.zeros((h, w), dtype=np.uint8)
    for m in seg.get('masks', []):
        if m.get('status') != 'rejected' and 'rle' in m:
            occupied |= rle_decode(m['rle'], (h, w)).astype(np.uint8)
    # Dilate to add margin — prevents thin halo masks
    if margin_px > 0 and occupied.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px * 2, margin_px * 2))
        occupied = cv2.dilate(occupied, kernel)
    return occupied.astype(bool)


def _filter_exemplar_results(results, seg, species, category, threshold):
    """Filter exemplar scan results: no overlap, no halos, score >= threshold.
    Returns list of mask dicts, sorted by score descending."""
    from mask_ops import clean_mask
    h, w = seg['image_height'], seg['image_width']
    exclusion = _build_exclusion_mask(seg, margin_px=15)

    next_id = max((m['id'] for m in seg.get('masks', [])), default=-1) + 1
    candidates = []

    for r in results:
        if r['score'] < threshold:
            continue
        binary = r['mask']
        # Resize mask to image dimensions if needed (SAM3 may return at model resolution)
        if binary.shape != (h, w):
            binary = cv2.resize(binary.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        # Subtract exclusion zone (existing masks + margin)
        binary = binary & ~exclusion
        binary = clean_mask(binary, min_fragment=cfg.MIN_MASK_AREA_PX).astype(bool)
        if binary.sum() < cfg.MIN_MASK_AREA_PX:
            continue

        point_info = {
            'label': f"E{next_id}",
            'species': species,
            'name': species,
            'category': category,
            'x': float(np.mean(np.where(binary)[1])),
            'y': float(np.mean(np.where(binary)[0])),
        }

        mask_dict = build_mask_dict(
            mask_id=next_id,
            binary_mask=binary,
            score=r['score'],
            point_info=point_info,
            source_type='exemplar',
            simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON,
        )
        if mask_dict is not None:
            candidates.append((r['score'], mask_dict, binary))
            next_id += 1

    # Sort by score descending, remove inter-candidate overlaps (keep higher score)
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept = []
    kept_union = np.zeros((h, w), dtype=bool)
    for score, md, binary in candidates:
        overlap = (binary & kept_union).sum()
        if overlap > binary.sum() * 0.3:
            continue  # too much overlap with a higher-scored candidate
        kept.append(md)
        kept_union |= binary

    return kept


class _StubExemplarEngine:
    """TEST-ONLY fake engine. Returns ``n`` synthetic candidate masks so the
    no-persist contract test can run ``_exemplar_candidates`` headlessly without
    loading SAM3. NOT used by the running app — the route always passes the real
    ``session['sam_engine']``. Kept here (not in tests/) so the pure helper has a
    drop-in stand-in with the exact ``exemplar_scan(bbox) -> [{mask, score, ...}]``
    contract.
    """

    def __init__(self, n=2):
        self.n = n

    def exemplar_scan(self, bbox):  # pragma: no cover - exercised only by tests
        # Emit n disjoint rectangles with high scores. Each is well over
        # MIN_MASK_AREA_PX and sits at x>=28 (clear of a seed mask in the
        # top-left + its exclusion margin), in distinct y-bands, so
        # _filter_exemplar_results keeps every one rather than subtracting or
        # de-duping them away. Assumes the test image is 48x64.
        out = []
        for i in range(self.n):
            y0 = 6 + i * 20
            binary = np.zeros((48, 64), dtype=bool)
            binary[y0:y0 + 16, 30:62] = True   # 16 x 32 = 512 px
            out.append({'mask': binary, 'score': 0.95 - i * 0.01, 'bbox': None})
        return out


def _exemplar_candidates(seg, mode, threshold, mask_id, engine):
    """Pure candidate finder for the exemplar scan — RETURNS a list of candidate
    mask dicts and NEVER mutates ``seg['masks']``.

    This holds the exact candidate-finding math the old route ran inline; the
    only change is WHERE results accumulate. To preserve the two behaviours the
    old code got from ``seg['masks'].extend(...)`` between species —
      (1) later species treat earlier candidates as occupied (exclusion), and
      (2) ids stay unique across species (``next_id = max(id)+1``) —
    we accumulate into a *working copy* of the mask list (``work_seg``) and pass
    that to ``_filter_exemplar_results``. The caller's ``seg`` is untouched, so
    candidates are ephemeral until the client decides to keep them (Task 13).

    Returns a flat list of candidate mask dicts (the route derives the
    ``species_scanned`` summary itself from ``seg``).
    """
    # Shallow working seg: same dims, a *copy* of the masks list (the dicts are
    # shared, but we only ever read them or append our own new dicts — never edit
    # the originals — so the real seg['masks'] list is never extended).
    work_seg = {
        'image_height': seg['image_height'],
        'image_width': seg['image_width'],
        'masks': list(seg.get('masks', [])),
    }

    all_new_masks = []

    if mode == 'this':
        source_mask = None
        for m in seg.get('masks', []):
            if m['id'] == mask_id:
                source_mask = m
                break
        if source_mask is None:
            raise LookupError('mask not found')
        bbox = source_mask.get('bbox')
        if not bbox:
            raise ValueError('mask has no bounding box')

        results = engine.exemplar_scan(bbox)
        all_new_masks = _filter_exemplar_results(
            results, work_seg,
            source_mask.get('species', 'UNK'),
            source_mask.get('category', ''),
            threshold,
        )

    elif mode == 'all':
        # Scan using every unique accepted/pending species as exemplar.
        for source in _exemplar_sources_all(seg):
            results = engine.exemplar_scan(source['bbox'])
            new_masks = _filter_exemplar_results(
                results, work_seg,
                source.get('species', 'UNK'),
                source.get('category', ''),
                threshold,
            )
            # Accumulate into the WORKING copy only — so the next species sees
            # these as occupied and gets fresh ids. seg['masks'] stays untouched.
            work_seg['masks'].extend(new_masks)
            all_new_masks.extend(new_masks)

    return all_new_masks


def _exemplar_sources_all(seg):
    """The unique accepted/pending masks (one per species, first seen) used as
    exemplars for a mode='all' scan. Shared by the helper and the route's
    species_scanned summary so they can never disagree."""
    seen_species = set()
    sources = []
    for m in seg.get('masks', []):
        if m.get('status') not in ('accepted', 'pending'):
            continue
        sp = m.get('species', '')
        if sp and sp not in seen_species and m.get('bbox'):
            seen_species.add(sp)
            sources.append(m)
    return sources


@app.route('/api/image/<path:filename>/exemplar', methods=['POST'])
def exemplar_scan(filename):
    """Scan for similar regions. mode='this' uses one mask, mode='all' uses all
    accepted masks. Candidates are RETURNED as new_masks and are NOT persisted to
    seg['masks'] — the client holds them ephemerally in a review panel (Task 12)."""
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mode = data.get('mode', 'this')  # 'this' or 'all'
    threshold = data.get('threshold', 0.4)
    mask_id = data.get('mask_id')

    if mode == 'this' and mask_id is None:
        return jsonify({'error': 'mask_id required for mode=this'}), 400

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    try:
        new_masks = _exemplar_candidates(seg, mode, threshold, mask_id, engine)
    except LookupError:
        return jsonify({'error': 'mask not found'}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    species_scanned = ([s.get('species', '') for s in _exemplar_sources_all(seg)]
                       if mode == 'all' else [])

    return jsonify({
        'ok': True,
        'count': len(new_masks),
        'new_masks': new_masks,
        'species_scanned': sorted(species_scanned),
    })


@app.route('/api/image/<path:filename>/scrap', methods=['POST'])
def scrap_segmentation(filename):
    """Mark a frame as scrapped — excluded from review and export.
    Mirrors the step 4 scrap API so the two apps behave the same way.
    """
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = True
    seg['reviewed'] = True
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


@app.route('/api/image/<path:filename>/unscrap', methods=['POST'])
def unscrap_segmentation(filename):
    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = False
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


def _blocking_masks(masks, rejected_ids=None):
    """Masks that a new or edited mask must clip against: the live (non-rejected)
    ones, minus any the client has just deleted (rejected_ids). The client sends
    its authoritative deleted set so a fast delete -> add/refine is NOT clipped by
    a mask whose 'rejected' status PUT has not landed yet. Flask runs threaded, so
    the reject PUT and the add/refine POST can race; without this, the just-deleted
    mask would intermittently block a new overlapping mask in its location.
    Backend status='rejected' is honored too.
    """
    rej = set(rejected_ids or [])
    return [m for m in (masks or [])
            if m.get('status') != 'rejected' and m.get('id') not in rej]


def _clip_to_neighbors(new_binary, blocking):
    """Subtract every blocking mask's footprint from new_binary so a new or
    edited mask stays on its own side of live neighbors (existing masks win).
    Same rule /add and /refine use. `blocking` is the output of _blocking_masks.
    """
    import numpy as np
    from mask_ops import rle_decode
    if not blocking:
        return new_binary
    h, w = new_binary.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for m in blocking:
        rle = m.get('rle')
        if not rle:
            continue
        try:
            forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            continue
    return new_binary & ~forbidden


def _overlapping_blockers(new_binary, blocking):
    """Which blocking masks actually intersect new_binary — for user-facing
    'blocked by mask #N' messages. `blocking` is the output of _blocking_masks.
    Returns [{'id','species','label'}, ...] ordered by descending overlap area.
    """
    import numpy as np
    from mask_ops import rle_decode
    h, w = new_binary.shape
    hits = []
    for m in blocking:
        rle = m.get('rle')
        if not rle:
            continue
        try:
            b = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            continue
        inter = int((new_binary & b).sum())
        if inter > 0:
            hits.append((inter, {'id': m.get('id'),
                                 'species': m.get('species', ''),
                                 'label': m.get('label', '')}))
    hits.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in hits]


@app.route('/api/image/<path:filename>/add', methods=['POST'])
def add_mask(filename):
    """Add a new mask at a click point (SAM3 point prompt).

    Edited-wins policy: the new mask claims whatever region SAM3 gives it.
    Any live (non-rejected) neighbor that overlaps yields that region —
    its RLE + polygon get re-computed without the new mask's footprint.
    Rejected masks are untouched (they're "deleted" in the user's mental
    model, so they can't block a new mask either).
    """
    import numpy as np

    seg = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    click_x = data['click_x']
    click_y = data['click_y']
    species = data.get('species', '')
    category = data.get('category', '')
    label = data.get('label', '?')
    is_review = bool(data.get('review'))

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    result = engine.segment_point(click_x, click_y, label=1)
    if result is None:
        return jsonify({'error': 'no mask generated at this point'}), 400

    new_binary = np.asarray(result['mask'], dtype=bool)
    if not new_binary.any():
        return jsonify({'error': 'no mask generated at this point'}), 400

    # Strict clip-to-neighbors: the new mask gets trimmed to stay on its side
    # of any LIVE (non-rejected) neighbor. Rejected masks are "deleted" in the
    # user's mental model and do NOT block a new overlapping mask.
    existing_masks = seg.get('masks', [])
    rejected_ids = data.get('rejected_ids') or []
    blocking = _blocking_masks(existing_masks, rejected_ids)
    if blocking:
        from mask_ops import rle_decode
        h, w = new_binary.shape
        forbidden = np.zeros((h, w), dtype=bool)
        for m in blocking:
            rle = m.get('rle')
            if not rle:
                continue
            try:
                forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
            except Exception:
                continue
        pre_clip = np.asarray(result['mask'], dtype=bool)
        new_binary = new_binary & ~forbidden
        if not new_binary.any():
            blockers = _overlapping_blockers(pre_clip, blocking)
            names = ', '.join(f"#{b['id']} ({b['species'] or '?'})" for b in blockers) or 'existing masks'
            return jsonify({
                'error': f'new mask fully overlaps {names} — delete it or click a clearer spot',
                'blocked_by': blockers,
            }), 400

    next_id = max((m['id'] for m in existing_masks), default=-1) + 1
    point_info = {
        'label': label, 'species': species, 'name': species,
        'category': category, 'x': click_x, 'y': click_y,
    }

    mask_dict = build_mask_dict(
        mask_id=next_id,
        binary_mask=new_binary,
        score=result['score'],
        point_info=point_info,
        source_type='manual_click',
        simplify_epsilon=cfg.POLYGON_SIMPLIFY_EPSILON,
    )
    if mask_dict is None:
        return jsonify({'error': 'mask too small or empty after clip'}), 400

    # REVIEW arming: flag the mask so YOLO export skips it and the batch export
    # ships it to the expert repo (parity with _process_single_image). The
    # source_type stays manual_click — a human placed it.
    if is_review:
        mask_dict['review'] = True
        mask_dict['species'] = 'REVIEW'
        mask_dict['category'] = 'Review'

    seg['masks'].append(mask_dict)
    return jsonify({'ok': True, 'mask': mask_dict})


def _derive_project_id(export_dir):
    parts = os.path.abspath(export_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith('run_'):
            return p
    return os.path.basename(os.path.dirname(os.path.abspath(export_dir))) or 'project'


def _stamp_provenance(seg, filename):
    """Compute per-(image,label) outcomes for a finished/exported frame and
    upsert the flat ledger. Stamps seg['label_outcomes'] in place. The CSV
    write is wrapped non-fatally so a provenance hiccup never blocks review or
    export. Persistence of the block rides on the existing flush.
    """
    export_dir = session['export_dir']
    outcomes = compute_label_outcomes(
        seg, TARGET_SPECIES, source=cfg.PROVENANCE_SOURCE,
        reviewer=session.get('reviewer', cfg.REVIEWER))
    seg['label_outcomes'] = outcomes
    try:
        write_provenance_csv(
            export_dir,
            os.path.splitext(filename)[0],
            outcomes,
            cfg.PROJECT_ID or _derive_project_id(export_dir))
    except Exception as e:
        log(f"[provenance] csv write failed for {filename} (non-fatal): {e}")


def _export_review_bundle(to_export, batch_files):
    """Ship this batch's REVIEW-flagged masks to the expert-review repo +
    library, and auto-relabel any that a past expert already identified.
    Best-effort: any failure is logged and never breaks the YOLO export."""
    if not _REVIEW_OK:
        # Surface the disabled state instead of dropping REVIEW masks silently:
        # if any frame in this batch carries a REVIEW-flagged mask, the operator
        # must know it did not ship to the expert repo.
        n_review = 0
        for fn in batch_files:
            seg = to_export.get(fn) or {}
            for m in seg.get('masks', []):
                if m.get('review') or m.get('species') == 'REVIEW':
                    n_review += 1
        if n_review:
            log(f"[review] WARNING: expert-review export DISABLED (_reefreview "
                f"import failed) but {n_review} REVIEW mask(s) in this batch were "
                f"NOT shipped. Fix the _reefreview import and re-export.")
        else:
            log("[review] expert-review export disabled (_reefreview import "
                "failed); no REVIEW masks in this batch.")
        return {'disabled': True, 'review_masks_skipped': n_review}
    try:
        export_dir = session['export_dir']
        contacts = session.get('contacts') or [
            c.strip() for c in (cfg.REVIEW_CONTACTS or '').split(',') if c.strip()]
        # Canonical identity from the orchestrator (project.json); fall back to
        # the path-derived id (and id-as-name) when launched standalone.
        project_id = cfg.PROJECT_ID or _derive_project_id(export_dir)
        project_name = cfg.PROJECT_NAME or project_id
        stats = _review_export.export_flagged_masks(
            to_export, batch_files,
            export_dir=export_dir,
            review_dir=session.get('review_dir') or cfg.REVIEW_DIR,
            repo_url=cfg.REVIEW_REPO_URL,
            library_dir=cfg.EXPERT_LIBRARY_DIR or None,
            master_codes=cfg.MASTER_CODES_CSV,
            contacts=contacts,
            featured_codes=TARGET_SPECIES,
            project_id=project_id,
            project_name=project_name,
            pad_px=cfg.REVIEW_CROP_PAD_PX,
            max_edge=cfg.REVIEW_MAX_EDGE,
            full_max_edge=cfg.REVIEW_FULL_MAX_EDGE,
            overlap_thresh=cfg.REVIEW_OVERLAP_THRESH,
            git_push=cfg.REVIEW_GIT_PUSH,
            log_fn=lambda m: log(m),
        )
        if stats.get('new') or stats.get('auto_relabeled') or stats.get('skipped_missing_image'):
            log(f"[review] batch: +{stats.get('new', 0)} queued, "
                f"{stats.get('auto_relabeled', 0)} auto-relabeled, "
                f"{stats.get('skipped_missing_image', 0)} skipped (missing image), "
                f"pushed={stats.get('pushed')}, pending={stats.get('pending_total')}")
        return stats
    except Exception as e:
        log(f"[review] export bundle failed (non-fatal): {e}")
        return {'error': str(e)}


@app.route('/api/export_batch', methods=['POST'])
def do_export():
    """Export current batch to YOLO Segmentation format."""
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']
    export_dir = session['export_dir']

    batch_files = files[offset:offset + batch_size]
    if not batch_files:
        return jsonify({'error': 'No images to export'}), 400

    # Collect segmentations for export
    to_export = {}
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            to_export[fn] = seg

    stats = export_batch(
        to_export, export_dir, session['class_map'],
        symlink=cfg.SYMLINK_IMAGES,
    )

    # Ship REVIEW-flagged masks to the expert-review GitHub-Pages repo + the
    # permanent cross-project library (auto-relabels any already known to a past
    # expert). Runs before flush_all so in-place relabels persist.
    review_stats = _export_review_bundle(to_export, batch_files)

    # Mark exported and generate reviewed overlay images
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if seg:
            seg['exported'] = True
            _stamp_provenance(seg, fn)
            render_segmentation_overlays(seg, fn, export_dir, stage="reviewed")

    # Build per-frame manifest entries (one row per routed frame in this batch)
    batch_name = _next_batch_name(export_dir)
    manifest_entries = []
    for fn in batch_files:
        seg = _find_segmentation(fn)
        if not seg:
            continue
        stem = os.path.splitext(fn)[0]
        accepted = [m for m in seg.get('masks', [])
                    if m.get('status') == 'accepted' and not m.get('review')
                    and m.get('species') != 'REVIEW' and not _needs_species(m)]
        has_review = any(m.get('review') or m.get('species') == 'REVIEW'
                         for m in seg.get('masks', []))
        image_path = seg.get('image_path_abs')
        if seg.get('scrapped'):
            outcome, lbl, img = 'scrapped', '', ''
        elif not image_path or not os.path.exists(image_path):
            outcome, lbl, img = 'image_missing', '', ''   # export_yolo skips it; no label is written
        elif accepted:
            outcome, lbl, img = 'exported_with_masks', stem + '.txt', fn
        elif has_review:
            outcome, lbl, img = 'review_only', '', ''
        else:
            outcome, lbl, img = 'exported_empty', stem + '.txt', fn
        preview = ''
        if outcome not in ('scrapped', 'image_missing'):
            preview = _publish_preview(export_dir, stem)
        from collections import Counter
        species_counts = Counter(m.get('species', 'UNK') for m in accepted)
        entry = {
            'basename': stem, 'n_masks': len(accepted),
            'label_file': lbl, 'image_file': img, 'outcome': outcome,
            'preview_file': preview, 'batch_name': batch_name,
            'image_path': image_path or '',
        }
        entry.update({sp: c for sp, c in species_counts.items()})
        manifest_entries.append(entry)
    missing_images = [e['basename'] for e in manifest_entries if e['outcome'] == 'image_missing']
    if missing_images:
        log(f"[export] WARNING: {len(missing_images)} frame(s) skipped — image file missing on disk "
            f"(Step-3 eligibility should prevent this): {', '.join(missing_images)}")
    write_export_manifest(export_dir, manifest_entries)
    reconcile_export(export_dir, manifest_entries)

    flush_all()

    # Rebuild review list under the removed-frames model: exported + scrapped
    # frames drop out, review_offset resets to 0. The rebuilt list IS the next
    # batch; "done" is review_files == [].
    _build_review_list()

    # GUARANTEE: when the queue is empty every routed (non-scrapped) frame must
    # be accounted for as exported. _build_review_list includes every
    # non-scrapped, non-exported frame, so an empty list means
    # exported + scrapped == routed. Assert it so a silently-dropped frame is
    # caught loudly instead of vanishing from the YOLO export.
    if not session['review_files']:
        routed = exported = scrapped = 0
        for _s in session['segmentations'].values():
            routed += 1
            if _s.get('scrapped', False):
                scrapped += 1
            elif _s.get('exported', False):
                exported += 1
        accounted = exported + scrapped
        if accounted != routed:
            log(f"[batch] ALL-DONE RECONCILE MISMATCH: routed={routed} "
                f"exported={exported} scrapped={scrapped} "
                f"accounted={accounted} (lost {routed - accounted})")
        else:
            log(f"[batch] all done: routed={routed} = exported={exported} "
                f"+ scrapped={scrapped}; every frame accounted for")
        assert accounted == routed, (
            f"review queue empty but {routed - accounted} routed frame(s) "
            f"unaccounted (exported={exported} scrapped={scrapped} routed={routed})")

    out = {'ok': True, **stats}
    if review_stats:
        out['review'] = review_stats
    return jsonify(out)


@app.route('/api/status')
def status():
    segs = session['segmentations']
    agg = {
        'images': len(segs),
        'reviewed': sum(1 for s in segs.values() if s.get('reviewed', False)),
        'exported': sum(1 for s in segs.values() if s.get('exported', False)),
        'total_masks': sum(len(s.get('masks', [])) for s in segs.values()),
        'custom_imports': sum(1 for s in segs.values() if s.get('custom_import', False)),
    }
    # Batch progress under the removed-frames model: do_export() drops exported
    # frames from review_files and resets review_offset to 0, so the offset is
    # always 0 between batches. The number of batches still to do is simply how
    # many batch-sized chunks the remaining review_files split into. When
    # review_files is empty, nothing remains and the frontend shows "All Done".
    batch_size = session.get('review_batch_size') or cfg.REVIEW_BATCH_SIZE
    n_review = len(session['review_files'])
    batches_remaining = ((n_review + batch_size - 1) // batch_size) if (n_review and batch_size) else 0
    total_batches = batches_remaining
    batches_done = 0

    return jsonify({
        'configured': session['configured'],
        'phase': session['phase'],
        'processing_done': session['processing_idx'],
        'processing_total': len(session['processing_queue']),
        'review_total': n_review,
        'review_offset': session['review_offset'],
        'review_batch_size': batch_size,
        'total_classes': len(session['class_map']),
        'stats': agg,
        # Batch progress (orchestrator panel reads these).
        'batches_total': total_batches,
        'batches_done': batches_done,
        'batches_remaining': batches_remaining,
        'orchestrator_url': _orchestrator_url(),
    })


def _host_open(path):
    """xdg-open a path on the host (host-only by design). Returns (ok, err)."""
    if not path or not os.path.exists(path):
        return False, f'path not found: {path}'
    try:
        subprocess.Popen(['xdg-open', path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, ''
    except FileNotFoundError:
        return False, 'xdg-open not available on this host'
    except Exception as e:
        return False, str(e)


@app.route('/api/open_export_folder', methods=['POST'])
def open_export_folder():
    export_dir = session.get('export_dir', '')
    ok, err = _host_open(export_dir)
    return (jsonify({'ok': True, 'path': export_dir}) if ok
            else (jsonify({'error': err}), 400))


@app.route('/api/open_export_manifest', methods=['POST'])
def open_export_manifest():
    export_dir = session.get('export_dir', '')
    manifest = os.path.join(export_dir, 'export_manifest.csv') if export_dir else ''
    ok, err = _host_open(manifest)
    return (jsonify({'ok': True, 'path': manifest}) if ok
            else (jsonify({'error': err}), 400))


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve raw image file."""
    seg = _find_segmentation(filename)
    if seg:
        path = seg.get('image_path_abs', '')
        if path and os.path.exists(path):
            return send_file(path)
    return jsonify({'error': 'not found'}), 404


# ── Main ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TCRMP Segment Images - SAM3 Review App')
    parser.add_argument('--port', type=int, default=cfg.PORT)
    args = parser.parse_args()

    print(f"Loading SAM3 model...")
    from sam_engine import SAM3Engine
    session['sam_engine'] = SAM3Engine(cfg)

    log("SAM3 model loaded, server starting")
    print(f"Server ready on http://localhost:{args.port}")

    app.run(host='0.0.0.0', port=args.port, debug=False)
