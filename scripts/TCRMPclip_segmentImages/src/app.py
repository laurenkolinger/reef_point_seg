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
    REVIEW_TAGGING_ENABLED, REVIEW_LOCKED_MSG, review_tag_requested,
)
from export_yolo import load_class_map, save_class_map, export_batch, update_data_yaml
from render_overlay import render_segmentation_overlays

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

# Where the label list is cleaned of the expert-review tag (Lauren Olinger,
# 2026-09-06). A target label of REVIEW draws a live, armable REVIEW control in
# the label row, which offers the operator a control every add route answers
# with a 409. The entry is DROPPED and logged rather than refused, because
# TCRMP_TARGET_SPECIES comes from a launcher and refusing a whole launch over
# one entry would cost the operator the session. Logged with print(): log() is
# not defined yet when this runs at import time.
def _drop_review_labels(labels, source):
    """Remove every label that spells out the expert-review tag.

    Parameters:
        labels: the label codes read from one source, in their original order.
        source: plain words naming where they came from, for the log line.

    Returns:
        list: the labels that are not the review tag, in their original order.
        The list comes back unchanged when review tagging is enabled again, so
        flipping REVIEW_TAGGING_ENABLED reverses this with no other edit.

    Example:
        _drop_review_labels(['PA', 'review'], 'TCRMP_TARGET_SPECIES') returns
        ['PA'] and prints one line naming 'review'.
    """
    if REVIEW_TAGGING_ENABLED:
        return list(labels)
    kept, dropped = [], []
    for label in labels:
        if review_tag_requested({'species': label}):
            dropped.append(label)
        else:
            kept.append(label)
    if dropped:
        print(f"[review-lock] dropped {len(dropped)} target label(s) spelling "
              f"out the expert-review tag from {source}: {', '.join(dropped)}. "
              f"{REVIEW_LOCKED_MSG}")
    return kept


def _load_target_species():
    """The target species codes this app offers, in priority order.

    Reads the orchestrator override TCRMP_TARGET_SPECIES first, then the
    TARGET_SPECIES line of TCRMPcvr_chooseImages' config.py, then a built-in
    default list. Every source is passed through _drop_review_labels, so no
    source can arm the suspended expert-review flag.

    Returns:
        list: the species codes, in the order the source gave them.
    """
    import re
    # Pipeline orchestrator override
    env_sp = os.environ.get('TCRMP_TARGET_SPECIES', '')
    if env_sp:
        return _drop_review_labels(
            [s.strip() for s in env_sp.split(',') if s.strip()],
            'TCRMP_TARGET_SPECIES')
    config_path = os.path.join(REPO_DIR, 'TCRMPcvr_chooseImages', 'src', 'config.py')
    try:
        with open(config_path) as f:
            for line in f:
                if line.strip().startswith('TARGET_SPECIES'):
                    match = re.search(r'\[(.+)\]', line)
                    if match:
                        return _drop_review_labels(
                            [s.strip().strip('"').strip("'")
                             for s in match.group(1).split(',')],
                            config_path)
    except FileNotFoundError:
        pass
    return _drop_review_labels(
        ['OFRA', 'PA', 'OA', 'OFAV', 'AL', 'MC', 'AA'],
        'the built-in default target list')

TARGET_SPECIES = _load_target_species()


def _is_review(p):
    """A REVIEW-flagged prompt/mask: segmented as its OWN positive mask (unlike
    a non-target '?' point, which is a negative refinement click), shipped to the
    expert-review site, and never trained."""
    return bool(p.get('review')) or p.get('species', p.get('species_code', '')) == 'REVIEW'


def _review_tag_refused(payload):
    """Report whether a request body must be refused for asking to tag a mask.

    Expert-review flagging is locked (Lauren Olinger, 2026-09-06): no mask may
    GAIN a review tag. This app is archived from the orchestrator sidebar but is
    still launchable through POST /api/step/5/run, and it writes into the same
    segmentation files and the same mask registry as the named annotators, so
    the lock covers it too.

    This test is deliberately broader than _is_review above, which classifies
    masks already stored on disk and must not move.

    The gap between this test and _is_review widened on 2026-09-06, when the
    category joined the spellings the refusal catches. This test recognises
    FOUR, each normalised for case and whitespace and each read on its own: a
    truthy 'review' key, a 'species', a 'species_code' or a 'category' that
    spells out REVIEW. _is_review recognises TWO: a truthy 'review' key or a
    species of exactly 'REVIEW'. The asymmetry is deliberate. What may be
    WRITTEN is judged as widely as the tag can be spelled; what the masks
    already on disk MEAN is judged by _is_review, which drives the expert
    export, the training filter and the provenance ledger.

    Args:
        payload: the parsed request body, one relabel entry, a point info dict
            out of a prompt file, or a candidate mask dict. A value with no
            dict-like get() answers False rather than raising, so an unparsed
            body can be passed straight in.

    Returns:
        bool: True when the lock is on and this payload asks for a review tag,
        in any spelling mask_ops.review_tag_requested recognises.
    """
    return not REVIEW_TAGGING_ENABLED and review_tag_requested(payload)


def _review_locked_response():
    """The refusal every route returns when a request asks for a new tag.

    Returns:
        tuple: a Flask (response, status) pair carrying HTTP 409 and
        {'ok': False, 'review_locked': True, 'error': REVIEW_LOCKED_MSG}. Every
        VICARIUS annotator answers with this exact shape, so the browser has a
        single case to recognise.
    """
    return jsonify({'ok': False, 'review_locked': True,
                    'error': REVIEW_LOCKED_MSG}), 409


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

    # Per-image segmentation data, organized by year
    'segmentations_by_year': {},  # year -> {image_filename: seg_dict}

    # Processing queue
    'processing_queue': [],      # list of {image_path, year, filename, prompts}
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
}


# ── Reviewer identity ────────────────────────────────────────────
# Same contract as the annotator twins: the orchestrator seeds TCRMP_REVIEWER
# at launch and POSTs /api/reviewer to a reused process, so the session value
# tracks whoever opened the project last. Blank stays blank.

def _norm_reviewer(value):
    """Initials as stored everywhere: stripped, uppercased, at most 8 chars."""
    return str(value or '').strip().upper()[:8]


def _current_reviewer():
    """Session reviewer, else the TCRMP_REVIEWER env default; '' when unknown."""
    return _norm_reviewer(session.get('reviewer') or os.environ.get('TCRMP_REVIEWER', ''))


def log(msg, year=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)


# ── Persistence (JSON per year) ──────────────────────────────────

def _seg_path(export_dir, year):
    return os.path.join(export_dir, 'segmentations', str(year), 'segmentations.json')


def _manifest_path(export_dir, year):
    return os.path.join(export_dir, 'segmentations', str(year), 'processed_manifest.json')


def load_segmentations(export_dir, year):
    path = _seg_path(export_dir, year)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_segmentations(export_dir, year, data):
    path = _seg_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=1)


def save_manifest(export_dir, year, filenames):
    path = _manifest_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'processed': sorted(filenames),
            'last_updated': datetime.now().isoformat(),
            'count': len(filenames),
        }, f, indent=2)


def flush_all():
    export_dir = session['export_dir']
    if not export_dir:
        return
    for year, segs in session['segmentations_by_year'].items():
        save_segmentations(export_dir, year, segs)
        save_manifest(export_dir, year, list(segs.keys()))
    # Also save class map
    if session['class_map']:
        save_class_map(export_dir, session['class_map'])


# ── Review helpers ───────────────────────────────────────────────

def _build_review_list():
    """Build the review queue.

    Mirrors the step 4 policy: once a frame is exported or scrapped it is
    considered done and does not re-enter the queue. Done-is-done.

    If `session['shuffle']` is set, the unexported list is shuffled so the
    user sees a random sample instead of year-first ordering.
    """
    import random as _random
    target_only = getattr(cfg, 'TARGET_SPECIES_ONLY', 0)
    unexported = []
    for year in sorted(session['segmentations_by_year']):
        segs = session['segmentations_by_year'][year]
        for fn in sorted(segs):
            seg = segs[fn]
            if seg.get('scrapped', False):
                continue
            if seg.get('exported', False):
                continue
            if target_only:
                masks = seg.get('masks', [])
                if not any(m.get('species', '') in TARGET_SPECIES for m in masks):
                    continue
            unexported.append(fn)
    if session.get('shuffle'):
        _random.shuffle(unexported)
    session['review_files'] = unexported
    session['review_offset'] = 0


def _find_segmentation(filename):
    """Find a segmentation dict by filename. Returns (seg_dict, year) or (None, None)."""
    for year, segs in session['segmentations_by_year'].items():
        if filename in segs:
            return segs[filename], year
    return None, None


def _ensure_image_embedded(filename):
    """Ensure the SAM3 engine has the given image embedded for refinement."""
    engine = session.get('sam_engine')
    if engine is None:
        return False

    seg, year = _find_segmentation(filename)
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
                           orchestrator_url=_orchestrator_url(),
                           orchestrated=bool(os.environ.get('TCRMP_INPUT_DIR')))


@app.route('/api/configure', methods=['POST'])
def configure():
    """Load sam_click_prompts.json for selected years/categories, build queue."""
    data = request.json
    input_dir = data.get('input_dir', cfg.INPUT_DIR)
    export_dir = data.get('export_dir', cfg.EXPORT_DIR)
    categories = data.get('categories', cfg.DEFAULT_CATEGORIES)
    batch_size = data.get('review_batch_size', cfg.REVIEW_BATCH_SIZE)

    session['input_dir'] = os.path.abspath(input_dir)
    session['export_dir'] = os.path.abspath(export_dir)
    session['categories'] = categories
    session['review_batch_size'] = batch_size
    session['shuffle'] = bool(data.get('shuffle', False))

    # Expert-review settings: where to push the queue, and who gets the CSV.
    session['review_dir'] = (data.get('review_dir') or cfg.REVIEW_DIR)
    _raw_contacts = data.get('contacts', cfg.REVIEW_CONTACTS)
    if isinstance(_raw_contacts, str):
        _raw_contacts = _raw_contacts.split(',')
    session['contacts'] = [c.strip() for c in (_raw_contacts or []) if c and c.strip()]

    os.makedirs(session['export_dir'], exist_ok=True)

    # Load existing class map
    session['class_map'] = load_class_map(session['export_dir'])

    # Discover years from input directory
    input_dir_abs = session['input_dir']
    years = []
    for entry in sorted(os.listdir(input_dir_abs)):
        entry_path = os.path.join(input_dir_abs, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            prompts_path = os.path.join(entry_path, 'ids', 'sam_click_prompts.json')
            if os.path.exists(prompts_path):
                years.append(int(entry))

    if not years:
        return jsonify({'error': 'No sam_click_prompts.json found in input directory'}), 404

    # Load existing segmentations
    session['segmentations_by_year'] = {}
    already_processed = set()

    for year in years:
        existing = load_segmentations(session['export_dir'], year)
        session['segmentations_by_year'][year] = existing
        already_processed.update(existing.keys())

    # Build processing queue
    queue = []
    skipped = 0

    for year in years:
        prompts_path = os.path.join(input_dir_abs, str(year), 'ids', 'sam_click_prompts.json')
        with open(prompts_path) as f:
            prompts = json.load(f)

        raw_dir = os.path.join(input_dir_abs, str(year), 'raw')

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

            # Resolve raw image path
            raw_rel = img_data.get('raw_image', '')
            raw_path = os.path.join(input_dir_abs, str(year), raw_rel) if raw_rel else None
            if not raw_path or not os.path.exists(raw_path):
                raw_path = os.path.join(raw_dir, img_filename)
            if not os.path.exists(raw_path):
                log(f"WARNING: Raw image not found for {img_filename}")
                continue

            queue.append({
                'filename': img_filename,
                'image_path': raw_path,
                'year': year,
                'points': selected_points,
            })

    session['processing_queue'] = queue
    session['processing_idx'] = 0

    if not queue:
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

    log(f"Configured: {len(years)} years, {len(queue)} images to process, "
        f"{skipped} already done, categories={categories}")

    return jsonify({
        'years': years,
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

    # Discover years
    seg_dir = os.path.join(export_dir, 'segmentations')
    discovered_years = []
    if os.path.isdir(seg_dir):
        for entry in sorted(os.listdir(seg_dir)):
            if entry.isdigit():
                if os.path.exists(_seg_path(export_dir, int(entry))):
                    discovered_years.append(int(entry))

    if not discovered_years:
        return jsonify({'error': 'No segmentation data found'}), 404

    session['segmentations_by_year'] = {}
    total = 0
    for year in discovered_years:
        segs = load_segmentations(export_dir, year)
        session['segmentations_by_year'][year] = segs
        total += len(segs)

    session['phase'] = 'review'
    session['processing_queue'] = []
    _build_review_list()
    session['configured'] = True

    return jsonify({
        'total_processed': total,
        'years': discovered_years,
        'phase': 'review',
        'review_count': len(session['review_files']),
        'total_classes': len(session['class_map']),
    })


def _steal_region_from_others(seg, new_binary, target_id, min_area_px=100):
    """When a mask is added or edited, it CLAIMS the region it occupies.
    Any other non-rejected mask that overlaps with that region loses the
    overlap (its RLE / polygon gets re-computed with the new mask subtracted).
    Masks that shrink below `min_area_px` are dropped entirely.

    Returns a list of mask dicts that were modified or dropped; the caller
    can forward that to the client so it merges the state without a full
    refetch. Keeps rejected masks untouched (they're "deleted" in the user's
    mental model).
    """
    import numpy as np
    from mask_ops import rle_decode, update_mask_geometry

    h, w = new_binary.shape
    new_any = new_binary.any()
    if not new_any:
        return {'updated': [], 'removed': []}

    updated = []
    removed = []
    survivors = []
    for m in seg.get('masks', []):
        if m.get('id') == target_id or m.get('status') == 'rejected':
            survivors.append(m)
            continue
        rle = m.get('rle')
        if not rle:
            survivors.append(m)
            continue
        try:
            other = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            survivors.append(m)
            continue
        overlap = other & new_binary
        if not overlap.any():
            survivors.append(m)
            continue
        reduced = other & ~new_binary
        if int(reduced.sum()) < min_area_px:
            # Whatever's left is too small to be useful — drop.
            removed.append({'id': m.get('id')})
            continue
        updated_m = update_mask_geometry(
            m, reduced, cfg.POLYGON_SIMPLIFY_EPSILON)
        if updated_m is None:
            removed.append({'id': m.get('id')})
            continue
        updated.append(m)
        survivors.append(m)
    seg['masks'] = survivors
    return {'updated': updated, 'removed': removed}


def _process_single_image(engine, item, export_dir):
    """Segment one image. Returns (filename, year, n_masks, n_points).

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
    year = item['year']
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
    # build_mask_dict is the one place a mask is born, and it counts every
    # expert-review tag it drops on a process-wide tally. This frame's own
    # figure is counted here rather than read as a difference of that tally,
    # because the tally is not this frame's alone: the exemplar scan route
    # derives its point_info from an existing mask, so a scan seeded from one of
    # the masks flagged before the lock builds a tagged point_info on its own
    # request thread while this loop runs. Counting per call keeps the logged
    # number exact; the running total stays with review_tags_dropped().
    review_tags_lost = 0
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
                # Expert-review flagging is locked (2026-09-06). A REVIEW prompt
                # read back out of ids/sam_click_prompts.json still produces a
                # real mask, but the mask is born untagged: build_mask_dict
                # blanks a REVIEW species and category itself, and the stamp
                # that used to set the review boolean here is gone. This pass
                # has no request body for a route-level refusal to inspect, so
                # the count logged below is how the run says what it dropped
                # instead of doing it silently. The condition is the one
                # build_mask_dict counted its own drop on, read on the same
                # prompt and only where a mask was actually built.
                if _review_tag_refused(pt):
                    review_tags_lost += 1
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

    if review_tags_lost > 0:
        log(f"[review-lock] {filename}: {review_tags_lost} point(s) asked for an "
            f"expert-review tag and got an ordinary mask instead (flagging is "
            f"locked; masks flagged before 2026-09-06 keep theirs)", year)

    session['segmentations_by_year'].setdefault(year, {})[filename] = seg_dict
    render_segmentation_overlays(seg_dict, filename, export_dir, year, stage="auto")

    return filename, year, len(final_masks), len(points)


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
                fn, year, n_masks, n_pts = _process_single_image(engine, item, export_dir)
                log(f"SAM3 {idx + 1}/{total}: {fn} ({n_pts} pts -> {n_masks} masks)")
                # Persist this year's segmentations after EVERY image so an
                # orchestrator restart (or crash, or ctrl+C) can't rewind
                # progress. Cost: one ~KB JSON rewrite per image — cheap.
                try:
                    save_segmentations(
                        export_dir, year,
                        session['segmentations_by_year'].get(year, {}),
                    )
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
        seg, year = _find_segmentation(fn)
        if seg:
            n_masks = len(seg.get('masks', []))
            n_pending = sum(1 for m in seg.get('masks', []) if m.get('status') == 'pending')
            n_accepted = sum(1 for m in seg.get('masks', []) if m.get('status') == 'accepted')
            images.append({
                'filename': fn,
                'year': year,
                'n_masks': n_masks,
                'n_pending': n_pending,
                'n_accepted': n_accepted,
                'reviewed': seg.get('reviewed', False),
                'exported': seg.get('exported', False),
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


@app.route('/api/next_review_batch', methods=['POST'])
def next_review_batch():
    files = session['review_files']
    batch_size = session['review_batch_size']
    if session['review_offset'] + batch_size < len(files):
        session['review_offset'] += batch_size
    else:
        return jsonify({
            'offset': session['review_offset'],
            'total_images': len(files),
            'done': True,
        })
    return jsonify({
        'offset': session['review_offset'],
        'total_images': len(files),
    })


@app.route('/api/image/<path:filename>')
def get_image_data(filename):
    """Get all mask data for a specific image."""
    seg, year = _find_segmentation(filename)
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
        'year': year,
    })


@app.route('/api/image/<path:filename>/masks', methods=['PUT'])
def update_masks(filename):
    """Bulk update mask statuses (accept/reject).

    Statuses only: this route reads `updates` and nothing else, so it carries no
    relabel field and cannot write a species or a review flag. A relabel field
    added here in future needs the same refusal POST /add carries, or the
    2026-09-06 expert-review lock reopens on this app.
    """
    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    data = request.json
    updates = data.get('updates', {})  # {mask_id: status}

    for mask in seg.get('masks', []):
        mid = str(mask['id'])
        if mid in updates:
            mask['status'] = updates[mid]

    seg['reviewed'] = True

    # Provenance: stamp per-(image,label) outcomes the moment the operator
    # finishes this frame, and upsert the flat ledger. Non-fatal on failure.
    from provenance import compute_label_outcomes, write_provenance_csv
    _source = os.environ.get('TCRMP_PROVENANCE_SOURCE', 'step5')
    _reviewer = _current_reviewer()
    outcomes = compute_label_outcomes(
        seg, TARGET_SPECIES, source=_source, reviewer=_reviewer)
    seg['label_outcomes'] = outcomes
    try:
        write_provenance_csv(
            session['export_dir'],
            os.path.splitext(filename)[0],
            outcomes,
            cfg.PROJECT_ID or _derive_project_id(session['export_dir']))
    except Exception as e:
        log(f"[provenance] csv write failed for {filename} (non-fatal): {e}")
    try:
        save_segmentations(session['export_dir'], year,
                           session['segmentations_by_year'].get(year, {}))
    except Exception as e:
        log(f"[provenance] seg flush failed for {filename}: {e}")

    return jsonify({'ok': True})


@app.route('/api/image/<path:filename>/accept_all', methods=['POST'])
def accept_all_masks(filename):
    """Accept all pending masks for an image."""
    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    count = 0
    for mask in seg.get('masks', []):
        if mask.get('status') == 'pending':
            mask['status'] = 'accepted'
            count += 1
    seg['reviewed'] = True

    # Provenance: stamp per-(image,label) outcomes the moment the operator
    # finishes this frame, and upsert the flat ledger. Non-fatal on failure.
    from provenance import compute_label_outcomes, write_provenance_csv
    _source = os.environ.get('TCRMP_PROVENANCE_SOURCE', 'step5')
    _reviewer = _current_reviewer()
    outcomes = compute_label_outcomes(
        seg, TARGET_SPECIES, source=_source, reviewer=_reviewer)
    seg['label_outcomes'] = outcomes
    try:
        write_provenance_csv(
            session['export_dir'],
            os.path.splitext(filename)[0],
            outcomes,
            cfg.PROJECT_ID or _derive_project_id(session['export_dir']))
    except Exception as e:
        log(f"[provenance] csv write failed for {filename} (non-fatal): {e}")
    try:
        save_segmentations(session['export_dir'], year,
                           session['segmentations_by_year'].get(year, {}))
    except Exception as e:
        log(f"[provenance] seg flush failed for {filename}: {e}")

    return jsonify({'ok': True, 'accepted': count})


@app.route('/api/image/<path:filename>/refine', methods=['POST'])
def refine_mask(filename):
    """Refine a mask with an additional positive/negative click."""
    seg, year = _find_segmentation(filename)
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
    # Rejected masks are always ignored (they're "deleted" in the UI).
    import numpy as np
    from mask_ops import rle_decode
    new_binary = np.asarray(result['mask'], dtype=bool)
    h, w = new_binary.shape
    other_masks = [m for m in seg.get('masks', [])
                   if m.get('id') != mask_id and m.get('status') != 'rejected']
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
            return jsonify({'error': 'refinement fully overlapped other masks'}), 400

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
    seg, year = _find_segmentation(filename)
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
    # masks are ignored so you can redraw cleanly over a deleted one.
    import numpy as np
    from mask_ops import rle_decode
    new_binary = np.asarray(result['mask'], dtype=bool)
    h, w = new_binary.shape
    other_masks = [m for m in seg.get('masks', [])
                   if m.get('id') != mask_id and m.get('status') != 'rejected']
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
            return jsonify({'error': 'rebox fully overlapped other masks'}), 400

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

    In both cases we then subtract other masks on this frame so the edit
    can't eat into neighbors (same rule as /refine).
    """
    import numpy as np
    try:
        import cv2
    except ImportError:
        return jsonify({'error': 'cv2 not available — draw edit needs opencv'}), 500
    from mask_ops import rle_decode, update_mask_geometry

    seg, year = _find_segmentation(filename)
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

    # Edited-wins policy: the drawn stroke claims the region. Overlapping
    # live neighbors yield. (Refine keeps strict clipping by contrast.)
    updated = update_mask_geometry(
        target_mask, new_binary, cfg.POLYGON_SIMPLIFY_EPSILON)
    if updated is None:
        return jsonify({'error': 'mask disappeared after draw edit'}), 400

    overlap_delta = _steal_region_from_others(seg, new_binary, mask_id)

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
    seg, year = _find_segmentation(filename)
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
    # refined neighbor from spilling into that region.
    other_masks = [m for m in seg.get('masks', [])
                   if m.get('id') != mask_id and m.get('status') != 'rejected']
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
    seg, year = _find_segmentation(filename)
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
    seg, year = _find_segmentation(filename)
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
    seg, year = _find_segmentation(filename)
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
# Deliberately exempt from the 2026-09-06 review-flag lock: restoring a tag that
# is already written on disk is preservation, not addition, and it is the leg of
# the expert round trip that brings an identification home. The lock refuses new
# tags arriving from a request body or a prompt file, never one already stored.
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

    seg, year = _find_segmentation(filename)
    if not seg or year is None:
        return jsonify({'error': 'not found'}), 404

    export_dir = session.get('export_dir')
    if not export_dir:
        return jsonify({'error': 'no export dir configured'}), 400

    disk_segs = load_segmentations(export_dir, year)
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


@app.route('/api/image/<path:filename>/exemplar', methods=['POST'])
def exemplar_scan(filename):
    """Scan for similar regions. mode='this' uses one mask, mode='all' uses all accepted masks."""
    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    mode = data.get('mode', 'this')  # 'this' or 'all'
    threshold = data.get('threshold', 0.4)

    if not _ensure_image_embedded(filename):
        return jsonify({'error': 'could not embed image'}), 500

    all_new_masks = []

    if mode == 'this':
        # Scan using one specific mask as exemplar
        mask_id = data.get('mask_id')
        if mask_id is None:
            return jsonify({'error': 'mask_id required for mode=this'}), 400
        source_mask = None
        for m in seg.get('masks', []):
            if m['id'] == mask_id:
                source_mask = m
                break
        if source_mask is None:
            return jsonify({'error': 'mask not found'}), 404

        bbox = source_mask.get('bbox')
        if not bbox:
            return jsonify({'error': 'mask has no bounding box'}), 400

        results = engine.exemplar_scan(bbox)
        all_new_masks = _filter_exemplar_results(
            results, seg,
            source_mask.get('species', 'UNK'),
            source_mask.get('category', ''),
            threshold,
        )

    elif mode == 'all':
        # Scan using every unique accepted species as exemplar
        seen_species = set()
        exemplar_sources = []
        for m in seg.get('masks', []):
            if m.get('status') not in ('accepted', 'pending'):
                continue
            sp = m.get('species', '')
            if sp and sp not in seen_species and m.get('bbox'):
                seen_species.add(sp)
                exemplar_sources.append(m)

        for source in exemplar_sources:
            results = engine.exemplar_scan(source['bbox'])
            new_masks = _filter_exemplar_results(
                results, seg,
                source.get('species', 'UNK'),
                source.get('category', ''),
                threshold,
            )
            # Add to seg so subsequent species see these as occupied
            seg['masks'].extend(new_masks)
            all_new_masks.extend(new_masks)

        # Return early — masks already added to seg above
        return jsonify({
            'ok': True,
            'new_masks': all_new_masks,
            'count': len(all_new_masks),
            'species_scanned': sorted(seen_species),
        })

    # For 'this' mode, add to seg
    seg['masks'].extend(all_new_masks)

    return jsonify({
        'ok': True,
        'new_masks': all_new_masks,
        'count': len(all_new_masks),
    })


@app.route('/api/image/<path:filename>/scrap', methods=['POST'])
def scrap_segmentation(filename):
    """Mark a frame as scrapped — excluded from review and export.
    Mirrors the step 4 scrap API so the two apps behave the same way.
    """
    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = True
    seg['reviewed'] = True
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


@app.route('/api/image/<path:filename>/unscrap', methods=['POST'])
def unscrap_segmentation(filename):
    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404
    seg['scrapped'] = False
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


@app.route('/api/image/<path:filename>/add', methods=['POST'])
def add_mask(filename):
    """Add a new mask at a click point (SAM3 point prompt).

    Edited-wins policy: the new mask claims whatever region SAM3 gives it.
    Any live (non-rejected) neighbor that overlaps yields that region —
    its RLE + polygon get re-computed without the new mask's footprint.
    Rejected masks are untouched (they're "deleted" in the user's mental
    model, so they can't block a new mask either).

    Review-flag lock (2026-09-06): a body asking for an expert-review tag is
    refused with the shared 409 and creates nothing. This route read the species
    straight out of the body and had no guard of any kind, so both arms of the
    tag reached the stored mask.
    """
    import numpy as np

    seg, year = _find_segmentation(filename)
    if not seg:
        return jsonify({'error': 'not found'}), 404

    engine = session.get('sam_engine')
    if engine is None:
        return jsonify({'error': 'SAM3 engine not loaded'}), 500

    data = request.json
    if _review_tag_refused(data):
        return _review_locked_response()
    click_x = data['click_x']
    click_y = data['click_y']
    species = data.get('species', '')
    category = data.get('category', '')
    label = data.get('label', '?')

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
    blocking = [m for m in existing_masks if m.get('status') != 'rejected']
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
        new_binary = new_binary & ~forbidden
        if not new_binary.any():
            return jsonify({'error': 'new mask region fully overlaps existing (non-rejected) masks — try a cleaner click'}), 400

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

    seg['masks'].append(mask_dict)
    return jsonify({'ok': True, 'mask': mask_dict})


def _derive_project_id(export_dir):
    parts = os.path.abspath(export_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith('run_'):
            return p
    return os.path.basename(os.path.dirname(os.path.abspath(export_dir))) or 'project'


def _count_review_masks(to_export, batch_files):
    """How many masks in this batch carry an expert-review tag.

    Parameters:
        to_export: {filename: segmentation dict} for the batch.
        batch_files: the filenames in the batch, in export order.

    Returns:
        The number of tagged masks, counted with the same two spellings the
        export selector itself reads: the review boolean, or a species of
        REVIEW. A frame missing from to_export contributes nothing.
    """
    n_review = 0
    for fn in batch_files:
        seg = to_export.get(fn) or {}
        for m in seg.get('masks', []):
            if m.get('review') or m.get('species') == 'REVIEW':
                n_review += 1
    return n_review


def _export_review_bundle(to_export, batch_files):
    """Ship this batch's REVIEW-flagged masks to the expert-review repo +
    library, and auto-relabel any that a past expert already identified.
    Best-effort: any failure is logged and never breaks the YOLO export.

    SUSPENDED while REVIEW_TAGGING_ENABLED is False (2026-09-06). This app is
    archived from the orchestrator sidebar but is still launchable through
    POST /api/step/5/run, and this is its one path that publishes outward: it
    mints a fresh review identifier onto every already-flagged mask and pushes
    the bundle to a public GitHub Pages repository. The function returns early
    with the count of masks it did not publish, in the shape the two named
    annotators use. Reversing it is the one constant, never the environment
    variable TCRMP_REVIEW_GIT_PUSH, which any hand launch can flip back.
    """
    if not REVIEW_TAGGING_ENABLED:
        n_review = _count_review_masks(to_export, batch_files)
        log(f"[review-lock] expert-review export SUSPENDED: {n_review} flagged "
            f"mask(s) in this batch were not published and not given a review "
            f"identifier. {REVIEW_LOCKED_MSG}")
        return {'disabled': True, 'review_masks_skipped': n_review}
    if not _REVIEW_OK:
        return None
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


@app.route('/api/reviewer', methods=['GET', 'POST'])
def api_reviewer():
    """GET -> {'reviewer'}; POST {'reviewer'} sets session['reviewer']
    (normalized) and echoes it back. The orchestrator pushes the latest
    opener here when it reuses a live step 5 process."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        session['reviewer'] = _norm_reviewer(data.get('reviewer'))
    return jsonify({'reviewer': _current_reviewer()})


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
        seg, year = _find_segmentation(fn)
        if seg:
            to_export[fn] = seg

    stats = export_batch(
        to_export, export_dir, session['class_map'],
        symlink=cfg.SYMLINK_IMAGES,
        reviewer=_current_reviewer(),
    )

    # Ship REVIEW-flagged masks to the expert-review GitHub-Pages repo + the
    # permanent cross-project library (auto-relabels any already known to a past
    # expert). Runs before flush_all so in-place relabels persist.
    review_stats = _export_review_bundle(to_export, batch_files)

    # Provenance: stamp/refresh per-(image,label) outcomes at export. The
    # flat ledger upsert means this supersedes any earlier finish-time row.
    from provenance import compute_label_outcomes, write_provenance_csv
    _source = os.environ.get('TCRMP_PROVENANCE_SOURCE', 'step5')
    _reviewer = _current_reviewer()
    # Mark exported and generate reviewed overlay images
    for fn in batch_files:
        seg, year = _find_segmentation(fn)
        if seg:
            seg['exported'] = True
            outcomes = compute_label_outcomes(
                seg, TARGET_SPECIES, source=_source, reviewer=_reviewer)
            seg['label_outcomes'] = outcomes
            try:
                write_provenance_csv(
                    export_dir, os.path.splitext(fn)[0], outcomes,
                    cfg.PROJECT_ID or _derive_project_id(export_dir))
            except Exception as e:
                log(f"[provenance] csv write failed for {fn} (non-fatal): {e}")
            render_segmentation_overlays(seg, fn, export_dir, year, stage="reviewed")

    flush_all()

    # Rebuild review list: unexported first, exported at end
    _build_review_list()

    out = {'ok': True, **stats}
    if review_stats:
        out['review'] = review_stats
    return jsonify(out)


@app.route('/api/status')
def status():
    year_stats = {}
    for year, segs in session['segmentations_by_year'].items():
        n_exported = sum(1 for s in segs.values() if s.get('exported', False))
        n_reviewed = sum(1 for s in segs.values() if s.get('reviewed', False))
        n_masks = sum(len(s.get('masks', [])) for s in segs.values())
        year_stats[year] = {
            'images': len(segs),
            'reviewed': n_reviewed,
            'exported': n_exported,
            'total_masks': n_masks,
        }
    # Batch progress for the orchestrator panel: how many review batches are
    # done vs. remaining. A "done" batch is one whose offset is fully behind the
    # current review_offset (the operator has exported/advanced past it).
    batch_size = session.get('review_batch_size') or cfg.REVIEW_BATCH_SIZE
    n_review = len(session['review_files'])
    total_batches = ((n_review + batch_size - 1) // batch_size) if n_review else 0
    batches_done = (session['review_offset'] // batch_size) if batch_size else 0
    if batches_done > total_batches:
        batches_done = total_batches
    batches_remaining = max(total_batches - batches_done, 0)

    return jsonify({
        'configured': session['configured'],
        'phase': session['phase'],
        'processing_done': session['processing_idx'],
        'processing_total': len(session['processing_queue']),
        'review_total': n_review,
        'review_offset': session['review_offset'],
        'review_batch_size': batch_size,
        'total_classes': len(session['class_map']),
        'year_stats': year_stats,
        # Batch progress (orchestrator panel reads these).
        'batches_total': total_batches,
        'batches_done': batches_done,
        'batches_remaining': batches_remaining,
        'orchestrator_url': _orchestrator_url(),
    })


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve raw image file."""
    seg, year = _find_segmentation(filename)
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
