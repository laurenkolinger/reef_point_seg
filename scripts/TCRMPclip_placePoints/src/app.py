"""
TCRMP Place Points — Unified CPC + OCR Review App

Takes selected_frames.csv from TCRMPcvr_chooseImages and:
  - CPC frames (pre-2020): loads existing (x,y) coords, remaps species codes
    to recoded versions using the recoded all_points.csv
  - OCR frames (2020+): runs GPU OCR detection (same as TCRMPclip_ocrID_batch)

All frames go through the same QAQC review UI for verification, point editing,
and unified export in SAM3-compatible format.

CRITICAL: The import directory (TCRMP_clip) is READ-ONLY.

Usage:
    python app.py [--port 5060] [--no-gpu]
"""

import os
import sys
import csv
import json
import glob
import re
import argparse
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import detect_annotations, get_ocr_reader
from species import SpeciesLookup, ocr_date_to_lookup_date, parse_transect_frame
from export import export_batch
from summarize import write_summary
import config as cfg

app = Flask(__name__, template_folder='templates')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)
USE_GPU = True

# ── Paths from config ─────────────────────────────────────────
DEFAULT_SELECTED_FRAMES = cfg.SELECTED_FRAMES
DEFAULT_CLIP_DIR = cfg.CLIP_DIR
DEFAULT_CPC_DIR = cfg.CPC_DIR
DEFAULT_EXPORT_DIR = cfg.EXPORT_DIR

# Species list from chooseImages config
CHOOSE_CONFIG_PATH = os.path.join(
    REPO_DIR, 'TCRMPcvr_chooseImages', 'src', 'config.py')


def _load_target_species():
    """Read TARGET_SPECIES from env var or TCRMPcvr_chooseImages/src/config.py."""
    # Pipeline orchestrator override
    env_sp = os.environ.get('TCRMP_TARGET_SPECIES', '')
    if env_sp:
        return [s.strip() for s in env_sp.split(',') if s.strip()]
    try:
        with open(CHOOSE_CONFIG_PATH) as f:
            for line in f:
                if line.strip().startswith('TARGET_SPECIES'):
                    # Extract the list
                    match = re.search(r'\[(.+)\]', line)
                    if match:
                        return [s.strip().strip('"').strip("'")
                                for s in match.group(1).split(',')]
    except FileNotFoundError:
        pass
    return ['OFRA', 'PA', 'OA', 'OFAV', 'AL', 'MC', 'AA']


ALL_TARGET_SPECIES = _load_target_species()


def _is_review(pt):
    """A point flagged for outside expert review: segmented into its own mask
    downstream and shipped to the expert-review site, never used for training."""
    return bool(pt.get('review')) or pt.get('species_code', '') == 'REVIEW'


def _load_target_label_info():
    """Return [{code, name, category}, ...] for each target label.

    Prefers master_codes_recoded.csv from the same dir as the recoded
    all_points.csv. Falls back to scanning all_points for the first row
    matching each code. Always returns an entry for every target (with
    empty name/category if not found) so the UI has a stable index.
    """
    lookup = {}  # code -> {name, category}

    # Prefer master_codes_recoded.csv
    ap_path = _find_recoded_all_points()
    ap_dir = os.path.dirname(ap_path)
    master_candidates = [
        os.path.join(ap_dir, 'master_codes_recoded.csv'),
        os.path.join(ap_dir, 'master_codes.csv'),
        os.path.join(REPO_DIR, 'output', 'master_codes.csv'),
    ]
    for mc in master_candidates:
        if not os.path.exists(mc):
            continue
        try:
            with open(mc, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get('species_code') or row.get('code') or '').strip()
                    if not code:
                        continue
                    lookup[code] = {
                        'name': (row.get('species_name') or row.get('name') or '').strip(),
                        'category': (row.get('category') or '').strip(),
                    }
            if lookup:
                log(f"Loaded {len(lookup)} codes from {os.path.basename(mc)}")
                break
        except Exception as e:
            log(f"Failed to read {mc}: {e}")

    # Fallback: scan all_points for any missing codes
    missing = [c for c in ALL_TARGET_SPECIES if c not in lookup]
    if missing and os.path.exists(ap_path):
        try:
            still_missing = set(missing)
            with open(ap_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = (row.get('species_code') or '').strip()
                    if code in still_missing:
                        lookup[code] = {
                            'name': (row.get('species_name') or '').strip(),
                            'category': (row.get('category') or '').strip(),
                        }
                        still_missing.discard(code)
                        if not still_missing:
                            break
        except Exception as e:
            log(f"Failed to scan {ap_path} for label info: {e}")

    out = []
    for code in ALL_TARGET_SPECIES:
        info = lookup.get(code, {'name': '', 'category': ''})
        out.append({'code': code, 'name': info['name'], 'category': info['category']})
    return out


def _load_species_remap():
    """Build old_code -> new_code dict from recodeSpecies remap_log."""
    remap = {}
    import glob as _glob
    import json as _json

    # Pipeline orchestrator override: check env var for remap_log path
    env_remap = os.environ.get('TCRMP_REMAP_LOG', '')
    if env_remap and os.path.isfile(env_remap):
        logs = [env_remap]
    else:
        recode_dir = os.path.join(REPO_DIR, 'TCRMPcvr_recodeSpecies', 'output')
        logs = sorted(_glob.glob(os.path.join(recode_dir, 'remap_log_*.json')))

    if not logs:
        return remap
    with open(logs[-1]) as f:
        data = _json.load(f)
    for entry in data.get('remaps', []):
        old = entry.get('old_code', '')
        new = entry.get('new_code', '')
        if old and new and old != new:
            remap[old] = new
    print(f"Loaded species remap: {len(remap)} entries from {os.path.basename(logs[-1])}")
    return remap


SPECIES_REMAP = _load_species_remap()

# ── Session state ──────────────────────────────────────────────

session = {
    'selected_frames_path': '',
    'clip_dir': '',
    'cpc_dir': '',
    'export_dir': '',
    'species_filter': [],       # species codes to include (empty = all)
    'review_batch_size': 10,
    'configured': False,
    'phase': 'startup',         # startup | processing | review
    'all_images': [],           # full list of selected frames
    'processing_queue': [],     # OCR frames not yet processed
    'processing_idx': 0,
    'annotations_by_year': {},  # year -> {filename: detection_dict}
    'review_files': [],         # ordered list of unreviewed filenames
    'review_offset': 0,
    'species_lookup': None,
    'flush_counter': 0,
    'flush_interval': 50,
}

LOG_LINES = {}


def log(msg, year=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    if year is not None:
        LOG_LINES.setdefault(year, []).append(line)


def flush_logs(export_dir):
    for year, lines in LOG_LINES.items():
        if not lines:
            continue
        log_path = os.path.join(export_dir, str(year), 'log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            for line in lines:
                f.write(line + '\n')
    LOG_LINES.clear()


# ── Detection DB persistence (JSON per year) ─────────────────

def _detections_path(export_dir, year):
    return os.path.join(export_dir, str(year), 'detections.json')


def _manifest_path(export_dir, year):
    return os.path.join(export_dir, str(year), 'processed_manifest.json')


def load_detections(export_dir, year):
    path = _detections_path(export_dir, year)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_detections(export_dir, year, detections):
    path = _detections_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(detections, f, indent=2)


def save_manifest(export_dir, year, processed_set):
    path = _manifest_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'processed': sorted(processed_set),
            'last_updated': datetime.now().isoformat(),
            'count': len(processed_set),
        }, f, indent=2)


def flush_all():
    export_dir = session['export_dir']
    if not export_dir:
        return
    for year, dets in session['annotations_by_year'].items():
        save_detections(export_dir, year, dets)
        save_manifest(export_dir, year, set(dets.keys()))
    flush_logs(export_dir)


# ── Species enrichment ───────────────────────────────────────

def _find_recoded_all_points():
    """Find the most recent recoded all_points.csv."""
    if cfg.ALL_POINTS_OVERRIDE and os.path.exists(cfg.ALL_POINTS_OVERRIDE):
        return cfg.ALL_POINTS_OVERRIDE
    recode_dir = os.path.join(REPO_DIR, 'TCRMPcvr_recodeSpecies', 'output')
    files = sorted(glob.glob(os.path.join(recode_dir, 'all_points_*.csv')))
    if files:
        return files[-1]
    return os.path.join(REPO_DIR, 'output', 'all_points.csv')


# Resolve target label info now that _find_recoded_all_points is defined
ALL_TARGET_LABELS = _load_target_label_info()


def _log_config(export_dir, frames_path, species_filter):
    """Write config snapshot to output for reproducibility."""
    cfg_log = {
        'timestamp': datetime.now().isoformat(),
        'selected_frames': frames_path,
        'clip_dir': DEFAULT_CLIP_DIR,
        'cpc_dir': DEFAULT_CPC_DIR,
        'export_dir': export_dir,
        'all_points': _find_recoded_all_points(),
        'species_filter': species_filter,
        'target_species': ALL_TARGET_SPECIES,
    }
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, 'config_log.json')
    with open(path, 'w') as f:
        json.dump(cfg_log, f, indent=2)
    log(f"Config logged to {path}")


_ROUTING_REPORT_FIELDS = [
    'frame_id', 'basename', 'year', 'route',
    'outcome', 'resolved_path', 'note',
    # OCR-failure reserve refill (Task 9): when an ocr_failed frame is replaced
    # by a reserve frame, the reserve's report row records which frame it stood
    # in for (refilled_from) and why (refill_reason).
    'refilled_from', 'refill_reason',
]


def _write_routing_report(export_dir, routing_report):
    """Write the per-frame routing report (JSON + CSV) next to config_log.json.

    One row per selected_frames row. outcome is one of:
      routed | image_missing | cpc_missing | ocr_failed | already_processed.
    This guarantees every selected frame is accounted for: a frame that did
    not route is recorded here (and mirrored into the /api/configure dropped[])
    instead of being silently swallowed by a counter.
    """
    os.makedirs(export_dir, exist_ok=True)

    json_path = os.path.join(export_dir, 'routing_report.json')
    with open(json_path, 'w') as f:
        json.dump(routing_report, f, indent=2)

    csv_path = os.path.join(export_dir, 'routing_report.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_ROUTING_REPORT_FIELDS)
        w.writeheader()
        for entry in routing_report:
            w.writerow({k: entry.get(k, '') for k in _ROUTING_REPORT_FIELDS})

    log(f"Routing report written: {json_path} ({len(routing_report)} frames)")


def _downgrade_routing_outcome(frame_id, outcome, note=''):
    """Downgrade a frame's routing outcome after the fact (e.g. OCR detected
    nothing -> ocr_failed) and keep the persisted report + dropped list in sync.

    The OCR queue provisionally records 'routed' at configure time because the
    detection has not run yet. When detect returns nothing we re-classify the
    frame as ocr_failed here so it is recorded (not silently lost) and surfaces
    in the dropped[] list the orchestrator consumes.
    """
    report = session.get('routing_report') or []
    dropped = session.get('routing_dropped') or []
    changed = False
    for entry in report:
        if entry.get('frame_id') == frame_id and entry.get('outcome') == 'routed':
            entry['outcome'] = outcome
            if note:
                entry['note'] = note
            if outcome not in ('routed', 'already_processed'):
                if not any(d.get('frame_id') == frame_id for d in dropped):
                    dropped.append({
                        'frame_id': frame_id,
                        'basename': entry.get('basename', ''),
                        'year': entry.get('year', ''),
                        'route': entry.get('route', ''),
                        'reason': outcome,
                    })
            changed = True
            break
    if changed:
        session['routing_report'] = report
        session['routing_dropped'] = dropped
        export_dir = session.get('export_dir')
        if export_dir:
            _write_routing_report(export_dir, report)


# ── OCR-failure reserve refill (Task 9) ──────────────────────

def _load_reserve(path):
    """Load reserve_frames.csv (Step 3 / build_reserve_rows schema) into a list
    of dicts, lowest reserve_rank first.

    Step 3 writes one eligible-but-unselected frame per row, tagging each with
    the target species it carries (semicolon-joined in the `species` column) and
    its already-resolved source_image / pts_image paths. When a 2020+ frame
    routes ocr_failed, Step 4 draws the next species-matched reserve from here to
    keep the per-species count whole. Returns [] when the file is absent (the
    honest-shortfall fallback) so a pipeline run without Step 3's reserve still
    behaves exactly as before.
    """
    if not path or not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize the semicolon-joined species tag into a set so the
                # species-match test below is a cheap intersection.
                sp = (row.get('species') or '').strip()
                row['_species_set'] = {
                    s.strip() for s in sp.split(';') if s.strip()
                }
                try:
                    row['_rank'] = int(float(row.get('reserve_rank', 0) or 0))
                except (ValueError, TypeError):
                    row['_rank'] = 0
                rows.append(row)
    except Exception as e:  # noqa: BLE001
        log(f"reserve-frames load failed ({path}): {e}")
        return []
    rows.sort(key=lambda r: r['_rank'])
    log(f"Loaded {len(rows)} reserve frames from {path}")
    return rows


def _reserve_key(row):
    """Stable identity for a reserve row so it isn't pulled twice."""
    return row.get('frame_id') or row.get('basename') or str(row.get('_rank'))


def _pull_reserve_frame(species_present):
    """Return the next unused reserve frame as an OCR-queue item, or None.

    Selection: lowest reserve_rank unused row whose species intersect
    `species_present`; if none match, fall back to any unused row (keeps the
    overall count whole even when no same-species reserve remains). A reserve
    whose _pts image cannot be resolved on disk is skipped (it is useless for
    OCR) but still marked used so we don't reconsider it every failure. The
    chosen row is marked used and resolved to the same dict shape as the
    configure-time ocr_queue.append({...}) block, with refilled_from left for
    the caller to stamp.
    """
    reserve = session.get('reserve_frames') or []
    used = session.setdefault('reserve_used', set())
    clip_index = session.get('clip_index', {})
    want = set(species_present or [])

    # OCR failures can only be refilled by a frame that itself has a _pts image
    # for OCR. CPC reserve rows (route=='cpc') have no _pts and will always be
    # skipped by _resolve_pts, so they wastefully burn through the candidate
    # list before a usable 2020+ row is found. Prefer ocr_needed rows first;
    # only fall back to non-ocr_needed rows if none remain.
    def _ocr_candidates(pool):
        preferred = [r for r in pool if r.get('route') == 'ocr_needed']
        return preferred if preferred else pool

    unused = [r for r in reserve if _reserve_key(r) not in used]
    # Species-matched candidates first (rank order), then any unused row.
    matched = [r for r in unused if r['_species_set'] & want]
    if want and matched:
        candidates = _ocr_candidates(matched)
    else:
        candidates = _ocr_candidates(unused)

    for row in candidates:
        key = _reserve_key(row)
        basename = (row.get('basename') or '').strip()
        if not basename:
            used.add(key)
            continue
        # Trust Step 3's resolved pts_image/source_image, else the clip index.
        pts_path = _resolve_pts(row, basename, clip_index)
        raw_path = _resolve_raw(row, basename, clip_index)
        if not pts_path:
            # No usable _pts on disk: burn this reserve and try the next one.
            used.add(key)
            log(f"reserve {basename}: no _pts image resolvable, skipping")
            continue

        used.add(key)
        try:
            year = int(float(row.get('year') or 0))
        except (ValueError, TypeError):
            year = 0
        date_str = str(row.get('date') or '').replace('-', '')
        site = row.get('site', '')
        try:
            transect = int(float(row.get('transect') or 0))
            frame = int(float(row.get('frame') or 0))
            t_id = f"{transect}{frame:02d}"
        except (ValueError, TypeError):
            t_id = ''
        return {
            'pts_path': pts_path,
            'raw_path': raw_path,
            'filename': f"{basename}_pts.jpg",
            'year': year,
            'date_str': date_str,
            'site': site,
            't_id': t_id,
            'frame_id': f"{basename}@{year}",
            'basename': basename,
            'species_present': sorted(row['_species_set']),
            'refilled_from': '',
        }
    return None


def _record_refill_entry(repl):
    """Append a routing-report row for a refilled reserve frame and persist.

    The reserve's row is recorded as routed (it now occupies the OCR queue slot
    of the frame it replaced) and carries refill provenance so the operator can
    see exactly which ocr_failed frame each reserve stood in for.
    """
    report = session.get('routing_report')
    if report is None:
        return
    # Idempotent: don't double-record if the same reserve is pulled again.
    if any(e.get('frame_id') == repl['frame_id']
           and e.get('refilled_from') for e in report):
        return
    report.append({
        'frame_id': repl['frame_id'],
        'basename': repl['basename'],
        'year': repl['year'],
        'route': 'ocr_needed',
        'outcome': 'routed',
        'resolved_path': repl.get('pts_path', '') or '',
        'note': f"refill for {repl.get('refilled_from', '')}",
        'refilled_from': repl.get('refilled_from', ''),
        'refill_reason': repl.get('refill_reason', 'ocr_failed'),
    })
    session['routing_report'] = report
    export_dir = session.get('export_dir')
    if export_dir:
        _write_routing_report(export_dir, report)


def _load_species_lookup():
    csv_path = _find_recoded_all_points()
    if not os.path.exists(csv_path):
        log("WARNING: all_points.csv not found — species codes will be empty")
        return None
    log(f"Loading RECODED species data from {csv_path}")
    sp = SpeciesLookup(csv_path)
    log(f"Loaded {len(sp)} species records")
    return sp


def enrich_with_species(points, date_str, site, t_id, species_lookup):
    """Add species_code, species_name, category from recoded all_points."""
    if not species_lookup or not date_str:
        return points

    lookup_date = ocr_date_to_lookup_date(date_str)
    transect, frame = parse_transect_frame(t_id) if t_id else (0, 0)

    for pt in points:
        sp, matched_date, exact = species_lookup.lookup(
            lookup_date, site, transect, frame, pt['label']
        )
        if sp:
            pt['species_code'] = sp['species_code']
            pt['species_name'] = sp['species_name']
            pt['category'] = sp['category']
        else:
            pt.setdefault('species_code', '')
            pt.setdefault('species_name', '')
            pt.setdefault('category', '')

    return points


# ── CPC loading ──────────────────────────────────────────────

def load_cpc_points(basename, year, cpc_dir, species_lookup):
    """Load (x,y) coords from CPC file directly, using border-based scaling.

    Reads the .cpc file from CLIP_DIR to get raw coords + border rectangle,
    then scales to image pixel space. Falls back to point_coords.csv if no
    CPC file is found.
    """
    from PIL import Image as PILImage

    # Resolve the .cpc file via the prebuilt whole-tree index (season-agnostic:
    # the .cpc may sit in a season folder that does not match the date year).
    cpc_file_path = session.get('cpc_index', {}).get(basename)

    if cpc_file_path:
        # ── Read CPC file directly ──
        with open(cpc_file_path, 'r', errors='ignore') as f:
            lines = [l.strip() for l in f.readlines()]

        # Header: code_path, image_path, canvas_w, canvas_h, img_area_w, img_area_h
        import ntpath
        header_parts = lines[0].replace('"', '').split(',')
        cpc_header_img = ntpath.basename(header_parts[1].strip()) if len(header_parts) > 1 else ''
        cpc_header_stem = os.path.splitext(cpc_header_img)[0]

        # Verify name match (allow SCTLD<->TCRMP prefix swap)
        def _strip_prefix(s):
            for pfx in ('TCRMP', 'SCTLD', 'CIG'):
                if s.startswith(pfx):
                    return s[len(pfx):]
            return s

        if _strip_prefix(cpc_header_stem) != _strip_prefix(basename):
            log(f"WARNING: CPC header image '{cpc_header_stem}' != basename '{basename}', skipping CPC file")
            cpc_file_path = None

    if cpc_file_path:
        # Border corners: line1=BL, line2=BR, line3=TR, line4=TL
        tl = [float(v) for v in lines[4].split(',')]
        bl = [float(v) for v in lines[1].split(',')]
        tr = [float(v) for v in lines[3].split(',')]
        border_w = tr[0] - tl[0]
        border_h = bl[1] - tl[1]

        if border_w <= 0 or border_h <= 0:
            log(f"WARNING: Invalid CPC borders for {basename}, falling back to CSV")
            cpc_file_path = None

    if cpc_file_path:
        n_points = int(lines[5])
        label_start = 6 + n_points

        # Get image dimensions for scaling (season-agnostic clip index).
        raw_path = session.get('clip_index', {}).get(basename)
        if not raw_path:
            log(f"WARNING: No raw image for {basename}, cannot scale CPC coords")
            return []
        with PILImage.open(raw_path) as img:
            img_w, img_h = img.size

        # CPC file found but we use cpc_all for coords+species — fall through
        pass

    # ── Read from cpc_all/point_coords.csv (correct coords + species) ──
    csv_path = os.path.join(cpc_dir, str(year), 'ids', 'point_coords.csv')
    if not os.path.exists(csv_path):
        return []

    points = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_stem = os.path.splitext(row['raw_image'])[0]
            if raw_stem != basename:
                continue
            px, py = float(row['x']), float(row['y'])
            # Apply species recode (e.g. MFAV->OFAV, MA->OA)
            sp_code = row.get('species_code', '')
            sp_code = SPECIES_REMAP.get(sp_code, sp_code)
            points.append({
                'label': row['label'],
                'x': px,
                'y': py,
                'letter_cx': px,
                'letter_cy': py - 50,
                'species_code': sp_code,
                'species_name': row.get('species_name', ''),
                'category': row.get('category', ''),
                'ocr_confidence': 100,
                'has_crosshair': True,
                'source': 'cpc',
            })

    return points


# ── OCR detection ────────────────────────────────────────────

def run_detection(image_path, filename, date_str, site, t_id):
    """Run OCR detection and enrich with recoded species data."""
    annotations = detect_annotations(image_path, gpu=USE_GPU)
    if not annotations:
        return []

    points = []
    for letter in sorted(annotations.keys()):
        ann = annotations[letter]
        points.append({
            'label': letter,
            'x': round(ann['cross_x'], 1),
            'y': round(ann['cross_y'], 1),
            'letter_cx': round(ann['letter_cx'], 1),
            'letter_cy': round(ann['letter_cy'], 1),
            'has_crosshair': ann['has_crosshair'],
            'ocr_confidence': ann['ocr_conf'],
            'species_code': '',
            'species_name': '',
            'category': '',
            'source': 'ocr',
        })

    enrich_with_species(points, date_str, site, t_id, session['species_lookup'])
    return points


# ── Image path finding ───────────────────────────────────────

def _clip_path_rank(path):
    """Lower rank = more canonical. When the same frame stem exists in more than
    one place (a high-res JPEG/ re-export or an _edit variant alongside the
    original), prefer the flat original so the displayed image matches the one
    the CPC/OCR points were computed for. Deterministic so Step 3 and Step 4
    resolve to the same file."""
    parts = path.split(os.sep)
    noncanonical = any(p == 'JPEG' or p.endswith('_edit') for p in parts[:-1])
    return (1 if noncanonical else 0, len(parts))


def _persisted_file_list(clip_dir):
    """All file paths under clip_dir via the shared persisted index
    (scripts/clip_index.py: one walk, dir-mtime staleness sentinel, auto
    rebuild). Returns None when the module is unavailable or errors, so
    _walk_index keeps its original walk as the fallback. The routing
    configure calls _walk_index twice (jpg + .cpc); with the index both
    calls are dict lookups over one cached listing instead of two full
    167k-file tree walks."""
    try:
        scripts_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        if scripts_root not in sys.path:
            sys.path.insert(0, scripts_root)
        import clip_index
        return clip_index.list_files(clip_dir)
    except Exception:
        return None


def _walk_index(clip_dir, exts):
    """Whole-tree scan: stem -> canonical path for the given extensions.
    Basenames are globally unique per logical frame, so this resolves a frame
    regardless of which TCRMP{season}_clip / period subfolder it lives in.
    Hidden dirs + dot-files skipped; collisions resolve to the most canonical
    copy (see _clip_path_rank). Served from the persisted clip index when
    current (Task 1.4, 2026-08-26); falls back to the original walk."""
    index = {}
    if not clip_dir or not os.path.isdir(clip_dir):
        return index
    paths = _persisted_file_list(clip_dir)
    if paths is None:
        paths = []
        for root, dirs, files in os.walk(clip_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fn in files:
                if not fn.startswith('.'):
                    paths.append(os.path.join(root, fn))
    for path in paths:
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() in exts:
            prev = index.get(stem)
            if prev is None or _clip_path_rank(path) < _clip_path_rank(prev):
                index[stem] = path
    return index


def build_clip_index(clip_dir):
    """Whole-tree image index (jpg/jpeg). See _walk_index."""
    return _walk_index(clip_dir, ('.jpg', '.jpeg'))


def build_cpc_index(clip_dir):
    """Whole-tree .cpc index, so the per-frame CPC lookup is an O(1) dict hit
    instead of a full-tree recursive glob per frame."""
    return _walk_index(clip_dir, ('.cpc',))


def _csv_image_path(row, col):
    """Read a path column from a selected_frames row, treating blanks/NaN as
    absent. pandas reads empty CSV cells as float('nan')."""
    if not hasattr(row, 'get'):
        return ''
    v = row.get(col)
    if v is None:
        return ''
    s = str(v).strip()
    return '' if s.lower() == 'nan' else s


def _resolve_raw(row, basename, clip_index):
    """Trust Step 3's already-resolved source_image when it names a real file;
    else resolve by basename from the recursive clip index."""
    p = _csv_image_path(row, 'source_image')
    if p and os.path.isfile(p):
        return p
    return clip_index.get(basename)


def _resolve_pts(row, basename, clip_index):
    """Trust Step 3's pts_image when real; else resolve {basename}_pts from index."""
    p = _csv_image_path(row, 'pts_image')
    if p and os.path.isfile(p):
        return p
    return clip_index.get(f'{basename}_pts')


def _basename_to_raw_name(basename):
    """Convert basename to likely raw filename with extension."""
    return basename + '.jpeg'


# ── Review helpers ───────────────────────────────────────────

def _build_review_list():
    """Build the review queue.

    Rules:
      - Scrapped frames: excluded (they stay excluded forever).
      - Exported frames: excluded (they stay excluded until user un-scraps /
        un-exports).
      - Reviewed-but-not-exported frames: pushed to the END of the queue so
        the user sees fresh frames first on each rebuild.

    Shuffle behavior: when `session['shuffle']` is set, we shuffle once per
    session and preserve that ordering across subsequent rebuilds (which
    happen after every export/scrap). Newly-appearing frames get appended
    (shuffled amongst themselves) rather than causing a global re-shuffle
    that would re-surface frames the reviewer already looked at.
    """
    import random as _random

    fresh = []     # not reviewed and not exported
    reviewed = []  # reviewed but not exported (bottom of queue)
    for year in sorted(session['annotations_by_year']):
        dets = session['annotations_by_year'][year]
        for fn in sorted(dets):
            det = dets[fn]
            if det.get('scrapped', False):
                continue
            if det.get('exported', False):
                continue
            if det.get('reviewed', False):
                reviewed.append(fn)
            else:
                fresh.append(fn)

    if session.get('shuffle'):
        prior = session.get('shuffled_order') or []
        prior_set = set(prior)
        # 1) Keep prior ordering for frames that are still fresh (stable).
        fresh_set = set(fresh)
        fresh_kept = [f for f in prior if f in fresh_set]
        fresh_new = [f for f in fresh if f not in prior_set]
        if fresh_new:
            _random.shuffle(fresh_new)
        fresh_ordered = fresh_kept + fresh_new
        # 2) Also shuffle the reviewed-pile once (stably across rebuilds).
        rev_set = set(reviewed)
        rev_prior_key = 'shuffled_reviewed_order'
        rev_prior = session.get(rev_prior_key) or []
        rev_prior_set = set(rev_prior)
        rev_kept = [f for f in rev_prior if f in rev_set]
        rev_new = [f for f in reviewed if f not in rev_prior_set]
        if rev_new:
            _random.shuffle(rev_new)
        rev_ordered = rev_kept + rev_new
        # 3) Cache for next rebuild and emit final review list.
        session['shuffled_order'] = fresh_ordered
        session[rev_prior_key] = rev_ordered
        session['review_files'] = fresh_ordered + rev_ordered
    else:
        session['review_files'] = fresh + reviewed
    session['review_offset'] = 0


def _find_detection(filename):
    for year, dets in session['annotations_by_year'].items():
        if filename in dets:
            return dets[filename], year
    return None, None


# ── Routes ───────────────────────────────────────────────────

@app.route('/')
def index():
    # Orchestrator can pre-seed these options so the auto-boot respects what
    # the user picked in the Step 4 panel (no more silently-ignored choices).
    _env_ref = os.environ.get('TCRMP_REFERENCE_MODE', '') == '1'
    _env_shuf = os.environ.get('TCRMP_SHUFFLE', '') == '1'
    _env_batch = os.environ.get('TCRMP_REVIEW_BATCH_SIZE', '10') or '10'

    return render_template('index.html',
                           target_species=ALL_TARGET_SPECIES,
                           target_labels=ALL_TARGET_LABELS,
                           default_frames=DEFAULT_SELECTED_FRAMES,
                           default_export=DEFAULT_EXPORT_DIR,
                           orchestrated=bool(os.environ.get('TCRMP_ORCHESTRATOR_URL')),
                           orchestrator_url=os.environ.get('TCRMP_ORCHESTRATOR_URL', '').rstrip('/'),
                           default_reference_mode=_env_ref,
                           default_shuffle=_env_shuf,
                           default_batch_size=_env_batch)


@app.route('/api/configure', methods=['POST'])
def configure():
    """Load selected_frames.csv, split CPC/OCR, build processing state."""
    data = request.json
    frames_path = data.get('selected_frames', DEFAULT_SELECTED_FRAMES)
    export_dir = data.get('export_dir', DEFAULT_EXPORT_DIR)
    species_filter = data.get('species_filter', [])
    batch_size = data.get('review_batch_size', 10)

    # If TARGET_SPECIES_ONLY is set, auto-apply target species as filter
    if getattr(cfg, 'TARGET_SPECIES_ONLY', 0) and not species_filter:
        species_filter = list(ALL_TARGET_SPECIES)
        log(f"TARGET_SPECIES_ONLY=1: auto-filtering to {species_filter}")

    session['selected_frames_path'] = frames_path
    # Honor an explicit clip_dir from the caller; else the env/config default.
    session['clip_dir'] = data.get('clip_dir') or DEFAULT_CLIP_DIR
    # One-pass recursive indexes of the whole clip tree (season-agnostic), so
    # routing resolves every frame + .cpc by basename regardless of season
    # folder, in O(1) instead of a per-frame glob.
    session['clip_index'] = build_clip_index(session['clip_dir'])
    session['cpc_index'] = build_cpc_index(session['clip_dir'])
    log(f"Indexed {len(session['clip_index'])} clip images + "
        f"{len(session['cpc_index'])} .cpc files under {session['clip_dir']}")
    session['cpc_dir'] = DEFAULT_CPC_DIR
    session['export_dir'] = os.path.abspath(export_dir)
    session['species_filter'] = species_filter
    session['review_batch_size'] = batch_size
    session['reference_mode'] = bool(data.get('reference_mode', False))
    session['shuffle'] = bool(data.get('shuffle', False))

    # OCR-failure reserve pool (Task 9): Step 3 drops reserve_frames.csv next to
    # selected_frames.csv (or the orchestrator points at it via env). When a
    # 2020+ frame routes ocr_failed, _pull_reserve_frame draws the next
    # species-matched reserve to keep the per-species count whole. Absent file
    # => empty pool => existing honest-shortfall behavior is unchanged.
    reserve_path = (
        os.environ.get('TCRMP_RESERVE_FRAMES')
        or os.path.join(os.path.dirname(frames_path), 'reserve_frames.csv')
    )
    session['reserve_frames'] = _load_reserve(reserve_path)
    session['reserve_used'] = set()

    os.makedirs(session['export_dir'], exist_ok=True)

    # Load aux_points from added_points.csv if it exists (persists across sessions).
    session['aux_by_year'] = _load_aux_points()

    # Log config for reproducibility
    _log_config(session['export_dir'], frames_path, species_filter)

    # Load species lookup (recoded)
    session['species_lookup'] = _load_species_lookup()

    # Load selected frames
    if not os.path.exists(frames_path):
        return jsonify({'error': f'File not found: {frames_path}'}), 404

    df = pd.read_csv(frames_path)
    log(f"Loaded {len(df)} selected frames from {frames_path}")

    # Apply species filter
    if species_filter:
        mask = df['species_present'].apply(
            lambda s: any(sp in str(s).split(';') for sp in species_filter)
        )
        df = df[mask]
        log(f"Species filter [{', '.join(species_filter)}]: {len(df)} frames remain")

    # Load existing detections from export dir.
    #
    # NOTE: we deliberately do NOT pre-create routed_input/<year>/ dirs here
    # for every year in the selection. A year whose frames all drop (missing
    # images / missing CPC) must NOT leave behind an empty stamped year dir
    # that looks like a real route. We only touch a year dir if it already
    # holds detections to load, and the export path creates real year dirs
    # on demand when a frame actually routes.
    session['annotations_by_year'] = {}
    already_processed = set()

    for year in df['year'].unique():
        dets = load_detections(session['export_dir'], int(year))
        if dets:
            session['annotations_by_year'][int(year)] = dets
            already_processed.update(dets.keys())

    # Per-frame routing accountability: every selected_frames row gets exactly
    # one routing_report entry so no frame is ever silently lost. The
    # reconciliation assertion below proves that.
    routing_report = []      # one dict per selected row
    dropped = []             # subset of routing_report that did not route

    def _record(frame_id, basename, year, route, outcome,
                resolved_path='', note=''):
        entry = {
            'frame_id': frame_id,
            'basename': basename,
            'year': year,
            'route': route,
            'outcome': outcome,
            'resolved_path': resolved_path or '',
            'note': note or '',
        }
        routing_report.append(entry)
        if outcome not in ('routed', 'already_processed'):
            dropped.append({
                'frame_id': frame_id,
                'basename': basename,
                'year': year,
                'route': route,
                'reason': outcome,
            })
        return entry

    # Process each frame
    cpc_loaded = 0
    ocr_queue = []
    cpc_missing = 0
    image_missing = 0
    ocr_failed = 0
    already_processed_hits = 0

    for ridx, row in df.iterrows():
        basename = row['basename']
        year = int(row['year'])
        route = row['route']
        date_str = str(row['date']).replace('-', '')
        site = row['site']
        transect = int(row['transect'])
        frame = int(row['frame'])
        t_id = f"{transect}{frame:02d}"
        frame_id = f"{basename}@{year}"

        # Build a _pts filename for OCR frames
        pts_filename = f"{basename}_pts.jpg"
        # For CPC frames, use the raw filename as the key
        det_key = pts_filename if route == 'ocr_needed' else f"{basename}_cpc.json"

        if det_key in already_processed:
            already_processed_hits += 1
            _record(frame_id, basename, year, route, 'already_processed',
                    note='detection already present in export dir')
            continue

        if route == 'cpc':
            # Load CPC points (with recoded species)
            points = load_cpc_points(
                basename, year, session['cpc_dir'], session['species_lookup'])

            if not points:
                cpc_missing += 1
                _record(frame_id, basename, year, route, 'cpc_missing',
                        note='no CPC points found for basename')
                continue

            # Resolve raw image: trust Step 3's source_image, else clip index.
            raw_path = _resolve_raw(row, basename, session['clip_index'])

            detection = {
                'pts_path': raw_path or '',  # CPC uses raw image for display
                'raw_path': raw_path or '',
                'points': points,
                'date_str': date_str,
                'site': site,
                't_id': t_id,
                'year': year,
                'processed_at': datetime.now().isoformat(),
                'reviewed': False,
                'exported': False,
                'source_type': 'cpc',
            }

            session['annotations_by_year'].setdefault(year, {})[det_key] = detection
            cpc_loaded += 1
            _record(frame_id, basename, year, route, 'routed',
                    resolved_path=raw_path or '',
                    note='' if raw_path else 'CPC points routed but raw image not found')

        elif route == 'ocr_needed':
            # Resolve _pts + raw: trust Step 3's resolved paths, else clip index.
            pts_path = _resolve_pts(row, basename, session['clip_index'])
            raw_path = _resolve_raw(row, basename, session['clip_index'])

            if not pts_path:
                # image_missing: no _pts image on disk at all.
                image_missing += 1
                _record(frame_id, basename, year, route, 'image_missing',
                        note='no _pts image found on disk')
                continue

            # species_present (semicolon-joined in selected_frames) is carried on
            # the queue item so an ocr_failed refill can pick a species-matched
            # reserve frame (Task 9).
            sp_present = [
                s.strip() for s in str(row.get('species_present', '')).split(';')
                if s.strip() and s.strip().lower() != 'nan'
            ]
            ocr_queue.append({
                'pts_path': pts_path,
                'raw_path': raw_path,
                'filename': pts_filename,
                'year': year,
                'date_str': date_str,
                'site': site,
                't_id': t_id,
                'frame_id': frame_id,
                'basename': basename,
                'species_present': sp_present,
            })
            # OCR frames route into the queue here; whether OCR actually
            # detects anything (ocr_failed) is resolved when the queue runs
            # (see _record_ocr_outcome). We provisionally mark routed so the
            # reconciliation below balances; a later ocr_failed re-records it.
            _record(frame_id, basename, year, route, 'routed',
                    resolved_path=pts_path or '')

        else:
            # Unknown route value: record rather than silently skip.
            _record(frame_id, basename, year, route or '', 'image_missing',
                    note=f'unrecognized route value: {route!r}')

    session['processing_queue'] = ocr_queue
    session['processing_idx'] = 0
    # Stash the report so the OCR phase can downgrade routed -> ocr_failed and
    # so a final reconciliation can run after OCR completes.
    session['routing_report'] = routing_report
    session['routing_dropped'] = dropped

    if len(ocr_queue) == 0:
        session['phase'] = 'review'
        _build_review_list()
    else:
        session['phase'] = 'processing'

    # Flush CPC detections immediately
    flush_all()

    session['configured'] = True

    # ── Routing reconciliation ──
    # routed: CPC frames loaded + OCR frames queued (provisionally routed).
    # already_processed_hits: rows skipped because their detection already
    #   exists in the export dir (counted per selected row, NOT the loaded set).
    # len(dropped): every recorded non-routing outcome.
    # These three must partition len(df) exactly: no frame is unaccounted for.
    routed_count = cpc_loaded + len(ocr_queue)
    recon = routed_count + already_processed_hits + len(dropped)
    if recon != len(df):
        log("ROUTING RECONCILIATION MISMATCH: "
            f"routed({routed_count}) + already_processed({already_processed_hits}) "
            f"+ dropped({len(dropped)}) = {recon} != selected rows ({len(df)}). "
            "A selected frame was neither routed nor recorded as dropped.")
    else:
        log(f"Routing reconciliation OK: {recon} == {len(df)} selected rows.")

    # Write the per-frame routing report next to config_log.json so the
    # operator (and _route_drive) can see exactly what happened to every frame.
    _write_routing_report(session['export_dir'], routing_report)

    log(f"CPC loaded: {cpc_loaded}, already processed: {already_processed_hits}, "
        f"OCR queue: {len(ocr_queue)}, CPC missing: {cpc_missing}, "
        f"image missing: {image_missing}, dropped total: {len(dropped)}")

    return jsonify({
        'total_frames': len(df),
        'cpc_loaded': cpc_loaded,
        'already_processed': already_processed_hits,
        'ocr_remaining': len(ocr_queue),
        'cpc_missing': cpc_missing,
        'image_missing': image_missing,
        'ocr_failed': ocr_failed,
        'dropped': dropped,
        'reconciled': recon == len(df),
        'phase': session['phase'],
        'review_count': len(session.get('review_files', [])),
        'species_filter': species_filter,
    })


@app.route('/api/resume', methods=['POST'])
def resume():
    """Continue existing project from export directory."""
    data = request.json
    session['export_dir'] = os.path.abspath(data['export_dir'])
    session['review_batch_size'] = data.get('review_batch_size', 10)
    session['reference_mode'] = bool(data.get('reference_mode', False))
    session['shuffle'] = bool(data.get('shuffle', False))
    session['aux_by_year'] = _load_aux_points()

    if not os.path.isdir(session['export_dir']):
        return jsonify({'error': 'Export directory not found'}), 404

    export_dir = session['export_dir']
    session['species_lookup'] = _load_species_lookup()

    # Discover years
    discovered_years = []
    for entry in sorted(os.listdir(export_dir)):
        entry_path = os.path.join(export_dir, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            yr = int(entry)
            if os.path.exists(_detections_path(export_dir, yr)):
                discovered_years.append(yr)

    if not discovered_years:
        return jsonify({'error': 'No detection data found'}), 404

    session['annotations_by_year'] = {}
    total = 0
    year_stats = {}

    for year in discovered_years:
        dets = load_detections(export_dir, year)
        session['annotations_by_year'][year] = dets
        total += len(dets)
        n_exp = sum(1 for d in dets.values() if d.get('exported', False))
        year_stats[year] = {
            'total': len(dets), 'exported': n_exp,
            'remaining_review': len(dets) - n_exp,
        }

    session['phase'] = 'review'
    session['processing_queue'] = []
    _build_review_list()
    session['configured'] = True

    return jsonify({
        'total_processed': total,
        'years': discovered_years,
        'phase': 'review',
        'year_stats': year_stats,
        'review_count': len(session['review_files']),
    })


def _process_one_item(item, position_label=''):
    """Run OCR for a single queue item, record the detection, and — on
    ocr_failed — pull a reserve frame in to keep the per-species count whole.

    Shared by the live /api/process route and the headless
    run_ocr_queue_for_test() seam so both drive identical logic (detection +
    ocr_failed downgrade + reserve refill). Returns the number of detected
    points. A pulled-in reserve is appended to session['processing_queue'] so it
    gets its own OCR pass in turn (the queue-driven loop sees len(queue) grow).
    """
    pts_path = item['pts_path']
    filename = item['filename']
    year = item['year']

    if position_label:
        log(f"OCR {position_label}: {filename}", year=year)
    else:
        log(f"OCR: {filename}", year=year)
    points = run_detection(pts_path, filename, item['date_str'],
                           item['site'], item['t_id'])

    # The image was found on disk (it routed into the OCR queue) but detection
    # returned nothing: record ocr_failed so the frame is accounted for and
    # surfaces in dropped[] rather than disappearing as an empty detection.
    if not points and item.get('frame_id'):
        _downgrade_routing_outcome(
            item['frame_id'], 'ocr_failed',
            note='image found but OCR detected no annotations')

        # Reserve refill (Task 9): swap in the next species-matched reserve so
        # the per-species target stays whole. If the reserve is exhausted (or
        # absent), _pull_reserve_frame returns None and we fall back to the
        # honest shortfall (the ocr_failed downgrade above stands).
        if not item.get('_is_refill'):
            repl = _pull_reserve_frame(item.get('species_present', []))
            if repl:
                repl['refill_reason'] = 'ocr_failed'
                repl['refilled_from'] = item.get('basename', '')
                repl['_is_refill'] = True
                session['processing_queue'].append(repl)
                _record_refill_entry(repl)
                log(f"Reserve refill: {repl['basename']} replaces "
                    f"{item.get('basename', '')} (ocr_failed)")

    detection = {
        'pts_path': pts_path,
        'raw_path': item['raw_path'],
        'points': points,
        'date_str': item['date_str'],
        'site': item['site'],
        't_id': item['t_id'],
        'year': year,
        'processed_at': datetime.now().isoformat(),
        'reviewed': False,
        'exported': False,
        'source_type': 'ocr',
    }

    session['annotations_by_year'].setdefault(year, {})[filename] = detection
    return len(points)


def run_ocr_queue_for_test():
    """Headless driver over session['processing_queue'] (no Flask request, no
    GPU required when the fixture's _pts images carry no red annotations).

    Mirrors the /api/process loop: walks the queue from processing_idx to the
    end, running _process_one_item on each. Because an ocr_failed refill appends
    a reserve item to the queue, the while-loop naturally consumes those too.
    Returns {'refilled': <count of routing-report rows with refilled_from>} so a
    test can assert a reserve was pulled in and recorded.
    """
    queue = session.get('processing_queue', [])
    idx = session.get('processing_idx', 0)
    while idx < len(queue):
        item = queue[idx]
        _process_one_item(item, position_label=f"{idx + 1}/{len(queue)}")
        idx += 1
        session['processing_idx'] = idx
    flush_all()
    session['phase'] = 'review'
    _build_review_list()
    report = session.get('routing_report') or []
    refilled = sum(1 for e in report if e.get('refilled_from'))
    return {'refilled': refilled,
            'processed': idx,
            'queue_total': len(queue)}


@app.route('/api/process', methods=['POST'])
def process_one():
    """Process ONE OCR image from the queue."""
    idx = session['processing_idx']
    queue = session['processing_queue']

    if idx >= len(queue):
        flush_all()
        session['phase'] = 'review'
        _build_review_list()
        return jsonify({
            'done': True, 'processed': idx, 'total': len(queue),
            'review_count': len(session['review_files']),
        })

    item = queue[idx]
    filename = item['filename']
    year = item['year']

    n_points = _process_one_item(item, position_label=f"{idx + 1}/{len(queue)}")

    session['processing_idx'] = idx + 1
    session['flush_counter'] += 1

    if session['flush_counter'] >= session['flush_interval']:
        flush_all()
        session['flush_counter'] = 0

    return jsonify({
        # len(queue) is read live: a reserve refill may have grown the queue,
        # so the UI's "total" reflects the added frame on the next poll.
        'done': False, 'processed': idx + 1, 'total': len(session['processing_queue']),
        'filename': filename, 'year': year, 'n_points': n_points,
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
    target_only = getattr(cfg, 'TARGET_SPECIES_ONLY', 0)
    images = []
    for fn in batch_files:
        det, year = _find_detection(fn)
        if det:
            pts = det.get('points', [])
            if target_only and ALL_TARGET_SPECIES:
                pts = [p for p in pts
                       if p.get('species_code', '') in ALL_TARGET_SPECIES
                       or _is_review(p)]
            images.append({
                'filename': fn,
                'year': year,
                'n_points': len(pts),
                'status': 'reviewed' if det.get('reviewed') else 'detected',
                'source_type': det.get('source_type', 'ocr'),
            })

    total_batches = (len(files) + batch_size - 1) // batch_size if files else 0
    current_batch = (offset // batch_size) + 1 if files else 0

    return jsonify({
        'images': images,
        'batch_number': current_batch,
        'total_batches': total_batches,
        'total_unreviewed': len(files),
        'offset': offset,
        'batch_size': batch_size,
    })


@app.route('/api/image/<path:filename>')
def get_image_data(filename):
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    points = det['points']
    # If TARGET_SPECIES_ONLY, only return target-species points — plus REVIEW
    # points, which must stay visible/editable so the reviewer can see what they
    # flagged for the expert (they round-trip via the same merge path below).
    if getattr(cfg, 'TARGET_SPECIES_ONLY', 0) and ALL_TARGET_SPECIES:
        points = [p for p in points
                  if p.get('species_code', '') in ALL_TARGET_SPECIES
                  or _is_review(p)]
    aux_points = session.get('aux_by_year', {}).get(year, {}).get(filename, [])
    return jsonify({
        'points': points,
        'status': 'reviewed' if det.get('reviewed') else 'detected',
        'year': year,
        'source_type': det.get('source_type', 'ocr'),
        'reference_mode': bool(session.get('reference_mode', False)),
        'aux_points': aux_points,
    })


def _aux_csv_path():
    export_dir = session.get('export_dir', '')
    if not export_dir:
        return ''
    return os.path.join(export_dir, 'added_points.csv')


def _load_aux_points():
    """Load existing added_points.csv on configure/resume."""
    path = _aux_csv_path()
    out = {}
    if not path or not os.path.exists(path):
        return out
    try:
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    year = int(float(row.get('year', 0) or 0))
                except (ValueError, TypeError):
                    continue
                fn = row.get('frame_filename', '')
                if not year or not fn:
                    continue
                # Coerce x/y to floats for rendering.
                try:
                    row['x'] = float(row.get('x', 0))
                    row['y'] = float(row.get('y', 0))
                except (ValueError, TypeError):
                    continue
                out.setdefault(year, {}).setdefault(fn, []).append(row)
    except Exception as e:
        log(f"aux-points load failed ({path}): {e}")
    return out


@app.route('/api/image/<path:filename>/add_aux', methods=['POST'])
def add_aux(filename):
    """Log a "side observation" on this frame. Written to a separate
    added_points.csv so it's easy for downstream users to see what the
    reviewer spotted beyond the main targets. Not touched by step 5.
    """
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'frame not found'}), 404
    data = request.json or {}

    row = {
        'frame_filename': filename,
        'year': year,
        'date': det.get('date_str', ''),
        'site': det.get('site', ''),
        'transect': det.get('t_id', ''),
        'x': round(float(data.get('x', 0)), 2),
        'y': round(float(data.get('y', 0)), 2),
        'species_code': (data.get('species_code') or '').strip(),
        'species_name': (data.get('species_name') or '').strip(),
        'category': (data.get('category') or 'Aux').strip(),
        'notes': (data.get('notes') or '').strip(),
        'added_at': datetime.now().isoformat(),
    }

    session.setdefault('aux_by_year', {}).setdefault(year, {}).setdefault(filename, []).append(row)

    path = _aux_csv_path()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if need_header:
                writer.writeheader()
            writer.writerow(row)

    return jsonify({'ok': True, 'point': row})


@app.route('/api/image/<path:filename>/delete_aux', methods=['POST'])
def delete_aux(filename):
    """Remove an aux point by its added_at timestamp (unique enough in practice)."""
    data = request.json or {}
    stamp = data.get('added_at', '')
    if not stamp:
        return jsonify({'error': 'added_at is required'}), 400

    removed = 0
    aux_by_year = session.get('aux_by_year', {})
    for year, by_file in aux_by_year.items():
        pts = by_file.get(filename, [])
        kept = [p for p in pts if p.get('added_at') != stamp]
        removed += len(pts) - len(kept)
        by_file[filename] = kept

    # Rewrite CSV without the removed row(s). Simpler than in-place edit.
    path = _aux_csv_path()
    if path and os.path.exists(path):
        rows = []
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for r in reader:
                    if r.get('added_at') != stamp:
                        rows.append(r)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            log(f"aux delete rewrite failed: {e}")

    return jsonify({'ok': True, 'removed': removed})


@app.route('/api/image/<path:filename>/scrap', methods=['POST'])
def scrap_image(filename):
    """Mark a frame as scrapped — excluded from review and export."""
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    det['scrapped'] = True
    det['reviewed'] = True  # so it doesn't re-appear in queue counts
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


@app.route('/api/image/<path:filename>/unscrap', methods=['POST'])
def unscrap_image(filename):
    """Undo scrap (user changed their mind)."""
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    det['scrapped'] = False
    flush_all()
    _build_review_list()
    return jsonify({'ok': True, 'remaining': len(session['review_files'])})


@app.route('/api/image/<path:filename>/points', methods=['PUT'])
def update_points(filename):
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    data = request.json
    incoming = data['points']

    if getattr(cfg, 'TARGET_SPECIES_ONLY', 0) and ALL_TARGET_SPECIES:
        # UI shows target-species points AND REVIEW points. Merge incoming back:
        # keep the hidden non-target points unchanged; replace the editable
        # (target + REVIEW) points with incoming. REVIEW points are excluded from
        # the preserved bucket so they aren't duplicated (they return via incoming).
        non_target = [p for p in det['points']
                      if p.get('species_code', '') not in ALL_TARGET_SPECIES
                      and not _is_review(p)]
        det['points'] = non_target + incoming
    else:
        det['points'] = incoming

    det['reviewed'] = True
    flush_all()
    return jsonify({'ok': True})


@app.route('/api/export_batch', methods=['POST'])
def do_export():
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']
    export_dir = session['export_dir']

    batch_files = files[offset:offset + batch_size]
    if not batch_files:
        return jsonify({'error': 'No images to export'}), 400

    reference_mode = bool(session.get('reference_mode', False))
    detections = []
    re_export_count = 0
    for fn in batch_files:
        det, year = _find_detection(fn)
        if det:
            if det.get('exported', False):
                re_export_count += 1
            pts = det['points']
            # Reference mode: the user placed their own points instead of
            # editing the detected ones. Export only their additions.
            if reference_mode:
                pts = [p for p in pts if p.get('added') is True]
            detections.append({
                'filename': fn,
                'pts_path': det['pts_path'],
                'raw_path': det.get('raw_path'),
                'points': pts,
                'year': det['year'],
                'date_str': det.get('date_str', ''),
                'site': det.get('site', ''),
                't_id': det.get('t_id', ''),
            })

    stats = export_batch(detections, export_dir, log_fn=lambda msg: log(msg))

    for fn in batch_files:
        det, year = _find_detection(fn)
        if det:
            det['exported'] = True

    for year in stats.get('years_affected', []):
        write_summary(year, export_dir)

    flush_all()

    # Rebuild review list: unexported first, exported at end
    _build_review_list()

    # Count remaining unexported
    n_unexported = sum(
        1 for dets in session['annotations_by_year'].values()
        for d in dets.values() if not d.get('exported', False)
    )

    # Auto-start SAM3 head-start: if the orchestrator set AUTO_START_SAM3=1
    # and gave us its URL, nudge step 5 to (re)configure and process the new
    # frames. Fire and forget — any failure here does not affect the export.
    _nudge_sam3_if_configured(exported_count=stats.get('exported_raw', 0))

    return jsonify({
        'ok': True,
        'exported_raw': stats['exported_raw'],
        'exported_images': stats['exported_pts'],
        'point_records': stats['point_records'],
        'test_images': stats['test_images'],
        'years_affected': stats['years_affected'],
        'remaining_review': n_unexported,
        're_exported': re_export_count,
    })


@app.route('/api/export_all', methods=['POST'])
def do_export_all():
    """Headless flush: export EVERY routed frame across ALL years in one call.

    Unlike /api/export_batch (which exports only the current review batch
    slice), this exports the entire detection set regardless of review
    batches, forcing reference_mode OFF so detected (CPC/OCR) points are
    written, not just user-added ones. Used by the orchestrator's background
    routing driver so 4.test can fully replace Step 4 without a manual review
    pass. Reuses export_batch (the same JSON/CSV writer as do_export).
    """
    export_dir = session['export_dir']

    # Gather ALL frames across all years (not a review-batch slice).
    detections = []
    all_keys = []
    for year, dets in session['annotations_by_year'].items():
        for fn, det in dets.items():
            if det.get('scrapped', False):
                continue
            all_keys.append((fn, year))
            detections.append({
                'filename': fn,
                'pts_path': det['pts_path'],
                'raw_path': det.get('raw_path'),
                'points': det['points'],  # reference_mode OFF: keep all points
                'year': det['year'],
                'date_str': det.get('date_str', ''),
                'site': det.get('site', ''),
                't_id': det.get('t_id', ''),
            })

    if not detections:
        return jsonify({'error': 'No frames to export'}), 400

    stats = export_batch(detections, export_dir, log_fn=lambda msg: log(msg))

    for fn, year in all_keys:
        det, _ = _find_detection(fn)
        if det:
            det['exported'] = True

    for year in stats.get('years_affected', []):
        write_summary(year, export_dir)

    flush_all()
    _build_review_list()

    return jsonify({
        'ok': True,
        'exported_frames': len(detections),
        'years': stats['years_affected'],
        'exported_raw': stats['exported_raw'],
        'point_records': stats['point_records'],
    })


def _nudge_sam3_if_configured(exported_count=0):
    """Tell the orchestrator to head-start SAM3 after a successful export.

    Controlled by env vars set when launched from the orchestrator:
      TCRMP_ORCHESTRATOR_URL (e.g. http://localhost:5050)
      TCRMP_AUTO_START_SAM3  (set to "1" to enable)
    Runs in a background thread — the export response should never block on
    a slow or unreachable orchestrator.
    """
    url = os.environ.get('TCRMP_ORCHESTRATOR_URL', '').rstrip('/')
    auto_flag = os.environ.get('TCRMP_AUTO_START_SAM3', '') == '1'
    if not url or not auto_flag:
        return
    if exported_count <= 0:
        return  # nothing new worth nudging for

    def _call():
        import urllib.request as _urllib
        try:
            req = _urllib.Request(
                f"{url}/api/step/5/kick",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with _urllib.urlopen(req, timeout=5) as r:
                log(f"SAM3 kick sent: HTTP {r.status}")
        except Exception as e:
            log(f"SAM3 kick failed (non-fatal): {e}")

    import threading as _threading
    _threading.Thread(target=_call, daemon=True).start()


@app.route('/api/next_review_batch', methods=['POST'])
def next_review_batch():
    files = session['review_files']
    batch_size = session['review_batch_size']
    if session['review_offset'] + batch_size < len(files):
        session['review_offset'] += batch_size
    else:
        return jsonify({
            'offset': session['review_offset'],
            'total_unreviewed': len(files),
            'done': True,
        })
    return jsonify({
        'offset': session['review_offset'],
        'total_unreviewed': len(files),
    })


@app.route('/api/update_batch_size', methods=['POST'])
def update_batch_size():
    data = request.json
    session['review_batch_size'] = data.get('batch_size', 10)
    return jsonify({'ok': True, 'batch_size': session['review_batch_size']})


@app.route('/api/status')
def status():
    year_stats = {}
    for year, dets in session['annotations_by_year'].items():
        n_exported = sum(1 for d in dets.values() if d.get('exported', False))
        n_reviewed = sum(1 for d in dets.values() if d.get('reviewed', False))
        year_stats[year] = {
            'processed': len(dets),
            'reviewed': n_reviewed,
            'exported': n_exported,
        }
    return jsonify({
        'configured': session['configured'],
        'phase': session['phase'],
        'total_images': len(session.get('all_images', [])),
        'processing_done': session['processing_idx'],
        'processing_total': len(session['processing_queue']),
        'review_total': len(session['review_files']),
        'review_offset': session['review_offset'],
        'review_batch_size': session['review_batch_size'],
        'year_stats': year_stats,
    })


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve image from its original path (READ-ONLY)."""
    det, year = _find_detection(filename)
    if det:
        # For CPC frames, pts_path points to the raw image
        path = det.get('pts_path', '')
        if path and os.path.exists(path):
            return send_file(path)
        # Try raw_path as fallback
        path = det.get('raw_path', '')
        if path and os.path.exists(path):
            return send_file(path)
    return jsonify({'error': 'not found'}), 404


# ── Main ─────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TCRMP Place Points - Unified QAQC App')
    parser.add_argument('--port', type=int, default=5065)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    if args.no_gpu:
        USE_GPU = False

    print(f"Pre-loading EasyOCR model (GPU={USE_GPU})...")
    get_ocr_reader(gpu=USE_GPU)
    log("EasyOCR model loaded, server starting")
    print(f"Target species: {ALL_TARGET_SPECIES}")
    print(f"Server ready on http://localhost:{args.port}")

    app.run(host='0.0.0.0', port=args.port, debug=False)
