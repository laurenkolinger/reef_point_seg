"""
TCRMP OCR Batch — Process-All-Then-Review Flask App

Two-phase architecture:
  Phase 1 (Processing): OCR all _pts images across selected years, save
      detection results to per-year JSON files. Resumable on crash/restart.
  Phase 2 (Review): Load detections from JSON, present in review batches
      of 5/10/20 for QAQC, export to cpc_all-compatible year directories.

CRITICAL: The import directory is READ-ONLY. Never modify, delete, or write
to the import folder — it is a shared Dropbox with the only copy of images.

Usage:
    python app.py [--port 5050] [--no-gpu]
"""

import os
import sys
import json
import argparse
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import detect_annotations, get_ocr_reader
from scanner import scan_all_years, get_available_years, parse_pts_filename, pts_to_raw_name
from species import SpeciesLookup, ocr_date_to_lookup_date, parse_transect_frame
from export import export_batch
from summarize import write_summary

app = Flask(__name__, template_folder='templates')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_GPU = True

# ── Session state ──────────────────────────────────────────────────

session = {
    'import_dir': '',
    'export_dir': '',
    'selected_years': [],
    'review_batch_size': 10,
    'configured': False,
    'phase': 'startup',           # startup | processing | review
    'all_images': [],             # full scan results
    'processing_queue': [],       # images not yet OCR'd
    'processing_idx': 0,
    'annotations_by_year': {},    # year -> {filename: detection_dict}
    'review_files': [],           # ordered list of unreviewed filenames
    'review_offset': 0,
    'species_lookup': None,
    'flush_counter': 0,
    'flush_interval': 50,         # auto-flush every N images
}

LOG_LINES = {}  # year -> [log lines]


def log(msg, year=None):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    if year is not None:
        LOG_LINES.setdefault(year, []).append(line)


def flush_logs(export_dir):
    """Write buffered log lines to per-year log.txt files."""
    for year, lines in LOG_LINES.items():
        if not lines:
            continue
        log_path = os.path.join(export_dir, str(year), 'log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            for line in lines:
                f.write(line + '\n')
    LOG_LINES.clear()


# ── Detection DB persistence (JSON per year) ─────────────────────

def _detections_path(export_dir, year):
    return os.path.join(export_dir, str(year), 'ocr_detections.json')


def _manifest_path(export_dir, year):
    return os.path.join(export_dir, str(year), 'processed_manifest.json')


def load_detections(export_dir, year):
    """Load existing detection results for a year. Returns dict."""
    path = _detections_path(export_dir, year)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_detections(export_dir, year, detections):
    """Save detection results for a year to JSON."""
    path = _detections_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(detections, f, indent=2)


def load_manifest(export_dir, year):
    """Load processed manifest for a year. Returns set of filenames."""
    path = _manifest_path(export_dir, year)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return set(data.get('processed', []))
    return set()


def save_manifest(export_dir, year, processed_set):
    """Save processed manifest for a year."""
    path = _manifest_path(export_dir, year)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            'processed': sorted(processed_set),
            'last_updated': datetime.now().isoformat(),
            'count': len(processed_set),
        }, f, indent=2)


def flush_all():
    """Flush all in-memory annotations to disk."""
    export_dir = session['export_dir']
    if not export_dir:
        return
    for year, dets in session['annotations_by_year'].items():
        save_detections(export_dir, year, dets)
        save_manifest(export_dir, year, set(dets.keys()))
    flush_logs(export_dir)


# ── Species enrichment ───────────────────────────────────────────

def _load_species_lookup():
    """Load all_points.csv species data."""
    csv_path = os.path.join(BASE_DIR, '..', '..', 'output', 'all_points.csv')
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        log("WARNING: all_points.csv not found — species codes will be empty")
        return None
    log(f"Loading species data from {csv_path}")
    sp = SpeciesLookup(csv_path)
    log(f"Loaded {len(sp)} species records")
    return sp


def enrich_with_species(points, date_str, site, t_id, species_lookup):
    """Add species_code, species_name, category to each point."""
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


def run_detection(image_path, filename, date_str, site, t_id):
    """Run OCR detection and enrich with species data."""
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
        })

    enrich_with_species(points, date_str, site, t_id, session['species_lookup'])
    return points


# ── Review helpers ───────────────────────────────────────────────

def _build_review_list():
    """Build ordered list of unreviewed + unexported filenames."""
    review = []
    for year in sorted(session['annotations_by_year']):
        dets = session['annotations_by_year'][year]
        for fn in sorted(dets):
            det = dets[fn]
            if not det.get('exported', False):
                review.append(fn)
    session['review_files'] = review
    session['review_offset'] = 0


def _find_detection(filename):
    """Find a detection dict by filename across all years."""
    for year, dets in session['annotations_by_year'].items():
        if filename in dets:
            return dets[filename], year
    return None, None


# ── Routes ───────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/available_years', methods=['POST'])
def available_years():
    """Return available years for an import directory."""
    data = request.json
    import_dir = data.get('import_dir', '')
    if not import_dir or not os.path.isdir(import_dir):
        return jsonify({'years': [], 'error': 'Invalid directory'})
    years = get_available_years(import_dir)
    return jsonify({'years': years})


@app.route('/api/configure', methods=['POST'])
def configure():
    """New project: scan dirs, load species, check existing detections."""
    data = request.json
    session['import_dir'] = os.path.abspath(data['import_dir'])
    session['export_dir'] = os.path.abspath(data['export_dir'])
    session['selected_years'] = data.get('selected_years', [])
    session['review_batch_size'] = data.get('review_batch_size', 10)

    os.makedirs(session['export_dir'], exist_ok=True)

    log(f"Configured: import={session['import_dir']}, "
        f"export={session['export_dir']}, "
        f"years={session['selected_years']}, "
        f"review_batch={session['review_batch_size']}")

    # Load species lookup
    session['species_lookup'] = _load_species_lookup()

    # Scan input directory (READ-ONLY)
    all_images = scan_all_years(session['import_dir'], session['selected_years'])
    session['all_images'] = all_images

    # Load existing detections from export dir
    session['annotations_by_year'] = {}
    already_processed = set()
    year_stats = {}

    for year in session['selected_years']:
        year_dir = os.path.join(session['export_dir'], str(year))
        os.makedirs(year_dir, exist_ok=True)

        dets = load_detections(session['export_dir'], year)
        session['annotations_by_year'][year] = dets
        already_processed.update(dets.keys())

        year_total = sum(1 for img in all_images if img['year'] == year)
        year_stats[year] = {
            'total': year_total,
            'processed': len(dets),
            'remaining': year_total - len(dets),
        }

    # Build processing queue (images not yet in any detection DB)
    queue = [img for img in all_images if img['filename'] not in already_processed]
    session['processing_queue'] = queue
    session['processing_idx'] = 0

    # Determine phase
    if len(queue) == 0:
        session['phase'] = 'review'
        _build_review_list()
    else:
        session['phase'] = 'processing'

    session['configured'] = True

    log(f"Found {len(all_images)} total images, "
        f"{len(already_processed)} already processed, "
        f"{len(queue)} remaining to process")

    return jsonify({
        'total_images': len(all_images),
        'already_processed': len(already_processed),
        'remaining': len(queue),
        'phase': session['phase'],
        'year_stats': year_stats,
        'review_count': len(session.get('review_files', [])),
    })


@app.route('/api/resume', methods=['POST'])
def resume():
    """Continue existing project: load export dir, discover state."""
    data = request.json
    session['export_dir'] = os.path.abspath(data['export_dir'])
    session['review_batch_size'] = data.get('review_batch_size', 10)

    if not os.path.isdir(session['export_dir']):
        return jsonify({'error': 'Export directory not found'}), 404

    # If user pointed to a year subdir (e.g. ocr_all/2023), walk up to parent
    export_dir = session['export_dir']
    basename = os.path.basename(export_dir)
    if basename.isdigit() and os.path.exists(os.path.join(export_dir, 'ocr_detections.json')):
        export_dir = os.path.dirname(export_dir)
        session['export_dir'] = export_dir

    # Discover years from existing year subdirs
    discovered_years = []
    for entry in sorted(os.listdir(export_dir)):
        entry_path = os.path.join(export_dir, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            yr = int(entry)
            det_path = _detections_path(export_dir, yr)
            if os.path.exists(det_path):
                discovered_years.append(yr)

    if not discovered_years:
        return jsonify({'error': 'No detection data found in export directory'}), 404

    session['selected_years'] = discovered_years

    # Load species
    session['species_lookup'] = _load_species_lookup()

    # Load all detections
    session['annotations_by_year'] = {}
    total_processed = 0
    year_stats = {}

    for year in discovered_years:
        dets = load_detections(session['export_dir'], year)
        session['annotations_by_year'][year] = dets
        total_processed += len(dets)
        n_exported = sum(1 for d in dets.values() if d.get('exported', False))
        year_stats[year] = {
            'total': len(dets),
            'processed': len(dets),
            'exported': n_exported,
            'remaining_review': len(dets) - n_exported,
        }

    # Try to discover import_dir from detection data
    for year in discovered_years:
        dets = session['annotations_by_year'][year]
        for fn, det in dets.items():
            pts_path = det.get('pts_path', '')
            if pts_path and os.path.exists(pts_path):
                parts = pts_path.split(os.sep)
                for i, part in enumerate(parts):
                    if part == 'TCRMP_clip':
                        session['import_dir'] = os.sep.join(parts[:i + 1])
                        break
            if session['import_dir']:
                break
        if session['import_dir']:
            break

    # Check if there are unprocessed images (need import_dir + rescan)
    needs_processing = False
    if session['import_dir'] and os.path.isdir(session['import_dir']):
        all_images = scan_all_years(session['import_dir'], discovered_years)
        session['all_images'] = all_images
        already = set()
        for dets in session['annotations_by_year'].values():
            already.update(dets.keys())
        queue = [img for img in all_images if img['filename'] not in already]
        session['processing_queue'] = queue
        session['processing_idx'] = 0
        needs_processing = len(queue) > 0

        for year in discovered_years:
            yt = sum(1 for img in all_images if img['year'] == year)
            year_stats[year]['total'] = yt
            year_stats[year]['remaining'] = yt - year_stats[year]['processed']
    else:
        session['all_images'] = []
        session['processing_queue'] = []

    if needs_processing:
        session['phase'] = 'processing'
    else:
        session['phase'] = 'review'
        _build_review_list()

    session['configured'] = True

    log(f"Resumed project: {len(discovered_years)} years, "
        f"{total_processed} total detections")

    return jsonify({
        'total_processed': total_processed,
        'years': discovered_years,
        'phase': session['phase'],
        'year_stats': year_stats,
        'import_dir': session.get('import_dir', ''),
        'needs_processing': needs_processing,
        'processing_remaining': len(session['processing_queue']),
        'review_count': len(session.get('review_files', [])),
    })


@app.route('/api/process', methods=['POST'])
def process_one():
    """Process ONE image from the queue. Called repeatedly for progress."""
    idx = session['processing_idx']
    queue = session['processing_queue']

    if idx >= len(queue):
        flush_all()
        session['phase'] = 'review'
        _build_review_list()
        return jsonify({
            'done': True,
            'processed': idx,
            'total': len(queue),
            'review_count': len(session['review_files']),
        })

    item = queue[idx]
    pts_path = item['pts_path']
    filename = item['filename']
    year = item['year']

    log(f"Processing {idx + 1}/{len(queue)}: {filename}", year=year)
    points = run_detection(pts_path, filename, item['date_str'],
                           item['site'], item['t_id'])

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
    }

    session['annotations_by_year'].setdefault(year, {})[filename] = detection
    session['processing_idx'] = idx + 1
    session['flush_counter'] += 1

    if session['flush_counter'] >= session['flush_interval']:
        flush_all()
        session['flush_counter'] = 0

    return jsonify({
        'done': False,
        'processed': idx + 1,
        'total': len(queue),
        'filename': filename,
        'year': year,
        'n_points': len(points),
    })


@app.route('/api/flush', methods=['POST'])
def force_flush():
    """Force-write all in-memory annotations to disk."""
    flush_all()
    return jsonify({'ok': True})


@app.route('/api/review_batch')
def review_batch():
    """Return current review batch."""
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']

    batch_files = files[offset:offset + batch_size]
    images = []
    for fn in batch_files:
        det, year = _find_detection(fn)
        if det:
            images.append({
                'filename': fn,
                'year': year,
                'n_points': len(det.get('points', [])),
                'status': 'reviewed' if det.get('reviewed') else 'detected',
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
    """Return detection data for one image."""
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    return jsonify({
        'points': det['points'],
        'status': 'reviewed' if det.get('reviewed') else 'detected',
        'year': year,
    })


@app.route('/api/image/<path:filename>/points', methods=['PUT'])
def update_points(filename):
    """Save QAQC edits for an image."""
    det, year = _find_detection(filename)
    if not det:
        return jsonify({'error': 'not found'}), 404
    data = request.json
    det['points'] = data['points']
    det['reviewed'] = True
    return jsonify({'ok': True})


@app.route('/api/export_batch', methods=['POST'])
def do_export():
    """Export current review batch to year-organized output."""
    batch_size = session['review_batch_size']
    offset = session['review_offset']
    files = session['review_files']
    export_dir = session['export_dir']

    batch_files = files[offset:offset + batch_size]
    if not batch_files:
        return jsonify({'error': 'No images to export'}), 400

    detections = []
    for fn in batch_files:
        det, year = _find_detection(fn)
        if det:
            detections.append({
                'filename': fn,
                'pts_path': det['pts_path'],
                'raw_path': det.get('raw_path'),
                'points': det['points'],
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
    _build_review_list()

    return jsonify({
        'ok': True,
        'exported_raw': stats['exported_raw'],
        'exported_images': stats['exported_pts'],
        'point_records': stats['point_records'],
        'test_images': stats['test_images'],
        'years_affected': stats['years_affected'],
        'remaining_review': len(session['review_files']),
    })


@app.route('/api/next_review_batch', methods=['POST'])
def next_review_batch():
    """Advance to next review batch."""
    files = session['review_files']
    batch_size = session['review_batch_size']

    if session['review_offset'] + batch_size < len(files):
        session['review_offset'] += batch_size
    else:
        session['review_offset'] = 0

    return jsonify({
        'offset': session['review_offset'],
        'total_unreviewed': len(files),
    })


@app.route('/api/update_batch_size', methods=['POST'])
def update_batch_size():
    """Update review batch size."""
    data = request.json
    session['review_batch_size'] = data.get('batch_size', 10)
    return jsonify({'ok': True, 'batch_size': session['review_batch_size']})


@app.route('/api/status')
def status():
    """Overall progress status."""
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
        'total_images': len(session['all_images']),
        'processing_done': session['processing_idx'],
        'processing_total': len(session['processing_queue']),
        'review_total': len(session['review_files']),
        'review_offset': session['review_offset'],
        'review_batch_size': session['review_batch_size'],
        'year_stats': year_stats,
    })


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve a _pts image from its original path (READ-ONLY)."""
    det, year = _find_detection(filename)
    if det and det.get('pts_path') and os.path.exists(det['pts_path']):
        return send_file(det['pts_path'])
    return jsonify({'error': 'not found'}), 404


# ── Main ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCRMP OCR Batch - QAQC App')
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    if args.no_gpu:
        USE_GPU = False

    print("Pre-loading EasyOCR model (GPU={})...".format(USE_GPU))
    get_ocr_reader(gpu=USE_GPU)
    log("EasyOCR model loaded, server starting")
    print(f"Server ready on http://localhost:{args.port}")

    app.run(host='0.0.0.0', port=args.port, debug=False)
