"""
Batch-review flow regression test for the combined annotator (Step 4.test).

No pytest dependency; run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_batch_flow.py

Guards the removed-frames batch model (plan Task 1):

  - 11 manual-mode seg_dicts, batch_size 10.
  - Export batch 1 (10 frames): exported frames drop out of review_files,
    review_offset resets to 0, and the rebuilt list IS the next batch, so the
    leftover 1 frame loads (NOT done).
  - /api/status reports exactly 1 review batch still remaining at this point.
  - Export the final 1 frame: review_files goes empty and ONLY THEN is the
    queue done (status batches_remaining == 0), with the all-done reconcile
    assertion (routed == exported + scrapped) holding.

The SAM3 engine is never loaded. We drive the module-level `session` dict
directly (the app uses a process-global session, not Flask session) and call
the /api/export_batch + /api/status routes through the Flask test client.
"""

import os
import sys
import csv
import json
import shutil
import tempfile
import traceback
from datetime import datetime

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
sys.path.insert(0, _SRC)

# Pin config at import time (mirrors test_segment_app.py): deterministic
# target species + manual-annotate mode, and a fixed batch size for the math.
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'DICT')
os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')
os.environ.setdefault('TCRMP_REVIEW_BATCH_SIZE', '10')

import app  # noqa: E402

# ── tiny harness ────────────────────────────────────────────────────
_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ''))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _seed_session(export_dir, n_frames, batch_size):
    """Install n_frames empty manual-mode seg_dicts in the flat segmentations dict,
    configured for review at offset 0 with the given batch size."""
    segs = {}
    for i in range(n_frames):
        fn = f"frame_{i:03d}.jpg"
        segs[fn] = {
            'image_path': fn,
            'image_path_abs': os.path.join(export_dir, fn),  # may not exist
            'image_width': 10,
            'image_height': 10,
            'masks': [],
            'reference_points': [],
            'processed_at': datetime.now().isoformat(),
            'reviewed': False,
            'exported': False,
        }
    app.session['export_dir'] = export_dir
    app.session['input_dir'] = export_dir
    app.session['segmentations'] = segs
    app.session['class_map'] = {}
    app.session['review_batch_size'] = batch_size
    app.session['shuffle'] = False
    app.session['configured'] = True
    app.session['phase'] = 'review'
    app.session['contacts'] = []
    app.session['reviewer'] = ''
    app._build_review_list()


def _seed_session_real_images(export_dir, n_frames, batch_size):
    """Like _seed_session but writes REAL image files and gives each frame one
    accepted mask, so export_batch actually populates all_images/all_labels.

    This is the C1 regression path: all_images/ accumulates across export
    batches, so a per-batch reconcile fails from batch 2 on. The image_missing
    fixture above never writes to all_images, so it can NOT exercise this.
    """
    segs = {}
    for i in range(n_frames):
        fn = f"frame_{i:03d}.jpg"
        path = os.path.join(export_dir, fn)
        Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
        segs[fn] = {
            'image_path': fn,
            'image_path_abs': path,
            'image_width': 32,
            'image_height': 24,
            'masks': [{
                'status': 'accepted', 'species': 'DICT',
                'polygon_norm': [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]],
            }],
            'reference_points': [],
            'processed_at': datetime.now().isoformat(),
            'reviewed': False,
            'exported': False,
        }
    app.session['export_dir'] = export_dir
    app.session['input_dir'] = export_dir
    app.session['segmentations'] = segs
    app.session['class_map'] = {}
    app.session['review_batch_size'] = batch_size
    app.session['shuffle'] = False
    app.session['configured'] = True
    app.session['phase'] = 'review'
    app.session['contacts'] = []
    app.session['reviewer'] = ''
    app._build_review_list()


def _status(client):
    return client.get('/api/status').get_json()


def _review_batch(client):
    return client.get('/api/review_batch').get_json()


# ── the regression ──────────────────────────────────────────────────
def test_eleven_frames_two_batches_no_lost_final():
    app.app.config['TESTING'] = True
    client = app.app.test_client()

    tmp = tempfile.mkdtemp(prefix='batchflow_')
    try:
        _seed_session(tmp, n_frames=11, batch_size=10)

        # Initial state: 11 frames queued, 2 batches (10 + 1).
        st = _status(client)
        check(st['review_total'] == 11, f"expected 11 review_total, got {st['review_total']}")
        check(st['batches_remaining'] == 2, f"expected 2 batches, got {st['batches_remaining']}")
        check(st['review_offset'] == 0, f"offset must start at 0, got {st['review_offset']}")

        rb = _review_batch(client)
        check(len(rb['images']) == 10, f"batch 1 must serve 10 frames, got {len(rb['images'])}")

        # Export batch 1 (10 frames).
        r1 = client.post('/api/export_batch')
        j1 = r1.get_json()
        check(j1.get('ok'), f"export batch 1 not ok: {j1}")

        # Removed-frames model: 10 exported drop out, 1 remains, offset reset to 0.
        check(len(app.session['review_files']) == 1,
              f"after batch 1 export, review_files must have 1, got {len(app.session['review_files'])}")
        check(app.session['review_offset'] == 0,
              f"review_offset must reset to 0, got {app.session['review_offset']}")

        st = _status(client)
        check(st['review_total'] == 1, f"review_total must be 1 after batch 1, got {st['review_total']}")
        # NOT done yet: exactly one partial batch remains.
        check(st['batches_remaining'] == 1,
              f"final partial batch must still remain (1), got {st['batches_remaining']}")

        # The "next batch" reload (what nextReviewBatch() does) serves the
        # leftover frame; it is NOT lost.
        rb = _review_batch(client)
        check(len(rb['images']) == 1,
              f"the leftover frame must load as the next batch, got {len(rb['images'])} images")

        # next_review_batch route reports not-done while a frame remains.
        nb = client.post('/api/next_review_batch').get_json()
        check(nb['done'] is False, f"next_review_batch must report done=False with 1 frame left, got {nb}")

        # Export the final 1 frame.
        r2 = client.post('/api/export_batch')
        j2 = r2.get_json()
        check(j2.get('ok'), f"export final frame not ok: {j2}")

        # ONLY NOW is the queue empty + done.
        check(len(app.session['review_files']) == 0,
              f"review_files must be empty after final export, got {len(app.session['review_files'])}")
        st = _status(client)
        check(st['review_total'] == 0, f"review_total must be 0 when done, got {st['review_total']}")
        check(st['batches_remaining'] == 0, f"batches_remaining must be 0 when done, got {st['batches_remaining']}")

        # And next_review_batch now reports done.
        nb = client.post('/api/next_review_batch').get_json()
        check(nb['done'] is True, f"next_review_batch must report done=True when empty, got {nb}")

        # All 11 frames exported (reconcile assertion in do_export held).
        exported = sum(1 for s in app.session['segmentations'].values()
                       if s.get('exported'))
        check(exported == 11, f"all 11 frames must be exported, got {exported}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_multi_batch_export_cumulative_manifest():
    """C1 regression: a two-batch export over 11 frames WITH real image files.

    all_images/ accumulates (export_yolo only adds), so before the cumulative
    fix reconcile_export asserted all_images(20) == this-batch(10) and threw a
    500 on batch 2. Assert: neither export raises, all_images ends with 11
    images, and export_manifest.csv documents all 11 basenames (one row each).
    """
    app.app.config['TESTING'] = True
    client = app.app.test_client()

    tmp = tempfile.mkdtemp(prefix='batchflow_real_')
    try:
        _seed_session_real_images(tmp, n_frames=11, batch_size=10)

        # Batch 1 (10 frames). Must not raise (no 500).
        r1 = client.post('/api/export_batch')
        check(r1.status_code == 200, f"batch 1 HTTP {r1.status_code}, body={r1.get_data(as_text=True)}")
        j1 = r1.get_json()
        check(j1.get('ok'), f"export batch 1 not ok: {j1}")

        # Batch 2 (leftover 1 frame). This is the call that 500'd pre-fix
        # because all_images already held 10 while the batch manifest had 1.
        r2 = client.post('/api/export_batch')
        check(r2.status_code == 200, f"batch 2 HTTP {r2.status_code}, body={r2.get_data(as_text=True)}")
        j2 = r2.get_json()
        check(j2.get('ok'), f"export batch 2 not ok: {j2}")

        # all_images accumulated to all 11 frames across the two batches.
        imgs = sorted(os.listdir(os.path.join(tmp, 'all_images')))
        check(len(imgs) == 11, f"all_images must hold all 11 frames, got {len(imgs)}: {imgs}")

        # The CUMULATIVE manifest documents every routed basename, one row each.
        with open(os.path.join(tmp, 'export_manifest.csv')) as f:
            rows = list(csv.DictReader(f))
        basenames = sorted(r['basename'] for r in rows)
        check(len(rows) == 11, f"manifest must document all 11 basenames, got {len(rows)}: {basenames}")
        expected = sorted(f"frame_{i:03d}" for i in range(11))
        check(basenames == expected, f"manifest basenames mismatch: got {basenames}")
        # No duplicate rows (merge-by-basename, not append).
        check(len(set(basenames)) == 11, f"duplicate basenames in manifest: {basenames}")

        # And all 11 are flagged exported in the session.
        exported = sum(1 for s in app.session['segmentations'].values()
                       if s.get('exported'))
        check(exported == 11, f"all 11 frames must be exported, got {exported}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    print("BATCH-FLOW TESTS")
    run(test_eleven_frames_two_batches_no_lost_final)
    run(test_multi_batch_export_cumulative_manifest)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print(f"\n{n_pass} passed, {n_fail} failed")
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"\n--- {name} ---\n{detail}")
    sys.exit(1 if n_fail else 0)
