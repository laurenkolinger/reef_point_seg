"""Tests for the annotator's resume session mode (active-learning loop).

Where this fits: seed_from_predictions.py (Task 3) seeds model masks into an
export's segmentations.json as status='pending', source_type='model'. This
test covers the OTHER half of the loop -- the annotator booting straight into
a review queue of exactly those seeded (non-exported) frames via POST
/api/resume, instead of re-running the configure/routing path.

House harness style: standalone script, no pytest. Env vars are set BEFORE
`import app as A` (mirrors test_batch_flow.py / test_segment_app.py). SAM3
loads only under `__main__` in app.py, so the test client never touches the
GPU; the routes exercised here (resume, image GET, masks PUT) are SAM-free
per the module's own review-batch/masks-update code paths.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_loop_resume.py
"""
import json
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

# ── Build the temp seeded export BEFORE import (env vars must be set first) ──
_ROOT = tempfile.mkdtemp(prefix='loop_resume_test_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations')
os.makedirs(_SEG_DIR, exist_ok=True)

_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_RAW_DIR, exist_ok=True)

_SEEDED_NAME = 'TCRMPX_clip_AAA_T118.jpeg'
_EXPORTED_NAME = 'TCRMPX_clip_AAA_T218.jpeg'


def _make_image(path, w=200, h=100):
    im = Image.new('RGB', (w, h), (10, 20, 30))
    im.save(path, 'JPEG', quality=90)


_make_image(os.path.join(_RAW_DIR, _SEEDED_NAME))
_make_image(os.path.join(_RAW_DIR, _EXPORTED_NAME))

# Set env BEFORE importing app/mask_ops/config so cfg.* picks these up.
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT_DIR
os.environ['TCRMP_SESSION_MODE'] = 'resume'
os.environ['TCRMP_MANUAL_ANNOTATE'] = '1'
os.environ['TCRMP_PROVENANCE_SOURCE'] = 'step4loop'
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'AA')

import config as cfg  # noqa: E402
import mask_ops  # noqa: E402

# Build one pending, source_type='model' mask on the seeded frame using the
# same helper seed_from_predictions.py uses, so the fixture matches real
# seeder output shape exactly.
_binary_mask = np.zeros((100, 200), dtype=bool)
_binary_mask[20:80, 20:180] = True  # well above MIN_MASK_AREA_PX default guard
_point_info = {'label': 'M', 'species': 'AA', 'name': 'Acropora agaricites',
               'category': 'Coral', 'x': 100, 'y': 50}
_seeded_mask = mask_ops.build_mask_dict(
    mask_id=0, binary_mask=_binary_mask, score=0.9,
    point_info=_point_info, source_type='model',
)
assert _seeded_mask is not None, 'fixture mask must build (check MIN_MASK_AREA_PX)'
_seeded_mask['status'] = 'pending'

_seg_dict = {
    _SEEDED_NAME: {
        'image_path': _SEEDED_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _SEEDED_NAME),
        'image_width': 200,
        'image_height': 100,
        'masks': [_seeded_mask],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': False,
        'exported': False,
    },
    # A pre-existing EXPORTED frame: must NOT appear in the resume review
    # queue (done-is-done, mirrors _build_review_list's exported filter).
    _EXPORTED_NAME: {
        'image_path': _EXPORTED_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _EXPORTED_NAME),
        'image_width': 200,
        'image_height': 100,
        'masks': [],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': True,
    },
}
with open(os.path.join(_SEG_DIR, 'segmentations.json'), 'w') as f:
    json.dump(_seg_dict, f)

import app as A  # noqa: E402

_fail = 0
def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)
    else:
        print("PASS:", msg)


# ── config.SESSION_MODE reflects the env set above ──────────────────────────
check(cfg.SESSION_MODE == 'resume', f"cfg.SESSION_MODE == 'resume' (got {cfg.SESSION_MODE!r})")

A.app.config['TESTING'] = True

with A.app.test_client() as c:
    # ── POST /api/resume reads export_dir from the POST BODY ────────────────
    r = c.post('/api/resume', json={'export_dir': _EXPORT_DIR})
    check(r.status_code == 200, f"POST /api/resume -> 200 (got {r.status_code}, body={r.data!r})")

    # ── Review queue contains ONLY the seeded (non-exported) frame ──────────
    r_batch = c.get('/api/review_batch')
    check(r_batch.status_code == 200, f"GET /api/review_batch -> 200 (got {r_batch.status_code})")
    batch = json.loads(r_batch.data)
    review_filenames = [im['filename'] for im in batch.get('images', [])]
    check(review_filenames == [_SEEDED_NAME],
          f"review queue == [seeded frame] only (got {review_filenames})")
    check(batch.get('total_images') == 1,
          f"total_images == 1 (got {batch.get('total_images')})")
    check(_EXPORTED_NAME not in review_filenames,
          "pre-existing exported frame excluded from review queue")

    # ── GET /api/image/<seeded> -> masks[0].status=='pending', source_type=='model' ──
    r_img = c.get(f'/api/image/{_SEEDED_NAME}')
    check(r_img.status_code == 200, f"GET /api/image/<seeded> -> 200 (got {r_img.status_code})")
    img_data = json.loads(r_img.data)
    masks = img_data.get('masks', [])
    check(len(masks) == 1, f"seeded frame has exactly 1 mask (got {len(masks)})")
    if masks:
        check(masks[0].get('status') == 'pending',
              f"masks[0].status == 'pending' (got {masks[0].get('status')})")
        check(masks[0].get('source_type') == 'model',
              f"masks[0].source_type == 'model' (got {masks[0].get('source_type')})")

    # ── Accept the mask -> label_provenance.csv gains a step4loop/found_ai row ──
    mask_id = str(masks[0]['id']) if masks else '0'
    r_put = c.put(f'/api/image/{_SEEDED_NAME}/masks',
                  json={'updates': {mask_id: 'accepted'}})
    check(r_put.status_code == 200, f"PUT /api/image/<seeded>/masks -> 200 (got {r_put.status_code})")

    prov_path = os.path.join(_EXPORT_DIR, 'label_provenance.csv')
    check(os.path.exists(prov_path), f"label_provenance.csv written at {prov_path}")
    if os.path.exists(prov_path):
        import csv as _csv
        with open(prov_path, newline='') as f:
            rows = list(_csv.DictReader(f))
        match = [r for r in rows if r.get('basename') == os.path.splitext(_SEEDED_NAME)[0]
                  and r.get('label') == 'AA']
        check(len(match) == 1, f"exactly 1 provenance row for (seeded frame, AA) (got {len(match)})")
        if match:
            check(match[0].get('source') == 'step4loop',
                  f"provenance row source == 'step4loop' (got {match[0].get('source')!r})")
            check(match[0].get('outcome') == 'found_ai',
                  f"provenance row outcome == 'found_ai' (got {match[0].get('outcome')!r})")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
