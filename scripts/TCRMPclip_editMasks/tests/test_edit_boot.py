"""Tests for the editMasks app booting in 'edit' mode against a project export.

Where this fits: TCRMPclip_editMasks is a standalone fork of the combined
annotator dedicated to editing an EXISTING project export (relabel / delete /
redraw / add masks). It boots via the SAME POST /api/resume path a resume
session uses, but with config.SESSION_MODE == 'edit' (default) so provenance
rows are tagged 'edit'. This test proves the edit-mode boot loads the export's
segmentations.json editable and serves the mask data back to the client.

House harness style: standalone script, no pytest. Env vars are set BEFORE
`import app as A` (mirrors test_loop_resume.py). SAM3 loads only under
`__main__` in app.py, so the test client never touches the GPU; the routes
exercised here (resume, image GET) are SAM-free.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_edit_boot.py
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

# -- Build the temp seeded export BEFORE import (env vars must be set first) --
_ROOT = tempfile.mkdtemp(prefix='edit_boot_test_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations')
os.makedirs(_SEG_DIR, exist_ok=True)

_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_RAW_DIR, exist_ok=True)

_FRAME_NAME = 'TCRMPX_clip_AAA_T118.jpeg'


def _make_image(path, w=200, h=100):
    im = Image.new('RGB', (w, h), (10, 20, 30))
    im.save(path, 'JPEG', quality=90)


_make_image(os.path.join(_RAW_DIR, _FRAME_NAME))

# Set env BEFORE importing app/mask_ops/config so cfg.* picks these up.
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT_DIR
os.environ['TCRMP_SESSION_MODE'] = 'edit'
os.environ['TCRMP_PROVENANCE_SOURCE'] = 'edit'
os.environ['TCRMP_MANUAL_ANNOTATE'] = '1'
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'AA')

import config as cfg  # noqa: E402
import mask_ops  # noqa: E402

# Build one committed PA (present/accepted) mask on the frame using the same
# helper the annotator uses, so the fixture matches real export shape exactly.
_binary_mask = np.zeros((100, 200), dtype=bool)
_binary_mask[20:80, 20:180] = True  # well above MIN_MASK_AREA_PX default guard
_point_info = {'label': 'M', 'species': 'AA', 'name': 'Acropora agaricites',
               'category': 'Coral', 'x': 100, 'y': 50}
_seeded_mask = mask_ops.build_mask_dict(
    mask_id=0, binary_mask=_binary_mask, score=0.9,
    point_info=_point_info, source_type='auto',
)
assert _seeded_mask is not None, 'fixture mask must build (check MIN_MASK_AREA_PX)'
_seeded_mask['status'] = 'accepted'

_seg_dict = {
    _FRAME_NAME: {
        'image_path': _FRAME_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _FRAME_NAME),
        'image_width': 200,
        'image_height': 100,
        'masks': [_seeded_mask],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
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


# -- config reflects the edit-mode defaults / env set above ------------------
check(cfg.SESSION_MODE == 'edit', f"cfg.SESSION_MODE == 'edit' (got {cfg.SESSION_MODE!r})")
check(cfg.PROVENANCE_SOURCE == 'edit',
      f"cfg.PROVENANCE_SOURCE == 'edit' (got {cfg.PROVENANCE_SOURCE!r})")
check(cfg.PORT == 5085, f"cfg.PORT default == 5085 (got {cfg.PORT})")

A.app.config['TESTING'] = True

with A.app.test_client() as c:
    # -- The index page routes an edit session through the resume boot --------
    r_idx = c.get('/')
    check(r_idx.status_code == 200, f"GET / -> 200 (got {r_idx.status_code})")
    idx_html = r_idx.data.decode('utf-8', 'replace')
    check("const SESSION_MODE = \"resume\"" in idx_html,
          "edit session hands the template SESSION_MODE 'resume' so it boots via /api/resume")
    check("const ORCHESTRATED = true" in idx_html,
          "edit session is orchestrated (auto-boots headlessly from TCRMP_EXPORT_DIR)")

    # -- POST /api/resume reads export_dir from the POST BODY -> 200 ----------
    r = c.post('/api/resume', json={'export_dir': _EXPORT_DIR})
    check(r.status_code == 200, f"POST /api/resume -> 200 (got {r.status_code}, body={r.data!r})")
    resume_body = json.loads(r.data)
    check(resume_body.get('total_processed') == 1,
          f"resume loaded 1 processed frame (got {resume_body.get('total_processed')})")

    # -- GET /api/image/<f> returns the export's masks editable --------------
    r_img = c.get(f'/api/image/{_FRAME_NAME}')
    check(r_img.status_code == 200, f"GET /api/image/<frame> -> 200 (got {r_img.status_code})")
    img_data = json.loads(r_img.data)
    masks = img_data.get('masks', [])
    check(len(masks) == 1, f"frame has exactly 1 editable mask (got {len(masks)})")
    if masks:
        check(masks[0].get('species') == 'AA',
              f"masks[0].species == 'AA' (got {masks[0].get('species')})")
        check(masks[0].get('status') == 'accepted',
              f"masks[0].status == 'accepted' (got {masks[0].get('status')})")
    check(img_data.get('image_width') == 200,
          f"served image_width == 200 (got {img_data.get('image_width')})")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
