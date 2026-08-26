"""Tests for registry upsert on edit + source='edit' provenance in TCRMPclip_editMasks.

Where this fits: editing a mask must (1) update its canonical record in the
cross-project mask registry immediately -- not just at export -- and (2) the
editor must be able to reach frames that are ALREADY exported, because the
whole point of this app is to fix a backlog of masks that were accepted
(unlabeled) and exported before this editor existed. The base combined
annotator's _build_review_list excludes exported frames from review; this
app must include them instead, gated on config.SESSION_MODE == 'edit'.

This test seeds an export with two frames:
  - FRAME_EXPORTED: seg['exported'] = True, carries one unlabeled
    (species=='') accepted mask -- the "82-backlog" shape.
  - FRAME_NORMAL: seg['exported'] = False, carries one labeled mask, used
    to relabel/delete and check the registry + provenance side effects.

Assertions:
  (a) the exported frame is reachable: it appears in session review_files
      (via /api/resume -> review_count) and its unlabeled mask is listed by
      GET /api/unlabeled.
  (b) relabeling a mask to 'PA' via PUT /api/image/<f>/masks upserts a
      registry row with species 'PA' and a newer updated_at than the row's
      first upsert.
  (c) deleting a mask (status -> rejected) via the same route sets its
      registry row status to 'rejected'.
  (d) label_provenance.csv rows written by the editor carry source == 'edit'.

House harness style: standalone script, no pytest. Env vars (including
TCRMP_MASK_REGISTRY_DIR, pointed at a temp dir so this test never touches the
real cross-project registry) are set BEFORE `import app as A` (mirrors
test_edit_boot.py / test_expert_lock.py). SAM3 loads only under __main__ in
app.py, so the test client never touches the GPU; PUT /api/image/<f>/masks
and GET /api/unlabeled are SAM-free.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_edit_registry.py
"""
import csv
import json
import os
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src')
sys.path.insert(0, SRC)

# -- Build the temp seeded export BEFORE import (env vars must be set first) --
_ROOT = tempfile.mkdtemp(prefix='edit_registry_test_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations')
os.makedirs(_SEG_DIR, exist_ok=True)

_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_RAW_DIR, exist_ok=True)

_REGISTRY_DIR = os.path.join(_ROOT, 'mask_registry')

_FRAME_EXPORTED = 'TCRMPX_clip_AAA_T118.jpeg'
_FRAME_NORMAL = 'TCRMPX_clip_AAA_T119.jpeg'


def _make_image(path, w=200, h=100):
    im = Image.new('RGB', (w, h), (10, 20, 30))
    im.save(path, 'JPEG', quality=90)


_make_image(os.path.join(_RAW_DIR, _FRAME_EXPORTED))
_make_image(os.path.join(_RAW_DIR, _FRAME_NORMAL))

# Set env BEFORE importing app/mask_ops/config so cfg.* picks these up.
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT_DIR
os.environ['TCRMP_SESSION_MODE'] = 'edit'
os.environ['TCRMP_PROVENANCE_SOURCE'] = 'edit'
os.environ['TCRMP_MANUAL_ANNOTATE'] = '1'
os.environ['TCRMP_MASK_REGISTRY_DIR'] = _REGISTRY_DIR
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'AA,PA')

import config as cfg  # noqa: E402
import mask_ops  # noqa: E402


def _make_mask(mask_id, x0, species, name, category='Coral', status='accepted'):
    binary_mask = np.zeros((100, 200), dtype=bool)
    binary_mask[20:80, x0:x0 + 60] = True  # well above MIN_MASK_AREA_PX default guard
    point_info = {'label': chr(ord('M') + mask_id), 'species': species, 'name': name,
                  'category': category, 'x': x0 + 30, 'y': 50}
    m = mask_ops.build_mask_dict(
        mask_id=mask_id, binary_mask=binary_mask, score=0.9,
        point_info=point_info, source_type='auto',
    )
    assert m is not None, f'fixture mask {mask_id} must build (check MIN_MASK_AREA_PX)'
    m['status'] = status
    return m


# FRAME_EXPORTED: already exported, one UNLABELED accepted mask (the 82-backlog shape).
_mask_unlabeled = _make_mask(0, 20, species='', name='', status='accepted')

# FRAME_NORMAL: not yet exported, one labeled mask to relabel/delete.
_mask_labeled = _make_mask(0, 20, species='AA', name='Acropora agaricites')

_seg_dict = {
    _FRAME_EXPORTED: {
        'image_path': _FRAME_EXPORTED,
        'image_path_abs': os.path.join(_RAW_DIR, _FRAME_EXPORTED),
        'image_width': 200,
        'image_height': 100,
        'masks': [_mask_unlabeled],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': True,
    },
    _FRAME_NORMAL: {
        'image_path': _FRAME_NORMAL,
        'image_path_abs': os.path.join(_RAW_DIR, _FRAME_NORMAL),
        'image_width': 200,
        'image_height': 100,
        'masks': [_mask_labeled],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
    },
}
with open(os.path.join(_SEG_DIR, 'segmentations.json'), 'w') as f:
    json.dump(_seg_dict, f)

import app as A  # noqa: E402
from _reefreview.mask_registry import MaskRegistry, build_registry_record  # noqa: E402

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

A.app.config['TESTING'] = True

with A.app.test_client() as c:
    # -- Boot via the edit-mode resume path --------------------------------
    r = c.post('/api/resume', json={'export_dir': _EXPORT_DIR})
    check(r.status_code == 200, f"POST /api/resume -> 200 (got {r.status_code}, body={r.data!r})")
    resume_body = json.loads(r.data)
    check(resume_body.get('total_processed') == 2,
          f"resume loaded 2 processed frames (got {resume_body.get('total_processed')})")

    # -- (a) the exported frame IS reachable in edit mode -------------------
    check(resume_body.get('review_count') == 2,
          f"review_count includes the already-exported frame too (got "
          f"{resume_body.get('review_count')}, expected 2)")
    check(_FRAME_EXPORTED in A.session.get('review_files', []),
          f"exported frame is present in session review_files (got {A.session.get('review_files')})")

    r_unl = c.get('/api/unlabeled')
    check(r_unl.status_code == 200, f"GET /api/unlabeled -> 200 (got {r_unl.status_code})")
    unl_body = json.loads(r_unl.data)
    check(_FRAME_EXPORTED in (unl_body.get('frames') or []),
          f"/api/unlabeled lists the already-exported frame's unlabeled mask "
          f"(got frames={unl_body.get('frames')})")

    # -- (b) relabel the FRAME_NORMAL mask to 'PA' -> registry upsert -------
    # Mint the uid the same way the registry does, so we can look the row up.
    expected_uid = build_registry_record(
        _FRAME_NORMAL, _mask_labeled, 'proj', 'proj')['uid']

    reg = MaskRegistry(root=_REGISTRY_DIR)
    row_before = reg.lookup(expected_uid)
    check(row_before is None,
          f"registry row does not exist before any edit (got {row_before})")

    time.sleep(1.1)  # ensure a distinguishable (second-precision) updated_at

    r_relabel = c.put(f'/api/image/{_FRAME_NORMAL}/masks', json={
        'relabel': {'0': {'species': 'PA', 'name': 'Porites astreoides', 'category': 'Coral'}},
    })
    check(r_relabel.status_code == 200,
          f"PUT masks (relabel) -> 200 (got {r_relabel.status_code})")
    relabel_body = json.loads(r_relabel.data)
    check(relabel_body.get('ok') is True,
          f"relabel reports ok=True (got {relabel_body})")

    row_after_relabel = reg.lookup(expected_uid)
    check(row_after_relabel is not None,
          "registry row now exists immediately after relabel (upserted on edit, not export)")
    if row_after_relabel:
        check(row_after_relabel.get('species') == 'PA',
              f"registry row species == 'PA' after relabel (got {row_after_relabel.get('species')!r})")
        check(bool(row_after_relabel.get('updated_at')),
              f"registry row has a non-empty updated_at (got {row_after_relabel.get('updated_at')!r})")
        first_updated_at = row_after_relabel.get('updated_at')

    time.sleep(1.1)

    # -- (c) delete (status -> rejected) the same mask -> registry status ---
    r_del = c.put(f'/api/image/{_FRAME_NORMAL}/masks', json={
        'updates': {'0': 'rejected'},
    })
    check(r_del.status_code == 200, f"PUT masks (delete) -> 200 (got {r_del.status_code})")
    del_body = json.loads(r_del.data)
    check(del_body.get('ok') is True, f"delete reports ok=True (got {del_body})")

    row_after_delete = reg.lookup(expected_uid)
    check(row_after_delete is not None, "registry row still exists after delete")
    if row_after_delete:
        check(row_after_delete.get('status') == 'rejected',
              f"registry row status == 'rejected' after delete (got {row_after_delete.get('status')!r})")
        check(row_after_delete.get('updated_at') != first_updated_at,
              f"registry row updated_at bumped again on the delete edit "
              f"(before={first_updated_at!r}, after={row_after_delete.get('updated_at')!r})")

    # -- (d) provenance CSV rows carry source == 'edit' ----------------------
    prov_path = os.path.join(_EXPORT_DIR, 'label_provenance.csv')
    check(os.path.exists(prov_path), f"label_provenance.csv was written at {prov_path}")
    if os.path.exists(prov_path):
        with open(prov_path, newline='') as f:
            rows = list(csv.DictReader(f))
        check(len(rows) > 0, "label_provenance.csv has at least one row")
        non_edit = [r for r in rows if r.get('source') != 'edit']
        check(len(non_edit) == 0,
              f"every label_provenance.csv row has source=='edit' (found non-edit rows: {non_edit})")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
