"""
Phase 0 unit tests — canonical project identity.

Proves project_id + project_name are threaded through export_flagged_masks into
BOTH the review manifest item and the library record, that project_name defaults
to project_id, that the library persists the new column, and that the
project_manager resolver maps a project_id back to its step-5 dir.

No pytest:  env/bin/python scripts/_reefreview/tests/test_project_identity.py
"""

import os
import sys
import json
import tempfile
import traceback

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, 'pipeline_orchestrator'))  # project_manager

from _reefreview import mask_geom, review_export
from _reefreview.library import Library
from _reefreview.review_repo import ReviewRepo

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn(); _RESULTS.append((fn.__name__, True, '')); print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── helpers (mirrors test_reefreview.py) ────────────────────────────
def _square_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m


def _review_mask(rle, bbox, poly, sx, sy, status='pending'):
    return {'id': 0, 'label': 'A', 'species': 'REVIEW', 'name': '', 'category': 'Review',
            'source_x': sx, 'source_y': sy, 'polygon_px': poly,
            'polygon_norm': [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]], 'rle': rle,
            'bbox': bbox, 'area': 100, 'score': 0.9, 'status': status, 'review': True}


def _make_project(root, fn, mask):
    proj = os.path.join(root, 'inprocess', 'p_20990101_aaa', 'step5_segmentImages')
    segdir = os.path.join(proj, 'segmentations', '2020'); os.makedirs(segdir, exist_ok=True)
    raw = os.path.join(proj, 'raw'); os.makedirs(raw, exist_ok=True)
    img = os.path.join(raw, fn)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
    seg = {'image_path': 'raw/' + fn, 'image_path_abs': img, 'image_width': 200,
           'image_height': 200, 'masks': [mask], 'reviewed': False, 'exported': False}
    return proj, {fn: seg}


def _export(t, *, project_id, project_name=None):
    fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
    mask = _review_mask(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                        [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
    proj, segmap = _make_project(t, fn, mask)
    review_dir = os.path.join(t, 'REVIEW'); lib_dir = os.path.join(t, '_lib')
    kw = dict(export_dir=proj, review_dir=review_dir, repo_url='', library_dir=lib_dir,
              master_codes=None, contacts=['x@y.com'], featured_codes=['PA'],
              project_id=project_id, git_push=False, log_fn=lambda m: None)
    if project_name is not None:
        kw['project_name'] = project_name
    st = review_export.export_flagged_masks(segmap, [fn], **kw)
    return st, review_dir, lib_dir


# ── tests ───────────────────────────────────────────────────────────
def test_export_stamps_project_name_into_manifest_and_library():
    with tempfile.TemporaryDirectory() as t:
        st, review_dir, lib_dir = _export(t, project_id='demo_20260101_abc', project_name='Demo Reef X')
        check(st['new'] == 1, st)
        uid = st['uids'][0]
        # manifest item
        repo = ReviewRepo(review_dir, remote_url='')
        items = repo.load_manifest().get('items', [])
        item = next((i for i in items if i.get('uid') == uid), None)
        check(item is not None, 'item in manifest')
        check(item.get('project_id') == 'demo_20260101_abc', f"manifest project_id: {item}")
        check(item.get('project_name') == 'Demo Reef X', f"manifest project_name: {item}")
        # library record
        rec = Library(lib_dir).lookup(uid)
        check(rec.get('project_id') == 'demo_20260101_abc', f"lib project_id: {rec}")
        check(rec.get('project_name') == 'Demo Reef X', f"lib project_name: {rec}")


def test_project_name_defaults_to_project_id_when_absent():
    with tempfile.TemporaryDirectory() as t:
        st, review_dir, lib_dir = _export(t, project_id='only_id_20260101_xyz')  # no project_name
        uid = st['uids'][0]
        item = next(i for i in ReviewRepo(review_dir, remote_url='').load_manifest()['items']
                    if i.get('uid') == uid)
        check(item.get('project_name') == 'only_id_20260101_xyz', f"defaulted name: {item}")
        check(Library(lib_dir).lookup(uid).get('project_name') == 'only_id_20260101_xyz', 'lib defaulted name')


def test_library_persists_project_name_column():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(os.path.join(t, '_lib')); lib.ensure()
        lib.upsert({'uid': 'u1', 'code': 'REVIEW', 'mode': 'USER',
                    'project_id': 'pid1', 'project_name': 'Project One'})
        # reload from disk via a fresh instance to prove it round-trips the CSV column
        rec = Library(os.path.join(t, '_lib')).lookup('u1')
        check(rec.get('project_name') == 'Project One', f"persisted: {rec}")


def test_resolve_step_dir_maps_project_id_to_step5():
    import project_manager as pm
    with tempfile.TemporaryDirectory() as t:
        root = os.path.join(t, 'inprocess'); os.makedirs(root)
        for pid in ('alpha_20200101_aaa', 'beta_20200202_bbb'):
            d = os.path.join(root, pid); os.makedirs(os.path.join(d, 'step5_segmentImages'))
            with open(os.path.join(d, 'project.json'), 'w') as f:
                json.dump({'id': pid, 'name': pid.split('_')[0]}, f)
        got = pm.resolve_step_dir('beta_20200202_bbb', root, step='5')
        check(got == os.path.join(root, 'beta_20200202_bbb', 'step5_segmentImages'), got)
        check(pm.resolve_step_dir('nope_x', root) == '', 'unknown id -> empty')
        check(pm.resolve_step_dir('', root) == '', 'empty id -> empty')
        ids = sorted(pid for pid, _ in pm.find_projects(root))
        check(ids == ['alpha_20200101_aaa', 'beta_20200202_bbb'], ids)


if __name__ == '__main__':
    for fn in (test_export_stamps_project_name_into_manifest_and_library,
               test_project_name_defaults_to_project_id_when_absent,
               test_library_persists_project_name_column,
               test_resolve_step_dir_maps_project_id_to_step5):
        run(fn)
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{len(_RESULTS) - n_fail} passed, {n_fail} failed")
    for name, ok, err in _RESULTS:
        if not ok:
            print(f"\n--- {name} ---\n{err}")
    sys.exit(1 if n_fail else 0)
