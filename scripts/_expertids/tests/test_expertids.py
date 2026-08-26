"""
Add-Expert-IDs blueprint tests — Phase 5 routing + C5 rolling + C6 consensus/accept
(CONTRACTS §4,§5,§6,§8 + C2,C3,C6).

NO-pytest harness (mirrors _reefreview/tests): check()/run(), __main__ exits
nonzero on any failure. Runnable with the unified env python:

    env/bin/python scripts/_expertids/tests/test_expertids.py

Three layers:
  SMOKE       blueprint imports, routes register, GET / renders 200, classifier
  UNIT        project routing (group + resolve dir), rolling add_review upsert,
              accept sets expert_id + removes, consensus agreement classifier,
              operator-setup persistence, import/accept regenerate the
              label_provenance.csv ledger (pending_expert -> found_expert)
  ADVERSARIAL unrouted UID reported (never silent); identical frame names across
              two projects don't cross-relabel; re-drop same reviewer upserts (no
              dupes); conflict not auto-accepted; provenance refresh failures
              (module unavailable, unwritable ledger path) never break
              import/accept
"""

import os
import sys
import json
import tempfile
import traceback

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # scripts/_expertids
_SCRIPTS = os.path.dirname(_PKG)                    # scripts
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, 'pipeline_orchestrator'))

from _reefreview import mask_geom, review_export
from _reefreview.review_repo import ReviewRepo
from _reefreview.library import Library
from _expertids import importer, make_blueprint
import project_manager

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


# ── fixtures ─────────────────────────────────────────────────────────
def _square_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m


def _review_mask(rle, bbox, poly, sx, sy, status='pending'):
    return {'id': 0, 'label': 'A', 'species': 'REVIEW', 'name': '', 'category': 'Review',
            'source_x': sx, 'source_y': sy, 'polygon_px': poly,
            'polygon_norm': [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]], 'rle': rle,
            'bbox': bbox, 'area': 6400, 'score': 0.9, 'status': status, 'review': True}


def _mk_project(projects_root, pid, fn, mask):
    """Create <projects_root>/<pid>/ with a project.json (id==pid) + step-5 dir,
    seed a raw image, return (export_dir, segdir, segmap) ready for
    export_flagged_masks. The export_dir == <project_dir>/step5_segmentImages so
    resolve_step_dir maps pid -> this dir."""
    pdir = os.path.join(projects_root, pid)
    export_dir = os.path.join(pdir, 'step5_segmentImages')
    segdir = os.path.join(export_dir, 'segmentations', '2020'); os.makedirs(segdir, exist_ok=True)
    raw = os.path.join(export_dir, 'raw'); os.makedirs(raw, exist_ok=True)
    with open(os.path.join(pdir, 'project.json'), 'w') as f:
        json.dump({'id': pid, 'name': pid.split('_')[0]}, f)
    img = os.path.join(raw, fn)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
    seg = {'image_path': 'raw/' + fn, 'image_path_abs': img, 'image_width': 200,
           'image_height': 200, 'masks': [mask], 'reviewed': False, 'exported': False}
    return export_dir, segdir, {fn: seg}


def _export(projects_root, pid, fn, *, review_dir, library_dir, sq=(60, 140, 60, 140),
            project_name='', status='pending'):
    """Export one flagged mask for project pid; returns (export_dir, uid). The
    exporter stamps review_uid onto the in-memory masks, so we persist the
    post-export segmap to disk for the importer's relabel matcher."""
    y0, y1, x0, x1 = sq
    mask = _review_mask(mask_geom.rle_encode(_square_mask(200, 200, y0, y1, x0, x1)),
                        [x0, y0, x1 - 1, y1 - 1],
                        [[x0, y0, x1, y0, x1, y1, x0, y1]], (x0 + x1) // 2, (y0 + y1) // 2,
                        status=status)
    export_dir, segdir, segmap = _mk_project(projects_root, pid, fn, mask)
    st = review_export.export_flagged_masks(
        segmap, [fn], export_dir=export_dir, review_dir=review_dir, repo_url='',
        library_dir=library_dir, master_codes=None, contacts=[], featured_codes=['PA'],
        project_id=pid, project_name=project_name, git_push=False, log_fn=lambda m: None)
    # Persist the (now review_uid-stamped) segmap so the importer can match it.
    with open(os.path.join(segdir, 'segmentations.json'), 'w') as f:
        json.dump(segmap, f)
    return export_dir, st['uids'][0]


def _env(t):
    return {'projects_root': os.path.join(t, 'inprocess'),
            'review_dir': os.path.join(t, 'REVIEW'),
            'library_dir': os.path.join(t, '_lib')}


def _seg_mask(export_dir, fn):
    """Return the first mask dict in export_dir's segmentations for frame fn."""
    import glob
    for p in sorted(glob.glob(os.path.join(export_dir, 'segmentations', '*', 'segmentations.json'))):
        data = json.load(open(p))
        for f, seg in data.items():
            if os.path.basename(f) == os.path.basename(fn):
                return seg['masks'][0]
    return None


def _ledger_rows(export_dir):
    """All rows of export_dir's label_provenance.csv ([] when absent)."""
    import csv
    path = os.path.join(export_dir, 'label_provenance.csv')
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _client(e, git_push=False, repo_url=''):
    """Flask test client with the blueprint mounted at /expertids."""
    from flask import Flask

    def paths():
        return {'export_dir': '', 'review_dir': e['review_dir'],
                'library_dir': e['library_dir'], 'review_repo_url': repo_url,
                'master_codes': None, 'projects_root': e['projects_root'],
                'overlap_thresh': 0.5, 'git_push': git_push}

    app = Flask(__name__)
    app.register_blueprint(make_blueprint(paths, log_fn=lambda m: None), url_prefix='/expertids')
    return app.test_client()


# ══ SMOKE ════════════════════════════════════════════════════════════
def test_smoke_blueprint_routes_register_and_index_renders():
    from flask import Flask
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)

        def paths():
            return {'export_dir': '', 'review_dir': e['review_dir'],
                    'library_dir': e['library_dir'], 'review_repo_url': 'https://github.com/o/r.git',
                    'master_codes': None, 'projects_root': e['projects_root'],
                    'overlap_thresh': 0.5, 'git_push': False}

        app = Flask(__name__)
        app.register_blueprint(make_blueprint(paths, log_fn=lambda m: None), url_prefix='/expertids')
        rules = {r.rule for r in app.url_map.iter_rules()}
        for need in ('/expertids/', '/expertids/api/import', '/expertids/api/pending_by_project',
                     '/expertids/api/consensus', '/expertids/api/accept',
                     '/expertids/api/operator_setup', '/expertids/api/email_requests',
                     '/expertids/api/delete_project'):
            check(need in rules, f"route missing: {need}")
        c = app.test_client()
        r = c.get('/expertids/')
        check(r.status_code == 200, f"GET / -> {r.status_code}")
        check(b'eid-root' in r.data, 'panel fragment not rendered')
        check(b'cons-wrap' in r.data, 'consensus section not rendered')


def test_smoke_classifier_shapes():
    cls = importer.classify_reviews
    check(cls([])['status'] == 'none', 'empty -> none')
    check(cls([{'reviewer': 'a', 'code': 'PA'}])['status'] == 'single', 'one -> single')
    c2 = cls([{'reviewer': 'a', 'code': 'PA'}, {'reviewer': 'b', 'code': 'PA'}])
    check(c2['status'] == 'consensus' and c2['code'] == 'PA', f"agree -> consensus: {c2}")
    cf = cls([{'reviewer': 'a', 'code': 'PA'}, {'reviewer': 'b', 'code': 'AA'}])
    check(cf['status'] == 'conflict' and cf['code'] == '', f"disagree -> conflict: {cf}")


# ══ UNIT ═════════════════════════════════════════════════════════════
def test_default_projects_root_resolves_to_module_inprocess():
    root = importer.default_projects_root()
    check(root.endswith(os.path.join('reef_point_seg', 'inprocess')), root)


def test_routing_groups_by_project_and_resolves_dir():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed_a, ua = _export(e['projects_root'], 'alpha_20200101_aaa', fn,
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        ed_b, ub = _export(e['projects_root'], 'beta_20200101_bbb', 'TCRMP20201022_clip_BWR_T201.jpeg',
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        # resolve_step_dir maps each pid -> its own step-5 export dir
        check(project_manager.resolve_step_dir('alpha_20200101_aaa', e['projects_root']) == ed_a, 'pid alpha resolves')
        check(project_manager.resolve_step_dir('beta_20200101_bbb', e['projects_root']) == ed_b, 'pid beta resolves')
        rows = [{'uid': ua, 'code': 'PA', 'reviewer': 'jane', 'confidence': 'high', 'project_id': 'alpha_20200101_aaa'},
                {'uid': ub, 'code': 'AA', 'reviewer': 'jane', 'confidence': 'high', 'project_id': 'beta_20200101_bbb'}]
        st = importer.import_rows(rows, export_dir='', review_dir=e['review_dir'], repo_url='',
                                  library_dir=e['library_dir'], master_codes=None,
                                  projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        check(st['unrouted'] == [], f"nothing unrouted: {st}")
        check(len(st['projects']) == 2, f"two project groups: {st['projects']}")
        # tentative review landed on EACH project's own segmentation mask
        ma = _seg_mask(ed_a, fn); mb = _seg_mask(ed_b, 'TCRMP20201022_clip_BWR_T201.jpeg')
        check(ma['reviews'][0]['code'] == 'PA', f"alpha mask tentative PA: {ma.get('reviews')}")
        check(mb['reviews'][0]['code'] == 'AA', f"beta mask tentative AA: {mb.get('reviews')}")
        # ROLLING: no expert_id set on import (CONTRACTS §5)
        check('expert_id' not in ma, f"import must NOT set expert_id: {ma.get('expert_id')}")
        check(ma['review'] is True, 'mask stays pending until accept')


def test_rolling_add_review_upsert_no_duplicate():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'gamma_20200101_ccc', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        base = dict(export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                    master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'gamma_20200101_ccc'}], **base)
        # second reviewer accumulates
        importer.import_rows([{'uid': u, 'code': 'AA', 'reviewer': 'omar', 'project_id': 'gamma_20200101_ccc'}], **base)
        # jane re-drops with a corrected code -> UPSERT (replace HER row only)
        importer.import_rows([{'uid': u, 'code': 'OFAV', 'reviewer': 'jane', 'project_id': 'gamma_20200101_ccc'}], **base)
        item = next(i for i in ReviewRepo(e['review_dir']).load_manifest()['items'] if i['uid'] == u)
        revs = {r['reviewer']: r['code'] for r in item['reviews']}
        check(revs.get('jane') == 'OFAV', f"jane upserted: {item['reviews']}")
        check(revs.get('omar') == 'AA', f"omar preserved: {item['reviews']}")
        janes = [r for r in item['reviews'] if r['reviewer'] == 'jane']
        check(len(janes) == 1, f"jane appears once (no dupe): {item['reviews']}")
        # library reviews/<uid>.json mirrors it
        detail = Library(e['library_dir']).load_reviews(u)
        dr = {r['reviewer']: r['code'] for r in detail['reviews']}
        check(dr.get('jane') == 'OFAV' and dr.get('omar') == 'AA', f"library mirror: {detail}")


def test_accept_sets_expert_id_and_removes():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'delta_20200101_ddd', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'delta_20200101_ddd'},
                              {'uid': u, 'code': 'PA', 'reviewer': 'omar', 'project_id': 'delta_20200101_ddd'}],
                             export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                             master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        # consensus says PA -> accept it
        res = importer.accept_uid(u, 'PA', review_dir=e['review_dir'], repo_url='',
                                  library_dir=e['library_dir'], master_codes=None,
                                  projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        check(res['ok'] and res['relabeled_seg'], f"accept relabeled seg: {res}")
        # removed from the live site (accept is the ONLY removal path)
        check(u not in ReviewRepo(e['review_dir']).pending_uids(), 'item removed from repo')
        # expert_id stamped on the segmentation mask (mode EXPERT)
        m = _seg_mask(ed, fn)
        check(m.get('expert_id', {}).get('mode') == 'EXPERT' and m['species'] == 'PA',
              f"expert_id stamped: {m.get('expert_id')}")
        check(m['review'] is False, 'mask no longer pending')
        check(m.get('status') == 'accepted',
              f"expert accept flips status to accepted for YOLO export: {m.get('status')}")
        # library record -> EXPERT, reviews detail carries accepted
        rec = Library(e['library_dir']).lookup(u)
        check(rec['mode'] == 'EXPERT' and rec['code'] == 'PA', f"library EXPERT: {rec}")
        det = Library(e['library_dir']).load_reviews(u)
        check(det['accepted'] and det['accepted']['code'] == 'PA', f"accepted persisted: {det['accepted']}")


def test_provenance_ledger_refreshed_on_import_and_accept():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        stem = os.path.splitext(fn)[0]
        # The mask starts at the build_mask_dict default status 'pending':
        # provenance counts found_* only for ACCEPTED non-review masks, and
        # accept_uid itself flips status to 'accepted', so found_expert must
        # appear even when the annotator never accepted the mask.
        ed, u = _export(e['projects_root'], 'mu_20200101_mmm', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        base = dict(export_dir='', review_dir=e['review_dir'], repo_url='',
                    library_dir=e['library_dir'], master_codes=None,
                    projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        st = importer.import_rows(
            [{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'mu_20200101_mmm'},
             {'uid': u, 'code': 'PA', 'reviewer': 'omar', 'project_id': 'mu_20200101_mmm'}], **base)
        check(st['provenance_refreshed'] == 1, f"import refreshed the ledger: {st}")
        led = {(r['basename'], r['label']): r['outcome'] for r in _ledger_rows(ed)}
        check(led.get((stem, 'PA')) == 'pending_expert',
              f"tentative code lands as pending_expert: {led}")
        # accept flips the SAME (frame, label) row to found_expert with no
        # annotator re-touch
        res = importer.accept_uid(u, 'PA', review_dir=e['review_dir'], repo_url='',
                                  library_dir=e['library_dir'], master_codes=None,
                                  projects_root=e['projects_root'], labeler='LO',
                                  git_push=False, log_fn=lambda m: None)
        check(res['ok'] and res['provenance_refreshed'] == 1,
              f"accept refreshed the ledger: {res}")
        rows = _ledger_rows(ed)
        led = {(r['basename'], r['label']): r['outcome'] for r in rows}
        check(led.get((stem, 'PA')) == 'found_expert', f"accept -> found_expert: {led}")
        check(not [r for r in rows if r['outcome'] == 'pending_expert'],
              f"no stale pending_expert rows after accept: {rows}")
        # project_id column mirrors the annotator's derivation (project dir name)
        check(all(r['project_id'] == 'mu_20200101_mmm' for r in rows),
              f"ledger project_id derived from the project dir: {rows}")


def test_operator_setup_persists_email_and_candidate_codes():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        setup = importer.save_operator_setup(
            review_dir=e['review_dir'], repo_url='', master_codes=None,
            email='jane@uni.edu', candidate_codes='ofav, ofra', git_push=False, log_fn=lambda m: None)
        check(setup['email'] == 'jane@uni.edu', setup)
        check(setup['candidate_codes'] == ['OFAV', 'OFRA'], f"upper + split: {setup}")
        # codes.json reflects the candidate codes (viewer surfaces them)
        cj = json.load(open(os.path.join(e['review_dir'], 'codes.json')))
        check('OFAV' in cj['candidate_codes'] and 'OFRA' in cj['candidate_codes'], cj['candidate_codes'])
        # contact email reflected on the manifest
        man = ReviewRepo(e['review_dir']).load_manifest()
        check('jane@uni.edu' in man.get('contacts', []), f"contact set: {man.get('contacts')}")
        # re-load from disk survives
        again = importer.load_operator_setup(e['review_dir'])
        check(again['email'] == 'jane@uni.edu' and again['candidate_codes'] == ['OFAV', 'OFRA'], again)
        # a later write_codes (e.g. ensure_repo) must NOT drop the operator extras
        repo = ReviewRepo(e['review_dir'], master_codes_path=None)
        repo.write_codes()
        importer._merge_candidate_codes(repo, importer.load_operator_setup(e['review_dir'])['candidate_codes'])
        cj2 = json.load(open(os.path.join(e['review_dir'], 'codes.json')))
        check('OFAV' in cj2['candidate_codes'], f"extras survive re-seed merge: {cj2['candidate_codes']}")


def test_consensus_rows_and_email_requests():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'epsilon_20200101_eee', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'], project_name='Epsilon Run')
        importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'epsilon_20200101_eee'},
                              {'uid': u, 'code': 'PA', 'reviewer': 'omar', 'project_id': 'epsilon_20200101_eee'}],
                             export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                             master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        rows = importer.consensus_rows(e['review_dir'], master_codes=None, library_dir=e['library_dir'])
        check(len(rows) == 1, f"one pending consensus row: {len(rows)}")
        r = rows[0]
        check(r['status'] == 'consensus' and r['suggested'] == 'PA', f"consensus suggested PA: {r}")
        check(r['n_reviewers'] == 2, f"two reviewers: {r}")
        # pending_by_project bucket carries the project_name
        buckets = ReviewRepo(e['review_dir']).pending_by_project()
        check(buckets and buckets[0]['project_name'] == 'Epsilon Run' and buckets[0]['count'] == 1, buckets)


def test_delete_project_removes_only_target_project():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        ed_a, ua = _export(e['projects_root'], 'alpha_20200101_aaa', 'TCRMP20201022_clip_SCP_T101.jpeg',
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        ed_b, ub = _export(e['projects_root'], 'beta_20200101_bbb', 'TCRMP20201022_clip_BWR_T201.jpeg',
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        c = _client(e)
        r = c.post('/expertids/api/delete_project',
                   json={'project_id': 'alpha_20200101_aaa', 'review_dir': e['review_dir']})
        d = r.get_json()
        check(r.status_code == 200 and d['ok'], f"delete ok: {d}")
        check(d['removed'] == 1 and d['project_id'] == 'alpha_20200101_aaa', d)
        check(d['pushed'] is False, f"git_push False -> not pushed: {d}")
        check(d['pending_total'] == 1, f"one project left pending: {d}")
        repo = ReviewRepo(e['review_dir'])
        check(repo.pending_uids() == [ub], f"only beta remains: {repo.pending_uids()}")
        check(not os.path.isdir(os.path.join(repo.items_dir, ua)), 'alpha item assets gone')
        check(os.path.isdir(os.path.join(repo.items_dir, ub)), 'beta item assets intact')
        # the cross-project library is untouched by project deletion
        check(Library(e['library_dir']).lookup(ua) is not None, 'library record survives delete')


def test_delete_project_blank_refused_and_unknown_zero():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        _export(e['projects_root'], 'gamma_20200101_ccc', 'TCRMP20201022_clip_SCP_T101.jpeg',
                review_dir=e['review_dir'], library_dir=e['library_dir'])
        c = _client(e)
        check(c.post('/expertids/api/delete_project', json={}).status_code == 400, 'missing refused')
        check(c.post('/expertids/api/delete_project', json={'project_id': '   '}).status_code == 400,
              'whitespace refused')
        check(c.post('/expertids/api/delete_project', data={'project_id': ''}).status_code == 400,
              'form blank refused')
        r = c.post('/expertids/api/delete_project',
                   json={'project_id': 'nope_does_not_exist', 'review_dir': e['review_dir']})
        d = r.get_json()
        check(r.status_code == 200 and d['ok'] and d['removed'] == 0, f"unknown -> removed 0: {d}")
        check(d['pending_total'] == 1, f"pending unchanged for unknown project: {d}")


def test_delete_project_push_flag_respected():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        _export(e['projects_root'], 'push_20200101_aaa', 'TCRMP20201022_clip_SCP_T101.jpeg',
                review_dir=e['review_dir'], library_dir=e['library_dir'])
        _export(e['projects_root'], 'hold_20200101_bbb', 'TCRMP20201022_clip_BWR_T201.jpeg',
                review_dir=e['review_dir'], library_dir=e['library_dir'])
        calls = []
        orig = ReviewRepo.commit_push

        def fake(self, message, push=True):
            calls.append({'message': message, 'push': push})
            return bool(push)

        ReviewRepo.commit_push = fake
        try:
            c = _client(e, git_push=True)
            d = c.post('/expertids/api/delete_project',
                       json={'project_id': 'push_20200101_aaa',
                             'review_dir': e['review_dir']}).get_json()
            check(d['ok'] and d['pushed'] is True, f"git_push setting honored: {d}")
            check(calls[-1]['push'] is True, f"push=True passed through: {calls[-1]}")
            check(calls[-1]['message'] == 'remove project push_20200101_aaa (1 items) from review site',
                  f"commit message: {calls[-1]['message']}")
            # explicit push:false body override wins over the git_push setting
            d2 = c.post('/expertids/api/delete_project',
                        json={'project_id': 'hold_20200101_bbb',
                              'review_dir': e['review_dir'], 'push': False}).get_json()
            check(d2['ok'] and d2['pushed'] is False, f"push override False: {d2}")
            check(calls[-1]['push'] is False, f"push=False passed through: {calls[-1]}")
        finally:
            ReviewRepo.commit_push = orig


def test_delete_project_remote_isolation_never_brands_foreign_dir():
    """Hammering-pass regression (2026-07-09): delete_project against an
    overridden review_dir used to run commit_push's origin scrub even for a
    local-only commit, so a caller that pointed review_dir at a scratch tree
    and sent push:false still got that tree's git origin rewritten to the
    provider's PRODUCTION remote (config-only production branding). A
    local-only commit must leave origin alone, and the sanctioned isolation
    switch review_repo_url:'' (CONTRACTS §5) must fully detach even when the
    provider defaults to push-on. Real commit_push here — no mocks — because
    the git side effects ARE the thing under test; nothing can reach a real
    remote (push:false in case one, no remote at all in case two)."""
    import subprocess
    from _reefreview.review_repo import GIT
    PROD = 'https://github.com/prod-owner/prod-pages.git'
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        prov = dict(e)
        prov['review_dir'] = os.path.join(t, 'provider_review')  # NOT the caller's dir
        foreign = e['review_dir']                                # caller's scratch tree
        _export(e['projects_root'], 'brand_20200101_aaa', 'TCRMP20201022_clip_SCP_T101.jpeg',
                review_dir=foreign, library_dir=e['library_dir'])
        _export(e['projects_root'], 'detach_20200101_bbb', 'TCRMP20201022_clip_BWR_T201.jpeg',
                review_dir=foreign, library_dir=e['library_dir'])
        c = _client(prov, git_push=True, repo_url=PROD)

        def origin_of(d):
            r = subprocess.run([GIT, '-C', d, 'remote', 'get-url', 'origin'],
                               capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None

        check(origin_of(foreign) is None, 'precondition: scratch tree has no origin')
        # 1) push:false alone (remote keys absent): the commit stays local and
        #    the scratch tree's origin stays unset.
        d = c.post('/expertids/api/delete_project',
                   json={'project_id': 'brand_20200101_aaa',
                         'review_dir': foreign, 'push': False}).get_json()
        check(d['ok'] and d['removed'] == 1 and d['pushed'] is False,
              f"local-only delete ran: {d}")
        check(origin_of(foreign) is None,
              f"push:false must not brand the scratch tree with the production "
              f"origin: {origin_of(foreign)}")
        lg = subprocess.run([GIT, '-C', foreign, 'log', '--oneline'],
                            capture_output=True, text=True).stdout
        check('remove project brand_20200101_aaa' in lg, f"removal committed locally: {lg}")
        # 2) review_repo_url:'' detaches even though the push key is ABSENT and
        #    the provider pushes by default — commit_push has no remote, so
        #    nothing can leave the machine and origin stays unset.
        d2 = c.post('/expertids/api/delete_project',
                    json={'project_id': 'detach_20200101_bbb',
                          'review_dir': foreign, 'review_repo_url': ''}).get_json()
        check(d2['ok'] and d2['removed'] == 1 and d2['pushed'] is False,
              f"detached delete never pushes: {d2}")
        check(origin_of(foreign) is None, 'origin still unset after detached delete')
        cfg = os.path.join(foreign, '.git', 'config')
        check(PROD not in (open(cfg).read() if os.path.exists(cfg) else ''),
              'production URL never written into the scratch .git/config')
        check(not os.path.exists(prov['review_dir']),
              'provider (production) review_dir untouched throughout')


def test_email_requests_carry_through_text():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        _export(e['projects_root'], 'iota_20200101_iii', 'TCRMP20201022_clip_SCP_T101.jpeg',
                review_dir=e['review_dir'], library_dir=e['library_dir'], project_name='Iota Run')
        c = _client(e, repo_url='https://github.com/owner/repo.git')
        d = c.get('/expertids/api/email_requests',
                  query_string={'review_dir': e['review_dir']}).get_json()
        check(d['ok'] and len(d['requests']) == 1, f"one request: {d}")
        req = d['requests'][0]
        check(set(req.keys()) == {'project_id', 'project_name', 'count', 'text'},
              f"response shape kept: {sorted(req.keys())}")
        txt = req['text']
        check(txt.startswith('Subject: TCRMP expert IDs requested: Iota Run'), txt)
        check('Please review the 1 pending mask(s) for project Iota Run.' in txt, txt)
        check('https://owner.github.io/repo/' in txt, f"pages url in text: {txt}")
        check('lauren.olinger@uvi.edu' in txt, f"operator email fallback: {txt}")
        check('IDK is fine' in txt and 'email it back to' in txt, txt)
        check('\u2014' not in txt, 'no em dashes in email text')
        # a saved operator email replaces the fallback
        importer.save_operator_setup(review_dir=e['review_dir'], repo_url='',
                                     master_codes=None, email='op@uvi.edu',
                                     git_push=False, log_fn=lambda m: None)
        d2 = c.get('/expertids/api/email_requests',
                   query_string={'review_dir': e['review_dir']}).get_json()
        txt2 = d2['requests'][0]['text']
        check('op@uvi.edu' in txt2 and 'lauren.olinger@uvi.edu' not in txt2,
              f"saved operator email used: {txt2}")


def test_consensus_unaffected_by_confidence_removal():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        ed, u = _export(e['projects_root'], 'kappa_20200101_kkk', 'TCRMP20201022_clip_SCP_T101.jpeg',
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        # old CSVs with a confidence column still import; blank confidence is fine
        rows = importer.parse_csv_text(
            'uid,code,confidence,reviewer,project_id\n'
            + u + ',PA,high,jane,kappa_20200101_kkk\n'
            + u + ',PA,,omar,kappa_20200101_kkk\n')
        check(rows[0]['confidence'] == 'high' and rows[1]['confidence'] == '',
              f"confidence parsing kept: {rows}")
        importer.import_rows(rows, export_dir='', review_dir=e['review_dir'], repo_url='',
                             library_dir=e['library_dir'], master_codes=None,
                             projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        crows = importer.consensus_rows(e['review_dir'], master_codes=None, library_dir=e['library_dir'])
        r = next(x for x in crows if x['uid'] == u)
        check(r['status'] == 'consensus' and r['suggested'] == 'PA', f"consensus intact: {r}")
        # confidence stays in STORAGE (manifest reviews) even though the UI no longer shows it
        by = {rv['reviewer']: rv for rv in r['reviews']}
        check(by['jane'].get('confidence') == 'high', f"stored confidence intact: {by['jane']}")
        check(by['omar'].get('confidence') == '', f"blank confidence tolerated: {by['omar']}")


def test_parse_csv_text_tolerates_oversized_field():
    # Regression (2026-07-09): a CSV field larger than csv.field_size_limit's
    # default (131072) crashed parse_csv_text with an uncaught _csv.Error.
    # CONTRACTS §4: the importer is tolerant — parse must survive any field
    # size (bad UIDs are reported downstream by is_safe_uid, never a crash).
    big = 'x' * (1024 * 1024)   # 1MB field, far past the 131072 default
    rows = importer.parse_csv_text(
        'uid,code,confidence,reviewer,project_id\n'
        + big + ',OFAV,,jane,run_x\n'
        + 'GOOD-1,PA,,jane,run_x\n')
    check(len(rows) == 2, f"both rows parsed: {len(rows)}")
    check(rows[0]['uid'] == big, 'oversized field preserved verbatim')
    check(rows[1]['uid'] == 'GOOD-1', 'following row still parsed')
    # Oversized NON-uid column tolerated too.
    rows2 = importer.parse_csv_text('uid,code,confidence,reviewer,project_id\n'
                                    + 'GOOD-2,' + big + ',,jane,run_x\n')
    check(len(rows2) == 1 and rows2[0]['uid'] == 'GOOD-2',
          f"oversized code column tolerated: {len(rows2)}")


def test_accept_api_honors_body_overrides_and_push_flag():
    """Regression (2026-07-09): POST /api/accept ignored every body override
    except review_dir — library_dir fell through to the provider's (production)
    library, git_push/push to the provider's push default, projects_root /
    export_dir to the provider's roots. An 'isolated' accept therefore upserted
    an EXPERT record + reviews sidecar into the LIVE library, pushed to the
    production remote, and never relabeled the caller's segmentations. The
    route must be stateless like import: body overrides win, absent keys fall
    back to the provider."""
    with tempfile.TemporaryDirectory() as t:
        # provider = "production" paths; iso = the caller's isolated env
        prod = {'projects_root': os.path.join(t, 'prod', 'inprocess'),
                'review_dir': os.path.join(t, 'prod', 'REVIEW'),
                'library_dir': os.path.join(t, 'prod', '_lib')}
        iso = {'projects_root': os.path.join(t, 'iso', 'inprocess'),
               'review_dir': os.path.join(t, 'iso', 'REVIEW'),
               'library_dir': os.path.join(t, 'iso', '_lib')}
        fn_iso = 'TCRMP20201022_clip_SCP_T101.jpeg'
        fn_prod = 'TCRMP20201022_clip_BWR_T201.jpeg'
        ed_iso, u_iso = _export(iso['projects_root'], 'iso_20200101_iii', fn_iso,
                                review_dir=iso['review_dir'], library_dir=iso['library_dir'])
        ed_prod, u_prod = _export(prod['projects_root'], 'prod_20200101_ppp', fn_prod,
                                  review_dir=prod['review_dir'], library_dir=prod['library_dir'])
        calls = []
        orig = ReviewRepo.commit_push

        def fake(self, message, push=True):
            calls.append({'push': push, 'remote': self.remote_url})
            return bool(push)

        ReviewRepo.commit_push = fake
        try:
            # provider says: production library, push on, production remote
            c = _client(prod, git_push=True, repo_url='https://github.com/owner/prod-pages.git')
            r = c.post('/expertids/api/accept', json={
                'uid': u_iso, 'code': 'PA', 'basis': 'operator', 'labeler': 'LO',
                'review_dir': iso['review_dir'], 'library_dir': iso['library_dir'],
                'projects_root': iso['projects_root'], 'export_dir': ed_iso,
                'review_repo_url': '', 'push': False})
            d = r.get_json()
            check(r.status_code == 200 and d.get('ok') is True, f"accept ok: {d}")
            check(d['relabeled_seg'] is True and d['export_dir'] == ed_iso,
                  f"projects_root override routed the relabel to iso: {d}")
            check(d['pushed'] is False and calls[-1]['push'] is False,
                  f"push:false body override wins over provider git_push=True: {d}, {calls[-1]}")
            check(calls[-1]['remote'] == '',
                  f"review_repo_url:'' override detached the production remote: {calls[-1]}")
            m = _seg_mask(ed_iso, fn_iso)
            check(m.get('expert_id', {}).get('mode') == 'EXPERT' and m['species'] == 'PA',
                  f"iso segmentations relabeled: {m.get('expert_id')}")
            check(Library(iso['library_dir']).lookup(u_iso)['mode'] == 'EXPERT',
                  'override library received the EXPERT record')
            check(Library(prod['library_dir']).lookup(u_iso) is None,
                  'provider (production) library untouched by an overridden accept')
            check(not Library(prod['library_dir']).load_reviews(u_iso).get('accepted'),
                  'no accepted sidecar leaked into the provider library')
            # absent overrides fall back to the provider paths (default kept)
            d2 = c.post('/expertids/api/accept',
                        json={'uid': u_prod, 'code': 'PA', 'labeler': 'LO'}).get_json()
            check(d2.get('ok') is True and d2['export_dir'] == ed_prod,
                  f"no overrides -> provider routing: {d2}")
            check(d2['pushed'] is True and calls[-1]['push'] is True
                  and calls[-1]['remote'] == 'https://github.com/owner/prod-pages.git',
                  f"no overrides -> provider git_push + remote: {calls[-1]}")
            check(Library(prod['library_dir']).lookup(u_prod)['mode'] == 'EXPERT',
                  'no overrides -> provider library written')
        finally:
            ReviewRepo.commit_push = orig


# ══ ADVERSARIAL ══════════════════════════════════════════════════════
def test_adv_oversized_csv_field_returns_json_not_html_500():
    # Regression (2026-07-09): POST /api/import with a >131072-char field
    # returned a raw Flask HTML 500 because blueprint.do_import called
    # parse_csv_text OUTSIDE its try/except. The endpoint must always answer
    # JSON: the giant uid parses, then is_safe_uid reports it (unsafe_uid).
    import io as _io
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        os.makedirs(e['projects_root'], exist_ok=True)
        c = _client(e)
        big = 'x' * (1024 * 1024)
        csv_text = ('uid,code,confidence,reviewer,project_id\n'
                    + big + ',OFAV,,jane,run_x\n')
        # JSON-body path.
        r = c.post('/expertids/api/import',
                   json={'csv_text': csv_text, 'reviewer': 'jane'})
        check(r.status_code != 500, f"no 500 on oversized field: {r.status_code}")
        check(r.is_json, f"JSON response, not HTML: {r.headers.get('Content-Type')}")
        d = r.get_json()
        check(d.get('ok') is True and d.get('unsafe_uid') == 1,
              f"giant uid reported unsafe, import survives: {d}")
        check(d.get('reviews_added') == 0, f"nothing ingested for the bad row: {d}")
        # File-upload path.
        r2 = c.post('/expertids/api/import',
                    data={'file': (_io.BytesIO(csv_text.encode()), 'review.csv'),
                          'reviewer': 'jane'},
                    content_type='multipart/form-data')
        check(r2.status_code != 500, f"no 500 on uploaded file: {r2.status_code}")
        check(r2.is_json, f"JSON response on upload: {r2.headers.get('Content-Type')}")
        d2 = r2.get_json()
        check(d2.get('ok') is True and d2.get('unsafe_uid') == 1,
              f"upload path reports unsafe uid: {d2}")


def test_adv_unrouted_uid_reported_never_silent():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        os.makedirs(e['projects_root'], exist_ok=True)
        # A UID with a project_id that resolves to NO project on disk, no library
        # record, and no open export_dir -> must be reported as unrouted.
        st = importer.import_rows(
            [{'uid': 'GHOST-1', 'code': 'PA', 'reviewer': 'jane', 'project_id': 'nope_does_not_exist'}],
            export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
            master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        check('GHOST-1' in st['unrouted'], f"unrouted reported: {st['unrouted']}")
        check(st['reviews_added'] == 0, f"nothing added for unrouted: {st}")


def test_adv_identical_frame_names_across_projects_dont_cross_relabel():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'   # SAME filename in BOTH projects
        # Distinct mask geometry per project so the content-stable UIDs differ
        # while the frame FILENAME is identical (the cross-contamination risk).
        ed_a, ua = _export(e['projects_root'], 'projA_20200101_aaa', fn, sq=(40, 110, 40, 110),
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        ed_b, ub = _export(e['projects_root'], 'projB_20200101_bbb', fn, sq=(90, 160, 90, 160),
                           review_dir=e['review_dir'], library_dir=e['library_dir'])
        check(ua != ub, 'distinct UIDs for the same frame in two projects')
        # Accept ONLY project A's uid; project B's identical-named frame must be untouched.
        importer.import_rows([{'uid': ua, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'projA_20200101_aaa'}],
                             export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                             master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        res = importer.accept_uid(ua, 'PA', review_dir=e['review_dir'], repo_url='',
                                  library_dir=e['library_dir'], master_codes=None,
                                  projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        check(res['ok'] and res['export_dir'] == ed_a, f"accept routed to A only: {res}")
        ma = _seg_mask(ed_a, fn); mb = _seg_mask(ed_b, fn)
        check(ma.get('expert_id', {}).get('mode') == 'EXPERT', f"A relabeled: {ma.get('expert_id')}")
        check('expert_id' not in mb and mb['species'] == 'REVIEW',
              f"B's identical-named frame NOT cross-relabeled: {mb.get('expert_id')}, {mb.get('species')}")


def test_adv_conflict_not_auto_accepted():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'zeta_20200101_zzz', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'zeta_20200101_zzz'},
                              {'uid': u, 'code': 'AA', 'reviewer': 'omar', 'project_id': 'zeta_20200101_zzz'}],
                             export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                             master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        rows = importer.consensus_rows(e['review_dir'], master_codes=None, library_dir=e['library_dir'])
        r = next(x for x in rows if x['uid'] == u)
        check(r['status'] == 'conflict' and r['suggested'] == '', f"conflict, no auto code: {r}")
        # The item is STILL pending — a conflict is never auto-removed/accepted.
        check(u in ReviewRepo(e['review_dir']).pending_uids(), 'conflict item still pending (not auto-accepted)')
        m = _seg_mask(ed, fn)
        check('expert_id' not in m, f"no expert_id on a conflicted mask: {m.get('expert_id')}")


def test_adv_redrop_same_reviewer_upserts_in_segmentations():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'eta_20200101_hhh', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        base = dict(export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                    master_codes=None, projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'eta_20200101_hhh'}], **base)
        importer.import_rows([{'uid': u, 'code': 'OFAV', 'reviewer': 'jane', 'project_id': 'eta_20200101_hhh'}], **base)
        m = _seg_mask(ed, fn)
        janes = [r for r in (m.get('reviews') or []) if r['reviewer'] == 'jane']
        check(len(janes) == 1 and janes[0]['code'] == 'OFAV',
              f"seg mask upserts jane's row (no dupe): {m.get('reviews')}")


def test_adv_blank_reviewer_row_reported_not_added():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'theta_20200101_ttt', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        # No reviewer column, no default reviewer -> the row must be reported, not
        # silently added (CONTRACTS §4 reviewer REQUIRED).
        st = importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': '', 'project_id': 'theta_20200101_ttt'}],
                                  export_dir='', review_dir=e['review_dir'], repo_url='', library_dir=e['library_dir'],
                                  master_codes=None, projects_root=e['projects_root'], git_push=False,
                                  default_reviewer='', log_fn=lambda m: None)
        check(u in st['unrouted'], f"blank-reviewer row reported: {st['unrouted']}")
        item = next(i for i in ReviewRepo(e['review_dir']).load_manifest()['items'] if i['uid'] == u)
        check(item['reviews'] == [], f"no tentative review added for blank reviewer: {item['reviews']}")


def test_adv_provenance_unavailable_never_fails_import_or_accept():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'nu_20200101_nnn', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        base = dict(export_dir='', review_dir=e['review_dir'], repo_url='',
                    library_dir=e['library_dir'], master_codes=None,
                    projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        # IMPORT with the provenance module broken (helper's own guard): the
        # cross-package import resolves via sys.modules, so a broken module
        # there simulates an unavailable/failing provenance backend.
        import types
        fake = types.ModuleType('provenance')

        def _boom(*_a, **_k):
            raise RuntimeError('provenance backend unavailable')
        fake.compute_label_outcomes = _boom
        saved = sys.modules.get('provenance')
        sys.modules['provenance'] = fake
        try:
            st = importer.import_rows(
                [{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'nu_20200101_nnn'}], **base)
        finally:
            if saved is not None:
                sys.modules['provenance'] = saved
            else:
                sys.modules.pop('provenance', None)
        check(st['reviews_added'] == 1 and st['unrouted'] == [],
              f"import succeeds with provenance broken: {st}")
        check(st['provenance_refreshed'] == 0, f"refresh reported 0: {st}")
        m = _seg_mask(ed, fn)
        check(m['reviews'][0]['code'] == 'PA', f"tentative review still landed: {m.get('reviews')}")
        check(not os.path.exists(os.path.join(ed, 'label_provenance.csv')),
              'no ledger written when provenance is broken')
        # ACCEPT with the whole helper crashing (call-site guard).
        orig = importer._refresh_provenance_ledger

        def _crash(*_a, **_k):
            raise RuntimeError('refresh machinery exploded')
        importer._refresh_provenance_ledger = _crash
        try:
            res = importer.accept_uid(u, 'PA', review_dir=e['review_dir'], repo_url='',
                                      library_dir=e['library_dir'], master_codes=None,
                                      projects_root=e['projects_root'], labeler='LO',
                                      git_push=False, log_fn=lambda m: None)
        finally:
            importer._refresh_provenance_ledger = orig
        check(res['ok'] and res['relabeled_seg'],
              f"accept succeeds with refresh crashing: {res}")
        check(res['provenance_refreshed'] == 0, f"refresh reported 0: {res}")
        m = _seg_mask(ed, fn)
        check(m.get('expert_id', {}).get('mode') == 'EXPERT',
              f"expert_id stamped despite refresh crash: {m.get('expert_id')}")


def test_adv_unwritable_ledger_path_never_fails_import_or_accept():
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'xi_20200101_xxx', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        # A DIRECTORY squatting on the ledger path makes the csv upsert
        # (os.replace onto it) fail -> the refresh must swallow that.
        os.makedirs(os.path.join(ed, 'label_provenance.csv'))
        st = importer.import_rows(
            [{'uid': u, 'code': 'PA', 'reviewer': 'jane', 'project_id': 'xi_20200101_xxx'}],
            export_dir='', review_dir=e['review_dir'], repo_url='',
            library_dir=e['library_dir'], master_codes=None,
            projects_root=e['projects_root'], git_push=False, log_fn=lambda m: None)
        check(st['reviews_added'] == 1 and st['unrouted'] == [],
              f"import succeeds with unwritable ledger path: {st}")
        check(st['provenance_refreshed'] == 0, f"refresh reported 0: {st}")
        m = _seg_mask(ed, fn)
        check(m['reviews'][0]['code'] == 'PA', f"tentative review still landed: {m.get('reviews')}")
        res = importer.accept_uid(u, 'PA', review_dir=e['review_dir'], repo_url='',
                                  library_dir=e['library_dir'], master_codes=None,
                                  projects_root=e['projects_root'], labeler='LO',
                                  git_push=False, log_fn=lambda m: None)
        check(res['ok'] and res['relabeled_seg'] and res['provenance_refreshed'] == 0,
              f"accept succeeds with unwritable ledger path: {res}")
        m = _seg_mask(ed, fn)
        check(m.get('expert_id', {}).get('mode') == 'EXPERT',
              f"expert_id stamped despite unwritable ledger: {m.get('expert_id')}")


def test_adv_concurrent_imports_same_project_lose_nothing():
    """Two simultaneous CSV imports (different reviewers, same project) must
    not race on segmentations.json (CONTRACTS §5 rolling upsert). Pre-fix,
    importer._save used an unlocked read-modify-write with a FIXED '<path>.tmp'
    name: one import could crash (FileNotFoundError on os.replace, surfacing
    as HTTP 500 mid-import) or silently lose the other reviewer's seg-level
    reviews[] row while the flock-protected manifest kept both, so manifest
    and segmentations diverged. Multiple frames per project widen the write
    window; a barrier makes the two imports collide."""
    import threading
    fns = ['TCRMP20201022_clip_SCP_T1%02d.jpeg' % i for i in (1, 2, 3)]
    for trial in range(8):
        with tempfile.TemporaryDirectory() as t:
            e = _env(t)
            pid = 'rho_20200101_rrr'
            # ONE project, THREE flagged frames in one segmentations.json (the
            # per-fn _export helper would overwrite it): build the combined
            # segmap by hand, export all frames in one pass, persist the
            # review_uid-stamped segmap.
            pdir = os.path.join(e['projects_root'], pid)
            ed = os.path.join(pdir, 'step5_segmentImages')
            segdir = os.path.join(ed, 'segmentations', '2020')
            raw = os.path.join(ed, 'raw')
            os.makedirs(segdir, exist_ok=True)
            os.makedirs(raw, exist_ok=True)
            with open(os.path.join(pdir, 'project.json'), 'w') as f:
                json.dump({'id': pid, 'name': 'rho'}, f)
            segmap = {}
            for fn in fns:
                mask = _review_mask(
                    mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                    [60, 60, 139, 139], [[60, 60, 140, 60, 140, 140, 60, 140]],
                    100, 100)
                img = os.path.join(raw, fn)
                Image.fromarray(
                    (np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
                segmap[fn] = {'image_path': 'raw/' + fn, 'image_path_abs': img,
                              'image_width': 200, 'image_height': 200,
                              'masks': [mask], 'reviewed': False,
                              'exported': False}
            st = review_export.export_flagged_masks(
                segmap, fns, export_dir=ed, review_dir=e['review_dir'],
                repo_url='', library_dir=e['library_dir'], master_codes=None,
                contacts=[], featured_codes=['PA'], project_id=pid,
                project_name='rho', git_push=False, log_fn=lambda m: None)
            uids = st['uids']
            check(len(uids) == len(fns), f'trial {trial}: export seeded {st}')
            with open(os.path.join(segdir, 'segmentations.json'), 'w') as f:
                json.dump(segmap, f)
            base = dict(export_dir='', review_dir=e['review_dir'], repo_url='',
                        library_dir=e['library_dir'], master_codes=None,
                        projects_root=e['projects_root'], git_push=False,
                        log_fn=lambda m: None)
            errs = []
            gate = threading.Barrier(2)

            def worker(reviewer, code):
                try:
                    gate.wait(timeout=30)
                    importer.import_rows(
                        [{'uid': u, 'code': code, 'reviewer': reviewer,
                          'project_id': pid} for u in uids], **base)
                except Exception as ex:
                    errs.append(f'{reviewer}: {type(ex).__name__}: {ex}')

            th = [threading.Thread(target=worker, args=('jane', 'PA')),
                  threading.Thread(target=worker, args=('omar', 'AA'))]
            for x in th:
                x.start()
            for x in th:
                x.join()
            check(errs == [],
                  f'trial {trial}: concurrent import crashed: {errs}')
            # The atomic replace must not narrow permissions (mkstemp tmp
            # files are 0600; the target keeps its umask-default mode).
            segp = os.path.join(segdir, 'segmentations.json')
            check(os.stat(segp).st_mode & 0o044,
                  f'trial {trial}: seg file lost group/world read after import')
            items = {i['uid']: i for i in
                     ReviewRepo(e['review_dir']).load_manifest()['items']}
            for fn, u in zip(fns, uids):
                m = _seg_mask(ed, fn)
                srev = {r['reviewer'] for r in (m.get('reviews') or [])}
                check(srev == {'jane', 'omar'},
                      f'trial {trial} {fn}: seg reviews[] lost a reviewer '
                      f'(manifest/seg diverge): {sorted(srev)}')
                mrev = {r['reviewer'] for r in items[u]['reviews']}
                check(mrev == {'jane', 'omar'},
                      f'trial {trial} {fn}: manifest lost a reviewer: '
                      f'{sorted(mrev)}')
                # Library detail mirror raced the same way (fixed tmp name in
                # save_reviews + unlocked read-modify-write in import_rows).
                lrev = {r['reviewer'] for r in
                        (Library(e['library_dir']).load_reviews(u)
                         .get('reviews') or [])}
                check(lrev == {'jane', 'omar'},
                      f'trial {trial} {fn}: library reviews/<uid>.json lost '
                      f'a reviewer: {sorted(lrev)}')


def test_adv_import_remote_overrides_isolate_from_provider_repo():
    """Hammering-pass regression (2026-07-09): /api/import honored path
    overrides but NOT review_repo_url/git_push, so an import pointed at a
    scratch review_dir was still git-inited with the provider's PRODUCTION
    remote and its manifest pushed to the live Pages repo. The body's remote
    overrides must isolate (explicit review_repo_url '' = no remote,
    git_push false = no push) on BOTH request paths, and ABSENT keys must
    keep provider behavior (CONTRACTS §5 per-request remote isolation)."""
    import io
    PROD = 'https://github.com/prod-owner/prod-pages.git'
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        pid = 'sigma_20200101_sss'
        ed, u = _export(e['projects_root'], pid, fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        # Provider mimics production: real remote, push on, and a review_dir of
        # its own that the overridden requests must never touch.
        prov = dict(e)
        prov['review_dir'] = os.path.join(t, 'provider_review')
        calls = []
        orig = ReviewRepo.commit_push

        def fake(self, message, push=True):
            calls.append({'remote': self.remote_url, 'push': push})
            return bool(push and self.remote_url)

        ReviewRepo.commit_push = fake
        try:
            c = _client(prov, git_push=True, repo_url=PROD)
            csv_line = f'uid,code,confidence,reviewer,project_id\n{u},PA,,jane,{pid}\n'
            # 1) FULL isolation (JSON path): scratch review_dir + explicit
            #    no-remote/no-push.
            d = c.post('/expertids/api/import', json={
                'csv_text': csv_line, 'review_dir': e['review_dir'],
                'library_dir': e['library_dir'], 'export_dir': '',
                'review_repo_url': '', 'git_push': False}).get_json()
            check(d.get('ok') and d.get('reviews_added') == 1, f"isolated import ran: {d}")
            check(d.get('pushed') is False, f"isolated import must not push: {d}")
            check(calls[-1]['remote'] == '' and calls[-1]['push'] is False,
                  f"remote detached + push off: {calls[-1]}")
            cfg = os.path.join(e['review_dir'], '.git', 'config')
            cfg_text = open(cfg).read() if os.path.exists(cfg) else ''
            check(PROD not in cfg_text,
                  'production remote never wired into the scratch repo')
            check(not os.path.exists(prov['review_dir']),
                  'provider (production) review_dir untouched by the overridden import')
            # 2) The multipart-form path honors the same overrides.
            d2 = c.post('/expertids/api/import', data={
                'review_dir': e['review_dir'], 'library_dir': e['library_dir'],
                'export_dir': '', 'review_repo_url': '', 'git_push': '0',
                'file': (io.BytesIO(csv_line.encode()), 'ids.csv')}).get_json()
            check(d2.get('ok') and d2.get('pushed') is False,
                  f"form-path overrides honored: {d2}")
            check(calls[-1]['remote'] == '' and calls[-1]['push'] is False,
                  f"form path detached too: {calls[-1]}")
            # 3) ABSENT keys keep production behavior (provider remote + push);
            #    the panel UI never sends the remote keys.
            d3 = c.post('/expertids/api/import', json={
                'csv_text': csv_line, 'review_dir': e['review_dir'],
                'library_dir': e['library_dir'], 'export_dir': ''}).get_json()
            check(d3.get('ok') and d3.get('pushed') is True,
                  f"no overrides -> provider remote push preserved: {d3}")
            check(calls[-1]['remote'] == PROD and calls[-1]['push'] is True,
                  f"provider fallback intact: {calls[-1]}")
            # 4) Accept honors the same isolation switches.
            d4 = c.post('/expertids/api/accept', json={
                'uid': u, 'code': 'PA', 'review_dir': e['review_dir'],
                'library_dir': e['library_dir'],
                'projects_root': e['projects_root'],
                'review_repo_url': '', 'git_push': False}).get_json()
            check(d4.get('ok') and d4.get('pushed') is False, f"isolated accept: {d4}")
            check(calls[-1]['remote'] == '' and calls[-1]['push'] is False,
                  f"accept remote detached: {calls[-1]}")
        finally:
            ReviewRepo.commit_push = orig


def test_parse_csv_text_strips_utf8_bom():
    # Regression (2026-07-09): an Excel round-trip prepends a UTF-8 BOM, so the
    # first header cell read '\ufeffuid', header detection failed, and the file
    # parsed positionally with the header leaking as a phantom data row.
    rows = importer.parse_csv_text(
        '\ufeffuid,code,confidence,reviewer,project_id\r\n'
        'BOM-1,PA,high,jane,run_x\r\n'
        'BOM-2,SS,,omar,run_y\r\n')
    check(len(rows) == 2, f"header consumed, 2 data rows: {len(rows)}")
    check(rows[0]['uid'] == 'BOM-1' and rows[0]['reviewer'] == 'jane',
          f"header-mapped fields: {rows[0]}")
    check(rows[1]['project_id'] == 'run_y', f"column order honored: {rows[1]}")
    check(not any(r['uid'].lower() == 'uid' or '\ufeff' in r['uid'] for r in rows),
          'no phantom header row leaked as data')


def test_adv_unwritable_seg_tree_reports_not_crashes():
    # Regression (2026-07-09): a read-only segmentations/<year>/ dir made
    # _tentative_in_segmentations raise PermissionError out of import_rows AFTER
    # the manifest review landed - crash plus partial state. Now the import
    # succeeds and reports the failure per uid in stats['seg_write_failed'].
    import stat
    with tempfile.TemporaryDirectory() as t:
        e = _env(t)
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        ed, u = _export(e['projects_root'], 'ro_20200101_rrr', fn,
                        review_dir=e['review_dir'], library_dir=e['library_dir'])
        # Find the segmentations file import would write and freeze its dir.
        seg_files = list(importer._iter_segmentation_files(ed))
        check(len(seg_files) >= 1, f"fixture has a seg file: {seg_files}")
        seg_dir = os.path.dirname(seg_files[0])
        os.chmod(seg_dir, stat.S_IRUSR | stat.S_IXUSR)
        try:
            st = importer.import_rows(
                [{'uid': u, 'code': 'PA', 'reviewer': 'jane',
                  'project_id': 'ro_20200101_rrr'}],
                export_dir='', review_dir=e['review_dir'], repo_url='',
                library_dir=e['library_dir'], master_codes=None,
                projects_root=e['projects_root'], git_push=False,
                log_fn=lambda m: None)
        finally:
            os.chmod(seg_dir, stat.S_IRWXU)
        check(st['reviews_added'] == 1, f"manifest review still landed: {st}")
        check(len(st['seg_write_failed']) >= 1
              and st['seg_write_failed'][0]['uid'] == u,
              f"failure reported per uid: {st['seg_write_failed']}")
        # The tentative review is on the site even though the seg write failed.
        item = [i for i in ReviewRepo(e['review_dir']).load_manifest()['items']
                if i['uid'] == u][0]
        check(any(r['reviewer'] == 'jane' for r in item['reviews']),
              'tentative review visible on the site')


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    print(f"Running {len(tests)} expertids tests...\n")
    for fn in tests:
        run(fn)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [(n, e) for n, ok, e in _RESULTS if not ok]
    print(f"\n==== {passed}/{len(_RESULTS)} passed ====")
    if failed:
        print("\nFAILURES:")
        for n, e in failed:
            print(f"\n--- {n} ---\n{e}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == '__main__':
    main()
