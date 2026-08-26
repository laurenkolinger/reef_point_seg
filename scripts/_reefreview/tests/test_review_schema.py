"""
Per-reviewer + multi-project + site-name schema tests (CONTRACTS §2,§3,§5,§6,§7,§8).

Covers the review-core cluster:
  review_repo.add_review (rolling upsert keyed by reviewer)
  review_repo.accept_item (sets accepted + removes; accept is the ONLY removal)
  review_repo.pending_by_project (project bucketing)
  review_repo.reviewer_names (distinct-reviewer tracking)
  review_repo.write_codes -> codes.json carries sites + candidate_codes
  review_export site_full resolution + new-item reviews:[]/accepted:null
  library.load_reviews/save_reviews per-uid round-trip
  codes.load_sites helper

Three layers: smoke (imports + codes.json emission), unit (logic round-trips),
adversarial (skeptic edges: blank reviewer, re-drop same reviewer, unsafe UID,
accept-then-gone, missing project bucket, already-published item missing fields).

No pytest:  env/bin/python scripts/_reefreview/tests/test_review_schema.py
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

from _reefreview import mask_geom, review_export, codes as codes_mod
from _reefreview.library import Library, UnsafeUidError
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


# ── helpers ─────────────────────────────────────────────────────────
def _repo(t, name='R'):
    repo = ReviewRepo(os.path.join(t, name), remote_url='',
                      master_codes_path=None, log_fn=lambda m: None)
    repo.ensure_repo()
    return repo


def _square_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m


def _review_mask(rle, bbox, poly, sx, sy, status='pending'):
    return {'id': 0, 'label': 'A', 'species': 'REVIEW', 'name': '', 'category': 'Review',
            'source_x': sx, 'source_y': sy, 'polygon_px': poly,
            'polygon_norm': [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]], 'rle': rle,
            'bbox': bbox, 'area': 100, 'score': 0.9, 'status': status, 'review': True}


def _make_project(root, fn, mask, pid='run_t_20990101_aaa'):
    proj = os.path.join(root, 'inprocess', pid, 'step5_segmentImages')
    raw = os.path.join(proj, 'raw'); os.makedirs(raw, exist_ok=True)
    img = os.path.join(raw, fn)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
    seg = {'image_path': 'raw/' + fn, 'image_path_abs': img, 'image_width': 200,
           'image_height': 200, 'masks': [mask], 'reviewed': False, 'exported': False}
    return proj, {fn: seg}


def _export_one(t, *, fn='TCRMP20201022_clip_SCP_T101.jpeg', candidate_codes=None,
                site_codes=None, project_id='run_t', project_name=''):
    mask = _review_mask(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                        [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
    proj, segmap = _make_project(t, fn, mask)
    rd = os.path.join(t, 'REVIEW'); ld = os.path.join(t, '_lib')
    st = review_export.export_flagged_masks(
        segmap, [fn], export_dir=proj, review_dir=rd, repo_url='', library_dir=ld,
        master_codes=None, contacts=['x@y.com'], featured_codes=['PA'],
        project_id=project_id, project_name=project_name,
        candidate_codes=candidate_codes, site_codes=site_codes,
        git_push=False, log_fn=lambda m: None)
    return st, rd, ld


# ══ SMOKE ════════════════════════════════════════════════════════════
def test_smoke_imports_and_new_api_present():
    repo = ReviewRepo.__init__
    for attr in ('add_review', 'accept_item', 'pending_by_project',
                 'reviewer_names', 'write_codes'):
        check(hasattr(ReviewRepo, attr), f"ReviewRepo missing {attr}")
    for attr in ('load_reviews', 'save_reviews', 'reviews_path'):
        check(hasattr(Library, attr), f"Library missing {attr}")
    check(hasattr(codes_mod, 'load_sites'), 'codes.load_sites missing')


def test_smoke_codes_json_carries_sites_and_candidate_codes():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        cj = json.load(open(os.path.join(repo.root, 'codes.json')))
        check('sites' in cj and isinstance(cj['sites'], dict), 'sites present + dict')
        check('candidate_codes' in cj and isinstance(cj['candidate_codes'], list),
              'candidate_codes present + list')
        # default repo loads the real supporting_data/site_codes.csv
        check(cj['sites'].get('GRP') == 'Great Pond', f"GRP map: {cj['sites'].get('GRP')}")
        check(cj['candidate_codes'] == [], f"default candidate_codes empty: {cj['candidate_codes']}")


# ══ UNIT ═════════════════════════════════════════════════════════════
def test_load_sites_helper():
    sites = codes_mod.load_sites()  # real csv
    check(sites.get('GRP') == 'Great Pond', sites.get('GRP'))
    check(sites.get('BWR') == 'Brewers Bay', sites.get('BWR'))
    # explicit small csv, tolerant of blanks
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, 's.csv')
        with open(p, 'w') as f:
            f.write('site,site_code\nFoo Reef,foo\n,\nBar,BAR\n')
        m = codes_mod.load_sites(p)
        check(m == {'FOO': 'Foo Reef', 'BAR': 'Bar'}, f"uppercased keys, blanks dropped: {m}")
    check(codes_mod.load_sites(os.path.join(t, 'gone.csv')) == {}, 'missing file -> {}')


def test_add_item_initializes_reviews_and_accepted():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1', 'site': 'SCP'})
        item = repo.load_manifest()['items'][0]
        check(item.get('reviews') == [], f"reviews default []: {item}")
        check(item.get('accepted') is None, f"accepted default None: {item}")


def test_add_review_rolling_upsert_by_reviewer():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1', 'site': 'SCP', 'project_id': 'p1'})
        repo.add_review('U1', 'lauren', 'PA', 'high')
        repo.add_review('U1', 'bob', 'SS', 'low')
        repo.add_review('U1', 'lauren', 'MM', 'high')   # re-drop overwrites lauren only
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'U1')
        by = {r['reviewer']: r for r in item['reviews']}
        check(len(item['reviews']) == 2, f"one row per reviewer: {item['reviews']}")
        check(by['lauren']['code'] == 'MM', f"lauren overwritten: {by['lauren']}")
        check(by['bob']['code'] == 'SS', f"bob untouched: {by['bob']}")
        check('at' in by['lauren'] and by['lauren']['confidence'] == 'high', by['lauren'])


def test_reviewer_names_tracking():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})
        repo.add_item('U2', {'uid': 'U2'})
        repo.add_review('U1', 'lauren', 'PA')
        repo.add_review('U2', 'bob', 'SS')
        repo.add_review('U1', 'lauren', 'PA')   # dup reviewer -> no dup name
        check(repo.reviewer_names() == ['lauren', 'bob'], repo.reviewer_names())
        check(repo.load_manifest().get('reviewer_names') == ['lauren', 'bob'],
              'top-level reviewer_names persisted')


def test_accept_item_sets_accepted_and_removes():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1', 'site': 'SCP'})
        repo.add_review('U1', 'lauren', 'PA', 'high')
        acc = repo.accept_item('U1', 'PA', labeler='operator', basis='consensus')
        check(acc['code'] == 'PA' and acc['mode'] == 'EXPERT', acc)
        check(acc['labeler'] == 'operator' and acc['basis'] == 'consensus', acc)
        check('at' in acc, 'accepted stamped at')
        check('U1' not in repo.pending_uids(), 'accept removes from repo')
        check(not os.path.isdir(os.path.join(repo.items_dir, 'U1')), 'items folder gone')


def test_pending_by_project_bucketing():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'beta', 'project_name': 'Beta Reef'})
        repo.add_item('A2', {'uid': 'A2', 'project_id': 'beta', 'project_name': 'Beta Reef'})
        repo.add_item('B1', {'uid': 'B1', 'project_id': 'alpha', 'project_name': 'Alpha Reef'})
        buckets = repo.pending_by_project()
        check([b['project_id'] for b in buckets] == ['alpha', 'beta'], f"sorted by name: {buckets}")
        check(buckets[0] == {'project_id': 'alpha', 'project_name': 'Alpha Reef', 'count': 1}, buckets[0])
        check(buckets[1]['count'] == 2, f"beta count: {buckets[1]}")


def test_remove_project_single_locked_pass():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.set_contacts(['op@uvi.edu'])
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'p1', 'project_name': 'One'})
        repo.add_item('A2', {'uid': 'A2', 'project_id': 'p1', 'project_name': 'One'})
        repo.add_item('B1', {'uid': 'B1', 'project_id': 'p2', 'project_name': 'Two'})
        repo.add_review('A1', 'lauren', 'PA')
        repo.add_review('B1', 'bob', 'SS')
        res = repo.remove_project('p1')
        check(res['removed'] == 2 and sorted(res['uids']) == ['A1', 'A2'], res)
        man = repo.load_manifest()
        check([i['uid'] for i in man['items']] == ['B1'], f"other project intact: {man['items']}")
        check(man['count'] == 1, f"count recomputed: {man['count']}")
        check(not os.path.isdir(os.path.join(repo.items_dir, 'A1')), 'A1 assets gone')
        check(not os.path.isdir(os.path.join(repo.items_dir, 'A2')), 'A2 assets gone')
        check(os.path.isdir(os.path.join(repo.items_dir, 'B1')), 'B1 assets intact')
        check(man.get('reviewer_names') == ['lauren', 'bob'],
              f"reviewer_names preserved: {man.get('reviewer_names')}")
        check(man.get('contacts') == ['op@uvi.edu'], f"contacts preserved: {man.get('contacts')}")


def test_export_stamps_site_full_and_candidate_codes():
    with tempfile.TemporaryDirectory() as t:
        st, rd, ld = _export_one(t, candidate_codes=['XCAND', 'YCAND'])
        uid = st['uids'][0]
        item = next(i for i in ReviewRepo(rd, remote_url='').load_manifest()['items']
                    if i['uid'] == uid)
        # SCP is in the real supporting_data/site_codes.csv -> South Capella
        check(item.get('site') == 'SCP', f"site code: {item}")
        check(item.get('site_full') == 'South Capella', f"site_full resolved: {item}")
        check(item.get('candidate_codes') == ['XCAND', 'YCAND'], f"candidate_codes stamped: {item}")
        check(item.get('reviews') == [] and item.get('accepted') is None,
              f"new item reviews/accepted empty: {item}")


def test_export_codes_json_carries_sites():
    with tempfile.TemporaryDirectory() as t:
        st, rd, ld = _export_one(t)
        cj = json.load(open(os.path.join(rd, 'codes.json')))
        check(cj['sites'].get('SCP') == 'South Capella', f"sites in export codes.json: {cj['sites'].get('SCP')}")


def test_candidate_codes_harvested_from_master_candidate_column():
    # CONTRACTS §2: candidate_codes default [] OR harvested from a master-codes
    # 'candidate' truthy column when present. Canonical master_codes.csv has no
    # such column -> stays []; a file that declares one is harvested.
    with tempfile.TemporaryDirectory() as t:
        mc = os.path.join(t, 'mc.csv')
        with open(mc, 'w') as f:
            f.write('code,category,name,candidate\n'
                    'PA,Coral,Porites,yes\n'      # truthy
                    'SS,Coral,Siderastrea,\n'     # blank -> excluded
                    'XX,Coral,Xeno,1\n'           # truthy
                    'YY,Coral,Yeno,false\n'       # falsey -> excluded
                    'ZZ,Coral,Zeno,candidate\n')  # truthy keyword
        repo = ReviewRepo(os.path.join(t, 'R'), remote_url='',
                          master_codes_path=mc, log_fn=lambda m: None)
        repo.ensure_repo()
        cj = json.load(open(os.path.join(repo.root, 'codes.json')))
        check(cj['candidate_codes'] == ['PA', 'XX', 'ZZ'],
              f"truthy candidate column harvested in order: {cj['candidate_codes']}")
        # the canonical (column-less) master codes must NOT invent the flag
        canon = codes_mod.load_codes()
        check(all('candidate' not in e for e in canon['codes']),
              'canonical master_codes gains no candidate key')
        check(ReviewRepo._candidate_codes(canon) == [],
              'canonical master_codes harvest stays []')


def test_add_review_preserves_at_and_other_reviewers_order():
    # An explicit `at` is stored verbatim; a re-drop by one reviewer must not
    # disturb the OTHER reviewers' rows (rolling upsert keyed by reviewer).
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})
        repo.add_review('U1', 'a', 'PA', 'high', at='2020-01-01T00:00:00')
        repo.add_review('U1', 'b', 'SS', 'low')
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'U1')
        by = {r['reviewer']: r for r in item['reviews']}
        check(by['a']['at'] == '2020-01-01T00:00:00', f"explicit at kept: {by['a']}")
        check('at' in by['b'] and by['b']['at'], f"auto at stamped for b: {by['b']}")
        # re-drop a -> b's row is byte-identical (untouched)
        b_before = dict(by['b'])
        repo.add_review('U1', 'a', 'MM', 'high')
        item2 = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'U1')
        by2 = {r['reviewer']: r for r in item2['reviews']}
        check(by2['b'] == b_before, f"other reviewer row untouched on re-drop: {by2['b']}")
        check(by2['a']['code'] == 'MM', f"re-dropper overwritten: {by2['a']}")
        check(len(item2['reviews']) == 2, f"still one row per reviewer: {item2['reviews']}")


def test_accept_item_default_basis_and_labeler():
    # accept_item with no basis/labeler defaults basis='consensus', labeler=''
    # and still stamps mode EXPERT + at (CONTRACTS §2,§6).
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})
        acc = repo.accept_item('U1', 'PA')   # no basis/labeler
        check(acc['basis'] == 'consensus', f"default basis: {acc}")
        check(acc['labeler'] == '', f"default labeler blank: {acc}")
        check(acc['mode'] == 'EXPERT' and acc['code'] == 'PA' and acc['at'], acc)
        # operator basis variant is honored too
        repo.add_item('U2', {'uid': 'U2'})
        acc2 = repo.accept_item('U2', 'SS', labeler='op', basis='operator')
        check(acc2['basis'] == 'operator' and acc2['labeler'] == 'op', acc2)


def test_library_reviews_round_trip():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(os.path.join(t, '_lib')); lib.ensure()
        # missing -> empty shell
        empty = lib.load_reviews('u1')
        check(empty == {'uid': 'u1', 'reviews': [], 'accepted': None}, empty)
        obj = {'uid': 'u1',
               'reviews': [{'reviewer': 'lauren', 'code': 'PA', 'confidence': 'high', 'at': 'T'}],
               'accepted': {'code': 'PA', 'mode': 'EXPERT', 'labeler': 'op', 'at': 'T', 'basis': 'consensus'}}
        lib.save_reviews('u1', obj)
        # fresh instance proves on-disk round-trip
        got = Library(os.path.join(t, '_lib')).load_reviews('u1')
        check(got['reviews'][0]['code'] == 'PA', got)
        check(got['accepted']['basis'] == 'consensus', got)
        # save tolerates a partial obj (defaults filled)
        lib.save_reviews('u2', {'reviews': [{'reviewer': 'b', 'code': 'SS'}]})
        g2 = lib.load_reviews('u2')
        check(g2['uid'] == 'u2' and g2['accepted'] is None, g2)


# ══ ADVERSARIAL ══════════════════════════════════════════════════════
def test_adv_blank_reviewer_rejected():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})
        check(repo.add_review('U1', '', 'PA') is None, 'empty reviewer rejected')
        check(repo.add_review('U1', '   ', 'PA') is None, 'whitespace reviewer rejected')
        item = repo.load_manifest()['items'][0]
        check(item['reviews'] == [], f"no review row written: {item}")
        check(repo.reviewer_names() == [], 'no reviewer tracked')


def test_adv_add_review_unknown_uid_is_noop():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        check(repo.add_review('GHOST', 'lauren', 'PA') is None, 'unknown uid -> None')
        check(repo.reviewer_names() == [], 'no reviewer tracked for ghost uid')


def test_adv_unsafe_uid_rejected_everywhere():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        check(repo.add_review('../evil', 'lauren', 'PA') is None, 'add_review unsafe')
        check(repo.accept_item('../evil', 'PA') is None, 'accept_item unsafe')
        lib = Library(os.path.join(t, '_lib')); lib.ensure()
        # load is tolerant (returns shell), save must refuse
        shell = lib.load_reviews('../evil')
        check(shell['reviews'] == [], 'load_reviews unsafe -> empty shell')
        raised = False
        try:
            lib.save_reviews('../evil', {'reviews': []})
        except UnsafeUidError:
            raised = True
        check(raised, 'save_reviews refuses unsafe uid')


def test_adv_accept_is_only_removal_and_idempotent_gone():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})
        repo.accept_item('U1', 'PA')
        check('U1' not in repo.pending_uids(), 'gone after accept')
        # second accept on an already-removed uid is a safe no-op (returns None)
        check(repo.accept_item('U1', 'PA') is None, 'accept of gone uid -> None')


def test_adv_re_add_preserves_tentative_reviews():
    # An item re-queued (refreshed imagery) must NOT lose tentative reviews.
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1', 'site': 'SCP'})
        repo.add_review('U1', 'lauren', 'PA', 'high')
        repo.add_item('U1', {'uid': 'U1', 'site': 'SCP', 'frame': 7})  # re-add
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'U1')
        check(len(item['reviews']) == 1 and item['reviews'][0]['code'] == 'PA',
              f"reviews carried forward on re-add: {item}")


def test_adv_pending_by_project_blank_project_bucket():
    # An older item missing project fields buckets under '' rather than crashing.
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('U1', {'uid': 'U1'})   # no project_id/name
        buckets = repo.pending_by_project()
        check(len(buckets) == 1 and buckets[0]['project_id'] == '', buckets)
        check(buckets[0]['project_name'] == '' and buckets[0]['count'] == 1, buckets)


def test_adv_pending_by_project_name_falls_back_to_id_when_blank():
    # An item with a project_id but no project_name labels its bucket with the
    # id (never an empty name), and items sharing one id collapse to one bucket
    # whose name is the first non-blank seen for that id.
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'pid_only'})           # no name
        repo.add_item('A2', {'uid': 'A2', 'project_id': 'pid_only', 'project_name': 'Named'})
        buckets = repo.pending_by_project()
        check(len(buckets) == 1, f"one bucket per project_id: {buckets}")
        b = buckets[0]
        check(b['project_id'] == 'pid_only' and b['count'] == 2, b)
        # first item had no name -> bucket name fell back to the id (never '')
        check(b['project_name'] == 'pid_only', f"name falls back to id: {b}")


def test_adv_already_published_item_missing_new_fields():
    # Simulate a manifest written by the OLD code (no reviews/accepted/reviewer_names).
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        legacy = {'generated_at': 'x', 'contacts': [], 'count': 1,
                  'items': [{'uid': 'U1', 'site': 'SCP', 'project_id': 'p1',
                             'project_name': 'P One'}]}
        repo._write_manifest_obj(legacy)   # no reviews/accepted/reviewer_names
        # add_review must tolerate the legacy item (no reviews key) and upsert.
        repo.add_review('U1', 'lauren', 'PA', 'high')
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'U1')
        check(item['reviews'][0]['reviewer'] == 'lauren', f"legacy item upserted: {item}")
        check(repo.reviewer_names() == ['lauren'], 'reviewer_names seeded on legacy manifest')
        # pending_by_project + accept also tolerate the legacy shape
        check(repo.pending_by_project()[0]['project_id'] == 'p1', 'bucket legacy item')
        check(repo.accept_item('U1', 'PA') is not None, 'accept legacy item')


def test_adv_identical_frame_names_across_projects_bucket_separately():
    # Same source frame filename in two projects must stay two distinct buckets/UIDs.
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('P1-x', {'uid': 'P1-x', 'source_image': 'F.jpeg',
                               'project_id': 'p1', 'project_name': 'One'})
        repo.add_item('P2-x', {'uid': 'P2-x', 'source_image': 'F.jpeg',
                               'project_id': 'p2', 'project_name': 'Two'})
        buckets = {b['project_id']: b['count'] for b in repo.pending_by_project()}
        check(buckets == {'p1': 1, 'p2': 1}, f"separate buckets: {buckets}")
        # accepting one leaves the other pending
        repo.accept_item('P1-x', 'PA')
        check(repo.pending_uids() == ['P2-x'], repo.pending_uids())


def test_adv_remove_project_unknown_and_blank_bucket():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'p1'})
        repo.add_item('N1', {'uid': 'N1'})   # legacy item, no project_id
        res = repo.remove_project('ghost')
        check(res == {'removed': 0, 'uids': []}, f"unknown project -> zero: {res}")
        check(len(repo.load_manifest()['items']) == 2, 'nothing dropped for unknown project')
        check(os.path.isdir(os.path.join(repo.items_dir, 'A1')), 'A1 assets intact')
        # '' matches ONLY the blank bucket (mirrors pending_by_project bucketing)
        res2 = repo.remove_project('')
        check(res2['removed'] == 1 and res2['uids'] == ['N1'], res2)
        check(repo.pending_uids() == ['A1'], repo.pending_uids())
        check(not os.path.isdir(os.path.join(repo.items_dir, 'N1')), 'N1 assets gone')


def test_adv_remove_project_then_re_export_re_adds_cleanly():
    with tempfile.TemporaryDirectory() as t:
        repo = _repo(t)
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'p1', 'project_name': 'One'})
        repo.add_review('A1', 'lauren', 'PA')
        check(repo.remove_project('p1')['removed'] == 1, 'p1 removed')
        check(repo.pending_uids() == [], 'p1 gone')
        # re-export (add_item again): fresh entry, no stale carried reviews
        repo.add_item('A1', {'uid': 'A1', 'project_id': 'p1', 'project_name': 'One'})
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'A1')
        check(item['reviews'] == [] and item['accepted'] is None,
              f"re-add starts clean after project delete: {item}")
        check(os.path.isdir(os.path.join(repo.items_dir, 'A1')), 'items dir recreated')
        check(repo.load_manifest()['count'] == 1, 'count recomputed on re-add')
        # reviewer_names history is deliberately preserved across the delete
        check(repo.reviewer_names() == ['lauren'], repo.reviewer_names())


def test_adv_unmapped_site_defaults_to_blank_site_full():
    with tempfile.TemporaryDirectory() as t:
        # ZZZ is not in the site table -> site_full empty, code preserved.
        st, rd, ld = _export_one(t, fn='TCRMP20201022_clip_ZZZ_T101.jpeg')
        uid = st['uids'][0]
        item = next(i for i in ReviewRepo(rd, remote_url='').load_manifest()['items']
                    if i['uid'] == uid)
        check(item.get('site') == 'ZZZ', f"site code kept: {item}")
        check(item.get('site_full') == '', f"unmapped site_full blank: {item}")


def test_adv_gh_token_fallback_chain():
    """GH_TOKEN env beats everything; hosts.yml parse is the last resort
    (snap gh cannot exec in capability-restricted process trees)."""
    import os as _os, tempfile as _tf
    from _reefreview import review_repo as _rr
    prev = _os.environ.get('GH_TOKEN')
    try:
        _os.environ['GH_TOKEN'] = 'tok_env_wins'
        assert _rr._gh_token() == 'tok_env_wins'
    finally:
        if prev is None:
            _os.environ.pop('GH_TOKEN', None)
        else:
            _os.environ['GH_TOKEN'] = prev
    with _tf.NamedTemporaryFile('w', suffix='.yml', delete=False) as f:
        f.write('github.com:\n    oauth_token: gho_hosts_yml\n')
        path = f.name
    try:
        assert _rr._token_from_hosts_yml(['/nope', path]) == 'gho_hosts_yml'
        assert _rr._token_from_hosts_yml(['/nope']) == ''
    finally:
        _os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    print(f"Running {len(tests)} review-schema tests...\n")
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
