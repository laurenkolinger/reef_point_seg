"""
Self-contained unit + integration tests for the expert-review round-trip.

No pytest dependency: run with the unified env python:
    env/bin/python scripts/_reefreview/tests/test_reefreview.py

Covers filename_parse, uid, mask_geom, codes, library, render_review,
review_repo, review_export, and the addExpertIDs importer. Uses tempdirs and
synthetic images/masks; no network (git push disabled).
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import traceback

import numpy as np
from PIL import Image

# scripts/ on path (parent of _reefreview) + importer src.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, 'TCRMPclip_addExpertIDs', 'src'))

from _reefreview import filename_parse, uid as uidmod, mask_geom, codes, render_review
from _reefreview.library import Library, UnsafeUidError
from _reefreview.review_repo import ReviewRepo
from _reefreview import review_repo as review_repo_mod
from _reefreview import review_export
import importer

# ── tiny test harness ───────────────────────────────────────────────
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


# ── synthetic helpers ───────────────────────────────────────────────
def _square_mask(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m

def _mask_dict(rle, bbox, polygon_px, sx, sy, species='REVIEW', review=True,
               status='pending'):
    d = {'id': 0, 'label': 'A', 'species': species, 'name': '', 'category': 'Review',
         'source_x': sx, 'source_y': sy, 'polygon_px': polygon_px,
         'polygon_norm': [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]], 'rle': rle,
         'bbox': bbox, 'area': 100, 'score': 0.9, 'status': status,
         'refinement_clicks': [], 'source_type': 'auto'}
    if review:
        d['review'] = True
    return d

def _make_project(root, fn, mask):
    """Create a project export dir with a raw image + a segmentations.json."""
    proj = os.path.join(root, 'inprocess', 'run_t_20990101_aaa', 'step5_segmentImages')
    segdir = os.path.join(proj, 'segmentations', '2020'); os.makedirs(segdir, exist_ok=True)
    raw = os.path.join(proj, 'raw'); os.makedirs(raw, exist_ok=True)
    img = os.path.join(raw, fn)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
    seg = {'image_path': 'raw/' + fn, 'image_path_abs': img, 'image_width': 200,
           'image_height': 200, 'masks': [mask], 'reviewed': False, 'exported': False}
    with open(os.path.join(segdir, 'segmentations.json'), 'w') as f:
        json.dump({fn: seg}, f)
    return proj, segdir, {fn: seg}


# ── filename_parse ──────────────────────────────────────────────────
def test_filename_standard():
    p = filename_parse.parse('TCRMP20201022_clip_SCP_T101.jpeg')
    check(p == {'year': 2020, 'date': '20201022', 'site': 'SCP', 'transect': 1, 'frame': 1}, p)

def test_filename_dialects_and_strip():
    check(filename_parse.parse('raw/TCRMP20140930_clip_BIT_T207.jpeg')['frame'] == 7, 'frame')
    check(filename_parse.parse('TCRMP20181101_clip_BIX_T101_pts.jpg')['site'] == 'BIX', 'pts strip')
    check(filename_parse.parse('TCRMP20140926_clip_BWR_T101_cpc.json')['site'] == 'BWR', 'cpc strip')
    check(filename_parse.parse('TCRMP_20250601_clip_FLC_T312.jpeg')['year'] == 2025, 'underscore dialect')
    check(filename_parse.parse('20250601TCRMP_clip_FLC_T312.jpeg')['site'] == 'FLC', 'datefirst dialect')
    check(filename_parse.parse('tcrmp20201022_clip_scp_t101.jpeg')['site'] == 'SCP', 'case-insensitive + upper')

def test_filename_multidigit_transect():
    p = filename_parse.parse('TCRMP20201022_clip_SCP_T1015.jpeg')  # transect 10, frame 15
    check(p['transect'] == 10 and p['frame'] == 15, p)

def test_filename_garbage_never_raises():
    p = filename_parse.parse('garbage.jpg')
    check(p == {'year': None, 'date': '', 'site': '', 'transect': None, 'frame': None}, p)
    check(filename_parse.parse('')['site'] == '', 'empty')
    check(filename_parse.parse(None)['site'] == '', 'None')


# ── uid ─────────────────────────────────────────────────────────────
def test_uid_determinism_and_rounding():
    a = uidmod.make_uid('TCRMP20201022_clip_SCP_T101.jpeg', 1234.6, 567.2)
    b = uidmod.make_uid('TCRMP20201022_clip_SCP_T101.jpeg', 1234.6, 567.2)
    check(a == b == 'SCP-20201022-T101-x1235y567', a)

def test_uid_fallback_for_unparseable():
    u = uidmod.make_uid('weird_name.jpg', 10, 20)
    check(u == 'weirdname-x10y20', u)

def test_uid_distinct_per_point():
    u1 = uidmod.make_uid('TCRMP20201022_clip_SCP_T101.jpeg', 10, 20)
    u2 = uidmod.make_uid('TCRMP20201022_clip_SCP_T101.jpeg', 11, 20)
    check(u1 != u2, 'distinct clicks -> distinct uid')


# ── mask_geom ───────────────────────────────────────────────────────
def test_rle_roundtrip():
    for m in [_square_mask(50, 60, 10, 30, 5, 25), np.ones((10, 10), bool),
              np.zeros((10, 10), bool)]:
        rle = mask_geom.rle_encode(m)
        back = mask_geom.rle_decode(rle, shape=tuple(rle['size']))
        check(np.array_equal(m, back), 'rle roundtrip')

def test_intersection_over_area():
    h = w = 100
    big = _square_mask(h, w, 10, 90, 10, 90)
    small = _square_mask(h, w, 20, 40, 20, 40)   # fully inside big
    check(abs(mask_geom.intersection_over_area(small, big) - 1.0) < 1e-9, 'small in big -> 1.0')
    disjoint = _square_mask(h, w, 95, 99, 95, 99)
    check(mask_geom.intersection_over_area(disjoint, big) == 0.0, 'disjoint -> 0')
    empty = np.zeros((h, w), bool)
    check(mask_geom.intersection_over_area(empty, big) == 0.0, 'empty new -> 0 (no div0)')

def test_decode_pair_size_mismatch():
    a = mask_geom.rle_encode(np.ones((10, 10), bool))
    b = mask_geom.rle_encode(np.ones((12, 12), bool))
    da, db = mask_geom.decode_pair(a, b)
    check(da is None and db is None, 'mismatched sizes -> (None,None)')

def test_iou_symmetry():
    a = _square_mask(50, 50, 0, 30, 0, 30)
    b = _square_mask(50, 50, 10, 40, 10, 40)
    check(abs(mask_geom.iou(a, b) - mask_geom.iou(b, a)) < 1e-9, 'iou symmetric')


# ── codes ───────────────────────────────────────────────────────────
def test_codes_load_and_grouping():
    d = codes.load_codes()
    check(len(d['codes']) >= 120, f"expected ~130 codes, got {len(d['codes'])}")
    grouped = sum(len(g['codes']) for g in d['groups'])
    check(grouped == len(d['codes']), f"every code grouped: {grouped} != {len(d['codes'])}")
    check(d['default_confidence'] == 'high', 'default conf high')
    check('high' in d['confidence'] and 'low' in d['confidence'], 'conf defs')
    check(d['idk']['code'] == 'IDK', 'idk sentinel')
    check(d['by_code'].get('PA', {}).get('group') == 'Coral', 'PA in Coral group')

def test_codes_dirty_categories_mapped():
    d = codes.load_codes()
    # DCA (category 'Dca') + ERSP (category 'Spo') must land in a real group, not crash.
    for c in ('DCA', 'ERSP'):
        if c in d['by_code']:
            check(d['by_code'][c]['group'] in [g['group'] for g in d['groups']], f"{c} grouped")


# ── library ─────────────────────────────────────────────────────────
def test_library_upsert_lookup_timestamps():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(t); lib.ensure()
        rec = lib.upsert({'uid': 'U1', 'code': 'REVIEW', 'mode': 'USER', 'source_image': 'f.jpeg'})
        created = rec['created_at']
        check(lib.lookup('U1')['code'] == 'REVIEW', 'lookup')
        rec2 = lib.upsert({'uid': 'U1', 'code': 'PA', 'mode': 'EXPERT'})
        check(rec2['code'] == 'PA' and rec2['mode'] == 'EXPERT', 'update fields')
        check(rec2['created_at'] == created, 'created_at preserved on update')
        check(rec2['source_image'] == 'f.jpeg', 'untouched field preserved')

def test_library_find_overlap_rules():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(t); lib.ensure()
        h = w = 100
        known = _square_mask(h, w, 10, 90, 10, 90)
        rle_known = mask_geom.rle_encode(known)
        # an EXPERT row on image f.jpeg
        lib.save_polygon('K1', {'rle': rle_known, 'source_image': 'TCRMP20201022_clip_SCP_T101.jpeg'})
        lib.upsert({'uid': 'K1', 'code': 'PA', 'mode': 'EXPERT',
                    'source_image': 'TCRMP20201022_clip_SCP_T101.jpeg'})
        new = _square_mask(h, w, 20, 40, 20, 40)  # inside known
        rle_new = mask_geom.rle_encode(new)
        m = lib.find_overlap('TCRMP20201022_clip_SCP_T101.jpeg', rle_new, thresh=0.5)
        check(m and m['code'] == 'PA', 'overlap match found')
        # different image -> no match
        check(lib.find_overlap('TCRMP20201022_clip_BIT_T101.jpeg', rle_new) is None, 'diff image no match')
        # exclude_uid skips self
        check(lib.find_overlap('TCRMP20201022_clip_SCP_T101.jpeg', rle_known, exclude_uid='K1') is None,
              'exclude_uid')

def test_library_find_overlap_requires_expert():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(t); lib.ensure()
        h = w = 100
        known = _square_mask(h, w, 10, 90, 10, 90)
        rle_known = mask_geom.rle_encode(known)
        lib.save_polygon('U1', {'rle': rle_known, 'source_image': 'f.jpeg'})
        # USER/REVIEW row must NOT be inheritable
        lib.upsert({'uid': 'U1', 'code': 'REVIEW', 'mode': 'USER', 'source_image': 'f.jpeg'})
        new = _square_mask(h, w, 20, 40, 20, 40)
        check(lib.find_overlap('f.jpeg', mask_geom.rle_encode(new)) is None,
              'USER/REVIEW row not inherited')


# ── render_review ───────────────────────────────────────────────────
def test_render_closeup_and_full():
    with tempfile.TemporaryDirectory() as t:
        img = os.path.join(t, 'TCRMP20201022_clip_SCP_T101.jpeg')
        Image.fromarray((np.random.rand(300, 300, 3) * 255).astype('uint8')).save(img)
        binary = _square_mask(300, 300, 80, 160, 90, 170)
        rle = mask_geom.rle_encode(binary)
        mask = _mask_dict(rle, [90, 80, 169, 159], [[90, 80, 169, 80, 169, 159, 90, 159]], 130, 120)
        crop = os.path.join(t, 'crop.jpg'); amask = os.path.join(t, 'mask.png'); full = os.path.join(t, 'full.jpg')
        info = render_review.render_closeup(img, mask, crop, amask, pad_px=20, max_edge=200)
        check(info is not None and os.path.exists(crop) and os.path.exists(amask), 'closeup written')
        check(render_review.render_full(img, mask, full, max_edge=250) is not None and os.path.exists(full), 'full written')

def test_render_missing_image_returns_none():
    with tempfile.TemporaryDirectory() as t:
        mask = _mask_dict(mask_geom.rle_encode(_square_mask(50, 50, 5, 15, 5, 15)),
                          [5, 5, 14, 14], [[5, 5, 14, 5, 14, 14, 5, 14]], 10, 10)
        check(render_review.render_closeup('/nope.jpg', mask, '/tmp/x.jpg', '/tmp/x.png') is None, 'missing img')


# ── review_repo ─────────────────────────────────────────────────────
def test_review_repo_lifecycle():
    with tempfile.TemporaryDirectory() as t:
        repo = ReviewRepo(os.path.join(t, 'REVIEW'), remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        check(os.path.exists(os.path.join(repo.root, '.nojekyll')), '.nojekyll')
        check(os.path.exists(os.path.join(repo.root, 'codes.json')), 'codes.json')
        check(os.path.exists(os.path.join(repo.root, 'index.html')), 'viewer copied')
        repo.set_contacts(['a@b.com'])
        check(repo.load_manifest()['contacts'] == ['a@b.com'], 'contacts set')
        repo.add_item('U1', {'site': 'SCP', 'uid': 'U1'})
        check(repo.pending_uids() == ['U1'], 'added')
        repo.add_item('U1', {'site': 'SCP', 'uid': 'U1'})  # idempotent (no dup)
        check(repo.pending_uids() == ['U1'], 'no dup on re-add')
        n = repo.remove_item('U1')
        check(n == 1 and repo.pending_uids() == [], 'removed')
        check(repo.commit_push('test', push=False) is False, 'commit-only returns False')


# ── review_export ───────────────────────────────────────────────────
def test_review_export_queue_then_autorelabel():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        binary = _square_mask(200, 200, 60, 140, 60, 140)
        rle = mask_geom.rle_encode(binary)
        mask = _mask_dict(rle, [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        proj, segdir, segmap = _make_project(t, fn, mask)
        review_dir = os.path.join(t, 'REVIEW'); lib_dir = os.path.join(t, '_lib')

        st = review_export.export_flagged_masks(
            segmap, [fn], export_dir=proj, review_dir=review_dir, repo_url='',
            library_dir=lib_dir, master_codes=None, contacts=['x@y.com'],
            featured_codes=['PA'], project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st['new'] == 1 and st['auto_relabeled'] == 0, st)
        u = st['uids'][0]
        lib = Library(lib_dir)
        check(lib.lookup(u)['mode'] == 'USER' and lib.lookup(u)['code'] == 'REVIEW', 'queued USER/REVIEW')
        check(os.path.exists(lib.image_path(u)) and os.path.exists(lib.mask_path(u)), 'image+mask sidecar stored')

        # Now flip the library row to EXPERT/PA and re-export an overlapping NEW mask -> auto-relabel
        lib.upsert({'uid': u, 'code': 'PA', 'mode': 'EXPERT'})
        binary2 = _square_mask(200, 200, 70, 130, 70, 130)
        mask2 = _mask_dict(mask_geom.rle_encode(binary2), [70, 70, 129, 129],
                           [[70, 70, 129, 70, 129, 129, 70, 129]], 99, 101)
        segmap2 = {fn: {'image_path_abs': segmap[fn]['image_path_abs'], 'image_width': 200,
                        'image_height': 200, 'masks': [mask2]}}
        st2 = review_export.export_flagged_masks(
            segmap2, [fn], export_dir=proj, review_dir=review_dir, repo_url='',
            library_dir=lib_dir, master_codes=None, contacts=['x@y.com'],
            featured_codes=['PA'], project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st2['auto_relabeled'] == 1 and st2['new'] == 0, f"auto-relabel: {st2}")
        check(mask2['species'] == 'PA' and mask2.get('review') is False, 'mask2 relabeled in place')

def test_review_export_rejected_skipped():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        binary = _square_mask(200, 200, 60, 140, 60, 140)
        mask = _mask_dict(mask_geom.rle_encode(binary), [60, 60, 139, 139],
                          [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100, status='rejected')
        proj, _, segmap = _make_project(t, fn, mask)
        st = review_export.export_flagged_masks(
            segmap, [fn], export_dir=proj, review_dir=os.path.join(t, 'REVIEW'), repo_url='',
            library_dir=os.path.join(t, '_lib'), master_codes=None, contacts=[], featured_codes=[],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st['new'] == 0 and st['skipped_rejected'] == 1, st)

def test_review_export_idempotent_reexport():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        mask = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                          [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        proj, _, segmap = _make_project(t, fn, mask)
        rd = os.path.join(t, 'REVIEW'); ld = os.path.join(t, '_lib')
        kw = dict(export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
                  contacts=[], featured_codes=['PA'], project_id='run_t', git_push=False, log_fn=lambda m: None)
        review_export.export_flagged_masks(segmap, [fn], **kw)
        review_export.export_flagged_masks(segmap, [fn], **kw)  # twice
        repo = ReviewRepo(rd, master_codes_path=None)
        check(len(repo.pending_uids()) == 1, 're-export does not duplicate the queue item')


# ── importer: ROLLING ingest (CONTRACTS §4,§5,§6) ───────────────────
# parse_csv_text now returns the 6-key row {uid, code, confidence, reviewer,
# labeler, project_id}; import_rows UPSERTs tentative reviews and NEVER sets
# expert_id / NEVER prunes; accept_uid is the only expert/prune path.

def test_csv_parse_variants():
    # New header (CONTRACTS §4): uid,code,confidence,reviewer,project_id. Every
    # row carries all six keys (reviewer + labeler + project_id added).
    r1 = importer.parse_csv_text('uid,code,confidence,reviewer,project_id\n'
                                 'A-1,PA,high,lauren,p1\n')
    check(r1 == [{'uid': 'A-1', 'code': 'PA', 'confidence': 'high',
                  'reviewer': 'lauren', 'labeler': '', 'project_id': 'p1'}], r1)
    # reviewer falls back to the legacy `labeler` column (reviewer wins when both).
    r1b = importer.parse_csv_text('uid,code,labeler\nA-9,SS,bob\n')
    check(r1b[0]['reviewer'] == 'bob' and r1b[0]['labeler'] == 'bob', r1b)
    r1c = importer.parse_csv_text('uid,code,reviewer,labeler\nA-9,SS,realname,legacy\n')
    check(r1c[0]['reviewer'] == 'realname', r1c)
    # Case-insensitive header; trailing human-readable columns ignored.
    r2 = importer.parse_csv_text('UID,Code,Reviewer,Project_Id,Site,Frame\n'
                                 'A-1,AA,kim,proj,SCP,3\n')
    check(r2[0]['code'] == 'AA' and r2[0]['reviewer'] == 'kim'
          and r2[0]['project_id'] == 'proj', r2)
    # Headerless positional: uid,code[,confidence,reviewer]; both reviewer+labeler
    # take the 4th column, project_id the 5th.
    r3 = importer.parse_csv_text('A-1,PA,high,dana\nA-2,AA,,sam,projX\n')
    check(len(r3) == 2 and r3[0]['reviewer'] == 'dana'
          and r3[1]['uid'] == 'A-2' and r3[1]['project_id'] == 'projX', r3)
    # Blank/space rows dropped; missing uid dropped.
    r4 = importer.parse_csv_text('\n\nuid,code,reviewer\n , , \nA-1,PA,lauren\n')
    check(len(r4) == 1 and r4[0]['uid'] == 'A-1' and r4[0]['reviewer'] == 'lauren', r4)

def _seg_mask(segdir, fn, idx=0):
    """First/idx mask of a project's on-disk segmentations.json (helper)."""
    seg = json.load(open(os.path.join(segdir, 'segmentations.json')))
    return seg[fn]['masks'][idx]


def test_importer_full_roundtrip():
    """ROLLING import (CONTRACTS §5): a reviewer's CSV row UPSERTs a tentative
    review onto the manifest item AND the project mask, but NEVER sets expert_id
    and NEVER prunes the repo. The item stays pending showing the tentative ID."""
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        mask = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                          [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        proj, segdir, segmap = _make_project(t, fn, mask)
        rd = os.path.join(t, 'REVIEW'); ld = os.path.join(t, '_lib')
        # export to populate library + queue
        st = review_export.export_flagged_masks(segmap, [fn], export_dir=proj, review_dir=rd,
            repo_url='', library_dir=ld, master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        u = st['uids'][0]
        # import a reviewer's CSV row (project_id unresolvable -> open export_dir fallback)
        ist = importer.import_rows([{'uid': u, 'code': 'PA', 'confidence': 'high',
                                     'reviewer': 'Dr. Expert'}],
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            overlap_thresh=0.5, git_push=False, log_fn=lambda m: None)
        # Rolling stats: one tentative review added, nothing unrouted.
        check(ist['rows'] == 1 and ist['reviews_added'] == 1, ist)
        check(ist['unrouted'] == [], f"nothing unrouted: {ist}")
        check(u in ist['uids'], ist)
        # NEVER set expert_id on the library record (still USER/REVIEW from export).
        lib = Library(ld)
        rec = lib.lookup(u)
        check(rec['mode'] == 'USER' and rec['code'] == 'REVIEW',
              f"import must not promote library record to EXPERT: {rec}")
        # The mask carries a tentative review[] keyed by reviewer; expert_id absent;
        # still flagged for review.
        m = _seg_mask(segdir, fn)
        check('expert_id' not in m, f"import must NOT set expert_id: {m}")
        check(m.get('review') is True and m.get('species') == 'REVIEW',
              f"mask stays pending: {m}")
        revs = m.get('reviews') or []
        check(len(revs) == 1 and revs[0]['reviewer'] == 'Dr. Expert'
              and revs[0]['code'] == 'PA' and revs[0]['confidence'] == 'high',
              f"tentative review upserted on mask: {revs}")
        # Per-uid library detail mirror also carries the tentative review.
        detail = lib.load_reviews(u)
        check(any(r['reviewer'] == 'Dr. Expert' and r['code'] == 'PA'
                  for r in detail['reviews']), f"library reviews mirror: {detail}")
        check(detail['accepted'] is None, f"not accepted by import: {detail}")
        # Repo: item still PENDING with a tentative review; nothing pruned.
        repo = ReviewRepo(rd, master_codes_path=None)
        check(repo.pending_uids() == [u], f"import must NOT prune: {repo.pending_uids()}")
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == u)
        check(item['reviews'][0]['reviewer'] == 'Dr. Expert' and item['accepted'] is None,
              f"manifest item tentative, not accepted: {item}")
        check(ist['pending_total'] == 1, f"pending_total reported: {ist}")
        # Re-import same reviewer -> UPSERT (one row per reviewer, not duplicated).
        ist2 = importer.import_rows([{'uid': u, 'code': 'PA', 'reviewer': 'Dr. Expert'}],
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            git_push=False, log_fn=lambda m: None)
        check(ist2['reviews_added'] == 1, 'idempotent re-import still upserts')
        m2 = _seg_mask(segdir, fn)
        check(len(m2.get('reviews') or []) == 1, f"re-import does not duplicate reviewer row: {m2}")
        check(ReviewRepo(rd, master_codes_path=None).pending_uids() == [u],
              're-import still does not prune')

def test_importer_unknown_code_and_new_uid():
    """Unknown master code is counted but the tentative review is still kept
    (CONTRACTS §5: never silent, never drop). A row with a project_id that does
    not resolve to any export dir AND no open export_dir fallback is REPORTED in
    `unrouted` — never silently dropped."""
    with tempfile.TemporaryDirectory() as t:
        proj = os.path.join(t, 'inprocess', 'run_x', 'step5_segmentImages')
        os.makedirs(os.path.join(proj, 'segmentations', '2020'), exist_ok=True)
        with open(os.path.join(proj, 'segmentations', '2020', 'segmentations.json'), 'w') as f:
            json.dump({}, f)
        ld = os.path.join(t, '_lib'); rd = os.path.join(t, 'REVIEW')
        pr = os.path.join(t, 'inprocess')   # projects_root with NO real project.json
        repo = ReviewRepo(rd, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        # Queue a pending item so the tentative review has something to land on.
        repo.add_item('GHOST-1', {'uid': 'GHOST-1', 'site': 'SCP'})
        # Unknown code, routed to the open export_dir (no project_id) -> kept tentative.
        ist = importer.import_rows([{'uid': 'GHOST-1', 'code': 'ZZZNOTACODE',
                                     'reviewer': 'kim'}],
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            projects_root=pr, git_push=False, log_fn=lambda m: None)
        check(ist['unknown_code'] == 1, f"unknown code counted: {ist}")
        check(ist['reviews_added'] == 1, f"row still kept as tentative: {ist}")
        check(ist['unrouted'] == [], f"open export_dir routed it: {ist}")
        # NEVER promote to EXPERT — import is rolling only.
        rec = Library(ld).lookup('GHOST-1')
        check(rec is None or rec.get('mode') != 'EXPERT',
              f"unknown-code import must not store EXPERT: {rec}")
        # Manifest item still pending with the tentative review.
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'GHOST-1')
        check(item['reviews'][0]['code'] == 'ZZZNOTACODE' and item['accepted'] is None, item)

        # Now a row whose project_id does not resolve AND no open export_dir
        # (export_dir='') -> the UID is REPORTED unrouted, never silent.
        repo.add_item('LOST-1', {'uid': 'LOST-1', 'project_id': 'no_such_project'})
        ist2 = importer.import_rows([{'uid': 'LOST-1', 'code': 'PA', 'reviewer': 'kim',
                                      'project_id': 'no_such_project'}],
            export_dir='', review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            projects_root=pr, git_push=False, log_fn=lambda m: None)
        check('LOST-1' in ist2['unrouted'], f"unroutable UID reported: {ist2}")
        check(ist2['reviews_added'] == 0, f"nothing added for the unrouted row: {ist2}")

def test_importer_overlap_cascade():
    """The per-export_dir overlap cascade (CONTRACTS §5) adds a TENTATIVE review
    (attributed to the synthetic 'overlap' reviewer) to any still-pending mask
    that overlaps a now-EXPERT library mask — it does NOT set expert_id and does
    NOT accept. A prior EXPERT library mask is what the cascade inherits from."""
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        m1 = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                        [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        proj, segdir, segmap = _make_project(t, fn, m1)
        rd = os.path.join(t, 'REVIEW'); ld = os.path.join(t, '_lib')
        st = review_export.export_flagged_masks(segmap, [fn], export_dir=proj, review_dir=rd,
            repo_url='', library_dir=ld, master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        u1 = st['uids'][0]
        lib = Library(ld)
        # Promote u1's library row to EXPERT/PA (as acceptance would) so it is an
        # inheritable source for the overlap cascade.
        lib.upsert({'uid': u1, 'code': 'PA', 'mode': 'EXPERT',
                    'source_image': fn})
        # Add a SECOND, still-pending review mask on the same frame overlapping m1.
        seg = json.load(open(os.path.join(segdir, 'segmentations.json')))
        m2 = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 70, 130, 70, 130)),
                        [70, 70, 129, 129], [[70, 70, 129, 70, 129, 129, 70, 129]], 99, 101)
        seg[fn]['masks'].append(m2)
        with open(os.path.join(segdir, 'segmentations.json'), 'w') as f:
            json.dump(seg, f)
        # Importing a reviewer row for u1 routes to this export_dir, then the
        # cascade runs there -> m2 inherits PA as a TENTATIVE review (not expert).
        ist = importer.import_rows([{'uid': u1, 'code': 'PA', 'reviewer': 'lauren'}],
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            overlap_thresh=0.5, git_push=False, log_fn=lambda m: None)
        check(ist['auto_tentative'] >= 1, f"cascade added tentative IDs: {ist}")
        m2_after = _seg_mask(segdir, fn, idx=1)
        # m2 stays REVIEW with NO expert_id, but gains an 'overlap' tentative review.
        check('expert_id' not in m2_after, f"overlap must NOT set expert_id: {m2_after}")
        check(m2_after.get('species') == 'REVIEW' and m2_after.get('review') is True,
              f"overlapped mask stays pending: {m2_after}")
        ov = [r for r in (m2_after.get('reviews') or []) if r['reviewer'] == 'overlap']
        check(len(ov) == 1 and ov[0]['code'] == 'PA',
              f"overlap reviewer carries inherited PA tentatively: {m2_after.get('reviews')}")
        # The overlapped mask's uid is now pending in the repo with an 'overlap'
        # review — still NOT accepted/pruned.
        repo = ReviewRepo(rd, master_codes_path=None)
        check(set(repo.pending_uids()) >= {u1}, f"nothing pruned by overlap: {repo.pending_uids()}")


# ── hardening from the adversarial review ───────────────────────────
def test_uid_safety_predicates():
    for u in ['SCP-20201022-T101-x100y100', 'A1', 'a.b_c+d-e']:
        check(uidmod.is_safe_uid(u), f"should be safe: {u}")
    for u in ['../etc/passwd', 'a/b', 'a\\b', '..', '.', '', None, 'a' * 200, 'a b']:
        check(not uidmod.is_safe_uid(u), f"should be unsafe: {u!r}")
    s = uidmod.sanitize_uid('../../etc/passwd')
    check(s and '/' not in s, 'sanitize strips separators')
    check(uidmod.sanitize_uid('') == '', 'sanitize empty -> empty')


def test_library_rejects_unsafe_uid_paths():
    with tempfile.TemporaryDirectory() as t:
        lib = Library(t); lib.ensure()
        for bad in ['../escape', 'a/b', '..']:
            try:
                lib.image_path(bad); raise AssertionError(f"image_path allowed unsafe {bad!r}")
            except UnsafeUidError:
                pass
            try:
                lib.save_polygon(bad, {'rle': None}); raise AssertionError("save_polygon allowed unsafe")
            except UnsafeUidError:
                pass
        check(lib.load_polygon('../escape') is None, 'load_polygon unsafe -> None')


def test_review_repo_rejects_unsafe_uid():
    with tempfile.TemporaryDirectory() as t:
        repo = ReviewRepo(os.path.join(t, 'R'), remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        check(repo.add_item('../evil', {'uid': '../evil'}) is None, 'add_item refuses unsafe uid')
        check(repo.remove_item('../evil') == 0, 'remove_item refuses unsafe uid')


def test_review_repo_scrub_remote_no_token():
    import subprocess
    with tempfile.TemporaryDirectory() as t:
        url = 'https://github.com/laurenkolinger/reefpointseg-review.git'
        repo = ReviewRepo(os.path.join(t, 'R'), remote_url=url, master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        cur = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=repo.root,
                             capture_output=True, text=True).stdout.strip()
        check(cur == url and 'x-access-token' not in cur, f"clean origin url, got {cur}")
        repo.commit_push('t', push=False)
        cur2 = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=repo.root,
                              capture_output=True, text=True).stdout.strip()
        check('x-access-token' not in cur2, 'no token persisted after commit_push')


def test_review_repo_local_commit_never_creates_or_rewrites_origin():
    """Hammering-pass regression (2026-07-09): commit_push ran _scrub_remote
    even for a local-only commit (push=False), so a ReviewRepo constructed with
    the production remote_url over a caller-supplied directory branded that
    directory's git origin with the production URL without ever pushing. Origin
    maintenance belongs to the push path (and ensure_repo's seeding of a fresh
    tree) only."""
    import subprocess
    with tempfile.TemporaryDirectory() as t:
        url = 'https://github.com/prod-owner/prod-pages.git'
        # A pre-existing repo with NO origin (a scratch/export tree): a
        # local-only commit through a production-configured ReviewRepo must not
        # attach one.
        seed = ReviewRepo(os.path.join(t, 'R'), remote_url='',
                          master_codes_path=None, log_fn=lambda m: None)
        seed.ensure_repo()
        repo = ReviewRepo(seed.root, remote_url=url, master_codes_path=None,
                          log_fn=lambda m: None)
        repo.commit_push('local only', push=False)
        r = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=repo.root,
                           capture_output=True, text=True)
        check(r.returncode != 0, f"push=False must not create origin: {r.stdout.strip()}")
        # A repo whose origin points somewhere ELSE keeps it on a local commit.
        other = 'https://example.com/somewhere/else.git'
        subprocess.run(['git', 'remote', 'add', 'origin', other], cwd=repo.root,
                       capture_output=True, text=True)
        repo.commit_push('local only again', push=False)
        cur = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=repo.root,
                             capture_output=True, text=True).stdout.strip()
        check(cur == other, f"push=False must not rewrite an existing origin: {cur}")


def test_review_repo_gitignores_lock():
    with tempfile.TemporaryDirectory() as t:
        repo = ReviewRepo(os.path.join(t, 'R'), remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        check('.reefreview.lock' in open(os.path.join(repo.root, '.gitignore')).read(),
              '.gitignore covers the lock file')


def test_export_uid_collision_same_pixel():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        proj = os.path.join(t, 'inprocess', 'run_t', 'step5_segmentImages')
        os.makedirs(os.path.join(proj, 'raw'), exist_ok=True)
        Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(
            os.path.join(proj, 'raw', fn))
        def mk(sx, sy, y0, y1, x0, x1):
            b = _square_mask(200, 200, y0, y1, x0, x1)
            return _mask_dict(mask_geom.rle_encode(b), [x0, y0, x1 - 1, y1 - 1],
                              [[x0, y0, x1 - 1, y0, x1 - 1, y1 - 1, x0, y1 - 1]], sx, sy, species='REVIEW')
        m1 = mk(100.2, 100.4, 60, 140, 60, 140)
        m2 = mk(100.4, 100.1, 20, 50, 20, 50)   # same rounded pixel (100,100)
        seg = {'image_path_abs': os.path.join(proj, 'raw', fn), 'image_width': 200,
               'image_height': 200, 'masks': [m1, m2]}
        st = review_export.export_flagged_masks(
            {fn: seg}, [fn], export_dir=proj, review_dir=os.path.join(t, 'R'), repo_url='',
            library_dir=os.path.join(t, '_lib'), master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st['new'] == 2 and len(set(st['uids'])) == 2, f"distinct uids: {st['uids']}")


def test_export_missing_image_recorded_not_queued():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        mask = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                          [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        seg = {'image_path_abs': '/does/not/exist.jpeg', 'image_width': 200,
               'image_height': 200, 'masks': [mask]}
        st = review_export.export_flagged_masks(
            {fn: seg}, [fn], export_dir=os.path.join(t, 'p'), review_dir=os.path.join(t, 'R'),
            repo_url='', library_dir=os.path.join(t, '_lib'), master_codes=None, contacts=[],
            featured_codes=['PA'], project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st['skipped_missing_image'] == 1 and st['new'] == 0, st)
        check(len(Library(os.path.join(t, '_lib')).load()) == 1, 'geometry row recorded')
        check(ReviewRepo(os.path.join(t, 'R'), master_codes_path=None).pending_uids() == [],
              'missing-image item not queued')


def test_export_skips_non_pending_status():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        for status in ('rejected', 'merged_away', 'deleted'):
            mask = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                              [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100,
                              status=status)
            seg = {'image_path_abs': '/x.jpeg', 'image_width': 200, 'image_height': 200, 'masks': [mask]}
            st = review_export.export_flagged_masks(
                {fn: seg}, [fn], export_dir=os.path.join(t, 'p'), review_dir=os.path.join(t, 'R' + status),
                repo_url='', library_dir=os.path.join(t, '_lib' + status), master_codes=None,
                contacts=[], featured_codes=[], project_id='r', git_push=False, log_fn=lambda m: None)
            check(st['skipped_rejected'] == 1 and st['new'] == 0, (status, st))


def test_importer_default_labeler_and_unsafe_uid():
    """An unsafe UID from the CSV is REPORTED (stats.unsafe_uid) and skipped — no
    path traversal. A safe row with no per-row reviewer falls back to
    default_reviewer/default_labeler and is UPSERTed as a tentative review (still
    no expert_id, still no prune)."""
    with tempfile.TemporaryDirectory() as t:
        proj = os.path.join(t, 'inprocess', 'run_x', 'step5_segmentImages')
        os.makedirs(os.path.join(proj, 'segmentations', '2020'), exist_ok=True)
        with open(os.path.join(proj, 'segmentations', '2020', 'segmentations.json'), 'w') as f:
            f.write('{}')
        ld = os.path.join(t, '_lib'); rd = os.path.join(t, 'R')
        pr = os.path.join(t, 'inprocess')
        repo = ReviewRepo(rd, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        repo.add_item('GOOD-1', {'uid': 'GOOD-1', 'site': 'SCP'})
        # CSV has no reviewer column -> reviewer falls back to default_reviewer.
        rows = importer.parse_csv_text('uid,code\n../../etc/evil,PA\nGOOD-1,AA\n')
        ist = importer.import_rows(rows, export_dir=proj, review_dir=rd, repo_url='',
            library_dir=ld, master_codes=None, projects_root=pr, git_push=False,
            default_reviewer='Dr. X', default_labeler='Dr. X', log_fn=lambda m: None)
        check(ist.get('unsafe_uid', 0) == 1, f"unsafe uid counted: {ist}")
        check(not os.path.exists(os.path.join(t, 'etc', 'evil.json')), 'no traversal escape')
        check(ist['rows'] == 1 and ist['reviews_added'] == 1,
              f"only the one safe row processed: {ist}")
        # The tentative review is attributed to the default reviewer.
        item = next(i for i in repo.load_manifest()['items'] if i['uid'] == 'GOOD-1')
        check(item['reviews'][0]['reviewer'] == 'Dr. X'
              and item['reviews'][0]['code'] == 'AA', f"default reviewer used: {item}")
        check(item['accepted'] is None, 'import never accepts')
        check(repo.pending_uids() == ['GOOD-1'], 'import never prunes')


# ── regression tests for the SECOND adversarial pass ────────────────
def test_uid_content_disc_stable_and_distinct():
    # Two masks, same source pixel, different geometry -> distinct, stable discs.
    m_a = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                     [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
    m_b = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 20, 50, 20, 50)),
                     [20, 20, 49, 49], [[20, 20, 49, 20, 49, 49, 20, 49]], 100, 100)
    fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
    ua, ub = uidmod.mask_uid(fn, m_a), uidmod.mask_uid(fn, m_b)
    check(ua != ub, f"distinct geometry -> distinct uid: {ua} {ub}")
    check(ua == uidmod.mask_uid(fn, m_a), "mask_uid stable across calls")
    check(uidmod.content_disc(m_a) and uidmod.content_disc(m_a) == uidmod.content_disc(m_a),
          "content_disc deterministic")


def test_export_persists_review_uid_and_importer_matches_it():
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        proj = os.path.join(t, 'inprocess', 'run_t', 'step5_segmentImages')
        segdir = os.path.join(proj, 'segmentations', '2020'); os.makedirs(segdir, exist_ok=True)
        os.makedirs(os.path.join(proj, 'raw'), exist_ok=True)
        Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(
            os.path.join(proj, 'raw', fn))
        m1 = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                        [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        m2 = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 20, 50, 20, 50)),
                        [20, 20, 49, 49], [[20, 20, 49, 20, 49, 49, 20, 49]], 100, 100)  # same pixel
        seg = {'image_path_abs': os.path.join(proj, 'raw', fn), 'image_width': 200,
               'image_height': 200, 'masks': [m1, m2]}
        json.dump({fn: seg}, open(os.path.join(segdir, 'segmentations.json'), 'w'))
        ld = os.path.join(t, '_lib'); rd = os.path.join(t, 'R')
        st = review_export.export_flagged_masks({fn: seg}, [fn], export_dir=proj, review_dir=rd,
            repo_url='', library_dir=ld, master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(st['new'] == 2 and len(set(st['uids'])) == 2, f"distinct uids: {st['uids']}")
        check(m1.get('review_uid') and m2.get('review_uid') and m1['review_uid'] != m2['review_uid'],
              'review_uid persisted + distinct')
        # re-export is dedupe-stable (same uids, no extra rows)
        st2 = review_export.export_flagged_masks({fn: seg}, [fn], export_dir=proj, review_dir=rd,
            repo_url='', library_dir=ld, master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        check(set(st2['uids']) == set(st['uids']), 're-export uids stable')
        check(len(Library(ld).load()) == 2, 're-export does not grow the library')
        # ROLLING import matches the persisted review_uid and adds a TENTATIVE
        # review to EXACTLY that one mask — the sibling mask (same source pixel,
        # different geometry/uid) is untouched, and neither mask gets expert_id.
        import importer
        json.dump({fn: seg}, open(os.path.join(segdir, 'segmentations.json'), 'w'))
        ist = importer.import_rows(
            importer.parse_csv_text(f"uid,code,reviewer\n{m1['review_uid']},PA,lauren\n"),
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            git_push=False, log_fn=lambda m: None)
        check(ist['reviews_added'] == 1 and ist['unrouted'] == [], ist)
        out = json.load(open(os.path.join(segdir, 'segmentations.json')))
        by_uid = {mm['review_uid']: mm for mm in out[fn]['masks']}
        target = by_uid[m1['review_uid']]
        sibling = by_uid[m2['review_uid']]
        check((target.get('reviews') or [])[0]['reviewer'] == 'lauren'
              and target['reviews'][0]['code'] == 'PA',
              f"exactly the matched mask got the tentative review: {target.get('reviews')}")
        check('expert_id' not in target, 'matched mask still has no expert_id')
        check(not (sibling.get('reviews') or []),
              f"sibling mask (distinct uid) untouched: {sibling.get('reviews')}")
        # Both masks still REVIEW species (no relabel by import).
        check(all(mm.get('species') == 'REVIEW' for mm in out[fn]['masks']),
              'import never relabels species')


def test_relabel_hits_all_segmentation_files():
    # Same frame present in TWO project segmentation files within ONE export_dir
    # -> the rolling tentative review lands on the matching mask in BOTH files
    # (and never sets expert_id / never relabels species).
    with tempfile.TemporaryDirectory() as t:
        import importer
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        proj = os.path.join(t, 'inprocess', 'run_t', 'step5_segmentImages')
        os.makedirs(os.path.join(proj, 'raw'), exist_ok=True)
        Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(
            os.path.join(proj, 'raw', fn))
        m = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                       [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        # Export once (puts a library row + persists review_uid on the mask).
        seg = {'image_path_abs': os.path.join(proj, 'raw', fn), 'image_width': 200,
               'image_height': 200, 'masks': [m]}
        ld = os.path.join(t, '_lib'); rd = os.path.join(t, 'R')
        st = review_export.export_flagged_masks({fn: seg}, [fn], export_dir=proj, review_dir=rd,
            repo_url='', library_dir=ld, master_codes=None, contacts=[], featured_codes=['PA'],
            project_id='run_t', git_push=False, log_fn=lambda m: None)
        uid = st['uids'][0]
        # Now write the SAME frame (with the persisted review_uid) into TWO seg files.
        for yr in ('2020', '2021'):
            d = os.path.join(proj, 'segmentations', yr); os.makedirs(d, exist_ok=True)
            json.dump({fn: {'image_width': 200, 'image_height': 200, 'masks': [dict(m)]}},
                      open(os.path.join(d, 'segmentations.json'), 'w'))
        ist = importer.import_rows(
            importer.parse_csv_text(f"uid,code,reviewer\n{uid},PA,lauren\n"),
            export_dir=proj, review_dir=rd, repo_url='', library_dir=ld, master_codes=None,
            git_push=False, log_fn=lambda m: None)
        check(ist['reviews_added'] == 1 and ist['unrouted'] == [], 'one CSV row routed')
        for yr in ('2020', '2021'):
            out = json.load(open(os.path.join(proj, 'segmentations', yr, 'segmentations.json')))
            mm = out[fn]['masks'][0]
            check((mm.get('reviews') or [])[0]['reviewer'] == 'lauren'
                  and mm['reviews'][0]['code'] == 'PA', f"{yr} file got tentative review")
            check('expert_id' not in mm and mm.get('species') == 'REVIEW',
                  f"{yr} file: no expert_id, no relabel")


# ── importer: project_id ROUTING (CONTRACTS §5) ─────────────────────
def _routable_project(root, fn, mask, project_id, projects_root_name='inprocess',
                      dir_name=None):
    """A project that resolve_step_dir(project_id) CAN find: writes a real
    project.json carrying `id`=project_id under <root>/<projects_root_name>/<dir>.
    Returns (projects_root, export_dir, segdir)."""
    pr = os.path.join(root, projects_root_name)
    pdir = os.path.join(pr, dir_name or project_id)
    export_dir = os.path.join(pdir, 'step5_segmentImages')
    segdir = os.path.join(export_dir, 'segmentations', '2020')
    raw = os.path.join(export_dir, 'raw')
    os.makedirs(segdir, exist_ok=True); os.makedirs(raw, exist_ok=True)
    with open(os.path.join(pdir, 'project.json'), 'w') as f:
        json.dump({'id': project_id, 'name': project_id}, f)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(
        os.path.join(raw, fn))
    json.dump({fn: {'image_path_abs': os.path.join(raw, fn), 'image_width': 200,
                    'image_height': 200, 'masks': [mask]}},
              open(os.path.join(segdir, 'segmentations.json'), 'w'))
    return pr, export_dir, segdir


def test_importer_routes_rows_by_project_id():
    """import_rows GROUPS rows by project_id and routes each group to its own
    resolved export_dir; an unknown project_id with no open export_dir fallback
    is REPORTED (CONTRACTS §5). The per-project stat block reflects routed vs
    unrouted counts."""
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        m = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                       [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        pr, export_a, segdir_a = _routable_project(t, fn, m, 'alpha_20200101_aaa')
        rd = os.path.join(t, 'R'); ld = os.path.join(t, '_lib')
        repo = ReviewRepo(rd, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        # The persisted review_uid is what the importer matches in alpha's seg file.
        m_a = json.load(open(os.path.join(segdir_a, 'segmentations.json')))[fn]['masks'][0]
        from _reefreview.uid import mask_uid
        uid_a = m_a.get('review_uid') or mask_uid(fn, m_a, filename_parse.parse(fn))
        repo.add_item(uid_a, {'uid': uid_a, 'project_id': 'alpha_20200101_aaa',
                              'source_image': fn})
        repo.add_item('GHOST', {'uid': 'GHOST', 'project_id': 'no_such'})
        rows = [{'uid': uid_a, 'code': 'PA', 'reviewer': 'lauren',
                 'project_id': 'alpha_20200101_aaa'},
                {'uid': 'GHOST', 'code': 'SS', 'reviewer': 'bob',
                 'project_id': 'no_such'}]
        ist = importer.import_rows(rows, export_dir='', review_dir=rd, repo_url='',
            library_dir=ld, master_codes=None, projects_root=pr, git_push=False,
            log_fn=lambda m: None)
        # alpha row routed + tentative on alpha's mask; ghost row unrouted+reported.
        check(ist['reviews_added'] == 1, f"only the routable row added: {ist}")
        check('GHOST' in ist['unrouted'] and uid_a not in ist['unrouted'],
              f"unknown project_id reported, known one not: {ist}")
        seg_a = json.load(open(os.path.join(segdir_a, 'segmentations.json')))
        rv = seg_a[fn]['masks'][0].get('reviews') or []
        check(rv and rv[0]['reviewer'] == 'lauren',
              f"routed to alpha's resolved export_dir: {rv}")
        # Per-project stats present for both groups.
        pids = {p['project_id']: p for p in ist['projects']}
        check(pids['alpha_20200101_aaa']['routed'] == 1, f"alpha routed: {pids}")
        check(pids['no_such']['unrouted'] == 1, f"no_such unrouted: {pids}")


# ── ACCEPT path: the ONLY expert_id + prune path (CONTRACTS §5,§6) ───
def test_accept_uid_sets_expert_id_removes_and_prunes():
    """accept_uid is the consensus/accept path: it sets expert_id (mode EXPERT)
    on the routed project's step-5 mask, removes the UID from the review repo,
    prunes, and stamps the library record EXPERT. import did none of this."""
    with tempfile.TemporaryDirectory() as t:
        fn = 'TCRMP20201022_clip_SCP_T101.jpeg'
        m = _mask_dict(mask_geom.rle_encode(_square_mask(200, 200, 60, 140, 60, 140)),
                       [60, 60, 139, 139], [[60, 60, 139, 60, 139, 139, 60, 139]], 100, 100)
        pid = 'alpha_20200101_aaa'
        pr, export_dir, segdir = _routable_project(t, fn, m, pid)
        rd = os.path.join(t, 'R'); ld = os.path.join(t, '_lib')
        # Persist a review_uid on the mask so accept can match it.
        from _reefreview.uid import mask_uid
        seg = json.load(open(os.path.join(segdir, 'segmentations.json')))
        target = seg[fn]['masks'][0]
        uid = mask_uid(fn, target, filename_parse.parse(fn))
        target['review_uid'] = uid
        json.dump(seg, open(os.path.join(segdir, 'segmentations.json'), 'w'))
        # Queue the item with its project_id (accept routes off this).
        repo = ReviewRepo(rd, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        repo.add_item(uid, {'uid': uid, 'project_id': pid, 'source_image': fn})
        # Two reviewers already agreed PA (tentative), simulating consensus input.
        repo.add_review(uid, 'lauren', 'PA', 'high')
        repo.add_review(uid, 'bob', 'PA', 'high')
        check(repo.pending_uids() == [uid], 'pending before accept')

        res = importer.accept_uid(uid, 'PA', review_dir=rd, repo_url='', library_dir=ld,
            master_codes=None, projects_root=pr, labeler='operator', basis='consensus',
            git_push=False, log_fn=lambda m: None)
        check(res['ok'] and res['relabeled_seg'] is True, f"accept relabeled seg: {res}")
        check(res['accepted']['mode'] == 'EXPERT' and res['accepted']['basis'] == 'consensus', res)
        # Segmentation mask now carries the FINAL expert_id (mode EXPERT).
        out = json.load(open(os.path.join(segdir, 'segmentations.json')))
        mm = out[fn]['masks'][0]
        check(mm.get('species') == 'PA' and mm.get('review') is False, f"mask relabeled: {mm}")
        check(mm.get('expert_id', {}).get('mode') == 'EXPERT'
              and mm['expert_id']['labeler'] == 'operator', f"expert_id stamped: {mm}")
        # Library record promoted to EXPERT.
        rec = Library(ld).lookup(uid)
        check(rec['mode'] == 'EXPERT' and rec['code'] == 'PA', f"library EXPERT: {rec}")
        # Accept is the ONLY removal -> the repo is pruned.
        check(ReviewRepo(rd, master_codes_path=None).pending_uids() == [],
              'accept prunes the review repo')
        # Second accept on the now-gone uid is a safe no-op error (not pending).
        res2 = importer.accept_uid(uid, 'PA', review_dir=rd, repo_url='', library_dir=ld,
            master_codes=None, projects_root=pr, git_push=False, log_fn=lambda m: None)
        check(res2['ok'] is False, f"re-accept of pruned uid -> not ok: {res2}")


def test_accept_uid_rejects_unsafe_uid():
    with tempfile.TemporaryDirectory() as t:
        rd = os.path.join(t, 'R'); ld = os.path.join(t, '_lib')
        res = importer.accept_uid('../evil', 'PA', review_dir=rd, repo_url='',
            library_dir=ld, master_codes=None, git_push=False, log_fn=lambda m: None)
        check(res['ok'] is False and 'unsafe' in res['error'], res)
        check(not os.path.exists(os.path.join(t, 'evil')), 'no traversal escape')


# ── consensus classifier (CONTRACTS §6) ─────────────────────────────
def test_classify_reviews_consensus_conflict_single_none():
    cl = importer.classify_reviews
    # 0 reviewers -> none
    r0 = cl([])
    check(r0['status'] == 'none' and r0['n_reviewers'] == 0 and r0['code'] == '', r0)
    # 1 reviewer -> single (suggested code = the lone code)
    r1 = cl([{'reviewer': 'a', 'code': 'PA'}])
    check(r1['status'] == 'single' and r1['code'] == 'PA' and r1['n_reviewers'] == 1, r1)
    # 2 agree -> consensus (>=2 real agreeing codes)
    r2 = cl([{'reviewer': 'a', 'code': 'PA'}, {'reviewer': 'b', 'code': 'PA'}])
    check(r2['status'] == 'consensus' and r2['code'] == 'PA' and r2['codes'] == ['PA'], r2)
    # 2 distinct -> conflict (no suggested code)
    r3 = cl([{'reviewer': 'a', 'code': 'PA'}, {'reviewer': 'b', 'code': 'SS'}])
    check(r3['status'] == 'conflict' and r3['code'] == ''
          and set(r3['codes']) == {'PA', 'SS'}, r3)
    # blanks/IDK ignored for agreement: one real PA + one IDK -> single (not consensus)
    r4 = cl([{'reviewer': 'a', 'code': 'PA'}, {'reviewer': 'b', 'code': 'IDK'}])
    check(r4['status'] == 'single' and r4['code'] == 'PA', r4)
    # 3 reviewers, all PA -> consensus
    r5 = cl([{'reviewer': x, 'code': 'PA'} for x in ('a', 'b', 'c')])
    check(r5['status'] == 'consensus' and r5['n_reviewers'] == 3, r5)


def test_consensus_rows_table_from_repo():
    """consensus_rows builds one row per pending uid with its computed status —
    a pure read that mutates nothing (CONTRACTS §6)."""
    with tempfile.TemporaryDirectory() as t:
        rd = os.path.join(t, 'R'); ld = os.path.join(t, '_lib')
        repo = ReviewRepo(rd, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        repo.add_item('AGREE', {'uid': 'AGREE', 'project_id': 'p', 'project_name': 'P'})
        repo.add_review('AGREE', 'a', 'PA', 'high')
        repo.add_review('AGREE', 'b', 'PA', 'low')
        repo.add_item('CLASH', {'uid': 'CLASH', 'project_id': 'p'})
        repo.add_review('CLASH', 'a', 'PA')
        repo.add_review('CLASH', 'b', 'SS')
        repo.add_item('LONE', {'uid': 'LONE', 'project_id': 'p'})
        repo.add_review('LONE', 'a', 'AA')
        rows = importer.consensus_rows(rd, repo_url='', master_codes=None, library_dir=ld,
                                       log_fn=lambda m: None)
        by = {r['uid']: r for r in rows}
        check(by['AGREE']['status'] == 'consensus' and by['AGREE']['suggested'] == 'PA', by['AGREE'])
        check(by['CLASH']['status'] == 'conflict' and by['CLASH']['suggested'] == '', by['CLASH'])
        check(by['LONE']['status'] == 'single' and by['LONE']['suggested'] == 'AA', by['LONE'])
        # Conflicts sort first (operator attention), then consensus, then single.
        check(rows[0]['uid'] == 'CLASH', f"conflict first: {[r['uid'] for r in rows]}")
        # Pure read: nothing pruned.
        check(set(ReviewRepo(rd, master_codes_path=None).pending_uids())
              == {'AGREE', 'CLASH', 'LONE'}, 'consensus_rows mutates nothing')


def test_commit_push_releases_lock_before_push():
    # The flock must NOT be held across the network push: a second instance can
    # acquire the lock while a (no-op) push runs. We assert the index work is
    # locked but the lock is free afterward, and a concurrent add_item succeeds.
    with tempfile.TemporaryDirectory() as t:
        repo = ReviewRepo(os.path.join(t, 'R'), remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        repo.add_item('UID-1', {'uid': 'UID-1', 'site': 'SCP'})
        ok = repo.commit_push('t', push=False)  # push disabled -> returns False, lock released
        check(ok is False, 'commit-only returns False')
        # a fresh instance (separate in-process lock-depth) can lock immediately
        repo2 = ReviewRepo(os.path.join(t, 'R'), remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo2.add_item('UID-2', {'uid': 'UID-2', 'site': 'SCP'})
        check(set(repo2.pending_uids()) == {'UID-1', 'UID-2'}, 'both items present, no deadlock')


def test_ensure_repo_atomic_and_gitignore_reasserted():
    with tempfile.TemporaryDirectory() as t:
        root = os.path.join(t, 'R')
        repo = ReviewRepo(root, remote_url='', master_codes_path=None, log_fn=lambda m: None)
        repo.ensure_repo()
        # Simulate a pre-existing tree that lost its .gitignore (the live-repo case)
        os.remove(os.path.join(root, '.gitignore'))
        repo.ensure_repo()   # must re-assert it
        gi = open(os.path.join(root, '.gitignore')).read()
        check('.reefreview.lock' in gi, '.gitignore re-asserted when missing')
        # no leftover .tmp files from atomic writes
        check(not any(f.endswith('.tmp') for f in os.listdir(root)), 'no leftover .tmp')


def test_noop_preamble_never_dirties_clean_worktree():
    # Regression (2026-07-09): the export_flagged_masks preamble (ensure_repo +
    # set_contacts with unchanged contacts) against an already-seeded repo
    # rewrote codes.json + review_manifest.json with pure generated_at churn,
    # rewrote README.md, and touched .git/config — leaving the PRODUCTION
    # working tree dirty (' M codes.json / M review_manifest.json') with no
    # commit to sweep it (a zero-mask export never commits). All seeds must be
    # idempotent: a no-op preamble leaves the tree byte-identical + git-clean.
    with tempfile.TemporaryDirectory() as t:
        root = os.path.join(t, 'R')

        def _git(*args):
            return subprocess.run([review_repo_mod.GIT, *args], cwd=root,
                                  capture_output=True, text=True)

        def _preamble():
            r = ReviewRepo(root, remote_url='', master_codes_path=None,
                           log_fn=lambda m: None)
            r.ensure_repo()
            r.set_contacts(['lauren.olinger@uvi.edu'])
            return r

        repo = _preamble()
        _git('add', '-A')
        _git('commit', '-m', 'baseline')
        check(not _git('status', '--porcelain').stdout.strip(), 'baseline clean')

        watched = ['review_manifest.json', 'codes.json', 'README.md',
                   'index.html', os.path.join('.git', 'config')]
        def _stats():
            return {fn: (lambda s: (s.st_ino, s.st_mtime_ns, s.st_size))
                        (os.stat(os.path.join(root, fn))) for fn in watched}
        before = _stats()
        gen_before = repo.load_manifest()['generated_at']

        _preamble()   # the no-op export preamble, fresh instance

        status = _git('status', '--porcelain').stdout
        check(not status.strip(), f'no-op preamble dirtied the tree: {status!r}')
        after = _stats()
        churned = [fn for fn in watched if before[fn] != after[fn]]
        check(not churned, f'no-op preamble rewrote files: {churned}')
        check(repo.load_manifest()['generated_at'] == gen_before,
              'generated_at churned with no semantic change')

        # A REAL change still writes, dirties, and is visible on the manifest.
        repo.set_contacts(['someone.else@uvi.edu'])
        check(repo.load_manifest()['contacts'] == ['someone.else@uvi.edu'],
              'real contact change persisted')
        check(' M review_manifest.json' in _git('status', '--porcelain').stdout,
              'real change dirties the manifest for the next commit')

        # Operator sidecar extras survive a codes.json re-seed (write_codes
        # reads operator_setup.json directly).
        with open(os.path.join(root, 'operator_setup.json'), 'w') as f:
            json.dump({'email': '', 'candidate_codes': ['ZZTEST']}, f)
        ReviewRepo(root, remote_url='', master_codes_path=None,
                   log_fn=lambda m: None).write_codes()
        cj = json.load(open(os.path.join(root, 'codes.json')))
        check('ZZTEST' in cj['candidate_codes'],
              f"sidecar extras survive re-seed: {cj['candidate_codes']}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    print(f"Running {len(tests)} reviewer tests...\n")
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
