"""
End-to-end expert-review pathway test (CONTRACTS §2,§4,§5,§6 + the just-landed
remove_project batch delete and blank-confidence viewer CSV).

NO-pytest harness (mirrors _reefreview/tests and _expertids/tests): check()/
run(), __main__ exits nonzero on any failure. Runnable with the unified env
python:

    env/bin/python scripts/_expertids/tests/test_e2e_expert_review.py

One sequential scenario in temp dirs only (never the live review repo, git push
disabled everywhere):

  10  two synthetic projects (project.json id/name) with REVIEW-flagged masks
      exported to a temp review repo; manifest items carry project identity and
      their crop/mask/full assets exist on disk
  20  reviewers A and B return viewer CSVs (header uid,code,confidence,
      reviewer,project_id with BLANK confidence) across BOTH projects;
      import_rows upserts tentative reviews into the manifest AND the routed
      segmentations.json masks; nothing unrouted; no expert_id yet
  30  consensus_rows shows both reviewers with the right agreement status:
      one consensus uid, one conflict, two singles
  40  accepting the consensus uid stamps expert_id (mode EXPERT) + species in
      the right project's segmentations.json, prunes the repo item and its
      items/<uid>/ assets, and promotes the library record
  50  the importer refreshed each touched project's label_provenance.csv
      ITSELF during import (20) and accept (40): the ledger already shows
      found_expert / pending_expert with NO annotator re-touch and feeds
      _matrix.builder.build_matrix with matching cell outcomes; an
      annotator-style manual re-stamp is redundant-by-design (idempotent,
      matrix unchanged)
  60  ReviewRepo.remove_project batch-deletes project 2's items + asset dirs,
      leaves project 1 intact, and recomputes the manifest count
  70  export_yolo.export_batch excludes pending/REVIEW masks and includes the
      accepted expert mask under its accepted code
"""

import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                       # scripts/_expertids
_SCRIPTS = os.path.dirname(_PKG)                    # scripts
sys.path.insert(0, _SCRIPTS)
# The ACTIVE step4test combined annotator owns provenance + the YOLO export.
sys.path.insert(0, os.path.join(_SCRIPTS, 'TCRMPclip_combinedAnnotate', 'src'))

from _reefreview import mask_geom, review_export
from _reefreview.review_repo import ReviewRepo
from _reefreview.library import Library
from _expertids import importer
from _matrix import builder as matrix_builder
import provenance
import export_yolo

_RESULTS = []
_SILENT = lambda _m: None  # noqa: E731

# Shared scenario state: stage tests run in sorted order and build on each
# other (a true round-trip, not isolated units).
CTX = {}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn(); _RESULTS.append((fn.__name__, True, '')); print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── synthetic fixtures ───────────────────────────────────────────────
def _square(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool); m[y0:y1, x0:x1] = True; return m


def _mask(sq, sx, sy, species='REVIEW', review=True, status='pending'):
    y0, y1, x0, x1 = sq
    rle = mask_geom.rle_encode(_square(200, 200, y0, y1, x0, x1))
    d = {'id': 0, 'label': 'A', 'species': species, 'name': '',
         'category': 'Review' if review else '',
         'source_x': sx, 'source_y': sy,
         'polygon_px': [[x0, y0, x1, y0, x1, y1, x0, y1]],
         'polygon_norm': [[x0 / 200, y0 / 200, x1 / 200, y0 / 200,
                           x1 / 200, y1 / 200, x0 / 200, y1 / 200]],
         'rle': rle, 'bbox': [x0, y0, x1 - 1, y1 - 1],
         'area': (y1 - y0) * (x1 - x0), 'score': 0.9, 'status': status,
         'refinement_clicks': [], 'source_type': 'auto'}
    if review:
        d['review'] = True
    return d


def _mk_project(root, pid, name, frames):
    """<root>/inprocess/<pid>/ with project.json (id/name) + the ACTIVE
    step4test export dir: raw imagery + segmentations/2020/segmentations.json.
    The dir is named run_* so build_matrix scans it, and project.json id==pid
    so resolve_step_dir routes the importer/accept here. frames maps
    filename -> [mask dicts]. Returns (export_dir, segdir, segmap)."""
    pdir = os.path.join(root, 'inprocess', pid)
    export_dir = os.path.join(pdir, 'step4test_combinedAnnotate')
    segdir = os.path.join(export_dir, 'segmentations', '2020')
    raw = os.path.join(export_dir, 'raw')
    os.makedirs(segdir, exist_ok=True); os.makedirs(raw, exist_ok=True)
    with open(os.path.join(pdir, 'project.json'), 'w') as f:
        json.dump({'id': pid, 'name': name}, f)
    segmap = {}
    for fn, masks in frames.items():
        img = os.path.join(raw, fn)
        Image.fromarray((np.random.rand(200, 200, 3) * 255).astype('uint8')).save(img)
        segmap[fn] = {'image_path': 'raw/' + fn, 'image_path_abs': img,
                      'image_width': 200, 'image_height': 200, 'masks': masks,
                      'reviewed': False, 'exported': False}
    _write_seg(segdir, segmap)
    return export_dir, segdir, segmap


def _write_seg(segdir, segmap):
    with open(os.path.join(segdir, 'segmentations.json'), 'w') as f:
        json.dump(segmap, f)


def _read_seg(segdir):
    with open(os.path.join(segdir, 'segmentations.json')) as f:
        return json.load(f)


# ── 10: build two projects + export flagged masks to the review repo ─
def test_10_two_projects_export_review_masks():
    root = CTX['root']
    pr = os.path.join(root, 'inprocess')
    rd = os.path.join(root, 'REVIEW')
    ld = os.path.join(root, '_lib')
    f1, f2 = 'TCRMP20201022_clip_SCP_T101.jpeg', 'TCRMP20201022_clip_SCP_T102.jpeg'
    f3, f4 = 'TCRMP20211023_clip_BWR_T201.jpeg', 'TCRMP20211023_clip_BWR_T202.jpeg'
    p1, p2 = 'run_alpha_20200101_aaa', 'run_beta_20200101_bbb'
    CTX.update(pr=pr, rd=rd, ld=ld, f1=f1, f2=f2, f3=f3, f4=f4, p1=p1, p2=p2)

    # Project 1: m1 = the future consensus mask, left at the build_mask_dict
    # default status 'pending' so the expert accept must flip it itself before
    # YOLO export or found_expert can see it; m1b = a plain pending AA mask
    # that must never train; m2 = the future conflict mask, annotator-accepted
    # (covers the other pre-accept state: accepted geometry, species REVIEW).
    m1 = _mask((60, 140, 60, 140), 100, 100)
    m1b = _mask((10, 40, 10, 40), 25, 25, species='AA', review=False)
    m2 = _mask((30, 120, 40, 130), 85, 75, status='accepted')
    ed1, segdir1, segmap1 = _mk_project(root, p1, 'Alpha Run', {f1: [m1, m1b], f2: [m2]})
    st1 = review_export.export_flagged_masks(
        segmap1, [f1, f2], export_dir=ed1, review_dir=rd, repo_url='',
        library_dir=ld, master_codes=None, contacts=['expert@example.com'],
        featured_codes=['PA'], project_id=p1, project_name='Alpha Run',
        git_push=False, log_fn=_SILENT)
    check(st1['new'] == 2 and st1['auto_relabeled'] == 0, st1)
    check(st1['pushed'] is False, f"push disabled: {st1}")
    _write_seg(segdir1, segmap1)   # persist the review_uid the exporter stamped
    u1, u2 = st1['uids']

    # Project 2: two still-pending REVIEW masks (the future singles).
    m3 = _mask((50, 130, 50, 130), 90, 90)
    m4 = _mask((20, 100, 60, 150), 105, 60)
    ed2, segdir2, segmap2 = _mk_project(root, p2, 'Beta Run', {f3: [m3], f4: [m4]})
    st2 = review_export.export_flagged_masks(
        segmap2, [f3, f4], export_dir=ed2, review_dir=rd, repo_url='',
        library_dir=ld, master_codes=None, contacts=['expert@example.com'],
        featured_codes=['PA'], project_id=p2, project_name='Beta Run',
        git_push=False, log_fn=_SILENT)
    check(st2['new'] == 2 and st2['pushed'] is False, st2)
    _write_seg(segdir2, segmap2)
    u3, u4 = st2['uids']
    CTX.update(ed1=ed1, ed2=ed2, segdir1=segdir1, segdir2=segdir2,
               u1=u1, u2=u2, u3=u3, u4=u4)
    check(len({u1, u2, u3, u4}) == 4, f"four distinct uids: {[u1, u2, u3, u4]}")

    # Manifest items carry project identity + on-disk assets.
    repo = ReviewRepo(rd, master_codes_path=None, log_fn=_SILENT)
    man = repo.load_manifest()
    check(man['count'] == 4 and len(man['items']) == 4, f"4 items queued: {man['count']}")
    by = {it['uid']: it for it in man['items']}
    for uid, pid, pname in ((u1, p1, 'Alpha Run'), (u2, p1, 'Alpha Run'),
                            (u3, p2, 'Beta Run'), (u4, p2, 'Beta Run')):
        it = by[uid]
        check(it['project_id'] == pid and it['project_name'] == pname,
              f"{uid} carries project identity: {it.get('project_id')}, {it.get('project_name')}")
        for key in ('crop', 'mask', 'full'):
            rel = it.get(key)
            check(rel and os.path.exists(os.path.join(rd, rel)),
                  f"{uid} {key} asset exists: {rel}")
        check(it['reviews'] == [] and it['accepted'] is None,
              f"{uid} starts with no reviews and no acceptance: {it}")
    check('expert@example.com' in man.get('contacts', []),
          f"contacts stamped: {man.get('contacts')}")


# ── 20: reviewers A + B return viewer CSVs (blank confidence) ────────
def test_20_viewer_csvs_import_tentative_reviews():
    rd, ld, pr = CTX['rd'], CTX['ld'], CTX['pr']
    p1, p2 = CTX['p1'], CTX['p2']
    u1, u2, u3, u4 = CTX['u1'], CTX['u2'], CTX['u3'], CTX['u4']

    # The new viewer omits confidence: the column is present but BLANK.
    csv_a = ('uid,code,confidence,reviewer,project_id\n'
             f"{u1},PA,,Reviewer A,{p1}\n"
             f"{u2},PA,,Reviewer A,{p1}\n"
             f"{u3},AA,,Reviewer A,{p2}\n")
    rows_a = importer.parse_csv_text(csv_a)
    check(len(rows_a) == 3 and all(r['confidence'] == '' for r in rows_a),
          f"blank confidence parsed: {rows_a}")
    check(rows_a[0]['reviewer'] == 'Reviewer A' and rows_a[0]['project_id'] == p1,
          rows_a[0])
    ist_a = importer.import_rows(rows_a, export_dir='', review_dir=rd, repo_url='',
                                 library_dir=ld, master_codes=None, projects_root=pr,
                                 git_push=False, log_fn=_SILENT)
    check(ist_a['rows'] == 3 and ist_a['reviews_added'] == 3, ist_a)
    check(ist_a['unrouted'] == [], f"nothing unrouted for A: {ist_a}")
    check(ist_a['auto_tentative'] == 0, f"no overlap cascade rows: {ist_a}")
    check(ist_a['provenance_refreshed'] == 2,
          f"import refreshed both touched projects' ledgers: {ist_a}")

    csv_b = ('uid,code,confidence,reviewer,project_id\n'
             f"{u1},PA,,Reviewer B,{p1}\n"
             f"{u2},SS,,Reviewer B,{p1}\n"
             f"{u4},SS,,Reviewer B,{p2}\n")
    ist_b = importer.import_rows(importer.parse_csv_text(csv_b), export_dir='',
                                 review_dir=rd, repo_url='', library_dir=ld,
                                 master_codes=None, projects_root=pr,
                                 git_push=False, log_fn=_SILENT)
    check(ist_b['rows'] == 3 and ist_b['reviews_added'] == 3, ist_b)
    check(ist_b['unrouted'] == [], f"nothing unrouted for B: {ist_b}")

    # Manifest: reviews[] upserted per reviewer, no acceptance, no prune.
    repo = ReviewRepo(rd, master_codes_path=None, log_fn=_SILENT)
    by = {it['uid']: it for it in repo.load_manifest()['items']}
    want = {u1: {'Reviewer A': 'PA', 'Reviewer B': 'PA'},
            u2: {'Reviewer A': 'PA', 'Reviewer B': 'SS'},
            u3: {'Reviewer A': 'AA'},
            u4: {'Reviewer B': 'SS'}}
    for uid, expected in want.items():
        revs = {r['reviewer']: r for r in by[uid]['reviews']}
        check({k: v['code'] for k, v in revs.items()} == expected,
              f"{uid} manifest reviews: {by[uid]['reviews']}")
        check(all(v['confidence'] == '' for v in revs.values()),
              f"{uid} blank confidence stored: {by[uid]['reviews']}")
        check(by[uid]['accepted'] is None, f"{uid} not accepted by import")
    check(len(repo.pending_uids()) == 4, 'import never prunes')
    check(set(repo.load_manifest().get('reviewer_names', []))
          == {'Reviewer A', 'Reviewer B'}, 'both reviewers tracked')

    # Segmentations: tentative reviews landed on the routed project's masks.
    seg1 = _read_seg(CTX['segdir1'])
    mm1 = seg1[CTX['f1']]['masks'][0]
    check({r['reviewer'] for r in mm1['reviews']} == {'Reviewer A', 'Reviewer B'},
          f"consensus mask carries both reviewers: {mm1.get('reviews')}")
    m1b = seg1[CTX['f1']]['masks'][1]
    check(not m1b.get('reviews') and m1b['species'] == 'AA',
          f"non-review AA mask untouched: {m1b.get('reviews')}")
    mm2 = seg1[CTX['f2']]['masks'][0]
    check({(r['reviewer'], r['code']) for r in mm2['reviews']}
          == {('Reviewer A', 'PA'), ('Reviewer B', 'SS')},
          f"conflict mask carries both codes: {mm2.get('reviews')}")
    seg2 = _read_seg(CTX['segdir2'])
    mm3 = seg2[CTX['f3']]['masks'][0]
    mm4 = seg2[CTX['f4']]['masks'][0]
    check([(r['reviewer'], r['code']) for r in mm3['reviews']] == [('Reviewer A', 'AA')],
          f"single A routed to project 2: {mm3.get('reviews')}")
    check([(r['reviewer'], r['code']) for r in mm4['reviews']] == [('Reviewer B', 'SS')],
          f"single B routed to project 2: {mm4.get('reviews')}")
    for mm in (mm1, mm2, mm3, mm4):
        check('expert_id' not in mm, f"import must NOT set expert_id: {mm.get('expert_id')}")
        check(mm['species'] == 'REVIEW' and mm.get('review') is True,
              f"mask stays pending after import: {mm.get('species')}")
        check(all((r.get('reviewer') or '') != 'overlap' for r in mm['reviews']),
              f"no synthetic overlap rows here: {mm['reviews']}")

    # Library: records still USER/REVIEW; per-uid detail mirrors the reviews.
    lib = Library(ld)
    for uid in (u1, u2, u3, u4):
        rec = lib.lookup(uid)
        check(rec['mode'] == 'USER' and rec['code'] == 'REVIEW',
              f"{uid} library not promoted by import: {rec}")
    det = lib.load_reviews(u1)
    check(len(det['reviews']) == 2 and det['accepted'] is None,
          f"library detail mirror: {det}")


# ── 30: consensus table ──────────────────────────────────────────────
def test_30_consensus_rows_agreement_status():
    rd, ld = CTX['rd'], CTX['ld']
    u1, u2, u3, u4 = CTX['u1'], CTX['u2'], CTX['u3'], CTX['u4']
    rows = importer.consensus_rows(rd, repo_url='', master_codes=None,
                                   library_dir=ld, log_fn=_SILENT)
    check(len(rows) == 4, f"one row per pending uid: {[r['uid'] for r in rows]}")
    by = {r['uid']: r for r in rows}
    r1 = by[u1]
    check(r1['status'] == 'consensus' and r1['suggested'] == 'PA'
          and r1['n_reviewers'] == 2, f"consensus row: {r1}")
    check({rv['reviewer'] for rv in r1['reviews']} == {'Reviewer A', 'Reviewer B'},
          f"both reviewer columns present: {r1['reviews']}")
    r2 = by[u2]
    check(r2['status'] == 'conflict' and r2['suggested'] == ''
          and set(r2['codes']) == {'PA', 'SS'}, f"conflict row: {r2}")
    check(by[u3]['status'] == 'single' and by[u3]['suggested'] == 'AA', by[u3])
    check(by[u4]['status'] == 'single' and by[u4]['suggested'] == 'SS', by[u4])
    check(rows[0]['uid'] == u2, f"conflict sorts first: {[r['uid'] for r in rows]}")
    # Pure read: nothing pruned.
    check(len(ReviewRepo(rd, master_codes_path=None).pending_uids()) == 4,
          'consensus_rows mutates nothing')


# ── 40: accept the consensus uid ─────────────────────────────────────
def test_40_accept_consensus_uid():
    rd, ld, pr = CTX['rd'], CTX['ld'], CTX['pr']
    u1, u2, u3, u4 = CTX['u1'], CTX['u2'], CTX['u3'], CTX['u4']
    res = importer.accept_uid(u1, 'PA', review_dir=rd, repo_url='', library_dir=ld,
                              master_codes=None, projects_root=pr, labeler='LO',
                              basis='consensus', git_push=False, log_fn=_SILENT)
    check(res['ok'] is True and res['relabeled_seg'] is True, res)
    check(res['export_dir'] == CTX['ed1'], f"routed to project 1: {res}")
    check(res['pushed'] is False, f"push disabled on accept: {res}")
    check(res['provenance_refreshed'] == 1,
          f"accept refreshed the relabeled tree's ledger: {res}")

    # Right project's mask relabeled with the final EXPERT id.
    seg1 = _read_seg(CTX['segdir1'])
    mm1 = seg1[CTX['f1']]['masks'][0]
    check(mm1['species'] == 'PA' and mm1['review'] is False,
          f"accepted mask relabeled: {mm1.get('species')}")
    eid = mm1.get('expert_id', {})
    check(eid.get('mode') == 'EXPERT' and eid.get('code') == 'PA'
          and eid.get('labeler') == 'LO' and eid.get('basis') == 'consensus',
          f"expert_id stamped: {eid}")
    check(mm1['status'] == 'accepted',
          f"expert accept flipped the pending mask to accepted: {mm1['status']}")
    # Everything else untouched.
    m1b = seg1[CTX['f1']]['masks'][1]
    check(m1b['species'] == 'AA' and 'expert_id' not in m1b, 'AA mask untouched')
    mm2 = seg1[CTX['f2']]['masks'][0]
    check(mm2['species'] == 'REVIEW' and 'expert_id' not in mm2,
          f"conflict mask untouched: {mm2.get('species')}")
    seg2 = _read_seg(CTX['segdir2'])
    for fn in (CTX['f3'], CTX['f4']):
        mm = seg2[fn]['masks'][0]
        check(mm['species'] == 'REVIEW' and 'expert_id' not in mm,
              f"project 2 untouched by project 1 accept: {fn}")

    # Repo: item removed from the manifest AND its items/<uid>/ assets pruned.
    repo = ReviewRepo(rd, master_codes_path=None, log_fn=_SILENT)
    check(set(repo.pending_uids()) == {u2, u3, u4},
          f"only the accepted uid pruned: {repo.pending_uids()}")
    check(repo.load_manifest()['count'] == 3, 'manifest count recomputed')
    check(not os.path.isdir(os.path.join(repo.items_dir, u1)), 'accepted item assets gone')
    for uid in (u2, u3, u4):
        check(os.path.isdir(os.path.join(repo.items_dir, uid)),
              f"pending item assets intact: {uid}")

    # Library promoted to EXPERT and the acceptance persisted.
    lib = Library(ld)
    rec = lib.lookup(u1)
    check(rec['mode'] == 'EXPERT' and rec['code'] == 'PA', f"library EXPERT: {rec}")
    det = lib.load_reviews(u1)
    check(det['accepted'] and det['accepted']['code'] == 'PA'
          and det['accepted']['mode'] == 'EXPERT', f"accepted persisted: {det}")


# ── 50: provenance outcomes + coverage matrix ────────────────────────
def test_50_provenance_and_matrix():
    ed1, ed2, pr, p1 = CTX['ed1'], CTX['ed2'], CTX['pr'], CTX['p1']
    f1, f2 = CTX['f1'], CTX['f2']
    seg1 = _read_seg(CTX['segdir1'])

    def _ledger_map(export_dir):
        import csv
        path = os.path.join(export_dir, 'label_provenance.csv')
        check(os.path.exists(path), f"ledger exists: {path}")
        with open(path, newline='') as fh:
            return {(r['basename'], r['label']): r['outcome']
                    for r in csv.DictReader(fh)}

    # The importer regenerated the ledgers ITSELF during import (20) and
    # accept (40): found_expert / pending_expert are already on disk with NO
    # annotator re-touch and no manual provenance call.
    s1 = os.path.splitext(f1)[0]
    s2 = os.path.splitext(f2)[0]
    s3 = os.path.splitext(CTX['f3'])[0]
    s4 = os.path.splitext(CTX['f4'])[0]
    led1 = _ledger_map(ed1)
    check(led1.get((s1, 'PA')) == 'found_expert',
          f"accept refreshed the ledger to found_expert: {led1}")
    check(led1.get((s2, 'PA')) == 'pending_expert'
          and led1.get((s2, 'SS')) == 'pending_expert',
          f"import refreshed pending tentatives: {led1}")
    led2 = _ledger_map(ed2)
    check(led2.get((s3, 'AA')) == 'pending_expert'
          and led2.get((s4, 'SS')) == 'pending_expert',
          f"project 2 ledger carries its tentatives: {led2}")

    mx = matrix_builder.build_matrix(pr)
    check(mx['cells'][s1]['PA']['outcome'] == 'found_expert',
          f"matrix cell 1: {mx['cells'].get(s1)}")
    check(mx['cells'][s2]['PA']['outcome'] == 'pending_expert',
          f"matrix cell 2: {mx['cells'].get(s2)}")
    check(mx['cells'][s2]['SS']['outcome'] == 'pending_expert',
          f"matrix cell 2 SS: {mx['cells'].get(s2)}")
    check(mx['cells'][s3]['AA']['outcome'] == 'pending_expert'
          and mx['cells'][s4]['SS']['outcome'] == 'pending_expert',
          f"project 2 pending cells surface: {mx['cells'].get(s3)}, {mx['cells'].get(s4)}")
    pids = {p['project_id'] for p in mx['projects']}
    check({p1, CTX['p2']} <= pids, f"both projects scanned: {pids}")
    check(mx['stats']['by_outcome']['found_expert'] == 1
          and mx['stats']['by_outcome']['pending_expert'] == 4,
          f"matrix stats: {mx['stats']['by_outcome']}")

    # An annotator-style manual re-stamp is now redundant-by-design: the
    # outcomes agree with the on-disk ledger and regenerating changes nothing.
    out_f1 = provenance.compute_label_outcomes(seg1[f1], ['PA'], 'step4test',
                                               reviewer='LO')
    check(set(out_f1) == {'PA'} and out_f1['PA']['outcome'] == 'found_expert',
          f"accepted expert mask -> found_expert: {out_f1}")
    out_f2 = provenance.compute_label_outcomes(seg1[f2], ['PA'], 'step4test',
                                               reviewer='LO')
    check(out_f2['PA']['outcome'] == 'pending_expert',
          f"pending tentative PA -> pending_expert: {out_f2}")
    check(out_f2['SS']['outcome'] == 'pending_expert',
          f"pending tentative SS -> pending_expert: {out_f2}")
    provenance.write_provenance_csv(ed1, s1, out_f1, p1)
    provenance.write_provenance_csv(ed1, s2, out_f2, p1)
    mx2 = matrix_builder.build_matrix(pr)
    check({b: {l: c['outcome'] for l, c in by.items()}
           for b, by in mx2['cells'].items()}
          == {b: {l: c['outcome'] for l, c in by.items()}
              for b, by in mx['cells'].items()},
          'manual re-stamp is idempotent: no matrix cell changed')
    check(mx2['stats']['by_outcome'] == mx['stats']['by_outcome'],
          f"stats unchanged by re-stamp: {mx2['stats']['by_outcome']}")


# ── 60: batch delete project 2 from the review site ──────────────────
def test_60_remove_project_batch_delete():
    rd, ld = CTX['rd'], CTX['ld']
    p1, p2 = CTX['p1'], CTX['p2']
    u2, u3, u4 = CTX['u2'], CTX['u3'], CTX['u4']
    repo = ReviewRepo(rd, master_codes_path=None, log_fn=_SILENT)
    res = repo.remove_project(p2)
    check(res['removed'] == 2 and set(res['uids']) == {u3, u4},
          f"project 2 batch-removed: {res}")
    for uid in (u3, u4):
        check(not os.path.isdir(os.path.join(repo.items_dir, uid)),
              f"deleted project's item assets gone: {uid}")
    check(os.path.isdir(os.path.join(repo.items_dir, u2)),
          'project 1 item assets intact')
    man = repo.load_manifest()
    check(man['count'] == 1 and repo.pending_uids() == [u2],
          f"manifest count right after delete: {man['count']}, {repo.pending_uids()}")
    check(man['items'][0]['project_id'] == p1, 'remaining item belongs to project 1')
    # The cross-project library is untouched by the site prune.
    lib = Library(ld)
    check(lib.lookup(u3) is not None and lib.lookup(u4) is not None,
          'library records survive remove_project')
    # Unknown project id is a safe zero, not an error.
    res2 = repo.remove_project('nope_does_not_exist')
    check(res2 == {'removed': 0, 'uids': []}, f"unknown project: {res2}")
    check(repo.load_manifest()['count'] == 1, 'unknown project changed nothing')


# ── 70: training handoff (YOLO export selection) ─────────────────────
def test_70_yolo_export_training_handoff():
    f1, f2 = CTX['f1'], CTX['f2']
    seg1 = _read_seg(CTX['segdir1'])
    ydir = os.path.join(CTX['root'], 'yolo_export')
    cm = export_yolo.load_class_map(ydir)
    check(cm == {}, f"fresh class map: {cm}")
    st = export_yolo.export_batch(seg1, ydir, cm, symlink=False)
    # f1 exports (one accepted expert PA mask); f2 is review-only and skipped
    # entirely so the flagged coral never becomes a false negative.
    check(st['exported_images'] == 1 and st['review_only_skipped'] == 1, st)
    check(st['exported_masks'] == 1 and st['exported_empty'] == 0, st)

    check('PA' in cm, f"accepted code registered: {cm}")
    for excluded in ('REVIEW', 'AA', 'SS'):
        check(excluded not in cm, f"{excluded} must not reach training: {cm}")

    s1 = os.path.splitext(f1)[0]
    s2 = os.path.splitext(f2)[0]
    label1 = os.path.join(ydir, 'all_labels', s1 + '.txt')
    check(os.path.exists(label1), 'label written for the accepted frame')
    lines = [ln for ln in open(label1).read().splitlines() if ln.strip()]
    check(len(lines) == 1, f"only the accepted expert mask labeled: {lines}")
    toks = lines[0].split()
    check(toks[0] == str(cm['PA']) and len(toks) == 1 + 8,
          f"YOLO line under the accepted class: {lines[0]}")
    check(os.path.exists(os.path.join(ydir, 'all_images', f1)), 'image exported')
    check(not os.path.exists(os.path.join(ydir, 'all_labels', s2 + '.txt')),
          'no label for the review-only frame')
    check(not os.path.exists(os.path.join(ydir, 'all_images', f2)),
          'review-only frame image not exported')
    # class_map.json + data.yaml persisted for the trainer.
    check(export_yolo.load_class_map(ydir) == cm, 'class_map.json persisted')
    data_yaml = open(os.path.join(ydir, 'data.yaml')).read()
    check('PA' in data_yaml and 'REVIEW' not in data_yaml,
          f"data.yaml carries the accepted class only: {data_yaml}")


def main():
    CTX['root'] = tempfile.mkdtemp(prefix='e2e_expert_review_')
    try:
        tests = [v for k, v in sorted(globals().items())
                 if k.startswith('test_') and callable(v)]
        print(f"Running {len(tests)} end-to-end expert-review stages...\n")
        for fn in tests:
            run(fn)
    finally:
        shutil.rmtree(CTX['root'], ignore_errors=True)
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
