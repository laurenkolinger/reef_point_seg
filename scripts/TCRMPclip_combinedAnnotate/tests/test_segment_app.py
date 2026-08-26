"""
Self-contained smoke + unit tests for the Step-5 Segment app.

No pytest dependency — run with the unified env python:
    env/bin/python scripts/TCRMPclip_segmentImages/tests/test_segment_app.py

Layers (CONTRACTS §9):
  SMOKE  — app.py imports, Flask GET / renders 200, the index template's
           inline <script> passes `node --check` after Jinja stripping.
  UNIT   — mask_ops.merge_overlapping_same_id unions two overlapping
           same-species masks (and leaves a different-species overlap alone);
           app._merge_imported_review_fields merges imported expert/pending
           fields onto in-memory masks without clobbering operator edits.

The SAM3 engine is never loaded (none of these paths need it).
"""

import os
import sys
import re
import subprocess
import tempfile
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
sys.path.insert(0, _SRC)

# The combined-annotator backend reads TARGET_SPECIES + MANUAL_ANNOTATE from
# config at IMPORT time. Pin them here, before `import app` happens anywhere in
# this module, so the manual-mode smoke test below sees a deterministic config.
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'DICT')
os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')

import cv2
import mask_ops
from mask_ops import rle_encode, build_mask_dict

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
H, W = 60, 60


def _rect_mask(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _mask(mid, species, binary, status='pending'):
    return {
        'id': mid, 'label': mask_ops_label(mid), 'species': species,
        'name': species, 'category': 'Coral',
        'rle': rle_encode(binary), 'bbox': mask_ops.mask_bbox(binary),
        'area': int(binary.sum()), 'score': 0.9, 'status': status,
        'polygon_px': [[x for x in (0, 0, 1, 0, 1, 1)]],
        'refinement_clicks': [],
    }


def mask_ops_label(n):
    s = ''
    n = n + 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ── SMOKE ───────────────────────────────────────────────────────────
def test_smoke_import_app():
    # app.py must import without loading SAM3 (engine stays None).
    import app  # noqa: F401
    check(hasattr(app, 'app'), 'Flask app object missing')
    check(hasattr(app, 'merge_same_id'), '/merge_same_id route fn missing')
    check(hasattr(app, 'reload_segmentations'), '/reload_segmentations route fn missing')
    check(hasattr(app, '_merge_imported_review_fields'), 'field-merge helper missing')


def test_blocking_masks_excludes_client_rejected_ids():
    # The /add and /refine clip step must treat a mask the CLIENT has just
    # deleted as non-blocking, even if its 'rejected' status PUT has not landed
    # on the backend yet (Flask is threaded, so the reject and the add race).
    # The client passes its authoritative rejected ids; they win.
    import app
    a = _rect_mask(10, 30, 10, 30)
    live = _mask(5, 'OFAV', a, status='accepted')   # still 'live' on the backend
    # In client rejected_ids -> excluded from blocking.
    check(app._blocking_masks([live], [5]) == [], 'client rejected_ids must exclude mask 5')
    # Not client-rejected and live -> blocks.
    check(len(app._blocking_masks([live], [])) == 1, 'live mask must block when not client-rejected')
    # Backend status='rejected' -> excluded regardless of ids.
    dead = _mask(6, 'OFAV', a, status='rejected')
    check(app._blocking_masks([dead], []) == [], 'backend-rejected mask must be excluded')


def test_smoke_index_renders_200():
    import app
    app.app.config['TESTING'] = True
    client = app.app.test_client()
    r = client.get('/')
    check(r.status_code == 200, f'GET / returned {r.status_code}')
    body = r.get_data(as_text=True)
    # New UI affordances are present in the rendered page.
    check('Target points only' in body, 'target-only toggle missing from page')
    check('id="qa-btn-review"' in body, 'REVIEW quick-arm button missing from page')
    check('return to main menu' in body, 'done-return button missing from page')
    check('remove / shrink region' in body, 'refine hint missing from page')
    # Relics removed in the one-way / cleanup pass must be GONE.
    check('Merge same-ID masks' not in body, 'merge-same-id button should be removed')
    check('Reload IDs' not in body, 'reload-ids button should be removed')
    check('Accept All Pending' not in body, 'accept-all button should be removed')
    check('Export Batch' not in body, 'export-batch button should be removed')


def test_smoke_status_exposes_batches():
    import app
    client = app.app.test_client()
    r = client.get('/api/status')
    check(r.status_code == 200, f'/api/status returned {r.status_code}')
    js = r.get_json()
    for k in ('batches_total', 'batches_done', 'batches_remaining', 'orchestrator_url'):
        check(k in js, f'/api/status missing {k}')


def test_smoke_template_js_node_check():
    # Extract inline <script>, strip Jinja, node --check. Skips gracefully if
    # node is unavailable in the environment.
    node = None
    for cand in ('node', '/usr/bin/node'):
        try:
            subprocess.run([cand, '--version'], capture_output=True, check=True)
            node = cand
            break
        except Exception:
            continue
    if node is None:
        print('    (node unavailable — skipping JS node --check)')
        return
    tpl = os.path.join(_SRC, 'templates', 'index.html')
    src = open(tpl).read()
    blocks = re.findall(r'<script>(.*?)</script>', src, flags=re.DOTALL)
    js = '\n'.join(blocks)
    js = re.sub(r'{%.*?%}', '', js, flags=re.DOTALL)
    js = re.sub(r'{{.*?}}', '0', js, flags=re.DOTALL)
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(js)
        path = f.name
    try:
        res = subprocess.run([node, '--check', path], capture_output=True, text=True)
        check(res.returncode == 0, f'node --check failed: {res.stderr}')
    finally:
        os.unlink(path)


# ── UNIT: merge_overlapping_same_id ─────────────────────────────────
def test_union_merge_two_overlapping_same_species():
    # Two OFAV masks that overlap → unioned into one survivor (lowest id).
    a = _rect_mask(10, 30, 10, 30)   # 20x20
    b = _rect_mask(20, 40, 20, 40)   # 20x20, overlaps a in [20:30,20:30]
    masks = [_mask(0, 'OFAV', a), _mask(1, 'OFAV', b)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 1, f'expected 1 merge, got {merged}')
    check(len(new_masks) == 1, f'expected 1 survivor mask, got {len(new_masks)}')
    surv = new_masks[0]
    check(surv['id'] == 0, f'survivor should be lowest id (0), got {surv["id"]}')
    # The survivor covers (at least) the raw union footprint. update_mask_geometry
    # runs the tool's standard clean_mask (morphological close + hole fill), so a
    # concave union may pick up a few extra pixels at the notch — that's expected
    # and applies to every edited mask in this app. The invariant that matters:
    # the survivor is bigger than either input alone and >= the raw union.
    raw_union = int((a | b).sum())
    area_a, area_b = int(a.sum()), int(b.sum())
    check(surv['area'] >= raw_union,
          f'unioned area {surv["area"]} < raw union {raw_union}')
    check(surv['area'] > area_a and surv['area'] > area_b,
          f'unioned area {surv["area"]} must exceed each input ({area_a}, {area_b})')
    # And no more than the bounding-box span (sanity: closing can not balloon it).
    check(surv['area'] <= raw_union + 200,
          f'unioned area {surv["area"]} ballooned past tolerance of raw union {raw_union}')


def test_union_merge_leaves_different_species_alone():
    # Overlapping but DIFFERENT species → never merged.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    masks = [_mask(0, 'OFAV', a), _mask(1, 'PAST', b)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f'different species must not merge, got {merged}')
    check(len(new_masks) == 2, f'expected 2 masks, got {len(new_masks)}')


def test_union_merge_skips_non_overlapping_same_species():
    # Same species but disjoint → stay separate.
    a = _rect_mask(0, 10, 0, 10)
    b = _rect_mask(40, 50, 40, 50)
    masks = [_mask(0, 'OFAV', a), _mask(1, 'OFAV', b)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f'disjoint same-species must not merge, got {merged}')
    check(len(new_masks) == 2, f'expected 2 masks, got {len(new_masks)}')


def test_union_merge_ignores_rejected():
    # A rejected overlapping same-species mask is "deleted" — left untouched.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    masks = [_mask(0, 'OFAV', a), _mask(1, 'OFAV', b, status='rejected')]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f'rejected mask must not be merged, got {merged}')
    check(len(new_masks) == 2, f'expected 2 masks (1 rejected), got {len(new_masks)}')


def test_union_merge_transitive_chain():
    # a-b overlap, b-c overlap, a-c disjoint → all three collapse via b.
    a = _rect_mask(0, 20, 0, 20)
    b = _rect_mask(10, 30, 10, 30)
    c = _rect_mask(25, 45, 25, 45)
    masks = [_mask(0, 'OFAV', a), _mask(1, 'OFAV', b), _mask(2, 'OFAV', c)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(len(new_masks) == 1, f'expected 1 chained survivor, got {len(new_masks)}')
    check(merged == 2, f'expected 2 absorbed, got {merged}')


def test_union_merge_never_merges_two_review_masks():
    # Two overlapping REVIEW masks are DIFFERENT organisms awaiting different
    # expert IDs — they must NEVER be merged (would drop one's review state).
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    ma = _mask(0, 'REVIEW', a)
    mb = _mask(1, 'REVIEW', b)
    ma['review_uid'] = 'uid-A'
    ma['reviews'] = [{'reviewer': 'Dr A', 'code': 'OFAV', 'confidence': 'high'}]
    mb['review_uid'] = 'uid-B'
    mb['reviews'] = [{'reviewer': 'Dr B', 'code': 'PAST', 'confidence': 'low'}]
    masks = [ma, mb]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f'two REVIEW masks must not merge, got {merged}')
    check(len(new_masks) == 2, f'expected 2 REVIEW masks kept, got {len(new_masks)}')
    by_uid = {m.get('review_uid'): m for m in new_masks}
    check(by_uid.get('uid-A'), 'first REVIEW mask review_uid lost')
    check(by_uid.get('uid-B'), 'second REVIEW mask review_uid lost')
    check(by_uid['uid-A']['reviews'][0]['reviewer'] == 'Dr A', 'reviews of A lost')
    check(by_uid['uid-B']['reviews'][0]['reviewer'] == 'Dr B', 'reviews of B lost')


def test_union_merge_never_merges_blank_or_question_placeholder():
    # Empty-string and '?' species are placeholders too — never grouped.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    masks = [_mask(0, '', a), _mask(1, '', b)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f"blank-species masks must not merge, got {merged}")
    check(len(new_masks) == 2, f'expected 2 blank masks, got {len(new_masks)}')

    masks = [_mask(0, '?', a), _mask(1, '?', b)]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f"'?'-species masks must not merge, got {merged}")
    check(len(new_masks) == 2, f"expected 2 '?' masks, got {len(new_masks)}")


def test_union_merge_absorbed_expert_id_and_reviews_survive():
    # Two overlapping OFAV masks: the ABSORBED (higher-id) one carries the
    # accepted expert_id + reviews[]. After merge they must survive on the
    # lowest-id survivor, and the survivor must NOT be downgraded to pending.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    surv_in = _mask(0, 'OFAV', a, status='pending')   # bare survivor (no review)
    absorbed_in = _mask(1, 'OFAV', b)
    absorbed_in['review_uid'] = 'uid-keep'
    absorbed_in['expert_id'] = {'code': 'OFAV', 'mode': 'EXPERT', 'labeler': 'Dr X'}
    absorbed_in['reviews'] = [{'reviewer': 'Dr X', 'code': 'OFAV', 'confidence': 'high'}]
    masks = [surv_in, absorbed_in]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 1, f'expected 1 merge, got {merged}')
    check(len(new_masks) == 1, f'expected 1 survivor, got {len(new_masks)}')
    surv = new_masks[0]
    check(surv['id'] == 0, f'survivor should be id 0, got {surv["id"]}')
    # Absorbed review state carried forward onto the survivor.
    check(surv.get('review_uid') == 'uid-keep', 'absorbed review_uid lost on merge')
    check(surv.get('expert_id', {}).get('mode') == 'EXPERT',
          'absorbed expert_id dropped on merge (data loss)')
    check(surv.get('reviews') and surv['reviews'][0]['reviewer'] == 'Dr X',
          'absorbed reviews[] dropped on merge (data loss)')
    check(not refused, f'no conflict expected, got refused={refused}')


def test_union_merge_pending_survivor_keeps_absorbed_accepted_state():
    # Survivor (lowest id) is PENDING; absorbed (higher id) is ACCEPTED expert.
    # The pending survivor must NOT downgrade the absorbed accepted state — the
    # accepted expert_id is promoted onto the survivor.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    surv_in = _mask(0, 'OFAV', a, status='pending')
    absorbed_in = _mask(1, 'OFAV', b, status='accepted')
    absorbed_in['expert_id'] = {'code': 'OFAV', 'mode': 'EXPERT', 'labeler': 'Dr Y'}
    masks = [surv_in, absorbed_in]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 1, f'expected 1 merge, got {merged}')
    surv = new_masks[0]
    check(surv.get('expert_id', {}).get('mode') == 'EXPERT',
          'pending survivor silently downgraded an accepted absorbed mask')


def test_union_merge_refuses_conflicting_expert_ids():
    # Two overlapping same-species masks with DIFFERENT accepted expert IDs —
    # merging would force one organism to claim two codes, so REFUSE + report.
    a = _rect_mask(10, 30, 10, 30)
    b = _rect_mask(20, 40, 20, 40)
    ma = _mask(0, 'OFAV', a, status='accepted')
    mb = _mask(1, 'OFAV', b, status='accepted')
    ma['expert_id'] = {'code': 'OFAV', 'mode': 'EXPERT', 'labeler': 'Dr A'}
    mb['expert_id'] = {'code': 'OANN', 'mode': 'EXPERT', 'labeler': 'Dr B'}
    masks = [ma, mb]
    new_masks, merged, refused = mask_ops.merge_overlapping_same_id(masks, H, W)
    check(merged == 0, f'conflicting expert IDs must not merge, got {merged}')
    check(len(new_masks) == 2, f'both conflicting masks must survive, got {len(new_masks)}')
    check(len(refused) == 1, f'expected 1 refused report, got {refused}')
    check(refused[0]['reason'] == 'conflicting accepted expert IDs',
          f'unexpected refusal reason: {refused[0]}')
    # Each keeps its own accepted code.
    by_id = {m['id']: m for m in new_masks}
    check(by_id[0]['expert_id']['code'] == 'OFAV', 'survivor code corrupted')
    check(by_id[1]['expert_id']['code'] == 'OANN', 'absorbed code corrupted')


# ── UNIT: _merge_imported_review_fields ─────────────────────────────
def test_field_merge_imports_expert_without_clobbering_edits():
    import app
    # Operator's in-memory mask: they REJECTED it and edited its geometry/area.
    current = [{
        'id': 7, 'review_uid': 'uidX', 'species': 'REVIEW', 'status': 'rejected',
        'area': 1234, 'rle': {'counts': [1], 'size': [H, W]},
        'refinement_clicks': [{'x': 1, 'y': 2, 'label': 0}],
    }]
    # Disk copy carries an accepted expert ID + reviews[] + different geometry.
    disk = [{
        'id': 7, 'review_uid': 'uidX', 'species': 'OFAV', 'name': 'Orbicella',
        'category': 'Coral', 'status': 'accepted', 'area': 9999,
        'rle': {'counts': [9], 'size': [H, W]},
        'expert_id': {'code': 'OFAV', 'mode': 'EXPERT', 'labeler': 'Dr. X'},
        'reviews': [{'reviewer': 'Dr. X', 'code': 'OFAV', 'confidence': 'high'}],
    }]
    merged, expert = app._merge_imported_review_fields(current, disk)
    cm = current[0]
    check(merged == 1, f'expected 1 merged, got {merged}')
    check(expert == 1, f'expected 1 expert, got {expert}')
    # Imported review fields merged in:
    check(cm['expert_id']['mode'] == 'EXPERT', 'expert_id not merged')
    check(cm.get('reviews') and cm['reviews'][0]['code'] == 'OFAV', 'reviews[] not merged')
    check(cm['species'] == 'OFAV', 'accepted species not promoted')
    # Operator-owned fields preserved (NOT clobbered):
    check(cm['status'] == 'rejected', 'operator status was clobbered')
    check(cm['area'] == 1234, 'operator geometry area was clobbered')
    check(cm['rle']['counts'] == [1], 'operator RLE was clobbered')
    check(cm['refinement_clicks'][0]['x'] == 1, 'operator refinement edits clobbered')


def test_field_merge_pending_no_expert_does_not_promote_species():
    import app
    # Pending review on disk (no accepted expert id) → reviews/review merge in,
    # but species is NOT promoted (no acceptance yet).
    current = [{'id': 3, 'review_uid': 'uidP', 'species': 'REVIEW', 'status': 'pending'}]
    disk = [{
        'id': 3, 'review_uid': 'uidP', 'species': 'REVIEW', 'review': True,
        'reviews': [{'reviewer': 'A', 'code': 'OFAV', 'confidence': ''}],
    }]
    merged, expert = app._merge_imported_review_fields(current, disk)
    check(merged == 1, f'expected 1 merged, got {merged}')
    check(expert == 0, f'expected 0 expert, got {expert}')
    check(current[0].get('reviews'), 'reviews[] not merged for pending')
    check(current[0]['species'] == 'REVIEW', 'species must stay REVIEW until accepted')


def test_field_merge_matches_by_id_when_no_uid():
    import app
    current = [{'id': 5, 'species': 'REVIEW', 'status': 'pending'}]
    disk = [{
        'id': 5, 'species': 'OFAV',
        'expert_id': {'code': 'OFAV', 'mode': 'EXPERT', 'labeler': 'Z'},
        'reviews': [{'reviewer': 'Z', 'code': 'OFAV'}],
    }]
    merged, expert = app._merge_imported_review_fields(current, disk)
    check(merged == 1, f'expected 1 merged by id, got {merged}')
    check(expert == 1, f'expected 1 expert, got {expert}')
    check(current[0]['species'] == 'OFAV', 'species not promoted on id match')


def test_field_merge_no_disk_match_is_noop():
    import app
    current = [{'id': 1, 'review_uid': 'nope', 'species': 'REVIEW', 'status': 'pending'}]
    disk = [{'id': 99, 'review_uid': 'other', 'expert_id': {'mode': 'EXPERT'}}]
    merged, expert = app._merge_imported_review_fields(current, disk)
    check(merged == 0 and expert == 0, 'unmatched masks must be a no-op')
    check(current[0]['species'] == 'REVIEW', 'unmatched mask must be untouched')


# ── STRUCTURAL: merge button wired to mergeAction() ─────────────────
def test_merge_button_uses_merge_action():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    check('function mergeAction(' in html, "mergeAction() defined")
    check(html.count('mergeAction(') >= 2, "M-key AND button both call mergeAction()")


# ── STRUCTURAL: s key triggers scan-this ────────────────────────────
def test_s_triggers_scan_this():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    block = html[html.index("case 's':"):html.index("case 's':") + 240]
    check("doExemplarScan('this')" in block, "plain s runs scan-this")
    check("saveCurrentMasks" in block, "Ctrl+S still saves")


# ── STRUCTURAL: scan-candidate review loop (Task 13) ────────────────
def test_scan_review_loop_present_and_keydown_delegates():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    # The three review-loop entry points plus the bbox zoom helper exist.
    check('function startScanReview(' in html, 'startScanReview() defined')
    check('function scanReviewAccept(' in html, 'scanReviewAccept() defined')
    check('function scanReviewReject(' in html, 'scanReviewReject() defined')
    check('function focusScanCandidate(' in html, 'focusScanCandidate() defined')
    check('function centerOnBBox(' in html, 'centerOnBBox() helper defined')
    # Keep persists via commit_mask; reject is client-only (never POSTs).
    check('/commit_mask' in html, 'scanReviewAccept must POST /commit_mask')
    # doExemplarScan drops into the loop after populating candidates.
    check(html.count('startScanReview(') >= 2,
          'doExemplarScan must call startScanReview() (defn + call site)')
    # The keydown handler delegates Enter/Delete to the loop BEFORE the normal
    # switch, guarded by scanReviewIdx >= 0, so a scan-review Enter/Del does NOT
    # also accept/reject a main-list mask.
    kd = html.index("document.addEventListener('keydown'")
    sw = html.index('switch (e.key)', kd)
    guard = html[kd:sw]
    check('if (scanReviewIdx >= 0)' in guard,
          'keydown must guard the review loop with scanReviewIdx >= 0 before the switch')
    check('scanReviewAccept()' in guard, 'Enter must delegate to scanReviewAccept() in review')
    check('scanReviewReject()' in guard, 'Delete/x must delegate to scanReviewReject() in review')


# ── COMBINED ANNOTATOR: manual-mode backend smoke ───────────────────
def test_manual_mode_configure_reference_points_and_provenance():
    """End-to-end manual-mode contract:
      1. configure() in MANUAL_ANNOTATE builds empty-mask seg_dicts that carry
         the FULL prompt-point set as reference_points (no SAM3 pass).
      2. /api/image/<fn> returns masks==[] and the reference points.
      3. An accepted manual_click mask + export_batch writes label_provenance.csv
         with a found_manual row.
    """
    import app
    check(app.cfg.MANUAL_ANNOTATE, 'MANUAL_ANNOTATE must be on for this test')

    work = tempfile.mkdtemp(prefix='combined_smoke_')
    input_dir = os.path.join(work, 'input')
    export_dir = os.path.join(work, 'export')
    ids_dir = os.path.join(input_dir, 'ids')
    raw_dir = os.path.join(input_dir, 'raw')
    os.makedirs(ids_dir)
    os.makedirs(raw_dir)

    fn = 'IMG_combined_001.jpeg'
    # A small raw jpeg the app can decode for dims + overlay rendering.
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[:] = (20, 60, 90)
    raw_path = os.path.join(raw_dir, fn)
    check(cv2.imwrite(raw_path, img), 'failed to write test jpeg')

    # Two prompt points (the FULL set must survive as reference_points).
    prompts = {
        fn: {
            'raw': os.path.join('raw', fn),
            'points': [
                {'label': 'A', 'species': 'DICT', 'name': 'Dictyota spp.',
                 'category': 'Macroalgae', 'x': 12.0, 'y': 18.0, 'point_type': 1},
                {'label': 'B', 'species': 'DICT', 'name': 'Dictyota spp.',
                 'category': 'Macroalgae', 'x': 30.0, 'y': 25.0, 'point_type': 1},
            ],
        }
    }
    with open(os.path.join(ids_dir, 'sam_click_prompts.json'), 'w') as f:
        import json as _json
        _json.dump(prompts, f)

    app.app.config['TESTING'] = True
    client = app.app.test_client()

    r = client.post('/api/configure', json={
        'input_dir': input_dir, 'export_dir': export_dir,
        'categories': ['Target species only'], 'reviewer': 'LO',
    })
    check(r.status_code == 200, f'/api/configure returned {r.status_code}: {r.get_data(as_text=True)}')
    cfg_js = r.get_json()
    check(cfg_js['phase'] == 'review', f"manual mode must go straight to review, got {cfg_js['phase']}")
    check(cfg_js['to_process'] == 0, f"manual mode builds no queue, got to_process={cfg_js['to_process']}")
    check(cfg_js['review_count'] == 1, f"empty manual frame must enter review, got {cfg_js['review_count']}")
    check('images' in cfg_js, f"configure response must include 'images' count, got {cfg_js}")
    check(cfg_js['images'] == 1, f"expected 1 image in configure response, got {cfg_js.get('images')}")
    check(app.session.get('reviewer') == 'LO', 'reviewer not captured from configure body')

    # /api/image/<fn>: empty masks + the two reference points.
    r = client.get(f'/api/image/{fn}')
    check(r.status_code == 200, f'/api/image returned {r.status_code}')
    ij = r.get_json()
    check(ij['masks'] == [], f"manual frame must start with no masks, got {ij['masks']}")
    check(len(ij['reference_points']) == 2, f"expected 2 reference points, got {len(ij['reference_points'])}")
    rp = ij['reference_points'][0]
    check(rp['species_code'] == 'DICT' and rp['label'] == 'A',
          f'reference point mapping wrong: {rp}')

    # Inject an ACCEPTED manual_click mask into the in-memory seg, then export.
    seg = app._find_segmentation(fn)
    check(seg is not None, 'seg_dict missing after configure')
    binary = np.zeros((40, 50), dtype=bool)
    binary[8:24, 6:30] = True
    md = build_mask_dict(
        mask_id=0, binary_mask=binary, score=0.95,
        point_info={'label': 'A', 'species': 'DICT', 'name': 'Dictyota spp.',
                    'category': 'Macroalgae', 'x': 12.0, 'y': 18.0},
        source_type='manual_click', simplify_epsilon=app.cfg.POLYGON_SIMPLIFY_EPSILON)
    check(md is not None, 'build_mask_dict returned None for the test mask')
    md['status'] = 'accepted'
    seg['masks'].append(md)

    r = client.post('/api/export_batch', json={})
    check(r.status_code == 200, f'/api/export_batch returned {r.status_code}: {r.get_data(as_text=True)}')

    # label_provenance.csv exists with a found_manual row for this image/species.
    csv_path = os.path.join(os.path.abspath(export_dir), 'label_provenance.csv')
    check(os.path.exists(csv_path), f'label_provenance.csv not written at {csv_path}')
    import csv as _csv
    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    match = [r for r in rows if r['basename'] == os.path.splitext(fn)[0] and r['label'] == 'DICT']
    check(len(match) == 1, f'expected one DICT provenance row, got {match}')
    check(match[0]['outcome'] == 'found_manual',
          f"manual_click accepted mask must be found_manual, got {match[0]['outcome']}")
    check(match[0]['source'] == app.cfg.PROVENANCE_SOURCE,
          f"source must be {app.cfg.PROVENANCE_SOURCE}, got {match[0]['source']}")
    check(match[0]['reviewer'] == 'LO', f"reviewer must be carried, got {match[0]['reviewer']}")
    # The in-JSON block was stamped too.
    check(seg.get('label_outcomes', {}).get('DICT', {}).get('outcome') == 'found_manual',
          'seg["label_outcomes"] block not stamped on export')


# ── STRUCTURAL: hide-rejected toggle (Task 14) ──────────────────────
def test_hide_rejected_default_on():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    check('id="chk-hide-rejected"' in html, "hide-rejected checkbox present")
    blk = html[html.index('id="chk-hide-rejected"') - 60:html.index('id="chk-hide-rejected"') + 60]
    check('checked' in blk, "hide-rejected default checked")
    check('hideRejected' in html, "renderMaskList consults hideRejected")
    # Verify the ROW-render path (order.map loop) applies the combined guard,
    # not just the summary-count filter.  A regression that only filters the
    # `visible` array but leaves the row loop unguarded would fail here.
    # Locate the function *definition* (not a call site) by searching for
    # "function renderMaskList" or "renderMaskList() {".
    func_marker = 'renderMaskList() {'
    func_start  = html.index(func_marker)
    render_body = html[func_start:func_start + 5000]
    check(
        "hideRejected && m.status === 'rejected'" in render_body,
        "renderMaskList row-render loop guards against rejected rows when hideRejected is on"
    )


# ── STRUCTURAL: scan-candidate overlay + double-commit guard (Task 13b) ─
def test_scan_candidate_overlay_and_commit_guard():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    # _committingCandidate guard variable is declared at module level.
    check('_committingCandidate' in html, '_committingCandidate guard variable not declared')
    # scanReviewAccept() rejects re-entry while a commit is in flight.
    accept_start = html.index('async function scanReviewAccept(')
    accept_end = html.index('\n}', accept_start) + 2
    accept_body = html[accept_start:accept_end]
    check('if (_committingCandidate) return' in accept_body,
          'scanReviewAccept must bail out when _committingCandidate is true')
    check('_committingCandidate = true' in accept_body,
          'scanReviewAccept must set _committingCandidate = true before the POST')
    check('_committingCandidate = false' in accept_body,
          'scanReviewAccept must reset _committingCandidate = false in finally')
    check('finally' in accept_body,
          'scanReviewAccept must use try/finally to guarantee _committingCandidate reset')
    # render() draws the scan-candidate overlay (references scanCandidates).
    render_start = html.index('\nfunction render()')
    render_end = html.index('\n}', render_start) + 2
    render_body = html[render_start:render_end]
    check('scanCandidates' in render_body,
          'render() must reference scanCandidates for the overlay pass')
    check('setLineDash' in render_body,
          'render() must use setLineDash for dashed candidate outlines')
    # The overlay resets the dash after drawing (so normal strokes are not dashed).
    check(render_body.count('setLineDash') >= 2,
          'render() must call setLineDash at least twice (set + reset)')


# ── STRUCTURAL: prominent lock-in hint (Task 16) ────────────────────
def test_lockin_hint_present():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    check('id="lockin-hint"' in html, "lock-in hint element present")
    check('double-click' in html.lower() and 'lock in' in html.lower(), "lock-in copy present")
    check('function updateLockinHint(' in html, "updateLockinHint() defined")
    # updateLockinHint must be called from render() and/or renderMaskList() —
    # at least two call sites so both code paths trigger it.
    check(html.count('updateLockinHint()') >= 2, "updateLockinHint() called from render/renderMaskList")


# ── STRUCTURAL: draw_edit sends rejected_ids (Task B2) ─────────────
def test_draw_edit_sends_rejected_ids():
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    i = html.index('/draw_edit')
    block = html[i:i + 400]
    check('rejected_ids: rejectedMaskIds()' in block,
          "draw_edit fetch body includes rejected_ids: rejectedMaskIds()")


# ── STRUCTURAL: recentlyDeletedIds feeds rejectedMaskIds (Task 17) ──
def test_recently_deleted_feeds_rejected_ids():
    # Delete-then-remask bug: after deleting a mask, a new overlapping mask in
    # its freed region was sometimes clipped to a sliver. Root cause is client
    # side — a wholesale `currentData.masks = data.masks` reload (or full
    # `currentData` reload) can restore a just-deleted mask's pre-reject status
    # before its reject PUT lands, so rejectedMaskIds() misses it and /add then
    # clips against it. Fix: track deletions in a module-level recentlyDeletedIds
    # Set, union it into rejectedMaskIds(), clear it on frame change.
    html = open(os.path.join(_SRC, 'templates', 'index.html')).read()
    check('recentlyDeletedIds' in html, 'recentlyDeletedIds tracked')

    # rejectedMaskIds() must union recentlyDeletedIds so a just-deleted mask is
    # always reported to /add until the frame changes.
    i = html.index('function rejectedMaskIds')
    rj = html[i:i + 400]
    check('recentlyDeletedIds' in rj, 'rejectedMaskIds unions recentlyDeletedIds')

    # Both delete paths must record the id into the set.
    rm = html[html.index('function rejectMask'):html.index('function rejectMask') + 600]
    check('recentlyDeletedIds.add' in rm, 'rejectMask adds to recentlyDeletedIds')
    dc = html[html.index('function deleteCheckedMasks'):html.index('function deleteCheckedMasks') + 900]
    check('recentlyDeletedIds.add' in dc, 'deleteCheckedMasks adds to recentlyDeletedIds')

    # selectImage must clear the set on frame load so deletions never leak across
    # frames (an id would otherwise wrongly free a region on a different image).
    si = html[html.index('function selectImage'):html.index('function selectImage') + 1600]
    check('recentlyDeletedIds.clear' in si, 'selectImage clears recentlyDeletedIds on frame change')


# ── main ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Step-5 Segment app tests')
    print('— smoke —')
    run(test_smoke_import_app)
    run(test_smoke_index_renders_200)
    run(test_smoke_status_exposes_batches)
    run(test_smoke_template_js_node_check)
    print('— unit: merge_overlapping_same_id —')
    run(test_union_merge_two_overlapping_same_species)
    run(test_union_merge_leaves_different_species_alone)
    run(test_union_merge_skips_non_overlapping_same_species)
    run(test_union_merge_ignores_rejected)
    run(test_union_merge_transitive_chain)
    run(test_union_merge_never_merges_two_review_masks)
    run(test_union_merge_never_merges_blank_or_question_placeholder)
    run(test_union_merge_absorbed_expert_id_and_reviews_survive)
    run(test_union_merge_pending_survivor_keeps_absorbed_accepted_state)
    run(test_union_merge_refuses_conflicting_expert_ids)
    print('— unit: _merge_imported_review_fields —')
    run(test_field_merge_imports_expert_without_clobbering_edits)
    run(test_field_merge_pending_no_expert_does_not_promote_species)
    run(test_field_merge_matches_by_id_when_no_uid)
    run(test_field_merge_no_disk_match_is_noop)
    print('— combined annotator: manual-mode backend —')
    run(test_manual_mode_configure_reference_points_and_provenance)
    print('— add/refine clip: client-authoritative rejected set —')
    run(test_blocking_masks_excludes_client_rejected_ids)
    print('— structural: merge button wired to mergeAction() —')
    run(test_merge_button_uses_merge_action)
    print('— structural: s key triggers scan-this —')
    run(test_s_triggers_scan_this)
    print('— structural: scan-candidate review loop (Task 13) —')
    run(test_scan_review_loop_present_and_keydown_delegates)
    print('— structural: scan-candidate overlay + commit guard (Task 13b) —')
    run(test_scan_candidate_overlay_and_commit_guard)
    print('— structural: hide-rejected toggle (Task 14) —')
    run(test_hide_rejected_default_on)
    print('— structural: lock-in hint (Task 16) —')
    run(test_lockin_hint_present)
    print('— structural: recentlyDeletedIds feeds rejectedMaskIds (Task 17) —')
    run(test_recently_deleted_feeds_rejected_ids)
    print('— structural: draw_edit sends rejected_ids (Task B2) —')
    run(test_draw_edit_sends_rejected_ids)

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass} passed, {n_fail} failed")
    if n_fail:
        for name, ok, err in _RESULTS:
            if not ok:
                print(f"\nFAIL {name}:\n{err}")
        sys.exit(1)
    sys.exit(0)
