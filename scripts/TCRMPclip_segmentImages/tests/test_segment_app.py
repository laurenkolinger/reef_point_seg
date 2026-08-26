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

import mask_ops
from mask_ops import rle_encode

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


def test_smoke_index_renders_200():
    import app
    app.app.config['TESTING'] = True
    client = app.app.test_client()
    r = client.get('/')
    check(r.status_code == 200, f'GET / returned {r.status_code}')
    body = r.get_data(as_text=True)
    # New UI affordances are present in the rendered page.
    check('Merge same-ID masks' in body, 'merge-same-id button missing from page')
    check('Reload IDs' in body, 'reload-ids button missing from page')
    check('return to main menu' in body, 'done-return button missing from page')
    check('remove / shrink region' in body, 'refine hint missing from page')


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

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass} passed, {n_fail} failed")
    if n_fail:
        for name, ok, err in _RESULTS:
            if not ok:
                print(f"\nFAIL {name}:\n{err}")
        sys.exit(1)
    sys.exit(0)
