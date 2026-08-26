"""Tests for the expert-review lock in TCRMPclip_editMasks.

Where this fits: editMasks fixes an EXISTING export's masks, but a mask that
is out with (or has returned from) outside taxonomic expert review must not
be editable here -- editing it would silently diverge from what the expert
looked at, and this app must never spawn a NEW expert-review round-trip
either. app.py's _is_expert_locked(mask) is the server-side guard on every
route that mutates a specific existing mask; this test seeds an export with
an ordinary mask and two locked masks (a REVIEW mask and an expert_id mask)
and proves the ordinary mask can still be relabeled while the locked ones are
refused and left unchanged. It also greps the template to confirm the
REVIEW-arming control was removed, not just hidden.

Also covers two mutate-existing-mask routes found unguarded by a reviewer:
accept_all (POST /api/image/<f>/accept_all), which must never flip a locked
PENDING mask to 'accepted', and merge_same_id (POST
/api/image/<f>/merge_same_id), which must never union a locked mask carrying
a real species code into a neighbor and remove it.

House harness style: standalone script, no pytest. Env vars are set BEFORE
`import app as A` (mirrors test_edit_boot.py). SAM3 loads only under
`__main__` in app.py, so the test client never touches the GPU; PUT
/api/image/<f>/masks (relabel/status) is SAM-free.

Run: env/bin/python scripts/TCRMPclip_editMasks/tests/test_expert_lock.py
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
_ROOT = tempfile.mkdtemp(prefix='expert_lock_test_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations')
os.makedirs(_SEG_DIR, exist_ok=True)

_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_RAW_DIR, exist_ok=True)

_FRAME_NAME = 'TCRMPX_clip_AAA_T118.jpeg'


def _make_image(path, w=300, h=100):
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


def _make_mask(mask_id, x0, species='AA', name='Acropora agaricites', category='Coral'):
    binary_mask = np.zeros((100, 300), dtype=bool)
    binary_mask[20:80, x0:x0 + 60] = True  # well above MIN_MASK_AREA_PX default guard
    point_info = {'label': chr(ord('M') + mask_id), 'species': species, 'name': name,
                  'category': category, 'x': x0 + 30, 'y': 50}
    m = mask_ops.build_mask_dict(
        mask_id=mask_id, binary_mask=binary_mask, score=0.9,
        point_info=point_info, source_type='auto',
    )
    assert m is not None, f'fixture mask {mask_id} must build (check MIN_MASK_AREA_PX)'
    m['status'] = 'accepted'
    return m


# (a) an ordinary PA mask -- editable
_mask_ordinary = _make_mask(0, 10, species='PA', name='Porites astreoides')

# (b) a REVIEW mask -- expert-locked via species=='REVIEW' + review=True
_mask_review = _make_mask(1, 100, species='REVIEW', name='Flagged for expert review',
                           category='Review')
_mask_review['review'] = True

# (c) a mask carrying expert_id={'mode': 'EXPERT'} -- expert-locked via expert_id
_mask_expert = _make_mask(2, 190, species='AA', name='Acropora agaricites')
_mask_expert['expert_id'] = {'mode': 'EXPERT', 'reviewer': 'expert@example.com'}

# (d) a PENDING mask locked via expert_id -- for accept_all: must stay pending,
# never flipped to accepted by the unguarded bulk-accept loop.
_mask_pending_locked = _make_mask(3, 260, species='MC', name='Montastraea cavernosa')
_mask_pending_locked['status'] = 'pending'
_mask_pending_locked['expert_id'] = {'mode': 'EXPERT', 'reviewer': 'expert@example.com'}

# (e) an ordinary same-species ('AA') mask placed to physically OVERLAP the
# locked expert_id mask (2, x0=190..250) -- for merge_same_id: this pair would
# be unioned (same real species code, overlapping footprints) by an unguarded
# route; the locked member must be held out of the merge instead.
_mask_overlap_locked = _make_mask(4, 210, species='AA', name='Acropora agaricites')

# Separate frame used only by the merge_same_id test so its mutation of
# seg['masks'] cannot disturb the accept_all frame state above.
_MERGE_FRAME_NAME = 'TCRMPX_clip_AAA_T119.jpeg'
_make_image(os.path.join(_RAW_DIR, _MERGE_FRAME_NAME))

_seg_dict = {
    _FRAME_NAME: {
        'image_path': _FRAME_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _FRAME_NAME),
        'image_width': 300,
        'image_height': 100,
        'masks': [_mask_ordinary, _mask_review, _mask_expert, _mask_pending_locked],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
    },
    _MERGE_FRAME_NAME: {
        'image_path': _MERGE_FRAME_NAME,
        'image_path_abs': os.path.join(_RAW_DIR, _MERGE_FRAME_NAME),
        'image_width': 300,
        'image_height': 100,
        'masks': [
            _make_mask(2, 190, species='AA', name='Acropora agaricites'),
            _make_mask(4, 210, species='AA', name='Acropora agaricites'),
        ],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
    },
}
# The merge-frame's mask 2 must also carry the expert_id lock (build a fresh
# dict per frame so the two frames' mask-2 objects are independent).
_seg_dict[_MERGE_FRAME_NAME]['masks'][0]['expert_id'] = {
    'mode': 'EXPERT', 'reviewer': 'expert@example.com'}

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


# -- _is_expert_locked unit checks (server-side helper) ----------------------
check(A._is_expert_locked(_mask_ordinary) is False,
      "_is_expert_locked(ordinary PA mask) is False")
check(A._is_expert_locked(_mask_review) is True,
      "_is_expert_locked(REVIEW mask) is True")
check(A._is_expert_locked(_mask_expert) is True,
      "_is_expert_locked(expert_id mask) is True")
check(A._is_expert_locked({'review_uid': 'abc123'}) is True,
      "_is_expert_locked(review_uid-only mask) is True")
check(A._is_expert_locked({'reviews': [{'by': 'x'}]}) is True,
      "_is_expert_locked(non-empty reviews[] mask) is True")
check(A._is_expert_locked({'reviews': []}) is False,
      "_is_expert_locked(empty reviews[] mask) is False")
check(A._is_expert_locked({}) is False,
      "_is_expert_locked({}) is False")

A.app.config['TESTING'] = True

with A.app.test_client() as c:
    # Boot via the edit-mode resume path (same as test_edit_boot.py).
    r = c.post('/api/resume', json={'export_dir': _EXPORT_DIR})
    check(r.status_code == 200, f"POST /api/resume -> 200 (got {r.status_code}, body={r.data!r})")

    r_img = c.get(f'/api/image/{_FRAME_NAME}')
    check(r_img.status_code == 200, f"GET /api/image/<frame> -> 200 (got {r_img.status_code})")
    img_data = json.loads(r_img.data)
    check(len(img_data.get('masks', [])) == 4,
          f"frame has exactly 4 seeded masks (got {len(img_data.get('masks', []))})")

    # -- Relabeling the ORDINARY mask SUCCEEDS -------------------------------
    r_ok = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'relabel': {'0': {'species': 'OFRA', 'name': 'Orbicella franksi', 'category': 'Coral'}},
    })
    check(r_ok.status_code == 200, f"PUT masks (ordinary relabel) -> 200 (got {r_ok.status_code})")
    ok_body = json.loads(r_ok.data)
    check(ok_body.get('ok') is True,
          f"ordinary mask relabel reports ok=True (got {ok_body})")
    check(not ok_body.get('locked'),
          f"ordinary mask relabel reports no lock (got {ok_body})")

    r_check = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after = {m['id']: m for m in json.loads(r_check.data)['masks']}
    check(masks_after[0]['species'] == 'OFRA',
          f"ordinary mask species actually changed to OFRA (got {masks_after[0]['species']!r})")

    # -- Relabeling the REVIEW mask is REFUSED (locked, unchanged) ----------
    r_locked1 = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'relabel': {'1': {'species': 'PA', 'name': 'Porites astreoides', 'category': 'Coral'}},
    })
    locked1_body = json.loads(r_locked1.data)
    check(locked1_body.get('ok') is False,
          f"REVIEW mask relabel reports ok=False (got {locked1_body})")
    check(locked1_body.get('locked') is True,
          f"REVIEW mask relabel reports locked=True (got {locked1_body})")
    check(1 in (locked1_body.get('locked_ids') or []),
          f"REVIEW mask id (1) is in locked_ids (got {locked1_body.get('locked_ids')})")

    r_check2 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after2 = {m['id']: m for m in json.loads(r_check2.data)['masks']}
    check(masks_after2[1]['species'] == 'REVIEW',
          f"REVIEW mask species unchanged (got {masks_after2[1]['species']!r})")

    # -- Deleting (status -> rejected) the REVIEW mask is REFUSED -----------
    r_del1 = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'updates': {'1': 'rejected'},
    })
    del1_body = json.loads(r_del1.data)
    check(del1_body.get('ok') is False,
          f"REVIEW mask delete-via-reject reports ok=False (got {del1_body})")
    check(del1_body.get('locked') is True,
          f"REVIEW mask delete-via-reject reports locked=True (got {del1_body})")

    r_check3 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after3 = {m['id']: m for m in json.loads(r_check3.data)['masks']}
    check(masks_after3[1]['status'] == 'accepted',
          f"REVIEW mask status unchanged, still 'accepted' (got {masks_after3[1]['status']!r})")

    # -- Relabeling the expert_id mask is REFUSED (locked, unchanged) -------
    r_locked2 = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'relabel': {'2': {'species': 'MC', 'name': 'Montastraea cavernosa', 'category': 'Coral'}},
    })
    locked2_body = json.loads(r_locked2.data)
    check(locked2_body.get('ok') is False,
          f"expert_id mask relabel reports ok=False (got {locked2_body})")
    check(locked2_body.get('locked') is True,
          f"expert_id mask relabel reports locked=True (got {locked2_body})")
    check(2 in (locked2_body.get('locked_ids') or []),
          f"expert_id mask id (2) is in locked_ids (got {locked2_body.get('locked_ids')})")

    # -- Deleting (status -> rejected) the expert_id mask is REFUSED --------
    r_del2 = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'updates': {'2': 'rejected'},
    })
    del2_body = json.loads(r_del2.data)
    check(del2_body.get('ok') is False,
          f"expert_id mask delete-via-reject reports ok=False (got {del2_body})")

    r_check4 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after4 = {m['id']: m for m in json.loads(r_check4.data)['masks']}
    check(masks_after4[2]['species'] == 'AA',
          f"expert_id mask species unchanged (got {masks_after4[2]['species']!r})")
    check(masks_after4[2]['status'] == 'accepted',
          f"expert_id mask status unchanged, still 'accepted' (got {masks_after4[2]['status']!r})")
    check(masks_after4[2].get('expert_id', {}).get('mode') == 'EXPERT',
          f"expert_id mask still carries expert_id.mode == 'EXPERT' (got {masks_after4[2].get('expert_id')})")

    # -- A mixed batch (one ordinary + one locked) still applies the ordinary
    #    edit and reports the locked one refused. --------------------------
    r_mixed = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'relabel': {
            '0': {'species': 'MC', 'name': 'Montastraea cavernosa', 'category': 'Coral'},
            '1': {'species': 'PA', 'name': 'Porites astreoides', 'category': 'Coral'},
        },
    })
    mixed_body = json.loads(r_mixed.data)
    check(mixed_body.get('ok') is True,
          f"mixed batch (1 ordinary + 1 locked) still reports ok=True (got {mixed_body})")
    check(1 in (mixed_body.get('locked_ids') or []),
          f"mixed batch reports the locked mask id in locked_ids (got {mixed_body.get('locked_ids')})")
    r_check5 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after5 = {m['id']: m for m in json.loads(r_check5.data)['masks']}
    check(masks_after5[0]['species'] == 'MC',
          f"mixed batch: ordinary mask DID change to MC (got {masks_after5[0]['species']!r})")
    check(masks_after5[1]['species'] == 'REVIEW',
          f"mixed batch: locked mask did NOT change (got {masks_after5[1]['species']!r})")

    # -- A relabel that would ARM review on the ordinary mask is refused ----
    r_arm = c.put(f'/api/image/{_FRAME_NAME}/masks', json={
        'relabel': {'0': {'species': 'REVIEW', 'review': True}},
    })
    arm_body = json.loads(r_arm.data)
    check(arm_body.get('ok') is False,
          f"relabel arming REVIEW on an ordinary mask is refused (got {arm_body})")
    check(arm_body.get('locked') is True,
          f"relabel arming REVIEW on an ordinary mask reports locked=True (got {arm_body})")
    r_check6 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after6 = {m['id']: m for m in json.loads(r_check6.data)['masks']}
    check(masks_after6[0]['species'] == 'MC',
          f"ordinary mask species unchanged by the refused arm-REVIEW attempt (got {masks_after6[0]['species']!r})")

    # -- accept_all does NOT accept a locked PENDING mask --------------------
    # Mask 3 is pending and expert_id-locked. An unguarded accept_all would
    # flip it straight to 'accepted' with only the _needs_species check;
    # _is_expert_locked must skip it and report it in locked_ids instead.
    r_accept_all = c.post(f'/api/image/{_FRAME_NAME}/accept_all')
    check(r_accept_all.status_code == 200,
          f"POST accept_all -> 200 (got {r_accept_all.status_code})")
    accept_all_body = json.loads(r_accept_all.data)
    check(accept_all_body.get('locked') is True,
          f"accept_all reports locked=True when a pending mask is locked (got {accept_all_body})")
    check(3 in (accept_all_body.get('locked_ids') or []),
          f"accept_all reports locked mask id (3) in locked_ids (got {accept_all_body.get('locked_ids')})")

    r_check7 = c.get(f'/api/image/{_FRAME_NAME}')
    masks_after7 = {m['id']: m for m in json.loads(r_check7.data)['masks']}
    check(masks_after7[3]['status'] == 'pending',
          f"locked mask (3) status is STILL 'pending' after accept_all (got {masks_after7[3]['status']!r})")

    # -- merge_same_id does not mutate/remove a locked mask -------------------
    # On the dedicated merge frame: mask 2 is expert_id-locked with species
    # 'AA', mask 4 is an ordinary 'AA' mask that physically OVERLAPS mask 2.
    # An unguarded merge_overlapping_same_id would union them (same real
    # species code, overlapping footprints), absorbing one and removing it.
    # The locked mask must come out completely unchanged and mask 4 must
    # survive as its own mask (not silently absorbed/removed).
    r_before_merge = c.get(f'/api/image/{_MERGE_FRAME_NAME}')
    merge_masks_before = {m['id']: m for m in json.loads(r_before_merge.data)['masks']}
    check(set(merge_masks_before.keys()) == {2, 4},
          f"merge frame starts with exactly masks 2 and 4 (got {sorted(merge_masks_before.keys())})")

    r_merge = c.post(f'/api/image/{_MERGE_FRAME_NAME}/merge_same_id')
    check(r_merge.status_code == 200,
          f"POST merge_same_id -> 200 (got {r_merge.status_code})")
    merge_body = json.loads(r_merge.data)

    r_after_merge = c.get(f'/api/image/{_MERGE_FRAME_NAME}')
    merge_masks_after = {m['id']: m for m in json.loads(r_after_merge.data)['masks']}
    check(2 in merge_masks_after,
          f"locked mask (2) was NOT removed/absorbed by merge_same_id (got ids {sorted(merge_masks_after.keys())})")
    check(merge_masks_after.get(2, {}).get('rle') == merge_masks_before[2]['rle'],
          "locked mask (2) RLE geometry is byte-identical after merge_same_id (not unioned into)")
    check(merge_masks_after.get(2, {}).get('expert_id', {}).get('mode') == 'EXPERT',
          f"locked mask (2) still carries expert_id.mode == 'EXPERT' (got {merge_masks_after.get(2, {}).get('expert_id')})")
    check(4 in merge_masks_after,
          f"non-locked overlapping mask (4) still exists, not silently absorbed away (got ids {sorted(merge_masks_after.keys())})")
    check(merge_body.get('merged', 0) == 0,
          f"merge_same_id reports 0 masks merged (the only overlapping pair is locked) (got {merge_body.get('merged')})")


# -- Template no longer contains the REVIEW-arming control -------------------
_TEMPLATE_PATH = os.path.join(SRC, 'templates', 'index.html')
with open(_TEMPLATE_PATH) as f:
    _template_src = f.read()

check('qa-btn-review' not in _template_src,
      "template has no qa-btn-review (quickadd REVIEW-arm button) element")
check('assignNewMaskReview' not in _template_src,
      "template has no assignNewMaskReview function/call (REVIEW popup option)")
check('toggleQuickReview' not in _template_src,
      "template has no toggleQuickReview function/call")
check('armReview(' not in _template_src,
      "template has no armReview(...) call (function itself was removed)")
check('persistMaskReview' not in _template_src,
      "template has no persistMaskReview function (dead REVIEW-persist path)")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
