"""Server-side refusals that lock expert-review flagging in Segment Images.

Where this fits: Lauren Olinger locked expert-review flagging across every
annotator on 2026-09-06. No mask may GAIN a review tag from now on, in any
spelling, from any route. Segment Images is archived from the orchestrator
sidebar but is still launchable through POST /api/step/5/run, and it writes
into the same segmentation files and the same mask registry as the named
annotators, so "all usages" covers it.

A review tag has two independent arms, and the app's own read predicate joins
them with OR: the boolean mask['review'], and the species string 'REVIEW'
(with mask['species_code'] as the fallback the predicate reads when 'species'
is absent). This app never had any pin at all: POST /add read the species
straight out of the request body, and the batch segmentation pass stamped both
arms onto any mask built from a REVIEW prompt. This suite proves both arms are
now closed on every route of this app that can write a tag, and that nothing
which reads an existing tag moved.

Two entry points the named annotators have do not exist here, and the suite
asserts their absence rather than assuming it: this app has no /add_box route,
and its PUT /masks carries statuses only, with no relabel field.

House harness style: standalone script, no pytest, named test functions run
through run(), PASS/FAIL per case, nonzero exit on any failure. Env vars are
set BEFORE `import app as A`. SAM3 loads only under __main__ in app.py, so the
test client never touches the GPU: a fake engine stands in for POST /add.

Run: env/bin/python scripts/TCRMPclip_segmentImages/tests/test_review_lock_routes.py
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import contextlib
import copy
import inspect
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import datetime

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
sys.path.insert(0, _SRC)

# ── fixture tree, built BEFORE importing app so config picks the env up ──
_ROOT = tempfile.mkdtemp(prefix='review_lock_si_')
_EXPORT_DIR = os.path.join(_ROOT, 'export')
_YEAR = 2026
_SEG_DIR = os.path.join(_EXPORT_DIR, 'segmentations', str(_YEAR))
_RAW_DIR = os.path.join(_ROOT, 'raw')
os.makedirs(_SEG_DIR, exist_ok=True)
os.makedirs(_RAW_DIR, exist_ok=True)

IMG_W, IMG_H = 300, 120
FRAME = 'TCRMPX_clip_AAA_T118.jpeg'
BATCH_FRAME = 'TCRMPX_clip_AAA_T200.jpeg'


def _make_image(path, w=IMG_W, h=IMG_H):
    """Write a small solid JPEG so the app can open a real image file."""
    Image.new('RGB', (w, h), (10, 20, 30)).save(path, 'JPEG', quality=90)


_make_image(os.path.join(_RAW_DIR, FRAME))
_make_image(os.path.join(_RAW_DIR, BATCH_FRAME))

os.environ['TCRMP_EXPORT_DIR'] = _EXPORT_DIR
os.environ['TCRMP_MANUAL_ANNOTATE'] = '1'
os.environ['TCRMP_REVIEWER'] = 'LO'
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA,OFRA')
# A registry row must never reach the real inprocess/_mask_registry/manifest.csv
# from a fixture. run_all_tests.sh sets this for the whole board; setdefault
# keeps a hand run just as safe.
os.environ.setdefault('TCRMP_MASK_REGISTRY_DIR',
                      os.path.join(_ROOT, 'mask_registry'))
os.environ.setdefault('TCRMP_CLIP_INDEX_DIR', os.path.join(_ROOT, 'clip_index'))

import config as cfg  # noqa: E402
import mask_ops  # noqa: E402


def _rect(x0, x1, y0=20, y1=100):
    """A filled rectangle well above the app's minimum mask area."""
    binary = np.zeros((IMG_H, IMG_W), dtype=bool)
    binary[y0:y1, x0:x1] = True
    return binary


def _seed_mask(mask_id, x0, species, name, category='Coral', status='accepted'):
    """Build one stored mask the way the app itself builds one."""
    point_info = {'label': f'M{mask_id}', 'species': species, 'name': name,
                  'category': category, 'x': x0 + 30, 'y': 60}
    m = mask_ops.build_mask_dict(mask_id=mask_id, binary_mask=_rect(x0, x0 + 60),
                                 score=0.9, point_info=point_info,
                                 source_type='auto')
    assert m is not None, f'fixture mask {mask_id} must build'
    m['status'] = status
    # build_mask_dict is under the same 2026-09-06 lock, so it will not store a
    # REVIEW species. A mask flagged BEFORE the lock is what this fixture needs,
    # so the pre-lock fields are stamped on here directly.
    m['species'] = species
    m['name'] = name
    m['category'] = category
    return m


_ORDINARY = _seed_mask(0, 10, 'PA', 'Porites astreoides')
_FLAGGED = _seed_mask(1, 100, 'REVIEW', 'Flagged for expert review',
                      category='Review')
_FLAGGED['review'] = True
_FLAGGED['review_uid'] = 'uid-existing-0001'
_FLAGGED['reviews'] = [{'reviewer': 'Dr A', 'code': 'OFAV', 'confidence': 'high'}]

_SEED = {
    FRAME: {
        'image_path': FRAME,
        'image_path_abs': os.path.join(_RAW_DIR, FRAME),
        'image_width': IMG_W,
        'image_height': IMG_H,
        'masks': [_ORDINARY, _FLAGGED],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': True,
        'exported': False,
    },
}
with open(os.path.join(_SEG_DIR, 'segmentations.json'), 'w') as f:
    json.dump(_SEED, f)

import app as A  # noqa: E402

# ── tiny test harness (house style: test_segment_app.py) ────────────────
_RESULTS = []


def check(cond, msg):
    """Fail the current test with msg when cond is falsy."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, record PASS or FAIL, never abort the suite."""
    try:
        fn()
        _RESULTS.append((fn.__name__, True))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False))
        print(f"  FAIL {fn.__name__}: {e}")
        print(traceback.format_exc())


# ── fake SAM3 engine: POST /add needs a mask, not a model ───────────────
class FakeEngine:
    """Stands in for the SAM3 engine: returns a rectangle for any prompt."""

    _current_image_path = 'x'

    def set_image(self, path):
        """Report the fixture image size without opening a model."""
        return (IMG_W, IMG_H)

    def segment_point(self, x, y, label=1):
        """Return a small rectangle centred on the click."""
        return {'mask': _rect(int(x) - 20, int(x) + 20, int(y) - 20, int(y) + 20),
                'score': 0.9, 'bbox': None}


A.app.config['TESTING'] = True
_CLIENT = A.app.test_client()
_BOOT = _CLIENT.post('/api/resume', json={'export_dir': _EXPORT_DIR})
assert _BOOT.status_code == 200, f'fixture resume failed: {_BOOT.data!r}'
A.session['sam_engine'] = FakeEngine()
A._ensure_image_embedded = lambda filename: True

# Every spelling the refusal must catch, as (body fragment, why) pairs. The
# platform coerces the boolean with bool(), so 1 and the string "no" are both
# truthy; the species string is compared case- and whitespace-insensitively;
# species_code is read independently of species.
REVIEW_SPELLINGS = [
    ({'review': True}, 'review boolean True'),
    ({'review': 1}, 'review integer 1'),
    ({'review': 'no'}, 'review truthy string "no"'),
    ({'species': 'REVIEW'}, 'species REVIEW, no boolean'),
    ({'species': ' review '}, 'species " review " (lower case, padded)'),
    ({'species': '', 'species_code': 'REVIEW'}, 'species_code REVIEW with an empty species'),
    ({'category': 'Review'}, 'category Review, no species and no boolean'),
]


def _masks():
    """The live in-memory mask list for the fixture frame."""
    return A.session['segmentations_by_year'][_YEAR][FRAME]['masks']


def _by_id(mask_id):
    """The live in-memory mask with this id, or None."""
    for m in _masks():
        if m.get('id') == mask_id:
            return m
    return None


def _refusal_is_well_formed(response, where):
    """Assert one response is the shared HTTP 409 review-locked refusal."""
    body = response.get_json()
    check(response.status_code == 409,
          f'{where}: expected 409, got {response.status_code} {body}')
    check(body.get('ok') is False, f'{where}: ok must be False, got {body}')
    check(body.get('review_locked') is True,
          f'{where}: review_locked must be True, got {body}')
    check(body.get('error') == mask_ops.REVIEW_LOCKED_MSG,
          f'{where}: error must be the shared message, got {body.get("error")!r}')


# ── refusals ────────────────────────────────────────────────────────────
def test_add_refuses_every_review_spelling():
    before = len(_masks())
    for fragment, why in REVIEW_SPELLINGS:
        body = {'click_x': 150, 'click_y': 60, 'label': 'M'}
        body.update(fragment)
        r = _CLIENT.post(f'/api/image/{FRAME}/add', json=body)
        _refusal_is_well_formed(r, f'/add with {why}')
    check(len(_masks()) == before,
          f'a refused /add must create no mask (was {before}, now {len(_masks())})')


def test_box_add_refuses_every_review_spelling():
    # This app has no box-add route, so there is no second add path to tag
    # through. Asserted rather than assumed: if a box add is ever ported over
    # from the named annotators, this case goes red and the refusal must be
    # ported with it.
    check(not hasattr(A, 'add_mask_box'),
          'Segment Images has no add_mask_box handler; a new one needs the '
          'same refusal as /add before it ships')
    r = _CLIENT.post(f'/api/image/{FRAME}/add_box',
                     json={'box': [200, 20, 260, 90], 'review': True})
    check(r.status_code in (404, 405),
          f'/add_box is not a route on this app, got {r.status_code}')


def test_species_string_alone_is_refused():
    # The named annotator Edit Masks pinned the BOOLEAN arm of the tag on its
    # add routes in 2026-07 and left the species string open. This app never
    # pinned either arm, so both spellings reached the stored mask. The gap the
    # species string opens is asserted with the app's own read predicate: a
    # mask carrying nothing but species 'REVIEW' reads as flagged, so a
    # boolean-only pin would not have stopped it here either.
    stored_under_a_boolean_only_pin = {'id': 99, 'species': 'REVIEW', 'review': False}
    check(A._is_review(stored_under_a_boolean_only_pin) is True,
          'a mask carrying only species REVIEW reads as flagged, so pinning '
          'the boolean alone cannot close the tag')
    check(A._is_review({'species': '', 'species_code': 'REVIEW'}) is False,
          'the read predicate MISSES a species_code of REVIEW beside an empty '
          'species, because dict.get returns the stored empty string rather '
          'than falling through to species_code. That is exactly why the '
          'refusal tests the two keys independently instead of reusing the '
          'read predicate, and why the read predicate itself is left alone')

    before = len(_masks())
    r = _CLIENT.post(f'/api/image/{FRAME}/add',
                     json={'click_x': 150, 'click_y': 60, 'species': 'REVIEW'})
    _refusal_is_well_formed(r, '/add with the species string alone')
    check(len(_masks()) == before, 'the species-only add created no mask')


def test_add_refuses_a_category_only_payload():
    # The tag has a third spelling: the mask's category set to 'Review'. It is
    # not a real category (config.py ships 'Target species only' and nothing
    # else) and across the segmentation files under inprocess/ it pairs one for
    # one with the flag, so it is the same tag written a third way. POST /add
    # reads the category straight out of the body into point_info, so before
    # the refusal caught this spelling the category rode into the stored mask
    # verbatim. This app has no box add and no relabel field, so /add is the
    # only route the spelling could arrive on.
    before = len(_masks())
    for spelling in ('Review', 'REVIEW', ' review ', 'ReViEw'):
        r = _CLIENT.post(f'/api/image/{FRAME}/add',
                         json={'click_x': 150, 'click_y': 60, 'label': 'M',
                               'category': spelling})
        _refusal_is_well_formed(r, f'/add with category {spelling!r} alone')
    check(len(_masks()) == before,
          f'no category-only add created a mask (was {before}, now {len(_masks())})')

    r = _CLIENT.post(f'/api/image/{FRAME}/add',
                     json={'click_x': 150, 'click_y': 60, 'label': 'M',
                           'species': 'PA', 'name': 'Porites astreoides',
                           'category': 'Coral'})
    body = r.get_json()
    check(r.status_code == 200 and body.get('ok') is True,
          f'a real category still creates a mask, got {r.status_code} {body}')
    check(body['mask']['category'] == 'Coral',
          f'and keeps that category, got {body["mask"]["category"]!r}')
    _by_id(body['mask']['id'])['status'] = 'rejected'  # keep the frame tidy


def test_relabel_refuses_adding_a_tag():
    # PUT /masks on this app carries statuses only: there is no relabel field,
    # so a relabel body is inert and cannot write either arm of the tag. The
    # status half of the same call must keep working.
    source = inspect.getsource(A.update_masks)
    body = source.replace(A.update_masks.__doc__ or '', '')
    check('relabel' not in body,
          'PUT /masks reads no relabel field, so it cannot write a species or '
          'a review flag; a relabel field added here needs the same refusal '
          'POST /add carries')
    before = copy.deepcopy(_by_id(0))
    for fragment, why in REVIEW_SPELLINGS:
        r = _CLIENT.put(f'/api/image/{FRAME}/masks',
                        json={'relabel': {'0': dict(fragment)}})
        check(r.status_code == 200,
              f'relabel body with {why}: route still answers, got {r.status_code}')
        after = _by_id(0)
        check(after['species'] == before['species'],
              f'relabel body with {why}: species unchanged, got {after["species"]!r}')
        check(A._is_review(after) is False,
              f'relabel body with {why}: no tag was written, got {after}')


def test_batch_pass_never_mints_a_flagged_mask():
    # The batch segmentation pass takes its prompts from a file, not a request
    # body, so no request-level gate can see it. A prompt carrying either arm
    # of the tag must still produce a real mask, born untagged.
    item = {'filename': BATCH_FRAME, 'year': _YEAR,
            'image_path': os.path.join(_RAW_DIR, BATCH_FRAME),
            'points': [{'x': 60, 'y': 60, 'species': 'REVIEW', 'review': True,
                        'label': 'A'},
                       {'x': 200, 'y': 60, 'species': 'PA', 'label': 'B'}]}
    A._process_single_image(FakeEngine(), item, _EXPORT_DIR)
    produced = A.session['segmentations_by_year'][_YEAR][BATCH_FRAME]['masks']
    check(len(produced) == 2,
          f'both prompts still produce a mask, got {len(produced)}')
    flagged = [m for m in produced if A._is_review(m)]
    check(not flagged,
          f'no batch mask is born flagged, got {[m["species"] for m in flagged]}')


def test_refusal_answers_409_with_the_shared_message():
    r = _CLIENT.post(f'/api/image/{FRAME}/add',
                     json={'click_x': 150, 'click_y': 60, 'review': True})
    _refusal_is_well_formed(r, 'the shared refusal shape')
    body = r.get_json()
    check(set(body.keys()) == {'ok', 'review_locked', 'error'},
          f'the refusal carries exactly ok, review_locked and error, got {sorted(body)}')
    check(mask_ops.REVIEW_LOCKED_MSG ==
          'Flagging for expert review is temporarily unavailable. VICARIUS is '
          'accepting no new review tags.',
          f'the shared message text is the approved copy, got '
          f'{mask_ops.REVIEW_LOCKED_MSG!r}')


# ── nothing else moved ──────────────────────────────────────────────────
def test_ordinary_labelling_is_untouched():
    before = len(_masks())
    r = _CLIENT.post(f'/api/image/{FRAME}/add',
                     json={'click_x': 250, 'click_y': 60, 'species': 'OFRA',
                           'name': 'Orbicella franksi', 'category': 'Coral',
                           'label': 'M'})
    body = r.get_json()
    check(r.status_code == 200 and body.get('ok') is True,
          f'an ordinary /add still succeeds, got {r.status_code} {body}')
    check(body['mask']['species'] == 'OFRA',
          f'the new mask carries its species, got {body["mask"]["species"]!r}')
    new_id = body['mask']['id']
    check(len(_masks()) == before + 1, 'the ordinary add is stored')

    r = _CLIENT.put(f'/api/image/{FRAME}/masks',
                    json={'updates': {str(new_id): 'accepted'}})
    check(r.status_code == 200 and r.get_json().get('ok') is True,
          f'a status update still succeeds, got {r.status_code} {r.get_json()}')
    check(_by_id(new_id)['status'] == 'accepted',
          f'the status update was applied, got {_by_id(new_id)["status"]!r}')
    _by_id(new_id)['status'] = 'rejected'  # keep the frame tidy


def test_existing_flagged_mask_still_reads_back():
    r = _CLIENT.get(f'/api/image/{FRAME}')
    body = r.get_json()
    check(r.status_code == 200, f'GET the frame -> 200, got {r.status_code}')
    wire = [m for m in body['masks'] if m['id'] == 1]
    check(len(wire) == 1, 'the pre-existing flagged mask is still served')
    check(wire[0]['species'] == 'REVIEW',
          f'it still reads as REVIEW, got {wire[0]["species"]!r}')
    check(wire[0].get('review') is True,
          f'it still carries the review boolean, got {wire[0].get("review")!r}')
    check(wire[0].get('review_uid') == 'uid-existing-0001',
          'its review_uid still travels to the client')
    check(A._is_review(_by_id(1)) is True,
          'the read predicate still calls it flagged')

    # A status change on a flagged mask is not a tag write and still works:
    # this app is where an operator accepts or rejects the backlog.
    r = _CLIENT.put(f'/api/image/{FRAME}/masks', json={'updates': {'1': 'accepted'}})
    check(r.status_code == 200 and _by_id(1)['status'] == 'accepted',
          f'a flagged mask can still be accepted, got {r.status_code}')
    check(A._is_review(_by_id(1)) is True,
          'and it is still flagged afterwards')


def test_disk_read_back_still_restores_an_existing_tag():
    # Deliberately exempt from the lock: restoring a tag that is already on
    # disk is preservation, and it is how an expert identification comes home.
    current = [{'id': 5, 'review_uid': 'uid-round-trip', 'species': 'REVIEW',
                'status': 'pending'}]
    disk = [{'id': 5, 'review_uid': 'uid-round-trip', 'species': 'OFAV',
             'name': 'Orbicella faveolata', 'category': 'Coral', 'review': True,
             'expert_id': {'mode': 'EXPERT', 'code': 'OFAV'},
             'reviews': [{'reviewer': 'Dr X', 'code': 'OFAV', 'confidence': 'high'}]}]
    merged, expert = A._merge_imported_review_fields(current, disk)
    check(merged == 1, f'the disk read-back still merges, got {merged}')
    check(expert == 1, f'the expert-accepted id is still counted, got {expert}')
    check(current[0].get('review') is True,
          f'the review boolean is still restored from disk, got '
          f'{current[0].get("review")!r}')
    check(current[0].get('reviews'), 'reviews[] is still restored from disk')
    check(current[0]['species'] == 'OFAV',
          f'the expert-accepted species is still promoted, got '
          f'{current[0]["species"]!r}')

    # An in-memory mask with no tag still gains one from its disk twin: that is
    # the documented exemption, not a hole this change may close.
    plain = [{'id': 6, 'species': 'PA', 'status': 'pending'}]
    disk_flagged = [{'id': 6, 'species': 'REVIEW', 'review': True,
                     'review_uid': 'uid-on-disk'}]
    A._merge_imported_review_fields(plain, disk_flagged)
    check(plain[0].get('review') is True,
          'a tag already written to disk is still restored to memory')


# ── the outward-publishing export ───────────────────────────────────────
# What the real exporter does to a mask it publishes, so a leak is visible on
# the mask itself and not only on the tripwire counter. review_export.py:122
# stamps mask['review_uid'] onto every flagged mask it ships; the reviewer of
# 2026-09-06 saw exactly this identifier minted by this app.
MINTED_UID = 'FSB-20141208-T301-x0y0-085a4ba6'


class _MintingTripwire:
    """Stands where scripts/_reefreview/review_export stands, and mints like it.

    export_flagged_masks() writes a fresh review identifier onto every flagged
    mask it publishes and pushes the bundle to the public GitHub Pages repo.
    This stand-in records the call and performs the same minting, so a leak
    shows up twice: the call list is not empty, and an already-flagged mask
    comes back carrying an identifier it never had. Nothing is written to disk
    and nothing is pushed.
    """

    calls = []

    @staticmethod
    def export_flagged_masks(to_export, batch_files, **kwargs):
        """Record the call, mint onto every flagged mask, report a push."""
        _MintingTripwire.calls.append({'git_push': kwargs.get('git_push')})
        for fn in batch_files:
            seg = to_export.get(fn) or {}
            for m in seg.get('masks', []):
                if m.get('review') or m.get('species') == 'REVIEW':
                    m['review_uid'] = MINTED_UID
        return {'new': 1, 'pushed': True, 'pending_total': 1}


def _export_fixture_frame(name, masks):
    """A segmentation dict for the export test, outside the live session."""
    return name, {'image_path': name,
                  'image_path_abs': os.path.join(_RAW_DIR, FRAME),
                  'image_width': IMG_W, 'image_height': IMG_H,
                  'masks': masks, 'exported': False}


def _export_flagged(mask_id, spelling='both'):
    """A mask flagged before the lock, carrying no review identifier yet."""
    m = {'id': mask_id, 'status': 'accepted', 'area': 1600}
    if spelling in ('both', 'boolean'):
        m['review'] = True
    if spelling in ('both', 'species'):
        m['species'] = 'REVIEW'
        m['category'] = 'Review'
    return m


def test_export_bundle_is_suspended_and_reports_a_count():
    # This app is archived from the sidebar but still launchable through
    # POST /api/step/5/run, and its export reaches the same public repo. Plan
    # decision 4 suspends it: nothing is published and no fresh review
    # identifier is minted while the lock holds.
    fn_a, seg_a = _export_fixture_frame(
        'EXPORT_A.jpeg', [_export_flagged(0),
                          {'id': 1, 'status': 'accepted', 'species': 'PA'}])
    fn_b, seg_b = _export_fixture_frame(
        'EXPORT_B.jpeg', [_export_flagged(0), _export_flagged(1, 'species')])

    _MintingTripwire.calls = []
    prior_ok, prior_export = A._REVIEW_OK, A._review_export
    A._REVIEW_OK, A._review_export = True, _MintingTripwire
    try:
        stats = A._export_review_bundle({fn_a: seg_a, fn_b: seg_b}, [fn_a, fn_b])
    finally:
        A._REVIEW_OK, A._review_export = prior_ok, prior_export

    check(_MintingTripwire.calls == [],
          f'the real exporter was never called, so nothing was pushed, got '
          f'{_MintingTripwire.calls!r}')
    minted = [m['id'] for seg in (seg_a, seg_b) for m in seg['masks']
              if m.get('review_uid')]
    check(minted == [],
          f'no already-flagged mask gained a review identifier, got {minted!r}')
    check(isinstance(stats, dict) and stats.get('disabled') is True,
          f'the export reports itself disabled, got {stats!r}')
    check(stats.get('review_masks_skipped') == 3,
          f'it counts the three flagged masks it did not publish, got {stats!r}')
    check(seg_a['masks'][1].get('review_uid') is None,
          'the ordinary mask was left alone as well')

    # Hostile and empty batches: a filename with no segmentation, a frame with
    # no masks key, and an empty batch all count zero rather than raising. This
    # route is served on 0.0.0.0 with no authentication, so a batch list can
    # carry anything at all.
    _MintingTripwire.calls = []
    A._REVIEW_OK, A._review_export = True, _MintingTripwire
    try:
        odd = A._export_review_bundle(
            {'PRESENT.jpeg': {'image_path': 'PRESENT.jpeg'}},
            ['PRESENT.jpeg', 'ABSENT.jpeg'])
        empty = A._export_review_bundle({}, [])
    finally:
        A._REVIEW_OK, A._review_export = prior_ok, prior_export
    check(odd.get('review_masks_skipped') == 0,
          f'a frame with no masks key and a missing frame count zero, got {odd!r}')
    check(empty == {'disabled': True, 'review_masks_skipped': 0},
          f'an empty batch is still suspended and counts zero, got {empty!r}')
    check(_MintingTripwire.calls == [],
          f'and neither reached the exporter, got {_MintingTripwire.calls!r}')


def test_target_species_env_drops_a_review_entry():
    """A REVIEW entry in TCRMP_TARGET_SPECIES is dropped, the rest survive.

    The target list draws the label row, so a REVIEW entry there would put a
    live, armable REVIEW control in front of the operator that every add route
    answers with a 409. The value comes from a launcher, so the entry is
    dropped and logged rather than the whole launch refused.
    """
    prior = os.environ.get('TCRMP_TARGET_SPECIES')
    said = io.StringIO()
    try:
        os.environ['TCRMP_TARGET_SPECIES'] = 'PA, review ,OFRA,Review'
        with contextlib.redirect_stdout(said):
            codes = A._load_target_species()
    finally:
        if prior is None:
            os.environ.pop('TCRMP_TARGET_SPECIES', None)
        else:
            os.environ['TCRMP_TARGET_SPECIES'] = prior

    check(codes == ['PA', 'OFRA'],
          f'both REVIEW spellings are dropped and the real codes survive, got '
          f'{codes!r}')
    logged = said.getvalue()
    check('[review-lock]' in logged,
          f'the drop is logged under the lock own tag, got {logged!r}')
    check('review' in logged and 'Review' in logged,
          f'the log names both dropped entries, got {logged!r}')

    # An environment with no REVIEW entry passes through untouched and in
    # silence: the drop must not become a line on every launch.
    quiet = io.StringIO()
    try:
        os.environ['TCRMP_TARGET_SPECIES'] = 'PA,OFRA'
        with contextlib.redirect_stdout(quiet):
            plain = A._load_target_species()
    finally:
        if prior is None:
            os.environ.pop('TCRMP_TARGET_SPECIES', None)
        else:
            os.environ['TCRMP_TARGET_SPECIES'] = prior
    check(plain == ['PA', 'OFRA'],
          f'an ordinary environment value is unchanged, got {plain!r}')
    check(quiet.getvalue() == '',
          f'and nothing is logged for it, got {quiet.getvalue()!r}')


TESTS = [
    test_add_refuses_every_review_spelling,
    test_box_add_refuses_every_review_spelling,
    test_species_string_alone_is_refused,
    test_add_refuses_a_category_only_payload,
    test_relabel_refuses_adding_a_tag,
    test_batch_pass_never_mints_a_flagged_mask,
    test_refusal_answers_409_with_the_shared_message,
    test_ordinary_labelling_is_untouched,
    test_existing_flagged_mask_still_reads_back,
    test_disk_read_back_still_restores_an_existing_tag,
    test_export_bundle_is_suspended_and_reports_a_count,
    test_target_species_env_drops_a_review_entry,
]


if __name__ == '__main__':
    print('review-flag lock, Segment Images routes:')
    for _t in TESTS:
        run(_t)
    _failed = [name for name, ok in _RESULTS if not ok]
    shutil.rmtree(_ROOT, ignore_errors=True)
    print(f"\n{len(_RESULTS) - len(_failed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
