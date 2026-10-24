"""A species picked in the label popup reaches the server with the accept.

The defect this closes (2026-10-15): a mask added with no label armed is
labelled from the popup (a click on a target-label row, or a number key).
assignNewMaskSpecies set species, category and name on the BROWSER copy only,
then PUT /api/image/<f>/masks with nothing but {updates: {id: 'accepted'}}.
The server's label guard refuses to accept a mask whose stored species is
blank, so it kept the mask pending and blank while the page showed it
accepted as PA. A real project ended up with 19 pending and 62 rejected
blank masks this way.

Two halves, run for both annotator pages (combinedAnnotate, then this same
file re-runs itself for editMasks):

  Server (Flask test client): one PUT carrying the additive `relabel` field
  beside the accept stores species, category and name AND accepts; the
  status-only body the old page sent is refused by the label guard and
  leaves the mask pending and blank; an unnamed session is refused whole.

  Page (the template's own functions run under node against a stubbed
  fetch, through _annotator_page_harness.py in this folder): every popup
  path sends the species in the same request as the status; any refusal
  (label guard, missing initials, a non-2xx answer, a body with ok false or
  an error, an unreadable body, no answer at all, no answer in time, and
  editMasks' expert lock) puts the mask back the way the server holds it and
  leaves a persistent message, next step first, naming the mask; two writes
  to one mask end where the server is in every answer order; a later
  confirmed write clears the message; a refusal that arrives after a frame
  switch never touches the new frame's mask of the same id; a frame save
  waits for a write still in flight and leaves out a write that timed out;
  editMasks judges the expert lock exactly as the server does.

No pytest, GPU-free (CUDA_VISIBLE_DEVICES may be empty; nothing here loads a
model). subprocess.run and Popen are stubbed at module level, the standing
rule for annotator tests; node and the editMasks re-run use the saved real
functions. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_label_persists_with_accept.py
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_label_persists_with_accept.py editMasks
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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_real_run = subprocess.run
_real_popen = subprocess.Popen


def _stubbed_spawn(*args, **kwargs):
    """Refuse any spawn the code under test attempts (standing annotator rule)."""
    raise RuntimeError("subprocess is stubbed in test_label_persists_with_accept")


subprocess.run = _stubbed_spawn
subprocess.Popen = _stubbed_spawn


def _spawn(args, **kwargs):
    """Run one child this suite starts itself (node, the editMasks re-run).

    subprocess.run builds its child through the module's Popen, so the real
    Popen is put back for exactly the length of this one call.
    """
    subprocess.Popen = _real_popen
    try:
        return _real_run(args, **kwargs)
    finally:
        subprocess.Popen = _stubbed_spawn

APPS = ('combinedAnnotate', 'editMasks')
WHICH = sys.argv[1] if len(sys.argv) > 1 else APPS[0]
if WHICH not in APPS:
    raise SystemExit("usage: test_label_persists_with_accept.py [combinedAnnotate|editMasks]")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_SCRIPTS, 'TCRMPclip_' + WHICH, 'src')

_ROOT = tempfile.mkdtemp(prefix='label_persists_')
_EXPORT = os.path.join(_ROOT, 'run_20260101_lblacc', 'step4test_combinedAnnotate')
os.makedirs(_EXPORT, exist_ok=True)
os.environ.pop('TCRMP_REVIEWER', None)
os.environ['TCRMP_PROJECT_ID'] = '20260101_lblacc'
os.environ['TCRMP_EXPORT_DIR'] = _EXPORT
os.environ['TCRMP_INPUT_DIR'] = os.path.join(_ROOT, 'routed_input')
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA,AT,MCAV')

sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from PIL import Image  # noqa: E402

import app as A  # noqa: E402
import _annotator_page_harness as H  # noqa: E402

FRAME = 'TCRMP20201022_clip_SCP_T101.jpeg'

_fail = 0


def check(cond, msg):
    """Record one assertion, printing PASS or FAIL with the app name."""
    global _fail
    print(("  PASS " if cond else "  FAIL ") + "[%s] %s" % (WHICH, msg))
    if not cond:
        _fail += 1


# ── Server half ──────────────────────────────────────────────────────────────

def _mask(mid, status, species, category='', name=''):
    """One stored mask with the fields the /masks route reads and writes."""
    return {"id": mid, "status": status, "species": species, "category": category,
            "name": name, "label": chr(ord('A') + mid), "source_type": "manual_click",
            "source_x": 10 + mid, "source_y": 10, "review": False,
            "rle": {"counts": "c%d" % mid, "size": [48, 64]},
            "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]], "area": 100}


def _seed_frame():
    """Install one frame with four masks into the app's session and return it."""
    img = os.path.join(_ROOT, FRAME)
    Image.new("RGB", (64, 48), (1, 2, 3)).save(img)
    seg = {"image_path": FRAME, "image_path_abs": img,
           "image_width": 64, "image_height": 48,
           "masks": [_mask(0, "pending", ""),
                     _mask(1, "pending", ""),
                     _mask(2, "accepted", "AT", "Coral", "Agaricia tenuifolia"),
                     _mask(3, "accepted", "PA", "Coral", "Porites astreoides")],
           "reviewed": False, "exported": False}
    A.session['export_dir'] = _EXPORT
    A.session['segmentations'] = {FRAME: seg}
    return {m['id']: m for m in seg['masks']}


def _put(client, body):
    """PUT one body to the frame's /masks route; return (status, json)."""
    res = client.put('/api/image/%s/masks' % FRAME, json=body)
    return res.status_code, json.loads(res.data)


def test_server_relabel_with_accept():
    """The route stores the species and accepts it when both come together."""
    A.app.config['TESTING'] = True
    by_id = _seed_frame()
    pa = {"species": "PA", "category": "Coral", "name": "Porites astreoides"}
    with A.app.test_client() as c:
        # Nobody named: the accept with a relabel is refused whole.
        c.post('/api/reviewer', json={'reviewer': ''})
        code, body = _put(c, {"updates": {"0": "accepted"}, "relabel": {"0": pa}})
        check(body.get('needs_reviewer') is True and body.get('ok') is False,
              "an unnamed session's relabel plus accept is refused (got %s)" % body)
        check(by_id[0]['species'] == '' and by_id[0]['status'] == 'pending',
              "nothing in the refused request was applied")

        c.post('/api/reviewer', json={'reviewer': 'LO'})
        # What the page used to send: the accept alone. The guard refuses it.
        code, body = _put(c, {"updates": {"1": "accepted"}})
        check(code == 200 and 1 in (body.get('needs_species') or []),
              "a status-only accept of a blank mask is refused by the label guard (got %s)" % body)
        check(by_id[1]['status'] == 'pending' and by_id[1]['species'] == '',
              "the refused mask stays pending and blank on the server")

        # What the page sends now: the species and the accept in one request.
        code, body = _put(c, {"updates": {"0": "accepted"}, "relabel": {"0": pa}})
        check(code == 200 and body.get('ok') is True and not body.get('needs_species'),
              "species plus accept in one PUT is confirmed (got %s %s)" % (code, body))
        m = by_id[0]
        check(m['status'] == 'accepted', "the mask is accepted (got %r)" % m['status'])
        check(m['species'] == 'PA', "the species is stored (got %r)" % m['species'])
        check(m['category'] == 'Coral' and m['name'] == 'Porites astreoides',
              "the category and full name are stored (got %r, %r)" % (m['category'], m['name']))
        check(m.get('reviewer') == 'LO', "the accept is credited to LO (got %r)" % m.get('reviewer'))

        # The C popup on an accepted mask: relabel AT to PA, still accepted.
        code, body = _put(c, {"updates": {"2": "accepted"}, "relabel": {"2": pa}})
        check(body.get('ok') is True and by_id[2]['species'] == 'PA'
              and by_id[2]['name'] == 'Porites astreoides' and by_id[2]['status'] == 'accepted',
              "relabelling an accepted mask stores the new species (got %s)" % by_id[2]['species'])

        # Unlabel from the popup: blank species and pending, both stored.
        blank = {"species": "", "category": "", "name": ""}
        code, body = _put(c, {"updates": {"3": "pending"}, "relabel": {"3": blank}})
        check(body.get('ok') is True and by_id[3]['species'] == '' and by_id[3]['status'] == 'pending',
              "unlabel stores the blank species and the pending status (got %s, %s)"
              % (by_id[3]['species'], by_id[3]['status']))


# ── Page half ────────────────────────────────────────────────────────────────

# The visible messages, next step first and no HTTP codes (mask C, on screen).
NO_SPECIES = "Pick a species and accept mask C again: it was not saved because it has no species."
NO_INITIALS = "Type your initials in the top bar, then try mask C again: it was not saved without them."
NO_ANSWER = "Try mask C again: the annotator server did not answer, so it was not saved."
UNREADABLE = "Try mask C again: the server's answer could not be read, so it was not saved."
EXPERT_LOCKED = "Mask C is in expert review and read-only here, so the change was not saved."
BLANK_ON_SCREEN = ("Select mask C and pick a species: it has no species, so the server will "
                   "not accept it.")
FRAME_REFUSED = "Save this frame again (Ctrl+S): the server refused its statuses. Boom."

# Masks whose expert-lock state the page must judge exactly as the server's
# _is_expert_locked does (Edit Masks only): Python truthiness, a dict for
# expert_id, the exact REVIEW species.
LOCK_CASE_MASKS = [
    {}, {"review": True}, {"review": 1}, {"review": "yes"}, {"review": False}, {"review": 0},
    {"review": ""}, {"review": []}, {"review": {}}, {"species": "REVIEW"}, {"species": "review"},
    {"expert_id": {}}, {"expert_id": {"mode": "EXPERT"}}, {"expert_id": []}, {"expert_id": None},
    {"expert_id": "x"}, {"review_uid": "u1"}, {"review_uid": ""}, {"reviews": []},
    {"reviews": [{}]}, {"reviews": "x"}, {"reviews": {}}, {"reviews": {"a": 1}},
]

SCENARIOS = r"""
const blankC = () => ({id: 5, label: 'C', status: 'pending', species: '', category: '', name: ''});
const atC = () => ({id: 5, label: 'C', status: 'accepted', species: 'AT', category: 'Hard coral',
                    name: 'Agaricia tenuifolia'});
const paC = (status) => ({id: 5, label: 'C', status, species: 'PA', category: 'Hard coral',
                          name: 'Porites astreoides'});
const relabelOf = (call, id) => ((call && call.body && call.body.relabel) || {})[String(id)];
const NO_SPECIES = %(no_species)s;
const NO_INITIALS = %(no_initials)s;
const NO_ANSWER = %(no_answer)s;
const UNREADABLE = %(unreadable)s;
const EXPERT_LOCKED = %(expert_locked)s;
const BLANK_ON_SCREEN = %(blank_on_screen)s;
const FRAME_REFUSED = %(frame_refused)s;
const refused = {status: 200, body: {ok: false, needs_reviewer: true, error: 'x'}};
const taken = {status: 200, body: {ok: true}};

function wantReverted(m, label) {
  want(m.status === 'pending' && m.species === '' && !m.name && !m.category,
       label + ': the mask is back to pending and blank, as the server holds it (got '
       + m.status + ' ' + JSON.stringify(m.species) + ')');
}
function wantPersistentMessage(text, label) {
  want(statusEl().textContent === text, label + ': the status row says ' + JSON.stringify(text)
       + ' (got ' + JSON.stringify(statusEl().textContent) + ')');
  want(statusEl().cls.has('conflict'), label + ': the message is the persistent kind, not the 3 s flash');
  want(!/HTTP/.test(statusEl().textContent), label + ': no HTTP code in the visible text');
}
async function popupAccept(species, answer) {
  reset([blankC()]);
  pendingAddMaskIdx = 0;
  respondWith(answer);
  assignNewMaskSpecies(species);
  await settle();
  return currentData.masks[0];
}

// 1. The popup sends the species with the accept, and a confirmed answer sticks.
let m = await popupAccept('PA', taken);
want(puts().length === 1, 'popup label: exactly one PUT is sent');
want(puts()[0] && puts()[0].url === '/api/image/F1.jpg/masks', 'popup label: it goes to the frame /masks route');
want(puts()[0] && JSON.stringify(puts()[0].body.updates) === '{"5":"accepted"}', 'popup label: the PUT accepts mask C');
let rl = relabelOf(puts()[0], 5);
want(!!rl && rl.species === 'PA', 'popup label: the same PUT carries species PA (got ' + JSON.stringify(rl) + ')');
want(!!rl && rl.category === 'Hard coral', 'popup label: the category comes from the label list, not a hard-coded Coral');
want(!!rl && rl.name === 'Porites astreoides', 'popup label: the name is the full name, not the code');
want(m.status === 'accepted' && m.species === 'PA' && m.name === 'Porites astreoides',
     'popup label: a confirmed accept stays accepted as PA on the page');
want(!statusEl().cls.has('conflict'), 'popup label: a confirmed accept shows no refusal');
want(flushes === 1, 'popup label: a confirmed accept arms the trailing flush');

// 2. The label guard refuses: the page goes back to what the server holds.
m = await popupAccept('PA', {status: 200, body: {ok: true, needs_species: [5]}});
wantReverted(m, 'label guard refusal');
wantPersistentMessage(NO_SPECIES, 'label guard refusal');
want(flushes === 0, 'label guard refusal: no flush is armed for a write that was not applied');

// 3. Nobody is named: refused whole, the initials box is pointed at.
m = await popupAccept('PA', {status: 200, body: {ok: false, needs_reviewer: true,
                                                 error: 'Set your initials first (top bar) so this work is credited.'}});
wantReverted(m, 'missing initials');
wantPersistentMessage(NO_INITIALS, 'missing initials');
want(document.getElementById('reviewer-input').cls.has('missing'), 'missing initials: the initials box is marked');

// 4. A non-2xx answer: the reason shows, the status code does not.
m = await popupAccept('PA', {status: 500, body: {error: 'boom'}});
wantReverted(m, 'HTTP 500');
wantPersistentMessage('Try mask C again: the server refused it. Boom.', 'HTTP 500');

// 5. No answer at all.
m = await popupAccept('PA', {throws: true});
wantReverted(m, 'network failure');
wantPersistentMessage(NO_ANSWER, 'network failure');

// 6. A 2xx answer that is not JSON.
m = await popupAccept('PA', {status: 200});
wantReverted(m, 'unreadable answer');
wantPersistentMessage(UNREADABLE, 'unreadable answer');

// 7. ok false with an error (the expert-review lock answers this way).
m = await popupAccept('PA', {status: 409, body: {ok: false, review_locked: true, error: REVIEW_LOCKED_MSG}});
wantReverted(m, 'ok false with an error');
wantPersistentMessage('Try mask C again: the server refused it. ' + REVIEW_LOCKED_MSG, 'ok false with an error');

// 8. The C popup on an accepted mask: relabel AT to PA.
reset([atC()]);
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskSpecies('PA');
await settle();
rl = relabelOf(puts()[0], 5);
want(!!rl && rl.species === 'PA' && puts()[0].body.updates['5'] === 'accepted',
     'C popup relabel: species PA and the accept go in one PUT');
reset([atC()]);
pendingAddMaskIdx = 0;
respondWith(refused);
assignNewMaskSpecies('PA');
await settle();
m = currentData.masks[0];
want(m.species === 'AT' && m.name === 'Agaricia tenuifolia' && m.category === 'Hard coral' && m.status === 'accepted',
     'C popup relabel refused: the mask is AT and accepted again (got ' + m.species + ' ' + m.status + ')');

// 9. A refusal arriving after a frame switch never touches the new frame.
reset([blankC()]);
pendingAddMaskIdx = 0;
const slow = respondWith({deferred: true});
assignNewMaskSpecies('PA');
await settle();
const oldMask = currentData.masks[0];
switchTo('F2.jpg', [{id: 5, label: 'C', status: 'pending', species: 'AT', category: 'Hard coral',
                     name: 'Agaricia tenuifolia'}]);
slow.release({status: 200, body: {ok: true, needs_species: [5]}});
await settle();
const newMask = currentData.masks[0];
want(newMask.species === 'AT' && newMask.status === 'pending', 'frame switch: the new frame mask with the same id is untouched');
wantReverted(oldMask, 'frame switch');
want(statusEl().textContent === 'Pick a species and accept mask C on F1.jpg again: it was not saved because it has no species.',
     'frame switch: the message names the mask and the frame it is on (got ' + JSON.stringify(statusEl().textContent) + ')');

// 10. A category chip sends its code, category and name.
reset([blankC()]);
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskCategory('Sponge');
await settle();
rl = relabelOf(puts()[0], 5);
want(!!rl && rl.species === 'SPO' && rl.category === 'Sponge' && rl.name === 'Sponge'
     && puts()[0].body.updates['5'] === 'accepted',
     'category chip: SPO, Sponge and the accept go in one PUT (got ' + JSON.stringify(rl) + ')');
reset([blankC()]);
pendingAddMaskIdx = 0;
respondWith({status: 200, body: {ok: true, needs_species: [5]}});
assignNewMaskCategory('Sponge');
await settle();
wantReverted(currentData.masks[0], 'category chip refused');
want(statusEl().cls.has('conflict'), 'category chip refused: the message is persistent');

// 11. Unlabel stores the blank species with the pending status.
reset([paC('accepted')]);
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskCategory('');
await settle();
rl = relabelOf(puts()[0], 5);
want(!!rl && rl.species === '' && puts()[0].body.updates['5'] === 'pending',
     'unlabel: the blank species and pending go in one PUT (got ' + JSON.stringify(rl) + ')');
want(currentData.masks[0].species === '' && currentData.masks[0].status === 'pending',
     'unlabel: the page shows the mask pending and blank');

// 12. Fallbacks when the page knows no name or category for a code.
reset([blankC()]);
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskSpecies('ZZZ');
await settle();
rl = relabelOf(puts()[0], 5);
want(!!rl && rl.name === 'ZZZ' && rl.category === 'Coral',
     'no name or category known: the code stands as the name and Coral as the category (got ' + JSON.stringify(rl) + ')');

// 13. A refused scan candidate picks up the label list category and name.
reset([]);
scanCandidates = [{label: 'S1', species: '', status: 'pending'}];
pendingScanCandidateIdx = 0;
assignNewMaskSpecies('PA');
await settle();
const kept = keptCandidates[0] || {};
want(kept.species === 'PA' && kept.category === 'Hard coral' && kept.name === 'Porites astreoides',
     'scan candidate: species, category and full name go with the keep (got ' + JSON.stringify(kept) + ')');
want(puts().length === 0, 'scan candidate: no /masks PUT is sent for a candidate');

// 14. A refused reject is undone too.
reset([paC('pending')]);
selectedMask = 0;
respondWith({status: 404, body: {error: 'not found'}});
rejectMask();
await settle();
want(currentData.masks[0].status === 'pending', 'refused reject: the mask is pending again');
want(!recentlyDeletedIds.has(5), 'refused reject: the mask no longer counts as deleted for /add');
wantPersistentMessage('Try mask C again: the server refused it. Not found.', 'refused reject');

// 15. Enter on a mask the server holds blank: the status-only accept is checked too.
reset([paC('pending')]);
selectedMask = 0;
respondWith({status: 200, body: {ok: true, needs_species: [5]}});
acceptMask();
await settle();
want(currentData.masks[0].status === 'pending', 'refused Enter accept: the mask is pending again');
wantPersistentMessage(NO_SPECIES, 'refused Enter accept');

// 16. A frame save waits for an accept in flight and carries what the server kept.
reset([blankC()]);
pendingAddMaskIdx = 0;
const inFlight = respondWith({deferred: true});
respondWith(taken);
assignNewMaskSpecies('PA');
const save = saveCurrentMasks();
await settle();
want(puts().length === 1, 'frame save: it waits while the accept is still in flight (PUTs so far: ' + puts().length + ')');
inFlight.release({status: 200, body: {ok: true, needs_species: [5]}});
await save;
await settle();
want(puts().length === 2 && puts()[1].body.updates['5'] === 'pending',
     'frame save: it restates the status the server kept, not the refused accept (got '
     + JSON.stringify(puts()[1] && puts()[1].body.updates) + ')');

// 17. Ctrl+S never says saved when the frame save was refused.
reset([paC('accepted')]);
respondWith({status: 500, body: {error: 'boom'}});
respondWith(taken);
await saveFrameNow();
await settle();
wantPersistentMessage(FRAME_REFUSED, 'Ctrl+S with a refused frame save');

// 18. A refused label never undoes a reject the server took, in every answer order.
for (const order of ['label last', 'label first']) {
  reset([blankC()]);
  pendingAddMaskIdx = 0;
  const lbl = respondWith({deferred: true});
  const rej = respondWith({deferred: true});
  assignNewMaskSpecies('PA');
  selectedMask = 0;
  rejectMask();
  await settle();
  if (order === 'label last') { rej.release(taken); await settle(); lbl.release(refused); }
  else { lbl.release(refused); await settle(); rej.release(taken); }
  await settle();
  m = currentData.masks[0];
  want(m.status === 'rejected' && m.species === '' && !m.name,
       'refused label, saved reject (' + order + '): rejected, species put back (got ' + m.status + ' ' + JSON.stringify(m.species) + ')');
  respondWith(taken);
  await saveCurrentMasks();
  const last = puts()[puts().length - 1];
  want(last && last.body.updates['5'] === 'rejected',
       'refused label, saved reject (' + order + '): the next frame save keeps it rejected');
}

// 19. Two quick writes to one mask: the page ends where the server is, in
// either answer order. `second` names the second write: 'label' (AT) or
// 'reject'. With `laterFirst` the second write's answer arrives first.
const twoWrites = async (second, firstAnswer, secondAnswer, laterFirst) => {
  reset([blankC()]);
  pendingAddMaskIdx = 0;
  const w1 = respondWith({deferred: true});
  assignNewMaskSpecies('PA');
  const w2 = respondWith({deferred: true});
  if (second === 'label') { pendingAddMaskIdx = 0; assignNewMaskSpecies('AT'); }
  else { selectedMask = 0; rejectMask(); }
  await settle();
  if (laterFirst) {
    w2.release(secondAnswer); await settle();
    const between = Object.assign({}, currentData.masks[0]);
    w1.release(firstAnswer); await settle();
    return {between, end: currentData.masks[0]};
  }
  w1.release(firstAnswer); await settle();
  w2.release(secondAnswer); await settle();
  return {end: currentData.masks[0]};
};
wantReverted((await twoWrites('label', refused, refused, false)).end, 'two labels, both refused');
m = (await twoWrites('label', refused, taken, false)).end;
want(m.species === 'AT' && m.status === 'accepted', 'two labels, first refused, second taken: AT accepted (got ' + m.species + ' ' + m.status + ')');
m = (await twoWrites('label', taken, refused, false)).end;
want(m.species === 'PA' && m.status === 'accepted' && m.name === 'Porites astreoides',
     'two labels, first taken, second refused: PA accepted (got ' + m.species + ' ' + m.status + ')');
let r = await twoWrites('label', taken, refused, true);
want(r.between.species === 'PA' && r.between.status === 'accepted',
     'two labels, second refused before the first answers: the page hands back to the first label, PA (got '
     + r.between.species + ' ' + r.between.status + ')');
want(r.end.species === 'PA' && r.end.status === 'accepted' && r.end.name === 'Porites astreoides',
     'two labels, second refused, then first taken: PA accepted (got ' + r.end.species + ' ' + r.end.status + ')');
r = await twoWrites('label', refused, refused, true);
wantReverted(r.end, 'two labels, both refused, later answer first');
r = await twoWrites('reject', taken, refused, true);
want(r.between.status === 'accepted' && r.between.species === 'PA',
     'label then reject, reject refused first: the page shows the label still out, PA accepted (got '
     + r.between.species + ' ' + r.between.status + ')');
want(r.end.status === 'accepted' && r.end.species === 'PA',
     'label then reject, reject refused, label taken: PA accepted (got ' + r.end.species + ' ' + r.end.status + ')');
r = await twoWrites('reject', refused, refused, true);
wantReverted(r.end, 'label then reject, both refused, later answer first');

// 20. A frame save names the frame only when it is not the one on screen.
reset([{id: 5, label: 'C', status: 'accepted', species: '', category: '', name: ''}]);
respondWith({status: 200, body: {ok: true, needs_species: [5]}});
let frameSaved = await saveCurrentMasks();
want(frameSaved === false, 'blank accepted mask: the frame save reports not saved');
wantPersistentMessage(BLANK_ON_SCREEN, 'blank accepted mask on screen');
want(statusEl().title === BLANK_ON_SCREEN, 'blank accepted mask: the full text is in the status row title');
reset([{id: 5, label: 'C', status: 'accepted', species: '', category: '', name: ''}]);
const late = respondWith({deferred: true});
const lateSave = saveCurrentMasks();
await settle();
switchTo('F2.jpg', []);
late.release({status: 200, body: {ok: true, needs_species: [5]}});
await lateSave;
want(statusEl().textContent.startsWith('Reopen frame F1.jpg, then select mask C and pick a species:'),
     'blank accepted mask after a frame switch: the message names the frame (got ' + JSON.stringify(statusEl().textContent) + ')');
reset([paC('accepted')]);
respondWith({status: 500, body: {error: 'boom'}});
await saveCurrentMasks();
wantPersistentMessage(FRAME_REFUSED, 'refused frame save on screen');
want(statusEl().title === statusEl().textContent, 'refused frame save: the full text is in the title');

// 21. A confirmed later write to a mask clears that mask's not-saved message,
// and only that mask's.
m = await popupAccept('PA', {status: 200, body: {ok: true, needs_species: [5]}});
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskSpecies('PA');
await settle();
want(statusEl().textContent === '' && !statusEl().cls.has('conflict'),
     'a confirmed write clears that mask\'s not-saved message (got ' + JSON.stringify(statusEl().textContent) + ')');
reset([blankC(), {id: 6, label: 'D', status: 'pending', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]);
pendingAddMaskIdx = 0;
respondWith({status: 200, body: {ok: true, needs_species: [5]}});
assignNewMaskSpecies('PA');
await settle();
selectedMask = 1;
respondWith(taken);
acceptMask();
await settle();
want(statusEl().textContent === NO_SPECIES, 'a confirmed write to another mask leaves mask C\'s message up');

%(app_specific)s

// 22. A write that never answers: put back, said plainly, and kept out of
// frame saves until a later write to it is confirmed.
reset([paC('accepted'), {id: 6, label: 'D', status: 'pending', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]);
selectedMask = 1;
respondWith({hang: true});
acceptMask();
await sleep(MASK_WRITE_TIMEOUT_MS * 3);
await settle();
want(currentData.masks[1].status === 'pending', 'no answer in time: mask D is put back to pending');
want(statusEl().textContent.startsWith('Reopen the frame to see what mask D holds:')
     && statusEl().textContent.includes('may still have saved it') && statusEl().cls.has('conflict'),
     'no answer in time: the message says the server may still have saved it, next step first (got '
     + JSON.stringify(statusEl().textContent) + ')');
respondWith(taken);
await saveCurrentMasks();
let lastSave = puts()[puts().length - 1];
want(lastSave && '5' in lastSave.body.updates && !('6' in lastSave.body.updates),
     'a timed-out mask is left out of the frame save (got ' + JSON.stringify(lastSave && lastSave.body.updates) + ')');
selectedMask = 1;
respondWith(taken);
acceptMask();
await settle();
respondWith(taken);
await saveCurrentMasks();
lastSave = puts()[puts().length - 1];
want(lastSave && lastSave.body.updates['6'] === 'accepted',
     'after a later confirmed write the mask is in frame saves again (got ' + JSON.stringify(lastSave && lastSave.body.updates) + ')');
reset([blankC()]);
pendingAddMaskIdx = 0;
respondWith({hang: true});
assignNewMaskSpecies('PA');
const outcome = await Promise.race([saveCurrentMasks().then(() => 'finished'),
                                    sleep(MASK_WRITE_TIMEOUT_MS * 6).then(() => 'still waiting')]);
want(outcome === 'finished', 'a frame save behind a write that never answers still finishes (got ' + outcome + ')');
reset([paC('accepted')]);
respondWith({hang: true});
const hungSave = await Promise.race([saveCurrentMasks(), sleep(MASK_WRITE_TIMEOUT_MS * 6).then(() => 'still waiting')]);
want(hungSave === false && statusEl().textContent.startsWith('Reopen this frame to see what the server holds:'),
     'a frame save that never answers is given up and reported (got ' + JSON.stringify(hungSave) + ', '
     + JSON.stringify(statusEl().textContent) + ')');
"""

APP_SPECIFIC = {
    'combinedAnnotate': r"""
// CA: a code the label list leaves unnamed takes a name learned elsewhere on the page.
reset([blankC()]);
pendingAddMaskIdx = 0;
respondWith(taken);
assignNewMaskSpecies('LRN');
await settle();
rl = relabelOf(puts()[0], 5);
want(!!rl && rl.name === 'Learned name' && rl.category === 'Hard coral',
     'learned name: a name the page learned is sent (got ' + JSON.stringify(rl) + ')');
""",
    'editMasks': r"""
// EM: a frame save leaves out masks that are out with an expert, and still
// calls the route so the server marks the frame reviewed.
const lockedC = () => Object.assign(paC('accepted'), {expert_id: {mode: 'EXPERT'}});
reset([lockedC()]);
const allLocked = await saveCurrentMasks();
want(allLocked === true && puts().length === 1 && JSON.stringify(puts()[0].body.updates) === '{}',
     'all-locked frame: the save sends no status, still reaches the route, and reports saved (got '
     + allLocked + ', ' + JSON.stringify(puts().map(p => p.body.updates)) + ')');
want(!statusEl().cls.has('conflict'), 'all-locked frame: no refusal message');
reset([]);
const empty = await saveCurrentMasks();
want(empty === true && puts().length === 1 && JSON.stringify(puts()[0].body.updates) === '{}',
     'a frame with no masks still reaches the route so the server marks it reviewed');
reset([lockedC()]);
await saveFrameNow();
await settle();
want(statusEl().textContent === 'Saved to disk', 'all-locked frame: Ctrl+S says Saved to disk (got ' + JSON.stringify(statusEl().textContent) + ')');
reset([lockedC(), {id: 6, label: 'D', status: 'pending', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]);
respondWith({match: '/masks', status: 200, body: {ok: true}});
const mixed = await saveCurrentMasks();
want(mixed === true && puts().length === 1 && JSON.stringify(puts()[0].body.updates) === '{"6":"pending"}',
     'mixed frame: only the unlocked mask is sent, and it saves (got ' + JSON.stringify(puts()[0] && puts()[0].body.updates) + ')');

// EM: the expert lock refuses a mask that is out with an expert.
m = await popupAccept('PA', {status: 200, body: {ok: false, locked: true, locked_ids: [5]}});
wantReverted(m, 'expert lock');
wantPersistentMessage(EXPERT_LOCKED, 'expert lock');

// EM: the page judges the expert lock exactly as the server does.
for (const c of LOCK_CASES) {
  want(isExpertLocked(c.mask) === c.locked,
       'isExpertLocked matches the server for ' + JSON.stringify(c.mask) + ' (server: ' + c.locked + ')');
}
""",
}


def test_page_still_renders():
    """GET / renders the template and serves the label-write code.

    The page half reads the raw template; this proves Jinja still compiles it
    (a JSDoc type in double braces once broke the whole page).
    """
    A.app.config['TESTING'] = True
    res = A.app.test_client().get('/')
    page = res.get_data(as_text=True)
    check(res.status_code == 200, "GET / renders the annotator page (got %s)" % res.status_code)
    check("function labelMask(" in page and "relabel: { [key]: relabel.fields }" in page,
          "the rendered page carries the one-request label write")


# ── Static guards ────────────────────────────────────────────────────────────

# The one function allowed to hand persistMaskStatus a fourth argument (the
# label fields). Any other caller passing one could send a review tag past the
# expert-review lock, which every label path checks before it calls labelMask.
LABEL_FIELD_SENDER = 'labelMask'
WRITE_CALLEE = 'persistMaskStatus'
MAX_PLAIN_ARGS = 3
# The time limit on a mask write, in milliseconds: long enough for a slow
# save, short enough that a save path never waits on a dead server for long.
TIMEOUT_RANGE_MS = (5000, 30000)


def test_only_label_mask_sends_fields():
    """Only labelMask hands persistMaskStatus the label fields, anywhere in the script."""
    src = H.template_source(WHICH)
    stray = H.calls_with_more_args(src, WRITE_CALLEE, MAX_PLAIN_ARGS, LABEL_FIELD_SENDER)
    check(not stray, "only %s passes label fields to %s, anywhere in the page (stray calls on lines %s)"
          % (LABEL_FIELD_SENDER, WRITE_CALLEE, stray or "none"))
    check(bool(H.calls_with_more_args(src.replace("function labelMask(", "function notLabelMask("),
                                      WRITE_CALLEE, MAX_PLAIN_ARGS, LABEL_FIELD_SENDER)),
          "labelMask's own call is the one the guard allows (renaming it makes the guard fire)")
    decoy = ("function labelMask(m, s, f) {\n  return persistMaskStatus(m.id, s, 'p', {fields: f});\n}\n"
             "document.addEventListener('x', () => persistMaskStatus(m.id, 'accepted', 'p', "
             "{fields: {species: 'REVIEW', review: true}}));\n"
             "const later = function (m) {\n  setTimeout(() => persistMaskStatus(m.id, 'a', 'p', {}), 0);\n};\n"
             "function quiet(m) {\n  persistMaskStatus(m.id, f(a, b), 'pending');\n}\n")
    check(len(H.calls_with_more_args(decoy, WRITE_CALLEE, MAX_PLAIN_ARGS, LABEL_FIELD_SENDER)) == 2,
          "the guard catches calls in a listener and in a nested function, and lets a nested "
          "call's commas and labelMask's own call pass")


def test_write_time_limit():
    """The template declares a time limit on mask writes within a sane range."""
    decl = H.top_level_constant(H.template_source(WHICH), 'MASK_WRITE_TIMEOUT_MS')
    value = int(re.search(r"=\s*([\d_]+)", decl).group(1).replace('_', '')) if decl else None
    check(value is not None and TIMEOUT_RANGE_MS[0] <= value <= TIMEOUT_RANGE_MS[1],
          "MASK_WRITE_TIMEOUT_MS is declared between %d and %d ms (got %r)"
          % (TIMEOUT_RANGE_MS + (value,)))


def _lock_cases():
    """The LOCK_CASE_MASKS with the server's own verdict on each (Edit Masks)."""
    if WHICH != 'editMasks':
        return []
    return [{"mask": m, "locked": bool(A._is_expert_locked(dict(m)))} for m in LOCK_CASE_MASKS]


def test_page_sends_species_and_honours_refusals():
    """Run every page scenario under node against the template's own code."""
    constants = {k: json.dumps(v) for k, v in {
        "no_species": NO_SPECIES, "no_initials": NO_INITIALS, "no_answer": NO_ANSWER,
        "unreadable": UNREADABLE, "expert_locked": EXPERT_LOCKED,
        "blank_on_screen": BLANK_ON_SCREEN, "frame_refused": FRAME_REFUSED}.items()}
    constants["app_specific"] = APP_SPECIFIC[WHICH]
    script = H.build(WHICH, SCENARIOS % constants,
                     extra_prelude="const LOCK_CASES = %s;" % json.dumps(_lock_cases()))
    H.run(script, _ROOT, _spawn, check)


def _rerun_for_edit_masks():
    """Run this same file for the editMasks twin; return its exit code."""
    print("\n-- editMasks --", flush=True)
    res = _spawn([sys.executable, os.path.abspath(__file__), 'editMasks'], timeout=600)
    return res.returncode


if __name__ == "__main__":
    try:
        print("-- %s: server --" % WHICH)
        test_server_relabel_with_accept()
        test_page_still_renders()
        test_only_label_mask_sends_fields()
        test_write_time_limit()
        print("-- %s: page --" % WHICH)
        test_page_sends_species_and_honours_refusals()
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
    twin_code = 0
    if WHICH == H.APPS[0]:
        twin_code = _rerun_for_edit_masks()
    print("PASS" if _fail == 0 else "%d FAILED" % _fail)
    sys.exit(1 if (_fail or twin_code) else 0)
