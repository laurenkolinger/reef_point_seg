"""Every save goes to the frame whose masks are loaded, however fast the
frames change.

The defect this closes (2026-10-16): selectImage moved the current frame
index at once but loaded the new frame's masks later. Every save read its
filename from the current index and its statuses from the loaded masks, so a
second Prev (or any frame change) during a load sent one frame's statuses to
another frame's file. Mask ids start over on every frame, so the server
applied them to the other frame's masks: an accepted mask on T308 became
pending. A frame load that finished after the operator had moved on also
replaced the newer frame on screen.

What this suite proves, for both annotator pages (combinedAnnotate, then
this file re-runs itself for editMasks), against a fake server that keeps
every frame and holds chosen frame loads open:
  - double Prev, Prev then Next, a held arrow key (key-repeat burst, both
    directions), the unlabeled-frame finder, a batch change, Done and Export,
    each started while a frame is still loading, leave the server holding
    each frame's own statuses and the page showing the frame last asked for;
  - an accept pressed while the next frame loads goes to the frame it was
    pressed on;
  - Export and Done stop, and say so, while a frame save is refused;
  - a merge, merge same ID, Reload IDs, an add, a box add, a scan keep or scan
    results, a refine, a redraw, a re-run, an undo or a draw edit never sends
    a request once the frame changed while it waited, and never applies an
    answer to a frame (data object) no longer loaded; a late merge or add
    after Next leaves the next frame and its server copy untouched;
  - scan candidates are cleared when the next frame arrives;
  - an accept pressed while the same frame reloads is either kept on both
    sides or refused with a reason, never dropped in silence;
  - a frame that fails to load puts the index and picture back on the loaded
    frame and says so;
  - Reload IDs sends no frame save first;
  - a frame reopen, a merge and an undo wait for writes still out on their
    frame, so the server's answer already holds them;
  - a merge held behind an unanswered accept, then Right (or Right and Left),
    on a server that answers writes late: the merged mask stays accepted on
    the server and a return to the frame shows what the server holds;
  - each of the 13 frame requests counts as out on its frame from the click,
    so a save started while it waits sends nothing until it is answered; it
    never waits on itself, one that fails frees its frame at once, and two
    requests go out in click order;
  - a frame load with no answer gives up after the time limit, puts the
    index and picture back, says so, and lets edits run again;
  - a save whose frame still has a request out when its wait ends sends
    nothing, says "Still saving the last change", and stops Done and Export;
    a merge slower than the wait, then Right, leaves the merged mask accepted;
  - each of the 13 frame requests gives up after its own time limit, frees
    its frame and changes no mask;
  - Reload IDs with nothing new says "nothing new" on a frame that shows
    imported IDs and "no imported IDs yet" only on a frame with none.
And on the Flask test client: Reload IDs reports only the masks it changed,
so a frame with nothing imported shows 0.

The page's own functions run under node (_annotator_page_harness.py in this
folder); no browser, no server, no GPU. Neither page has a frame list or
frame picker: these are all the ways to change frame.

No pytest. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_frame_switch_saves.py
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_frame_switch_saves.py editMasks
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
import shutil
import subprocess
import sys
import tempfile

import _annotator_page_harness as H

_real_run = subprocess.run
_real_popen = subprocess.Popen


def _stubbed_spawn(*args, **kwargs):
    """Refuse any spawn the code under test attempts (standing annotator rule)."""
    raise RuntimeError("subprocess is stubbed in test_frame_switch_saves")


subprocess.run = _stubbed_spawn
subprocess.Popen = _stubbed_spawn


def _spawn(args, **kwargs):
    """Run one child this suite starts itself (node, the editMasks re-run)."""
    subprocess.Popen = _real_popen
    try:
        return _real_run(args, **kwargs)
    finally:
        subprocess.Popen = _stubbed_spawn


WHICH = sys.argv[1] if len(sys.argv) > 1 else H.APPS[0]
if WHICH not in H.APPS:
    raise SystemExit("usage: test_frame_switch_saves.py [combinedAnnotate|editMasks]")

_fail = 0


def check(cond, msg):
    """Record one assertion, printing PASS or FAIL with the app name."""
    global _fail
    print(("  PASS " if cond else "  FAIL ") + "[%s] %s" % (WHICH, msg))
    if not cond:
        _fail += 1


SCENARIOS = r"""
// Three frames whose mask 0 differ, so a status saved to the wrong frame shows.
const FRAMES = () => ({
  'F1.jpg': {masks: [{id: 0, label: 'A', status: 'pending', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]},
  'F2.jpg': {masks: [{id: 0, label: 'A', status: 'accepted', species: 'AT', category: 'Hard coral', name: 'Agaricia tenuifolia'}]},
  'F3.jpg': {masks: [{id: 0, label: 'A', status: 'rejected', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]},
});
let own = {};
const reading = (m) => (m ? m.status + ' ' + m.species : 'none');
async function bootAt(i) {
  reset([]);
  serverFrames = FRAMES();
  own = Object.fromEntries(Object.entries(serverFrames).map(([f, fr]) => [f, reading(fr.masks[0])]));
  images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}, {filename: 'F3.jpg'}];
  currentData = null; currentIdx = -1;
  await selectImage(i);
  await settle();
}
function serverHoldsOwn(label) {
  const got = Object.fromEntries(Object.entries(serverFrames).map(([f, fr]) => [f, reading(fr.masks[0])]));
  want(Object.keys(own).every(f => got[f] === own[f]),
       label + ': the server holds each frame\'s own statuses (got ' + JSON.stringify(got) + ')');
}
function pageShows(file, label) {
  const m = currentData && currentData.masks && currentData.masks[0];
  const onIndex = images[currentIdx] ? images[currentIdx].filename : 'none';
  want(onIndex === file && reading(m) === own[file],
       label + ': the page shows ' + file + ' with its own masks (index on ' + onIndex + ', masks ' + reading(m) + ')');
}
async function finish(label, file) {
  await saveCurrentMasks();
  await settle();
  serverHoldsOwn(label);
  pageShows(file, label);
}

// 1. Double Prev while the first Prev is still loading, loads answering in order.
await bootAt(2);
gate('F2.jpg'); gate('F1.jpg');
prevImage(); await settle();
prevImage(); await settle();
openGate('F2.jpg'); await settle();
openGate('F1.jpg'); await settle();
await finish('double Prev', 'F1.jpg');

// 2. Double Prev, the first load answering last.
await bootAt(2);
gate('F2.jpg'); gate('F1.jpg');
prevImage(); await settle();
prevImage(); await settle();
openGate('F1.jpg'); await settle();
openGate('F2.jpg'); await settle();
await finish('double Prev, older load answers last', 'F1.jpg');

// 3. Prev then Next (the Next button), the older load answering last.
await bootAt(1);
gate('F1.jpg'); gate('F2.jpg');
prevImage(); await settle();
advanceToNextFrame(); await settle();
openGate('F2.jpg'); await settle();
openGate('F1.jpg'); await settle();
await finish('Prev then Next', 'F2.jpg');

// 4. A held Left arrow: one press, then key-repeat events while frames load.
await bootAt(2);
gate('F2.jpg'); gate('F1.jpg');
pressKey('ArrowLeft'); await settle();
pressKey('ArrowLeft', true); pressKey('ArrowLeft', true); pressKey('ArrowLeft', true);
await settle();
openGate('F2.jpg'); await settle();
openGate('F1.jpg'); await settle();
await finish('Left arrow key-repeat burst', 'F1.jpg');

// 5. A held Right arrow from the first frame.
await bootAt(0);
gate('F2.jpg'); gate('F3.jpg');
pressKey('ArrowRight'); await settle();
pressKey('ArrowRight', true); pressKey('ArrowRight', true);
await settle();
openGate('F3.jpg'); await settle();
openGate('F2.jpg'); await settle();
await finish('Right arrow key-repeat burst', 'F3.jpg');

// 6. The unlabeled-frame finder while the next frame is loading.
await bootAt(0);
unlabeledSet = new Set(['F3.jpg']);
gate('F2.jpg'); gate('F3.jpg');
advanceToNextFrame(); await settle();
jumpNextUnlabeled(); await settle();
openGate('F3.jpg'); await settle();
openGate('F2.jpg'); await settle();
await finish('finder jump during a slow load', 'F3.jpg');

// 7. A batch change while a frame of the old batch is still loading.
await bootAt(0);
gate('F2.jpg');
advanceToNextFrame(); await settle();
serverFrames['G1.jpg'] = {masks: [{id: 0, label: 'A', status: 'accepted', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'}]};
own['G1.jpg'] = 'accepted PA';
serverBatch = {images: [{filename: 'G1.jpg'}], total_batches: 1, batch_number: 1, offset: 0,
               batch_size: 50, total_images: 1};
await nextReviewBatch(); await settle();
openGate('F2.jpg'); await settle();
await finish('batch change during a slow load', 'G1.jpg');

// 8. Done while the previous frame is still loading.
await bootAt(2);
gate('F2.jpg');
prevImage(); await settle();
await doneAndReturn(); await settle();
serverHoldsOwn('Done during a slow load');
openGate('F2.jpg'); await settle();

// 9. Export while the previous frame is still loading.
await bootAt(2);
gate('F2.jpg');
prevImage(); await settle();
await doExport(); await settle();
serverHoldsOwn('Export during a slow load');
openGate('F2.jpg'); await settle();

// 10. Enter pressed while the next frame loads is ignored, and says why: the
// loaded masks belong to the frame being left, drawn over the next one's picture.
await bootAt(0);
gate('F2.jpg');
advanceToNextFrame(); await settle();
selectedMask = 0;
acceptMask(); await settle();
want(statusEl().textContent === 'This frame is still loading. Try again in a moment.',
     'accept pressed during a slow load: ignored, with a plain reason (got ' + JSON.stringify(statusEl().textContent) + ')');
openGate('F2.jpg'); await settle();
await finish('accept pressed during a slow load', 'F2.jpg');

// 11. Export and Done stop while a frame save is refused, and say so.
const paA = () => ({id: 5, label: 'C', status: 'accepted', species: 'PA', category: 'Hard coral', name: 'Porites astreoides'});
reset([paA()]);
respondWith({match: '/masks', status: 500, body: {error: 'boom'}});
await doExport(); await settle();
want(!fetchLog.some(c => c.url === '/api/export_batch'), 'refused frame save: Export does not run');
want(statusEl().textContent.startsWith('Export stopped because this frame did not save. Save this frame again (Ctrl+S)')
     && statusEl().cls.has('conflict'),
     'refused frame save: Export says why it stopped, with the next step (got ' + JSON.stringify(statusEl().textContent) + ')');
reset([paA()]);
respondWith({match: '/masks', status: 500, body: {error: 'boom'}});
await doneAndReturn(); await settle();
want(!windowClosed && !window.location.href, 'refused frame save: Done does not leave the page');
want(statusEl().textContent.startsWith('The window stays open because this frame did not save. Save this frame again (Ctrl+S)'),
     'refused frame save: Done says why the window stays open (got ' + JSON.stringify(statusEl().textContent) + ')');
reset([paA()]);
await doneAndReturn(); await settle();
want(windowClosed, 'a frame save that is taken lets Done leave as before');

// 13 to 15. A merge or an add answered after the operator moved on never
// touches the next frame. Two masks per frame here.
const mk = (id, status, species) => ({id, label: 'AB'[id], status, species, category: 'Hard coral',
                                      name: species === 'AT' ? 'Agaricia tenuifolia' : 'Porites astreoides'});
const readAll = (fr) => fr.masks.map(x => x.id + ':' + x.status + ' ' + x.species).join(', ');
async function bootPair(i) {
  reset([]);
  serverFrames = {'F1.jpg': {masks: [mk(0, 'accepted', 'PA'), mk(1, 'pending', 'PA')]},
                  'F2.jpg': {masks: [mk(0, 'rejected', 'PA'), mk(1, 'accepted', 'AT')]}};
  own = Object.fromEntries(Object.entries(serverFrames).map(([f, fr]) => [f, readAll(fr)]));
  images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}];
  currentData = null; currentIdx = -1;
  await selectImage(i);
  await settle();
}
async function finishPair(label, file) {
  await saveCurrentMasks(); await settle();
  const got = Object.fromEntries(Object.entries(serverFrames).map(([f, fr]) => [f, readAll(fr)]));
  want(Object.keys(own).every(f => got[f] === own[f]),
       label + ': the server holds each frame\'s own masks (got ' + JSON.stringify(got) + ')');
  want(currentData && currentData.filename === file && readAll(currentData) === own[file],
       label + ': the page shows ' + file + ' with its own masks (got ' + (currentData ? readAll(currentData) : 'none') + ')');
}
const foreign0 = {id: 0, label: 'A', status: 'accepted', species: 'XX', category: 'Hard coral', name: 'Foreign'};
await bootPair(0);
checkedMaskIds = new Set([0, 1]);
let late = respondWith({match: '/merge_many', deferred: true});
mergeCheckedMasks(); await settle();
advanceToNextFrame(); await settle();
late.release({status: 200, body: {ok: true, mask: foreign0, removed_ids: [1]}});
await settle(); await sleep(50); await settle();
await finishPair('merge of checked masks answered after Next', 'F2.jpg');
await bootPair(0);
late = respondWith({match: '/add', deferred: true});
addMaskAt(10, 10); await settle();
advanceToNextFrame(); await settle();
late.release({status: 200, body: {ok: true, mask: {id: 1, label: 'B', status: 'pending', species: ''}, affected: {}}});
await settle(); await sleep(50); await settle();
await finishPair('add answered after Next', 'F2.jpg');
await bootPair(0);
late = respondWith({match: '/merge_same_id', deferred: true});
mergeSameId(); await settle();
advanceToNextFrame(); await settle();
late.release({status: 200, body: {ok: true, masks: [foreign0], merged: 1, refused: []}});
await settle(); await sleep(50); await settle();
await finishPair('merge same ID answered after Next', 'F2.jpg');

// 16. Every frame request: nothing is sent when the frame changes while the
// request waits on a write, and an answer for a frame object no longer loaded
// changes nothing. A reload of the same frame is a new object, so this also
// covers A to B and back to A.
const pairMasks = () => [Object.assign(mk(0, 'accepted', 'PA'), {refinement_clicks: [{x: 1, y: 1, label: 1}],
                                                                  bbox: [1, 1, 8, 8]}),
                         mk(1, 'pending', 'PA')];
const foreignMask = {ok: true, mask: foreign0, affected: {updated: [{id: 1, species: 'XX', status: 'rejected'}]}};
const cand = {label: 'S1', species: 'PA', status: 'pending', bbox: [1, 1, 5, 5]};
const ROUTES = [
  ['merge of checked masks', '/merge_many', () => { checkedMaskIds = new Set([0, 1]); return mergeCheckedMasks(); },
   {ok: true, mask: foreign0, removed_ids: [1]}],
  ['merge of two clicked masks', '/merge', () => doMerge(0, 1), {ok: true, mask: foreign0, removed_id: 1}],
  ['merge same ID', '/merge_same_id', () => mergeSameId(), {ok: true, masks: [foreign0], merged: 1, refused: []}],
  ['Reload IDs', '/api/reload_segmentations', () => reloadSegmentations(), {ok: true, masks: [foreign0], merged: 1, expert: 0}],
  ['add', '/add', () => addMaskAt(10, 10), {ok: true, mask: Object.assign({}, foreign0, {id: 9}), affected: {updated: [{id: 1, species: 'XX'}]}}],
  ['box add', '/add_box', () => doBoxAdd([1, 1, 9, 9]), {ok: true, mask: Object.assign({}, foreign0, {id: 9}), affected: {updated: [{id: 1, species: 'XX'}]}}],
  ['scan keep', '/commit_mask', () => { scanCandidates = [Object.assign({}, cand)]; scanReviewIdx = 0; return scanReviewAccept(); },
   {ok: true, mask: Object.assign({}, foreign0, {id: 9})}],
  ['scan results', '/exemplar', () => doExemplarScan('all'), {ok: true, new_masks: [cand], count: 1, species_scanned: ['PA']}],
  ['refine', '/refine', () => { selectedMask = 0; return doRefine(3, 3, 1); }, foreignMask],
  ['redraw from a box', '/rebox', () => { selectedMask = 0; return doBoxRedraw([1, 1, 9, 9]); }, foreignMask],
  ['re-run of a click', '/refine', () => rerunPrompt(currentData.masks[0]), foreignMask],
  ['undo', '/undo_refine', () => { selectedMask = 0; return undoRefine(); }, Object.assign({remaining_undos: 0}, foreignMask)],
  ['draw edit', '/draw_edit', () => { selectedMask = 0; return submitDrawEdit([{x: 1, y: 1}, {x: 5, y: 5}, {x: 9, y: 1}]); },
   Object.assign({op: 'add'}, foreignMask)],
];
const sentTo = (route, since) => fetchLog.slice(since).some(c => String(c.url).endsWith(route));
for (const [name, route, act, answer] of ROUTES) {
  try {
    // While it waits on a write, the frame changes: nothing goes out.
    reset(pairMasks());
    const held = respondWith({match: '/masks', deferred: true});
    selectedMask = 1; acceptMask(); await settle();
    selectedMask = -1;
    const waiting = act(); await settle();
    const mark = fetchLog.length;
    currentData = {masks: pairMasks(), filename: 'F1.jpg'};
    held.release({status: 200, body: {ok: true}}); await settle(); await waiting;
    want(!sentTo(route, mark), name + ': a request held behind a write is not sent once the frame has changed');
    // The answer arrives after the frame changed: nothing on the page moves.
    reset(pairMasks());
    const slow = respondWith({match: route, deferred: true});
    const asked = act(); await settle();
    const fresh = {masks: pairMasks(), filename: 'F1.jpg'};
    currentData = fresh;
    const before = JSON.stringify(fresh.masks);
    const putsBefore = puts().length;
    slow.release({status: 200, body: answer}); await settle(); await asked; await settle();
    want(currentData === fresh && JSON.stringify(fresh.masks) === before && scanCandidates.length <= 1
         && (route !== '/exemplar' || scanCandidates.length === 0) && puts().length === putsBefore,
         name + ': an answer for a frame no longer loaded changes nothing (masks ' + readAll(fresh)
         + ', candidates ' + scanCandidates.length + ', saves ' + (puts().length - putsBefore) + ')');
  } catch (e) {
    want(false, name + ': the scenario raised ' + (e && e.message));
  }
}

// 17. Scan candidates belong to the frame they were found on.
await bootAt(0);
scanCandidates = [Object.assign({}, cand)]; scanReviewIdx = 0; pendingScanCandidateIdx = -1;
advanceToNextFrame(); await settle();
want(scanCandidates.length === 0 && scanReviewIdx === -1 && pendingScanCandidateIdx === -1,
     'scan candidates are cleared when the next frame arrives (left ' + scanCandidates.length + ')');
pressKey('Enter'); await settle();
want(!fetchLog.some(c => String(c.url).endsWith('/commit_mask')), 'Enter on the next frame keeps no candidate from the last one');
serverHoldsOwn('scan candidates across a frame switch');

// 18. An accept pressed while the same frame reloads is not lost silently.
for (const how of ['Next then Prev', 'the finder landing on this frame']) {
  await bootAt(0);
  gate('F2.jpg'); gate('F1.jpg');
  if (how === 'Next then Prev') { advanceToNextFrame(); await settle(); prevImage(); await settle(); }
  else { unlabeledSet = new Set(['F1.jpg']); jumpNextUnlabeled(); await settle(); }
  selectedMask = 0;
  pressKey('Enter'); await settle();
  const said = statusEl().textContent;
  openGate('F2.jpg'); await settle(); openGate('F1.jpg'); await settle();
  const pageSays = reading(currentData && currentData.masks[0]);
  const serverSays = reading(serverFrames['F1.jpg'].masks[0]);
  want(pageSays === serverSays && (serverSays === 'accepted PA' || said === 'This frame is still loading. Try again in a moment.'),
       'accept during a reload (' + how + '): kept on both sides, or refused with a reason (page ' + pageSays
       + ', server ' + serverSays + ', said ' + JSON.stringify(said) + ')');
}

// 19. A frame that does not load leaves the loaded frame in charge, and says so.
for (const how of ['no answer', 'not found']) {
  await bootAt(0);
  if (how === 'no answer') failGets.add('F2.jpg'); else delete serverFrames['F2.jpg'];
  advanceToNextFrame(); await settle(); await sleep(20); await settle();
  want(images[currentIdx] && images[currentIdx].filename === 'F1.jpg' && currentData && currentData.filename === 'F1.jpg',
       'frame load failed (' + how + '): the index is back on the loaded frame (index on '
       + (images[currentIdx] ? images[currentIdx].filename : 'none') + ')');
  want(img && String(img.src).endsWith('F1.jpg'), 'frame load failed (' + how + '): the picture is back on the loaded frame (got ' + (img && img.src) + ')');
  want(statusEl().textContent.includes('F2.jpg did not load'), 'frame load failed (' + how + '): a plain message says so (got '
       + JSON.stringify(statusEl().textContent) + ')');
}

// 20. Reload IDs does not restate the frame first, so an outside change is
// not reported as an edit made here.
reset([mk(0, 'accepted', 'PA')]);
respondWith({url: '/api/reload_segmentations', status: 200, body: {ok: true, masks: [mk(0, 'accepted', 'PA')], merged: 0, expert: 0}});
await reloadSegmentations(); await settle();
want(puts().length === 0, 'Reload IDs sends no frame save before it reloads (saves sent: ' + puts().length + ')');

// 12. A reopen, a merge and an undo wait for writes still out on their frame.
const blankC = () => ({id: 5, label: 'C', status: 'pending', species: '', category: '', name: '',
                       refinement_clicks: [{x: 1, y: 1, label: 1}]});
const heldWrite = () => {
  reset([blankC()]);
  pendingAddMaskIdx = 0;
  const w = respondWith({deferred: true});
  assignNewMaskSpecies('PA');
  return w;
};
let w = heldWrite(); await settle();
selectImage(0); await settle();
want(!fetchLog.some(c => c.url === '/api/image/F1.jpg' && c.method === 'GET'),
     'a reopen of the frame waits while a write to it is out');
w.release({status: 200, body: {ok: true}}); await settle();
want(fetchLog.some(c => c.url === '/api/image/F1.jpg' && c.method === 'GET'), 'the reopen goes out once the write is answered');
w = heldWrite(); await settle();
mergeSameId(); await settle();
want(!fetchLog.some(c => String(c.url).endsWith('/merge_same_id')), 'a merge waits while a write to its frame is out');
w.release({status: 200, body: {ok: true}}); await settle();
want(fetchLog.some(c => String(c.url).endsWith('/merge_same_id')), 'the merge goes out once the write is answered');
w = heldWrite(); await settle();
selectedMask = 0;
undoRefine(); await settle();
want(!fetchLog.some(c => String(c.url).endsWith('/undo_refine')), 'an undo waits while a write to its frame is out');
w.release({status: 200, body: {ok: true}}); await settle();
want(fetchLog.some(c => String(c.url).endsWith('/undo_refine')), 'the undo goes out once the write is answered');

// 21. A merge held behind an unanswered accept, then a frame switch, on a busy
// server (every mask write and merge answers SLOW_WRITE_MS late). The merged
// mask must stay accepted on the server, as it does with no switch, and a
// return to the frame must show the masks the server holds.
const SLOW_WRITE_MS = 30;
const SLOW_SETTLE_MS = SLOW_WRITE_MS * 10;
const lettered = (id, status) => ({id, label: 'ABCD'[id], status, species: 'PA', category: 'Hard coral',
                                   name: 'Porites astreoides'});
const statuses = (masks) => masks.map(m => m.label + ':' + m.status).join(' ');
async function bootFour() {
  reset([]);
  serverFrames = {'F1.jpg': {masks: [lettered(0, 'pending'), lettered(1, 'accepted'),
                                     lettered(2, 'pending'), lettered(3, 'pending')]},
                  'F2.jpg': {masks: [mk(0, 'rejected', 'PA')]}};
  images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}];
  currentData = null; currentIdx = -1;
  await selectImage(0); await settle();
  serverWriteDelayMs = SLOW_WRITE_MS;
  // Accept C, then merge A with B while that accept is still unanswered.
  selectedMask = 2; acceptMask(); await settle();
  selectedMask = -1;
}
const MERGES = [['merge of two clicked masks', () => doMerge(0, 1)],
                ['merge of checked masks', () => { checkedMaskIds = new Set([0, 1]); return mergeCheckedMasks(); }]];
for (const [name, merge] of MERGES) {
  await bootFour();
  merge(); await settle();
  await sleep(SLOW_SETTLE_MS); await settle();
  const base = statuses(serverFrames['F1.jpg'].masks);
  want(base === 'A:accepted C:accepted D:pending',
       name + ', no frame switch: the server holds the merged mask accepted (got ' + base + ')');

  await bootFour();
  merge(); await settle();
  pressKey('ArrowRight'); await settle();
  await sleep(SLOW_SETTLE_MS); await settle();
  const afterRight = statuses(serverFrames['F1.jpg'].masks);
  want(afterRight === base, name + ', then Right: the server holds what it holds with no switch (got '
       + afterRight + ', want ' + base + ')');
  want(currentData && currentData.filename === 'F2.jpg' && statuses(currentData.masks) === 'A:rejected'
       && statuses(serverFrames['F2.jpg'].masks) === 'A:rejected',
       name + ', then Right: the next frame is untouched on the page and the server');

  await bootFour();
  merge(); await settle();
  pressKey('ArrowRight'); await settle();
  pressKey('ArrowLeft'); await settle();
  await sleep(SLOW_SETTLE_MS); await settle();
  const page = currentData && currentData.filename === 'F1.jpg' ? statuses(currentData.masks) : 'not on F1.jpg';
  const server = statuses(serverFrames['F1.jpg'].masks);
  want(page === server && server === base, name + ', then Right and Left: the page shows the masks the server holds (page '
       + page + ', server ' + server + ')');
}

// 22. Every frame request counts as out on its frame from the moment it is
// asked for: a save started while it still waits behind a write waits for it
// too, and sends nothing until its answer is in. A request with nothing ahead
// of it never waits on itself, and two requests go out in click order.
for (const [name, route, act, answer] of ROUTES) {
  try {
    reset(pairMasks());
    const held = respondWith({match: '/masks', deferred: true});
    selectedMask = 1; acceptMask(); await settle();
    selectedMask = -1;
    const slow = respondWith({match: route, deferred: true});
    const asked = act(); await settle();
    const saving = saveCurrentMasks(); await settle();
    held.release({status: 200, body: {ok: true}}); await settle();
    want(sentTo(route, 0) && puts().length === 1,
         name + ': a save started while the request waits behind a write sends nothing before the request is answered (saves sent: '
         + (puts().length - 1) + ')');
    slow.release({status: 200, body: answer}); await settle(); await asked; await saving; await settle();
    want(puts().length >= 2, name + ': the save goes out once the request is answered');

    reset(pairMasks());
    const alone = respondWith({match: route, deferred: true});
    const lone = act(); await settle();
    want(sentTo(route, 0), name + ': with nothing else out on the frame, the request goes out at once');
    alone.release({status: 200, body: answer}); await settle(); await lone;

    // A request that fails frees its frame at once, not at the time limit.
    reset(pairMasks());
    const doomed = respondWith({match: route, deferred: true});
    const tried = act(); await settle();
    const behind = saveCurrentMasks(); await settle();
    const heldBack = puts().length === 0;
    doomed.release({throws: true}); await settle();
    want(heldBack && puts().length === 1, name + ': a request that fails frees its frame at once, and the save waiting on it goes out');
    await tried; await behind;
  } catch (e) {
    want(false, name + ': the scenario raised ' + (e && e.message));
  }
}
reset(pairMasks());
const first = respondWith({match: '/merge_same_id', deferred: true});
mergeSameId(); await settle();
reloadSegmentations(); await settle();
want(sentTo('/merge_same_id', 0) && !sentTo('/api/reload_segmentations', 0),
     'two requests in a row: the second waits for the first');
first.release({status: 200, body: {ok: true, masks: pairMasks(), merged: 0, refused: []}}); await settle();
want(sentTo('/api/reload_segmentations', 0), 'two requests in a row: the second goes out as soon as the first is answered');

// 23. A frame load the server never answers gives up after the time limit:
// the index and the picture go back to the loaded frame, a plain message says
// so, and edits work again. An answer that comes after the limit changes nothing.
for (const how of ['no answer at all', 'an answer after the time limit']) {
  await bootAt(0);
  if (how === 'no answer at all') hangGets.add('F2.jpg'); else gate('F2.jpg');
  advanceToNextFrame(); await settle();
  await sleep(MASK_WRITE_TIMEOUT_MS + 50); await settle();
  if (how !== 'no answer at all') { openGate('F2.jpg'); await settle(); }
  want(images[currentIdx] && images[currentIdx].filename === 'F1.jpg' && currentData && currentData.filename === 'F1.jpg',
       'frame load timed out (' + how + '): the index is back on the loaded frame (index on '
       + (images[currentIdx] ? images[currentIdx].filename : 'none') + ', masks of '
       + (currentData ? currentData.filename : 'none') + ')');
  want(img && String(img.src).endsWith('F1.jpg'),
       'frame load timed out (' + how + '): the picture is back on the loaded frame (got ' + (img && img.src) + ')');
  const said = statusEl().textContent;
  want(said === 'Try again in a moment: frame F2.jpg did not answer within ' + (MASK_WRITE_TIMEOUT_MS / 1000)
       + ' seconds, so F1.jpg stays open.',
       'frame load timed out (' + how + '): a plain message says so (got ' + JSON.stringify(said) + ')');
  const putsBefore = puts().length;
  selectedMask = 0; acceptMask(); await settle();
  want(puts().length === putsBefore + 1 && reading(serverFrames['F1.jpg'].masks[0]) === 'accepted PA',
       'frame load timed out (' + how + '): edits work again (server holds ' + reading(serverFrames['F1.jpg'].masks[0]) + ')');
}

// 24. A save whose frame still has a request out when its wait ends (a merge
// slower than the wait's cap) sends nothing: restating the frame then would
// undo the request's answer. It says so, and Done and Export stop on it.
const STILL_SAVING = 'Still saving the last change to this frame. Try again in a moment.';
const sameIdAnswer = () => ({status: 200, body: {ok: true, masks: pairMasks(), merged: 0, refused: []}});
reset(pairMasks());
let stuck = respondWith({match: '/merge_same_id', deferred: true});
mergeSameId(); await settle();
const savedWhileOut = await saveCurrentMasks();
want(savedWhileOut === false && puts().length === 0,
     'a save while a request is still out after the wait sends nothing and reports not saved (saves sent: ' + puts().length + ')');
want(statusEl().textContent === STILL_SAVING && statusEl().cls.has('conflict'),
     'a save while a request is still out says so (got ' + JSON.stringify(statusEl().textContent) + ')');
stuck.release(sameIdAnswer()); await settle();
want(await saveCurrentMasks() === true && puts().length === 1, 'once the request is answered the save goes out');
reset(pairMasks());
stuck = respondWith({match: '/merge_same_id', deferred: true});
mergeSameId(); await settle();
await doneAndReturn(); await settle();
want(!windowClosed && statusEl().textContent === 'The window stays open because this frame did not save. ' + STILL_SAVING,
     'Done stops while a request is still out (got ' + JSON.stringify(statusEl().textContent) + ')');
stuck.release(sameIdAnswer()); await settle();
reset(pairMasks());
stuck = respondWith({match: '/merge_same_id', deferred: true});
mergeSameId(); await settle();
await doExport(); await settle();
want(!fetchLog.some(c => c.url === '/api/export_batch') && statusEl().textContent.startsWith('Export stopped because this frame did not save. ')
     && statusEl().textContent.endsWith(STILL_SAVING),
     'Export stops while a request is still out (got ' + JSON.stringify(statusEl().textContent) + ')');
stuck.release(sameIdAnswer()); await settle();

// The reported case: a merge slower than the wait's cap, then Right. The
// server keeps the merged mask accepted, and the message names the frame.
const SLOW_MERGE_MS = MASK_WRITE_TIMEOUT_MS + 150;
reset([]);
serverFrames = {'F1.jpg': {masks: [lettered(0, 'pending'), lettered(1, 'accepted'),
                                   lettered(2, 'pending'), lettered(3, 'pending')]},
                'F2.jpg': {masks: [mk(0, 'rejected', 'PA')]}};
images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}];
currentData = null; currentIdx = -1;
await selectImage(0); await settle();
serverWriteDelayMs = SLOW_MERGE_MS;
doMerge(0, 1); await settle();
pressKey('ArrowRight'); await settle();
await sleep(SLOW_MERGE_MS * 2 + 200); await settle();
const slowMerged = statuses(serverFrames['F1.jpg'].masks);
want(slowMerged === 'A:accepted C:pending D:pending',
     'a merge slower than the wait, then Right: the server keeps the merged mask accepted (got ' + slowMerged + ')');
want(!puts().some(c => String(c.url).includes('F1.jpg')), 'a merge slower than the wait, then Right: no frame save goes to its frame');
want(statusEl().textContent === 'Still saving the last change to frame F1.jpg. Try again in a moment.',
     'a merge slower than the wait, then Right: the message names the frame left (got ' + JSON.stringify(statusEl().textContent) + ')');

// 25. Every frame request gives up when no answer comes within its time
// limit: it frees its frame, changes no mask, and the next save goes out at once.
for (const [name, route, act] of ROUTES) {
  try {
    reset(pairMasks());
    respondWith({match: route, hang: true});
    const before = JSON.stringify(currentData.masks);
    act(); await settle();
    await sleep(FRAME_REQUEST_TIMEOUT_MS + 100); await settle();
    const stillOut = [..._maskWritesInFlight.values()].includes('F1.jpg');
    want(!stillOut && JSON.stringify(currentData.masks) === before,
         name + ': a request with no answer gives up after its time limit, frees its frame and changes no mask');
    const started = Date.now();
    const saved = await saveCurrentMasks();
    want(saved && puts().length === 1 && Date.now() - started < MASK_WRITE_TIMEOUT_MS / 2,
         name + ': after the time limit the next save goes out at once (' + (Date.now() - started) + ' ms)');
  } catch (e) {
    want(false, name + ': the scenario raised ' + (e && e.message));
  }
}
reset(pairMasks());
respondWith({match: '/api/reload_segmentations', hang: true});
reloadSegmentations(); await settle();
await sleep(FRAME_REQUEST_TIMEOUT_MS + 100); await settle();
want(statusEl().textContent === 'Reload error: no answer within ' + (FRAME_REQUEST_TIMEOUT_MS / 1000) + ' seconds',
     'a request that times out says so in plain words (got ' + JSON.stringify(statusEl().textContent) + ')');

// 26. Reload IDs with nothing new: "nothing new" on a frame that already
// shows imported IDs, "no imported IDs yet" only on a frame with none.
const expertA = () => Object.assign(mk(0, 'accepted', 'PA'), {expert_id: {mode: 'EXPERT', code: 'PA'}});
const reviewedA = () => Object.assign(mk(0, 'accepted', 'PA'), {reviews: [{reviewer: 'XY', code: 'PA'}]});
const RELOADS = [
  ['a frame showing an expert ID', [expertA()], 0, 1, 'Reloaded: nothing new for this frame.'],
  ['a frame showing a returned review', [reviewedA()], 0, 0, 'Reloaded: nothing new for this frame.'],
  ['a frame with no import record', [mk(0, 'accepted', 'PA')], 0, 0, 'Reloaded: no imported IDs for this frame yet'],
  ['a frame with one new expert ID', [expertA()], 1, 1, 'Reloaded: 1 mask updated (1 expert ID)'],
];
for (const [label, masks, merged, expert, said] of RELOADS) {
  reset([mk(0, 'accepted', 'PA')]);
  respondWith({url: '/api/reload_segmentations', status: 200, body: {ok: true, masks, merged, expert}});
  await reloadSegmentations(); await settle();
  want(statusEl().textContent === said, 'Reload IDs on ' + label + ' says ' + JSON.stringify(said)
       + ' (got ' + JSON.stringify(statusEl().textContent) + ')');
}
"""

# Every frame request goes through its entry's send(), which carries the time limit.
FRAME_REQUESTS = ('scanReviewAccept', 'doRefine', 'doBoxRedraw', 'rerunPrompt', 'mergeCheckedMasks',
                  'doMerge', 'mergeSameId', 'reloadSegmentations', 'doExemplarScan', 'addMaskAt',
                  'doBoxAdd', 'undoRefine', 'submitDrawEdit')
# The request limit must leave a slow model call on a busy card far inside it.
MIN_FRAME_REQUEST_TIMEOUT_MS = 60000


def check_request_limits():
    """The template's request time limit and its use by all 13 requests."""
    src = H.template_source(WHICH)
    decl = H.top_level_constant(src, 'FRAME_REQUEST_TIMEOUT_MS') or ''
    value = decl.split('=')[-1].strip().rstrip(';').strip()
    check(value.isdigit() and int(value) >= MIN_FRAME_REQUEST_TIMEOUT_MS,
          "FRAME_REQUEST_TIMEOUT_MS is a named constant of at least %d ms (got %r)"
          % (MIN_FRAME_REQUEST_TIMEOUT_MS, value or None))
    bodies = dict(H.named_functions(src))
    for name in FRAME_REQUESTS:
        body = bodies.get(name, '')
        check('request.send(' in body and 'fetch(' not in body,
              "%s sends through its request entry (time limit), never a bare fetch" % name)


def _import_app():
    """Import this suite's app.py fresh, with no graphics card visible."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    src = os.path.join(os.path.dirname(H.template_path(WHICH)), os.pardir)
    src = os.path.abspath(src)
    sys.path.insert(0, src)
    try:
        import app
    finally:
        sys.path.remove(src)
    return app


def _lettered(mid, **extra):
    """One accepted, labelled mask with id `mid` plus `extra` fields."""
    mask = {"id": mid, "label": "ABCD"[mid], "status": "accepted", "species": "PA",
            "category": "Hard coral", "name": "Porites astreoides"}
    mask.update(extra)
    return mask


EXPERT_PA = {"mode": "EXPERT", "code": "PA"}


def _reload_answer(A, root, on_disk, held, dirty=False):
    """Load `on_disk` as the project, replace the frame F1.jpg in memory with
    `held` (marking it edited here when `dirty`), press Reload IDs on F1.jpg
    and return the route's JSON answer."""
    export_dir = tempfile.mkdtemp(prefix="reload_", dir=root)
    path = A._seg_path(export_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(on_disk, fh)
    A.session["export_dir"] = export_dir
    A.session["segmentations"] = A.load_segmentations(export_dir)
    A.session["class_map"] = {}
    A._segmentations_saved_version = A._segmentations_dirty_counter
    A._dirty_frames.clear()
    if held is not None:
        A.session["segmentations"]["F1.jpg"] = {"masks": held}
    if dirty:
        A._mark_dirty(filename="F1.jpg")
    with A.app.test_client() as client:
        res = client.post("/api/reload_segmentations", json={"filename": "F1.jpg"})
    check(res.status_code == 200, "Reload IDs answers 200 (got %d)" % res.status_code)
    return res.get_json() or {}


def check_reload_counts(root):
    """Reload IDs reports only the masks it changed (both apps' route)."""
    A = _import_app()
    A.app.config["TESTING"] = True
    four = [_lettered(i) for i in range(4)]

    body = _reload_answer(A, root, {"F1.jpg": {"masks": four}}, None)
    check(body.get("merged") == 0 and body.get("expert") == 0,
          "a frame with nothing imported reports 0 masks updated (got %s updated, %s expert)"
          % (body.get("merged"), body.get("expert")))

    imported = [_lettered(0, expert_id=EXPERT_PA)] + [_lettered(i) for i in range(1, 4)]
    body = _reload_answer(A, root, {"F1.jpg": {"masks": imported}}, [_lettered(i) for i in range(4)])
    check(body.get("merged") == 1 and body.get("expert") == 1,
          "an expert ID imported outside onto one mask reports 1 mask updated, 1 expert ID (got %s, %s)"
          % (body.get("merged"), body.get("expert")))
    shown = {m["id"]: m for m in body.get("masks", [])}
    check((shown.get(0) or {}).get("expert_id") == EXPERT_PA, "the answer's masks carry the imported expert ID")

    body = _reload_answer(A, root, {"F1.jpg": {"masks": imported}},
                          [_lettered(i, status="pending") for i in range(4)], dirty=True)
    check(body.get("merged") == 1 and body.get("expert") == 1,
          "a frame edited here with one expert ID imported outside still reports 1 updated (got %s, %s)"
          % (body.get("merged"), body.get("expert")))

    body = _reload_answer(A, root, {"F1.jpg": {"masks": imported}},
                          [_lettered(0, expert_id=EXPERT_PA)] + [_lettered(i) for i in range(1, 4)])
    check(body.get("merged") == 0,
          "an expert ID this window already held is not counted again (got %s updated)" % body.get("merged"))

    body = _reload_answer(A, root, {"F2.jpg": {"masks": [_lettered(0)]}}, four)
    check(body.get("merged") == 0,
          "a frame held here and not on disk reports 0 masks updated (got %s)" % body.get("merged"))


def main():
    """Run the scenarios for this app, then for the editMasks twin."""
    root = tempfile.mkdtemp(prefix='frame_switch_')
    try:
        print("-- %s: frame switches --" % WHICH)
        H.run(H.build(WHICH, SCENARIOS, extra_prelude="const LOCK_CASES = [];", real=("scanReviewAccept",)),
              root, _spawn, check)
        print("-- %s: Reload IDs counts --" % WHICH)
        check_reload_counts(root)
        print("-- %s: request time limits --" % WHICH)
        check_request_limits()
    finally:
        shutil.rmtree(root, ignore_errors=True)
    twin_code = 0
    if WHICH == H.APPS[0]:
        print("\n-- editMasks --", flush=True)
        twin_code = _spawn([sys.executable, os.path.abspath(__file__), 'editMasks'], timeout=600).returncode
    print("PASS" if _fail == 0 else "%d FAILED" % _fail)
    return 1 if (_fail or twin_code) else 0


if __name__ == "__main__":
    sys.exit(main())
