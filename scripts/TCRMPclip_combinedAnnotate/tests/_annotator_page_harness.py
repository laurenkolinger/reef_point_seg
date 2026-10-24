"""Module: _annotator_page_harness.py

Purpose: run an annotator page's own JavaScript under node, with no browser
and no server, for the page suites in this folder (test_label_persists_with_
accept.py, test_frame_switch_saves.py). Both annotator templates are served
by Flask from Jinja, but no function body in either holds Jinja, so every
top-level function can be lifted out verbatim and run against:

  - a small fake DOM (the status row, the initials box, anything looked up by
    id), a captured `document.addEventListener` so the page's own keydown
    handler can be pressed, and a stub `Image`;
  - a scripted `fetch`: each test either queues answers (respondWith, matched
    in order, by URL text, or by exact URL; deferred, hung or thrown), or runs
    a fake annotator server (serverFrames) that keeps every frame's masks,
    applies /masks PUTs with the server's label guard and /merge and
    /merge_many with the server's status rule, can apply those writes after a
    delay (a busy server), can hold a frame's GET behind a gate to stand in
    for a slow load, and can leave a frame's GET unanswered for good. Every
    held request fails as an abort when the caller's AbortController fires,
    as a browser's fetch does.

Inputs: an app name ('combinedAnnotate' or 'editMasks') and a block of
scenario JavaScript. Outputs: the assembled harness text, and the node run's
PASS/FAIL lines.

Run: imported by the suites above; not a suite itself.
"""
import os
import re
import subprocess
import tempfile

SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APPS = ('combinedAnnotate', 'editMasks')
NODE_CANDIDATES = ('node', '/usr/bin/node')
NODE_TIMEOUT_S = 120

# Top-level declarations the page's functions read, copied from the template
# when it has them (a fallback only for a name the template lacks, so a test
# never shadows the page's own value). MASK_WRITE_TIMEOUT_MS and
# FRAME_REQUEST_TIMEOUT_MS are deliberately absent: the harness sets its own
# short limits (the request limit above the write limit, as in the page) so
# the no-answer scenarios run in a fraction of a second, and the suites check
# the template's own values.
TEMPLATE_CONSTANTS = ('CATEGORY_CODE_NAMES', 'DEFAULT_TARGET_CATEGORY', 'MASK_WRITE_FIELDS',
                      '_maskWritesInFlight', '_maskWriteState', '_maskWriteSeq',
                      '_timedOutMaskIds', '_maskNotice', '_frameLoadSeq', '_frameLoading')
CONSTANT_FALLBACKS = {
    'CATEGORY_CODE_NAMES': "const CATEGORY_CODE_NAMES = {SPO: 'Sponge', MACA: 'Macroalgae', "
                           "GORG: 'Gorgonian', TURF: 'Turf'};",
}
KEYDOWN_OPENER = "document.addEventListener('keydown', e => {"
KEYDOWN_CLOSER = "});"

_QUOTES = "'\"`"
_OPENERS, _CLOSERS = "([{", ")]}"


def template_path(app):
    """Path of one annotator's template."""
    if app not in APPS:
        raise ValueError("unknown annotator %r; expected one of %s" % (app, ", ".join(APPS)))
    return os.path.join(SCRIPTS, "TCRMPclip_" + app, "src", "templates", "index.html")


def template_source(app):
    """The raw text of one annotator's template."""
    path = template_path(app)
    try:
        with open(path) as fh:
            return fh.read()
    except OSError as exc:
        raise RuntimeError("cannot read the %s template at %s: %s" % (app, path, exc)) from exc


def named_functions(src):
    """Yield (name, source) for every top-level function declaration in `src`.

    A top-level function starts in column one with 'function' or 'async
    function' and ends at the next line that is exactly '}', unless its
    braces already balance on the first line (a one-line function).
    """
    lines = src.splitlines()
    head = re.compile(r"^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")
    i = 0
    while i < len(lines):
        found = head.match(lines[i])
        if not found:
            i += 1
            continue
        opened, closed = lines[i].count('{'), lines[i].count('}')
        j = i
        if not (opened and opened == closed):
            j = i + 1
            while j < len(lines) and lines[j] != "}":
                j += 1
        yield found.group(1), "\n".join(lines[i:j + 1])
        i = j + 1


def all_functions(src):
    """Every top-level function declaration in `src`, verbatim."""
    return "\n".join(body for _, body in named_functions(src))


def top_level_constant(src, name):
    """The template's own `const`/`let` declaration of `name`, or None.

    The declaration runs from its first line to the first line ending in ';'.
    """
    lines = src.splitlines()
    start = re.compile(r"^(?:const|let)\s+%s\b" % re.escape(name))
    for i, line in enumerate(lines):
        if start.match(line):
            j = i
            while j < len(lines) and not lines[j].rstrip().endswith(';'):
                j += 1
            return "\n".join(lines[i:j + 1])
    return None


def constants(src):
    """The declarations of TEMPLATE_CONSTANTS, from the template or a fallback."""
    out = []
    for name in TEMPLATE_CONSTANTS:
        decl = top_level_constant(src, name) or CONSTANT_FALLBACKS.get(name)
        if decl:
            out.append(decl)
    return "\n".join(out)


def keydown_handler(src):
    """The page's main keydown listener, verbatim, or '' when it has none."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line == KEYDOWN_OPENER:
            for j in range(i + 1, len(lines)):
                if lines[j] == KEYDOWN_CLOSER:
                    return "\n".join(lines[i:j + 1])
    return ""


def argument_count(text, open_at):
    """Count the arguments of the call whose '(' is at `open_at` in `text`.

    Commas count only at the call's own nesting level and outside quotes, so
    an object literal or a nested call is one argument.
    """
    depth, quote, commas, k = 0, None, 0, open_at
    while k < len(text):
        ch = text[k]
        if quote:
            if ch == '\\':
                k += 2
                continue
            if ch == quote:
                quote = None
        elif ch in _QUOTES:
            quote = ch
        elif ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth -= 1
            if depth == 0:
                inner = text[open_at + 1:k].strip()
                return commas + 1 if inner else 0
        elif ch == ',' and depth == 1:
            commas += 1
        k += 1
    raise ValueError("unbalanced call at offset %d" % open_at)


def calls_with_more_args(src, callee, max_args, allowed_function):
    """Offsets (line numbers) of calls to `callee` anywhere in `src` that pass
    more than `max_args` arguments outside the body of `allowed_function`.

    The whole text is scanned, so a call inside an arrow function, a listener
    or any other non-top-level code is caught as well as one in a function.
    """
    allowed = [(src.find(body), src.find(body) + len(body))
               for name, body in named_functions(src) if name == allowed_function]
    declaration = re.compile(r"function\s+%s\s*\($" % re.escape(callee))
    call = re.compile(r"(?<![\w$.])%s\(" % re.escape(callee))
    found = []
    for hit in call.finditer(src):
        before = src[max(0, hit.start() - 40):hit.end()]
        if declaration.search(before):
            continue
        if argument_count(src, hit.end() - 1) <= max_args:
            continue
        if any(lo <= hit.start() < hi for lo, hi in allowed):
            continue
        found.append(src.count("\n", 0, hit.start()) + 1)
    return found


PRELUDE = r"""
// ---- a browser just big enough for the page's status row, inputs and keys ----
class FakeEl {
  constructor(id) {
    this.id = id; this.textContent = ''; this.title = ''; this.value = ''; this.innerHTML = '';
    this.style = {}; this.cls = new Set(); this.focused = false; this.disabled = false;
    const self = this;
    this.classList = {
      add: (...c) => c.forEach(x => self.cls.add(x)),
      remove: (...c) => c.forEach(x => self.cls.delete(x)),
      toggle: (c, on) => { const want = on === undefined ? !self.cls.has(c) : !!on;
                           if (want) self.cls.add(c); else self.cls.delete(c); return want; },
      contains: (c) => self.cls.has(c),
    };
  }
  focus() { this.focused = true; }
  select() {}
  blur() {}
  setAttribute() {}
  removeAttribute() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener() {}
}
const _els = {};
const _docHandlers = {};
const document = {
  getElementById(id) { return _els[id] || (_els[id] = new FakeEl(id)); },
  querySelectorAll() { return []; },
  querySelector(sel) { return sel === '.screen.active' ? {id: 'review'} : null; },
  addEventListener(type, fn) { (_docHandlers[type] = _docHandlers[type] || []).push(fn); },
  createElement() { return new FakeEl(''); },
  body: new FakeEl('body'),
};
let windowClosed = false;
const window = { location: { search: '', href: '' }, addEventListener() {},
                 close() { windowClosed = true; } };
const localStorage = { getItem() { return null; }, setItem() {} };
globalThis.Image = class {
  constructor() { this.complete = false; this.onload = null; this._src = ''; }
  set src(v) { this._src = v; }
  get src() { return this._src; }
};

// ---- fetch: queued answers, or a fake annotator server ----
// A queued answer answers the next fetch, or with `match` the next fetch whose
// URL holds that text, or with `url` the next fetch to exactly that URL;
// `deferred` waits for release(), `hang` never answers, and both reject as an
// abort when the caller's AbortController fires.
let fetchLog = [];
let fetchPlan = [];
function respondWith(spec) { fetchPlan.push(spec); return spec; }
// The queued answer the next fetch to `url` takes, or -1 when none matches.
function _planIndex(url) {
  return fetchPlan.findIndex(p => (!p.url || String(url) === p.url)
                                  && (!p.match || String(url).includes(p.match)));
}
function _resp(spec) {
  return { ok: spec.status >= 200 && spec.status < 300, status: spec.status,
           json: async () => { if (spec.body === undefined) throw new SyntaxError('not JSON');
                               return JSON.parse(JSON.stringify(spec.body)); } };
}
function _abortError() {
  const e = new Error('The operation was aborted.'); e.name = 'AbortError'; return e;
}
// Rejects when the caller's AbortController fires, as fetch does: with the
// reason the caller gave abort(), else as a plain abort.
function _abortable(o, rej) {
  if (o.signal) o.signal.addEventListener('abort', () => rej(o.signal.reason || _abortError()));
}
function _waitable(spec, o) {
  return new Promise((res, rej) => {
    spec.release = (s) => (s.throws ? rej(new TypeError('Failed to fetch')) : res(_resp(s)));
    _abortable(o, rej);
  });
}
// The fake server: every frame's masks, the label guard on accept, the merge
// status rule, writes applied after serverWriteDelayMs (a busy server), a
// gate per frame GET (a slow load: the snapshot is taken when the request
// arrives, as a real server does, and delivered when the gate opens), frame
// GETs that fail outright or never answer, and a count of /masks PUTs that
// never answer.
let serverFrames = null;
let serverBatch = null;
let serverHangPuts = 0;
let serverWriteDelayMs = 0;
let failGets = new Set();
let hangGets = new Set();
const getGates = {};
function gate(file) { let open; const p = new Promise(r => { open = r; }); getGates[file] = {p, open}; }
function openGate(file) { const g = getGates[file]; if (g) { delete getGates[file]; g.open(); } }
function _serverPut(file, body) {
  const frame = serverFrames[file];
  if (!frame) return {status: 404, body: {error: 'not found'}};
  const ups = body.updates || {}, rel = body.relabel || {}, needs = [];
  for (const m of frame.masks) {
    const k = String(m.id);
    if (rel[k]) Object.assign(m, rel[k]);
    if (k in ups) { if (ups[k] === 'accepted' && !m.species) needs.push(m.id); else m.status = ups[k]; }
  }
  frame.reviewed = true;
  return {status: 200, body: needs.length ? {ok: true, needs_species: needs} : {ok: true}};
}
// The server's merge (app.py _merge_ids): the lowest id survives, accepted
// when any input was accepted, and every other input is removed.
function _serverMerge(file, ids) {
  const frame = serverFrames[file];
  const order = ids.map(Number).sort((a, b) => a - b);
  const parts = frame ? order.map(i => frame.masks.find(m => m.id === i)) : [];
  if (!frame || order.length < 2 || parts.some(p => !p)) return {status: 404, body: {error: 'masks not found'}};
  if (parts.some(p => p.status === 'rejected')) return {status: 400, body: {error: 'cannot merge a rejected mask'}};
  const survivor = parts[0];
  if (parts.some(p => p.status === 'accepted')) survivor.status = 'accepted';
  const removed = order.slice(1);
  frame.masks = frame.masks.filter(m => !removed.includes(m.id));
  frame.reviewed = true;
  return {status: 200, body: {ok: true, mask: JSON.parse(JSON.stringify(survivor)),
                              removed_id: removed[0], removed_ids: removed}};
}
// A write the server applies once serverWriteDelayMs has passed, whether or
// not the caller gave up meanwhile, as a busy server does.
function _slowWrite(apply, o) {
  if (!serverWriteDelayMs) return Promise.resolve(_resp(apply()));
  return new Promise((res, rej) => {
    setTimeout(() => res(_resp(apply())), serverWriteDelayMs);
    _abortable(o, rej);
  });
}
function _serverAnswer(url, o) {
  const hit = String(url).match(/^\/api\/image\/([^/]+?)(\/masks|\/merge|\/merge_many)?$/);
  if (hit) {
    const file = decodeURIComponent(hit[1]);
    if (hit[2] === '/masks') {
      if (serverHangPuts > 0) { serverHangPuts -= 1; return _waitable({}, o); }
      return _slowWrite(() => _serverPut(file, JSON.parse(o.body)), o);
    }
    // A merge answer the scenario queued itself wins over the fake server's.
    if (hit[2] && _planIndex(url) < 0) {
      const b = JSON.parse(o.body);
      return _slowWrite(() => _serverMerge(file, b.mask_ids || [b.mask_id_a, b.mask_id_b]), o);
    }
    if (hit[2]) return null;
    if (failGets.has(file)) return Promise.reject(new TypeError('Failed to fetch'));
    if (hangGets.has(file)) return _waitable({}, o);
    const snapshot = serverFrames[file]
      ? {status: 200, body: JSON.parse(JSON.stringify(serverFrames[file]))}
      : {status: 404, body: {error: 'not found'}};
    const answer = () => _resp(snapshot);
    const g = getGates[file];
    if (!g) return Promise.resolve(answer());
    return new Promise((res, rej) => { g.p.then(() => res(answer())); _abortable(o, rej); });
  }
  if (url === '/api/review_batch' && serverBatch) return Promise.resolve(_resp({status: 200, body: serverBatch}));
  return null;
}
globalThis.fetch = (url, opts) => {
  const o = opts || {};
  fetchLog.push({ url, method: o.method || 'GET', body: o.body ? JSON.parse(o.body) : null });
  if (serverFrames) {
    const served = _serverAnswer(url, o);
    if (served) return served;
  }
  const at = _planIndex(url);
  const spec = at >= 0 ? fetchPlan.splice(at, 1)[0] : { status: 200, body: { ok: true } };
  if (spec.deferred || spec.hang) return _waitable(spec, o);
  if (spec.throws) return Promise.reject(new TypeError('Failed to fetch'));
  return Promise.resolve(_resp(spec));
};
async function settle() { for (let i = 0; i < 40; i++) await new Promise(r => setImmediate(r)); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ---- the page's own globals, as a fresh frame leaves them ----
const MASK_WRITE_TIMEOUT_MS = 200;
const FRAME_REQUEST_TIMEOUT_MS = 500;
const REVIEW_TAGGING_ENABLED = false;
const REVIEW_LOCKED_MSG = 'Flagging for expert review is temporarily unavailable. VICARIUS is accepting no new review tags.';
const ORCHESTRATOR_URL = '';
const SESSION_MODE = 'configure';
const TARGET_SPECIES = ['PA', 'AT', 'ZZZ', 'LRN'];
const TARGET_LABELS = [
  {code: 'PA', name: 'Porites astreoides', category: 'Hard coral'},
  {code: 'AT', name: 'Agaricia tenuifolia', category: 'Hard coral'},
  {code: 'ZZZ', name: '', category: ''},
  {code: 'LRN', name: '', category: 'Hard coral'},
];
const _knownLabelNames = {PA: 'Porites astreoides', AT: 'Agaricia tenuifolia', LRN: 'Learned name'};
const BINOMIAL_COMMON_WORDS = [];
let images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}];
let currentIdx = 0;
let currentData = null;
let img = null;
let _prefetchImg = null;
let pendingAddMaskIdx = -1;
let pendingScanCandidateIdx = -1;
let scanCandidates = [];
let scanReviewIdx = -1;
let stickyAddMode = false;
let stickyBoxMode = false;
let autoAdvance = false;
let autoCenter = false;
let selectedMask = -1;
let lastListClickIdx = -1;
let mergeFirstId = -1;
let refineInFlight = false;
let checkedMaskIds = new Set();
let recentlyDeletedIds = new Set();
let interactionMode = 'select';
let statusTimer = null;
let _flushTimer = null;
let numKeyDown = null;
let armedReview = false;
let _reviewerPushed = 'LO';
let unlabeledOnly = false;
let unlabeledSet = new Set();
let _batchInfoText = '';
let bgPollTimer = null;
let _committingCandidate = false;
"""

STUBS = r"""
// ---- browser-only helpers the scenarios do not exercise ----
let flushes = 0;
let exportConfirmOpened = 0;
function scheduleFlush() { flushes++; }
function render() {}
function renderMaskList() {}
function flashMode() {}
function showCatPopup() {}
function hideCatPopup() { pendingAddMaskIdx = -1; pendingScanCandidateIdx = -1; }
function setInteractionMode(m) { interactionMode = m; }
function resumeStickyMode() {}
function advanceToNextOrFrame() {}
function refreshUnlabeledSet() {}
function renderScanCandidates() {}
function setScanStatus() {}
function centerOnMask() {}
function _patchCatPopupNames() {}
function fitImage() {}
function updateModeButtons() {}
function updateCounter() {}
function _updateStickyAddIndicator() {}
function highlightQuickBtn() {}
function _updatePriorScreenedNote() {}
function hidePriorMaskTip() {}
function _timerStart() {}
function _timerRecordPrev() {}
function _updateLoopBanner() {}
function showScreen() {}
function openExportConfirm() { exportConfirmOpened++; }
function centerOnBBox() {}
let keptCandidates = [];
async function scanReviewAccept() { keptCandidates.push(Object.assign({}, scanCandidates[scanReviewIdx])); }
"""

HEAD = r"""
let fails = 0;
// A save path that waits forever leaves node with nothing to do, and node then
// exits 0 on its own. Count that as a failure, never as a clean finish.
let mainDone = false;
process.on('beforeExit', () => {
  if (mainDone) return;
  console.log('FAIL the scenarios stopped early: a save path is still waiting on an answer');
  process.exit(1);
});
// A page function that throws out of a promise nobody awaits (an uncaught
// error in the browser) is a failure too.
process.on('unhandledRejection', (e) => {
  console.log('FAIL an error escaped the page: ' + String(e && e.message || e).split('\n')[0]);
  fails++;
});
function want(cond, name) {
  if (cond) { console.log('PASS ' + name); } else { console.log('FAIL ' + name); fails++; }
}
const statusEl = () => document.getElementById('status-text');
// A frame with these masks loaded, as selectImage leaves it: the data carries
// the file it came from.
function reset(masks, file) {
  fetchLog = []; fetchPlan = []; flushes = 0; serverFrames = null; serverBatch = null;
  serverHangPuts = 0; serverWriteDelayMs = 0; failGets = new Set(); hangGets = new Set();
  windowClosed = false; exportConfirmOpened = 0;
  scanReviewIdx = -1; _committingCandidate = false;
  for (const k of Object.keys(_els)) delete _els[k];
  images = [{filename: 'F1.jpg'}, {filename: 'F2.jpg'}];
  const name = file || 'F1.jpg';
  currentIdx = images.findIndex(f => f.filename === name);
  currentData = {masks, filename: name}; selectedMask = -1;
  pendingAddMaskIdx = -1; pendingScanCandidateIdx = -1; scanCandidates = []; keptCandidates = [];
  recentlyDeletedIds = new Set(); checkedMaskIds = new Set(); unlabeledSet = new Set();
}
// The page as it is after a switch to another frame has finished loading.
function switchTo(file, masks) {
  if (!images.some(f => f.filename === file)) images.push({filename: file});
  currentIdx = images.findIndex(f => f.filename === file);
  currentData = {masks, filename: file};
}
function pressKey(key, repeat) {
  for (const h of (_docHandlers.keydown || [])) {
    h({key, repeat: !!repeat, target: {tagName: 'BODY', type: ''}, ctrlKey: false,
       metaKey: false, shiftKey: false, altKey: false, preventDefault() {}, stopPropagation() {}});
  }
}
const puts = () => fetchLog.filter(c => c.method === 'PUT');
"""

TAIL = r"""
  mainDone = true;
  process.exit(fails ? 1 : 0);
}
main().catch(e => { console.log('FAIL harness error: ' + (e && e.stack || e)); process.exit(1); });
"""


def build(app, scenarios, extra_prelude="", real=()):
    """Assemble the node script: fake browser, the template's code, stubs,
    the page's keydown handler, and `scenarios` (the body of async main).
    `real` names stubbed functions to leave out, so the page's own run."""
    src = template_source(app)
    stubs = "\n".join(line for line in STUBS.splitlines()
                      if not any(re.match(r"^(?:async\s+)?function\s+%s\(" % re.escape(n), line)
                                 for n in real))
    return "\n".join([PRELUDE, extra_prelude, constants(src), all_functions(src), stubs,
                      keydown_handler(src), HEAD, "async function main() {", scenarios, TAIL])


def find_node(spawn):
    """A working node binary, or None; `spawn` runs a child (subprocess.run)."""
    for cand in NODE_CANDIDATES:
        try:
            spawn([cand, "--version"], capture_output=True, check=True)
            return cand
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def run(script, workdir, spawn, check):
    """Run `script` under node and report each PASS/FAIL line through `check`.

    The no-answer scenarios log one expected console warning; only its first
    line is shown on a clean run, and all of stderr when node failed.
    """
    node = find_node(spawn)
    check(node is not None, "node is available to run the page scenarios")
    if node is None:
        return
    fd, path = tempfile.mkstemp(suffix='.js', dir=workdir)
    with os.fdopen(fd, 'w') as fh:
        fh.write(script)
    res = spawn([node, path], capture_output=True, text=True, timeout=NODE_TIMEOUT_S)
    for line in res.stdout.splitlines():
        if line.startswith('PASS '):
            check(True, line[5:])
        elif line.startswith('FAIL '):
            check(False, line[5:])
        else:
            print("    " + line)
    err = res.stderr.strip()
    if err:
        print("    [node stderr] " + (err[:3000] if res.returncode else err.splitlines()[0]))
    check(res.returncode == 0, "the page scenarios finished cleanly (node exit %d)" % res.returncode)
