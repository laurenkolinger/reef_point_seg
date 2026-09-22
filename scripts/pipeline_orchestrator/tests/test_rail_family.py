"""
Task T7: the family-aware rail in the pipeline window.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_rail_family.py

Every test drives the SHIPPED scripts/pipeline_orchestrator/static/orchestrator.js
inside a Node VM against a small stand-in for the page, built to match
templates/index.html: the ten rail tiles in their markup order, the two section
headings, one panel per tile, the Train panel's task select and the Trained runs
card. Nothing here reaches the network: every fetch stays pending, so the
asynchronous halves of the panel loaders never run and only the synchronous
wiring under test is exercised.

Covers:
  RAIL      a detection project shows Train, Evaluate and Inference and hides
            the six stages its family does not have; a segmentation project
            minted from a training set keeps them on the rail marked skipped;
            an ordinary project keeps everything; a heading with nothing left
            under it is hidden.
  ACTIVE    exactly one tile is ever painted as the current stage, including
            after a tile is taken out of the flow.
  RESUME    a project opens on the step the server recorded, archived chain
            slots 4 and 5 open the annotator's tile, and a training project is
            floored at Train instead of being pinned to it.
  PANELS    every reachable tile opens its panel, and a step that reaches no
            panel keeps the open one instead of blanking the window.
  TASK      the Training task control gives back the value it had when a box
            dataset pinned it to Detector.
  NAMES     the Train class picker resolves a code to its full name with the
            code beside it as a chip, and leaves an unresolved code bare.
  FINETUNE  Fine-tune is refused, and the button disabled with its reason, when
            a run's task does not match the task this project trains.
  STAGES    every stage is numbered off the rail this project shows, so a
            detection project's Model Inference panel is stage 3 of 3 and a
            segmentation project's is stage 7 of 7; each panel's number answers
            to its own id and survives every other stage being opened; a
            heading that spells its own number out has it rewritten; and no
            folder token or internal slot number is left in a sentence the
            window renders.
  SEEDS     a Refine round counts on one 13 px line everything it left out,
            the two drops the seeder used to make in silence included, names
            each of them on a focusable glyph rather than in a paragraph, and
            reads as a warning when it seeded no mask.
  PROMOTE   a run the server refuses on its own gates opens a confirmation
            that states all three verdicts in plain words, names every class
            the per-class gate flagged in full with its code beside it, and
            re-sends the promotion with confirm_gate on it when confirmed.
  HONESTY   a Stop the server refused (HTTP 409) reaches the screen word for
            word and leaves the tile running; a step whose process VEGA no
            longer owns says so and keeps its mark; another project's finished
            run is never claimed here, however it exited.
  GATES     the Rounds ledger's Gates column reads as plain words with the
            classes that dropped counted, never the ledger's own token.
  ROWS      the per-site table Choose Images reports reads at the table size
            and stripes, and its shortfall lines are focusable hints.
  EVALUATE  a saved Evaluate setting that is not the recommended one says so
            beside the field it belongs to, from project state and from
            /api/step/7/run_info alike, and the replaced-confidence note is one
            line with its reason on a tooltip.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_TPL = os.path.join(_ORCH, "templates", "index.html")

# ── tiny test harness (house style: no pytest) ──────────────────────
_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - a suite reports, never crashes
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _node():
    for cand in ("/usr/bin/node", "node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:  # noqa: BLE001 - probing for an optional binary
            continue
    return None


# ── the stand-in page ───────────────────────────────────────────────
# A small DOM with only what orchestrator.js touches: elements carry
# attributes, classes, a parent and children, and the three selector shapes the
# rail uses resolve against the real tree rather than a stub.
_NODE_DRIVER = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const scenario = fs.readFileSync(process.argv[3], 'utf8');

class El {
  constructor(tag, id) {
    this.tagName = String(tag || 'div').toUpperCase();
    this.id = id || '';
    this.style = {}; this.dataset = {}; this.attrs = {};
    this.children = []; this.parentNode = null;
    this.value = ''; this.checked = false; this.disabled = false; this.hidden = false;
    this.title = ''; this.textContent = ''; this.innerHTML = ''; this.options = [];
    this.selectedIndex = -1; this.tabIndex = -1;
    this._classes = new Set();
    const self = this;
    this.classList = {
      add() { for (const c of arguments) if (c) self._classes.add(c); },
      remove() { for (const c of arguments) self._classes.delete(c); },
      toggle(c, f) {
        const on = (f === undefined) ? !self._classes.has(c) : !!f;
        if (on) self._classes.add(c); else self._classes.delete(c);
        return on;
      },
      contains(c) { return self._classes.has(c); },
    };
  }
  get className() { return Array.from(this._classes).join(' '); }
  set className(v) { this._classes = new Set(String(v || '').split(/\s+/).filter(Boolean)); }
  // title is the one attribute a browser keeps in step with its property, and
  // the page both assigns el.title and removes the attribute, so the stand-in
  // reflects the two into each other.
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'title') this.title = String(v); }
  getAttribute(k) {
    if (k === 'title' && this.attrs[k] === undefined) return this.title || null;
    return this.attrs[k] === undefined ? null : this.attrs[k];
  }
  removeAttribute(k) { delete this.attrs[k]; if (k === 'title') this.title = ''; }
  hasAttribute(k) { return this.attrs[k] !== undefined; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  insertBefore(node, ref) {
    const i = this.children.indexOf(ref);
    node.parentNode = this;
    if (i < 0) this.children.push(node); else this.children.splice(i, 0, node);
    return node;
  }
  removeChild(c) { this.children = this.children.filter(x => x !== c); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  get childElementCount() { return this.children.length; }
  querySelector(sel) { const r = this.querySelectorAll(sel); return r.length ? r[0] : null; }
  querySelectorAll(sel) { return descendants(this).filter(el => matchesCompound(el, lastCompound(sel))); }
  addEventListener() {} removeEventListener() {} dispatchEvent() { return true; }
  closest() { return null; }
  scrollIntoView() {}
}

// A text node is an element with no tag, so a rebuilt heading keeps its words
// in order beside the span that holds its number.
class TextNode {
  constructor(text) {
    this.tagName = '#TEXT'; this.id = ''; this.children = []; this.parentNode = null;
    this.textContent = String(text === undefined ? '' : text);
    this._classes = new Set();
    this.classList = { add() {}, remove() {}, toggle() {}, contains() { return false; } };
  }
  get className() { return ''; }
}

function descendants(root) {
  const out = [];
  (function walk(el) { for (const c of el.children) { out.push(c); walk(c); } })(root);
  return out;
}

function lastCompound(sel) {
  const parts = String(sel).trim().split(/\s+/);
  return parts[parts.length - 1];
}

function parseCompound(text) {
  const out = { tag: '', id: '', classes: [], attrs: [] };
  const re = /([#.]?[A-Za-z0-9_-]+)|\[([A-Za-z0-9_-]+)(?:="([^"]*)")?\]/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m[2] !== undefined) { out.attrs.push([m[2], m[3]]); continue; }
    const t = m[1];
    if (t[0] === '#') out.id = t.slice(1);
    else if (t[0] === '.') out.classes.push(t.slice(1));
    else out.tag = t.toUpperCase();
  }
  return out;
}

function matchesCompound(el, compound) {
  const c = parseCompound(compound);
  if (c.tag && el.tagName !== c.tag) return false;
  if (c.id && el.id !== c.id) return false;
  for (const cl of c.classes) if (!el.classList.contains(cl)) return false;
  for (const pair of c.attrs) {
    const got = el.getAttribute(pair[0]);
    if (got === null) return false;
    if (pair[1] !== undefined && got !== pair[1]) return false;
  }
  return true;
}

const registry = new Map();
function reg(el) { if (el.id) registry.set(el.id, el); return el; }

const document = {
  body: new El('body', 'body'),
  documentElement: new El('html', ''),
  getElementById(id) {
    // Panels are exactly the ones templates/index.html carries, and the
    // fixture builds every one of them, so an unknown panel answers null the
    // way a browser does. Every other id is created on demand, because the
    // page carries far more of them than any one scenario needs.
    if (registry.has(id)) return registry.get(id);
    // An element the code under test made and put on the page is found by its
    // id from then on, the way a browser finds it.
    const onPage = descendants(document.body).find(el => el.id === id);
    if (onPage) return onPage;
    // Three more families answer null unless the fixture built them, because
    // the code under test decides what to do by whether the window ships one:
    // a per-panel stage number, the published stage number, and the note
    // beside an Evaluate field.
    if (/^(panel-|step[A-Za-z0-9]*-stage-num$|stage-display-|s7-[a-z-]+-off-recommended$)/.test(id)) {
      return null;
    }
    registry.set(id, new El('div', id));
    return registry.get(id);
  },
  createElement(tag) { return new El(tag, ''); },
  createTextNode(text) { return new TextNode(text); },
  querySelector(sel) { const r = this.querySelectorAll(sel); return r.length ? r[0] : null; },
  querySelectorAll(sel) {
    const seen = new Set();
    const out = [];
    const consider = (el) => {
      if (seen.has(el)) return;
      seen.add(el);
      if (matchesCompound(el, lastCompound(sel))) out.push(el);
    };
    descendants(document.body).forEach(consider);
    registry.forEach(el => { consider(el); descendants(el).forEach(consider); });
    return out;
  },
  addEventListener() {},
};
document.body.dataset.projectLoaded = 'false';

// ── the rail, in templates/index.html order ────────────────────────
// [key, title, display number or glyph, kind]
const RAIL = [
  ['1', 'Fetch Points', '1', 'step'],
  ['2', 'Fetch Labels', '2', 'step'],
  ['3', 'Choose Images', '3', 'step'],
  ['step4test', 'Place + Segment', '4', 'step'],
  ['6', 'Train Model', '5', 'step'],
  ['7', 'Evaluate Model', '6', 'step'],
  ['8', 'Model Inference', '7', 'step'],
  ['step4loop', 'Refine (model loop)', '↻', 'step'],
  ['review', 'Review', '', 'section'],
  ['editmasks', 'Edit Masks', '✎', 'step'],
  ['expertids', 'Expert Review I/O', '★', 'off'],
  ['tools', 'Tools', '', 'section'],
  ['matrix', 'Coverage Matrix', '▦', 'tool'],
];

function makeTile(key, title, num, kind) {
  const li = new El('li', '');
  if (kind === 'step') li.setAttribute('data-step', key);
  if (kind === 'off') { li.setAttribute('data-step-disabled', key); li.classList.add('step-disabled'); }
  if (kind === 'section') li.classList.add('nav-section');
  if (kind === 'tool') li.classList.add('nav-tool');
  const circle = new El('div', '');
  circle.classList.add('step-circle');
  circle.textContent = num;
  if (/^\d+$/.test(key) && num !== key) circle.dataset.num = num;
  const info = new El('div', '');
  info.classList.add('step-info');
  const t = new El('div', '');
  t.classList.add('step-title');
  t.textContent = title;
  info.appendChild(t);
  if (kind === 'step' || kind === 'off') {
    const st = new El('div', /^\d+$/.test(key) ? ('step' + key + '-status-text') : (key + '-status-text'));
    st.classList.add('step-status');
    st.textContent = 'locked';
    info.appendChild(reg(st));
  }
  li.appendChild(circle);
  li.appendChild(info);
  return li;
}

function makeRail() {
  const nav = reg(new El('ul', 'step-nav'));
  nav.classList.add('step-nav');
  RAIL.forEach(row => nav.appendChild(makeTile(row[0], row[1], row[2], row[3])));
  document.body.appendChild(nav);
  // One panel per tile that has one, the way index.html carries them.
  ['1', '2', '3', '6', '7', '8', 'step4test', 'step4loop', 'editmasks', 'expertids'].forEach(key => {
    const p = reg(new El('div', 'panel-' + key));
    p.classList.add('step-panel');
    document.body.appendChild(p);
  });
  return nav;
}

// The Training task select and the line under it.
function makeTaskControl(value) {
  const sel = reg(new El('select', 's6-task'));
  sel.value = value || 'segment';
  sel.title = 'What this run trains.';
  const note = reg(new El('div', 's6-task-note'));
  note.textContent = 'Changing this swaps the base model list to the weights that match.';
  document.body.appendChild(sel);
  document.body.appendChild(note);
  return sel;
}

// A training-source select whose options carry the kind the server reported.
function makeSourceSelect(kinds) {
  const sel = reg(new El('select', 's6-source'));
  sel.options = kinds.map(k => { const o = new El('option', ''); o.value = k.value; o.dataset.kind = k.kind; return o; });
  sel.selectedIndex = 0;
  sel.value = sel.options.length ? sel.options[0].value : '';
  document.body.appendChild(sel);
  return sel;
}

function selectSource(sel, index) {
  sel.selectedIndex = index;
  sel.value = sel.options[index].value;
}

// A panel heading, in either shape a window can carry it in:
//   'inline'   "Step <span class=stage-num>3</span>: Model Inference", which is
//              what templates/index.html ships.
//   'plain'    the number spelled into the heading words and no span at all,
//              which is the shape the pass under test rewrites.
function makeHeading(key, words, shape, stageText) {
  const panel = document.getElementById('panel-' + key);
  const head = new El('div', '');
  head.classList.add('panel-head');
  const h2 = new El('h2', '');
  if (shape === 'inline') {
    const span = new El('span', '');
    span.classList.add('stage-num');
    span.textContent = stageText === undefined ? '' : stageText;
    h2.appendChild(span);
  } else {
    h2.textContent = words;
  }
  head.appendChild(h2);
  panel.appendChild(head);
  return head;
}

// One Evaluate field inside its definition row, so the note beside it has a
// row to be appended to.
function makeEvalField(id, value) {
  const group = new El('div', '');
  group.classList.add('form-group');
  const input = reg(new El('input', id));
  input.value = value === undefined ? '' : String(value);
  group.appendChild(input);
  document.body.appendChild(group);
  return input;
}

// The Trained runs card with one action button per run.
function makeRunsCard(runs) {
  const card = reg(new El('div', 's6-runs-card'));
  runs.forEach(r => {
    const btn = new El('button', '');
    btn.setAttribute('data-action', 'finetune');
    btn.setAttribute('data-run-dir', r.dir);
    btn.setAttribute('data-task', r.task);
    btn.title = 'Start a training run from this run’s weights.';
    btn.disabled = !!r.disabled;
    card.appendChild(btn);
  });
  document.body.appendChild(card);
  return card;
}

// The "Continue from a VEGA model" advanced disclosure (2026-09-21 starting-
// model redesign): the select applyStep6ModelPath and onVegaFinetuneFrom
// write into, starting with only the "None" option, plus the two fields and
// the hint onStep6ContinueChange stashes, restores and rewrites.
function makeContinueFrom() {
  const details = reg(new El('details', 's6-continue-details'));
  details.open = false; // El has no default for this attribute; a real <details> starts closed.
  const summary = reg(new El('summary', 's6-continue-summary'));
  summary.textContent = 'Continue from a VEGA model (advanced)';
  details.appendChild(summary);
  const sel = reg(new El('select', 's6-continue-from'));
  const none = new El('option', '');
  none.value = '';
  sel.appendChild(none);
  details.appendChild(sel);
  document.body.appendChild(details);
  const modelNote = reg(new El('div', 's6-model-note'));
  modelNote.textContent = 'Ultralytics trained it on everyday photographs; VEGA teaches it your classes '
    + 'from the training data above.';
  document.body.appendChild(modelNote);
  const freeze = reg(new El('input', 's6-freeze'));
  freeze.value = '';
  document.body.appendChild(freeze);
  const lr0 = reg(new El('input', 's6-lr0'));
  lr0.value = '0.01';
  document.body.appendChild(lr0);
  return { details: details, summary: summary, sel: sel, modelNote: modelNote, freeze: freeze, lr0: lr0 };
}

// What the rail looks like right now.
function railInfo() {
  const nav = document.getElementById('step-nav');
  return nav.children.map(li => ({
    step: li.getAttribute('data-step'),
    disabled: li.getAttribute('data-step-disabled'),
    section: li.classList.contains('nav-section'),
    active: li.classList.contains('active'),
    display: li.style.display === undefined ? '' : li.style.display,
    title: (li.querySelector('.step-title') || { textContent: '' }).textContent,
    status: (li.querySelector('.step-status') || { textContent: '' }).textContent,
    classes: li.className,
  }));
}

function activePanels() {
  return document.querySelectorAll('.step-panel')
    .filter(p => p.classList.contains('active')).map(p => p.id);
}

// A full steps block, so populateConfigs() finds what it reads.
function makeSteps(statuses) {
  const out = {};
  ['1', '2', '3', '4', '5', '6', '7', '8'].forEach(k => {
    out[k] = { name: 'step ' + k, status: statuses[k] || 'locked', config: {}, outputs: {} };
  });
  return out;
}

// Timers are stubs: the page starts a 2.5 second tile poller and several
// pollers of its own, and a live interval would keep Node running long after
// the scenario finished. Nothing under test needs a timer to fire.
const timers = [];
const noTimer = (fn, ms) => { timers.push([fn, ms]); return timers.length; };

const sandbox = {
  document, console, timers,
  setTimeout: noTimer, clearTimeout() {}, setInterval: noTimer, clearInterval() {},
  fetch: () => new Promise(() => {}), alert() {}, prompt() { return null; },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  location: { href: 'http://127.0.0.1:5050/', search: '', hostname: '127.0.0.1', protocol: 'http:' },
  URLSearchParams, encodeURIComponent, decodeURIComponent,
  El, TextNode, makeRail, makeTaskControl, makeSourceSelect, selectSource, makeRunsCard, makeContinueFrom,
  makeHeading, makeEvalField, railInfo, activePanels, makeSteps, registry, reg,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'orchestrator.js' });
const out = vm.runInContext(scenario, sandbox, { filename: 'scenario.js' });
// A scenario that drives a request returns a promise; one that only reads the
// page returns its answer outright. Both are printed the same way.
Promise.resolve(out).then(value => {
  process.stdout.write(JSON.stringify(value === undefined ? null : value));
}, err => {
  process.stderr.write(String((err && err.stack) || err));
  process.exit(1);
});
"""


def _drive(scenario):
    """Run one scenario against orchestrator.js in Node; return its JSON result."""
    node = _node()
    check(node is not None, "node not available")
    tmp = tempfile.mkdtemp(prefix="rail_family_")
    try:
        driver = os.path.join(tmp, "driver.js")
        scen = os.path.join(tmp, "scenario.js")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(_NODE_DRIVER)
        with open(scen, "w", encoding="utf-8") as fh:
            fh.write(scenario)
        r = subprocess.run([node, driver, _JS, scen], capture_output=True, text=True, timeout=30)
        check(r.returncode == 0, f"node scenario failed:\n{r.stderr}")
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# The three project shapes every rail test opens.
_DETECTION_STATE = """
state = {
  is_training_project: true, project_type: 'bbox_training',
  training_source: { kind: 'bbox', name: 'coral_intel_subset', path: '/x/coral_intel_subset',
                     task_default: 'detect' },
  training_source_label: 'box dataset', training_source_name: 'coral_intel_subset',
  current_step: 6, steps: makeSteps({ 6: 'completed', 7: 'pending', 8: 'pending' }),
};
"""

_TRAINING_SET_STATE = """
state = {
  is_training_project: true, project_type: 'composite_training',
  composite: { name: 'qs20260902_pa_ss', source_projects: [] },
  training_source_label: 'training set', training_source_name: 'qs20260902_pa_ss',
  current_step: 6, steps: makeSteps({ 6: 'completed', 7: 'pending', 8: 'pending' }),
};
"""

_ANNOTATION_STATE = """
state = {
  is_training_project: false, project_type: 'annotation',
  current_step: 3, steps: makeSteps({ 1: 'completed', 2: 'completed', 3: 'running' }),
};
"""

_OPEN = "makeRail(); makeTaskControl('segment'); enterApp();"


# A JavaScript object entry mapping a short code to a two-word binomial, which
# is what a hardcoded species table looks like.
_HARDCODED_NAME_RE = r"""['"]?[A-Z]{2,5}['"]?\s*:\s*['"][A-Z][a-z]+ [a-z]+['"]"""


def _strip_comments(js):
    """JavaScript source with its block and line comments removed, so a rule
    about what the page renders is not tripped by a note to a reader."""
    no_block = re.sub(r"/\*[\s\S]*?\*/", "", js)
    return re.sub(r"(?m)//.*$", "", no_block)


def _tiles(info):
    """The rail rows keyed by their step or disabled key."""
    return {(row["step"] or row["disabled"]): row for row in info if row["step"] or row["disabled"]}


# ── RAIL ────────────────────────────────────────────────────────────
def test_detection_rail_shows_only_the_stages_that_family_has():
    """A box-dataset project trains a detector, so the four annotation stages
    and the two mask tools are not on its rail at all."""
    out = _drive(_DETECTION_STATE + _OPEN + """
({ rail: railInfo(), order: getNavOrder(), family: projectFamily(state), panels: activePanels() });
""")
    check(out["family"] == "detection", f"a bbox training_source is the detection family: {out['family']}")
    check(out["order"] == ["6", "7", "8"],
          f"a detection project offers Train, Evaluate and Inference only: {out['order']}")
    tiles = _tiles(out["rail"])
    for key in ("1", "2", "3", "step4test", "step4loop", "editmasks"):
        check(tiles[key]["display"] == "none",
              f"{key} is not part of the detection family and must be hidden: {tiles[key]}")
        check(tiles[key]["disabled"] == key,
              f"{key} must be out of the flow (data-step-disabled): {tiles[key]}")
    for key in ("6", "7", "8"):
        check(tiles[key]["display"] != "none", f"{key} must stay on the rail: {tiles[key]}")
        check(tiles[key]["step"] == key, f"{key} must stay in the flow: {tiles[key]}")
    check(out["panels"] == ["panel-6"], f"the detection project opens on Train: {out['panels']}")


def test_segmentation_training_set_rail_keeps_its_stages_and_says_skipped():
    """A project minted from a training set is the segmentation family, whose
    annotation stages exist; they were done in the projects the set was built
    from, so they stay on the rail with an honest caption."""
    out = _drive(_TRAINING_SET_STATE + _OPEN + """
({ rail: railInfo(), order: getNavOrder(), family: projectFamily(state) });
""")
    check(out["family"] == "segmentation",
          f"a project with no bbox source is the segmentation family: {out['family']}")
    check(out["order"] == ["6", "7", "8"],
          f"a training-set project still runs Train onward only: {out['order']}")
    tiles = _tiles(out["rail"])
    for key in ("1", "2", "3", "step4test", "step4loop", "editmasks"):
        check(tiles[key]["display"] != "none",
              f"{key} belongs to the segmentation family and must stay visible: {tiles[key]}")
        check(tiles[key]["status"] == "skipped",
              f"{key} must say skipped, not locked: {tiles[key]}")
        check("step-disabled" in tiles[key]["classes"],
              f"{key} must carry the disabled look: {tiles[key]}")


def test_ordinary_project_rail_is_untouched():
    """An annotation project keeps every stage and opens where it left off."""
    out = _drive(_ANNOTATION_STATE + _OPEN + """
({ rail: railInfo(), order: getNavOrder(), family: projectFamily(state), panels: activePanels() });
""")
    check(out["family"] == "segmentation", f"an annotation project is segmentation: {out['family']}")
    check(out["order"] == ["1", "2", "3", "step4test", "6", "7", "8", "step4loop", "editmasks"],
          f"every stage stays on an ordinary project's rail: {out['order']}")
    for row in out["rail"]:
        check(row["display"] != "none", f"nothing is hidden on an ordinary project: {row}")
    check(out["panels"] == ["panel-3"], f"the project opens on the step it recorded: {out['panels']}")


def test_a_heading_with_nothing_under_it_is_hidden():
    """A section heading whose every tile is hidden goes with them."""
    out = _drive("""
const nav = makeRail();
// Take away the tile that keeps the Review heading company, so the heading's
// only remaining tile is the one the rail hides.
const expert = nav.children.filter(li => li.getAttribute('data-step-disabled') === 'expertids')[0];
nav.removeChild(expert);
""" + _DETECTION_STATE + "makeTaskControl('segment'); enterApp();" + """
({ rail: railInfo() });
""")
    sections = [row for row in out["rail"] if row["section"]]
    check(len(sections) == 2, f"the fixture has two headings: {sections}")
    review = [s for s in sections if s["title"] == "Review"][0]
    tools = [s for s in sections if s["title"] == "Tools"][0]
    check(review["display"] == "none", f"the Review heading has nothing under it: {review}")
    check(tools["display"] != "none", f"the Tools heading keeps its tools: {tools}")


def test_rail_plan_is_a_pure_function_of_the_state():
    """railPlan() reads the state it is handed, so the rules can be checked
    without a page and cannot drift from projectFamily()."""
    out = _drive("""
const bbox = { is_training_project: true, training_source: { kind: 'bbox', task_default: 'detect' } };
const detectByTask = { is_training_project: true, training_source: { kind: 'staged', task_default: 'detection' } };
const set = { is_training_project: true, composite: { name: 'x' } };
const plain = { is_training_project: false };
({ bbox: railPlan(bbox), detectByTask: railPlan(detectByTask), set: railPlan(set),
   plain: railPlan(plain), nullState: railPlan(null),
   families: [projectFamily(bbox), projectFamily(set), projectFamily(plain), projectFamily(null)] });
""")
    check(out["bbox"]["hidden"] == ["1", "2", "3", "step4test", "step4loop", "editmasks"],
          f"a box dataset hides the annotation stages: {out['bbox']}")
    check(out["bbox"]["skipped"] == [], f"a hidden stage is not also skipped: {out['bbox']}")
    check(out["detectByTask"]["hidden"] == out["bbox"]["hidden"],
          f"a source asking for detection is the detection family too: {out['detectByTask']}")
    check(out["set"]["skipped"] == out["bbox"]["hidden"] and out["set"]["hidden"] == [],
          f"a training set keeps its stages, marked skipped: {out['set']}")
    check(out["plain"] == {"hidden": [], "skipped": []}, f"an ordinary project: {out['plain']}")
    check(out["nullState"] == {"hidden": [], "skipped": []}, f"no project: {out['nullState']}")
    check(out["families"] == ["detection", "segmentation", "segmentation", "segmentation"],
          f"families: {out['families']}")


# ── ACTIVE ──────────────────────────────────────────────────────────
def test_only_one_tile_is_ever_the_current_stage():
    """The skipped Fetch Points tile used to keep the magenta bar and the pink
    circle of the current stage while Train carried them too."""
    out = _drive(_DETECTION_STATE + _OPEN + """
const first = railInfo().filter(r => r.active).map(r => r.step || r.disabled);
switchStep(7);
const second = railInfo().filter(r => r.active).map(r => r.step || r.disabled);
switchStep(8);
const third = railInfo().filter(r => r.active).map(r => r.step || r.disabled);
({ first: first, second: second, third: third, panels: activePanels() });
""")
    check(out["first"] == ["6"], f"one active tile on open, and it is Train: {out['first']}")
    check(out["second"] == ["7"], f"one active tile after moving to Evaluate: {out['second']}")
    check(out["third"] == ["8"], f"one active tile after moving to Inference: {out['third']}")
    check(out["panels"] == ["panel-8"], f"one active panel: {out['panels']}")


def test_a_tile_taken_out_of_the_flow_loses_the_active_highlight():
    """updateSidebar() must find a tile by either spelling, so a tile that was
    active before the rail disabled it is reset on the next pass."""
    out = _drive("""
makeRail(); makeTaskControl('segment');
""" + _ANNOTATION_STATE + """
activeStep = 1;
updateSidebar();
const before = railInfo().filter(r => r.active).map(r => r.step || r.disabled);
""" + _DETECTION_STATE + """
applyCompositeTrainingMode();
activeStep = 6;
updateSidebar();
const after = railInfo().filter(r => r.active).map(r => r.step || r.disabled);
({ before: before, after: after });
""")
    check(out["before"] == ["1"], f"Fetch Points starts as the current stage: {out['before']}")
    check(out["after"] == ["6"],
          f"only Train is current once the rail disables Fetch Points: {out['after']}")


# ── RESUME ──────────────────────────────────────────────────────────
def test_a_project_opens_on_the_step_it_recorded():
    """A training project used to be pinned to Train on every open, so a
    student who left on Evaluate or Inference was thrown back every time."""
    out = _drive("""
const train = { is_training_project: true, training_source: { kind: 'bbox' } };
const plain = { is_training_project: false };
({
  trainEvaluate: openingStep(Object.assign({ current_step: 7 }, train)),
  trainInference: openingStep(Object.assign({ current_step: 8 }, train)),
  trainFloored: openingStep(Object.assign({ current_step: 1 }, train)),
  trainArchived: openingStep(Object.assign({ current_step: 5 }, train)),
  plainArchivedFour: openingStep(Object.assign({ current_step: 4 }, plain)),
  plainArchivedFive: openingStep(Object.assign({ current_step: 5 }, plain)),
  plainThree: openingStep(Object.assign({ current_step: 3 }, plain)),
  noProject: openingStep(null),
});
""")
    check(out["trainEvaluate"] == 7, f"Evaluate is kept: {out['trainEvaluate']}")
    check(out["trainInference"] == 8, f"Inference is kept: {out['trainInference']}")
    check(out["trainFloored"] == 6, f"a stage below Train opens Train: {out['trainFloored']}")
    check(out["trainArchived"] == 6, f"an archived slot opens Train on a training project: {out['trainArchived']}")
    check(out["plainArchivedFour"] == "step4test", f"slot 4 opens the annotator tile: {out['plainArchivedFour']}")
    check(out["plainArchivedFive"] == "step4test", f"slot 5 opens the annotator tile: {out['plainArchivedFive']}")
    check(out["plainThree"] == 3, f"an ordinary project opens where it was: {out['plainThree']}")
    check(out["noProject"] == 1, f"no project opens the first stage: {out['noProject']}")


def test_a_training_project_reopens_on_inference():
    """The whole path, not just the helper: a box project left on Inference
    comes back to Inference."""
    out = _drive(_DETECTION_STATE + """
state.current_step = 8;
state.steps['7'].status = 'completed';
""" + _OPEN + """
({ panels: activePanels(), active: railInfo().filter(r => r.active).map(r => r.step) });
""")
    check(out["panels"] == ["panel-8"], f"the window opens on Inference: {out['panels']}")
    check(out["active"] == ["8"], f"Inference is the current stage: {out['active']}")


# ── PANELS ──────────────────────────────────────────────────────────
def test_every_reachable_tile_opens_its_panel():
    out = _drive(_ANNOTATION_STATE + _OPEN + """
const seen = {};
getNavOrder().forEach(ds => { switchStep(navStepArg(ds)); seen[ds] = activePanels(); });
seen;
""")
    for key, panels in out.items():
        check(panels == [f"panel-{key}"], f"tile {key} must open panel-{key}, got {panels}")


def test_the_archived_slots_open_the_annotator_instead_of_blanking_the_window():
    """switchStep(4) and switchStep(5) used to clear every panel and find none,
    which is why the step4 and step5 screenshots were an empty window."""
    out = _drive(_ANNOTATION_STATE + _OPEN + """
switchStep(4);
const four = activePanels();
switchStep(3);
switchStep(5);
const five = activePanels();
switchStep(3);
switchStep(99);
const unknown = activePanels();
({ four: four, five: five, unknown: unknown, activeStep: activeStep });
""")
    check(out["four"] == ["panel-step4test"], f"step 4 opens the annotator's panel: {out['four']}")
    check(out["five"] == ["panel-step4test"], f"step 5 opens the annotator's panel: {out['five']}")
    check(out["unknown"] == ["panel-3"],
          f"a step that reaches no panel keeps the open one: {out['unknown']}")
    check(out["activeStep"] == 3, f"the current stage is unchanged too: {out['activeStep']}")


def test_a_stage_this_project_does_not_have_opens_one_it_does():
    """A detection project asked for an annotation stage opens its first real
    stage rather than a panel its rail cannot get back to."""
    out = _drive(_DETECTION_STATE + _OPEN + """
switchStep(3);
({ panels: activePanels(), active: railInfo().filter(r => r.active).map(r => r.step) });
""")
    check(out["panels"] == ["panel-6"], f"Choose Images is not offered, Train opens: {out['panels']}")
    check(out["active"] == ["6"], f"and Train is the only current stage: {out['active']}")


# ── TASK ────────────────────────────────────────────────────────────
def test_the_training_task_comes_back_after_a_box_dataset_unpins_it():
    """Browsing the training-source picker into a staged box dataset and back
    out used to leave a segmentation project training a detector."""
    out = _drive("""
makeRail();
const task = makeTaskControl('segment');
const source = makeSourceSelect([{ value: '/x/export', kind: 'composite' },
                                 { value: '/x/bbox_datasets/coral_intel', kind: 'bbox' }]);
""" + _ANNOTATION_STATE + """
selectSource(source, 1);
applyStep6BoxDatasetMode();
const pinned = { value: task.value, disabled: task.disabled, note: document.getElementById('s6-task-note').textContent };
selectSource(source, 0);
applyStep6BoxDatasetMode();
const freed = { value: task.value, disabled: task.disabled, note: document.getElementById('s6-task-note').textContent };
({ pinned: pinned, freed: freed });
""")
    check(out["pinned"]["value"] == "detect" and out["pinned"]["disabled"] is True,
          f"a box dataset pins the task to Detector: {out['pinned']}")
    check("only a detector can train on it" in out["pinned"]["note"],
          f"the line under the select says why: {out['pinned']}")
    check(out["freed"]["value"] == "segment",
          f"deselecting the box dataset gives the task back: {out['freed']}")
    check(out["freed"]["disabled"] is False, f"and hands the control back: {out['freed']}")
    check("Changing this" in out["freed"]["note"], f"and puts the free line back: {out['freed']}")


def test_pinning_twice_keeps_the_first_free_value():
    """The stash is written once, so a second pin cannot record Detector as
    the value to give back."""
    out = _drive("""
makeRail();
const task = makeTaskControl('segment');
pinStep6Task(true);
pinStep6Task(true);
pinStep6Task(false);
({ value: task.value, disabled: task.disabled });
""")
    check(out["value"] == "segment", f"the value the user had comes back: {out}")
    check(out["disabled"] is False, f"the control is free again: {out}")


# ── NAMES ───────────────────────────────────────────────────────────
def test_the_class_picker_shows_the_full_name_with_the_code_as_a_chip():
    out = _drive("""
const names = { PA: 'Porites astreoides', OFRA: 'Orbicella franksi',
                EN: 'Epinephelus striatus, Nassau grouper, dark phase',
                SS: 'SS', UG: 'grouper, species not identified', EVIL: '<b>x</b> y' };
({
  pa: step6ClassLabelHtml('PA', names),
  ofra: step6ClassLabelHtml('OFRA', names),
  en: step6ClassLabelHtml('EN', names),
  unresolved: step6ClassLabelHtml('ZZZ', names),
  selfNamed: step6ClassLabelHtml('SS', names),
  notASpecies: step6ClassLabelHtml('UG', names),
  hostile: step6ClassLabelHtml('EVIL', names),
  noRegistry: step6ClassLabelHtml('PA', null),
  text: step6ClassLabelText('PA', names),
  textUnresolved: step6ClassLabelText('ZZZ', names),
});
""")
    check(out["pa"] == '<em class="vega-report-binomial">Porites astreoides</em>'
          ' <span class="vega-report-chip">PA</span>', f"binomial italic, code as a chip: {out['pa']}")
    check("Orbicella franksi" in out["ofra"] and '<span class="vega-report-chip">OFRA</span>' in out["ofra"],
          f"every code resolves the same way: {out['ofra']}")
    check('<em class="vega-report-binomial">Epinephelus striatus</em>' in out["en"]
          and "Nassau grouper" in out["en"] and "<em" in out["en"],
          f"only the binomial is italic when qualifiers follow: {out['en']}")
    check("<em" not in out["en"].split("</em>")[1], f"the qualifiers are not italic: {out['en']}")
    check(out["unresolved"] == '<span class="vega-report-plain">ZZZ</span>',
          f"a code that resolves to nothing stays bare: {out['unresolved']}")
    check("chip" not in out["unresolved"], f"and carries no chip: {out['unresolved']}")
    check(out["selfNamed"] == '<span class="vega-report-plain">SS</span>',
          f"a registry row that repeats the code adds nothing: {out['selfNamed']}")
    check("<em" not in out["notASpecies"] and "grouper, species not identified" in out["notASpecies"],
          f"a name that is not a species is upright: {out['notASpecies']}")
    check("&lt;b&gt;" in out["hostile"] and "<b>" not in out["hostile"],
          f"a hostile registry row is escaped: {out['hostile']}")
    check(out["noRegistry"] == '<span class="vega-report-plain">PA</span>',
          f"with no registry every code stays bare: {out['noRegistry']}")
    check(out["text"] == "Porites astreoides (PA)", f"the plain-text form: {out['text']}")
    check(out["textUnresolved"] == "ZZZ", f"the plain-text form of an unresolved code: {out['textUnresolved']}")


def test_the_rendered_picker_rows_carry_the_name_and_a_useful_tooltip():
    out = _drive("""
makeRail();
const host = document.getElementById('s6-classes-list');
renderStep6Classes([
  { id: 0, name: 'OFRA', instance_count: 58, image_count: 12 },
  { id: 1, name: 'ZZZ', instance_count: 3, image_count: 1 },
], null, { OFRA: 'Orbicella franksi' });
({ html: host.innerHTML });
""")
    html = out["html"]
    check("Orbicella franksi" in html, f"the row names the species: {html}")
    check('<span class="vega-report-chip">OFRA</span>' in html, f"the code rides as a chip: {html}")
    check(">ZZZ</span>" in html, f"an unresolved code still shows: {html}")
    check("Tick this to train on Orbicella franksi (OFRA), id 0" in html,
          f"the tooltip names the species and its id: {html}")
    check("VEGA strips every class left unticked" in html, f"the tooltip says what happens next: {html}")
    check("#0" in html and "58 instances" in html, f"the id and the counts survive: {html}")


def test_the_picker_reads_the_label_registry_it_already_ships():
    """The names come from the module's own vocabulary route, not a map typed
    into the browser."""
    js = open(_JS, encoding="utf-8").read()
    i = js.index("async function refreshStep6Classes(")
    fn = js[i:js.index("\n}\n", i)]
    check("vegaLabelNames()" in fn,
          "refreshStep6Classes must read the label registry the page already fetches")
    check("renderStep6Classes(data.classes || [], data.include_classes, names)" in fn,
          "the resolved names must reach the picker")
    check("step6ClassLabelText(code, names)" in fn,
          "the dropped-class note must name species too")
    check("/labels/api/vocabulary" in js, "the vocabulary route is the name source")
    # A code-to-binomial table typed into the browser would go stale the moment
    # the Label Manager edits the vocabulary, so there must not be one. A
    # binomial inside a docstring example is not a lookup table.
    hardcoded = re.findall(_HARDCODED_NAME_RE, _strip_comments(js))
    check(not hardcoded, f"a species name table is hardcoded in the browser: {hardcoded[:3]}")


# ── FINETUNE ────────────────────────────────────────────────────────
def test_fine_tune_is_refused_across_families():
    out = _drive("""
({
  segIntoDetector: finetuneRefusal('seg', 'detect', true, FINETUNE_CROSS_FAMILY_TEXT),
  detectIntoSegmenter: finetuneRefusal('detect', 'segment', true, FINETUNE_CROSS_FAMILY_TEXT),
  matching: finetuneRefusal('detect', 'detect', true, FINETUNE_CROSS_FAMILY_TEXT),
  segMatching: finetuneRefusal('seg', 'segment', true, FINETUNE_CROSS_FAMILY_TEXT),
  notPinned: finetuneRefusal('seg', 'detect', false, FINETUNE_CROSS_FAMILY_TEXT),
  unknown: finetuneRefusal('', 'detect', true, FINETUNE_CROSS_FAMILY_TEXT),
  tip: finetuneRefusal('seg', 'detect', true, FINETUNE_CROSS_FAMILY_TIP),
});
""")
    check("segmenter" in out["segIntoDetector"] and "detector" in out["segIntoDetector"],
          f"the refusal names both tasks in plain words: {out['segIntoDetector']}")
    check("seg" not in out["segIntoDetector"].replace("segmenter", ""),
          f"the refusal never prints the raw task token: {out['segIntoDetector']}")
    check(out["detectIntoSegmenter"], "the refusal works in both directions")
    check(out["matching"] == "" and out["segMatching"] == "",
          f"a matching task is never refused: {out}")
    check(out["notPinned"] == "", f"a free task control follows the run instead: {out['notPinned']}")
    check(out["unknown"] == "", f"an unknown task is not evidence of a mismatch: {out['unknown']}")
    check("VEGA refuses" in out["tip"], f"the tooltip says who refuses and why: {out['tip']}")


def test_the_cross_family_hand_off_changes_nothing_on_the_panel():
    """A run's weights go into "Continue from a VEGA model" (#s6-continue-from,
    2026-09-21), not the starting-model select (#s6-model) any more; a refused
    hand-off must leave both alone, and an allowed one must leave the starting
    model untouched while setting the continue-from pick."""
    out = _drive("""
makeRail();
const task = makeTaskControl('detect');
task.disabled = true;
const model = document.getElementById('s6-model');
model.value = 'yolo11m.pt';
const cont = makeContinueFrom();
""" + _DETECTION_STATE + """
onVegaFinetuneFrom({ run_dir: '/runs/18aprilpm_PA_20260422_135837', task: 'seg' });
const refused = {
  model: model.value, continueFrom: cont.sel.value,
  result: document.getElementById('s6-result').innerHTML, task: task.value,
  detailsOpen: cont.details.open, modelDisabled: model.disabled,
};
onVegaFinetuneFrom({ run_dir: '/runs/coral_intel_det_260902', task: 'detect' });
({
  refused: refused, allowedModel: model.value, allowedContinueFrom: cont.sel.value,
  allowedDetailsOpen: cont.details.open, allowedModelDisabled: model.disabled,
});
""")
    check(out["refused"]["model"] == "yolo11m.pt",
          f"a refused hand-off must not touch the starting model: {out['refused']}")
    check(out["refused"]["continueFrom"] == "",
          f"a refused hand-off must not touch the continue-from pick: {out['refused']}")
    check(out["refused"]["task"] == "detect", f"nor the task: {out['refused']}")
    check(out["refused"]["detailsOpen"] is False,
          f"a refused hand-off must not open the continue-from disclosure: {out['refused']}")
    check(out["refused"]["modelDisabled"] is False,
          f"a refused hand-off must not disable the starting model: {out['refused']}")
    check("cannot start this run" in out["refused"]["result"],
          f"the panel says why in one line: {out['refused']['result']}")
    check(out["allowedModel"] == "yolo11m.pt",
          f"an allowed hand-off leaves the starting model's own value unchanged: {out['allowedModel']}")
    check(out["allowedContinueFrom"].endswith("/weights/best.pt"),
          f"an allowed hand-off hands the run's weights to the continue-from pick: {out['allowedContinueFrom']}")
    check(out["allowedDetailsOpen"] is True,
          f"an allowed hand-off must open the continue-from disclosure: {out}")
    check(out["allowedModelDisabled"] is True,
          f"choosing a run under continue-from must disable the starting model: {out}")


def test_the_fine_tune_button_is_disabled_with_its_reason():
    out = _drive("""
makeRail();
const task = makeTaskControl('detect');
task.disabled = true;
const card = makeRunsCard([
  { dir: '/runs/seg_one', task: 'seg' },
  { dir: '/runs/det_one', task: 'detect' },
  { dir: '/runs/no_weights', task: 'seg', disabled: true },
]);
""" + _DETECTION_STATE + """
applyStep6FinetuneFamily();
const read = () => document.getElementById('s6-runs-card')
  .querySelectorAll('[data-action="finetune"]')
  .map(b => ({ dir: b.getAttribute('data-run-dir'), disabled: b.disabled, title: b.title,
               wrapper: b.parentNode && b.parentNode.className,
               wrapperTitle: b.parentNode ? b.parentNode.title : '' }));
const pinned = read();
task.disabled = false;
task.value = 'segment';
applyStep6FinetuneFamily();
({ pinned: pinned, freed: read() });
""")
    seg = [b for b in out["pinned"] if b["dir"] == "/runs/seg_one"][0]
    det = [b for b in out["pinned"] if b["dir"] == "/runs/det_one"][0]
    none = [b for b in out["pinned"] if b["dir"] == "/runs/no_weights"][0]
    check(seg["disabled"] is True, f"a segmentation run cannot fine-tune a detector: {seg}")
    check(seg["wrapper"] == "vega-runs-action" and "VEGA refuses" in seg["wrapperTitle"],
          f"the reason rides on the focusable wrapper the card uses: {seg}")
    check(seg["title"] == "", f"a disabled button carries no title of its own: {seg}")
    check(det["disabled"] is False, f"a detector run is untouched: {det}")
    check(none["disabled"] is True and none["wrapperTitle"] == "",
          f"a button the card disabled for its own reason is left alone: {none}")
    freed_seg = [b for b in out["freed"] if b["dir"] == "/runs/seg_one"][0]
    check(freed_seg["disabled"] is False,
          f"the action comes back when the task is free again: {freed_seg}")
    check("weights" in freed_seg["title"],
          f"and so does its own tooltip: {freed_seg}")


# ── copy rules ──────────────────────────────────────────────────────
def test_the_new_copy_obeys_the_house_rules():
    """No em dash, no en dash, no emoji, no banned word, and no internal step
    number or directory token in anything this task added."""
    js = open(_JS, encoding="utf-8").read()
    for bad, label in (("—", "em dash"), ("–", "en dash")):
        check(bad not in js, f"{label} in orchestrator.js")
    check(not re.search(r"\bland(s|ed|ing)?\b", js.lower()), "the banned word is in orchestrator.js")
    # An internal directory token may be named in a comment about the server's
    # own layout; what matters is that none reaches a string the page renders.
    code = _strip_comments(js)
    for token in ("{step6_dir}", "{step7_dir}", "{step8_dir}", "step6_trainModel"):
        check(token not in code, f"{token} must never reach the screen")
    # Every sentence this task added, checked for the uppercase rule.
    for name in ("FINETUNE_CROSS_FAMILY_TEXT", "FINETUNE_CROSS_FAMILY_TIP", "TILE_SKIPPED_STATUS"):
        check(name in js, f"{name} is missing")
    i = js.index("const PIPELINE_LINES")
    block = js[i:js.index("\n};\n", i)]
    check("Box dataset -> YOLO11 detector" in block and "BOX DATASET" not in block,
          "the route line is sentence case, not uppercase")
    check("VEGA" in block, "the route line names who does the work")


def test_the_id_contract_is_kept():
    """No element id the JS reads may be renamed, and every id this task added
    a reference to must exist in templates/index.html."""
    # The window is index.html plus the fragments it includes, so the contract
    # is read across all of them.
    tpl_dir = os.path.join(_ORCH, "templates")
    tpl = "".join(open(os.path.join(tpl_dir, f), encoding="utf-8").read()
                  for f in sorted(os.listdir(tpl_dir)) if f.endswith(".html"))
    for element_id in ("step-nav", "topbar-pipeline", "s6-task", "s6-task-note", "s6-source",
                       "s6-classes-list", "s6-runs-card", "s6-result", "s6-model",
                       "panel-6", "panel-7", "panel-8",
                       "step1-status-text", "step6-status-text", "step8-status-text"):
        check(f'id="{element_id}"' in tpl, f"the window's templates carry no id {element_id}")
    js = open(_JS, encoding="utf-8").read()
    check("step${s}-status-text" in js, "the sidebar still writes the per-step caption ids")
    runs_js = open(os.path.join(_ORCH, "static", "orchestrator_runs.js"), encoding="utf-8").read()
    for stamp in ('data-action="', 'data-task="', 'finetune'):
        check(stamp in runs_js,
              f"the runs card no longer stamps {stamp!r}, which the family pass reads")


# ── STAGES ──────────────────────────────────────────────────────────
# What the three project shapes number their stages, read off the rail each one
# shows: a detection project has three stages, every other project has seven.
_HEADINGS_SHIPPED = (
    "makeHeading(6, 'Train Model', 'inline', '5');"
    "makeHeading(7, 'Evaluate Model', 'inline', '6');"
    "makeHeading(8, 'Model Inference', 'inline', '7');"
)
# What a scenario reads back about one stage.
_STAGE_READER = """
function stageRead(key) {
  const li = document.querySelector('.step-nav li[data-step="' + key + '"]');
  const circle = li ? li.querySelector('.step-circle') : null;
  const panel = document.getElementById('panel-' + key);
  const span = panel ? panel.querySelector('.stage-num') : null;
  return {
    num: stepDisplayNum(key),
    circle: circle ? circle.dataset.num : null,
    node: circle ? circle.textContent : null,
    span: span ? span.textContent : null,
  };
}
"""


def test_a_detection_project_numbers_its_three_stages():
    """A box-dataset project's rail holds Train, Evaluate and Model Inference,
    so its Inference panel is stage 3 of 3 and never stage 7."""
    out = _drive(_DETECTION_STATE + _STAGE_READER
                 + "makeRail(); makeTaskControl('segment');" + _HEADINGS_SHIPPED
                 + "enterApp();"
                 + "({ six: stageRead(6), seven: stageRead(7), eight: stageRead(8) });")
    check(out["six"]["num"] == "1", f"Train is stage 1 on a detection rail (got {out['six']['num']})")
    check(out["seven"]["num"] == "2", f"Evaluate is stage 2 (got {out['seven']['num']})")
    check(out["eight"]["num"] == "3", f"Inference is stage 3 (got {out['eight']['num']})")
    check(out["eight"]["span"] == "3",
          f"the Inference heading carries the stage number (got {out['eight']['span']!r})")
    check(out["six"]["span"] == "1",
          f"the Train heading was renumbered too (got {out['six']['span']!r})")
    check(out["eight"]["circle"] == "3",
          f"the rail node carries the same number (got {out['eight']['circle']!r})")
    check(out["eight"]["node"] == "3",
          f"the rail node draws the same number (got {out['eight']['node']!r})")


def test_a_training_set_project_keeps_the_seven_stage_numbering():
    """A project minted from a training set keeps every stage on its rail, so
    its numbering is the one the window ships."""
    out = _drive(_TRAINING_SET_STATE + _STAGE_READER
                 + "makeRail(); makeTaskControl('segment');" + _HEADINGS_SHIPPED
                 + "enterApp();"
                 + "({ six: stageRead(6), eight: stageRead(8) });")
    check(out["six"]["num"] == "5", f"Train is stage 5 (got {out['six']['num']})")
    check(out["eight"]["num"] == "7", f"Inference is stage 7 (got {out['eight']['num']})")
    check(out["eight"]["span"] == "7",
          f"the heading carries the seventh stage's number (got {out['eight']['span']!r})")


def test_an_ordinary_project_numbers_its_stages_as_the_window_ships_them():
    """A project that annotates its own images has every stage, and the numbers
    are the archived-chain ones: the combined annotator is 4 and Train is 5."""
    out = _drive(_ANNOTATION_STATE + _STAGE_READER
                 + "makeRail(); makeTaskControl('segment');" + _HEADINGS_SHIPPED
                 + "enterApp();"
                 + "({ three: stepDisplayNum(3), tool: stepDisplayNum('step4test'),"
                 + "   six: stepDisplayNum(6), seven: stepDisplayNum(7), eight: stepDisplayNum(8) });")
    check([out["three"], out["tool"], out["six"], out["seven"], out["eight"]]
          == ["3", "4", "5", "6", "7"], f"the ordinary numbering moved: {out}")


def test_every_panel_keeps_its_own_number_when_another_stage_opens():
    """Each heading number answers to its own panel's id. Opening another stage
    used to move one shared id onto the open panel's span, so the next stage
    opened overwrote that panel's number with its own: a detection project left
    the Model Inference heading reading "Step 1: Model Inference"."""
    out = _drive(_DETECTION_STATE + "state.current_step = 8;"
                 + "makeRail(); makeTaskControl('segment');" + _HEADINGS_SHIPPED
                 + "enterApp();"
                 + "switchStep(6); switchStep(7); switchStep(8); switchStep(6);"
                 + "const read = key => {"
                 + "  const span = document.getElementById('panel-' + key).querySelector('.stage-num');"
                 + "  return { num: span.textContent, id: span.id };"
                 + "};"
                 + "({ six: read(6), seven: read(7), eight: read(8) });")
    check([out["six"]["num"], out["seven"]["num"], out["eight"]["num"]] == ["1", "2", "3"],
          f"a panel was renumbered by another stage opening: {out}")
    check(out["six"]["id"] == "step6-stage-num", f"Train's number has its own id: {out['six']}")
    check(out["eight"]["id"] == "step8-stage-num", f"Inference's number has its own id: {out['eight']}")


def test_a_segmentation_project_keeps_its_headings_through_the_rail():
    """The same walk over the seven-stage rail, which is the family that had
    the most panels to corrupt: Fetch Points stayed "Step 5: Fetch Points"
    after Train Model was opened."""
    out = _drive(_TRAINING_SET_STATE + "state.current_step = 1;"
                 + "makeRail(); makeTaskControl('segment');"
                 + "makeHeading(1, 'Fetch Points', 'inline', '1');" + _HEADINGS_SHIPPED
                 + "enterApp();"
                 + "switchStep(1); switchStep(6); switchStep(8);"
                 + "const read = key =>"
                 + "  document.getElementById('panel-' + key).querySelector('.stage-num').textContent;"
                 + "({ one: read(1), six: read(6), eight: read(8) });")
    check([out["one"], out["six"], out["eight"]] == ["1", "5", "7"],
          f"a segmentation panel was renumbered by another stage opening: {out}")


def test_a_heading_that_spells_its_own_number_is_rewritten():
    """A window that ships no stage span has the number in its heading words.
    That number is this project's, not the internal slot: a detection project
    reads Step 3, never Step 7."""
    out = _drive(_DETECTION_STATE + "state.current_step = 8;"
                 + "makeRail(); makeTaskControl('segment');"
                 + "makeHeading(8, 'Step 7: Model Inference', 'plain');"
                 + "enterApp();"
                 + "const h = document.getElementById('panel-8').querySelector('h2');"
                 + "const slot = h.querySelector('.stage-num-inline');"
                 + "({ words: h.children.map(c => c.textContent),"
                 + "   number: slot ? slot.textContent : null,"
                 + "   published: slot ? slot.id : null,"
                 + "   classes: slot ? slot.className : null });")
    check(out["number"] == "3", f"the heading was renumbered (got {out['number']!r})")
    check(out["words"] == ["Step ", "3", ": Model Inference"],
          f"the heading keeps its words around the number (got {out['words']})")
    check(out["published"] == "step8-stage-num",
          f"the wrapped number answers to its own panel's id (got {out['published']!r})")
    check("stage-num" in (out["classes"] or ""),
          f"the wrapped number carries the stage-num class (got {out['classes']!r})")


def test_no_folder_token_or_internal_slot_reaches_a_sentence():
    """Nothing the window renders may carry a template placeholder or an
    internal chain slot: a student reads the stage number the rail shows."""
    js = open(_JS, encoding="utf-8").read()
    code = _strip_comments(js)
    leftovers = re.findall(r"\{step[A-Za-z0-9_]*_dir\}", code)
    check(not leftovers, f"a folder placeholder is still rendered: {leftovers}")
    check("`Step ${step}`" not in code,
          "a sentence still prints the internal chain slot instead of the stage number")
    check("Step ${step} has no output folder" not in code,
          "the open-folder refusal still prints the internal chain slot")
    check("stepDisplayNum(step)" in code,
          "the sentences that name a step read the display number")


# ── SEEDS ───────────────────────────────────────────────────────────
def _seed_summary(summary):
    """Render one seeder summary and hand back what the panel shows."""
    return _drive(f"renderStep4loopSeedSummary({summary});"
                  "({ html: document.getElementById('s4loop-seed-result').innerHTML });")


def _seed_line(html):
    """The words the summary puts on screen, with the glyph and its tooltip
    taken out, so a test can tell what a reader sees from what a reader
    hovers."""
    without_glyph = re.sub(r"<button.*?</button>", "", html, flags=re.S)
    return re.sub(r"<[^>]+>", "", without_glyph).strip()


def _seed_tip(html):
    """The tooltip the summary's glyph carries."""
    found = re.search(r'title="([^"]*)"', html)
    return found.group(1) if found else ""


def test_a_round_that_seeded_no_mask_says_why():
    """A Refine round seeded from a detector's predictions used to report a
    green "0 masks" with nothing wrong on screen. The drops are counted on the
    line and named in its tooltip, and the line reads as a warning."""
    out = _seed_summary("{seeded_frames: 12, seeded_masks: 0, skipped_holdout: 4,"
                        " skipped_box_only: 31, skipped_no_geometry: 2}")
    html = out["html"]
    check("badge-warning" in html, f"a round with no mask is not a success: {html}")
    check("badge-success" not in html, f"a round with no mask is not a success: {html}")
    check(_seed_line(html)
          == "Seeded 12 frames, 0 masks for review. Left out: 4 frames and 33 detections.",
          f"the line states both counts and what was left out: {_seed_line(html)!r}")
    tip = _seed_tip(html)
    check("31 detections that carried a box and no outline" in tip,
          f"the box-only drops are named in the tooltip: {tip!r}")
    check("2 detections that carried neither an outline nor a box" in tip,
          f"the shapeless drops are named in the tooltip: {tip!r}")
    check("4 frames on held-out transects" in tip,
          f"the held-out frames still read as expected: {tip!r}")
    check("segmentation model" in tip, f"the tooltip says what to do about it: {tip!r}")
    check("Lower the confidence" not in tip,
          f"the advice belongs to the largest drop, not the first one listed: {tip!r}")


def test_the_drop_reasons_sit_on_a_focusable_glyph_at_the_reading_size():
    """The reasons are a tooltip on a glyph reachable from the keyboard, not a
    paragraph of 12 px mono. The line itself is 13 px, the floor for text that
    explains anything (workflow page design, section 3)."""
    out = _seed_summary("{seeded_frames: 12, seeded_masks: 0, skipped_box_only: 31}")
    html = out["html"]
    check('class="tip"' in html and "<button" in html,
          f"the reasons hang off a button, which takes focus: {html}")
    check("font-size:13px" in html, f"the line is set at the reading size: {html}")
    check(len(_seed_line(html)) < 90, f"the line is one line, not a paragraph: {_seed_line(html)!r}")
    check("aria-label=" in html, f"the glyph says what it is about: {html}")


def test_a_round_that_seeded_masks_still_reads_as_a_success():
    """Masks to review is the good outcome, and it stays green even when some
    detections were dropped along the way."""
    out = _seed_summary("{seeded_frames: 9, seeded_masks: 14, skipped_box_only: 3}")
    html = out["html"]
    check("badge-success" in html, f"a round that seeded masks is a success: {html}")
    check(_seed_line(html)
          == "Seeded 9 frames, 14 masks for review. Left out: 3 detections.",
          f"the counts are stated: {_seed_line(html)!r}")
    check("3 detections that carried a box and no outline" in _seed_tip(html),
          f"the drop is still named: {_seed_tip(html)!r}")


def test_a_round_that_seeded_one_of_each_reads_in_the_singular():
    """One is one, not "1 frame(s)"."""
    out = _seed_summary("{seeded_frames: 1, seeded_masks: 1, skipped_holdout: 1}")
    check(_seed_line(out["html"])
          == "Seeded 1 frame, 1 mask for review. Left out: 1 frame.",
          f"the singular reads as a sentence: {_seed_line(out['html'])!r}")


def test_a_round_with_nothing_to_seed_says_so():
    """An empty round is not a silent zero either."""
    out = _seed_summary("{seeded_frames: 0, seeded_masks: 0}")
    html = out["html"]
    check("badge-warning" in html, f"an empty round is not a success: {html}")
    check("holds no prediction to review" in html, f"the empty round says so: {html}")
    check('class="tip"' not in html, f"a round that dropped nothing carries no glyph: {html}")


# ── PROMOTE ─────────────────────────────────────────────────────────
# A server that refuses the promotion on the run's own gates, then accepts the
# same promotion once the operator has confirmed it. The label registry answers
# beside it, because the refusal names the classes that dropped and a code
# never stands alone where it resolves to a name.
_VOCABULARY = ('[{ code: "PA", name: "Porites astreoides" },'
               ' { code: "SS", name: "Siderastrea siderea" }]')


def _gate_refusing_server(vocabulary=_VOCABULARY):
    """A fetch stub that refuses one promotion on the gates.

    Parameters:
        vocabulary: the rows GET /labels/api/vocabulary answers with, or
            'null' for a registry that is unavailable, which is how a window
            with no Label Manager behind it reads.
    Returns:
        The scenario JavaScript that installs the stub.
    """
    return """
const calls = [];
let refused = false;
const VOCABULARY = %s;
fetch = (url, opts) => {
  const body = JSON.parse((opts && opts.body) || '{}');
  calls.push({ url: url, body: body });
  if (url.indexOf('/labels/api/vocabulary') >= 0) {
    return Promise.resolve({ status: 200, json: () => Promise.resolve(
      VOCABULARY ? { rows: VOCABULARY } : { error: 'no label registry here' }) });
  }
  const promote = url.indexOf('/promote') >= 0;
  if (promote && !refused && body.confirm_gate !== true) {
    refused = true;
    return Promise.resolve({ status: 409, json: () => Promise.resolve({
      error: 'the gates did not pass', needs_confirmation: true,
      gate: { gate_map: 'fail', gate_class: 'flag:PA,SS', gate_recall: '' },
    })});
  }
  return Promise.resolve({ status: 200, json: () => Promise.resolve({ ok: true }) });
};
const settle = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
""" % vocabulary


_GATE_REFUSING_SERVER = _gate_refusing_server()


def test_promote_asks_before_it_overrides_a_gate():
    """The server refuses a run whose ledger recorded a failure, a flag or an
    incomparable comparison. The refusal opens a confirmation that states all
    three verdicts rather than an error nobody can act on."""
    out = _drive(_DETECTION_STATE + _GATE_REFUSING_SERVER
                 + "(async () => {"
                 + "  await promoteRun('/x/runs/round_3');"
                 + "  await settle();"
                 + "  const msg = document.getElementById('confirm-msg');"
                 + "  const ok = document.getElementById('confirm-ok-btn');"
                 + "  return { title: document.getElementById('confirm-title').textContent,"
                 + "           msg: msg.innerHTML, ok: ok.textContent,"
                 + "           wired: typeof ok.onclick === 'function',"
                 + "           visible: document.getElementById('confirm-overlay').className,"
                 + "           calls: calls.filter(c => c.url.indexOf('/promote') >= 0).length };"
                 + "})();")
    check(out["title"] == "Promote over the gate?", f"the dialog says what it asks: {out['title']!r}")
    check("round_3" in out["msg"], f"the dialog names the run: {out['msg']}")
    check("The mean average precision gate failed." in out["msg"],
          f"the failing gate is stated as its own sentence: {out['msg']}")
    check("The per-class gate flagged" in out["msg"],
          f"the flagged gate is stated in plain words: {out['msg']}")
    check("flag:PA,SS" not in out["msg"],
          f"the ledger's own token never reaches the screen: {out['msg']}")
    check("The recall gate recorded no verdict." in out["msg"],
          f"a gate that recorded nothing says so: {out['msg']}")
    check("confirm_gate" not in out["msg"], "the dialog never shows the request field's name")
    check(out["ok"] == "Promote Anyway", f"the button says what it does: {out['ok']!r}")
    check(out["wired"], "the confirmation button is wired")
    check("visible" in out["visible"], f"the dialog is on screen: {out['visible']!r}")
    check(out["calls"] == 1, f"one refused promotion so far (got {out['calls']})")


def test_the_flagged_classes_are_named_in_full_with_their_codes():
    """A gate that flagged two classes names both species in full, in italic,
    with the code beside each as a chip. The raw cell reads "flag:PA,SS"."""
    out = _drive(_DETECTION_STATE + _GATE_REFUSING_SERVER
                 + "(async () => {"
                 + "  await promoteRun('/x/runs/round_3');"
                 + "  await settle();"
                 + "  return { msg: document.getElementById('confirm-msg').innerHTML };"
                 + "})();")
    msg = out["msg"]
    check("Porites astreoides" in msg and "Siderastrea siderea" in msg,
          f"both flagged classes are named in full: {msg}")
    check(msg.count("vega-report-binomial") == 2,
          f"each binomial is italic: {msg}")
    check(msg.count("vega-report-chip") == 2, f"each code rides beside its name as a chip: {msg}")
    check("Porites astreoides</em> <span class=\"vega-report-chip\">PA" in msg,
          f"each code sits beside its own name: {msg}")
    check("and" in msg, f"the two names read as a phrase rather than a list token: {msg}")


def test_a_code_the_registry_cannot_resolve_stays_as_it_is():
    """A window with no label registry behind it shows the codes the ledger
    wrote rather than inventing a name for them."""
    out = _drive(_DETECTION_STATE + _gate_refusing_server("null")
                 + "(async () => {"
                 + "  await promoteRun('/x/runs/round_3');"
                 + "  await settle();"
                 + "  return { msg: document.getElementById('confirm-msg').innerHTML };"
                 + "})();")
    msg = out["msg"]
    check("The per-class gate flagged" in msg, f"the gate is still stated: {msg}")
    check("PA" in msg and "SS" in msg, f"the codes are still shown: {msg}")
    check("flag:PA,SS" not in msg, f"the ledger's own token never reaches the screen: {msg}")
    check("vega-report-binomial" not in msg, f"nothing is italicised as a species: {msg}")


def test_confirming_the_gate_sends_the_confirmation():
    """Confirming re-sends the same promotion with the operator's confirmation
    on it, which is the only thing the server accepts it over."""
    out = _drive(_DETECTION_STATE + _GATE_REFUSING_SERVER
                 + "(async () => {"
                 + "  await promoteRun('/x/runs/round_3');"
                 + "  await settle();"
                 + "  document.getElementById('confirm-ok-btn').onclick();"
                 + "  await settle();"
                 + "  const promotes = calls.filter(c => c.url.indexOf('/promote') >= 0);"
                 + "  return { promotes: promotes.map(c => c.body),"
                 + "           open: document.getElementById('confirm-overlay').className };"
                 + "})();")
    check(len(out["promotes"]) == 2, f"the promotion is sent again (got {out['promotes']})")
    check(out["promotes"][0] == {"run_dir": "/x/runs/round_3"},
          f"the first request states no confirmation: {out['promotes'][0]}")
    check(out["promotes"][1] == {"run_dir": "/x/runs/round_3", "confirm_gate": True},
          f"the second carries the confirmation: {out['promotes'][1]}")
    check("visible" not in out["open"], f"the dialog closes on confirm: {out['open']!r}")


# ── HONESTY ─────────────────────────────────────────────────────────
# Three answers the server gives that the window used to read off the wire and
# throw away, so the reader was told the opposite of what happened: a Stop the
# orchestrator refused, a step whose process outlived a VEGA restart, and a
# result that belongs to another project. Each scenario drives the shipped
# poller or the shipped Stop against a stub answering exactly as app.py does
# (step_stop, _survive_restart, _finish_for_owner).
_HONEST_STATE = """
state = { is_training_project: true, project_type: 'bbox_training',
          training_source: { kind: 'bbox', task_default: 'detect' },
          current_step: 6, steps: makeSteps({ 6: 'running' }) };
"""

_SETTLE = "const settle = async () => { for (let i = 0; i < 60; i++) await Promise.resolve(); };\n"

# What a panel is left showing: the state line the fixes write, the result
# badge beside it, the tile's own mark, and whether Stop is still pressable.
_PANEL_READER = """
const panelRead = (step) => ({
  note: document.getElementById('s' + step + '-status-note').innerHTML,
  noteShown: document.getElementById('s' + step + '-status-note').style.display,
  result: document.getElementById('s' + step + '-result').innerHTML,
  mark: state.steps[String(step)].status,
  stop: document.getElementById('s' + step + '-stop-btn').style.display,
  runDisabled: document.getElementById('s' + step + '-run-btn').disabled,
  polling: pollTimers[step] !== undefined,
});
"""


def _stop_server(status, body):
    """A fetch stub answering one POST to a step's Stop route.

    Parameters:
        status: the HTTP status the server answers with.
        body: the JSON body, as JavaScript source.
    Returns:
        The scenario JavaScript that installs the stub.
    """
    return _SETTLE + _PANEL_READER + """
const calls = [];
fetch = (url, opts) => {
  calls.push(url);
  return Promise.resolve({ status: %d, json: () => Promise.resolve(%s) });
};
""" % (status, body)


def _poll_server(status_body):
    """A fetch stub answering one status poll, its log read and a state read.

    Parameters:
        status_body: the status payload, as JavaScript source.
    Returns:
        The scenario JavaScript that installs the stub and a tick() that fires
        one beat of the shipped poller (setInterval is the harness stub, which
        records the callback instead of running it).
    """
    return _SETTLE + _PANEL_READER + """
const calls = [];
fetch = (url, opts) => {
  calls.push(url);
  const answer = (body) => Promise.resolve({ status: 200, json: () => Promise.resolve(body) });
  if (url.indexOf('/status') >= 0) return answer(%s);
  if (url.indexOf('/log') >= 0) return answer({ lines: [], offset: 0, dropped: 0 });
  return answer({});
};
const tick = async (step) => {
  startPolling(step);
  await timers[timers.length - 1][0]();
  await settle();
};
""" % status_body


# app.py's step_stop refusal, word for word (HTTP_CONFLICT branch).
_STOP_REFUSAL = ("Train Model (step 5) is still running as process 4242, which outlived a VEGA "
                 "restart, so the orchestrator cannot stop it. Stop process 4242 yourself, then "
                 "press Stop again.")
# app.py's RESTARTED_STEP_NOTE, filled in.
_RESTART_NOTE = ("VEGA was restarted while Train Model (step 5) was running, so it no longer owns "
                 "that process and cannot say how it ended. Nothing was recorded. Check "
                 "/x/step6_trainModel, then press Stop to clear the mark or run the step again.")
# app.py's FOREIGN_STEP_NOTE, filled in.
_FOREIGN_NOTE = ("Train Model (step 5) was started by project 'coral_intel', not the one open "
                 "here, so VEGA recorded the result there and left this project exactly as it "
                 "was.")


def test_a_stop_the_server_refused_is_not_reported_as_a_stop():
    """The server answers 409 when the process outlived a VEGA restart: it has
    no handle on it and cannot kill it. The window cleared the tile anyway and
    told the reader the run had stopped while the trainer went on writing."""
    out = _drive(_HONEST_STATE + _stop_server(409, """
{ success: false, stopped: false, orphan_pid: 4242, error: %s }
""" % json.dumps(_STOP_REFUSAL))
                 + "pollTimers[6] = 99;"
                 + "(async () => { await stopStep(6); await settle();"
                 + "  return { panel: panelRead(6) }; })();")
    panel = out["panel"]
    check(_STOP_REFUSAL in panel["note"],
          f"the refusal reaches the screen word for word: {panel['note']}")
    check("4242" in panel["note"], f"the note names the process still running: {panel['note']}")
    check("var(--error)" in panel["note"],
          f"a refused action reads as an error: {panel['note']}")
    check(panel["mark"] == "running",
          f"the tile stays running, because nothing stopped: {panel['mark']!r}")
    check(panel["polling"], "the poller keeps running: the step did not end")
    check("Stopped" not in panel["result"] and panel["result"] == "",
          f"nothing claims the step stopped: {panel['result']!r}")


def test_a_stop_the_server_made_clears_the_tile():
    """The ordinary answer is unchanged: the runner held the process, killed
    it, and the tile goes back to not done with no state line left over."""
    out = _drive(_HONEST_STATE + _stop_server(200, "{ success: true, stopped: true, message: 'Stopped.' }")
                 + "pollTimers[6] = 99;"
                 + "showStepStatusNote(6, 'error', 'an older refusal');"
                 + "(async () => { await stopStep(6); await settle();"
                 + "  return { panel: panelRead(6) }; })();")
    panel = out["panel"]
    check(panel["mark"] == "pending", f"a stopped step is not done: {panel['mark']!r}")
    check(not panel["polling"], "the poller is cleared with the run")
    check(panel["note"] == "", f"the older state line is cleared: {panel['note']!r}")
    check(panel["noteShown"] == "none", f"the empty line is hidden: {panel['noteShown']!r}")


def test_a_new_launch_takes_the_old_state_line_off_the_screen():
    """A state line is about the run that produced it. Pressing Run again
    clears it before the new run starts, so a sentence about a restart never
    sits over a live run."""
    out = _drive(_HONEST_STATE + _stop_server(200, "{ success: true, warnings: [] }")
                 + "showStepStatusNote(6, 'warning', 'a sentence about the last run');"
                 + "(async () => { const before = panelRead(6).note;"
                 + "  await runStep(6); await settle();"
                 + "  return { before: before, after: panelRead(6).note }; })();")
    check("the last run" in out["before"], f"the fixture starts with a line: {out['before']!r}")
    check(out["after"] == "", f"the new launch clears it: {out['after']!r}")


def test_a_restart_the_server_cannot_account_for_says_so():
    """poll_status answers untracked with one plain sentence when the runner
    holds no process for a step this project still has marked running. The
    poll read neither field, so the spinner stopped, nothing was said, and the
    step sat marked running for the life of the window."""
    out = _drive(_HONEST_STATE + _poll_server("""
{ running: false, exit_code: null, untracked: true, warnings: [%s] }
""" % json.dumps(_RESTART_NOTE))
                 + "(async () => { await tick(6); return { panel: panelRead(6), calls: calls }; })();")
    panel = out["panel"]
    check(_RESTART_NOTE in panel["note"],
          f"the server's sentence reaches the screen: {panel['note']}")
    check(panel["noteShown"] == "block", f"the line is shown: {panel['noteShown']!r}")
    check(panel["mark"] == "running",
          f"the mark is left as it is, because nothing here knows how it ended: {panel['mark']!r}")
    check(panel["result"] == "",
          f"nothing is claimed about the outcome: {panel['result']!r}")
    check(panel["stop"] == "",
          f"Stop stays pressable, because the sentence tells the reader to press it: {panel['stop']!r}")
    check(panel["runDisabled"] is False,
          "running the step again stays open, which the sentence offers as the other way out")
    check(not any("/api/project/state" in c for c in out["calls"]),
          f"nothing reloads the project for work that was never recorded: {out['calls']}")


def test_an_untracked_step_still_speaks_when_the_server_sends_no_sentence():
    """A server older than this window answers untracked with no warnings. The
    panel still refuses to claim an outcome and says what to do."""
    out = _drive(_HONEST_STATE + _poll_server("{ running: false, exit_code: null, untracked: true }")
                 + "(async () => { await tick(6); return { panel: panelRead(6) }; })();")
    panel = out["panel"]
    check("no longer owns" in panel["note"], f"the panel says what is known: {panel['note']}")
    check("Press Stop" in panel["note"], f"the panel says what to do: {panel['note']}")
    check(panel["mark"] == "running", f"the mark is still left alone: {panel['mark']!r}")


def test_another_projects_success_is_not_claimed_in_this_window():
    """_finish_for_owner answers completed false and names the project the
    result went to, and the finished process still carries exit code 0. The
    poll read the exit code alone, so this project said "Step completed
    successfully", flipped its button to done and reloaded its cards for work
    it did not do."""
    out = _drive(_HONEST_STATE + _poll_server("""
{ running: false, exit_code: 0, completed: false, warnings: [%s] }
""" % json.dumps(_FOREIGN_NOTE))
                 + "(async () => { await tick(6); return { panel: panelRead(6), calls: calls,"
                 + "   btn: document.getElementById('s6-run-btn').className }; })();")
    panel = out["panel"]
    check(_FOREIGN_NOTE in panel["note"],
          f"the sentence naming the other project reaches the screen: {panel['note']}")
    check("coral_intel" in panel["note"], f"the note names that project: {panel['note']}")
    check("Step completed successfully" not in panel["result"],
          f"this window claims no success: {panel['result']!r}")
    check(panel["result"] == "", f"the result line says nothing at all: {panel['result']!r}")
    check(panel["mark"] == "running",
          f"the step is not marked done off another project's work: {panel['mark']!r}")
    check("btn-done" not in out["btn"],
          f"the action button is not flipped to done: {out['btn']!r}")
    check(not any("/api/project/state" in c for c in out["calls"]),
          f"the project's cards are not reloaded for another project's run: {out['calls']}")


def test_a_genuine_completion_still_reads_as_a_success():
    """The ordinary finish is untouched: a zero exit with no owner dispute
    completes the step and says so."""
    out = _drive(_HONEST_STATE + _poll_server("{ running: false, exit_code: 0 }")
                 + "(async () => { await tick(6); return { panel: panelRead(6) }; })();")
    panel = out["panel"]
    check("Step completed successfully" in panel["result"],
          f"a real success still reads as one: {panel['result']!r}")
    check(panel["note"] == "", f"no state line is invented: {panel['note']!r}")


def test_a_process_that_outlived_a_restart_says_so_while_it_runs():
    """_survive_restart answers running true with the orphan's pid when the
    kernel says the process is alive. The tile spins on, and the sentence
    saying VEGA cannot stop it was dropped on every beat."""
    orphan = ("Train Model (step 5) is still running as process 4242, which outlived a VEGA "
              "restart, so the orchestrator cannot stop it. Its output goes on being written "
              "to /x/step6_trainModel.")
    out = _drive(_HONEST_STATE + _poll_server("""
{ running: true, exit_code: null, orphan_pid: 4242, warnings: [%s] }
""" % json.dumps(orphan))
                 + "(async () => { await tick(6); return { panel: panelRead(6) }; })();")
    panel = out["panel"]
    check(orphan in panel["note"], f"the sentence is shown while the run goes on: {panel['note']}")
    check(panel["polling"], "a live run keeps polling")
    check(panel["mark"] == "running", f"the tile stays running: {panel['mark']!r}")


# ── GATES ───────────────────────────────────────────────────────────
# The Rounds ledger's Gates column printed the ledger's own token in a chip
# ("flag:GL,EF,FP,LC", "unpinned"), which is an internal code a student cannot
# act on.
_ROUNDS_VOCABULARY = ('[{ code: "PA", name: "Porites astreoides" },'
                      ' { code: "SS", name: "Siderastrea siderea" },'
                      ' { code: "MC", name: "Montastraea cavernosa" },'
                      ' { code: "DL", name: "Diploria labyrinthiformis" },'
                      ' { code: "OA", name: "Orbicella annularis" },'
                      ' { code: "FF", name: "Favia fragum" }]')


def _rounds_server(rows, vocabulary=_ROUNDS_VOCABULARY):
    """A fetch stub answering the rounds ledger and the label vocabulary."""
    return _SETTLE + """
fetch = (url, opts) => {
  const answer = (body) => Promise.resolve({ status: 200, json: () => Promise.resolve(body) });
  if (url.indexOf('/labels/api/vocabulary') >= 0) return answer({ rows: %s });
  if (url.indexOf('/api/step/7/rounds') >= 0) return answer({ rounds: %s, champion: null });
  return answer({});
};
const roundsHtml = async () => {
  await refreshStep7Rounds();
  await settle();
  return document.getElementById('s7-rounds-tbody').innerHTML;
};
""" % (vocabulary, rows)


_ONE_ROUND = """
[{ round: 2, run_name: 'round_2', base_model: 'yolo11m.pt', metric_family: 'box',
   map50_95_B: 0.535, recall_B: 0.655, conf: 0.001, iou: 0.7, imgsz: 512,
   gate_map: 'fail', gate_class: 'flag:PA,SS', gate_recall: '' }]
"""


def test_the_gates_column_states_plain_words_not_the_ledger_token():
    """Every gate reads as the gate it is and what it did. The cell carries no
    ledger token, and the tooltip states the verdict as a sentence."""
    out = _drive(_rounds_server(_ONE_ROUND) + "roundsHtml();")
    check("flag:PA,SS" not in out, f"the ledger's own token never reaches the screen: {out}")
    check("mAP: failed" in out,
          f"the mAP gate says what it did: {out}")
    check("per class: 2 flagged" in out, f"the per-class gate counts what dropped: {out}")
    check("recall: no verdict" in out, f"a gate the ledger never wrote says so: {out}")
    check("The mean average precision gate failed." in out,
          f"the tooltip states the gate in full: {out}")
    check("Porites astreoides (PA)" in out and "Siderastrea siderea (SS)" in out,
          f"the tooltip names both flagged classes in full with their codes: {out}")
    check("The recall gate recorded no verdict." in out,
          f"the empty gate's tooltip says so: {out}")


def test_an_unpinned_gate_reads_as_words_rather_than_a_token():
    """"unpinned" and "incomparable" are ledger words, not English. The cell
    says what they mean for the run."""
    rows = ("[{ round: 1, run_name: 'round_1', map50_95_B: 0.098, gate_map: 'unpinned',"
            " gate_class: 'incomparable', gate_recall: 'pass' }]")
    out = _drive(_rounds_server(rows) + "roundsHtml();")
    check("mAP: split not pinned" in out, f"unpinned reads as plain words: {out}")
    check("per class: nothing to compare" in out, f"incomparable reads as plain words: {out}")
    check("recall: passed" in out, f"a gate that passed says so: {out}")
    check("unpinned<" not in out and "incomparable<" not in out,
          f"neither ledger word is left on screen: {out}")


def test_a_gate_that_flagged_many_classes_counts_them():
    """A detector can flag seventeen classes, and a cell is no place to name
    them: the count is on screen, four names are in the tooltip and the rest is
    a count too. A class the ledger repeats is one class."""
    rows = ("[{ round: 3, run_name: 'round_3', map50_95_B: 0.5, gate_map: 'pass',"
            " gate_class: 'flag:DL,FF,MC,OA,PA,SS,DL', gate_recall: 'pass' }]")
    out = _drive(_rounds_server(rows) + "roundsHtml();")
    check("per class: 6 flagged" in out,
          f"the repeated code is counted once, so six classes dropped: {out}")
    check("Diploria labyrinthiformis (DL)" in out, f"the first names are in full: {out}")
    check("and 2 others." in out, f"the rest is a count, not a wall of names: {out}")
    check("Porites astreoides" not in out,
          f"the tooltip stops at four names rather than listing every one: {out}")


def test_a_code_the_registry_cannot_resolve_stays_bare_in_the_gates_tooltip():
    """A class the Label Manager does not hold keeps the code the ledger
    wrote; nothing is invented for it."""
    rows = ("[{ round: 4, run_name: 'round_4', map50_95_B: 0.5, gate_map: 'pass',"
            " gate_class: 'flag:ZZ', gate_recall: 'pass' }]")
    out = _drive(_rounds_server(rows) + "roundsHtml();")
    check("per class: 1 flagged" in out, f"one class dropped: {out}")
    check("The per-class gate flagged ZZ." in out,
          f"an unresolved code is stated as it is: {out}")


# ── ROWS ────────────────────────────────────────────────────────────
def test_the_per_site_table_reads_at_the_table_size_and_stripes():
    """Choose Images reports its selection as a table of sites. It rendered at
    12 px with no striping, which is the texture Lauren called hard to read;
    the page's own table class carries 13 px cells, a hairline per row, the
    alternate row wash and the hover wash (design version 2, section 3)."""
    out = _drive("renderStep3SiteSummary(3, { n_selected: 40,"
                 " per_site: { BIT: 12, SVN: 10, BPT: 9, SRW: 5, CSE: 4 },"
                 " cap_frac: 0.25, cap_ok: true,"
                 " shortfalls: [{ species: 'Porites astreoides', achieved: 30,"
                 "                target: 50, reason: 'cap-limited' }] });"
                 "({ html: document.getElementById('s3-result').innerHTML });")
    html = out["html"]
    check('class="remap-table"' in html, f"the table takes the page's table styling: {html}")
    check("font-size:12px" not in html, f"nothing in the summary is set below the floor: {html}")
    check(html.count("<tr>") == 6, f"one head row and five site rows: {html}")
    check('class="hint" tabindex="0"' in html,
          f"a shortfall line is a hint that opens its reason from the keyboard: {html}")
    check("Porites astreoides</em>: 30/50 (capped by the max share per site)" in html,
          f"the shortfall states the counts and the reason: {html}")
    check("vega-report-binomial" in html, f"the species reads as a species, in italic: {html}")


# ── EVALUATE ────────────────────────────────────────────────────────
_EVAL_FIELDS = ("makeEvalField('s7-split', 'test'); makeEvalField('s7-conf', 0.001);"
                "makeEvalField('s7-iou', 0.7); makeEvalField('s7-preview-count', 8);")


def test_the_confidence_upgrade_note_is_one_line_with_its_reason_on_the_line():
    """A project saved before the measurement default was fixed said so in
    three sentences of amber prose on the Evaluate panel. The line now states
    what happened, and why it matters is a tooltip the line opens on hover and
    on keyboard focus (spec sections 6 and 7)."""
    out = _drive("showStep7ConfUpgrade(0.25);"
                 "const el = document.getElementById('s7-conf-upgraded');"
                 "const after = () => { showStep7ConfUpgrade(null); return el.textContent; };"
                 "({ text: el.textContent, tip: el.title, focusable: el.tabIndex,"
                 "   shown: el.style.display, cleared: after() });")
    check(out["text"] == "This project was saved with confidence 0.25, and VEGA set it to "
                         "0.001 when the project opened.",
          f"the line is one sentence stating both numbers: {out['text']!r}")
    check(out["text"].count(".") == 3,
          f"one sentence, the two decimals aside: {out['text']!r}")
    check("low-confidence tail" in out["tip"],
          f"the reason is a tooltip rather than a paragraph: {out['tip']!r}")
    check("measured with" in out["tip"],
          f"the ledger's own rule stays with the reason: {out['tip']!r}")
    check(out["focusable"] == 0, f"the line opens its tooltip from the keyboard (got {out['focusable']!r})")
    check(out["shown"] == "", f"the line is shown (got display {out['shown']!r})")
    check(out["cleared"] == "", "a project with nothing replaced says nothing")


def test_a_saved_setting_off_the_recommendation_says_so_beside_its_field():
    """A project opening Evaluate on a saved overlap of 0.7 says so beside the
    field whose help recommends 0.6, because the number decides which other
    runs the score may be ranked against."""
    out = _drive(_EVAL_FIELDS
                 + "renderEvalOffRecommended({ iou_threshold: { saved: 0.7, recommended: 0.6 } });"
                 + "const note = document.getElementById('s7-iou-off-recommended');"
                 + "({ text: note ? note.textContent : null,"
                 + "   tip: note ? note.title : null,"
                 + "   focusable: note ? note.tabIndex : null,"
                 + "   cls: note ? note.className : null,"
                 + "   colour: note ? note.style.color : null,"
                 + "   shown: note ? note.style.display : null,"
                 + "   beside: note && note.parentNode ? note.parentNode.className : null,"
                 + "   quiet: (document.getElementById('s7-conf-off-recommended') || {textContent: ''}).textContent });")
    check(out["text"] == "This project is saved with overlap 0.7, and VEGA recommends 0.6.",
          f"the line is one sentence stating both numbers: {out['text']!r}")
    check("not one measurement" in (out["tip"] or ""),
          f"the reason is a tooltip, not a paragraph on screen: {out['tip']!r}")
    check(out["focusable"] == 0, f"the line opens its tooltip from the keyboard (got {out['focusable']!r})")
    check(out["cls"] == "hint", f"the note is a hint line: {out['cls']!r}")
    check(out["colour"] == "var(--warning)", f"the note reads as a warning: {out['colour']!r}")
    check(out["shown"] == "", f"the note is shown (got display {out['shown']!r})")
    check(out["beside"] == "form-group", f"the note sits in the field's own row: {out['beside']!r}")
    check(out["quiet"] == "", "a setting that matches the recommendation says nothing")


def test_a_project_whose_settings_all_match_says_nothing():
    """An empty answer clears the notes; an answer the server never gave
    leaves what is on screen alone."""
    out = _drive(_EVAL_FIELDS
                 + "renderEvalOffRecommended({ iou_threshold: { saved: 0.7, recommended: 0.6 } });"
                 + "const note = document.getElementById('s7-iou-off-recommended');"
                 + "renderEvalOffRecommended(null);"
                 + "const kept = note.textContent;"
                 + "renderEvalOffRecommended({});"
                 + "({ kept: kept, cleared: note.textContent, tip: note.title,"
                 + "   hidden: note.style.display });")
    check(out["kept"].startswith("This project is saved with overlap 0.7"),
          "an answer the server never gave leaves the note alone")
    check(out["cleared"] == "", f"an empty answer clears the note: {out['cleared']!r}")
    check(out["tip"] in ("", None), f"a cleared note carries no tooltip either: {out['tip']!r}")
    check(out["hidden"] == "none", f"a cleared note is hidden: {out['hidden']!r}")


def test_every_evaluate_setting_the_server_reports_has_a_field_to_sit_beside():
    """The four settings VEGA recommends are the four the panel holds, so a
    difference the server reports can always be shown where it belongs."""
    js = open(_JS, encoding="utf-8").read()
    block = js[js.index("const EVAL_SETTING_FIELDS"):js.index("const EVAL_SETTING_NAMES")]
    for key, field in (("split", "s7-split"), ("conf_threshold", "s7-conf"),
                       ("iou_threshold", "s7-iou"), ("preview_count", "s7-preview-count")):
        check(f"{key}: '{field}'" in block, f"{key} is not bound to {field}")
    app = open(os.path.join(_ORCH, "app.py"), encoding="utf-8").read()
    recommended = app[app.index("EVAL_RECOMMENDED = {"):app.index("}", app.index("EVAL_RECOMMENDED = {"))]
    for key in ("split", "conf_threshold", "iou_threshold", "preview_count"):
        check(f'"{key}"' in recommended, f"the server no longer recommends {key}")


if __name__ == "__main__":
    print("T7: the family-aware rail\n")
    print("RAIL:")
    run(test_detection_rail_shows_only_the_stages_that_family_has)
    run(test_segmentation_training_set_rail_keeps_its_stages_and_says_skipped)
    run(test_ordinary_project_rail_is_untouched)
    run(test_a_heading_with_nothing_under_it_is_hidden)
    run(test_rail_plan_is_a_pure_function_of_the_state)
    print("ACTIVE:")
    run(test_only_one_tile_is_ever_the_current_stage)
    run(test_a_tile_taken_out_of_the_flow_loses_the_active_highlight)
    print("RESUME:")
    run(test_a_project_opens_on_the_step_it_recorded)
    run(test_a_training_project_reopens_on_inference)
    print("PANELS:")
    run(test_every_reachable_tile_opens_its_panel)
    run(test_the_archived_slots_open_the_annotator_instead_of_blanking_the_window)
    run(test_a_stage_this_project_does_not_have_opens_one_it_does)
    print("TASK:")
    run(test_the_training_task_comes_back_after_a_box_dataset_unpins_it)
    run(test_pinning_twice_keeps_the_first_free_value)
    print("NAMES:")
    run(test_the_class_picker_shows_the_full_name_with_the_code_as_a_chip)
    run(test_the_rendered_picker_rows_carry_the_name_and_a_useful_tooltip)
    run(test_the_picker_reads_the_label_registry_it_already_ships)
    print("FINETUNE:")
    run(test_fine_tune_is_refused_across_families)
    run(test_the_cross_family_hand_off_changes_nothing_on_the_panel)
    run(test_the_fine_tune_button_is_disabled_with_its_reason)
    print("STAGES:")
    run(test_a_detection_project_numbers_its_three_stages)
    run(test_a_training_set_project_keeps_the_seven_stage_numbering)
    run(test_an_ordinary_project_numbers_its_stages_as_the_window_ships_them)
    run(test_every_panel_keeps_its_own_number_when_another_stage_opens)
    run(test_a_segmentation_project_keeps_its_headings_through_the_rail)
    run(test_a_heading_that_spells_its_own_number_is_rewritten)
    run(test_no_folder_token_or_internal_slot_reaches_a_sentence)
    print("SEEDS:")
    run(test_a_round_that_seeded_no_mask_says_why)
    run(test_the_drop_reasons_sit_on_a_focusable_glyph_at_the_reading_size)
    run(test_a_round_that_seeded_masks_still_reads_as_a_success)
    run(test_a_round_that_seeded_one_of_each_reads_in_the_singular)
    run(test_a_round_with_nothing_to_seed_says_so)
    print("PROMOTE:")
    run(test_promote_asks_before_it_overrides_a_gate)
    run(test_the_flagged_classes_are_named_in_full_with_their_codes)
    run(test_a_code_the_registry_cannot_resolve_stays_as_it_is)
    run(test_confirming_the_gate_sends_the_confirmation)
    print("HONESTY:")
    run(test_a_stop_the_server_refused_is_not_reported_as_a_stop)
    run(test_a_stop_the_server_made_clears_the_tile)
    run(test_a_new_launch_takes_the_old_state_line_off_the_screen)
    run(test_a_restart_the_server_cannot_account_for_says_so)
    run(test_an_untracked_step_still_speaks_when_the_server_sends_no_sentence)
    run(test_another_projects_success_is_not_claimed_in_this_window)
    run(test_a_genuine_completion_still_reads_as_a_success)
    run(test_a_process_that_outlived_a_restart_says_so_while_it_runs)
    print("GATES:")
    run(test_the_gates_column_states_plain_words_not_the_ledger_token)
    run(test_an_unpinned_gate_reads_as_words_rather_than_a_token)
    run(test_a_gate_that_flagged_many_classes_counts_them)
    run(test_a_code_the_registry_cannot_resolve_stays_bare_in_the_gates_tooltip)
    print("ROWS:")
    run(test_the_per_site_table_reads_at_the_table_size_and_stripes)
    print("EVALUATE:")
    run(test_the_confidence_upgrade_note_is_one_line_with_its_reason_on_the_line)
    run(test_a_saved_setting_off_the_recommendation_says_so_beside_its_field)
    run(test_a_project_whose_settings_all_match_says_nothing)
    run(test_every_evaluate_setting_the_server_reports_has_a_field_to_sit_beside)
    print("COPY:")
    run(test_the_new_copy_obeys_the_house_rules)
    run(test_the_id_contract_is_kept)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} passed")
    for name, ok, err in _RESULTS:
        if not ok:
            print(f"\n--- {name} ---\n{err}")
    sys.exit(0 if passed == total else 1)
