"""
Starting model and training data redesign (2026-09-21): the orchestrator.js
half of the "Continue from a VEGA model" advanced disclosure that replaced
the old "Continue from a previous run" group inside the base-model select,
and the training data summary (Pictures / Split sentences) on the Train
panel. Spec: docs/superpowers/specs/2026-09-21-vega-starting-model-and-
training-data-design.md. Plan: docs/superpowers/plans/2026-09-21-vega-
starting-model-and-training-data.md.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_continue_from_js.py

Node drives orchestrator.js inside a fake DOM (the scenario driver pattern of
test_ui_sweep_20260905.py, adapted here to a version that can await async
functions: the scenario body runs inside an async IIFE the driver wraps it
in, so a scenario may use `await` and must end with an explicit `return`).

Covers:
  CONFIG     - collectConfig(6) sends model from #s6-model always, and
               model_path from #s6-continue-from only when it is not blank.
  CONTINUE   - onStep6ContinueChange stashes and restores Freeze layers and
               the learning rate, disables/enables #s6-model, and writes the
               replaced hint; applyStep6ModelPath adds a one-off option when
               the run is not already listed and opens the disclosure;
               gotoStep6Start('base') clears the pick; gotoStep6Start
               ('champion') selects the champion option for the panel's task
               or reports there is none.
  TASKFILTER - the task filter hides and disables a Continue-from option of
               the other task, clears a hidden selection, and writes the
               visible count into the disclosure summary.
  SUMMARY    - refreshStep6DataSummary builds the documented query string,
               writes both sentences verbatim on success, writes the error
               sentence into both lines on a 400, and a slow answer for a
               stale query never overwrites what a newer one already wrote.
  STATIC     - orchestrator.js no longer mentions the retired ids and
               functions; switchStep(6) no longer opens the runs card, which
               openStep7Panel opens instead.
  SMOKE      - node --check on orchestrator.js and (read-only) test_vega_runs.js.
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
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_VEGA_RUNS_TEST_JS = os.path.join(_HERE, "test_vega_runs.js")

_fail = 0
_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


_CACHE = {}


def _read(path):
    if path not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _node():
    for cand in ("/usr/bin/node", "node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


# Node driver, adapted from test_ui_sweep_20260905.py's _NODE_DRIVER: same
# fake El/document (getElementById mints one element per id, querySelectorAll
# always answers [], appendChild/removeChild track .children), same never-
# resolving default fetch so an un-stubbed background call parks harmlessly.
# The addition: the scenario body is wrapped in an async IIFE by the driver
# itself (a classic vm script cannot use top-level await), so a scenario may
# freely `await` an async orchestrator.js function; it must end with an
# explicit `return <value>;` rather than a bare trailing expression.
_NODE_DRIVER = r"""
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const scenario = fs.readFileSync(process.argv[3], 'utf8');
class El {
  constructor(id) {
    this.id = id || ''; this.style = {}; this.dataset = {}; this.value = '';
    this.checked = false; this.disabled = false; this.hidden = false;
    this.innerHTML = ''; this.textContent = ''; this.title = ''; this.options = [];
    this.children = []; this.attrs = {}; this.className = ''; this.tagName = 'DIV';
    this.selectedIndex = -1; this.childElementCount = 0; this.open = false;
    const classes = new Set();
    this.classList = {
      add: (c) => classes.add(c), remove: (c) => classes.delete(c),
      toggle: (c, f) => { if (f === undefined) f = !classes.has(c); f ? classes.add(c) : classes.delete(c); return f; },
      contains: (c) => classes.has(c),
    };
    this._listeners = {};
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) { this.children.push(c); this.childElementCount = this.children.length; return c; }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); this.childElementCount = this.children.length; return c; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener(type, fn) { (this._listeners[type] = this._listeners[type] || []).push(fn); }
  removeEventListener() {}
  dispatchEvent(evt) { (this._listeners[evt.type] || []).forEach((fn) => fn(evt)); return true; }
  closest() { return null; }
  scrollIntoView() {}
}
const els = new Map();
const document = {
  body: new El('body'), documentElement: new El('html'),
  getElementById(id) { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); },
  querySelector() { return null; },
  querySelectorAll(sel) {
    if (sel === '.s6-class-cb') return Array.from(els.values()).filter((e) => e.classList.contains('s6-class-cb'));
    return [];
  },
  createElement(tag) { const e = new El(''); e.tagName = String(tag || 'DIV').toUpperCase(); return e; },
  addEventListener() {},
};
document.body.dataset.projectLoaded = 'false';
const sandbox = {
  document, console, setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: () => new Promise(() => {}), alert() {}, prompt() { return null; },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  location: { href: 'http://127.0.0.1:5050/', search: '', hostname: '127.0.0.1', protocol: 'http:' },
  URLSearchParams, encodeURIComponent, decodeURIComponent, JSON, Promise,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'orchestrator.js' });
const wrapped = `(async () => {\n${scenario}\n})()`;
Promise.resolve(vm.runInContext(wrapped, sandbox, { filename: 'scenario.js' }))
  .then((out) => { process.stdout.write(JSON.stringify(out === undefined ? null : out)); })
  .catch((err) => { process.stderr.write(String((err && err.stack) || err)); process.exitCode = 1; });
"""


def _drive(scenario):
    """Run a scenario (an async-IIFE body; must end with `return <value>;`)
    against orchestrator.js in Node and return its JSON result."""
    node = _node()
    check(node is not None, "node not available")
    tmp = tempfile.mkdtemp(prefix="step6_continue_from_js_")
    try:
        driver = os.path.join(tmp, "driver.js")
        scen = os.path.join(tmp, "scenario.js")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(_NODE_DRIVER)
        with open(scen, "w", encoding="utf-8") as fh:
            fh.write(scenario)
        r = subprocess.run([node, driver, _JS, scen], capture_output=True, text=True, timeout=60)
        check(r.returncode == 0, f"node scenario failed:\n{r.stderr}")
        return json.loads(r.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── collectConfig(6) ────────────────────────────────────────────────
def test_collect_config_sends_model_path_only_when_continue_from_is_set():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-model').value = 'yolo11m-seg.pt';
$('s6-continue-from').value = '';
const blank = collectConfig(6);
$('s6-continue-from').value = '/x/runs/run_a/weights/best.pt';
const chosen = collectConfig(6);
return {
  blankModel: blank.model, blankHasPath: Object.prototype.hasOwnProperty.call(blank, 'model_path'),
  chosenModel: chosen.model, chosenPath: chosen.model_path,
};
""")
    check(result["blankModel"] == "yolo11m-seg.pt", f"model always comes from #s6-model: {result}")
    check(result["blankHasPath"] is False, f"a blank continue-from pick must send no model_path key: {result}")
    check(result["chosenModel"] == "yolo11m-seg.pt", f"model_path never overwrites model: {result}")
    check(result["chosenPath"] == "/x/runs/run_a/weights/best.pt",
          f"a continue-from pick must be sent as model_path: {result}")


# ── onStep6ContinueChange ───────────────────────────────────────────
def test_continue_change_stashes_and_restores_freeze_and_lr_and_toggles_model():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-freeze').value = '';
$('s6-lr0').value = '0.01';
$('s6-model').disabled = false;
$('s6-model-note').textContent = 'the free hint';
const sel = $('s6-continue-from');
const group = document.createElement('optgroup');
const opt = document.createElement('option');
opt.value = '/x/runs/run_a/weights/best.pt';
opt.dataset.task = 'seg';
opt.dataset.runName = 'run_a';
group.appendChild(opt);
sel.appendChild(group);
sel.value = opt.value;
onStep6ContinueChange();
const picked = {
  freeze: $('s6-freeze').value, lr0: $('s6-lr0').value,
  modelDisabled: $('s6-model').disabled, hint: $('s6-model-note').textContent,
};
sel.value = '';
onStep6ContinueChange();
const cleared = {
  freeze: $('s6-freeze').value, lr0: $('s6-lr0').value,
  modelDisabled: $('s6-model').disabled, hint: $('s6-model-note').textContent,
};
return { picked, cleared };
""")
    picked = result["picked"]
    check(picked["freeze"] == "10", f"freeze must become 10 while a run is chosen: {picked}")
    check(picked["lr0"] == "0.001", f"the learning rate must become 0.001 while a run is chosen: {picked}")
    check(picked["modelDisabled"] is True, f"#s6-model must disable while a run is chosen: {picked}")
    check(picked["hint"] == "Replaced: this run continues from run_a. "
          "Clear the choice under Continue from a VEGA model to start from a base model.",
          f"the replaced hint must name the run: {picked}")
    cleared = result["cleared"]
    check(cleared["freeze"] == "" and cleared["lr0"] == "0.01",
          f"clearing the pick must restore the operator's own numbers: {cleared}")
    check(cleared["modelDisabled"] is False, f"#s6-model must re-enable once cleared: {cleared}")
    check(cleared["hint"] == "the free hint", f"clearing the pick must restore the free hint: {cleared}")


def test_continue_change_does_not_clobber_stash_on_a_second_pick():
    """A second pick made without clearing first must not overwrite the
    stashed operator values with VEGA's own continuation values."""
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-freeze').value = '3';
$('s6-lr0').value = '0.02';
const sel = $('s6-continue-from');
const opt1 = document.createElement('option');
opt1.value = '/x/runs/run_a/weights/best.pt'; opt1.dataset.task = 'seg'; opt1.dataset.runName = 'run_a';
const opt2 = document.createElement('option');
opt2.value = '/x/runs/run_b/weights/best.pt'; opt2.dataset.task = 'seg'; opt2.dataset.runName = 'run_b';
sel.appendChild(opt1); sel.appendChild(opt2);
sel.value = opt1.value;
onStep6ContinueChange();
sel.value = opt2.value;
onStep6ContinueChange();
sel.value = '';
onStep6ContinueChange();
return { freeze: $('s6-freeze').value, lr0: $('s6-lr0').value };
""")
    check(result["freeze"] == "3" and result["lr0"] == "0.02",
          f"the operator's original numbers must survive a pick change: {result}")


# ── applyStep6ModelPath ─────────────────────────────────────────────
def test_apply_model_path_adds_a_one_off_option_and_opens_the_details():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-task').value = 'segment';
applyStep6ModelPath('/x/runs/run_z/weights/best.pt');
const sel = $('s6-continue-from');
return {
  value: sel.value, detailsOpen: $('s6-continue-details').open,
  childCount: sel.children.length,
};
""")
    check(result["value"] == "/x/runs/run_z/weights/best.pt", f"the value must be selected: {result}")
    check(result["detailsOpen"] is True, f"the disclosure must open: {result}")
    check(result["childCount"] == 1, f"exactly one option must be added: {result}")


def test_apply_model_path_reuses_an_existing_option_without_duplicating():
    result = _drive(r"""
const sel = document.getElementById('s6-continue-from');
const opt = document.createElement('option');
opt.value = '/x/runs/run_a/weights/best.pt'; opt.dataset.task = 'seg'; opt.dataset.runName = 'run_a';
sel.appendChild(opt);
applyStep6ModelPath('/x/runs/run_a/weights/best.pt');
return { value: sel.value, childCount: sel.children.length };
""")
    check(result["value"] == "/x/runs/run_a/weights/best.pt", f"value: {result}")
    check(result["childCount"] == 1, f"an already-listed run must not be duplicated: {result}")


# ── gotoStep6Start ───────────────────────────────────────────────────
def test_goto_step6_start_base_clears_the_pick():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-freeze').value = '';
$('s6-lr0').value = '0.01';
const sel = $('s6-continue-from');
const opt = document.createElement('option');
opt.value = '/x/runs/run_a/weights/best.pt'; opt.dataset.task = 'seg'; opt.dataset.runName = 'run_a';
sel.appendChild(opt);
sel.value = opt.value;
onStep6ContinueChange();
await gotoStep6Start('base');
return { value: sel.value, freeze: $('s6-freeze').value, lr0: $('s6-lr0').value,
         modelDisabled: $('s6-model').disabled };
""")
    check(result["value"] == "", f"'base' must clear the continue-from select: {result}")
    check(result["freeze"] == "" and result["lr0"] == "0.01",
          f"'base' must restore the operator's own freeze and learning rate: {result}")
    check(result["modelDisabled"] is False, f"'base' must re-enable #s6-model: {result}")


def test_goto_step6_start_champion_selects_the_champion_for_the_panel_task():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
api = async (url) => {
  if (url.indexOf('/api/step/6/list_runs') === 0) {
    return { champion_run_dir: '/x/runs/run_c', champion_task: 'seg' };
  }
  if (url.indexOf('/api/step/6/all_runs') === 0) {
    return { runs: [{ run_dir: '/x/runs/run_c', run_name: 'champion_run', origin: 'project',
                       project_name: 'p', task: 'seg', family: 'mask', map50_95: 0.842, has_weights: true }] };
  }
  return {};
};
$('s6-task').value = 'segment';
await gotoStep6Start('champion');
return { value: $('s6-continue-from').value, detailsOpen: $('s6-continue-details').open };
""")
    check(result["value"] == "/x/runs/run_c/weights/best.pt", f"the champion must be selected: {result}")
    check(result["detailsOpen"] is True, f"the disclosure must open on a champion pick: {result}")


def test_goto_step6_start_champion_reports_when_none_matches_the_task():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
api = async (url) => {
  if (url.indexOf('/api/step/6/list_runs') === 0) {
    return { champion_run_dir: '/x/runs/run_c', champion_task: 'detect' };
  }
  if (url.indexOf('/api/step/6/all_runs') === 0) {
    return { runs: [{ run_dir: '/x/runs/run_c', run_name: 'champion_run', origin: 'project',
                       project_name: 'p', task: 'detect', family: 'box', map50_95: 0.7, has_weights: true }] };
  }
  return {};
};
$('s6-task').value = 'segment';
await gotoStep6Start('champion');
return { value: $('s6-continue-from').value, resultHtml: document.getElementById('s6-result').innerHTML };
""")
    check(result["value"] == "", f"no segmentation champion exists, so nothing is picked: {result}")
    check("No champion has been promoted yet. Evaluate a run and press Promote first." in result["resultHtml"],
          f"the panel must say there is no champion for this task: {result}")


# ── task filter ──────────────────────────────────────────────────────
def test_task_filter_hides_disables_and_clears_a_mismatched_pick():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
const sel = $('s6-continue-from');
sel.dataset.loaded = '1';
const segOpt = document.createElement('option');
segOpt.value = '/x/runs/seg_run/weights/best.pt'; segOpt.dataset.task = 'seg'; segOpt.dataset.runName = 'seg_run';
const detOpt = document.createElement('option');
detOpt.value = '/x/runs/det_run/weights/best.pt'; detOpt.dataset.task = 'detect'; detOpt.dataset.runName = 'det_run';
sel.appendChild(segOpt); sel.appendChild(detOpt);
sel.value = segOpt.value;
$('s6-task').value = 'segment';
applyStep6ContinueTaskFilter();
const seg = { segHidden: segOpt.hidden, segDisabled: segOpt.disabled,
              detHidden: detOpt.hidden, detDisabled: detOpt.disabled,
              value: sel.value, summary: $('s6-continue-summary').textContent };
$('s6-task').value = 'detect';
applyStep6ContinueTaskFilter();
const det = { segHidden: segOpt.hidden, segDisabled: segOpt.disabled,
              detHidden: detOpt.hidden, detDisabled: detOpt.disabled,
              value: sel.value, summary: $('s6-continue-summary').textContent };
return { seg, det };
""")
    seg = result["seg"]
    check(seg["segHidden"] is False and seg["segDisabled"] is False, f"the matching run stays offered: {seg}")
    check(seg["detHidden"] is True and seg["detDisabled"] is True, f"the other task's run is hidden: {seg}")
    check(seg["value"] == "/x/runs/seg_run/weights/best.pt", f"the matching pick survives: {seg}")
    check(seg["summary"] == "Continue from a VEGA model (advanced): 1 run to choose from", f"summary count: {seg}")
    det = result["det"]
    check(det["detHidden"] is False and det["detDisabled"] is False, f"the now-matching run is offered: {det}")
    check(det["segHidden"] is True and det["segDisabled"] is True, f"the seg run is hidden now: {det}")
    check(det["value"] == "", f"switching task must clear a pick the filter just hid: {det}")
    check(det["summary"] == "Continue from a VEGA model (advanced): 1 run to choose from", f"summary count: {det}")


def test_task_filter_summary_counts_zero_and_plural():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
const sel = $('s6-continue-from');
sel.dataset.loaded = '1';
applyStep6ContinueTaskFilter();
const zero = $('s6-continue-summary').textContent;
for (const name of ['run_a', 'run_b']) {
  const opt = document.createElement('option');
  opt.value = '/x/runs/' + name + '/weights/best.pt'; opt.dataset.task = 'seg'; opt.dataset.runName = name;
  sel.appendChild(opt);
}
$('s6-task').value = 'segment';
applyStep6ContinueTaskFilter();
const two = $('s6-continue-summary').textContent;
return { zero, two };
""")
    check(result["zero"] == "Continue from a VEGA model (advanced): no runs to choose from yet", f"zero: {result}")
    check(result["two"] == "Continue from a VEGA model (advanced): 2 runs to choose from", f"two: {result}")


def test_task_filter_leaves_the_templates_summary_when_the_count_is_unknown():
    """Before refreshStep6ContinueOptions() has completed once (marked by
    #s6-continue-from's own dataset.loaded), the count is genuinely unread,
    not zero, so the template's own summary text must survive untouched."""
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-continue-summary').textContent = 'Continue from a VEGA model (advanced)';
$('s6-task').value = 'segment';
applyStep6ContinueTaskFilter();
return { summary: $('s6-continue-summary').textContent, loaded: $('s6-continue-from').dataset.loaded || '' };
""")
    check(result["loaded"] == "", f"the select must carry no loaded marker yet: {result}")
    check(result["summary"] == "Continue from a VEGA model (advanced)",
          f"an unread count must not overwrite the template's own summary text: {result}")


def test_refresh_continue_options_marks_the_select_loaded_on_success():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
api = async (url) => {
  if (url.indexOf('/api/step/6/list_runs') === 0) return { champion_run_dir: '', champion_task: '' };
  if (url.indexOf('/api/step/6/all_runs') === 0) return { runs: [] };
  return {};
};
$('s6-continue-summary').textContent = 'Continue from a VEGA model (advanced)';
$('s6-task').value = 'segment';
await refreshStep6ContinueOptions();
return {
  loaded: document.getElementById('s6-continue-from').dataset.loaded,
  summary: document.getElementById('s6-continue-summary').textContent,
};
""")
    check(result["loaded"] == "1", f"a successful fetch must mark the select loaded: {result}")
    check(result["summary"] == "Continue from a VEGA model (advanced): no runs to choose from yet",
          f"a known, empty count must now be reported: {result}")


def test_a_refresh_in_flight_when_a_hand_off_arrives_does_not_undo_it():
    """The parity-reviewer race (2026-09-22): switchStep(6)'s background
    refreshStep6ContinueOptions() call captured the select's value before its
    fetch, then restored that stale value once the fetch finally resolved,
    wiping a champion or run picked in between. A refresh started before
    applyStep6ModelPath and resolved after it must leave the chosen value and
    the open disclosure untouched: applyStep6ModelPath bumps the same request
    token, so the slower refresh finds itself stale on completion and
    discards its whole answer, never reaching the select at all."""
    result = _drive(r"""
let resolveList, resolveAll;
const listPromise = new Promise((resolve) => { resolveList = resolve; });
const allPromise = new Promise((resolve) => { resolveAll = resolve; });
api = async (url) => {
  if (url.indexOf('/api/step/6/list_runs') === 0) return listPromise;
  if (url.indexOf('/api/step/6/all_runs') === 0) return allPromise;
  return {};
};
const sel = document.getElementById('s6-continue-from');
const pending = refreshStep6ContinueOptions(); // starts; its fetch is still pending below
applyStep6ModelPath('/x/runs/hand_off/weights/best.pt'); // arrives while the refresh is in flight
const afterHandOff = {
  value: sel.value, detailsOpen: document.getElementById('s6-continue-details').open,
  childCount: sel.children.length,
};
resolveList({ champion_run_dir: '', champion_task: '' });
resolveAll({ runs: [{ run_dir: '/x/runs/other_run', run_name: 'other_run', origin: 'project',
                       project_name: 'p', task: 'seg', family: 'mask', map50_95: 0.5, has_weights: true }] });
await pending; // the stale refresh finally resolves
return {
  afterHandOff,
  finalValue: sel.value,
  finalDetailsOpen: document.getElementById('s6-continue-details').open,
  finalChildCount: sel.children.length,
};
""")
    hand_off = result["afterHandOff"]
    check(hand_off["value"] == "/x/runs/hand_off/weights/best.pt", f"the hand-off must apply at once: {hand_off}")
    check(hand_off["detailsOpen"] is True, f"the hand-off must open the disclosure at once: {hand_off}")
    check(result["finalValue"] == "/x/runs/hand_off/weights/best.pt",
          f"the stale refresh must not restore or blank the hand-off's value: {result}")
    check(result["finalDetailsOpen"] is True,
          f"the stale refresh must not close the disclosure the hand-off opened: {result}")
    check(result["finalChildCount"] == result["afterHandOff"]["childCount"],
          f"a stale refresh must never touch the option list at all: {result}")


def test_two_overlapping_refreshes_leave_only_the_newer_ones_options():
    """Two refreshStep6ContinueOptions() calls in flight together (the
    background one switchStep(6) fires and an explicit one gotoStep6Start
    awaits, say) must not both write the select: whichever started first must
    discard its answer once the second has started, however their fetches
    happen to finish."""
    result = _drive(r"""
let resolveList1, resolveAll1, resolveList2, resolveAll2;
const listP1 = new Promise((resolve) => { resolveList1 = resolve; });
const allP1 = new Promise((resolve) => { resolveAll1 = resolve; });
const listP2 = new Promise((resolve) => { resolveList2 = resolve; });
const allP2 = new Promise((resolve) => { resolveAll2 = resolve; });
let listCalls = 0, allCalls = 0;
api = async (url) => {
  if (url.indexOf('/api/step/6/list_runs') === 0) { listCalls += 1; return listCalls === 1 ? listP1 : listP2; }
  if (url.indexOf('/api/step/6/all_runs') === 0) { allCalls += 1; return allCalls === 1 ? allP1 : allP2; }
  return {};
};
const older = refreshStep6ContinueOptions(); // token 1
const newer = refreshStep6ContinueOptions(); // token 2, started while the older one is still in flight
// The newer call resolves first (a race says nothing about resolution order):
resolveList2({ champion_run_dir: '', champion_task: '' });
resolveAll2({ runs: [{ run_dir: '/x/runs/newer_run', run_name: 'newer_run', origin: 'project',
                       project_name: 'p', task: 'seg', family: 'mask', map50_95: 0.5, has_weights: true }] });
await newer;
const sel = document.getElementById('s6-continue-from');
const afterNewer = step6ContinueOptionElements(sel).map((o) => o.value);
// The older call resolves last, with different data:
resolveList1({ champion_run_dir: '', champion_task: '' });
resolveAll1({ runs: [{ run_dir: '/x/runs/older_run', run_name: 'older_run', origin: 'project',
                       project_name: 'p', task: 'seg', family: 'mask', map50_95: 0.9, has_weights: true }] });
await older;
return { afterNewer, afterOlder: step6ContinueOptionElements(sel).map((o) => o.value) };
""")
    check(result["afterNewer"] == ["/x/runs/newer_run/weights/best.pt"],
          f"the newer, first-issued-second call must write its own options: {result}")
    check(result["afterOlder"] == ["/x/runs/newer_run/weights/best.pt"],
          f"the older call must not add or replace anything once it resolves last: {result}")


def test_on_step6_task_change_reruns_the_continue_from_filter():
    """The plan's contract: onStep6TaskChange re-runs the continue-from task
    filter after the sibling swap and clears a hidden choice."""
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-model').options = [
  { value: 'yolo11m-seg.pt', hidden: false }, { value: 'yolo11m.pt', hidden: false },
];
$('s6-model').value = 'yolo11m-seg.pt';
const sel = $('s6-continue-from');
const segOpt = document.createElement('option');
segOpt.value = '/x/runs/seg_run/weights/best.pt'; segOpt.dataset.task = 'seg'; segOpt.dataset.runName = 'seg_run';
sel.appendChild(segOpt);
sel.value = segOpt.value;
$('s6-freeze').value = '';
onStep6ContinueChange();
$('s6-task').value = 'detect';
onStep6TaskChange();
return { value: sel.value, modelDisabled: $('s6-model').disabled, freeze: $('s6-freeze').value };
""")
    check(result["value"] == "", f"a task change must clear a continue-from pick of the other task: {result}")
    check(result["modelDisabled"] is False, f"clearing the pick must re-enable #s6-model: {result}")
    check(result["freeze"] == "", f"clearing the pick must restore the operator's freeze value: {result}")


# ── refreshStep6DataSummary ─────────────────────────────────────────
def test_data_summary_query_string_matches_the_documented_fields():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-source').value = '/x/inprocess/_composite/qs1';
$('s6-pin-split').checked = true;
$('s6-valid').value = '0.2';
$('s6-test').value = '0.1';
$('s6-task').value = 'detect';
return { query: step6DataSummaryQuery() };
""")
    check(result["query"] == "source_dir=%2Fx%2Finprocess%2F_composite%2Fqs1&pin_split=1"
          "&valid_ratio=0.2&test_ratio=0.1&task=detect",
          f"query string: {result['query']}")


def test_data_summary_query_blank_source_and_unchecked_split():
    result = _drive(r"""
const $ = (id) => document.getElementById(id);
$('s6-source').value = '';
$('s6-pin-split').checked = false;
$('s6-valid').value = '';
$('s6-test').value = '';
$('s6-task').value = 'segment';
return { query: step6DataSummaryQuery() };
""")
    check(result["query"] == "source_dir=&pin_split=0&valid_ratio=0.2&test_ratio=0.1&task=segment",
          f"a blank source and unticked split, with blank fractions taking their defaults: {result['query']}")


def test_data_summary_writes_both_sentences_verbatim():
    result = _drive(r"""
api = async (url) => ({
  ok: true, pictures_sentence: '1,036 images, 3,412 outlines, 2 classes',
  split: { rule: 'transect', sentence: 'Split by transect: transects 1 to 4 train, transect 5 validation, '
    + 'transect 6 test, about 812 / 118 / 106 images.' },
});
await refreshStep6DataSummary();
return {
  pictures: document.getElementById('s6-data-pictures').textContent,
  split: document.getElementById('s6-data-split').textContent,
};
""")
    check(result["pictures"] == "1,036 images, 3,412 outlines, 2 classes",
          f"the pictures sentence must render exactly as the server wrote it: {result}")
    check(result["split"] == "Split by transect: transects 1 to 4 train, transect 5 validation, "
          "transect 6 test, about 812 / 118 / 106 images.",
          f"the split sentence must render exactly as the server wrote it: {result}")


def test_data_summary_writes_the_servers_error_into_both_lines():
    result = _drive(r"""
api = async (url) => ({ error: 'That folder is outside the training roots VEGA reads.' });
await refreshStep6DataSummary();
return {
  pictures: document.getElementById('s6-data-pictures').textContent,
  split: document.getElementById('s6-data-split').textContent,
};
""")
    check(result["pictures"] == "That folder is outside the training roots VEGA reads.", f"{result}")
    check(result["split"] == "That folder is outside the training roots VEGA reads.", f"{result}")


def test_data_summary_a_slow_earlier_answer_never_overwrites_a_newer_one():
    result = _drive(r"""
let calls = 0;
api = async (url) => {
  calls += 1;
  if (calls === 1) {
    // The first (stale) query resolves last, after a macrotask delay, and
    // must lose the race against the second (current) query below.
    await new Promise((resolve) => setTimeout(resolve, 20));
    return { ok: true, pictures_sentence: 'STALE', split: { sentence: 'STALE split' } };
  }
  return { ok: true, pictures_sentence: 'CURRENT', split: { sentence: 'CURRENT split' } };
};
const first = refreshStep6DataSummary();
const second = refreshStep6DataSummary();
await Promise.all([first, second]);
return {
  pictures: document.getElementById('s6-data-pictures').textContent,
  split: document.getElementById('s6-data-split').textContent,
};
""")
    check(result["pictures"] == "CURRENT", f"a slow stale answer must not overwrite the current one: {result}")
    check(result["split"] == "CURRENT split", f"a slow stale answer must not overwrite the current one: {result}")


def test_schedule_data_summary_debounces_to_one_call():
    result = _drive(r"""
let calls = 0;
api = async (url) => { calls += 1; return { ok: true, pictures_sentence: 'X', split: { sentence: 'Y' } }; };
scheduleStep6DataSummary();
scheduleStep6DataSummary();
scheduleStep6DataSummary();
await new Promise((resolve) => setTimeout(resolve, STEP6_DATA_SUMMARY_DEBOUNCE_MS + 50));
return { calls };
""")
    check(result["calls"] == 1, f"three calls inside the debounce window must send one query: {result}")


# ── class picker count ───────────────────────────────────────────────
def test_classes_count_reads_the_checkboxes_on_the_page():
    """The DOM stub cannot parse the HTML string renderStep6Classes writes
    into innerHTML (no real parser in the fake document), so this drives
    updateStep6ClassesCount() directly against checkbox elements registered
    the way the stub's querySelectorAll('.s6-class-cb') can see them; a
    separate static check below confirms renderStep6Classes calls it."""
    result = _drive(r"""
const cb1 = document.getElementById('s6-cb-1');
cb1.classList.add('s6-class-cb'); cb1.checked = true;
const cb2 = document.getElementById('s6-cb-2');
cb2.classList.add('s6-class-cb'); cb2.checked = true;
updateStep6ClassesCount();
const bothTicked = document.getElementById('s6-classes-count').textContent;
cb1.checked = false;
updateStep6ClassesCount();
const oneTicked = document.getElementById('s6-classes-count').textContent;
return { bothTicked, oneTicked };
""")
    check(result["bothTicked"] == "2 of 2 ticked", f"both ticked: {result}")
    check(result["oneTicked"] == "1 of 2 ticked", f"unticking one must read 1 of 2: {result}")


def test_render_step6_classes_calls_the_count_update():
    js = _read(_JS)
    fn_start = js.index("function renderStep6Classes(")
    fn_end = js.index("\n}\n", fn_start)
    body = js[fn_start:fn_end]
    check(body.count("updateStep6ClassesCount();") >= 2,
          "renderStep6Classes must update the heading count on the no-classes branch and the normal render")


def test_select_all_step6_classes_updates_count_and_summary():
    js = _read(_JS)
    fn_start = js.index("function selectAllStep6Classes(")
    fn_end = js.index("\n}\n", fn_start)
    body = js[fn_start:fn_end]
    check("updateStep6ClassesCount();" in body, "Select all / Clear must update the heading count")
    check("refreshStep6DataSummary();" in body,
          "Select all / Clear must re-read the training data summary (setting .checked fires no 'change')")


# ── static sweeps ────────────────────────────────────────────────────
def test_retired_ids_and_functions_are_gone():
    js = _read(_JS)
    for needle in ("s6-model-runs-optgroup", "applyStep6BuiltinPreset", "gotoStep6Preset",
                   "s6-preset-retrain", "s6-preset-finetune"):
        check(needle not in js, f"retired reference {needle!r} still present in orchestrator.js")
    # refreshStep6ModelRunOptions is retired as a callable (no definition and
    # no live call site); its old name may still appear in a historical
    # comment ("Replaces refreshStep6ModelRunOptions(): ..."), which is not
    # a dangling reference, so this checks the definition and a call form
    # specifically rather than the bare substring.
    check("function refreshStep6ModelRunOptions(" not in js, "refreshStep6ModelRunOptions must no longer be defined")
    check("refreshStep6ModelRunOptions();" not in js, "no call site of refreshStep6ModelRunOptions may survive")
    check("await refreshStep6ModelRunOptions" not in js, "no awaited call site of refreshStep6ModelRunOptions may survive")


def test_switch_step_no_longer_opens_the_runs_card_but_evaluate_panel_does():
    js = _read(_JS)
    sw_start = js.index("function switchStep(")
    sw_end = js.index("function getNavOrder(")
    sw = js[sw_start:sw_end]
    check("openStep6RunsCard()" not in sw, "switchStep(6) must no longer initialise the runs card directly")
    check("refreshStep6ContinueOptions()" in sw, "switchStep(6) must load the continue-from options")
    check("refreshStep6DataSummary()" in sw, "switchStep(6) must load the training data summary")
    p7_start = js.index("async function openStep7Panel(")
    p7_end = js.index("\n}\n", p7_start)
    p7 = js[p7_start:p7_end]
    check("openStep6RunsCard()" in p7, "openStep7Panel must open the runs card when the disclosure is open")
    check("step7AllModelsOpen()" in p7, "openStep7Panel must gate the load on the disclosure's open state")
    check("function step7AllModelsOpen(" in js, "step7AllModelsOpen must be defined")
    check("function wireStep7AllModelsToggle(" in js, "wireStep7AllModelsToggle must be defined")


def test_promote_run_uses_the_renamed_continue_options_refresh():
    js = _read(_JS)
    fn_start = js.index("async function promoteRun(")
    fn_end = js.index("\n}\n", fn_start)
    body = js[fn_start:fn_end]
    check("refreshStep6ContinueOptions()" in body, "promoteRun must refresh the renamed continue-from options")


def test_on_vega_finetune_from_switches_to_train_before_applying_the_path():
    js = _read(_JS)
    fn_start = js.index("function onVegaFinetuneFrom(")
    fn_end = js.index("\n}\n", fn_start)
    body = js[fn_start:fn_end]
    switch_idx = body.index("switchStep(6)")
    apply_idx = body.index("applyStep6ModelPath(")
    check(switch_idx < apply_idx, "onVegaFinetuneFrom must switchStep(6) before applyStep6ModelPath")


# ── smoke ────────────────────────────────────────────────────────────
def test_node_check_orchestrator_js():
    node = _node()
    check(node is not None, "node not available")
    r = subprocess.run([node, "--check", _JS], capture_output=True, text=True)
    check(r.returncode == 0, f"orchestrator.js node --check failed:\n{r.stderr}")


def test_node_check_vega_runs_js_unbroken():
    """Read-only for this task: confirms nothing here broke the runs card's
    own suite by simply failing to parse; test_vega_runs.js's own assertions
    are C2's to run."""
    node = _node()
    check(node is not None, "node not available")
    check(os.path.isfile(_VEGA_RUNS_TEST_JS), f"{_VEGA_RUNS_TEST_JS} missing")
    r = subprocess.run([node, "--check", _VEGA_RUNS_TEST_JS], capture_output=True, text=True)
    check(r.returncode == 0, f"test_vega_runs.js node --check failed:\n{r.stderr}")


if __name__ == "__main__":
    print("Starting model and training data: orchestrator.js sweep (2026-09-21)\n")
    print("SMOKE:")
    run(test_node_check_orchestrator_js)
    run(test_node_check_vega_runs_js_unbroken)
    print("CONFIG:")
    run(test_collect_config_sends_model_path_only_when_continue_from_is_set)
    print("CONTINUE:")
    run(test_continue_change_stashes_and_restores_freeze_and_lr_and_toggles_model)
    run(test_continue_change_does_not_clobber_stash_on_a_second_pick)
    run(test_apply_model_path_adds_a_one_off_option_and_opens_the_details)
    run(test_apply_model_path_reuses_an_existing_option_without_duplicating)
    run(test_goto_step6_start_base_clears_the_pick)
    run(test_goto_step6_start_champion_selects_the_champion_for_the_panel_task)
    run(test_goto_step6_start_champion_reports_when_none_matches_the_task)
    print("TASKFILTER:")
    run(test_task_filter_hides_disables_and_clears_a_mismatched_pick)
    run(test_task_filter_summary_counts_zero_and_plural)
    run(test_task_filter_leaves_the_templates_summary_when_the_count_is_unknown)
    run(test_refresh_continue_options_marks_the_select_loaded_on_success)
    run(test_a_refresh_in_flight_when_a_hand_off_arrives_does_not_undo_it)
    run(test_two_overlapping_refreshes_leave_only_the_newer_ones_options)
    run(test_on_step6_task_change_reruns_the_continue_from_filter)
    print("SUMMARY:")
    run(test_data_summary_query_string_matches_the_documented_fields)
    run(test_data_summary_query_blank_source_and_unchecked_split)
    run(test_data_summary_writes_both_sentences_verbatim)
    run(test_data_summary_writes_the_servers_error_into_both_lines)
    run(test_data_summary_a_slow_earlier_answer_never_overwrites_a_newer_one)
    run(test_schedule_data_summary_debounces_to_one_call)
    run(test_classes_count_reads_the_checkboxes_on_the_page)
    run(test_render_step6_classes_calls_the_count_update)
    run(test_select_all_step6_classes_updates_count_and_summary)
    print("STATIC:")
    run(test_retired_ids_and_functions_are_gone)
    run(test_switch_step_no_longer_opens_the_runs_card_but_evaluate_panel_does)
    run(test_promote_run_uses_the_renamed_continue_options_refresh)
    run(test_on_vega_finetune_from_switches_to_train_before_applying_the_path)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
