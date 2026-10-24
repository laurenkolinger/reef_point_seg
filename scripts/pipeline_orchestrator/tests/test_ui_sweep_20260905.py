"""
Task T13 of the 2026-09-04 VEGA fold-in: the three new cards wired into the
main window's Train, Evaluate and Inference panels, and the control fixes
the audit found.

House harness, no pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_ui_sweep_20260905.py

The page is rendered through Jinja with stand-in variables (the same names
app.py's index() passes) so this suite never imports app.py, which another
task may be editing. Two checks drive orchestrator.js inside Node with a fake
document: the box-dataset mode and the Evaluate panel's numeric readers.

Covers:
  CARDS     - the three include tags at their insertion points, the old inline
              report and gallery blocks gone, assets pinned with asset_version,
              the host glue for every card event and loader.
  TRAIN     - Stop Training beside Start Training, a warnings container per
              panel, box-dataset mode hides only the inert split controls and
              pins the task, preset notes name config/presets/, the loaded
              preset's values are what the tooltips describe.
  EVALUATE  - a blank image size is sent blank, a typed 0 survives, the
              Rounds table renders each row's own metric family and knows the
              incomparable gate, the Base column cannot overflow.
  INFERENCE - the save-crops checkbox is gone, Fill opacity replaces Mask
              opacity with a detector default, a device control exists.
  WINDOW    - the wordmark and title read VEGA, the Train + Infer link is gone.
  PROSE     - tooltip rules on every control this task added or changed; no
              dashes, emoji, "UVI" or the banned word anywhere in either file.
  REGISTRY  - every suite of this build is listed in run_all_tests.sh and
              exists on disk.
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

import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
_REPO = os.path.dirname(_SCRIPTS)
_TPL = os.path.join(_ORCH, "templates", "index.html")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_RUNS_JS = os.path.join(_ORCH, "static", "orchestrator_runs.js")
_RUN_ALL = os.path.join(_SCRIPTS, "run_all_tests.sh")

EM = "—"
EN = "–"
ASSET_STAMP = "sweep-20260905"

# Every suite this build added; each must be listed in run_all_tests.sh and exist.
BUILD_SUITES = [
    "pipeline_orchestrator/tests/test_step6_launch_guard.py",
    "pipeline_orchestrator/tests/test_step6_runs_card.py",
    "pipeline_orchestrator/tests/test_step7_report_card.py",
    "pipeline_orchestrator/tests/test_step8_gallery_card.py",
    "pipeline_orchestrator/tests/test_vega_runs.js",
    "pipeline_orchestrator/tests/test_vega_report.js",
    "pipeline_orchestrator/tests/test_vega_gallery.js",
    "pipeline_orchestrator/tests/test_step6_all_runs.py",
    "pipeline_orchestrator/tests/test_step7_report_by_run.py",
    "pipeline_orchestrator/tests/test_step8_runs.py",
    "pipeline_orchestrator/tests/test_ui_sweep_20260905.py",
    "TCRMPtrain_oceankindCV/tests/test_evaluate_metrics.py",
    "TCRMPtrain_oceankindCV/tests/test_inference_manifest.py",
    "TCRMPtrain_oceankindCV/tests/test_evaluate_pdf.py",
]

# Controls this task added or whose tooltip it rewrote: (marker, how to find
# the tag). Each tag must carry a title that follows the platform rules.
CHANGED_CONTROLS = [
    'id="s6-stop-btn"', 'id="s6-run-btn"', 'id="s6-epochs"', 'id="s6-imgsz"',
    'id="s6-batch"', 'id="s6-patience"', 'id="s6-hsv-h"', 'id="s6-pin-split"',
    'id="s6-preset-select"', 'id="s6-device-preset"', 'id="s7-preset-select"', 'id="s7-imgsz"',
    'id="s7-preview-count"', 'id="s7-copy-pdf-btn"', 'id="s8-mask-alpha"',
    'id="s8-draw-boxes"', 'id="s8-device-preset"', 'id="s8-device-custom"',
    'onclick="refreshStep6Presets()"', 'onclick="refreshStep7Presets()"',
    'onclick="applyStep7Defaults()"',
]

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


# ── cached sources ──────────────────────────────────────────────────
_CACHE = {}


def _read(path):
    if path not in _CACHE:
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _render():
    """Render templates/index.html through Jinja with stand-ins for every
    variable app.py's index() passes, plus a stub url_for for the expertids
    static files. The card and expertids partials come from their real folders."""
    if "html" not in _CACHE:
        from jinja2 import ChoiceLoader, Environment, FileSystemLoader
        env = Environment(loader=ChoiceLoader([
            FileSystemLoader(os.path.join(_ORCH, "templates")),
            FileSystemLoader(os.path.join(_SCRIPTS, "_expertids", "templates")),
        ]), autoescape=False)
        env.globals["url_for"] = lambda endpoint, **kw: f"/stub/{endpoint}/{kw.get('filename', '')}"
        tpl = env.get_template("index.html")
        _CACHE["html"] = tpl.render(
            asset_version=ASSET_STAMP,
            repo_root="/stub/repo",
            projects_dir="/stub/projects",
            supporting_data_dir="/stub/supporting_data",
            all_points_csv="/stub/all_points.csv",
            master_codes_csv="/stub/master_codes.csv",
            project_loaded=False,
            project_state={},
        )
    return _CACHE["html"]


def _panel(html, key):
    """The markup of one step panel, from its opening tag to the next panel's."""
    start = html.index(f'id="panel-{key}"')
    nxt = html.find('class="step-panel"', start + 1)
    return html[start:nxt if nxt > 0 else len(html)]


def _tag_with(html, marker):
    """The whole opening tag that carries the marker (an id= or onclick=)."""
    idx = html.index(marker)
    start = html.rfind("<", 0, idx)
    end = html.index(">", idx)
    return html[start:end + 1]


def _title_of(tag):
    m = re.search(r'title="([^"]*)"', tag)
    return m.group(1) if m else None


def _node():
    for cand in ("/usr/bin/node", "node"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


# Node driver: loads orchestrator.js into a vm context with a fake document
# whose getElementById mints an element per id, a fetch that never resolves
# (so the page's init IIFE parks harmlessly), then runs a scenario script in
# the same context and prints its JSON result.
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
    this.selectedIndex = -1; this.childElementCount = 0;
    const classes = new Set();
    this.classList = {
      add: (c) => classes.add(c), remove: (c) => classes.delete(c),
      toggle: (c, f) => { if (f === undefined) f = !classes.has(c); f ? classes.add(c) : classes.delete(c); return f; },
      contains: (c) => classes.has(c),
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] === undefined ? null : this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  appendChild(c) { this.children.push(c); this.childElementCount = this.children.length; return c; }
  removeChild(c) { this.children = this.children.filter((x) => x !== c); this.childElementCount = this.children.length; return c; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  addEventListener() {}
  removeEventListener() {}
  dispatchEvent() { return true; }
  closest() { return null; }
}
const els = new Map();
const document = {
  body: new El('body'), documentElement: new El('html'),
  getElementById(id) { if (!els.has(id)) els.set(id, new El(id)); return els.get(id); },
  querySelector() { return null; },
  querySelectorAll() { return []; },
  createElement() { return new El(''); },
  addEventListener() {},
};
document.body.dataset.projectLoaded = 'false';
const sandbox = {
  document, console, setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: () => new Promise(() => {}), alert() {}, prompt() { return null; },
  CustomEvent: class { constructor(t, o) { this.type = t; this.detail = o && o.detail; } },
  localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
  location: { href: 'http://127.0.0.1:5050/', search: '', hostname: '127.0.0.1', protocol: 'http:' },
  URLSearchParams, encodeURIComponent, decodeURIComponent,
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox, { filename: 'orchestrator.js' });
const out = vm.runInContext(scenario, sandbox, { filename: 'scenario.js' });
process.stdout.write(JSON.stringify(out));
"""


def _drive(scenario):
    """Run a scenario against orchestrator.js in Node; return its JSON result."""
    import json
    node = _node()
    check(node is not None, "node not available")
    tmp = tempfile.mkdtemp(prefix="ui_sweep_20260905_")
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


# ── cards ───────────────────────────────────────────────────────────
def test_three_card_includes_present_and_old_blocks_gone():
    tpl = _read(_TPL)
    html = _render()
    for inc in ("{% include '_step6_runs.html' %}", "{% include '_step7_report.html' %}",
                "{% include '_step8_gallery.html' %}"):
        check(tpl.count(inc) == 1, f"expected exactly one {inc}")
    p6, p7, p8 = _panel(html, 6), _panel(html, 7), _panel(html, 8)
    # 2026-09-21: the runs card moved off the Train panel, into a closed
    # "All trained models" disclosure on the Evaluate panel, after the Rounds
    # ledger. See the VEGA starting-model-and-training-data design.
    check('id="s6-runs-card"' not in p6, "runs card must no longer render inside the Train panel")
    check('id="s6-runs-card"' in p7, "runs card must render inside the Evaluate panel")
    check(p7.index('id="s7-rounds-block"') < p7.index('id="s7-all-models"') < p7.index('id="s6-runs-card"'),
          "runs card sits inside the All trained models disclosure, after the Rounds ledger")
    check('id="s7-report-card"' in p7, "report card must render inside the Evaluate panel")
    check(p7.index('id="s7-result"') < p7.index('id="s7-report-card"') < p7.index('id="s7-rounds-block"'),
          "report card sits after the result line and before the Rounds table")
    check('id="s8-gallery-card"' in p8, "gallery card must render inside the Inference panel")
    check(p8.index('id="s8-result"') < p8.index('id="s8-gallery-card"'),
          "gallery card sits after the result line")
    for relic in ("s7-report-block", 'id="s7-report"', "s7-pdf-download", "s8-gallery-block",
                  "s8-counts", "s8-blind-spots", 'id="s8-gallery"', "s8-save-crops",
                  "save per-mask crops"):
        check(relic not in html and relic not in tpl, f"relic {relic} survives")


def test_card_assets_are_pinned_with_asset_version():
    tpl = _read(_TPL)
    html = _render()
    for name in ("orchestrator_runs", "orchestrator_report", "orchestrator_gallery"):
        css = f'href="/static/{name}.css?v={{{{ asset_version }}}}"'
        js = f'src="/static/{name}.js?v={{{{ asset_version }}}}"'
        check(css in tpl, f"stylesheet {name}.css is not pinned with asset_version")
        check(js in tpl, f"script {name}.js is not pinned with asset_version")
        check(f'/static/{name}.css?v={ASSET_STAMP}' in html, f"{name}.css pin does not render")
        check(f'/static/{name}.js?v={ASSET_STAMP}' in html, f"{name}.js pin does not render")
        check(os.path.isfile(os.path.join(_ORCH, "static", f"{name}.css")), f"{name}.css missing on disk")
        check(os.path.isfile(os.path.join(_ORCH, "static", f"{name}.js")), f"{name}.js missing on disk")
    # The stylesheets sit in the head beside style.css; the scripts follow orchestrator.js.
    head = tpl[:tpl.index("</head>")]
    check(head.count("orchestrator_runs.css") == 1 and head.count("orchestrator_report.css") == 1
          and head.count("orchestrator_gallery.css") == 1, "card stylesheets belong in the head")
    main = tpl.index('src="/static/orchestrator.js?v=')
    for name in ("orchestrator_runs.js", "orchestrator_report.js", "orchestrator_gallery.js"):
        check(tpl.index(f"/static/{name}") > main, f"{name} must load after orchestrator.js")


def test_card_events_and_loaders_are_wired():
    js = _read(_JS)
    for needle in ("'vega:evaluate-run'", "'vega:promote-run'", "'vega:finetune-from'", "'vega:verdict'",
                   "/api/step/8/verify", "loadStep7ReportCard(", "VegaGallery.load(", "initStep6Runs()",
                   "/labels/api/vocabulary", "labelNames"):
        check(needle in js, f"host glue missing {needle}")
    # api() carries the HTTP status so the report card can read a 404 as "not evaluated yet".
    api_fn = js[js.index("async function api("):js.index("\nfunction post(")]
    check("data._status = resp.status" in api_fn, "api() must attach the HTTP status as _status")
    # 2026-09-21: the Trained runs card moved off the Train panel into the
    # closed "All trained models" disclosure on the Evaluate panel (see the
    # VEGA starting-model-and-training-data design). switchStep(6) no longer
    # opens it; openStep7Panel() loads it only when the disclosure is already
    # open, and a toggle listener bound once loads it on every later open.
    sw = js[js.index("function switchStep("):js.index("function getNavOrder(")]
    check("openStep6RunsCard()" not in sw,
          "switchStep(6) must no longer initialise the runs card; it lives on the Evaluate panel now")
    check("openStep7Panel()" in sw, "switchStep(7) must open the Evaluate panel through its loader")
    check("refreshStep8GalleryCard()" in sw, "switchStep(8) must load the gallery card")

    panel7 = js[js.index("async function openStep7Panel("):js.index("\n}\n", js.index("async function openStep7Panel("))]
    check("wireStep7AllModelsToggle()" in panel7,
          "openStep7Panel must bind the All trained models toggle listener")
    check("step7AllModelsOpen()" in panel7 and "openStep6RunsCard()" in panel7,
          "openStep7Panel must load the runs card when All trained models is already open")

    toggle = js[js.index("function wireStep7AllModelsToggle("):
                js.index("\n}\n", js.index("function wireStep7AllModelsToggle("))]
    check("addEventListener('toggle'" in toggle,
          "the All trained models disclosure needs a toggle listener")
    check("openStep6RunsCard()" in toggle,
          "the toggle listener must load the runs card the moment the disclosure opens")

    promote = js[js.index("async function promoteRun("):js.index("\n}\n", js.index("async function promoteRun("))]
    check("openStep6RunsCard()" in promote, "a successful promote must reload the runs card")
    # Finished evaluation and inference runs reload their cards from the poller.
    poll = js[js.index("function startPolling("):js.index("function showResult(")]
    check("loadStep7ReportForSelectedRun()" in poll, "a finished evaluation must reload the report card")
    # The finished inference run is the one the gallery shows, so the poller
    # hands its output folder in rather than reloading whatever was newest.
    check("refreshStep8GalleryCard(newestStep8OutDir())" in poll,
          "a finished inference run must reload the gallery card on that run's output folder")
    # Copy PDF is a host control sending the selected run.
    copy_fn = js[js.index("async function copyStep7Pdf("):js.index("\n}\n", js.index("async function copyStep7Pdf("))]
    check("/api/step/7/copy_pdf" in copy_fn and "run_dir" in copy_fn, "Copy PDF must post run_dir to copy_pdf")
    # Continuing from a run picked on the card sets that run's best.pt as the
    # weights to continue from (under Continue from a VEGA model) and follows
    # the run's task, switching to the Train panel where the pick is shown.
    ft = js[js.index("function onVegaFinetuneFrom("):js.index("\n}\n", js.index("function onVegaFinetuneFrom("))]
    check("WEIGHTS_REL" in ft and "applyStep6ModelPath(" in ft and "s6-task" in ft,
          "continuing from a run must set its best.pt as the model to continue from and align the task")

    # The card's own champion tooltip carries the retired word's replacement:
    # "Continue from a VEGA model". orchestrator_runs.js is Task C2's file
    # (its own copy sweep is a separate task from this one), so only the one
    # string the plan's contract table gives verbatim is pinned here; the
    # property name "finetune" survives by design (TEXT.finetune's button
    # label is "Continue from", not its JS key).
    runs_js = _read(_RUNS_JS)
    check(
        'champion: "The promoted run, called the champion: the model Model Inference uses unless another '
        'is picked, and the first choice under Continue from a VEGA model. Pressing Promote on a different '
        'run makes that one the champion instead."' in runs_js,
        "TIPS.champion must carry the Continue from a VEGA model wording")
    check('finetune: "Continue from"' in runs_js, "TEXT.finetune's visible label must read Continue from")


def test_old_js_functions_are_gone():
    js = _read(_JS)
    tpl = _read(_TPL)
    for fn in ("function mdToHtml(", "function loadStep7Report(", "function refreshStep8Gallery(",
               "function verifyStep8("):
        check(fn not in js, f"{fn} still defined")
    for call in ("loadStep7Report()", "refreshStep8Gallery()", "verifyStep8("):
        check(call not in js and call not in tpl, f"call site {call} survives")


# ── train panel ─────────────────────────────────────────────────────
def test_train_panel_has_a_stop_button_and_warnings_container():
    html = _render()
    p6 = _panel(html, 6)
    check('id="s6-stop-btn"' in p6, "Train panel needs a Stop Training button")
    stop = _tag_with(p6, 'id="s6-stop-btn"')
    check('onclick="stopStep(6)"' in stop, "Stop Training must post /api/step/6/stop through stopStep(6)")
    check("display:none" in stop, "Stop Training is hidden until a run is live")
    check(p6.index('id="s6-run-btn"') < p6.index('id="s6-stop-btn"'), "Stop sits beside Start Training")
    check('id="s6-warnings"' in p6, "Train panel needs a warnings container")
    js = _read(_JS)
    check("function setStepRunControls(" in js, "orchestrator.js must define setStepRunControls")
    run_fn = js[js.index("async function runStep("):js.index("async function linkExistingStep1(")]
    check("setStepRunControls(step, true)" in run_fn, "runStep must disable Start and show Stop while live")
    poll = js[js.index("function startPolling("):js.index("function showResult(")]
    check("setStepRunControls(step, false)" in poll, "the poller must re-enable Start when the run ends")
    stop_fn = js[js.index("async function stopStep("):js.index("async function markDone(")]
    check("setStepRunControls(step, false)" in stop_fn, "stopStep must re-enable Start")
    ctl = js[js.index("function setStepRunControls("):js.index("\n}\n", js.index("function setStepRunControls("))]
    check("-run-btn" in ctl and "-stop-btn" in ctl and ".disabled = " in ctl,
          "setStepRunControls must toggle the run button's disabled state and the stop button")


def test_every_step_panel_has_a_warnings_container():
    html = _render()
    keys = re.findall(r'class="step-panel" id="panel-(\d+)"', html)
    check(keys == ["1", "2", "3", "6", "7", "8"], f"unexpected numeric panels: {keys}")
    for k in keys:
        p = _panel(html, k)
        check(f'id="s{k}-warnings"' in p, f"panel {k} has no warnings container")
        check(p.index(f'id="s{k}-result"') < p.index(f'id="s{k}-warnings"'),
              f"panel {k}: warnings sit after the result line")
    js = _read(_JS)
    check("function renderStepWarnings(" in js, "orchestrator.js must define renderStepWarnings")
    run_fn = js[js.index("async function runStep("):js.index("async function linkExistingStep1(")]
    check("renderStepWarnings(step, data.warnings)" in run_fn, "runStep must render the route's warnings")
    body = js[js.index("function renderStepWarnings("):js.index("\n}\n", js.index("function renderStepWarnings("))]
    check("-warnings" in body and "escapeHtml(" in body, "warnings are rendered escaped into s<step>-warnings")


def test_box_dataset_mode_hides_only_the_inert_split_controls():
    html = _render()
    p6 = _panel(html, 6)
    for gid in ("s6-valid-group", "s6-test-group", "s6-min-samples-group", "s6-pin-split-group"):
        check(f'id="{gid}"' in p6, f"Train panel needs the {gid} wrapper")
    js = _read(_JS)
    mode = js[js.index("function applyStep6BoxDatasetMode("):]
    mode = mode[:mode.index("\n}\n") + 3]
    for hid in ("s6-min-samples-group", "s6-pin-split-group"):
        check(hid in mode, f"box-dataset mode must hide {hid}")
    for keep in ("s6-valid-group", "s6-test-group"):
        check(keep not in mode, f"box-dataset mode must leave {keep} visible (the fractions reach the splitter)")
    # The pin itself lives in pinStep6Task(), which also restores the free
    # tooltip; the driven scenario below proves the value and the disabled state.
    check("pinStep6Task(" in mode, "box-dataset mode must pin the task to detect")
    result = _drive(r"""
state = { is_training_project: true, project_type: 'bbox_training',
          training_source_label: 'box dataset', training_source_name: 'coral_intel <b>x</b>',
          steps: { 6: { config: {} } }, composite: null };
applyCompositeTrainingMode();
const g = (id) => document.getElementById(id);
({ src: g('s6-source-card').style.display, cls: g('s6-classes-card').style.display,
   min: g('s6-min-samples-group').style.display, pin: g('s6-pin-split-group').style.display,
   valid: g('s6-valid-group').style.display || 'visible', test: g('s6-test-group').style.display || 'visible',
   task: g('s6-task').value, taskDisabled: g('s6-task').disabled,
   banner: g('s6-composite-banner').style.display || 'unset',
   note: g('s6-bbox-source-note').style.display, noteHtml: g('s6-bbox-source-note').innerHTML });
""")
    check(result["src"] == "none" and result["cls"] == "none", f"source and class cards hidden: {result}")
    check(result["min"] == "none" and result["pin"] == "none", f"min-samples and pinned-split hidden: {result}")
    check(result["valid"] == "visible" and result["test"] == "visible", f"fractions stay visible: {result}")
    check(result["task"] == "detect" and result["taskDisabled"] is True, f"task pinned to detect: {result}")
    check(result["banner"] == "unset", f"composite banner must not show a second source line: {result}")
    check(result["note"] == "block" and "&lt;b&gt;" in result["noteHtml"], f"dataset line shown, escaped: {result}")
    # An ordinary project is untouched by the mode.
    plain = _drive(r"""
state = { is_training_project: false, project_type: 'annotation', steps: { 6: { config: {} } } };
applyCompositeTrainingMode();
const g = (id) => document.getElementById(id);
({ min: g('s6-min-samples-group').style.display || 'visible', task: g('s6-task').disabled });
""")
    check(plain["min"] == "visible" and plain["task"] is False, f"ordinary project untouched: {plain}")


# The longest a training-source option's text may be. The 59-class staged box
# dataset wrote 984 characters into one option on 2026-09-05 and stretched the
# select to 7225 px inside a 946 px window; every class name now rides in the
# option's tooltip instead.
MAX_SOURCE_OPTION_CHARS = 160


def test_training_source_option_text_is_bounded_and_the_tooltip_carries_the_classes():
    """M2 (2026-09-05): a training-source option names at most the first few
    classes and counts the rest, and its tooltip carries every one."""
    js = _read(_JS)
    check("MAX" in js or "SHOWN" in js, "the truncation limit must be a named constant")
    result = _drive(r"""
const labels = [];
for (let i = 0; i < 59; i += 1) labels.push('Species_name_' + String(i).padStart(3, '0'));
const source = { name: 'coral_intel_smoke20', kind: 'bbox', n_images: 1234, n_masks: null,
                 labels: labels, path: '/x/supporting_data/bbox_datasets/coral_intel_smoke20',
                 created_at: '2026-09-02 01:03 AST' };
const short = { name: 'qs20260902_pa_ss', kind: 'composite', n_images: 40, n_masks: 120,
                labels: ['PA', 'SS'], path: '/x/inprocess/_composite/qs20260902_pa_ss',
                created_at: '2026-09-02 16:25 AST' };
({ text: step6SourceOptionText(source), title: step6SourceOptionTitle(source),
   shortText: step6SourceOptionText(short), shortTitle: step6SourceOptionTitle(short),
   missing: step6SourceOptionText({ name: 'n', kind: 'composite', n_images: 1, n_masks: 2 }) });
""")
    check(len(result["text"]) <= MAX_SOURCE_OPTION_CHARS,
          f"option text is {len(result['text'])} characters, over {MAX_SOURCE_OPTION_CHARS}: {result['text']}")
    check("and 56 more" in result["text"],
          f"the option must count the classes it does not name: {result['text']}")
    check(result["text"].count("Species_name_") == 3,
          f"the option names the first three classes: {result['text']}")
    check(result["title"].count("Species_name_") == 59,
          "the option tooltip must carry every class name")
    check("coral_intel_smoke20" in result["title"] and "2026-09-02 01:03 AST" in result["title"],
          f"the option tooltip keeps the path and the build stamp: {result['title']}")
    check(result["shortText"].endswith("(PA, SS)") and "more" not in result["shortText"],
          f"a two-class source is named in full with no counter: {result['shortText']}")
    check("(" not in result["missing"], f"a source with no class list gets no empty brackets: {result['missing']}")


def test_box_dataset_mode_says_why_the_task_is_pinned():
    """S5 (2026-09-05): the line under the pinned Training task select must
    say why it cannot be changed, not describe a change nobody can make."""
    html = _render()
    p6 = _panel(html, 6)
    check('id="s6-task-note"' in p6, "the line under the Training task select needs an id")
    result = _drive(r"""
const note = document.getElementById('s6-task-note');
note.textContent = 'Changing this swaps the base model list below to the weights that match.';
const free = note.textContent;
pinStep6Task(true);
const pinned = note.textContent;
pinStep6Task(false);
({ free: free, pinned: pinned, restored: note.textContent,
   disabled: document.getElementById('s6-task').disabled });
""")
    check(result["pinned"] == "A box dataset holds boxes and no outlines, so only a detector can train on it.",
          f"the pinned line must be the pin sentence: {result['pinned']!r}")
    check("Changing this" not in result["pinned"],
          f"the pinned line must not describe a change nobody can make: {result['pinned']!r}")
    check(result["restored"] == result["free"],
          f"unpinning restores the free line: {result['restored']!r}")
    check(result["disabled"] is False, f"unpinning re-enables the select: {result}")


def test_continue_from_options_are_filtered_by_the_training_task():
    """2026-09-21: the runs a project could start from used to live in a
    "Continue from a previous run" optgroup folded into the starting-model
    select itself (#s6-model-runs-optgroup), and the launch preflight judges
    only the weights file name, which is best.pt for every run. That optgroup
    is gone: the runs now live in their own "Continue from a VEGA model"
    select (#s6-continue-from, spec section 3.2), filtered by
    applyStep6ContinueTaskFilter(), which hides and disables an option of the
    other task, writes the visible count into the disclosure summary once the
    select has loaded for real (dataset.loaded, so a page that has not
    fetched yet never flashes "no runs to choose from"), and clears a pick
    the filter just hid rather than leaving it silently selected but inert."""
    js = _read(_JS)
    check("s6-model-runs-optgroup" not in js,
          "orchestrator.js must no longer mention the retired runs-optgroup id")
    check("function applyStep6ContinueTaskFilter(" in js, "applyStep6ContinueTaskFilter() missing")

    scenario_helpers = r"""
function opt(task, value) { return { dataset: { task: task }, value: value, hidden: false, disabled: false }; }
function group(children) { return { children: children }; }
"""

    detect = _drive(scenario_helpers + r"""
const sel = document.getElementById('s6-continue-from');
const segOpt = opt('seg', '/p/step6_trainModel/runs/seg_run/weights/best.pt');
const detOpt1 = opt('detect', '/p/step6_trainModel/runs/det_run_a/weights/best.pt');
const detOpt2 = opt('detect', '/p/step6_trainModel/runs/det_run_b/weights/best.pt');
sel.children = [group([segOpt]), group([detOpt1, detOpt2])];
sel.value = segOpt.value;
sel.dataset.loaded = '1';
document.getElementById('s6-task').value = 'detect';
applyStep6ContinueTaskFilter();
({ segHidden: segOpt.hidden, segDisabled: segOpt.disabled,
   det1Hidden: detOpt1.hidden, det1Disabled: detOpt1.disabled,
   det2Hidden: detOpt2.hidden, det2Disabled: detOpt2.disabled,
   value: sel.value,
   summary: document.getElementById('s6-continue-summary').textContent });
""")
    check(detect["segHidden"] is True and detect["segDisabled"] is True,
          f"a segmentation run must be hidden and unpickable on a detector run: {detect}")
    check(detect["det1Hidden"] is False and detect["det1Disabled"] is False
          and detect["det2Hidden"] is False and detect["det2Disabled"] is False,
          f"both detector runs stay offered: {detect}")
    check(detect["value"] == "",
          f"a selection the filter just hid must be cleared, not left silently selected: {detect}")
    check(detect["summary"] == "Continue from a VEGA model (advanced): 2 runs to choose from",
          f"the disclosure summary must count the runs the filter still shows: {detect}")

    segment = _drive(scenario_helpers + r"""
const sel = document.getElementById('s6-continue-from');
const detOpt = opt('detect', '/p/step6_trainModel/runs/det_run/weights/best.pt');
const segOpt = opt('seg', '/p/step6_trainModel/runs/seg_run/weights/best.pt');
sel.children = [group([detOpt]), group([segOpt])];
sel.value = segOpt.value;
sel.dataset.loaded = '1';
document.getElementById('s6-task').value = 'segment';
applyStep6ContinueTaskFilter();
({ detHidden: detOpt.hidden, segHidden: segOpt.hidden, value: sel.value,
   summary: document.getElementById('s6-continue-summary').textContent });
""")
    check(segment["detHidden"] is True, f"a detector run is hidden on a segmentation run: {segment}")
    check(segment["segHidden"] is False, f"the segmentation run stays offered: {segment}")
    check(segment["value"] != "",
          f"a selection the filter still shows must survive, not be cleared: {segment}")
    check(segment["summary"] == "Continue from a VEGA model (advanced): 1 run to choose from",
          f"the summary reads the singular for one run: {segment}")

    unloaded = _drive(scenario_helpers + r"""
const sel = document.getElementById('s6-continue-from');
const segOpt = opt('seg', '/p/step6_trainModel/runs/seg_run/weights/best.pt');
sel.children = [group([segOpt])];
document.getElementById('s6-continue-summary').textContent = 'Continue from a VEGA model (advanced)';
document.getElementById('s6-task').value = 'detect';
applyStep6ContinueTaskFilter();
({ summary: document.getElementById('s6-continue-summary').textContent });
""")
    check(unloaded["summary"] == "Continue from a VEGA model (advanced)",
          f"before the select has loaded for real, the summary is left as the template wrote it, "
          f"never rewritten to a misleading zero count: {unloaded}")


def test_preset_notes_name_the_real_folders():
    tpl = _read(_TPL)
    for ghost in ("pipeline_orchestrator/presets/", "pipeline_orchestrator/eval_presets/"):
        check(ghost not in tpl, f"{ghost} does not exist and must not be named")
    check("config/presets/" in tpl, "the Train preset note names config/presets/")
    check("config/eval_presets/" in tpl, "the Evaluate preset note names config/eval_presets/")
    check(os.path.isdir(os.path.join(_REPO, "config", "presets")), "config/presets/ missing on disk")
    check(os.path.isdir(os.path.join(_REPO, "config", "eval_presets")), "config/eval_presets/ missing on disk")


def test_loaded_preset_note_is_persistent_and_tooltips_defer_to_it():
    html = _render()
    p6 = _panel(html, 6)
    check('id="s6-preset-applied"' in p6, "a persistent line names the loaded preset")
    js = _read(_JS)
    imp = js[js.index("async function importStep6Preset("):js.index("\n}\n", js.index("async function importStep6Preset("))]
    check("setStep6PresetAppliedNote(" in imp, "importStep6Preset writes the loaded-preset line")
    check("setTimeout" not in imp, "the loaded-preset note must not flash and fade")
    note_fn = js[js.index("function setStep6PresetAppliedNote("):js.index("\n}\n", js.index("function setStep6PresetAppliedNote("))]
    check("s6-preset-applied" in note_fn, "the loaded-preset line writes into s6-preset-applied")
    check("setTimeout" not in note_fn, "the loaded-preset note must not flash and fade")
    for fid in ("s6-epochs", "s6-imgsz", "s6-batch", "s6-patience", "s6-hsv-h", "s6-device-preset"):
        tip = _title_of(_tag_with(p6, f'id="{fid}"')) or ""
        check("preset" in tip.lower(), f"{fid} tooltip must say the loaded preset sets the value")
        check("Default:" not in tip, f"{fid} tooltip must not advertise a fixed default the preset overrides")
    # The device options are labels, not tooltips, and one of them used to
    # call the dual-GPU choice the default while the preset picks cuda:0.
    dev = p6[p6.index('id="s6-device-preset"'):p6.index("</select>", p6.index('id="s6-device-preset"'))]
    check("(default)" not in dev, "no device option may call itself the default the preset overrides")
    pin = _title_of(_tag_with(p6, 'id="s6-pin-split"')) or ""
    check("seed" in pin and " 1 " in pin, "pinned-split tooltip must say the seed is fixed at 1")
    m = re.search(r'<input type="checkbox" id="s6-pin-split"([^>]*)>', p6)
    check(m is not None and "checked" in m.group(1), "pinned split stays ticked by default")


def test_train_progress_reads_the_epoch_column():
    js = _read(_JS)
    body = js[js.index("function parseProgressFromLog("):js.index("async function runStep(")]
    check("Epoch " in body and "step === 6" in body, "the Train step's progress must read the epoch column")


# ── evaluate panel ──────────────────────────────────────────────────
def test_evaluate_imgsz_is_not_coerced_client_side():
    html = _render()
    p7 = _panel(html, 7)
    tag = _tag_with(p7, 'id="s7-imgsz"')
    check('value="' not in tag or 'value=""' in tag, f"the image size field starts blank: {tag}")
    check('id="s7-imgsz-source"' in p7, "a line beside the field says where the value came from")
    js = _read(_JS)
    seven = js[js.index("} else if (step === 7) {"):js.index("} else if (step === 8) {")]
    check("|| 512" not in seven and "|| 1024" not in seven, "no client-side image size coercion")
    check("readOptionalNumberField('s7-imgsz')" in seven, "image size goes through the optional reader")
    for fid in ("s7-conf", "s7-iou", "s7-preview-count"):
        check(f"readNumberField('{fid}'" in seven, f"{fid} goes through the finite-number reader")
    result = _drive(r"""
const g = (id) => document.getElementById(id);
g('s7-run-dir').value = '/x/runs/r1'; g('s7-split').value = 'test';
g('s7-imgsz').value = ''; g('s7-conf').value = '0'; g('s7-iou').value = '';
g('s7-preview-count').value = '0'; g('s7-pdf-export-dir').value = '';
const c = collectConfig(7);
g('s7-imgsz').value = ' 640 ';
const c2 = collectConfig(7);
({ imgsz: c.imgsz, conf: c.conf_threshold, iou: c.iou_threshold, iouEcho: g('s7-iou').value,
   preview: c.preview_count, typed: c2.imgsz });
""")
    check(result["imgsz"] == "", f"blank image size must be sent blank: {result}")
    check(result["conf"] == 0, f"a typed 0 confidence must survive: {result}")
    check(result["preview"] == 0, f"a typed 0 preview count must survive: {result}")
    check(result["iou"] == 0.6 and result["iouEcho"] == "0.6", f"a blank IoU takes the default and is echoed: {result}")
    check(result["typed"] == 640, f"a typed size is sent as a number: {result}")
    # The run-change handler prefills from the run and loads the report card.
    change = js[js.index("async function onStep7RunChange("):js.index("\n}\n", js.index("async function onStep7RunChange("))]
    check("prefillStep7ImageSize(" in change and "loadStep7ReportForSelectedRun(" in change,
          "run change must prefill the image size and reload the report card")
    pre = js[js.index("async function prefillStep7ImageSize("):js.index("\n}\n", js.index("async function prefillStep7ImageSize("))]
    check("/api/step/7/run_info" in pre and "s7-imgsz-source" in pre and "args.yaml" in pre,
          "the prefill reads run_info and names args.yaml as the source")
    check("=== 1024" not in js and "=== 512" not in js, "the old 1024-or-512 overwrite guard is gone")


def test_rounds_table_renders_each_rows_own_family():
    html = _render()
    p7 = _panel(html, 7)
    check("mAP50-95(M)" not in p7 and "Recall(M)" not in p7, "Rounds headers are no longer mask-only")
    # 2026-09-17 (LO): Lauren dropped comparability language from the screen; the Gates header
    # says in plain words that a round scored on other images or settings is not compared.
    check("is not compared" in p7 and "incomparable" not in p7,
          "the Rounds gates are explained in plain words, never as the ledger token")
    js = _read(_JS)
    # From the family-column map (which names the box columns) through the
    # renderer, so both the column choice and the cells it writes are in scope.
    rounds = js[js.index("const ROUNDS_FAMILY_COLUMNS"):js.index("function confirmPromoteRun(")]
    for needle in ("map50_95_B", "recall_B", "roundFamily(", "overflow-wrap:anywhere", "title="):  # 2026-09-17: whole tokens that wrap, never an ellipsis
        check(needle in rounds, f"Rounds table must carry {needle}")
    tips = js[js.index("const _ROUNDS_GATE_TOOLTIPS"):js.index("function _roundsGateBadge(")]
    check("incomparable:" in tips, "the incomparable gate needs its own tooltip")


# ── inference panel ─────────────────────────────────────────────────
def test_inference_panel_has_device_control_and_fill_opacity():
    html = _render()
    p8 = _panel(html, 8)
    check('id="s8-device-preset"' in p8 and 'id="s8-device-custom"' in p8, "Inference panel needs a device control")
    check('onchange="onStepDeviceChange(8)"' in p8, "the device preset drives the custom field")
    check("Fill opacity" in p8 and "Mask opacity" not in p8, "the opacity control is relabelled Fill opacity")
    fill_tip = _title_of(_tag_with(p8, 'id="s8-mask-alpha"')) or ""
    check("box" in fill_tip.lower(), "the fill opacity tooltip says it fills boxes too")
    box_tip = _title_of(_tag_with(p8, 'id="s8-draw-boxes"')) or ""
    check("detector" in box_tip.lower(), "the draw-boxes tooltip says it adds nothing for a detector")
    js = _read(_JS)
    eight = js[js.index("} else if (step === 8) {"):js.index("\n    return cfg;")]
    check("save_crops" not in eight and "save_crops" not in js, "save_crops plumbing is gone")
    check("collectStepDevice(8)" in eight, "collectConfig(8) must send the device")
    check("FILL_ALPHA_DEFAULT_DETECT = 0.15" in js and "FILL_ALPHA_DEFAULT_SEGMENT = 0.45" in js,
          "fill opacity defaults are named constants")
    result = _drive(r"""
const g = (id) => document.getElementById(id);
g('s8-run-dir').value = '/x/runs/r1'; g('s8-run-name').value = ''; g('s8-imgsz').value = '';
g('s8-source-dir').value = ''; g('s8-sample-count').value = '';
g('s8-sample-mode').value = 'random'; g('s8-conf').value = '0'; g('s8-iou').value = '';
g('s8-mask-alpha').value = ''; g('s8-device-preset').value = '__custom__'; g('s8-device-custom').value = ' cuda:1 ';
state = { is_training_project: true, project_type: 'bbox_training', training_source_label: 'box dataset',
          training_source: { task_default: 'detect' }, steps: { 6: { config: {} } } };
const c = collectConfig(8);
state = { is_training_project: false, steps: { 6: { config: { task: 'segment' } } } };
const seg = step8FillAlphaDefault();
({ device: c.device, crops: Object.prototype.hasOwnProperty.call(c, 'save_crops'), conf: c.conf_threshold,
   alpha: c.mask_alpha, seg: seg, imgsz: c.imgsz, sample: c.sample_count });
""")
    check(result["device"] == "cuda:1", f"custom device is trimmed and sent: {result}")
    check(result["crops"] is False, f"no save_crops key: {result}")
    check(result["conf"] == 0, f"a typed 0 confidence survives: {result}")
    check(result["alpha"] == 0.15, f"a detector project defaults fill opacity to 0.15: {result}")
    check(result["seg"] == 0.45, f"a segmentation project keeps 0.45: {result}")
    check(result["imgsz"] == 512 and result["sample"] == 100, f"blank inference numbers take their defaults: {result}")


# ── whole window ────────────────────────────────────────────────────
def test_wordmark_reads_vega_and_title_too():
    html = _render()
    check("<title>VEGA</title>" in html, "the window title must read VEGA")
    check(html.count("<h1>VEGA</h1>") == 2, "both wordmarks (topbar and bounce screen) read VEGA")
    check("REEF POINT SEG" not in html, "the old wordmark survives")


def test_no_models_link_remains():
    html = _render()
    check('href="/models"' not in html, "the Train + Infer link must go")
    tools = html[html.index('class="nav-tool"'):html.index('<!-- Main Content -->')]
    check("Train + Infer" not in tools, "the Train + Infer label must go from the tools cluster")
    check('href="/boxes"' in tools, "the Bounding Boxes link stays")
    check("&#9881;" not in tools, "the Train + Infer glyph goes with its link")
    # The runs card's section tooltip still names the page, to say where a lab
    # run came from. Every surviving mention must read as history, not a label.
    for m in re.finditer(r"Train \+ Infer", html):
        before = html[max(0, m.start() - 40):m.start()]
        check("retired" in before, f"an unqualified Train + Infer mention survives: {before}")


# ── prose ───────────────────────────────────────────────────────────
def _tooltip_violations(tip):
    problems = []
    t = tip.strip()
    if not t:
        return ["empty"]
    if t.startswith("You "):
        problems.append("opens with You")
    if not t.endswith("."):
        problems.append("does not end a sentence")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s]
    if not 1 <= len(sentences) <= 2:
        problems.append(f"{len(sentences)} sentences")
    for ref in ("above", "below", "this tab", "this panel", "this window"):
        if re.search(rf"\b{ref}\b", t):
            problems.append(f"screen reference '{ref}'")
    if EM in t or EN in t:
        problems.append("dash")
    return problems


def test_every_new_or_changed_control_has_a_rule_compliant_tooltip():
    html = _render()
    bad = []
    for marker in CHANGED_CONTROLS:
        tag = _tag_with(html, marker)
        tip = _title_of(tag)
        if tip is None:
            bad.append(f"{marker}: no title")
            continue
        problems = _tooltip_violations(tip)
        if problems:
            bad.append(f"{marker}: {', '.join(problems)} :: {tip}")
    # Tooltips the host glue writes in code. The Stop button is not here: its
    # text is fixed markup, checked above through id="s6-stop-btn", and a
    # second copy in JavaScript would be the same sentence written twice.
    js = _read(_JS)
    for name in ("BOX_DATASET_TASK_TOOLTIP", "RUN_WARNING_TOOLTIP"):
        m = re.search(rf"const {name} =\s*'((?:[^'\\]|\\.)*)'", js)
        if not m:
            bad.append(f"{name}: constant missing in orchestrator.js")
            continue
        problems = _tooltip_violations(m.group(1).replace("\\'", "'"))
        if problems:
            bad.append(f"{name}: {', '.join(problems)}")
    tips = js[js.index("const _ROUNDS_GATE_TOOLTIPS"):js.index("function _roundsGateBadge(")]
    m = re.search(r"incomparable:\s*'((?:[^'\\]|\\.)*)'", tips)
    if not m:
        bad.append("incomparable gate tooltip missing")
    else:
        problems = _tooltip_violations(m.group(1).replace("\\'", "'"))
        if problems:
            bad.append(f"incomparable: {', '.join(problems)}")
    check(not bad, "tooltip rule violations:\n  " + "\n  ".join(bad))


def test_no_banned_prose_in_touched_strings():
    for path in (_TPL, _JS):
        text = _read(path)
        base = os.path.basename(path)
        check(EM not in text and EN not in text, f"{base} carries an em or en dash")
        check("UVI" not in text, f"{base} carries the string UVI")
        check(not re.search(r"[\U0001F000-\U0001FFFF]", text), f"{base} carries an emoji")
        m = re.search(r"\bland(s|ed|ing)?\b", text, re.I)
        check(m is None, f"{base} carries the banned word at offset {m.start() if m else -1}")


# ── registry ────────────────────────────────────────────────────────
def test_registered_suites_exist():
    sh = _read(_RUN_ALL)
    suites = sh[sh.index("SUITES=("):sh.index("\n)\n", sh.index("SUITES=("))]
    for rel in BUILD_SUITES:
        kind = "js" if rel.endswith(".js") else "py"
        check(f'"{kind} {rel}"' in suites, f"run_all_tests.sh does not list {rel}")
        check(os.path.isfile(os.path.join(_SCRIPTS, rel)), f"listed suite missing on disk: {rel}")


# ── smoke ───────────────────────────────────────────────────────────
def test_smoke_render():
    html = _render()
    check(len(html) > 100000, "rendered page is unexpectedly small")
    check('id="panel-8"' in html, "rendered page lacks the Inference panel")


def test_smoke_node_check():
    node = _node()
    check(node is not None, "node not available")
    r = subprocess.run([node, "--check", _JS], capture_output=True, text=True)
    check(r.returncode == 0, f"orchestrator.js node --check failed:\n{r.stderr}")


if __name__ == "__main__":
    print("T13 VEGA panel wiring sweep (2026-09-05)\n")
    print("SMOKE:")
    run(test_smoke_render)
    run(test_smoke_node_check)
    print("CARDS:")
    run(test_three_card_includes_present_and_old_blocks_gone)
    run(test_card_assets_are_pinned_with_asset_version)
    run(test_card_events_and_loaders_are_wired)
    run(test_old_js_functions_are_gone)
    print("TRAIN:")
    run(test_train_panel_has_a_stop_button_and_warnings_container)
    run(test_every_step_panel_has_a_warnings_container)
    run(test_box_dataset_mode_hides_only_the_inert_split_controls)
    run(test_continue_from_options_are_filtered_by_the_training_task)
    run(test_training_source_option_text_is_bounded_and_the_tooltip_carries_the_classes)
    run(test_box_dataset_mode_says_why_the_task_is_pinned)
    run(test_preset_notes_name_the_real_folders)
    run(test_loaded_preset_note_is_persistent_and_tooltips_defer_to_it)
    run(test_train_progress_reads_the_epoch_column)
    print("EVALUATE:")
    run(test_evaluate_imgsz_is_not_coerced_client_side)
    run(test_rounds_table_renders_each_rows_own_family)
    print("INFERENCE:")
    run(test_inference_panel_has_device_control_and_fill_opacity)
    print("WINDOW:")
    run(test_wordmark_reads_vega_and_title_too)
    run(test_no_models_link_remains)
    print("PROSE:")
    run(test_every_new_or_changed_control_has_a_rule_compliant_tooltip)
    run(test_no_banned_prose_in_touched_strings)
    print("REGISTRY:")
    run(test_registered_suites_exist)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
