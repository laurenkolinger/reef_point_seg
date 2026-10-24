"""Task 3: the training recipes, and how the Train panel chooses one.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_preset_default.py

A recipe is a preset YAML in config/presets/. Which one a project trains from
is decided by two explicit controls on the Train panel, Training task and
Subject, matched against what each recipe declares it is for. Nothing is
decided by a file name, a file time or a folder order. This suite pins that:

  PRESET    config/presets/fish_heavy_augment_640.yaml carries run3's recipe
            from the October 2025 fish ablation verbatim, declares task detect
            and subject fish, and names no parameter the Train form cannot
            apply (a key with no applier in orchestrator.js is dropped in
            silence at import time, which is the failure this catches).
  APPLIERS  every applier in STEP6_PRESET_APPLIERS resolves: the helper it
            calls is declared in orchestrator.js and the form field it writes
            exists in templates/index.html. An applier naming a helper that a
            later refactor renamed throws at import time, and the auto-apply
            path swallows the throw, so the rest of the recipe never reaches
            the form and nothing on the page says so.
  ROUTE     GET /api/step/6/presets reports what each file declares (task,
            subject, classified, warning) and picks nothing: no winner in the
            payload, and no per-entry flag the old file-name rule resolved.
            It also reports dataset_subject, the subject the open project's
            staged box dataset recorded for itself.
  SELECTION the panel's own rule, driven in Node against the live
            orchestrator.js and the real payload: exactly one match is
            offered and applied, more than one applies NOTHING and names the
            candidates, none applies nothing and says so, an unclassified
            recipe is listed and never matched, and a recipe whose file name
            sorts first wins nothing by it.
  ADVERSARIAL  an unparsable preset file, a file that parses to a list, a
            file whose _meta or params section is a scalar, a task and a
            subject in mixed case with stray spaces, non-string values, a
            subject outside the vocabulary, and two recipes claiming one pair.
  CORAL     config/presets/coral_heavy_augment_640.yaml carries the recipe
            Lauren approved for the Coral Intel subset on 2026-09-15: every
            value of it, no value it does not name, base weights that are
            staged on this box, a form applier behind every key, and both
            cautions written into its header.
  LIVE      with the shipped preset folder: Detector plus coral reaches the
            coral detector, Detector plus fish reaches the fish detector, and
            Segmentation plus coral matches BOTH coral segmentation recipes
            and therefore applies neither.
  DATASET   a staged box dataset's own subject sets where the Subject control
            starts, and never overrides a saved or operator choice.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, as every orchestrator suite does.
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
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
_REPO = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess. Both real
# functions are kept first: the selection checks below run the live
# orchestrator.js in Node, which is this suite's one deliberate subprocess, and
# subprocess.run() spawns through subprocess.Popen, so restoring one without
# the other answers every scenario with the stub's empty output.
_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen


@contextlib.contextmanager
def _node_allowed():
    """Let the two real subprocess functions through for one Node run.

    The stubs are put back in the finally block, so a scenario that throws
    cannot leave a later test able to launch an annotator.
    """
    subprocess.run, subprocess.Popen = _REAL_RUN, _REAL_POPEN
    try:
        yield
    finally:
        subprocess.run = lambda *a, **k: _FakeCompleted()
        subprocess.Popen = _blocked_popen

import yaml  # noqa: E402

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402

_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_TPL = os.path.join(_ORCH, "templates", "index.html")
_PRESET_DIR = os.path.join(_REPO, "config", "presets")
FISH_PRESET_ID = "fish_heavy_augment_640.yaml"
FISH_PRESET_PATH = os.path.join(_PRESET_DIR, FISH_PRESET_ID)

# The training arguments run3_heavy_augment received, read key by key from
# /home/bizon/Desktop/ai/oceankind_CV/fish_training/training_runs/
# run3_heavy_augment/args.yaml on 2026-09-14 and re-read on 2026-09-15. The
# shipped preset must carry these values unchanged, or a VEGA retrain is no
# longer the paper's recipe.
#
# Two entries are not verbatim copies of a line in that file, and a reader
# auditing the recipe against the paper needs to know which:
#   degrees          run3 recorded it as 15.0; the preset ships 15, the same
#                    number written without its decimal part.
#   label_smoothing  absent from run3's args.yaml altogether. The 0.0 here is
#                    VEGA's own trainer default (scripts/TCRMPtrain_oceankindCV/
#                    src/train_wrapper.py line 144), shipped explicitly so the
#                    Train form's field is written rather than keeping whatever
#                    a previously loaded preset left in it. Ultralytics 8.4
#                    treats the key as removed and drops it with a deprecation
#                    warning, so the value changes no training either way.
RECIPE = {
    "model": "yolo11l.pt", "epochs": 350, "imgsz": 640, "batch": 32,
    "patience": 50, "device": "0,1",
    "hsv_h": 0.3, "hsv_s": 0.5, "hsv_v": 0.4, "bgr": 0.0,
    "degrees": 15, "translate": 0.1, "scale": 0.5, "shear": 2.0,
    "perspective": 0.0001, "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.15, "cutmix": 0.0, "copy_paste": 0.1,
    "copy_paste_mode": "flip", "close_mosaic": 10, "optimizer": "auto",
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "cos_lr": False,
    "box": 7.5, "cls": 0.5, "dfl": 1.5, "label_smoothing": 0.0,
}

# The keys of RECIPE that args.yaml does not record. Named here so the
# provenance note above cannot drift away from what the suite asserts.
RECIPE_KEYS_NOT_IN_RUN3_ARGS = {"label_smoothing"}

# run3's own args.yaml, kept as a path so a reader can open it. The suite
# reads it when it is present on this box and skips that check when it is not,
# because the file is Lauren's Desktop copy and not part of the repo.
RUN3_ARGS_YAML = ("/home/bizon/Desktop/ai/oceankind_CV/fish_training/"
                  "training_runs/run3_heavy_augment/args.yaml")


# ---------------------------------------------------------------------------
# The coral recipe (Coral Intel wave, 2026-09-15)
# ---------------------------------------------------------------------------
CORAL_PRESET_ID = "coral_heavy_augment_640.yaml"
CORAL_PRESET_PATH = os.path.join(_PRESET_DIR, CORAL_PRESET_ID)

# The training arguments Lauren approved on 2026-09-15 for the Coral Intel
# subset: 256 by 256 pixel coral photographs, box labels only, kept to the 32
# species carrying at least 100 localised boxes.
#
# Unlike RECIPE above, this is not a transcript of a finished run. It is the
# fish run3 recipe re-aimed at these photographs, with the values Lauren set by
# hand, and it is written out here so that changing the shipped preset takes a
# deliberate edit in two files rather than one quiet one. The keys that differ
# from the fish recipe, and why, are in the preset's own header comment.
CORAL_RECIPE = {
    "model": "yolo11l.pt", "epochs": 600, "imgsz": 640, "batch": 32,
    "patience": 80, "device": "0,1",
    "cos_lr": True, "close_mosaic": 15, "optimizer": "auto",
    "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "box": 7.5, "cls": 0.5, "dfl": 1.5,
    "hsv_h": 0.1, "hsv_s": 0.7, "hsv_v": 0.5, "bgr": 0.0,
    "degrees": 12, "translate": 0.1, "scale": 0.6, "shear": 0,
    "perspective": 0, "flipud": 0.5, "fliplr": 0.5,
    "mosaic": 0.6, "mixup": 0.12, "cutmix": 0.0, "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "valid_ratio": 0.2, "test_ratio": 0.1,
}

# The hue jitter this recipe sits between, read from the neighbouring files
# themselves so that a change in either one is caught here. Both coral
# segmentation presets jitter hue by 0.03 and the fish detector by 0.3; 0.1 is
# the middle setting Lauren chose, and no ablation has tested it.
CORAL_SEG_PRESET_IDS = ("tuned_v1_regularized.yaml", "yolo26l_heavy_seg_aug.yaml")

# What every shipped recipe declares it is for, as (task, subject). Pinned
# here so that a recipe changing what it is written for, or a new recipe
# arriving beside these, fails this suite instead of quietly changing which
# project opens on which recipe. That is the failure this whole file exists
# for: on 2026-09-15 the coral DETECTION recipe took the coral segmentation
# default simply by sorting first.
SHIPPED_DECLARATIONS = {
    "coral_heavy_augment_640.yaml": ("detect", "coral"),
    "fish_heavy_augment_640.yaml": ("detect", "fish"),
    "tuned_v1_regularized.yaml": ("segment", "coral"),
    "yolo26l_heavy_seg_aug.yaml": ("segment", "coral"),
}

# The smallest shipped preset folder that is plausibly the real one. A helper
# that read an empty folder would otherwise pass every caller.
_MIN_SHIPPED_PRESETS = 3

# The base weights the recipe names have to be on disk, or the first press of
# Start Training reaches the network. Ultralytics resolves a bare checkpoint
# name against the training process's working directory, which is
# WORKING_DIRS[6] (scripts/TCRMPtrain_oceankindCV).
CORAL_MODEL_STAGED_AT = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "yolo11l.pt")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# JavaScript words that are followed by "(" without being a function call.
# Anything else an applier body calls has to be a helper declared in the file.
_JS_NON_CALL_WORDS = frozenset({
    "if", "for", "while", "switch", "catch", "return", "typeof", "function",
})

# Globals an applier body may legitimately call without orchestrator.js
# declaring them. Kept explicit so a typo cannot hide behind a broad rule.
_JS_ALLOWED_GLOBALS = frozenset({"Boolean", "Number", "String", "parseFloat", "parseInt"})

# The smallest applier map that is plausibly complete. A regex that matched
# nothing would otherwise pass every caller in silence.
_MIN_APPLIER_KEYS = 10


def _js_source():
    """The live orchestrator JavaScript as text.

    Returns the whole file. Raises OSError naming the path when it cannot be
    read, because every check below is meaningless without it.
    """
    try:
        with open(_JS, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise OSError(f"cannot read the orchestrator JavaScript at {_JS}: {exc}") from exc


def _applier_bodies():
    """The STEP6_PRESET_APPLIERS entries, parsed out of static/orchestrator.js.

    Returns a dict of preset parameter name to the text of the arrow function
    that writes it into the Train form. Raises AssertionError when the object
    literal cannot be found or bounded, because a silent empty dict would pass
    every caller.
    """
    src = _js_source()
    marker = "const STEP6_PRESET_APPLIERS = {"
    start = src.find(marker)
    check(start >= 0, f"STEP6_PRESET_APPLIERS not found in {_JS}")
    body_at = start + len(marker)
    end = src.find("\n};", body_at)
    check(end > body_at, "STEP6_PRESET_APPLIERS has no closing brace on its own line")
    body = src[body_at:end]
    entries = {}
    for line in body.splitlines():
        m = re.match(r"^\s{4}([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?),?\s*$", line)
        if m:
            entries[m.group(1)] = m.group(2)
    check(len(entries) > _MIN_APPLIER_KEYS,
          f"parsed too few applier entries ({sorted(entries)})")
    return entries


def _applier_keys():
    """The parameter names the Train form can apply. A set of key strings."""
    return set(_applier_bodies())


def _declared_js_names():
    """Every name orchestrator.js declares at the top level of the file.

    Returns a set of identifier strings covering `function f(`, `async
    function f(`, and `const`, `let` or `var` bindings written flush left.
    These are the helpers an applier body is allowed to call.
    """
    src = _js_source()
    names = set(re.findall(r"^(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
                           src, re.M))
    names |= set(re.findall(r"^(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
                            src, re.M))
    check(len(names) > _MIN_APPLIER_KEYS,
          f"parsed too few top-level declarations from {_JS} ({len(names)})")
    return names


def _calls_in(js_body):
    """The plain function names a snippet of JavaScript calls.

    js_body: the text of one applier arrow function.
    Returns a sorted list of identifiers called as `name(`, with member calls
    (`document.getElementById`) and language keywords left out.
    """
    raw = re.findall(r"(?<![.\w$])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", js_body)
    return sorted({n for n in raw if n not in _JS_NON_CALL_WORDS})


def _field_ids_in(js_body):
    """The Train form element ids a snippet of JavaScript names.

    js_body: the text of one applier arrow function.
    Returns a sorted list of the 's6-...' string literals it writes to.
    """
    return sorted(set(re.findall(r"['\"](s6-[A-Za-z0-9-]+)['\"]", js_body)))


def _make_project(tmp, extra_state=None):
    """A project state dict of the shape project_manager writes, with every
    step dir present on disk under tmp.

    tmp: an empty temp directory that becomes project_dir.
    extra_state: keys merged over the base state (project_type and its
    training_source or composite block).
    Returns the state dict.
    """
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    state = {
        "id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
        "created_at": "x", "updated_at": "x", "current_step": 6, "steps": steps,
    }
    state.update(extra_state or {})
    return state


_NO_PROJECT = object()


@contextlib.contextmanager
def _project_open(extra_state):
    """Yield a Flask test client with a project of the given kind loaded.

    extra_state: _NO_PROJECT for no open project, otherwise the keys merged
    into a temp project's state ({} for an ordinary annotation project).
    The temp project dir is removed and the global is cleared on exit.
    """
    tmp = tempfile.mkdtemp(prefix="step6_preset_default_")
    try:
        if extra_state is _NO_PROJECT:
            appmod.current_project = None
        else:
            appmod.current_project = _make_project(tmp, extra_state)
        yield appmod.create_app().test_client()
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


_BBOX_STATE = {"project_type": appmod.BBOX_PROJECT_TYPE,
               "training_source": {"kind": "bbox", "name": "fish_photo_28feb25",
                                   "path": "/nowhere/fish_photo_28feb25",
                                   "task_default": "detect"}}
_COMPOSITE_STATE = {"project_type": appmod.COMPOSITE_PROJECT_TYPE,
                    "composite": {"name": "set_a", "path": "/nowhere/set_a",
                                  "source_projects": [], "created_at": "x"}}


@contextlib.contextmanager
def _preset_dir(files):
    """Point the train preset folder at a temp dir holding the given files.

    files: mapping of file name to file text.
    Restores the real folder on exit, so a later test in this suite still sees
    the shipped presets.
    """
    tmp = tempfile.mkdtemp(prefix="step6_presets_")
    was = appmod.PRESET_DIRS.get("train", "")
    try:
        for name, text in files.items():
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        appmod.PRESET_DIRS["train"] = tmp
        yield tmp
    finally:
        appmod.PRESET_DIRS["train"] = was
        shutil.rmtree(tmp, ignore_errors=True)


def _presets(client):
    """The parsed /api/step/6/presets payload, with a status check."""
    r = client.get("/api/step/6/presets")
    check(r.status_code == 200, f"presets route answered {r.status_code}")
    return r.get_json() or {}


def _entry(data, pid):
    """The one listed entry with this id, or an AssertionError naming the ids
    that were listed."""
    hits = [p for p in (data.get("presets") or []) if p.get("id") == pid]
    check(len(hits) == 1,
          f"expected exactly one entry for {pid}, listed: "
          f"{[p.get('id') for p in (data.get('presets') or [])]}")
    return hits[0]


def _preset_body(client, pid):
    """The parsed /api/step/6/presets/<pid> payload plus its status code.

    Returns (status_code, body dict). Never raises on a non-200, because the
    damaged-file cases below assert on the status and the message.
    """
    r = client.get(f"/api/step/6/presets/{pid}")
    body = r.get_json()
    check(isinstance(body, dict),
          f"fetching {pid} answered {r.status_code} with a non-JSON body: {r.data[:200]!r}")
    return r.status_code, body


# Recipe fixtures. Each one is the smallest file the route will read: a _meta
# block with the two declarations, and one parameter.
def _recipe(name, task=None, subject=None, epochs=1):
    """A preset file's text, declaring whatever this fixture needs.

    Parameters:
      name: the _meta.name the picker shows.
      task: the _meta.task line, or None to leave the key out.
      subject: the _meta.subject line, or None to leave the key out.
      epochs: the single parameter, so two fixtures can be told apart after
        they reach the form.
    Returns:
      the file text.
    """
    lines = ["_meta:", f"  name: {name}"]
    if task is not None:
        lines.append(f"  task: {task}")
    if subject is not None:
        lines.append(f"  subject: {subject}")
    lines += ["params:", f"  epochs: {epochs}"]
    return "\n".join(lines) + "\n"


_CORAL_DETECT = _recipe("Coral detector", "detect", "coral", 3)
_FISH_DETECT = _recipe("Fish detector", "detect", "fish", 4)
_ANY_SEG = _recipe("Any segmenter", "segment", "any", 6)
# Unclassified shapes: no declarations at all, one of the two missing, and a
# word outside each vocabulary. Every one of them is listed and never matched.
_UNCLASSIFIED = _recipe("Plain")
_TASK_ONLY = _recipe("Task only", "detect", None)
_SUBJECT_ONLY = _recipe("Subject only", None, "coral")
_BAD_SUBJECT = _recipe("Odd subject", "detect", "reef")
_BAD_TASK = _recipe("Odd task", "pose", "coral")
_MESSY_CASE = _recipe("Messy", '"  Detection "', '"  Coral "', 7)
_LIST_SUBJECT = ("_meta:\n  name: Listy subject\n  task: detect\n"
                 "  subject:\n    - coral\nparams:\n  epochs: 8\n")
# Damaged shapes. Every one of these used to answer a 500 from one of the two
# routes, which takes out the whole picker.
_UNPARSABLE = "_meta: [unclosed\n  name: nope\n"
_NOT_A_MAPPING = "- one\n- two\n"
_SCALAR_META = "_meta: just a string\nparams:\n  epochs: 1\n"
_LIST_PARAMS = "_meta:\n  name: Listy\nparams:\n  - 1\n  - 2\n"


@contextlib.contextmanager
def _staged_dataset(manifest):
    """A folder shaped like a staged box dataset, holding the given manifest.

    manifest: a dict written as JSON, or a raw string or list written as it is,
      so a damaged file can be handed to the route.
    Yields the folder path and removes it on exit.
    """
    tmp = tempfile.mkdtemp(prefix="step6_dataset_")
    try:
        with open(os.path.join(tmp, "data.yaml"), "w", encoding="utf-8") as fh:
            fh.write("names:\n  0: a\nnc: 1\n")
        path = os.path.join(tmp, "ingest_manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            if isinstance(manifest, str):
                fh.write(manifest)
            else:
                json.dump(manifest, fh)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _bbox_state_for(dataset_dir):
    """The project state of a box-dataset project training from dataset_dir."""
    return {"project_type": appmod.BBOX_PROJECT_TYPE,
            "training_source": {"kind": "bbox", "name": os.path.basename(dataset_dir),
                                "path": dataset_dir, "task_default": "detect"}}


def _node():
    """The node binary, or None when this box has none."""
    for candidate in ("/usr/bin/node", "node"):
        try:
            with _node_allowed():
                _REAL_RUN([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except Exception:
            continue
    return None


# Node driver: loads the live orchestrator.js into a vm context with a fake
# document whose getElementById mints an element per id and a fetch that never
# resolves (so the page's init parks harmlessly), then runs a scenario in the
# same context and prints its JSON result. Started from the harness in
# test_ui_sweep_20260905.py and kept here so this suite stands on its own; this
# copy also awaits a scenario whose result is a promise, which is what lets a
# test drive the panel's own async loader rather than reading its source.
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
// A scenario whose last expression is a promise is awaited, so an async panel
// function can be run rather than read. A plain value passes straight through.
Promise.resolve(out).then(
  (value) => { process.stdout.write(JSON.stringify(value)); },
  (err) => { console.error(err && err.stack ? err.stack : String(err)); process.exit(3); },
);
"""


def _drive(scenario):
    """Run one scenario against the live orchestrator.js in Node.

    scenario: JavaScript whose last expression is the result to report.
    Returns the parsed JSON result.
    Raises AssertionError naming node's own error output when the scenario
    does not run, because a swallowed failure here would pass every caller.
    """
    node = _node()
    check(node is not None, "node is not on this box, so the panel's own rule cannot be run")
    tmp = tempfile.mkdtemp(prefix="step6_selection_")
    try:
        driver = os.path.join(tmp, "driver.js")
        scen = os.path.join(tmp, "scenario.js")
        with open(driver, "w", encoding="utf-8") as fh:
            fh.write(_NODE_DRIVER)
        with open(scen, "w", encoding="utf-8") as fh:
            fh.write(scenario)
        with _node_allowed():
            result = _REAL_RUN([node, driver, _JS, scen], capture_output=True,
                               text=True, timeout=60)
        check(result.returncode == 0,
              f"the selection scenario failed to run:\n{result.stderr}")
        return json.loads(result.stdout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _select(recipes, task, subject, *, applied="", dataset_note="", before=""):
    """What the Train panel does with these recipes, that task and that subject.

    recipes: the entries /api/step/6/presets would answer.
    task: 'detect' or 'segment', as the Training task control holds it.
    subject: as the Subject control holds it.
    applied: the recipe id whose values are already in the form, '' for none.
    dataset_note: the dataset warning the payload carried, '' for none.
    before: extra JavaScript run after the controls are set and before the
      render, for a test that changes a control twice.
    Returns a dict of matched, hidden and unclassified ids, the ids and labels
    the dropdown lists, the id it selects, and the line under it.
    """
    scenario = (
        f"_step6Recipes = {json.dumps(recipes)};\n"
        f"_step6DatasetNote = {json.dumps(dataset_note)};\n"
        "const sel = document.getElementById('s6-preset-select');\n"
        "sel.children = []; sel.value = '';\n"
        f"sel.dataset.appliedPresetId = {json.dumps(applied)};\n"
        f"document.getElementById('s6-task').value = {json.dumps(task)};\n"
        f"document.getElementById('s6-subject').value = {json.dumps(subject)};\n"
        f"{before}\n"
        "const sorted = renderStep6RecipeOptions();\n"
        "({ matched: sorted.matches.map(r => r.id),\n"
        "   hidden: sorted.hidden.map(r => r.id),\n"
        "   unclassified: sorted.unclassified.map(r => r.id),\n"
        "   listed: sel.children.map(o => o.value),\n"
        "   labels: sel.children.map(o => o.textContent),\n"
        "   selected: sel.value,\n"
        "   line: document.getElementById('s6-preset-selection').textContent })\n")
    return _drive(scenario)


def _shipped_recipes():
    """The entries /api/step/6/presets answers for the real preset folder."""
    with _project_open({}) as client:
        data = _presets(client)
    check(len(data["presets"]) >= _MIN_SHIPPED_PRESETS,
          f"the shipped folder answered too few recipes to test against: {data['presets']}")
    return data["presets"]


# ---------------------------------------------------------------------------
# PRESET: the shipped file is run3's recipe
# ---------------------------------------------------------------------------

def test_shipped_preset_matches_run3():
    check(os.path.isfile(FISH_PRESET_PATH), f"missing preset file {FISH_PRESET_PATH}")
    with open(FISH_PRESET_PATH, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    params = doc.get("params") or {}
    for key, want in RECIPE.items():
        got = params.get(key)
        check(got == want, f"preset {key} = {got!r}, run3 used {want!r}")
    meta = doc.get("_meta") or {}
    check((meta.get("task"), meta.get("subject")) == ("detect", "fish"),
          f"the fish preset must declare task detect and subject fish, it declares "
          f"{meta.get('task')!r} and {meta.get('subject')!r}")
    check(str(meta.get("name") or "").strip(), "preset needs a _meta.name")
    check(str(meta.get("description") or "").strip(),
          "preset needs a _meta.description for the picker")
    check(params.get("valid_ratio") == 0.2 and params.get("test_ratio") == 0.1,
          f"the paper's split fractions must ship with the recipe: {params.get('valid_ratio')!r} "
          f"and {params.get('test_ratio')!r}")
    # min_samples belongs to the outline splitter and is hidden for a box
    # dataset; a value here would write a knob the panel does not show.
    check("min_samples" not in params, "min_samples must not ship in a box-dataset preset")
    check("pin_split" not in params, "pin_split must not ship in a box-dataset preset")


def test_every_preset_key_has_a_form_applier():
    keys = _applier_keys()
    with open(FISH_PRESET_PATH, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    unknown = sorted(k for k in (doc.get("params") or {}) if k not in keys)
    check(not unknown,
          f"these preset keys have no applier in STEP6_PRESET_APPLIERS and would be "
          f"dropped in silence at import: {unknown}")


def test_run3_args_yaml_records_every_recipe_key_the_note_claims():
    """The provenance note above RECIPE says which keys run3's args.yaml does
    not record. Re-read the file and hold the note to it, so the record cannot
    drift from the evidence it cites."""
    if not os.path.isfile(RUN3_ARGS_YAML):
        print(f"    (skipped: {RUN3_ARGS_YAML} is not on this box)")
        return
    with open(RUN3_ARGS_YAML, encoding="utf-8") as fh:
        args = yaml.safe_load(fh) or {}
    missing = sorted(k for k in RECIPE if k not in args)
    check(missing == sorted(RECIPE_KEYS_NOT_IN_RUN3_ARGS),
          f"run3's args.yaml records every RECIPE key except {sorted(RECIPE_KEYS_NOT_IN_RUN3_ARGS)}, "
          f"but this run found {missing} absent; update the provenance note above RECIPE")
    for key, want in RECIPE.items():
        if key in RECIPE_KEYS_NOT_IN_RUN3_ARGS:
            continue
        check(args[key] == want,
              f"run3's args.yaml records {key} = {args[key]!r}, RECIPE claims {want!r}")


def test_every_applier_resolves_to_a_live_helper():
    """An applier that calls a helper a refactor renamed throws at import
    time. applyStep6PresetParams does not guard the call, and the auto-apply
    path swallows the throw, so every remaining key of the recipe is dropped
    and the page says nothing. This is that failure, caught statically."""
    declared = _declared_js_names() | _JS_ALLOWED_GLOBALS
    dangling = {}
    for key, body in _applier_bodies().items():
        unresolved = [n for n in _calls_in(body) if n not in declared]
        if unresolved:
            dangling[key] = unresolved
    check(not dangling,
          f"these STEP6_PRESET_APPLIERS entries call helpers that no longer exist in "
          f"{_JS}, so applying a preset throws and drops every key after them: {dangling}")


def test_every_applier_writes_a_field_that_exists():
    """setVal and getElementById both do nothing when the id is not on the
    page, so an applier naming a renamed field is a silent no-op."""
    tpl = open(_TPL, encoding="utf-8").read()
    missing = {}
    for key, body in _applier_bodies().items():
        absent = [i for i in _field_ids_in(body) if f'id="{i}"' not in tpl]
        if absent:
            missing[key] = absent
    check(not missing,
          f"these STEP6_PRESET_APPLIERS entries write form fields that are not in "
          f"{_TPL}, so the value is dropped in silence: {missing}")


def test_the_device_applier_drives_the_step_6_device_controls():
    """device is the one applier that writes a pair of controls through a
    shared helper, so the step number it passes is worth pinning: step 8 there
    would write the inference panel instead."""
    body = _applier_bodies().get("device", "")
    check(body, "STEP6_PRESET_APPLIERS has no device entry")
    check(re.search(r"applyStepDevice\(\s*6\s*,", body),
          f"the device applier must call applyStepDevice with step 6, it reads: {body!r}")
    tpl = open(_TPL, encoding="utf-8").read()
    for element_id in ("s6-device-preset", "s6-device-custom"):
        check(f'id="{element_id}"' in tpl,
              f"applyStepDevice(6, ...) writes #{element_id}, which is not in {_TPL}")


def test_the_device_applier_falls_back_to_the_train_panel_default():
    """A named constant, not a repeated string and not the inference panel's
    default: a preset that names no device must open the Train panel on the
    same device the panel restores a saved config with."""
    src = _js_source()
    body = _applier_bodies().get("device", "")
    check("TRAIN_DEVICE_DEFAULT" in body,
          f"the device applier must pass the named TRAIN_DEVICE_DEFAULT, it reads: {body!r}")
    check("INFER_DEVICE_DEFAULT" not in body,
          f"the device applier must not pass the inference panel's default: {body!r}")
    m = re.search(r"^const TRAIN_DEVICE_DEFAULT = '([^']*)';", src, re.M)
    check(m is not None and m.group(1).strip(),
          "TRAIN_DEVICE_DEFAULT is not declared with a value in orchestrator.js")


def test_preset_header_records_the_ablation():
    """The header comment is the only place a reader of the file meets the
    evidence, so it must carry the baseline and the best score and point at
    the model card."""
    text = open(FISH_PRESET_PATH, encoding="utf-8").read()
    head = text[:text.index("_meta:")]
    for needle in ("0.798", "0.904", "run3", "n-20260914-231000-fish-detector-model-report.md"):
        check(needle in head, f"the preset header must mention {needle!r}")
    check("hsv_h 0.3" in head or "hue" in head.lower(),
          "the header must carry the hue-jitter caution")
    # The platform prose rule: no em dash (U+2014) and no en dash (U+2013).
    for banned in ("\u2014", "\u2013"):
        check(banned not in text, f"the preset file must carry no {banned!r}")


def test_preset_base_weights_are_an_option_in_the_model_picker():
    """setVal on the model select silently blanks the field when the value is
    not one of its options, so the preset's base weights must be listed."""
    tpl = open(_TPL, encoding="utf-8").read()
    with open(FISH_PRESET_PATH, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    model = (doc.get("params") or {}).get("model")
    select_at = tpl.index('id="s6-model"')
    select = tpl[select_at:tpl.index("</select>", select_at)]
    check(f'value="{model}"' in select,
          f"the preset names base weights {model!r} that the Train model picker does not offer")


def test_shipped_preset_parses_through_the_route():
    with _project_open(_BBOX_STATE) as client:
        r = client.get(f"/api/step/6/presets/{FISH_PRESET_ID}")
        check(r.status_code == 200, f"fetching the preset answered {r.status_code}")
        body = r.get_json() or {}
        check("error" not in body, f"the shipped preset must parse: {body.get('error')!r}")
        check((body.get("params") or {}).get("model") == "yolo11l.pt",
              f"the fetched params must carry the base weights: {body.get('params')}")


# ---------------------------------------------------------------------------
# ROUTE: what each recipe declares, and that the route decides nothing
# ---------------------------------------------------------------------------

def test_every_entry_carries_task_subject_and_classified():
    """The documented contract of this route: EVERY entry carries task,
    subject and classified. The Train panel's filter reads all three on every
    entry, so one entry missing a key would break the whole picker; the
    fixture therefore holds two damaged shapes beside two healthy recipes.

    2026-09-21: the two built-in presets (finetune, retrain) that used to head
    this list are gone (design 2026-09-21-vega-starting-model-and-training-
    data-design.md section 3.2), so the listing now holds file recipes only."""
    files = {"a_broken.yaml": _UNPARSABLE, "b_list.yaml": _NOT_A_MAPPING,
             "c_coral.yaml": _CORAL_DETECT, "z_any.yaml": _ANY_SEG}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            ids = sorted(p["id"] for p in data["presets"])
            check(ids == ["a_broken.yaml", "b_list.yaml", "c_coral.yaml", "z_any.yaml"],
                  f"every file, and no built-in, must be listed: {ids}")
            check("builtin" not in str(data), f"no entry may carry a builtin flag: {data}")
            for p in data["presets"]:
                for key in ("task", "subject", "classified"):
                    check(key in p, f"entry {p.get('id')!r} is missing {key}")
                check(isinstance(p["task"], str) and isinstance(p["subject"], str),
                      f"entry {p.get('id')!r} must carry task and subject as strings: {p}")
                check(isinstance(p["classified"], bool),
                      f"entry {p.get('id')!r} classified must be a boolean: {p}")
            coral = _entry(data, "c_coral.yaml")
            check((coral["task"], coral["subject"], coral["classified"]) == ("detect", "coral", True),
                  f"a declared recipe reports what it declared: {coral}")
            any_seg = _entry(data, "z_any.yaml")
            check((any_seg["task"], any_seg["subject"]) == ("segment", "any"),
                  f"a recipe written for every subject reports subject any: {any_seg}")
            for pid in ("a_broken.yaml", "b_list.yaml"):
                entry = _entry(data, pid)
                check(entry["task"] == "" and entry["subject"] == "",
                      f"{pid} declares nothing and must report nothing: {entry}")
                check(entry["classified"] is False,
                      f"{pid} must never read as classified: {entry}")


def test_the_removed_builtin_preset_ids_answer_404():
    """2026-09-21: the two built-in presets are gone. A stale client, or a
    stored preset marker from before the change, asking for either id by name
    gets the same 404 an unknown file name gets, not a resurrected built-in."""
    with _project_open(_BBOX_STATE) as client:
        for preset_id in ("finetune", "retrain"):
            r = client.get(f"/api/step/6/presets/{preset_id}")
            body = r.get_json()
            check(r.status_code == 404,
                  f"{preset_id} must answer 404 like any other unknown file name: "
                  f"{r.status_code} {body}")
            check("error" in body, f"{preset_id} 404 must carry an error message: {body}")


def test_the_route_chooses_no_recipe_at_all():
    """The heart of the change. The route reports declarations and picks
    nothing: no default_id in the payload, and no per-entry flag a client could
    read as 'apply this one'. The two controls on the panel decide, and a
    second decider here is exactly what could disagree with them in silence."""
    files = {"a_first.yaml": _CORAL_DETECT, "z_last.yaml": _FISH_DETECT}
    with _preset_dir(files):
        for state in (_BBOX_STATE, _COMPOSITE_STATE, {}, _NO_PROJECT):
            with _project_open(state) as client:
                data = _presets(client)
                check("default_id" not in data,
                      f"the payload must name no default recipe: {sorted(data)}")
                for p in data["presets"]:
                    for gone in ("default", "default_for_source"):
                        check(gone not in p,
                              f"entry {p.get('id')!r} still carries {gone!r}, the flag "
                              f"the file-name rule resolved: {p}")


def test_declarations_are_read_case_and_space_insensitively():
    """An operator writing "  Detection " in a recipe means detect. Both keys
    are normalized the way the trainer normalizes a task, so a recipe is never
    unclassified over letter case."""
    with _preset_dir({"z_messy.yaml": _MESSY_CASE}):
        with _project_open(_BBOX_STATE) as client:
            entry = _entry(_presets(client), "z_messy.yaml")
            check((entry["task"], entry["subject"]) == ("detect", "coral"),
                  f"mixed case with stray spaces must still declare detect and coral: {entry}")
            check(entry["classified"] is True, f"and must count as classified: {entry}")


def test_a_recipe_missing_either_key_is_unclassified_and_says_why():
    """A recipe that declares neither key, or only one of them, is listed so it
    can still be imported by hand and is never matched. The entry carries the
    reason, because a recipe nobody can be offered must not look healthy."""
    files = {"a_none.yaml": _UNCLASSIFIED, "b_task.yaml": _TASK_ONLY,
             "c_subject.yaml": _SUBJECT_ONLY, "z_ok.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            for pid in ("a_none.yaml", "b_task.yaml", "c_subject.yaml"):
                entry = _entry(data, pid)
                check(entry["classified"] is False,
                      f"{pid} declares too little to be classified: {entry}")
                warning = str(entry.get("warning") or "")
                check(warning, f"{pid} must carry a warning saying why: {entry}")
                check("unclassified" in warning,
                      f"the warning must use the word the panel shows: {warning!r}")
            check("_meta.subject" in _entry(data, "b_task.yaml")["warning"],
                  "a recipe with a task and no subject must be told which key is missing")
            check("_meta.task" in _entry(data, "c_subject.yaml")["warning"],
                  "a recipe with a subject and no task must be told which key is missing")
            healthy = _entry(data, "z_ok.yaml")
            check(healthy["classified"] is True and "warning" not in healthy,
                  f"a fully declared recipe carries no warning: {healthy}")


def test_a_value_outside_the_vocabulary_is_named_in_the_warning():
    """A misspelled subject would match nothing and a misspelled task likewise,
    so the entry names the value it read and the words it accepts."""
    files = {"a_subject.yaml": _BAD_SUBJECT, "b_task.yaml": _BAD_TASK,
             "c_list.yaml": _LIST_SUBJECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            subject_warning = _entry(data, "a_subject.yaml")["warning"]
            check("'reef'" in subject_warning,
                  f"the warning must quote the value read: {subject_warning!r}")
            for word in appmod.STEP6_PRESET_SUBJECTS:
                check(word in subject_warning,
                      f"the warning must name the subject {word!r} a recipe may use: "
                      f"{subject_warning!r}")
            task_warning = _entry(data, "b_task.yaml")["warning"]
            check("'pose'" in task_warning,
                  f"the warning must quote the task read: {task_warning!r}")
            for word in ("detect", "segment"):
                check(word in task_warning,
                      f"the warning must name the task {word!r}: {task_warning!r}")
            listy = _entry(data, "c_list.yaml")
            check(listy["subject"] == "" and listy["classified"] is False,
                  f"a subject that is not a string declares no subject: {listy}")


def test_two_recipes_may_declare_the_same_pair_and_both_are_reported():
    """The route neither refuses the pair nor ranks it. Both entries come back
    classified, and what happens next is the panel's rule, which is that
    nothing is applied."""
    files = {"a_one.yaml": _CORAL_DETECT, "z_two.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            for pid in ("a_one.yaml", "z_two.yaml"):
                entry = _entry(data, pid)
                check(entry["classified"] is True and entry["task"] == "detect"
                      and entry["subject"] == "coral",
                      f"{pid} must be reported as declared: {entry}")


def test_an_empty_preset_folder_lists_nothing():
    """2026-09-21: with the two built-in presets removed, an empty recipe
    folder lists nothing at all, rather than falling back to Fine-tune and
    Full retrain."""
    with _preset_dir({}):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            ids = [p["id"] for p in data["presets"]]
            check(ids == [], f"an empty folder must list no recipe: {ids}")
            check("default_id" not in data,
                  f"an empty folder names no default either: {sorted(data)}")


def test_a_non_yaml_file_in_the_folder_is_skipped():
    files = {"notes.txt": "not a preset\n", "z_coral.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            ids = [p["id"] for p in _presets(client)["presets"]]
            check("notes.txt" not in ids, f"a non-YAML file must not be listed: {ids}")
            check("z_coral.yaml" in ids, f"the YAML recipe is listed: {ids}")


# ---------------------------------------------------------------------------
# ADVERSARIAL: one damaged file must never take out the picker
# ---------------------------------------------------------------------------

def test_unparsable_preset_is_reported_and_stays_unclassified():
    files = {"a_broken.yaml": _UNPARSABLE, "z_coral.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            broken = _entry(data, "a_broken.yaml")
            check("error" in broken, f"a damaged preset must report its error: {broken}")
            check(broken["classified"] is False,
                  f"a damaged preset can never be matched: {broken}")
            check(_entry(data, "z_coral.yaml")["classified"] is True,
                  "the healthy recipe beside it still reaches the picker")


def test_a_preset_that_is_not_a_mapping_is_reported_not_fatal():
    """Valid YAML that parses to a list used to reach .get() and answer a 500.
    It must be reported like any other damaged file, by both routes: the
    listing renders a damaged entry as a selectable option, so fetching one is
    a click away."""
    files = {"a_list.yaml": _NOT_A_MAPPING, "z_coral.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            broken = _entry(data, "a_list.yaml")
            check("error" in broken, f"a list-shaped preset must report its error: {broken}")
            check("mapping" in broken["error"],
                  f"the error must say what the file should have held: {broken['error']!r}")
            status, body = _preset_body(client, "a_list.yaml")
            check(status != 500 or "error" in body,
                  f"fetching a list-shaped preset must answer a message, not a bare 500: "
                  f"{status} {body}")
            check("mapping" in str(body.get("error", "")),
                  f"the fetch error must say what the file should have held: {body}")


def test_a_scalar_meta_section_is_survivable_in_both_routes():
    """A file whose _meta is a scalar used to reach meta.get() and answer a
    500 from the listing, which takes out the picker for every project. The
    section is unusable, not the file: the preset stays listed under its file
    name, reads as unclassified, and its params still load."""
    files = {"a_scalar_meta.yaml": _SCALAR_META, "z_coral.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            entry = _entry(data, "a_scalar_meta.yaml")
            check(entry["task"] == "" and entry["subject"] == "",
                  f"an unusable _meta declares nothing: {entry}")
            check(entry["classified"] is False,
                  f"and can never be matched: {entry}")
            check(entry["name"] == "a_scalar_meta.yaml",
                  f"the name falls back to the file name: {entry}")
            status, body = _preset_body(client, "a_scalar_meta.yaml")
            check(status == 200, f"fetching it must succeed: {status} {body}")
            check((body.get("params") or {}).get("epochs") == 1,
                  f"its params must still load: {body}")


def test_a_non_mapping_params_section_is_reported_not_applied():
    """params that parses to a list would reach the client as numeric keys and
    apply nothing, in silence. The fetch must say so instead."""
    files = {"a_list_params.yaml": _LIST_PARAMS, "z_coral.yaml": _CORAL_DETECT}
    with _preset_dir(files):
        with _project_open(_BBOX_STATE) as client:
            data = _presets(client)
            _entry(data, "a_list_params.yaml")   # still listed, not fatal
            status, body = _preset_body(client, "a_list_params.yaml")
            check("error" in body,
                  f"a list-shaped params section must be reported: {status} {body}")
            check("params" in str(body["error"]),
                  f"the error must name the params section: {body['error']!r}")


# ---------------------------------------------------------------------------
# DATASET: where the Subject control starts
# ---------------------------------------------------------------------------

def test_the_route_reports_the_subject_a_box_dataset_recorded():
    """A staged box dataset may record what it is about. The panel starts its
    Subject control there, which is the only thing this value does."""
    with _staged_dataset({"dataset": "d", "subject": "fish"}) as dataset_dir:
        with _preset_dir({"z_coral.yaml": _CORAL_DETECT}):
            with _project_open(_bbox_state_for(dataset_dir)) as client:
                data = _presets(client)
                check(data.get("dataset_subject") == "fish",
                      f"the dataset's own subject must reach the panel: {data.get('dataset_subject')!r}")
                check("dataset_subject_warning" not in data,
                      f"a readable manifest raises no warning: {data}")


def test_a_project_with_no_staged_dataset_reports_no_subject():
    with _preset_dir({"z_coral.yaml": _CORAL_DETECT}):
        for state in ({}, _COMPOSITE_STATE, _NO_PROJECT):
            with _project_open(state) as client:
                data = _presets(client)
                check(data.get("dataset_subject") == "",
                      f"a project with no staged box dataset reports no subject: {data!r}")


def test_a_dataset_subject_outside_the_vocabulary_is_ignored():
    """A hand-edited manifest cannot put the Subject control on a value the
    control does not offer, and cannot make the panel select anything."""
    for recorded in ("reef", "", 7, ["coral"], None):
        with _staged_dataset({"dataset": "d", "subject": recorded}) as dataset_dir:
            with _preset_dir({"z_coral.yaml": _CORAL_DETECT}):
                with _project_open(_bbox_state_for(dataset_dir)) as client:
                    data = _presets(client)
                    check(data.get("dataset_subject") == "",
                          f"subject {recorded!r} must be read as no subject, got "
                          f"{data.get('dataset_subject')!r}")


def test_a_damaged_dataset_manifest_says_so_instead_of_failing():
    """The listing must survive a manifest somebody broke, and must not go
    quiet about it: the panel shows the sentence beside the selection line."""
    with _staged_dataset("{not json at all") as dataset_dir:
        with _preset_dir({"z_coral.yaml": _CORAL_DETECT}):
            with _project_open(_bbox_state_for(dataset_dir)) as client:
                data = _presets(client)
                check(data.get("dataset_subject") == "",
                      f"a damaged manifest records no subject: {data!r}")
                warning = str(data.get("dataset_subject_warning") or "")
                check(warning, f"and the payload must say why: {data!r}")
                check(dataset_dir in warning,
                      f"the warning must name the file it could not read: {warning!r}")
    with _staged_dataset(["not", "a", "mapping"]) as dataset_dir:
        with _preset_dir({"z_coral.yaml": _CORAL_DETECT}):
            with _project_open(_bbox_state_for(dataset_dir)) as client:
                data = _presets(client)
                warning = str(data.get("dataset_subject_warning") or "")
                check("list" in warning,
                      f"a manifest holding a list must be named as such: {warning!r}")


def test_the_dataset_subject_vocabulary_is_the_stagers_own():
    """One vocabulary, two trees. The Train panel reads what the box-dataset
    stager writes, so the two lists are held against each other here rather
    than left to agree by memory."""
    sys.path.insert(0, _SCRIPTS)
    from _boxes import ingest_yolo_boxes as ing
    check(tuple(ing.DATASET_SUBJECTS) == tuple(appmod.STEP6_SUBJECTS),
          f"the stager records {ing.DATASET_SUBJECTS} and the panel reads "
          f"{appmod.STEP6_SUBJECTS}; a dataset staged with a word the panel does not "
          f"know would start the Subject control nowhere")


# ---------------------------------------------------------------------------
# SELECTION: the panel's own rule, run against the live orchestrator.js
# ---------------------------------------------------------------------------

# Two recipes claiming one task and subject, the first sorting before the
# second, so a rule that broke the tie by name would be visible here.
_TWO_CLAIMS = [
    {"id": "a_first.yaml", "name": "First by name", "task": "detect",
     "subject": "coral", "classified": True, "description": ""},
    {"id": "z_second.yaml", "name": "Second by name", "task": "detect",
     "subject": "coral", "classified": True, "description": ""},
]
_ONE_CLAIM = [_TWO_CLAIMS[0]]


def test_one_match_is_offered_and_the_line_names_the_task_and_subject():
    out = _select(_ONE_CLAIM, "detect", "coral")
    check(out["matched"] == ["a_first.yaml"], f"one recipe must match: {out}")
    check(out["selected"] == "a_first.yaml",
          f"the single match must be offered as the selection: {out}")
    check(out["line"].startswith("Training task detector and subject coral match one "
                                 'recipe, "First by name".'),
          f"the line must name the task, the subject and the recipe: {out['line']!r}")
    check("Press Import" in out["line"],
          f"and must say how to put it in the form: {out['line']!r}")


def test_the_line_says_so_when_the_form_was_filled_from_the_recipe():
    """The complaint this whole change answers: the panel must be able to say
    which recipe is in the form. Once a recipe has been applied, the line says
    it, rather than offering to apply it again."""
    out = _select(_ONE_CLAIM, "detect", "coral", applied="a_first.yaml")
    check("VICARIUS filled this form from it." in out["line"],
          f"the line must say the form carries it: {out['line']!r}")
    check("Press Import" not in out["line"],
          f"and must not still be offering it: {out['line']!r}")


def test_two_recipes_claiming_one_pair_apply_nothing_and_name_both():
    """The robustness rule. Two recipes for one task and subject is a choice a
    person makes: the panel applies neither, names both, and never breaks the
    tie by name, time, size or folder order."""
    out = _select(_TWO_CLAIMS, "detect", "coral")
    check(sorted(out["matched"]) == ["a_first.yaml", "z_second.yaml"],
          f"both recipes must match: {out}")
    check(out["selected"] == "",
          f"nothing may be selected while two recipes claim the pair: {out}")
    for name in ('"First by name"', '"Second by name"'):
        check(name in out["line"], f"the line must name {name}: {out['line']!r}")
    check("VICARIUS applied none of them" in out["line"],
          f"the line must say nothing was applied: {out['line']!r}")
    check("more than one recipe claims the same task and subject" in out["line"],
          f"the line must say why: {out['line']!r}")
    check("pick one" in out["line"],
          f"the line must say what the operator does next: {out['line']!r}")


def test_no_match_applies_nothing_and_says_so_plainly():
    out = _select(_TWO_CLAIMS, "segment", "algae")
    check(out["matched"] == [], f"nothing may match: {out}")
    check(out["selected"] == "", f"and nothing may be selected: {out}")
    check(out["line"].startswith("Training task segmentation and subject algae match no "
                                 "recipe in config/presets/."),
          f"the line must name the task and subject and say no recipe matched: "
          f"{out['line']!r}")
    check("this form keeps its own defaults" in out["line"],
          f"and must say what the form holds instead: {out['line']!r}")


# The states below are one Import click away and are the module's everyday
# ones, since segmentation plus coral matches both shipped coral segmentation
# recipes. Until 2026-09-15 the line went on saying VICARIUS had applied
# nothing while the form held an imported recipe's 30-odd values, and the
# adjacent applied-note line said the opposite an inch away.

def test_two_matches_still_name_the_recipe_that_filled_the_form():
    """Two recipes claim the pair, so VICARIUS applies neither, but one of them
    was imported by hand and its values are in the form. The line has to say
    both things, or it contradicts the form it sits under."""
    out = _select(_TWO_CLAIMS, "detect", "coral", applied="a_first.yaml")
    check(out["selected"] == "",
          f"nothing may be selected while two recipes claim the pair: {out}")
    check("VICARIUS applied none of them" in out["line"],
          f"the line must still say VICARIUS applied neither: {out['line']!r}")
    check('This form holds the values of "First by name"' in out["line"],
          f"and must name the recipe whose values are in the form: {out['line']!r}")


def test_no_match_still_names_the_recipe_that_filled_the_form():
    """The Subject control moved off the imported recipe. Nothing matches now,
    but the form is not back on its own defaults, so the line must not say it
    is."""
    out = _select(_TWO_CLAIMS, "segment", "algae", applied="a_first.yaml")
    check(out["matched"] == [], f"nothing may match: {out}")
    check("this form keeps its own defaults" not in out["line"],
          f"the line may not claim untouched defaults while a recipe filled the "
          f"form: {out['line']!r}")
    check("so VICARIUS applied none." in out["line"],
          f"the line must say VICARIUS applied nothing of its own: {out['line']!r}")
    check('This form holds the values of "First by name"' in out["line"],
          f"and must name what the form holds: {out['line']!r}")


def test_a_single_match_names_the_other_recipe_the_form_holds():
    """One recipe matches and a different one filled the form. The line offers
    the match and names the values that are actually there, so Import is a
    described change rather than a surprise."""
    recipes = _TWO_CLAIMS[:1] + [
        {"id": "held.yaml", "name": "Held by hand", "task": "segment",
         "subject": "coral", "classified": True, "description": ""},
    ]
    out = _select(recipes, "detect", "coral", applied="held.yaml")
    check(out["matched"] == ["a_first.yaml"], f"one recipe must match: {out}")
    check("Press Import to fill this form from it." in out["line"],
          f"the match must still be offered: {out['line']!r}")
    check('This form holds the values of "Held by hand"' in out["line"],
          f"and the held recipe must be named: {out['line']!r}")
    check("VICARIUS filled this form from it." not in out["line"],
          f"the line may not claim the match is in the form: {out['line']!r}")


def test_a_recipe_the_folder_no_longer_lists_is_still_named():
    """Adversarial: the form carries a recipe whose file was deleted or renamed
    since it was imported. The panel names the id it has rather than dropping
    the sentence, because a silent blank is the state this whole change removes."""
    out = _select(_TWO_CLAIMS, "detect", "coral", applied="deleted_recipe.yaml")
    check('This form holds the values of "deleted_recipe.yaml"' in out["line"],
          f"the line must name the id it was given: {out['line']!r}")
    check("no longer lists" in out["line"],
          f"and must say the folder does not hold it: {out['line']!r}")


def test_a_form_nothing_filled_claims_no_recipe():
    """The other side of the same rule: with no recipe applied, the line must
    not invent one."""
    for task, subject in (("detect", "coral"), ("detect", "fish"), ("segment", "algae")):
        out = _select(_TWO_CLAIMS, task, subject)
        check("This form holds the values of" not in out["line"],
              f"{task} plus {subject} named a held recipe with none applied: "
              f"{out['line']!r}")


def test_a_recipe_is_never_chosen_by_its_file_name():
    """The defect itself, pinned. A recipe whose name sorts before every other
    wins nothing by it: with the two controls on detector and fish, the
    alphabetically first recipe is not listed at all and the fish recipe is the
    one offered."""
    recipes = [
        {"id": "aaa_sorts_first.yaml", "name": "Sorts first", "task": "detect",
         "subject": "coral", "classified": True, "description": ""},
        {"id": "zzz_sorts_last.yaml", "name": "Sorts last", "task": "detect",
         "subject": "fish", "classified": True, "description": ""},
    ]
    out = _select(recipes, "detect", "fish")
    check(out["matched"] == ["zzz_sorts_last.yaml"],
          f"the recipe written for fish must win, whatever sorts first: {out}")
    check(out["selected"] == "zzz_sorts_last.yaml", f"and must be the selection: {out}")
    check("aaa_sorts_first.yaml" not in out["listed"],
          f"the coral recipe must not even be listed for a fish run: {out['listed']}")
    check(out["hidden"] == ["aaa_sorts_first.yaml"],
          f"it must be counted as written for another subject: {out}")
    check("1 other recipe in that folder is written for another task or subject"
          in out["line"],
          f"the line must account for what it hid: {out['line']!r}")


def test_changing_the_subject_refilters_the_list():
    """One recipe folder, two subjects, two different lists. The controls are
    read at render time, so a change to either re-filters at once."""
    recipes = [
        {"id": "coral.yaml", "name": "Coral", "task": "detect", "subject": "coral",
         "classified": True, "description": ""},
        {"id": "fish.yaml", "name": "Fish", "task": "detect", "subject": "fish",
         "classified": True, "description": ""},
    ]
    coral = _select(recipes, "detect", "coral")
    check(coral["listed"] == ["coral.yaml"] and coral["selected"] == "coral.yaml",
          f"coral must list and select the coral recipe: {coral}")
    fish = _select(recipes, "detect", "fish")
    check(fish["listed"] == ["fish.yaml"] and fish["selected"] == "fish.yaml",
          f"fish must list and select the fish recipe: {fish}")
    # And through the control's own handler, which is what the page calls.
    switched = _select(recipes, "detect", "coral", before=(
        "document.getElementById('s6-subject').value = 'fish';\n"
        "onStep6SubjectChange();\n"
        "sel.children = [];"))
    check(switched["listed"] == ["fish.yaml"],
          f"onStep6SubjectChange must re-filter the list: {switched}")


def test_changing_the_task_refilters_the_list():
    recipes = [
        {"id": "det.yaml", "name": "Detector", "task": "detect", "subject": "coral",
         "classified": True, "description": ""},
        {"id": "seg.yaml", "name": "Segmenter", "task": "segment", "subject": "coral",
         "classified": True, "description": ""},
    ]
    check(_select(recipes, "detect", "coral")["listed"] == ["det.yaml"],
          "the detector task lists the detector recipe")
    check(_select(recipes, "segment", "coral")["listed"] == ["seg.yaml"],
          "the segmentation task lists the segmentation recipe")


def test_a_recipe_for_any_subject_matches_every_subject():
    recipes = [{"id": "any.yaml", "name": "Any subject", "task": "segment",
                "subject": "any", "classified": True, "description": ""}]
    for subject in ("coral", "fish", "algae", "other"):
        out = _select(recipes, "segment", subject)
        check(out["matched"] == ["any.yaml"],
              f"a recipe written for any subject must match {subject}: {out}")
    check(_select(recipes, "detect", "coral")["matched"] == [],
          "a recipe written for any subject still belongs to one task")


def test_the_other_subject_reaches_only_the_any_recipes():
    recipes = [
        {"id": "any.yaml", "name": "Any subject", "task": "segment", "subject": "any",
         "classified": True, "description": ""},
        {"id": "coral.yaml", "name": "Coral", "task": "segment", "subject": "coral",
         "classified": True, "description": ""},
    ]
    out = _select(recipes, "segment", "other")
    check(out["matched"] == ["any.yaml"],
          f"other must reach the any recipe and nothing else: {out}")


def test_an_unclassified_recipe_is_listed_and_never_matched():
    """A recipe that declares too little stays importable by hand, is marked in
    the dropdown, and is never selected on its own."""
    recipes = _ONE_CLAIM + [
        {"id": "plain.yaml", "name": "Plain", "task": "", "subject": "",
         "classified": False, "description": "",
         "warning": "this recipe is unclassified, so VICARIUS never selects it on its own"},
    ]
    out = _select(recipes, "detect", "coral")
    check(out["unclassified"] == ["plain.yaml"], f"it must be sorted as unclassified: {out}")
    check("plain.yaml" in out["listed"], f"and must still be listed: {out['listed']}")
    check(out["matched"] == ["a_first.yaml"], f"and must never match: {out}")
    check(out["selected"] == "a_first.yaml",
          f"the classified recipe is still the one offered: {out}")
    check("Plain (unclassified)" in out["labels"],
          f"the dropdown must say it is unclassified: {out['labels']}")
    check("1 recipe declares no task or subject and is listed as unclassified"
          in out["line"],
          f"the line must account for it: {out['line']!r}")


def test_a_damaged_recipe_is_listed_with_its_error_and_never_matched():
    recipes = _ONE_CLAIM + [
        {"id": "broken.yaml", "task": "", "subject": "", "classified": False,
         "error": "parse error: mapping values are not allowed here"},
    ]
    out = _select(recipes, "detect", "coral")
    check("broken.yaml" in out["listed"], f"a damaged recipe stays visible: {out['listed']}")
    check(any("error:" in label for label in out["labels"]),
          f"and carries its error in the dropdown: {out['labels']}")
    check(out["matched"] == ["a_first.yaml"], f"and is never matched: {out}")


def test_the_dataset_note_is_shown_with_the_selection_line():
    out = _select(_ONE_CLAIM, "detect", "coral",
                  dataset_note="the subject recorded with this project's box dataset "
                               "could not be read from /nowhere/ingest_manifest.json: boom")
    check("could not be read" in out["line"],
          f"a dataset the panel could not read must be said out loud: {out['line']!r}")


def test_a_dataset_subject_only_sets_where_the_control_starts():
    """It fills in for a project with no subject of its own, and never
    overrides a value the saved config restored or the operator picked."""
    def start(before):
        return _drive(
            "const el = document.getElementById('s6-subject');\n"
            "el.dataset = {}; el.value = 'coral';\n"
            f"{before}\n"
            "applyStep6DatasetSubject('fish');\n"
            "({ value: el.value, source: el.dataset.subjectSource || '' })\n")

    fresh = start("")
    check(fresh["value"] == "fish" and fresh["source"] == "dataset",
          f"a project with no subject of its own starts on the dataset's: {fresh}")
    saved = start("el.dataset.subjectSource = 'saved'; el.value = 'coral';")
    check(saved["value"] == "coral",
          f"a subject restored from the saved config wins: {saved}")
    chosen = start("el.dataset.subjectSource = 'operator'; el.value = 'algae';")
    check(chosen["value"] == "algae",
          f"a subject the operator picked wins: {chosen}")
    unknown = _drive(
        "const el = document.getElementById('s6-subject');\n"
        "el.dataset = {}; el.value = 'coral';\n"
        "applyStep6DatasetSubject('reef');\n"
        "({ value: el.value, source: el.dataset.subjectSource || '' })\n")
    check(unknown["value"] == "coral" and unknown["source"] == "",
          f"a subject the control does not offer changes nothing: {unknown}")


def _auto_apply(recipes, task, subject, *, stored_preset_id="", dataset_subject=""):
    """Run the panel's real loader with auto-apply asked for, and report it.

    The whole loader is executed, not read: api() is replaced with a stub that
    answers a fixed presets payload and records every URL it was given, so
    whether a recipe was applied is a fact about what the code fetched rather
    than a fact about how it is written. A rule moved into a helper or behind
    an early return is caught here; a source-text check would not see it.

    recipes: the entries /api/step/6/presets would answer.
    task, subject: the two control values.
    stored_preset_id: the preset marker in the project's saved Step 6 config.
    dataset_subject: the subject the staged dataset recorded, '' for none.
    Returns a dict with every fetched URL, the fetches that imported a recipe,
    the selection, the applied marker and both lines under the picker.
    """
    scenario = (
        "_step6Recipes = null;\n"
        "_step6DatasetNote = '';\n"
        "window._step6PresetAutoApplied = false;\n"
        f"state = {{ steps: {{ '6': {{ config: "
        f"{{ preset_id: {json.dumps(stored_preset_id)} }} }} }} }};\n"
        "const fetched = [];\n"
        "api = async (url) => {\n"
        "  fetched.push(url);\n"
        "  if (url === '/api/step/6/presets') return "
        f"{{ presets: {json.dumps(recipes)}, dataset_subject: "
        f"{json.dumps(dataset_subject)} }};\n"
        "  return { name: 'imported recipe', params: {} };\n"
        "};\n"
        "const sel = document.getElementById('s6-preset-select');\n"
        "sel.children = []; sel.value = ''; sel.dataset = {};\n"
        "document.getElementById('s6-subject').dataset = {};\n"
        f"document.getElementById('s6-task').value = {json.dumps(task)};\n"
        f"document.getElementById('s6-subject').value = {json.dumps(subject)};\n"
        "document.getElementById('s6-preset-applied').textContent = '';\n"
        "refreshStep6Presets({ autoApply: true }).then(() => ({\n"
        "  fetched: fetched,\n"
        "  imported: fetched.filter(u => u.indexOf('/api/step/6/presets/') === 0),\n"
        "  selected: sel.value,\n"
        "  applied: sel.dataset.appliedPresetId || '',\n"
        "  line: document.getElementById('s6-preset-selection').textContent,\n"
        "  appliedNote: document.getElementById('s6-preset-applied').textContent,\n"
        "}))\n")
    return _drive(scenario)


def test_the_auto_apply_is_refused_when_two_recipes_claim_one_pair():
    """The apply half of the ambiguity rule, executed rather than read. Two
    recipes claiming detector plus coral must leave the loader having fetched
    the listing and nothing else: no recipe body is ever asked for, so none can
    reach the form. The single-match case beside it is the positive control,
    which is what proves this test could fail."""
    blocked = _auto_apply(_TWO_CLAIMS, "detect", "coral")
    check(blocked["fetched"] == ["/api/step/6/presets"],
          f"the loader may fetch only the listing while two recipes claim the "
          f"pair, it fetched: {blocked['fetched']}")
    check(blocked["imported"] == [],
          f"no recipe body may be read: {blocked['imported']}")
    check(blocked["selected"] == "" and blocked["applied"] == "",
          f"and nothing may be selected or marked as applied: {blocked}")
    check(blocked["appliedNote"] == "",
          f"and no line may claim a recipe filled the form: "
          f"{blocked['appliedNote']!r}")

    applied = _auto_apply(_ONE_CLAIM, "detect", "coral")
    check(applied["imported"] == ["/api/step/6/presets/a_first.yaml"],
          f"a single match must be fetched and applied, the loader fetched: "
          f"{applied['fetched']}")
    check(applied["applied"] == "a_first.yaml",
          f"and must be marked as the recipe in the form: {applied}")
    check("VICARIUS applied the one recipe written for this run" in applied["appliedNote"],
          f"and must say so in words: {applied['appliedNote']!r}")


def test_the_auto_apply_is_refused_when_no_recipe_matches():
    """Executed. No match means no fetch of any recipe body, so the platform
    defaults in the form are left exactly as they are."""
    out = _auto_apply(_TWO_CLAIMS, "segment", "algae")
    check(out["fetched"] == ["/api/step/6/presets"],
          f"the loader may fetch only the listing with no match: {out['fetched']}")
    check(out["applied"] == "", f"and nothing may be marked as applied: {out}")
    check("this form keeps its own defaults" in out["line"],
          f"and the line must say the form is untouched: {out['line']!r}")


def test_a_saved_project_is_never_auto_applied_over_even_on_a_single_match():
    """Executed. A project whose saved Step 6 config names a recipe was already
    configured, possibly hand-tuned afterwards, so one match does not license a
    re-apply. The panel names that recipe instead."""
    out = _auto_apply(_ONE_CLAIM, "detect", "coral", stored_preset_id="a_first.yaml")
    check(out["imported"] == [],
          f"a saved project may have no recipe applied over it: {out['fetched']}")
    check("saved training settings came from" in out["appliedNote"],
          f"and the panel must name the recipe those values came from: "
          f"{out['appliedNote']!r}")
    check('"First by name"' in out["appliedNote"],
          f"by name: {out['appliedNote']!r}")


def test_the_auto_apply_happens_only_on_a_single_match():
    """Read off the live refreshStep6Presets: the guard must require exactly
    one match. Kept beside the executed checks above, which catch a rule moved
    into a helper, while this one catches a loosened comparison in place."""
    body = _refresh_presets_body()
    guard = [ln for ln in body.splitlines() if "opts.autoApply" in ln and "if (" in ln]
    check(len(guard) == 1,
          f"expected exactly one auto-apply guard in refreshStep6Presets, found {guard}")
    check("sorted.matches.length === 1" in guard[0],
          f"the guard must require exactly one matching recipe, it reads: "
          f"{guard[0].strip()!r}")


def test_a_saved_preset_id_marker_still_blocks_the_auto_apply():
    """A training project saved with preset_id in its Step 6 config must open
    with its own values and must NOT have anything applied over them on a fresh
    page load. If a refactor drops that guard, a recipe overwrites Lauren's
    saved settings every time the panel opens, in silence."""
    body = _refresh_presets_body()
    check("config.preset_id" in body,
          f"refreshStep6Presets must read the saved marker as config.preset_id; it "
          f"does not, so a saved training project is re-defaulted on every open")
    check("!storedPresetId" in body,
          f"the auto-apply guard must skip a project that already carries a preset "
          f"marker: {body}")
    check("saved training settings came from" in body,
          "and the panel must name the recipe those saved values came from, so it can "
          "always answer which recipe is in the form")


def test_nothing_reads_the_old_file_name_flag():
    """The mechanism this replaced is gone, not merely unused: no preset file
    declares it, no code reads it, and no test pins it."""
    key = "default_for_source"
    for path in sorted(os.listdir(_PRESET_DIR)):
        full = os.path.join(_PRESET_DIR, path)
        text = open(full, encoding="utf-8").read()
        check(key not in text, f"{full} still mentions {key}")
    for path in (_JS, _TPL):
        text = open(path, encoding="utf-8").read()
        check(key not in text, f"{path} still mentions {key}")
    # This suite may name the flag only where it proves the flag is gone: the
    # two lines below, and nowhere else, or a later reader would think some
    # check still depends on it.
    own = open(os.path.abspath(__file__), encoding="utf-8").read()
    check(own.count(key) == 2,
          f"this suite mentions {key} {own.count(key)} times; only this check may "
          f"name it")
    app_text = open(os.path.join(_ORCH, "app.py"), encoding="utf-8").read()
    for quoted in (f'"{key}"', f"'{key}'"):
        check(quoted not in app_text,
              f"app.py still reads {quoted} as a key; the flag is meant to be gone, and "
              f"only the docstring line explaining its removal may name it")
    check("default_id" not in app_text,
          "app.py must no longer answer a default_id: the panel chooses")
    check("default_id" not in open(_JS, encoding="utf-8").read(),
          "orchestrator.js must no longer read a default_id")


# ---------------------------------------------------------------------------
# LIVE: the four recipes this module ships
# ---------------------------------------------------------------------------

def test_every_shipped_recipe_declares_what_it_is_for():
    """Pinned by name and pair. A recipe that stops declaring its pair, or
    starts declaring another one, changes which project opens on it, so it
    fails here rather than on the panel."""
    declared = {entry["id"]: (entry["task"], entry["subject"]) for entry in _shipped_recipes()}
    check(declared == SHIPPED_DECLARATIONS,
          f"the shipped recipes declare {declared}, this suite is written against "
          f"{SHIPPED_DECLARATIONS}")
    for entry in _shipped_recipes():
        check(entry["classified"] is True,
              f"every shipped recipe must be classified: {entry}")


def test_no_shipped_recipe_carries_a_banned_dash_or_a_stale_title():
    """House rules, on every recipe file rather than only the newest two. Em
    and en dashes are banned everywhere, code and comments included, and a
    header naming a title the _meta.name below it no longer uses sends a reader
    looking in the picker for a recipe that is not there."""
    for fname in sorted(os.listdir(_PRESET_DIR)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        path = os.path.join(_PRESET_DIR, fname)
        text = open(path, encoding="utf-8").read()
        bad = sorted({line_no + 1 for line_no, line in enumerate(text.splitlines())
                      if "—" in line or "–" in line})
        check(not bad, f"{path} carries an em or en dash on line(s) {bad}")
        name = str(((yaml.safe_load(text) or {}).get("_meta") or {}).get("name") or "")
        if not name:
            continue
        header = text.splitlines()[1] if len(text.splitlines()) > 1 else ""
        check(name in header,
              f"{path} line 2 reads {header!r}, which does not carry the recipe name "
              f"{name!r} the picker shows")


def test_detector_plus_coral_reaches_the_coral_recipe():
    out = _select(_shipped_recipes(), "detect", "coral")
    check(out["matched"] == [CORAL_PRESET_ID],
          f"Detector plus coral must reach {CORAL_PRESET_ID}: {out}")
    check(out["selected"] == CORAL_PRESET_ID, f"and must select it: {out}")
    check(FISH_PRESET_ID in out["hidden"],
          f"the fish detector must be hidden from a coral run: {out}")


def test_detector_plus_fish_reaches_the_fish_recipe():
    out = _select(_shipped_recipes(), "detect", "fish")
    check(out["matched"] == [FISH_PRESET_ID],
          f"Detector plus fish must reach {FISH_PRESET_ID}: {out}")
    check(out["selected"] == FISH_PRESET_ID, f"and must select it: {out}")
    check(CORAL_PRESET_ID in out["hidden"],
          f"the coral detector must be hidden from a fish run: {out}")


def test_segmentation_plus_coral_matches_both_and_applies_neither():
    """Two coral segmentation recipes ship today, so the panel applies neither
    and names both. This is the rule working, not a gap: picking between two
    recipes for one job is Lauren's choice to make."""
    out = _select(_shipped_recipes(), "segment", "coral")
    check(sorted(out["matched"]) == sorted(CORAL_SEG_PRESET_IDS),
          f"both coral segmentation recipes must match: {out}")
    check(out["selected"] == "", f"and nothing may be selected: {out}")
    check("VICARIUS applied none of them" in out["line"],
          f"the line must say nothing was applied: {out['line']!r}")
    for pid in CORAL_SEG_PRESET_IDS:
        name = _entry({"presets": _shipped_recipes()}, pid)["name"]
        check(f'"{name}"' in out["line"],
              f"the line must name {name!r}: {out['line']!r}")


def test_segmentation_plus_fish_reaches_nothing_and_says_so():
    out = _select(_shipped_recipes(), "segment", "fish")
    check(out["matched"] == [], f"no shipped recipe segments fish: {out}")
    check("match no recipe in config/presets/" in out["line"],
          f"and the line must say so: {out['line']!r}")


def test_every_shipped_recipe_key_reaches_the_form():
    """Every parameter of every shipped recipe, not only the two this suite
    reads value by value. A key with no applier in STEP6_PRESET_APPLIERS is
    dropped in silence at import time, which is how 30 of 36 fields once went
    missing from an applied recipe with nothing on the page saying so."""
    keys = _applier_keys()
    missing = {}
    for fname in sorted(os.listdir(_PRESET_DIR)):
        if not fname.endswith((".yaml", ".yml")):
            continue
        with open(os.path.join(_PRESET_DIR, fname), encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        params = (doc.get("params") or {}) if isinstance(doc, dict) else {}
        unknown = sorted(k for k in params if k not in keys)
        if unknown:
            missing[fname] = unknown
    check(not missing,
          f"these recipe keys have no applier in STEP6_PRESET_APPLIERS and would be "
          f"dropped in silence at import: {missing}")
    check(len(SHIPPED_DECLARATIONS) >= _MIN_SHIPPED_PRESETS,
          f"this check is written against the {len(SHIPPED_DECLARATIONS)} shipped "
          f"recipes and must not pass on an empty folder")


def test_the_shipped_folder_is_what_the_route_reads():
    check(os.path.isdir(_PRESET_DIR), f"missing preset folder {_PRESET_DIR}")
    check(appmod.PRESET_DIRS.get("train", "") == _PRESET_DIR,
          f"the route reads {appmod.PRESET_DIRS.get('train', '')!r}, this suite checks "
          f"{_PRESET_DIR!r}")


# ---------------------------------------------------------------------------
# DOC: the folder's README is what another agent reads to ship a recipe
# ---------------------------------------------------------------------------

def test_readme_documents_the_preset_id_spelling():
    """The README told a reader the preset id is the file name with .yaml
    removed. Both routes take the file name with its extension, so a reader who
    followed the README addressed a preset that answers 404."""
    readme = os.path.join(_PRESET_DIR, "README.md")
    check(os.path.isfile(readme), f"missing {readme}")
    text = open(readme, encoding="utf-8").read()
    check("minus `.yaml`" not in text,
          "the README must not say the preset id drops the .yaml extension; both "
          "preset routes take the file name with it")
    with _project_open({}) as client:
        data = _presets(client)
        listed = [p["id"] for p in data["presets"]]
        check(listed and all(i.endswith((".yaml", ".yml")) for i in listed),
              f"the route lists ids with their extension, which is what the README "
              f"must describe: {listed}")


def test_readme_documents_the_two_declarations_and_the_three_outcomes():
    """The folder README is what the next agent reads instead of the route. It
    has to carry the rule as it now is: two declarations, and what happens on
    one match, on more than one, and on none."""
    readme = os.path.join(_PRESET_DIR, "README.md")
    text = open(readme, encoding="utf-8").read()
    for needle in ("_meta.task", "_meta.subject", "task", "subject"):
        check(needle in text, f"the README must document {needle!r}")
    for needle in ("Exactly one recipe matches", "More than one matches", "None matches"):
        check(needle in text, f"the README must spell out the outcome {needle!r}")
    check("no tie-break" in text.lower() or "There is no tie-break" in text,
          "the README must say that a tie is never broken")
    for pid, (task, subject) in SHIPPED_DECLARATIONS.items():
        row = [ln for ln in text.splitlines() if pid in ln and "|" in ln]
        check(len(row) == 1,
              f"the README must carry one table row naming {pid}: {row}")
        check(task in row[0] and subject in row[0],
              f"the row for {pid} must name {task} and {subject}: {row[0]!r}")


def test_readme_documents_where_the_subject_control_starts():
    readme = os.path.join(_PRESET_DIR, "README.md")
    text = open(readme, encoding="utf-8").read()
    for needle in ("ingest_manifest.json", "--subject", "subset_by_class_count.py"):
        check(needle in text,
              f"the README must say how a dataset records its subject ({needle!r})")


# ---------------------------------------------------------------------------
# CORAL PRESET: the recipe Lauren presses play on
# ---------------------------------------------------------------------------

def _coral_doc():
    """The parsed coral preset file.

    Returns the whole YAML document as a dict. Raises AssertionError naming the
    path when the file is not there, because every check below reads it.
    """
    check(os.path.isfile(CORAL_PRESET_PATH), f"missing preset file {CORAL_PRESET_PATH}")
    with open(CORAL_PRESET_PATH, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    check(isinstance(doc, dict),
          f"{CORAL_PRESET_PATH} must parse to a mapping, it parsed to "
          f"{type(doc).__name__}")
    return doc


def _coral_header():
    """The comment block above the _meta section of the coral preset.

    Returns the leading text of the file. Raises AssertionError when the file
    has no _meta section to stop at, since a preset without one loads nothing.
    """
    text = open(CORAL_PRESET_PATH, encoding="utf-8").read()
    check("_meta:" in text, f"{CORAL_PRESET_PATH} has no _meta section")
    return text[:text.index("_meta:")]


def _preset_hue(preset_id):
    """The hsv_h value of a neighbouring preset, by file name."""
    path = os.path.join(_PRESET_DIR, preset_id)
    check(os.path.isfile(path), f"missing neighbouring preset {path}")
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return (doc.get("params") or {}).get("hsv_h")


def test_coral_preset_matches_the_approved_recipe():
    """Every value Lauren approved, key by key. A silent edit to one number
    here is a different detector on the next press of Start Training."""
    doc = _coral_doc()
    params = doc.get("params") or {}
    for key, want in CORAL_RECIPE.items():
        got = params.get(key)
        check(got == want, f"coral preset {key} = {got!r}, the approved recipe is {want!r}")
    check(isinstance(params.get("cos_lr"), bool),
          f"cos_lr must be a YAML boolean, not {type(params.get('cos_lr')).__name__} "
          f"({params.get('cos_lr')!r}): the form applier sets a checkbox from it")
    check(isinstance(params.get("device"), str),
          f"device must stay a string, not {type(params.get('device')).__name__}: "
          f"{params.get('device')!r} names two GPUs, not one number")
    meta = doc.get("_meta") or {}
    check(str(meta.get("name") or "").strip(), "the coral preset needs a _meta.name")
    check(str(meta.get("description") or "").strip(),
          "the coral preset needs a _meta.description for the picker")


def test_coral_preset_ships_no_key_outside_the_recipe():
    """The other direction of the check above: a key the recipe does not name
    would be applied to the Train form without anyone having approved it."""
    params = _coral_doc().get("params") or {}
    extra = sorted(k for k in params if k not in CORAL_RECIPE)
    check(not extra,
          f"the coral preset ships keys the approved recipe does not name, so they "
          f"reach the Train form unreviewed: {extra}")
    # min_samples belongs to the outline splitter and pin_split to the refine
    # loop; neither control is shown for a box dataset, so a value here would
    # write a knob the panel does not display.
    for hidden_key in ("min_samples", "pin_split"):
        check(hidden_key not in params,
              f"{hidden_key} must not ship in a box-dataset preset")


def test_coral_preset_declares_the_pair_it_is_written_for():
    """What the Train panel matches it on. Declaring detect and coral is what
    puts this recipe in front of a Coral Intel run, and declaring nothing else
    is what keeps it away from the fish detector and from the two coral
    SEGMENTATION recipes, which train a different task."""
    meta = _coral_doc().get("_meta") or {}
    check((meta.get("task"), meta.get("subject")) == ("detect", "coral"),
          f"the coral preset must declare task detect and subject coral, it declares "
          f"{meta.get('task')!r} and {meta.get('subject')!r}")
    check("default" not in meta,
          f"the coral preset must carry no default flag: the flag that once decided "
          f"this is gone, and a leftover would read as a second rule: {meta}")


def test_every_coral_preset_key_has_a_form_applier():
    """A key with no applier in STEP6_PRESET_APPLIERS is dropped in silence at
    import time. That is how 30 of 36 fields once went missing from an applied
    preset with nothing on the page saying so."""
    keys = _applier_keys()
    params = _coral_doc().get("params") or {}
    unknown = sorted(k for k in params if k not in keys)
    check(not unknown,
          f"these coral preset keys have no applier in STEP6_PRESET_APPLIERS and would "
          f"be dropped in silence at import: {unknown}")


def test_coral_preset_hue_sits_between_the_coral_and_fish_presets():
    """hsv_h 0.1 is a deliberate middle setting, and the two ends are files in
    this same folder. Reading them here means a change at either end shows up
    as a failure rather than as a caution that quietly stopped being true."""
    coral_hue = (_coral_doc().get("params") or {}).get("hsv_h")
    fish_hue = _preset_hue(FISH_PRESET_ID)
    check(coral_hue == 0.1, f"the approved hue jitter is 0.1, the preset ships {coral_hue!r}")
    for pid in CORAL_SEG_PRESET_IDS:
        seg_hue = _preset_hue(pid)
        check(seg_hue is not None and seg_hue < coral_hue,
              f"the header calls 0.1 higher than the coral segmentation presets, but "
              f"{pid} jitters hue by {seg_hue!r}")
    check(fish_hue is not None and coral_hue < fish_hue,
          f"the header calls 0.1 well below the fish detector, but {FISH_PRESET_ID} "
          f"jitters hue by {fish_hue!r}")


def test_coral_preset_header_records_why_and_both_cautions():
    """The header comment is where a reader of the file meets the reasoning,
    so it carries what the recipe is for and the two cautions that come with
    it: an unablated hue setting, and a copy-paste probability held at zero
    because ultralytics skips copy-paste on labels with no polygon."""
    head = _coral_header()
    for needle in ("Coral Intel", "256", "yolo11l.pt", "detect"):
        check(needle in head, f"the coral preset header must mention {needle!r}")
    for needle in ("0.1", "0.03", "0.3", "ablat"):
        check(needle in head,
              f"the hue caution must mention {needle!r}: 0.1 sits between the coral "
              f"segmentation presets' 0.03 and the fish detector's 0.3, untested")
    for needle in ("copy_paste", "polygon"):
        check(needle in head,
              f"the copy-paste caution must mention {needle!r}: ultralytics skips the "
              f"augmentation entirely when the labels carry no polygon")


def test_coral_preset_text_is_plain_ascii():
    """The platform prose rules ban em dashes, en dashes and emoji, and a
    preset header has no reason to leave ASCII. One check covers all three."""
    text = open(CORAL_PRESET_PATH, encoding="utf-8").read()
    offenders = sorted({ch for ch in text if ord(ch) > 127})
    check(not offenders,
          f"{CORAL_PRESET_PATH} must be plain ASCII; it carries "
          f"{[(c, hex(ord(c))) for c in offenders]}")


def test_coral_preset_base_weights_are_an_option_in_the_model_picker():
    """setVal on the model select silently blanks the field when the value is
    not one of its options, so the preset's base weights must be listed."""
    tpl = open(_TPL, encoding="utf-8").read()
    model = (_coral_doc().get("params") or {}).get("model")
    select_at = tpl.index('id="s6-model"')
    select = tpl[select_at:tpl.index("</select>", select_at)]
    check(f'value="{model}"' in select,
          f"the coral preset names base weights {model!r} that the Train model picker "
          f"does not offer")


def test_coral_preset_base_weights_are_staged_on_this_box():
    """Lab DNS is unreliable, so a recipe that names weights nobody staged
    turns the first press of Start Training into a download that can hang. The
    training process runs in WORKING_DIRS[6], which is where a bare checkpoint
    name resolves."""
    model = (_coral_doc().get("params") or {}).get("model")
    check(os.path.basename(CORAL_MODEL_STAGED_AT) == model,
          f"this check is pinned to {os.path.basename(CORAL_MODEL_STAGED_AT)!r} but the "
          f"preset names {model!r}")
    check(os.path.isfile(CORAL_MODEL_STAGED_AT),
          f"the coral preset trains from {model!r}, which is not staged at "
          f"{CORAL_MODEL_STAGED_AT}; the first run would try to download it")


def test_coral_preset_parses_through_the_route():
    with _project_open(_BBOX_STATE) as client:
        r = client.get(f"/api/step/6/presets/{CORAL_PRESET_ID}")
        check(r.status_code == 200, f"fetching the coral preset answered {r.status_code}")
        body = r.get_json() or {}
        check("error" not in body, f"the coral preset must parse: {body.get('error')!r}")
        params = body.get("params") or {}
        check(params.get("hsv_h") == 0.1 and params.get("epochs") == 600,
              f"the fetched params must carry the recipe: {params}")


def _number_field_ranges():
    """The min and max attributes of every numeric Step 6 input.

    Returns a dict of element id to (min, max), each a float or None when the
    attribute is absent. Raises AssertionError when the template yields too few
    fields to be plausible, since an empty dict would pass every caller.
    """
    tpl = open(_TPL, encoding="utf-8").read()
    ranges = {}
    for element_id, attrs in re.findall(
            r'<input type="number" id="(s6-[A-Za-z0-9-]+)"(.*?)>', tpl, re.S):
        def _attr(name):
            m = re.search(rf'\b{name}="(-?[0-9.]+)"', attrs)
            return float(m.group(1)) if m else None
        ranges[element_id] = (_attr("min"), _attr("max"))
    check(len(ranges) > _MIN_APPLIER_KEYS,
          f"parsed too few numeric Step 6 fields from {_TPL} ({sorted(ranges)})")
    return ranges


def test_every_coral_value_is_inside_its_form_fields_range():
    """Step 6 reads its fields with parseFloat and calls no validity check
    (static/orchestrator.js, collectConfig step 6), so a preset value outside a
    field's own min or max is handed to ultralytics exactly as written. The
    range the panel documents is the range the recipe has to stay inside."""
    ranges = _number_field_ranges()
    params = _coral_doc().get("params") or {}
    appliers = _applier_bodies()
    checked = 0
    for key, value in params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        field_ids = [i for i in _field_ids_in(appliers.get(key, "")) if i in ranges]
        if len(field_ids) != 1:
            continue
        low, high = ranges[field_ids[0]]
        check(low is None or value >= low,
              f"{key} = {value!r} is below the minimum #{field_ids[0]} documents ({low})")
        check(high is None or value <= high,
              f"{key} = {value!r} is above the maximum #{field_ids[0]} documents ({high})")
        checked += 1
    check(checked > _MIN_APPLIER_KEYS,
          f"only {checked} coral values could be matched to a numeric field; the "
          f"applier map or the template ids moved")


def _refresh_presets_body():
    """The text of refreshStep6Presets in static/orchestrator.js.

    Returns the function source. Raises AssertionError when it cannot be found
    or bounded, because the checks below would otherwise pass on an empty
    string.
    """
    src = _js_source()
    marker = "async function refreshStep6Presets("
    start = src.find(marker)
    check(start >= 0, f"refreshStep6Presets not found in {_JS}")
    end = src.find("\n}\n", start)
    check(end > start, "refreshStep6Presets has no closing brace on its own line")
    return src[start:end]


if __name__ == "__main__":
    print("TASK 3: the training recipes, and how the Train panel chooses one\n")
    print("PRESET FILE:")
    run(test_shipped_preset_matches_run3)
    run(test_run3_args_yaml_records_every_recipe_key_the_note_claims)
    run(test_every_preset_key_has_a_form_applier)
    run(test_preset_header_records_the_ablation)
    run(test_preset_base_weights_are_an_option_in_the_model_picker)
    run(test_shipped_preset_parses_through_the_route)
    print("FORM APPLIERS:")
    run(test_every_applier_resolves_to_a_live_helper)
    run(test_every_applier_writes_a_field_that_exists)
    run(test_the_device_applier_drives_the_step_6_device_controls)
    run(test_the_device_applier_falls_back_to_the_train_panel_default)
    print("WHAT THE ROUTE REPORTS:")
    run(test_every_entry_carries_task_subject_and_classified)
    run(test_the_removed_builtin_preset_ids_answer_404)
    run(test_the_route_chooses_no_recipe_at_all)
    run(test_declarations_are_read_case_and_space_insensitively)
    run(test_a_recipe_missing_either_key_is_unclassified_and_says_why)
    run(test_a_value_outside_the_vocabulary_is_named_in_the_warning)
    run(test_two_recipes_may_declare_the_same_pair_and_both_are_reported)
    run(test_an_empty_preset_folder_lists_nothing)
    run(test_a_non_yaml_file_in_the_folder_is_skipped)
    print("ADVERSARIAL:")
    run(test_unparsable_preset_is_reported_and_stays_unclassified)
    run(test_a_preset_that_is_not_a_mapping_is_reported_not_fatal)
    run(test_a_scalar_meta_section_is_survivable_in_both_routes)
    run(test_a_non_mapping_params_section_is_reported_not_applied)
    print("THE DATASET'S OWN SUBJECT:")
    run(test_the_route_reports_the_subject_a_box_dataset_recorded)
    run(test_a_project_with_no_staged_dataset_reports_no_subject)
    run(test_a_dataset_subject_outside_the_vocabulary_is_ignored)
    run(test_a_damaged_dataset_manifest_says_so_instead_of_failing)
    run(test_the_dataset_subject_vocabulary_is_the_stagers_own)
    print("THE PANEL'S SELECTION RULE:")
    run(test_one_match_is_offered_and_the_line_names_the_task_and_subject)
    run(test_the_line_says_so_when_the_form_was_filled_from_the_recipe)
    run(test_two_recipes_claiming_one_pair_apply_nothing_and_name_both)
    run(test_no_match_applies_nothing_and_says_so_plainly)
    run(test_two_matches_still_name_the_recipe_that_filled_the_form)
    run(test_no_match_still_names_the_recipe_that_filled_the_form)
    run(test_a_single_match_names_the_other_recipe_the_form_holds)
    run(test_a_recipe_the_folder_no_longer_lists_is_still_named)
    run(test_a_form_nothing_filled_claims_no_recipe)
    run(test_a_recipe_is_never_chosen_by_its_file_name)
    run(test_changing_the_subject_refilters_the_list)
    run(test_changing_the_task_refilters_the_list)
    run(test_a_recipe_for_any_subject_matches_every_subject)
    run(test_the_other_subject_reaches_only_the_any_recipes)
    run(test_an_unclassified_recipe_is_listed_and_never_matched)
    run(test_a_damaged_recipe_is_listed_with_its_error_and_never_matched)
    run(test_the_dataset_note_is_shown_with_the_selection_line)
    run(test_a_dataset_subject_only_sets_where_the_control_starts)
    run(test_the_auto_apply_is_refused_when_two_recipes_claim_one_pair)
    run(test_the_auto_apply_is_refused_when_no_recipe_matches)
    run(test_a_saved_project_is_never_auto_applied_over_even_on_a_single_match)
    run(test_the_auto_apply_happens_only_on_a_single_match)
    run(test_a_saved_preset_id_marker_still_blocks_the_auto_apply)
    run(test_nothing_reads_the_old_file_name_flag)
    print("CORAL PRESET FILE:")
    run(test_coral_preset_matches_the_approved_recipe)
    run(test_coral_preset_ships_no_key_outside_the_recipe)
    run(test_coral_preset_declares_the_pair_it_is_written_for)
    run(test_every_coral_preset_key_has_a_form_applier)
    run(test_coral_preset_hue_sits_between_the_coral_and_fish_presets)
    run(test_coral_preset_header_records_why_and_both_cautions)
    run(test_coral_preset_text_is_plain_ascii)
    run(test_coral_preset_base_weights_are_an_option_in_the_model_picker)
    run(test_coral_preset_base_weights_are_staged_on_this_box)
    run(test_coral_preset_parses_through_the_route)
    run(test_every_coral_value_is_inside_its_form_fields_range)
    print("THE SHIPPED FOLDER:")
    run(test_every_shipped_recipe_declares_what_it_is_for)
    run(test_no_shipped_recipe_carries_a_banned_dash_or_a_stale_title)
    run(test_detector_plus_coral_reaches_the_coral_recipe)
    run(test_detector_plus_fish_reaches_the_fish_recipe)
    run(test_segmentation_plus_coral_matches_both_and_applies_neither)
    run(test_segmentation_plus_fish_reaches_nothing_and_says_so)
    run(test_every_shipped_recipe_key_reaches_the_form)
    run(test_the_shipped_folder_is_what_the_route_reads)
    print("PRESET FOLDER README:")
    run(test_readme_documents_the_preset_id_spelling)
    run(test_readme_documents_the_two_declarations_and_the_three_outcomes)
    run(test_readme_documents_where_the_subject_control_starts)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
