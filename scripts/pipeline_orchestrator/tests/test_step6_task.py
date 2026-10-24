"""Task N2: the Step 6 "Training task" selector (segmentation or detector).

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_task.py

Segmentation is the default everywhere; the detector lives inside the Train
panel's existing advanced disclosure and is never styled as an accent.

Covers:
  UI       the control sits INSIDE the advanced details block, offers exactly
           two options with segment preselected, carries a tooltip that
           explains both, and wears no accent class or time-impact badge.
  JS       collectConfig(6) sends `task`, populateConfigs restores it, and the
           default is detect only for a project whose project.json
           training_source.task_default is "detect".
  SUBJECT  the Subject control sits in the same form row as the task control,
           offers coral, fish, algae and other with coral preselected, carries
           a rule-compliant tooltip, is saved and restored like every other
           control, and re-filters the training recipe list whenever it or the
           task changes. The two together are what choose a recipe.
  ENV      _run_step6 forwards TCRMP_STEP6_TASK, and every other knob
           (presets, augmentation, epochs, lr0, patience, freeze, device,
           batch, imgsz) reaches the engine identically for both tasks.
  MODEL    an empty model follows the task (yolo11m-seg.pt / yolo11m.pt), an
           explicit pick is respected, and segmentation weights with
           task=detect answer a pre-launch 400 carrying train_wrapper's own
           fail-fast wording rather than dying mid-launch.
  STEP 7   evaluating a detection run answers with a plain warning about the
           blank mask columns instead of crashing or looking broken.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level and runner.run_cli_stage is replaced per test.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402

_TPL = os.path.join(_ORCH, "templates", "index.html")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_TRAIN_WRAPPER = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "src", "train_wrapper.py")


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
# fixture
# ---------------------------------------------------------------------------

def _make_project(tmp, extra_state=None):
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    export_dir = os.path.join(tmp, pm.STEP_DIRS["step4test"])
    os.makedirs(os.path.join(export_dir, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "all_labels"), exist_ok=True)
    with open(os.path.join(export_dir, "data.yaml"), "w") as f:
        f.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")
    state = {
        "id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
        "created_at": "x", "updated_at": "x", "current_step": 6, "steps": steps,
    }
    state.update(extra_state or {})
    return state


def _client(extra_state=None):
    tmp = tempfile.mkdtemp(prefix="step6_task_")
    state = _make_project(tmp, extra_state)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    return client, tmp, state


def _close(tmp):
    appmod.current_project = None
    shutil.rmtree(tmp, ignore_errors=True)


def _launch(client, config):
    """POST the config, run step 6, return (response json, cmd, env)."""
    captured = {}
    real = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env_extra or {})
        return {"pid": 4242}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/6", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/6/run")
        return r, captured.get("cmd") or [], captured.get("env") or {}
    finally:
        appmod.runner.run_cli_stage = real


# ---------------------------------------------------------------------------
# UI: the control lives inside the advanced disclosure
# ---------------------------------------------------------------------------

def test_task_select_is_inside_advanced():
    tpl = open(_TPL).read()
    check(tpl.count('id="s6-task"') == 1, "exactly one Training task control")

    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    # The panel's advanced disclosure: the <details> whose summary opens the
    # augmentation / LR / loss-weight settings. Anchor on the SUMMARY text, not
    # on the "Advanced hyperparameters" HTML comment that precedes it: the
    # comment sits outside the tag, so anchoring there walks back to the
    # previous details (the DDP explainer) and computes the wrong window.
    # The bounds are then computed by matching nested details tags from the
    # opening tag, so the window does not depend on where the control happens
    # to sit: a control moved OUT lands outside [open_at, close_at) and fails.
    adv_at = panel.index("Advanced hyperparameters (augmentation")
    open_at = panel.rindex("<details", 0, adv_at)
    depth, cursor, close_at = 0, open_at, -1
    while cursor < len(panel):
        nxt_open = panel.find("<details", cursor + 1)
        nxt_close = panel.find("</details>", cursor + 1)
        if nxt_close < 0:
            break
        if 0 <= nxt_open < nxt_close:
            depth += 1
            cursor = nxt_open
            continue
        if depth == 0:
            close_at = nxt_close
            break
        depth -= 1
        cursor = nxt_close
    check(close_at > open_at,
          "the advanced details block must have a matching close tag")
    task_at = panel.index('id="s6-task"')
    check(open_at < task_at < close_at,
          f"the Training task control must sit inside the advanced details block "
          f"(block {open_at}..{close_at}, control at {task_at})")

    tag_start = panel.rindex("<select", 0, task_at)
    tag = panel[tag_start:panel.index(">", task_at) + 1]
    check("title=" in tag, f"the Training task control needs a tooltip (got {tag!r})")
    tip = tag[tag.index("title=") + 7:]
    tip = tip[:tip.index('"')]
    for word in ("polygon", "box"):
        check(word in tip.lower(),
              f"the tooltip must explain what each task trains on (missing {word!r} in {tip!r})")

    body = panel[task_at:close_at]
    check('value="segment"' in body and 'value="detect"' in body,
          "the control offers segmentation and detector")
    check(re.search(r'value="segment"[^>]*selected', body),
          "segmentation is the preselected option")
    check("Segmentation (default)" in body, "the segment option is labelled as the default")
    check("Detector (boxes only)" in body, "the detect option says it is boxes only")

    # Non-prominent: no accent button class and no time-impact star badge on
    # the control, its label, or the card that holds them. The card runs from
    # the form-row that opens it to the start of the next settings card.
    block_start = panel.rindex('<div class="form-row"', 0, task_at)
    next_card = panel.find('padding:12px;background:var(--surface-light)', task_at)
    block = panel[block_start:next_card if next_card > 0 else close_at]
    check('id="s6-task"' in block, "the slice really covers the control")
    check("btn-magenta" not in block, "the task control must not wear the accent button class")
    check("★" not in block, f"the task control must not carry a time-impact badge (got {block!r})")


def test_detect_weights_are_reachable_from_the_model_picker():
    tpl = open(_TPL).read()
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check('id="s6-model-detect-optgroup"' in panel,
          "the base model picker needs a detection group for a detector run")
    grp = panel[panel.index('id="s6-model-detect-optgroup"'):]
    grp = grp[:grp.index("</optgroup>")]
    check('value="yolo11m.pt"' in grp, "the detection group offers yolo11m.pt")
    check("hidden" in panel[panel.index('id="s6-model-detect-optgroup"') - 120:
                            panel.index('id="s6-model-detect-optgroup"') + 120],
          "the detection group is hidden until the task is set to detector")


# ---------------------------------------------------------------------------
# JS round trip
# ---------------------------------------------------------------------------

def test_js_round_trip():
    js = open(_JS).read()
    collect = js[js.index("function collectConfig"):js.index("} else if (step === 7)")]
    check("cfg.task" in collect, "collectConfig(6) must send the chosen task")
    check("s6-task" in collect, "collectConfig(6) must read #s6-task")

    pop = js[js.index("function populateConfigs"):js.index("// Step 7: evaluation")]
    check("s6-task" in pop, "populateConfigs must restore the saved task")
    check("defaultStep6Task" in pop or "s6.task" in pop,
          "populateConfigs must fall back to the project's own default task")

    check("function defaultStep6Task" in js,
          "orchestrator.js must define defaultStep6Task")
    dflt = js[js.index("function defaultStep6Task"):]
    dflt = dflt[:dflt.index("\n}") + 2]
    check("task_default" in dflt,
          "the default must read project.json training_source.task_default")
    check("'detect'" in dflt and "'segment'" in dflt,
          "the default resolves to detect or segment")
    check("training_source" in dflt,
          "the default keys off the training_source block, not the project type")

    check("function onStep6TaskChange" in js,
          "changing the task must swap the base-model catalog to match")
    swap = js[js.index("function onStep6TaskChange"):]
    swap = swap[:swap.index("\n}") + 2]
    check("applyStep6TaskToModelPicker" in swap,
          "changing the task must re-apply the model picker for that task")
    picker = js[js.index("function applyStep6TaskToModelPicker"):]
    picker = picker[:picker.index("\n}") + 2]
    check("STEP6_DETECT_OPTGROUP" in picker or "s6-model-detect-optgroup" in picker,
          "the detection model group is shown only for a detector run")
    check("hidden" in picker,
          "the non-matching catalog is hidden rather than left pickable")
    check("s6-model-detect-optgroup" in js,
          "orchestrator.js must know the detection optgroup id")


# ---------------------------------------------------------------------------
# preset marker: a deliberately-configured project must not be clobbered by
# the Step 6 default-preset auto-apply on a later page load (2026-09-02)
# ---------------------------------------------------------------------------

def test_preset_import_writes_the_marker():
    js = open(_JS).read()
    imp = js[js.index("async function importStep6Preset"):]
    imp = imp[:imp.index("\n// ── Step 7 preset loader")]
    check("sel.dataset.appliedPresetId = sel.value" in imp,
          "importStep6Preset must stamp the applied preset id onto the select's "
          "dataset (auto and manual imports share this one call)")
    # It must run for BOTH the auto path and the manual path, i.e. before the
    # opts.auto-gated status-line branch, not nested inside it. Anchored on the
    # word the branch turns on rather than on the shape of the ternary, which
    # was written over two lines on 2026-09-15.
    stamp_at = imp.index("sel.dataset.appliedPresetId = sel.value")
    auto_branch_at = imp.index("const prefix = opts.auto")
    check(stamp_at < auto_branch_at,
          "the marker must be written unconditionally, not only for a manual import")


def test_collect_config_saves_the_marker():
    js = open(_JS).read()
    collect = js[js.index("function collectConfig"):js.index("} else if (step === 7)")]
    check("cfg.preset_id" in collect,
          "collectConfig(6) must send preset_id so it round-trips to the server")
    check("appliedPresetId" in collect,
          "collectConfig(6) must read the marker importStep6Preset/populateConfigs wrote, "
          "not the select's raw current value (that would mark an untouched project the "
          "moment its dropdown happens to show a preset name)")


def test_populate_configs_seeds_the_marker_from_the_stored_config():
    js = open(_JS).read()
    pop = js[js.index("function populateConfigs"):js.index("// Step 7: evaluation")]
    check("s6.preset_id" in pop,
          "populateConfigs must read preset_id back out of the stored Step 6 config")
    check("appliedPresetId" in pop,
          "populateConfigs must seed the select's dataset from the stored preset_id "
          "so a plain Run (no re-import this session) still saves the existing marker "
          "forward instead of blanking it")


def test_auto_apply_is_skipped_when_the_stored_config_already_has_a_preset():
    js = open(_JS).read()
    fn = js[js.index("async function refreshStep6Presets"):]
    fn = fn[:fn.index("\nasync function importStep6Preset")]
    check("preset_id" in fn,
          "refreshStep6Presets must read the stored config's preset_id")
    check("storedPresetId" in fn,
          "refreshStep6Presets must name the stored marker distinctly from the "
          "dropdown's own current value (prev)")
    auto_if = fn[fn.index("if (opts.autoApply"):]
    auto_if = auto_if[:auto_if.index("{")]
    check("storedPresetId" in auto_if,
          f"the auto-apply condition itself must check storedPresetId, not just "
          f"read it elsewhere (got condition {auto_if!r})")
    check("!storedPresetId" in auto_if.replace(" ", ""),
          f"a truthy stored marker must SKIP auto-apply, not require it "
          f"(got condition {auto_if!r})")


def test_preset_id_round_trips_through_the_config_put():
    """Server side: /api/project/config/6 is a blind dict.update (project_config
    in app.py), so an unrecognized preset_id key must survive untouched and
    _run_step6 must never need to look at it to launch a training run."""
    client, tmp, _state = _client()
    try:
        r = client.put("/api/project/config/6",
                        json={"run_name": "marker_check", "preset_id": "tuned_v1_regularized"})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        check(appmod.current_project["steps"]["6"]["config"].get("preset_id") == "tuned_v1_regularized",
              "the config PUT must not strip an unrecognized preset_id key")
        r, cmd, _env = _launch(client, {"run_name": "marker_check", "preset_id": "tuned_v1_regularized",
                                        "model": "", "device": "0", "batch": 8})
        check(r.status_code == 200, f"a stored preset_id must not block a launch: {r.get_json()}")
        check(bool(cmd), "the run must still launch with preset_id present in the config")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# env composition + knob parity
# ---------------------------------------------------------------------------

_KNOBS = {
    "TCRMP_STEP6_BATCH": ("batch", 8),
    "TCRMP_STEP6_PATIENCE": ("patience", 30),
    "TCRMP_STEP6_LR0": ("lr0", 0.001),
    "TCRMP_STEP6_FREEZE": ("freeze", 10),
    "TCRMP_STEP6_HSV_H": ("hsv_h", 0.05),
    "TCRMP_STEP6_FLIPLR": ("fliplr", 0.25),
    "TCRMP_STEP6_MOSAIC": ("mosaic", 0.3),
    "TCRMP_STEP6_COPY_PASTE": ("copy_paste", 0.2),
    "TCRMP_STEP6_OPTIMIZER": ("optimizer", "AdamW"),
    "TCRMP_STEP6_DEVICE": ("device", "0"),
}


def _knob_config(task, model):
    cfg = {"run_name": "knobs_" + task, "task": task, "model": model,
           "epochs": 7, "imgsz": 320}
    for _env, (key, value) in _KNOBS.items():
        cfg[key] = value
    return cfg


def test_env_carries_the_task_and_every_other_knob_for_both():
    envs = {}
    cmds = {}
    for task, model in (("segment", "yolo11m-seg.pt"), ("detect", "yolo11m.pt")):
        client, tmp, _state = _client()
        try:
            r, cmd, env = _launch(client, _knob_config(task, model))
            check(r.status_code == 200, f"{task} run refused: {r.get_json()}")
            envs[task] = env
            cmds[task] = cmd
        finally:
            _close(tmp)

    for task in ("segment", "detect"):
        check(envs[task].get("TCRMP_STEP6_TASK") == task,
              f"TCRMP_STEP6_TASK must be {task!r} (got {envs[task].get('TCRMP_STEP6_TASK')!r})")
        for env_key, (_cfg_key, value) in _KNOBS.items():
            check(envs[task].get(env_key) == str(value),
                  f"{task}: {env_key} must be {value!r} (got {envs[task].get(env_key)!r})")

    # Parity: the two envs differ ONLY in the task. The engine guarantees every
    # other knob is task-agnostic, so anything else drifting here is a bug.
    diff = {k for k in set(envs["segment"]) | set(envs["detect"])
            if envs["segment"].get(k) != envs["detect"].get(k)}
    check(diff == {"TCRMP_STEP6_TASK"},
          f"only the task may differ between a segment and a detect launch (got {sorted(diff)})")

    # And the positional arguments differ only in the base weights.
    seg_cmd, det_cmd = list(cmds["segment"]), list(cmds["detect"])
    check(len(seg_cmd) == len(det_cmd) == 8, f"both launches take 8 arguments (got {seg_cmd})")
    check(seg_cmd[7] == "yolo11m-seg.pt" and det_cmd[7] == "yolo11m.pt",
          f"the base weights follow the task (got {seg_cmd[7]!r} / {det_cmd[7]!r})")
    check(seg_cmd[1] == det_cmd[1] == "all" and seg_cmd[5:7] == det_cmd[5:7] == ["7", "320"],
          f"epochs and image size pass through unchanged (got {seg_cmd} / {det_cmd})")


def test_blank_model_follows_the_task():
    for task, want in (("segment", "yolo11m-seg.pt"), ("detect", "yolo11m.pt")):
        client, tmp, _state = _client()
        try:
            r, cmd, env = _launch(client, {"run_name": "blank_" + task, "task": task,
                                           "model": "", "device": "0", "batch": 8})
            check(r.status_code == 200, f"{task} run refused: {r.get_json()}")
            check(cmd[7] == want,
                  f"an empty model must default to {want!r} for task {task} (got {cmd[7]!r})")
            check(env.get("TCRMP_STEP6_TASK") == task, f"task {task} forwarded")
        finally:
            _close(tmp)


def test_missing_task_defaults_to_segmentation():
    client, tmp, _state = _client()
    try:
        r, cmd, env = _launch(client, {"run_name": "legacy", "model": "", "device": "0"})
        check(r.status_code == 200, f"run refused: {r.get_json()}")
        check(env.get("TCRMP_STEP6_TASK") == "segment",
              f"a config with no task trains a segmentation model (got {env!r})")
        check(cmd[7] == "yolo11m-seg.pt", f"and takes the seg weights (got {cmd[7]!r})")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# model / task conflict
# ---------------------------------------------------------------------------

def test_seg_weights_with_detect_task_is_a_pre_launch_400():
    client, tmp, _state = _client()
    launched = []
    real = appmod.runner.run_cli_stage
    appmod.runner.run_cli_stage = lambda *a, **k: launched.append(a) or {"pid": 1}
    try:
        client.put("/api/project/config/6", json={
            "run_name": "conflict", "task": "detect", "model": "yolo11m-seg.pt",
            "device": "0", "batch": 8})
        r = client.post("/api/step/6/run")
        body = r.get_json() or {}
        check(r.status_code == 400, f"a seg model with task=detect must be 400 (got {r.status_code})")
        check(not launched, "nothing may be launched for a refused pairing")
        err = str(body.get("error") or "")
        check("cannot train from segmentation weights" in err,
              f"the refusal must carry the engine's own wording (got {err!r})")
        check("yolo11m-seg.pt" in err, f"the refusal names the weights (got {err!r})")

        # The wording is the engine's, not a paraphrase: pin it against
        # train_wrapper.check_model_task, the fail-fast this replaces.
        wrapper = open(_TRAIN_WRAPPER).read()
        check("cannot train from segmentation weights" in wrapper,
              "train_wrapper.py must still be the source of this wording")
        check("Pick a detection base model" in wrapper and "Pick a detection base model" in err,
              f"the way out must match train_wrapper's (got {err!r})")
    finally:
        appmod.runner.run_cli_stage = real
        _close(tmp)


def test_seg_weights_with_segment_task_still_runs():
    client, tmp, _state = _client()
    try:
        r, cmd, _env = _launch(client, {"run_name": "ok", "task": "segment",
                                        "model": "yolo11m-seg.pt", "device": "0", "batch": 8})
        check(r.status_code == 200, f"the ordinary pairing must still run: {r.get_json()}")
        check(cmd[7] == "yolo11m-seg.pt", f"the explicit pick is respected (got {cmd[7]!r})")
    finally:
        _close(tmp)


def test_explicit_detect_weights_are_respected():
    client, tmp, _state = _client()
    try:
        r, cmd, _env = _launch(client, {"run_name": "big", "task": "detect",
                                        "model": "yolo11x.pt", "device": "0", "batch": 8})
        check(r.status_code == 200, f"an explicit detection pick must run: {r.get_json()}")
        check(cmd[7] == "yolo11x.pt", f"the explicit pick is respected (got {cmd[7]!r})")
    finally:
        _close(tmp)


def test_unknown_task_is_refused_before_launch():
    client, tmp, _state = _client()
    launched = []
    real = appmod.runner.run_cli_stage
    appmod.runner.run_cli_stage = lambda *a, **k: launched.append(a) or {"pid": 1}
    try:
        client.put("/api/project/config/6", json={"run_name": "bad", "task": "sideways",
                                                  "device": "0", "batch": 8})
        r = client.post("/api/step/6/run")
        check(r.status_code == 400, f"an unknown task must be 400 (got {r.status_code})")
        check(not launched, "nothing may be launched for an unknown task")
        err = str((r.get_json() or {}).get("error") or "")
        check("segment" in err and "detect" in err,
              f"the refusal names the two tasks (got {err!r})")
    finally:
        appmod.runner.run_cli_stage = real
        _close(tmp)


# ---------------------------------------------------------------------------
# step 7 on a detection run
# ---------------------------------------------------------------------------

def test_step7_warns_plainly_for_a_detection_run():
    client, tmp, state = _client()
    captured = {}
    real = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        return {"pid": 7}

    appmod.runner.run_cli_stage = _fake
    try:
        step6_dir = state["steps"]["6"]["dir"]
        os.makedirs(os.path.join(step6_dir, "dataset"), exist_ok=True)
        run_dir = os.path.join(step6_dir, "runs", "det_run")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            f.write("task: detect\nimgsz: 256\n")
        r = client.put("/api/project/config/7", json={"run_dir": run_dir, "imgsz": 256})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/7/run")
        body = r.get_json() or {}
        check(r.status_code == 200, f"evaluating a detection run must not crash: {body}")
        check(captured.get("cmd"), "the evaluation still launches for a detection run")
        warnings = " ".join(body.get("warnings") or [])
        check("detector" in warnings.lower() or "box" in warnings.lower(),
              f"a detection run gets a plain note about its metrics (got {body.get('warnings')!r})")
        check("mask" in warnings.lower(),
              f"the note says the mask columns stay blank (got {warnings!r})")
    finally:
        appmod.runner.run_cli_stage = real
        _close(tmp)


def test_step7_says_nothing_extra_for_a_segmentation_run():
    client, tmp, state = _client()
    real = appmod.runner.run_cli_stage
    appmod.runner.run_cli_stage = lambda *a, **k: {"pid": 7}
    try:
        step6_dir = state["steps"]["6"]["dir"]
        os.makedirs(os.path.join(step6_dir, "dataset"), exist_ok=True)
        run_dir = os.path.join(step6_dir, "runs", "seg_run")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            f.write("task: segment\nimgsz: 512\n")
        client.put("/api/project/config/7", json={"run_dir": run_dir, "imgsz": 512})
        r = client.post("/api/step/7/run")
        body = r.get_json() or {}
        warnings = " ".join(body.get("warnings") or [])
        check("detector" not in warnings.lower(),
              f"a segmentation run gets no detector note (got {body.get('warnings')!r})")
    finally:
        appmod.runner.run_cli_stage = real
        _close(tmp)


# ---------------------------------------------------------------------------
# The Subject control: the other half of what chooses a training recipe
# (2026-09-15). Selection by file name was removed with the flags it read; a
# recipe is now offered only when its declared task and subject match these two
# controls, so the Subject control has to sit beside the task one, be saved
# with the project, and re-filter the recipe list the moment it changes.
# ---------------------------------------------------------------------------

# Every value the control offers, in the order the panel lists them. The same
# list is orchestrator.js's STEP6_PANEL_SUBJECTS; the check below holds the
# three copies (template, script, this suite) against each other.
SUBJECT_OPTIONS = ("coral", "fish", "algae", "other")


def test_subject_select_sits_beside_the_task_control():
    tpl = open(_TPL).read()
    check(tpl.count('id="s6-subject"') == 1, "exactly one Subject control")
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    task_at = panel.index('id="s6-task"')
    subject_at = panel.index('id="s6-subject"')
    row_of_task = panel.rindex('<div class="form-row"', 0, task_at)
    row_of_subject = panel.rindex('<div class="form-row"', 0, subject_at)
    check(row_of_task == row_of_subject,
          "the Subject control must sit in the same form row as the Training task "
          "control: the two together are what choose a recipe")
    tag = panel[panel.rindex("<select", 0, subject_at):panel.index(">", subject_at) + 1]
    check("title=" in tag, f"the Subject control needs a tooltip (got {tag!r})")
    tip = tag[tag.index("title=") + 7:]
    tip = tip[:tip.index('"')]
    check(not tip.startswith("You "), f"a tooltip may not open with You: {tip!r}")
    check(tip.endswith("."), f"a tooltip is complete sentences: {tip!r}")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", tip.strip()) if s]
    check(1 <= len(sentences) <= 2,
          f"a tooltip is one or two sentences, this has {len(sentences)}: {tip!r}")
    for banned in ("—", "–"):
        check(banned not in tip, f"a tooltip carries no dash: {tip!r}")
    check("recipe" in tip.lower(),
          f"the tooltip must say what the control does for the operator: {tip!r}")
    body = panel[subject_at:panel.index("</select>", subject_at)]
    values = re.findall(r'value="([a-z]+)"', body)
    check(tuple(values) == SUBJECT_OPTIONS,
          f"the control must offer {SUBJECT_OPTIONS}, it offers {tuple(values)}")
    check(re.search(r'value="coral"[^>]*selected', body),
          "coral is the preselected subject, which is what this module trains most")


def test_the_three_copies_of_the_subject_vocabulary_agree():
    """The template's options, the script's list and the server's vocabulary
    are three copies of one list. A value in one and not the others is a
    subject that either cannot be picked or can never match a recipe."""
    js = open(_JS).read()
    m = re.search(r"const STEP6_PANEL_SUBJECTS = \[([^\]]*)\];", js)
    check(m is not None, "orchestrator.js must declare STEP6_PANEL_SUBJECTS")
    listed = tuple(v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip())
    check(listed == SUBJECT_OPTIONS,
          f"orchestrator.js offers {listed}, the template offers {SUBJECT_OPTIONS}")
    check(tuple(appmod.STEP6_PANEL_SUBJECTS) == SUBJECT_OPTIONS,
          f"app.py knows {tuple(appmod.STEP6_PANEL_SUBJECTS)}, the panel offers "
          f"{SUBJECT_OPTIONS}")
    check(tuple(appmod.STEP6_PRESET_SUBJECTS)
          == tuple(appmod.STEP6_SUBJECTS) + (appmod.STEP6_SUBJECT_ANY,),
          "a recipe may declare every subject a dataset can, plus any")


def test_subject_round_trips_through_the_js():
    js = open(_JS).read()
    collect = js[js.index("function collectConfig"):js.index("} else if (step === 7)")]
    check("cfg.subject" in collect, "collectConfig(6) must send the chosen subject")
    check("step6SubjectValue()" in collect,
          "collectConfig(6) must read the control through its own normalizer, so a "
          "blank or unknown value is saved as the panel's default rather than as it is")
    pop = js[js.index("function populateConfigs"):js.index("// Step 7: evaluation")]
    check("s6-subject" in pop and "s6.subject" in pop,
          "populateConfigs must restore the saved subject")
    check("subjectSource = 'saved'" in pop,
          "populateConfigs must mark a restored subject as the project's own, or the "
          "dataset's recorded subject would overwrite it on the next panel open")
    check("delete subjectEl.dataset.subjectSource" in pop,
          "and must clear that mark for a project with no saved subject, or the mark "
          "left by the last project would block the new one's dataset subject")
    check("function onStep6SubjectChange" in js,
          "orchestrator.js must define the Subject control's change handler")
    handler = js[js.index("function onStep6SubjectChange"):]
    handler = handler[:handler.index("\n}") + 2]
    check("renderStep6RecipeOptions" in handler,
          "changing the subject must re-filter the recipe list at once")
    check("subjectSource = 'operator'" in handler,
          "a subject the operator picked must be marked as theirs")
    swap = js[js.index("function onStep6TaskChange"):]
    swap = swap[:swap.index("\n}") + 2]
    check("renderStep6RecipeOptions" in swap,
          "changing the training task must re-filter the recipe list too")


def test_subject_round_trips_through_the_config_put():
    """Server side: /api/project/config/6 is a blind dict.update (project_config
    in app.py), so the subject must survive untouched and must never be needed
    to launch a run."""
    client, tmp, _state = _client()
    try:
        r = client.put("/api/project/config/6",
                       json={"run_name": "subject_check", "subject": "fish"})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        check(appmod.current_project["steps"]["6"]["config"].get("subject") == "fish",
              "the config PUT must keep the saved subject")
        r, cmd, _env = _launch(client, {"run_name": "subject_check", "subject": "fish",
                                        "model": "", "device": "0", "batch": 8})
        check(r.status_code == 200, f"a stored subject must not block a launch: {r.get_json()}")
        check(bool(cmd), "the run must still launch with a subject in the config")
    finally:
        _close(tmp)


if __name__ == "__main__":
    print("TASK N2: Step 6 training task selector\n")
    print("UI:")
    run(test_task_select_is_inside_advanced)
    run(test_subject_select_sits_beside_the_task_control)
    run(test_the_three_copies_of_the_subject_vocabulary_agree)
    run(test_detect_weights_are_reachable_from_the_model_picker)
    print("JS:")
    run(test_js_round_trip)
    run(test_subject_round_trips_through_the_js)
    run(test_subject_round_trips_through_the_config_put)
    print("PRESET MARKER:")
    run(test_preset_import_writes_the_marker)
    run(test_collect_config_saves_the_marker)
    run(test_populate_configs_seeds_the_marker_from_the_stored_config)
    run(test_auto_apply_is_skipped_when_the_stored_config_already_has_a_preset)
    run(test_preset_id_round_trips_through_the_config_put)
    print("ENV:")
    run(test_env_carries_the_task_and_every_other_knob_for_both)
    run(test_blank_model_follows_the_task)
    run(test_missing_task_defaults_to_segmentation)
    print("MODEL / TASK CONFLICT:")
    run(test_seg_weights_with_detect_task_is_a_pre_launch_400)
    run(test_seg_weights_with_segment_task_still_runs)
    run(test_explicit_detect_weights_are_respected)
    run(test_unknown_task_is_refused_before_launch)
    print("STEP 7:")
    run(test_step7_warns_plainly_for_a_detection_run)
    run(test_step7_says_nothing_extra_for_a_segmentation_run)

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print()
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAILED {name}:\n{detail}")
    print("PASS" if not failed else f"{len(failed)} FAILED")
    sys.exit(1 if failed else 0)
