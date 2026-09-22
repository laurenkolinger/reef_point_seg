"""Task H: a composite training set opened as a pipeline project that starts
at Train.

No pytest: run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_composite_training_project.py

Covers:
  STATE    - /api/project/state exposes project_type + the composite block,
             opening lands current_step on 6 (Train) and unlocks Step 6.
  REFUSALS - the run routes for steps 1, 2, 3 and the annotator routes
             (step4test, step4loop, editmasks) answer 400 with the standard
             message; the step4test reset is refused too, so nothing can be
             pointed at the shared training set.
  PREFLIGHT- Step 6 accepts the symlinked export (all_images/ + data.yaml are
             read through the link, no copy).
  UI       - the disabled-tile wiring, the Train banner and the cache-bust.

Nothing here spawns a process: subprocess.run is stubbed module wide, the
VICARIUS event bridge is stubbed to no-ops, and the only stage launch a test
reaches is a stubbed runner.run_cli_stage.
"""

import json
import os
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


# Standing rule: no orchestrator test may reach a real subprocess. Popen is
# stubbed as well as run: a step-run route that slipped past its guard would
# otherwise launch a real pipeline stage from a throwaway fixture dir.
subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402
from _matrix.composite import create_training_project  # noqa: E402


class _FakeVic:
    """No-op stand-in for vicarius_bridge: these tests must never write a real
    event into the VICARIUS stream."""

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None

    @staticmethod
    def note(*_a, **_k):
        return None


appmod.vic = _FakeVic()

REFUSAL = ("Not available for a training project. "
           "This project trains from the training set or box dataset named "
           "in its README.")


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

def _make_composite(root, name="task_h_set"):
    """A minimal composite export the training project can link to. Built by
    hand (not build_composite) so the fixture needs no annotation projects."""
    out_dir = os.path.join(root, "_composite", name)
    os.makedirs(os.path.join(out_dir, "all_images"))
    os.makedirs(os.path.join(out_dir, "all_labels"))
    with open(os.path.join(out_dir, "all_images", "img1.jpg"), "w") as f:
        f.write("not really a jpeg, only the layout matters here")
    with open(os.path.join(out_dir, "all_labels", "img1.txt"), "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n" % out_dir)
    with open(os.path.join(out_dir, "composite.json"), "w") as f:
        json.dump({"name": name, "created_at": "2026-09-02 09:00:00", "by": "LO",
                   "labels": ["MC"], "n_images": 1, "n_masks": 1,
                   "project_filter": ["projA"],
                   "source_projects": [{"project_id": "projA", "n_masks": 1}],
                   "projects": ["projA"], "linked_projects": []}, f, indent=2)
    return out_dir


def _open_training_project():
    """(client, tmp_root, run_dir) with a training project opened."""
    tmp = tempfile.mkdtemp(prefix="comp_train_proj_")
    _make_composite(tmp)
    res = create_training_project(tmp, "task_h_set", "LO")
    client = appmod.create_app().test_client()
    r = client.post("/api/project/open", json={"project_dir": res["run_dir"]})
    check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
    return client, tmp, res["run_dir"]


def _close(tmp):
    appmod.current_project = None
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def test_state_exposes_project_type_and_composite():
    client, tmp, _run_dir = _open_training_project()
    try:
        data = client.get("/api/project/state").get_json()
        check(data.get("loaded") is True, f"project not loaded: {data}")
        st = data["state"]
        check(st.get("project_type") == "composite_training",
              f"project_type missing from the state payload: {st.get('project_type')!r}")
        comp = st.get("composite") or {}
        check(comp.get("name") == "task_h_set", f"composite block missing: {comp}")
        check(comp.get("source_projects") == [{"project_id": "projA", "n_masks": 1}],
              f"composite.source_projects missing: {comp}")
    finally:
        _close(tmp)


def test_open_lands_on_train_and_unlocks_step6():
    client, tmp, _run_dir = _open_training_project()
    try:
        st = client.get("/api/project/state").get_json()["state"]
        check(st.get("current_step") == 6,
              f"a training project must open on Train, got {st.get('current_step')!r}")
        check(st["steps"]["6"]["status"] != "locked",
              f"Step 6 must be unlocked from the start, got {st['steps']['6']['status']!r}")
        for s in ("1", "2", "3", "4"):
            check(st["steps"][s]["status"] == "skipped",
                  f"step {s} should stay skipped, got {st['steps'][s]['status']!r}")
    finally:
        _close(tmp)


def test_regular_project_is_not_treated_as_composite():
    """The guard keys on project.json only: an ordinary project is untouched."""
    check(appmod._is_composite_training({"project_type": "composite_training"}) is True,
          "the helper must recognize a training project")
    check(appmod._is_composite_training({"name": "ordinary"}) is False,
          "an ordinary project must not be treated as a training project")
    check(appmod._is_composite_training(None) is False, "no project loaded is not a training project")


def test_reopen_keeps_a_later_step():
    """A user who moved past Train reopens where they left off; only a step
    BELOW Train is pulled forward to it."""
    client, tmp, run_dir = _open_training_project()
    try:
        pj = os.path.join(run_dir, "project.json")
        for saved, expected in ((1, 6), (6, 6), (7, 7), (8, 8), (99, 8)):
            with open(pj) as f:
                doc = json.load(f)
            doc["current_step"] = saved
            # Train and everything after it must be reachable for the later
            # steps to mean anything, exactly as completing Step 6 leaves them.
            for s in ("6", "7", "8"):
                doc["steps"].setdefault(s, {"status": "pending", "name": s,
                                            "dir": os.path.join(run_dir, "x"),
                                            "config": {}, "outputs": {}})
                doc["steps"][s]["status"] = "pending"
            with open(pj, "w") as f:
                json.dump(doc, f)
            r = client.post("/api/project/open", json={"project_dir": run_dir})
            check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
            got = client.get("/api/project/state").get_json()["state"]["current_step"]
            check(got == expected,
                  f"saved current_step {saved} should reopen on {expected}, got {got}")
    finally:
        _close(tmp)


def _ordinary_project(tmp):
    """A minimal ordinary project on disk (no project_type), the same shape
    the launcher's stub has before the orchestrator fills it in."""
    for s in pm.STEP_KEYS:
        os.makedirs(os.path.join(tmp, pm.STEP_DIRS[s]), exist_ok=True)
    state = {
        "id": os.path.basename(tmp), "name": "ordinary", "created_at": "2026-09-02T09:00:00",
        "updated_at": "2026-09-02T09:00:00", "project_dir": tmp,
        "current_step": 1, "steps": {}, "access_log": [], "initials": "LO",
    }
    with open(os.path.join(tmp, "project.json"), "w") as f:
        json.dump(state, f)
    return tmp


def test_regular_project_passes_every_guard():
    """Every route the training project refuses must stay open for an ordinary
    one. Each stage entry point is stubbed, so a route that DOES proceed
    proceeds into a stub and nothing is launched."""
    tmp = tempfile.mkdtemp(prefix="comp_train_ordinary_")
    _ordinary_project(tmp)
    client = appmod.create_app().test_client()
    real_cli, real_flask, real_port = (appmod.runner.run_cli_stage,
                                       appmod.runner.run_flask_stage,
                                       appmod.find_free_port)
    appmod.runner.run_cli_stage = lambda *a, **k: {"success": True, "pid": 0}
    appmod.runner.run_flask_stage = lambda *a, **k: {"success": True, "pid": 0, "port": 0}
    appmod.find_free_port = lambda *a, **k: 0
    try:
        r = client.post("/api/project/open", json={"project_dir": tmp})
        check(r.status_code == 200, f"open failed: {r.status_code} {r.get_json()}")
        check(appmod.current_project.get("project_type") is None,
              "the fixture must be an ordinary project")
        paths = ["/api/step/1/run", "/api/step/2/run", "/api/step/3/run",
                 "/api/step/1/link",
                 "/api/step/1/reset", "/api/step/2/reset", "/api/step/3/reset",
                 "/api/step/4/reset", "/api/step/5/reset",
                 "/api/step/step4test/run", "/api/step/step4test/reset",
                 "/api/step/step4loop/run", "/api/step/step4loop/reset",
                 "/api/step/editmasks/run"]
        for path in paths:
            body = client.post(path, json={}).get_json() or {}
            check(body.get("error") != REFUSAL,
                  f"{path} refused an ORDINARY project with the training-set message")
    finally:
        appmod.runner.run_cli_stage = real_cli
        appmod.runner.run_flask_stage = real_flask
        appmod.find_free_port = real_port
        _close(tmp)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_step_run_routes_refused():
    client, tmp, _run_dir = _open_training_project()
    try:
        for step in (1, 2, 3):
            r = client.post(f"/api/step/{step}/run", json={})
            body = r.get_json() or {}
            check(r.status_code == 400, f"step {step} run: expected 400, got {r.status_code} {body}")
            check(body.get("error") == REFUSAL,
                  f"step {step} run: wrong message {body.get('error')!r}")
    finally:
        _close(tmp)


def test_annotator_routes_refused():
    client, tmp, _run_dir = _open_training_project()
    try:
        for path in ("/api/step/step4test/run", "/api/step/step4loop/run",
                     "/api/step/editmasks/run", "/api/step/step4test/reset",
                     # step4loop/reset drops masks and rewrites
                     # segmentations.json inside the export, which here is the
                     # SHARED training set.
                     "/api/step/step4loop/reset"):
            r = client.post(path, json={})
            body = r.get_json() or {}
            check(r.status_code == 400, f"{path}: expected 400, got {r.status_code} {body}")
            check(body.get("error") == REFUSAL, f"{path}: wrong message {body.get('error')!r}")
    finally:
        _close(tmp)


def test_reset_routes_refused():
    """pm.reset_step clears every step from the given one onward and relocks
    the rest, so a pre-Train reset would relock Train itself and wipe the
    skipped markers. Train and later stay resettable."""
    client, tmp, _run_dir = _open_training_project()
    try:
        for step in (1, 2, 3, 4, 5):
            r = client.post(f"/api/step/{step}/reset", json={})
            body = r.get_json() or {}
            check(r.status_code == 400, f"step {step} reset: expected 400, got {r.status_code}")
            check(body.get("error") == REFUSAL,
                  f"step {step} reset: wrong message {body.get('error')!r}")
        st = client.get("/api/project/state").get_json()["state"]
        check(st["steps"]["1"]["status"] == "skipped",
              "the refused resets left the skipped markers alone")
        check(st["steps"]["6"]["status"] != "locked", "the refused resets left Train unlocked")
        r = client.post("/api/step/6/reset", json={})
        check(r.status_code == 200, f"Train must stay resettable, got {r.status_code}")
    finally:
        _close(tmp)


def test_reset_never_reaches_the_shared_training_set():
    """The refusal above is what protects the composite: prove the symlinked
    export is still intact after the reset attempt."""
    client, tmp, run_dir = _open_training_project()
    try:
        client.post("/api/step/step4test/reset", json={})
        link = os.path.join(run_dir, "step4test_combinedAnnotate")
        check(os.path.islink(link), "the export symlink was removed by a refused reset")
        check(os.path.isfile(os.path.join(tmp, "_composite", "task_h_set", "data.yaml")),
              "the shared training set lost its data.yaml")
    finally:
        _close(tmp)


# ---------------------------------------------------------------------------
# step 6 preflight through the symlink
# ---------------------------------------------------------------------------

def test_step6_preflight_follows_the_symlink():
    client, tmp, run_dir = _open_training_project()
    calls = {}
    real_run_cli = appmod.runner.run_cli_stage

    def _fake_run_cli(step, cmd, cwd=None, env_extra=None):
        calls["cmd"] = list(cmd)
        return {"success": True, "pid": 0}

    appmod.runner.run_cli_stage = _fake_run_cli
    try:
        # Single GPU with an explicit batch: the multi-GPU/auto-batch preflight
        # is a separate guard and is not what this test is about.
        appmod.current_project["steps"]["6"]["config"].update({"device": "0", "batch": 8})
        r = client.post("/api/step/6/run", json={"config": {}})
        body = r.get_json() or {}
        check("error" not in body, f"step 6 refused a symlinked export: {body}")
        src = calls.get("cmd", [None, None, None])[2]
        check(src == os.path.join(run_dir, "step4test_combinedAnnotate"),
              f"the training source must be this project's export link, got {src!r}")
        check(os.path.isdir(os.path.join(src, "all_images")),
              "all_images/ must resolve through the symlink")
    finally:
        appmod.runner.run_cli_stage = real_run_cli
        _close(tmp)


# ---------------------------------------------------------------------------
# UI wiring
# ---------------------------------------------------------------------------

_TPL = os.path.join(_ORCH, "templates", "index.html")
_JS = os.path.join(_ORCH, "static", "orchestrator.js")
_TOOLTIP = "Not applicable: this project trains from a training set or box dataset"


def test_ui_disabled_tiles_and_banner():
    js = open(_JS).read()
    check("applyCompositeTrainingMode" in js, "orchestrator.js must define applyCompositeTrainingMode")
    check("function applyCompositeTrainingMode" in js, "applyCompositeTrainingMode must be a function")
    i = js.index("const COMPOSITE_DISABLED_TILES")
    body = js[i:js.index("function enterApp")]
    for key in ("'1'", "'2'", "'3'", "'step4test'", "'step4loop'", "'editmasks'"):
        check(key in body, f"the disabled-tile list must cover {key}")
    check("step-disabled" in body,
          "disabled tiles must reuse the existing .step-disabled visual language")
    check("data-step-disabled" in body,
          "a disabled tile must drop out of getNavOrder()/switchStep() via data-step-disabled")
    check(_TOOLTIP in body, f"the disabled tiles need the tooltip {_TOOLTIP!r}")
    check("composite_training" in body, "the mode must key on project_type composite_training")
    check("is_training_project" in body,
          "the mode must read the server's generic training-project flag, so the "
          "same path serves a bbox project")
    enter = js[js.index("function enterApp"):js.index("function enterApp") + 2000]
    check("applyCompositeTrainingMode()" in enter,
          "applyCompositeTrainingMode must be called from enterApp")
    check("s6-composite-banner" in body, "the Train banner must be filled from the composite block")
    # The banner is now generic: "Training from <training set|box dataset>
    # <name>", with the label handed over by the server per project type.
    check("'Training from '" in body, "the banner must name what this project trains from")
    check("training_source_label" in body and "training_source_name" in body,
          "the banner must read the server's plain-language source label and name")
    check("built from:" in body, "the banner must name the source projects")
    check("escapeHtml(" in body, "the banner text must be escaped")

    tpl = open(_TPL).read()
    check('id="s6-composite-banner"' in tpl, "index.html must carry the Train banner element")
    check(tpl.count('id="s6-composite-banner"') == 1, "exactly one Train banner element")


def test_asset_version_bumped():
    check(appmod.ASSET_VERSION == "20260921-start",
          f"ASSET_VERSION is {appmod.ASSET_VERSION}")
    sh = open(os.path.join(_SCRIPTS, "restart_reef_point_seg.sh")).read()
    check("orchestrator.js?v=20260921-start" in sh,
          "the restart script must fetch the fresh JS")
    check("assert_html_present '20260921-start'" in sh,
          "the restart self-test must pin the new cache-bust")
    check("20260901-quality-sweep" not in sh and "20260902-uniform-training" not in sh
          and "20260902-final-polish" not in sh,
          "a stale cache-bust marker is left in the restart script")


if __name__ == "__main__":
    print("TASK H: composite training projects\n")
    print("STATE:")
    run(test_state_exposes_project_type_and_composite)
    run(test_open_lands_on_train_and_unlocks_step6)
    run(test_reopen_keeps_a_later_step)
    run(test_regular_project_is_not_treated_as_composite)
    print("REFUSALS:")
    run(test_step_run_routes_refused)
    run(test_annotator_routes_refused)
    run(test_reset_routes_refused)
    run(test_reset_never_reaches_the_shared_training_set)
    run(test_regular_project_passes_every_guard)
    print("STEP 6 PREFLIGHT:")
    run(test_step6_preflight_follows_the_symlink)
    print("UI:")
    run(test_ui_disabled_tiles_and_banner)
    run(test_asset_version_bumped)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
