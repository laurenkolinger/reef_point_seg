"""No-re-screening at the orchestrator layer (2026-09-27 VEGA annotator
tidy-up plan): _run_step3 writes prior_screening_input.json and passes
--prior-screening; every annotator launch (configure and resume alike)
refreshes prior_screening.json in the project's step4test export dir.

No-pytest harness: run with env/bin/python <this file>.

subprocess.run/Popen are stubbed at module level before app.py is imported
(the standing rule for this directory): a route reachable from these tests
must never shell out to ss/pgrep/curl against this box's real processes.

Covers:
  _run_step3           writes <step3_dir>/prior_screening_input.json as
                       {stem: [labels]} from the coverage matrix, excluding
                       the open project; passes --prior-screening <path> in
                       the CLI command; a screening.screened_by_stem failure
                       is logged via vic.note and the flag is simply
                       omitted, Step 3 still runs.
  annotator launch     _launch_step4test_ui (mode="step4test" and
                       mode="step4loop") writes prior_screening.json under
                       the step4test export dir before spawning, built from
                       Step 3's own selected_frames.csv; a write failure is
                       logged and the launch still proceeds.
  no selection yet     _selected_frame_stems() answers [] (not a raise) when
                       Step 3 has not produced selected_frames.csv, and the
                       launch still writes a well-formed (empty) file.
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

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*a, **k):
    raise AssertionError(f"subprocess.Popen reached: {a[0] if a else k.get('args')!r}")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402
from _reefreview import screening  # noqa: E402

# Test isolation (2026-09-28 review): _kill_all_annotators' flush-before-kill
# (POST /api/flush, pipeline_orchestrator/app.py) reaches the network even
# when subprocess.run/Popen are stubbed, since it goes through urllib, not
# subprocess. This file's step4test/run, step4loop/run and editmasks/run
# calls all drive it, so appmod._flush_annotator is stubbed here (never a
# real save, never a real socket) and socket.socket is guarded so any code
# path that still tries to reach the orchestrator (5050) or the annotator
# port range (5080-5089) fails the test loudly instead of making a live
# network call.
appmod._flush_annotator = lambda port: (True, None)
import socket as _socket  # noqa: E402
_REAL_SOCKET = _socket.socket
_GUARDED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
_GUARDED_PORTS = {5050} | set(range(5080, 5090))


class _NoLivePortSocket(_REAL_SOCKET):
    def connect(self, address, *a, **k):
        if (isinstance(address, tuple) and len(address) >= 2
                and address[0] in _GUARDED_HOSTS and address[1] in _GUARDED_PORTS):
            raise AssertionError(
                f"test tried to open a real socket to {address[0]}:{address[1]} "
                f"(guarded: 5050, 5080-5089); stub the code path that reached this")
        return super().connect(address, *a, **k)


_socket.socket = _NoLivePortSocket

_REAL_INPROCESS_ROOT = appmod._inprocess_root
_RESULTS = []


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


# ── Fixtures ────────────────────────────────────────────────────────────

def _write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def _make_project(tmp, *, project_id="current123", target_species="PA, SS"):
    """Mirrors the other pipeline_orchestrator tests' fixture shape: every
    chain step dir under tmp, plus a project.json so screening.py's own
    identity lookup (used by _write_prior_screening_input /
    _write_prior_screening_for_launch) resolves this project correctly."""
    project_dir = tmp
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {"target_species": target_species}, "outputs": {}}
    export_dir = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    os.makedirs(export_dir, exist_ok=True)
    _write_json(os.path.join(project_dir, "project.json"), {
        "id": project_id, "name": "t",
        "steps": {"3": {"config": {"target_species": target_species}}},
    })
    state = {
        "id": project_id, "name": "t", "project_dir": project_dir,
        "created_at": "x", "updated_at": "x", "current_step": 3,
        "steps": steps,
    }
    return state, export_dir


def _build_matrix_fixture(runs_root):
    """One "other" project (otherProj, target PA) that screens IMG_a for PA
    with an accepted manual mask, so both the Step 3 input file and a
    launch's prior_screening.json have something real to carry."""
    other_dir = os.path.join(runs_root, "run_otherProj_20260920_bbbbbb")
    _write_json(os.path.join(other_dir, "project.json"), {
        "id": "otherProj", "name": "Other Project",
        "steps": {"3": {"config": {"target_species": "PA"}}},
    })
    seg_doc = {
        "IMG_a.jpeg": {
            "image_path": "IMG_a.jpeg", "image_width": 200, "image_height": 100,
            "reviewed": True, "exported": False,
            "processed_at": "2026-09-20T10:00:00-04:00",
            "masks": [{
                "id": 0, "species": "PA", "status": "accepted",
                "source_type": "manual_click", "reviewer": "at", "review": False,
                "polygon_px": [[0, 0, 10, 0, 10, 10, 0, 10]],
            }],
        },
    }
    _write_json(os.path.join(
        other_dir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
        seg_doc)


def _write_selected_frames_csv(step3_dir, basenames):
    path = os.path.join(step3_dir, "selected_frames.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "basename"])
        writer.writeheader()
        for i, base in enumerate(basenames):
            writer.writerow({"frame_id": f"f{i}", "basename": base})
    return path


def _client_with_project(runs_root, *, project_id="current123", target_species="PA, SS"):
    tmp = tempfile.mkdtemp(prefix="prior_screening_launch_")
    state, export_dir = _make_project(tmp, project_id=project_id, target_species=target_species)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    appmod.step4test_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}
    appmod.step4loop_state = dict(appmod.step4test_state)
    appmod.step4loop_cfg = {}
    appmod._inprocess_root = lambda: runs_root
    return client, tmp, state, export_dir


def _teardown(tmp, runs_root=None):
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    shutil.rmtree(tmp, ignore_errors=True)
    if runs_root:
        shutil.rmtree(runs_root, ignore_errors=True)


# ── _run_step3: writes the input file, passes the flag ──────────────────

def test_run_step3_writes_prior_screening_input_and_passes_flag():
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root)
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")

        cmd = captured.get("cmd") or []
        check("--prior-screening" in cmd, f"expected --prior-screening in the command: {cmd}")
        i = cmd.index("--prior-screening")
        input_path = cmd[i + 1]
        check(input_path == os.path.join(state["steps"]["3"]["dir"], "prior_screening_input.json"),
              f"unexpected input path: {input_path}")
        check(os.path.isfile(input_path), f"the input file must actually exist: {input_path}")
        doc = json.load(open(input_path))
        check(doc.get("IMG_a") == ["PA"], f"IMG_a should be screened for PA by otherProj: {doc}")
    finally:
        appmod.runner.run_cli_stage = orig
        _teardown(tmp, runs_root)


def test_run_step3_excludes_the_open_project_itself():
    """If the open project has ALREADY touched a frame (its own prior work,
    e.g. a re-run after Step 3 was reset), that frame must never show up in
    its own prior_screening_input.json as 'screened elsewhere'."""
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root, project_id="otherProj")
    # The "open" project here reuses id 'otherProj' on purpose: its own
    # contribution to the matrix (IMG_a/PA) must be excluded from its own
    # prior_screening_input.json.
    captured = {}
    appmod.runner.run_cli_stage = lambda step, cmd, cwd=None, env_extra=None: (
        captured.update(cmd=cmd) or {"pid": 1})
    try:
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")
        cmd = captured.get("cmd") or []
        i = cmd.index("--prior-screening")
        doc = json.load(open(cmd[i + 1]))
        check("IMG_a" not in doc,
              f"a project's own contribution must never screen itself: {doc}")
    finally:
        _teardown(tmp, runs_root)


def test_run_step3_omits_flag_and_logs_on_screening_failure():
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    client, tmp, state, export_dir = _client_with_project(runs_root)
    logged = []
    orig_screened = screening.screened_by_stem
    orig_note = appmod.vic.note
    captured = {}
    appmod.runner.run_cli_stage = lambda step, cmd, cwd=None, env_extra=None: (
        captured.update(cmd=cmd) or {"pid": 1})
    appmod._screening.screened_by_stem = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    appmod.vic.note = lambda msg, study=None: logged.append(msg)
    try:
        r = client.post("/api/step/3/run")
        data = r.get_json()
        check(r.status_code == 200, f"a screening failure must never block Step 3: {data}")
        cmd = captured.get("cmd") or []
        check("--prior-screening" not in cmd, f"the flag must be omitted on failure: {cmd}")
        check(any("prior_screening_input.json" in m for m in logged),
              f"the failure must be logged: {logged}")
    finally:
        appmod._screening.screened_by_stem = orig_screened
        appmod.vic.note = orig_note
        _teardown(tmp, runs_root)


# ── annotator launch: configure (step4test) and resume (step4loop) ──────
#
# _launch_step4test_ui is a closure nested inside create_app() with no
# module-level name (every existing suite in this directory reaches it only
# through the real HTTP routes that call it), so these tests do the same:
# drive the actual /api/step/step4test/run, /api/step/step4loop/run and
# /api/step/editmasks/run routes with the minimum real fixture each one's
# own preconditions require, exactly as test_flush_before_kill.py,
# test_editmasks_routes.py and test_route_clears_stale.py already do.

def _seed_fresh_routed_input(export_dir, selected_frames_path):
    """A routed_input/ already holding prompts NEWER than selected_frames.csv
    (os.utime, not a sleep-based race), so _run_step4test's fast path calls
    _launch_step4test_ui() synchronously on THIS request's thread instead of
    spawning the background routing driver this suite must never invoke for
    real (it drives a headless placePoints subprocess)."""
    routed_dir = os.path.join(export_dir, "routed_input")
    ids_dir = os.path.join(routed_dir, "ids")
    os.makedirs(ids_dir, exist_ok=True)
    prompt_path = os.path.join(ids_dir, "sam_click_prompts.json")
    with open(prompt_path, "w", encoding="utf-8") as fh:
        json.dump({}, fh)
    os.utime(selected_frames_path, (1000, 1000))
    os.utime(prompt_path, (2000, 2000))
    return routed_dir


def test_step4test_launch_writes_prior_screening_json():
    """Drives the real /api/step/step4test/run route with a fresh, already-
    routed cache, taking the instant-launch fast path (no live annotator to
    reuse -> _launch_step4test_ui() runs on this request)."""
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root)
    sel_path = _write_selected_frames_csv(state["steps"]["3"]["dir"], ["IMG_a", "IMG_b"])
    state["steps"]["3"]["outputs"]["selected_frames"] = sel_path
    _seed_fresh_routed_input(export_dir, sel_path)
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5080}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/step4test/run", json={})
        data = r.get_json()
        check(r.status_code == 200, f"launch failed: {data}")
        check(data.get("phase") == "ui_ready", f"expected an instant ui_ready launch: {data}")
        out_path = os.path.join(export_dir, "prior_screening.json")
        check(os.path.isfile(out_path), f"prior_screening.json must be written: {out_path}")
        doc = json.load(open(out_path))
        check(doc["frames"].get("IMG_a", {}).get("screened", {}).get("PA", {}).get("outcome")
              == "found_manual", f"IMG_a must carry otherProj's PA screening: {doc}")
        check("IMG_b" not in doc["frames"], f"IMG_b is not screened anywhere: {doc['frames']}")
    finally:
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp, runs_root)


def _seed_frozen_split_and_predictions(state):
    """The step4loop/run route's two hard preconditions (D8 fail-closed): a
    frozen split manifest under Step 6, and a predictions.json under Step 8.
    Mirrors test_flush_before_kill.py's own step4loop fixture."""
    dataset_dir = os.path.join(state["steps"]["6"]["dir"], "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, "split_manifest.json"), "w") as f:
        json.dump({"holdout_mode": "transect", "holdout_transects": {"val": [5], "test": [6]},
                   "pinned": {}}, f)
    step8_dir = state["steps"]["8"]["dir"]
    pred_dir = os.path.join(step8_dir, "infer_x")
    os.makedirs(pred_dir, exist_ok=True)
    raw = os.path.join(pred_dir, "img0.jpeg")
    open(raw, "w").close()
    with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
        json.dump({"schema_version": 1, "generated_at": "2026-09-27T00:00:00-04:00",
                   "items": [{"filename": "img0.jpeg", "raw": raw, "width": 100,
                              "height": 100, "detections": []}]}, f)
    return pred_dir


def test_step4loop_launch_writes_prior_screening_json():
    """Drives the real /api/step/step4loop/run route (the "resume" launch
    mode): the seeder subprocess is stubbed at module level (subprocess.run),
    so only the real Flask route and the real _launch_step4test_ui tail run."""
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root)
    sel_path = _write_selected_frames_csv(state["steps"]["3"]["dir"], ["IMG_a", "IMG_b"])
    state["steps"]["3"]["outputs"]["selected_frames"] = sel_path
    pred_dir = _seed_frozen_split_and_predictions(state)
    # The seeder subprocess below is stubbed (never really runs), but the
    # route verifies its claimed output exists on disk afterward, so this
    # pre-seeds the segmentations.json a real seed_from_predictions.py run
    # would have produced (mirrors test_flush_before_kill.py's own fixture).
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        json.dump({"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}}, f)

    def _fake_run(cmd, **kwargs):
        if cmd and len(cmd) > 1 and "seed_from_predictions.py" in str(cmd[1]):
            class _R:
                returncode = 0
                stdout = json.dumps({
                    "seeded_frames": 1, "seeded_masks": 1, "skipped_existing": 0,
                    "skipped_below_conf": 0, "skipped_missing_raw": 0,
                    "skipped_holdout": 0, "round_manifest": None,
                })
                stderr = ""
            return _R()
        return _FakeCompleted()

    orig_subprocess_run = subprocess.run
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    subprocess.run = _fake_run
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5081}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/step4loop/run", json={"config": {"predictions_dir": pred_dir}})
        data = r.get_json()
        check(r.status_code == 200, f"launch failed: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        out_path = os.path.join(export_dir, "prior_screening.json")
        check(os.path.isfile(out_path), f"prior_screening.json must be written on resume too: {out_path}")
        doc = json.load(open(out_path))
        check("IMG_a" in doc["frames"], f"resume-mode launch must also carry screened frames: {doc}")
    finally:
        subprocess.run = orig_subprocess_run
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp, runs_root)


# The editmasks route needs no selected_frames.csv or routing state at all
# (it opens straight on the export's existing segmentations, per its own
# docstring), so it is the simplest real trigger for the shared launch tail
# ( _write_prior_screening_for_launch ) when Step 3 has not run yet, and for
# proving a write failure never blocks a launch.

def test_launch_writes_empty_file_when_no_selection_yet():
    """No selected_frames.csv exists: _selected_frame_stems() must answer []
    rather than raising, and the launch must still write a well-formed
    (empty-frames) file."""
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root)
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json()
        check(r.status_code == 200, f"launch failed: {data}")
        out_path = os.path.join(export_dir, "prior_screening.json")
        check(os.path.isfile(out_path), "a well-formed file must still be written")
        doc = json.load(open(out_path))
        check(doc["frames"] == {}, f"no selection yet -> nothing to check -> empty frames: {doc}")
    finally:
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp, runs_root)


def test_launch_failure_is_logged_and_never_blocks():
    runs_root = tempfile.mkdtemp(prefix="prior_screening_matrix_")
    _build_matrix_fixture(runs_root)
    client, tmp, state, export_dir = _client_with_project(runs_root)
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        f.write("{}")
    logged = []
    orig_write = appmod._screening.write_prior_screening
    orig_note = appmod.vic.note
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    appmod._screening.write_prior_screening = lambda *a, **k: (_ for _ in ()).throw(
        OSError("disk full"))
    appmod.vic.note = lambda msg, study=None: logged.append(msg)
    appmod.runner.run_flask_stage = lambda *a, **k: {"pid": 1, "port": 5083}
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json()
        check(r.status_code == 200,
              f"a prior_screening.json write failure must never block the launch: {data}")
        check(any("prior_screening.json" in m for m in logged), f"the failure must be logged: {logged}")
    finally:
        appmod._screening.write_prior_screening = orig_write
        appmod.vic.note = orig_note
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        _teardown(tmp, runs_root)


if __name__ == "__main__":
    print("2026-09-27 VEGA no-re-screening -- orchestrator layer\n")
    print("_run_step3:")
    run(test_run_step3_writes_prior_screening_input_and_passes_flag)
    run(test_run_step3_excludes_the_open_project_itself)
    run(test_run_step3_omits_flag_and_logs_on_screening_failure)
    print("annotator launch (configure + resume):")
    run(test_step4test_launch_writes_prior_screening_json)
    run(test_step4loop_launch_writes_prior_screening_json)
    run(test_launch_writes_empty_file_when_no_selection_yet)
    run(test_launch_failure_is_logged_and_never_blocks)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
