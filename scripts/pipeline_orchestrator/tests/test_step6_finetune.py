"""Task 8: Step 6 fine-tune surfacing (weights picker + freeze + presets),
orchestrator-side only (train_wrapper.py --freeze itself is a separate task
and already implemented/tested elsewhere).

No-pytest harness: run with env/bin/python <this file>.

Covers:
  cfg model_path -> captured cmd's model positional (index 7) == the path,
    AND env TCRMP_STEP6_FREEZE forwarded when cfg freeze is set.
  cfg model_path pointing at a nonexistent file -> error response, runner's
    run_cli_stage NEVER called.
  cfg without model_path -> the catalog model name passes through unchanged.
  /api/step/6/presets -> lists 'finetune' and 'retrain' builtin entries.
"""
import os
import sys
import json
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm

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


def _make_project(tmp):
    """Fake project with a step4test export (so _run_step6's source guard
    passes) and a fabricated runs/<r>/weights/best.pt to use as a model_path."""
    project_dir = tmp
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}

    export_dir = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    os.makedirs(os.path.join(export_dir, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(export_dir, "all_labels"), exist_ok=True)
    with open(os.path.join(export_dir, "data.yaml"), "w") as f:
        f.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")

    # A fabricated previous run's best.pt, usable as a fine-tune model_path.
    prior_run_dir = os.path.join(steps["6"]["dir"], "runs", "prior_run")
    weights_dir = os.path.join(prior_run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_pt = os.path.join(weights_dir, "best.pt")
    with open(best_pt, "wb") as f:
        f.write(b"\x00" * 16)  # fake weights content; never actually loaded

    state = {
        "id": os.path.basename(project_dir), "name": "t", "project_dir": project_dir,
        "created_at": "x", "updated_at": "x", "current_step": 6,
        "steps": steps,
    }
    return state, best_pt


def _client_with_project():
    tmp = tempfile.mkdtemp()
    state, best_pt = _make_project(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    return client, tmp, state, best_pt


def test_model_path_override_and_freeze_env():
    client, tmp, state, best_pt = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        captured["env"] = env_extra or {}
        return {"pid": 12345}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/6", json={
            "model_path": best_pt, "freeze": 10, "run_name": "ft_run",
        })
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")

        r = client.post("/api/step/6/run")
        data = r.get_json()
        check(r.status_code == 200, f"run failed: {data}")

        cmd = captured.get("cmd") or []
        # cmd = [ENTRY_POINTS[6], "all", src_dir, step_dir, run_name, epochs, imgsz, model]
        check(len(cmd) >= 8, f"cmd too short: {cmd}")
        check(cmd[7] == best_pt, f"cmd[7] (model positional) {cmd[7]!r} != {best_pt!r}")

        env = captured.get("env") or {}
        check(env.get("TCRMP_STEP6_FREEZE") == "10",
              f"TCRMP_STEP6_FREEZE not forwarded correctly: {env.get('TCRMP_STEP6_FREEZE')!r}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_model_path_nonexistent_errors_without_calling_runner():
    client, tmp, state, best_pt = _client_with_project()
    called = {"n": 0}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        called["n"] += 1
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        bogus = os.path.join(tmp, "does_not_exist.pt")
        r = client.put("/api/project/config/6", json={"model_path": bogus})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")

        r = client.post("/api/step/6/run")
        data = r.get_json()
        check(r.status_code >= 400, f"expected error status, got {r.status_code}: {data}")
        check("not found" in str(data.get("error", "")).lower(),
              f"expected 'not found' error message: {data}")
        check(called["n"] == 0, "runner.run_cli_stage must NOT be called on a bad model_path")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_model_path_passes_catalog_name_through():
    client, tmp, state, best_pt = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        captured["env"] = env_extra or {}
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/6", json={"model": "yolo11m-seg.pt"})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")

        r = client.post("/api/step/6/run")
        data = r.get_json()
        check(r.status_code == 200, f"run failed: {data}")

        cmd = captured.get("cmd") or []
        check(cmd[7] == "yolo11m-seg.pt", f"cmd[7] {cmd[7]!r} != catalog name")
        env = captured.get("env") or {}
        check("TCRMP_STEP6_FREEZE" not in env,
              f"TCRMP_STEP6_FREEZE should be absent when freeze unset: {env}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_pin_split_env_always_forwarded_with_defaults():
    """The four pin-split/transect env vars are always forwarded (Task 8/T7),
    even when the panel never touched them, so pinned_split.py and the Refine
    loop's seeder always agree on the same holdout definition."""
    client, tmp, state, best_pt = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["env"] = env_extra or {}
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.post("/api/step/6/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")
        env = captured.get("env") or {}
        check(env.get("TCRMP_STEP6_PIN_SPLIT") == "1", f"pin_split default: {env}")
        check(env.get("TCRMP_STEP6_VAL_TRANSECTS") == "5", f"val_transects default: {env}")
        check(env.get("TCRMP_STEP6_TEST_TRANSECTS") == "6", f"test_transects default: {env}")
        check(env.get("TCRMP_STEP6_HOLDOUT_MODE") == "transect", f"holdout_mode default: {env}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_presets_endpoint_lists_finetune_and_retrain():
    client, tmp, state, best_pt = _client_with_project()
    try:
        r = client.get("/api/step/6/presets")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        ids = [p["id"] for p in data.get("presets", [])]
        check("finetune" in ids, f"'finetune' missing from presets: {ids}")
        check("retrain" in ids, f"'retrain' missing from presets: {ids}")

        r2 = client.get("/api/step/6/presets/finetune")
        d2 = r2.get_json()
        check(r2.status_code == 200, f"finetune preset fetch failed: {d2}")
        params = d2.get("params", {})
        check(params.get("epochs") == 150, f"finetune epochs != 150: {params}")
        check(params.get("patience") == 20, f"finetune patience != 20: {params}")
        check(params.get("lr0") == 0.001, f"finetune lr0 != 0.001: {params}")
        check(params.get("freeze") == 10, f"finetune freeze != 10: {params}")
        # No champion.json and no runs yet in a bare project except the
        # fixture's own prior_run - so model_path should resolve to it.
        check(params.get("model_path") == best_pt,
              f"finetune model_path should resolve to newest best.pt: {params}")

        r3 = client.get("/api/step/6/presets/retrain")
        d3 = r3.get_json()
        params3 = d3.get("params", {})
        check(params3.get("model") == "yolo11m-seg.pt", f"retrain model wrong: {params3}")
        check(params3.get("pin_split") == "1", f"retrain pin_split wrong: {params3}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TASK 8 -- Step 6 fine-tune surfacing (orchestrator)\n")
    print("MODEL PATH OVERRIDE:")
    run(test_model_path_override_and_freeze_env)
    run(test_model_path_nonexistent_errors_without_calling_runner)
    run(test_no_model_path_passes_catalog_name_through)
    print("PIN-SPLIT ENV DEFAULTS:")
    run(test_pin_split_env_always_forwarded_with_defaults)
    print("PRESETS:")
    run(test_presets_endpoint_lists_finetune_and_retrain)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
