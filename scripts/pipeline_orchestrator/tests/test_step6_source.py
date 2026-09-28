"""
Self-contained tests for Task 1: Step 6 train reads the 4.test combined-annotator
export dir (step4test_combinedAnnotate), not the legacy Step-5 dir.

No pytest dependency: run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_source.py

Covers:
  BEHAVIOR - /api/step/6/list_classes resolves classes from the 4.test export
             (data.yaml + all_labels under <project>/step4test_combinedAnnotate).
  SOURCE   - _run_step6 sources its training dir from _step4test_export_dir()
             and no longer reads current_project steps["5"] for the train source.
"""

import os
import re
import sys
import json
import shutil
import tempfile
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

_RESULTS = []

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Standing rule: no orchestrator test may reach a real subprocess.
subprocess.run = lambda *a, **k: _FakeCompleted()


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


def _make_export(tmp):
    """Build a minimal 4.test export dir under tmp/step4test_combinedAnnotate."""
    exp = os.path.join(tmp, "step4test_combinedAnnotate")
    os.makedirs(os.path.join(exp, "all_images"))
    os.makedirs(os.path.join(exp, "all_labels"))
    with open(os.path.join(exp, "data.yaml"), "w") as f:
        f.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")
    with open(os.path.join(exp, "all_labels", "img1.txt"), "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    return exp


def test_list_classes_reads_4test_export():
    """list_classes must read the class roster from the 4.test export dir."""
    import app as appmod
    tmp = tempfile.mkdtemp(prefix="step6_source_classes_")
    exp = _make_export(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = {"project_dir": tmp, "name": "t", "steps": {}}
    try:
        r = client.get("/api/step/6/list_classes")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        names = [c["name"] for c in (data.get("classes") or [])]
        check("MC" in names, f"expected MC in {names}")
        check(data.get("src_dir") == exp, f"src_dir {data.get('src_dir')} != {exp}")
        # instance + image counts come from the all_labels we wrote (1 polygon, 1 image)
        mc = [c for c in data["classes"] if c["name"] == "MC"][0]
        check(mc["instance_count"] == 1, f"instance_count {mc['instance_count']} != 1")
        check(mc["image_count"] == 1, f"image_count {mc['image_count']} != 1")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_step6_source_is_4test_not_step5():
    """Source guard: _run_step6 resolves its train source via _step4test_export_dir()
    and no longer reads steps['5'] for the source."""
    src = open(os.path.join(_ORCH, "app.py")).read()
    # Helper exists.
    check("def _step4test_export_dir(" in src, "_step4test_export_dir helper missing")
    # Slice out the _run_step6 body.
    m = re.search(r"\n    def _run_step6\(.*?(?=\n    def _run_step7\()", src, re.DOTALL)
    check(m is not None, "could not locate _run_step6 body")
    body = m.group(0)
    check("_step4test_export_dir()" in body,
          "_run_step6 does not source from _step4test_export_dir()")
    check('steps.get("5"' not in body and 'steps["5"]' not in body,
          "_run_step6 still references steps['5'] for the train source")


def test_load_project_unlocks_train_after_step3():
    """Review FIX-1: 4.test (non-chain) replaces Steps 4+5, so completing Step 3
    must unlock Step 6 (train) directly — otherwise train stays locked forever
    and /api/step/6/run returns 'Step 6 is locked'."""
    import project_manager as pm
    d = tempfile.mkdtemp(prefix="step6_source_unlock_")
    for sd in pm.STEP_DIRS.values():
        os.makedirs(os.path.join(d, sd), exist_ok=True)
    # Completed steps need non-empty dirs or load_project re-locks them.
    for s in ("1", "2", "3"):
        with open(os.path.join(d, pm.STEP_DIRS[s], "marker.txt"), "w") as f:
            f.write("x")
    state = {
        "id": os.path.basename(d), "name": "t", "project_dir": d,
        "created_at": "x", "updated_at": "x", "current_step": 4,
        "steps": {s: {"status": ("completed" if s in ("1", "2", "3") else "locked"),
                      "name": pm.STEP_NAMES[s],
                      "dir": os.path.join(d, pm.STEP_DIRS[s]),
                      "config": {}, "outputs": {}} for s in pm.STEP_KEYS},
    }
    with open(os.path.join(d, "project.json"), "w") as f:
        json.dump(state, f)
    try:
        loaded = pm.load_project(d)
        check(loaded["steps"]["6"]["status"] == "pending",
              f"train (6) should unlock after step 3 done; got {loaded['steps']['6']['status']}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---- Training source override (spec D5, 2026-08-26) ------------------------

def _project_for_step6(tmp):
    """Fake project whose Step 4 export exists (the default source)."""
    import project_manager as pm
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    _make_export(tmp)
    return {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
            "created_at": "x", "updated_at": "x", "current_step": 6, "steps": steps}


def _make_composite(root, name, with_yaml=True):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "all_images"), exist_ok=True)
    os.makedirs(os.path.join(d, "all_labels"), exist_ok=True)
    if with_yaml:
        with open(os.path.join(d, "data.yaml"), "w") as f:
            f.write("path: x\ntrain: all_images\nval: all_images\nnc: 1\nnames:\n  0: MC\n")
    return d


def _run_step6_with_source(source_dir):
    """POST /api/step/6/run with cfg.source_dir; returns (status, json, captured)."""
    import app as appmod
    tmp = tempfile.mkdtemp(prefix="step6_source_")
    comp_root = tempfile.mkdtemp(prefix="step6_composite_root_")
    orig_run = appmod.runner.run_cli_stage
    orig_root = appmod._composite_root
    captured = {}
    appmod._composite_root = lambda: comp_root
    appmod.runner.run_cli_stage = (
        lambda step, cmd, cwd=None, env_extra=None: captured.update(cmd=cmd) or {"pid": 1})
    try:
        state = _project_for_step6(tmp)
        client = appmod.create_app().test_client()
        appmod.current_project = state
        good = _make_composite(comp_root, "composite_26-08-26_1200")
        bad = _make_composite(comp_root, "no_yaml", with_yaml=False)
        outside = _make_composite(tmp, "outside_composite")
        target = {"good": good, "bad": bad, "outside": outside, "missing": os.path.join(comp_root, "nope")}[source_dir]
        state["steps"]["6"]["config"]["source_dir"] = target
        r = client.post("/api/step/6/run")
        return r.status_code, (r.get_json() or {}), captured, target, state
    finally:
        appmod.runner.run_cli_stage = orig_run
        appmod._composite_root = orig_root
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(comp_root, ignore_errors=True)


def test_source_dir_outside_composite_root_is_400():
    code, data, captured, target, _ = _run_step6_with_source("outside")
    check(code == 400, f"outside _composite -> 400, got {code} {data}")
    check(target in data.get("error", ""), f"error must name the path: {data}")
    check("http_status" not in data, "http_status is a transport hint, never a body field")
    check(not captured, "runner must not be called on a refused source")


def test_source_dir_missing_data_yaml_is_400():
    code, data, captured, _, _ = _run_step6_with_source("bad")
    check(code == 400 and "data.yaml" in data.get("error", ""), f"missing data.yaml -> 400: {code} {data}")
    check(not captured, "runner must not be called on a refused source")
    code, data, captured, _, _ = _run_step6_with_source("missing")
    check(code == 400, f"nonexistent dir -> 400: {code} {data}")


def test_valid_composite_source_is_used_and_saved():
    code, data, captured, target, state = _run_step6_with_source("good")
    check(code == 200, f"valid composite -> 200: {code} {data}")
    cmd = captured.get("cmd") or []
    check(len(cmd) >= 3 and cmd[2] == os.path.realpath(target),
          f"cmd[2] (train source) must be the composite dir: {cmd}")
    check(state["steps"]["6"]["config"].get("source_dir") == target, "source_dir saved in the step config")


def test_blank_source_dir_keeps_project_export():
    import app as appmod
    tmp = tempfile.mkdtemp(prefix="step6_source_blank_")
    orig_run = appmod.runner.run_cli_stage
    captured = {}
    appmod.runner.run_cli_stage = (
        lambda step, cmd, cwd=None, env_extra=None: captured.update(cmd=cmd) or {"pid": 1})
    try:
        state = _project_for_step6(tmp)
        client = appmod.create_app().test_client()
        appmod.current_project = state
        state["steps"]["6"]["config"]["source_dir"] = ""
        r = client.post("/api/step/6/run")
        check(r.status_code == 200, f"blank source -> project export: {r.status_code} {r.get_json()}")
        exp = os.path.join(tmp, "step4test_combinedAnnotate")
        check((captured.get("cmd") or [None] * 3)[2] == exp, f"train source must be the Step 4 export: {captured}")
        check(state["steps"]["6"]["config"].get("source_dir") is None, "blank saves as None")
    finally:
        appmod.runner.run_cli_stage = orig_run
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


# ---- H2 (2026-08-26): class selection must match the training source -------

def _write_yaml(d, names):
    lines = ["path: x", "train: all_images", "val: all_images", f"nc: {len(names)}", "names:"]
    lines += [f"  {i}: {n}" for i, n in names.items()]
    with open(os.path.join(d, "data.yaml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _h2_fixture(export_names=None):
    """Project export {0: OFAV, 1: PA} (+ extras) and a composite whose ids
    follow sorted label order {0: MC, 1: OFAV, 2: PA}. Returns a dict of
    handles; call fx['cleanup']() when done."""
    import app as appmod
    tmp = tempfile.mkdtemp(prefix="step6_h2_")
    comp_root = tempfile.mkdtemp(prefix="step6_h2_composite_root_")
    state = _project_for_step6(tmp)
    exp = os.path.join(tmp, "step4test_combinedAnnotate")
    _write_yaml(exp, export_names or {0: "OFAV", 1: "PA"})
    with open(os.path.join(exp, "all_labels", "img1.txt"), "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.1 0.1 0.2 0.2 0.3 0.3\n")
    comp = _make_composite(comp_root, "composite_26-08-26_1200", with_yaml=False)
    _write_yaml(comp, {0: "MC", 1: "OFAV", 2: "PA"})
    with open(os.path.join(comp, "all_labels", "a.txt"), "w") as f:
        f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.1 0.1 0.2 0.2 0.3 0.3\n2 0.1 0.1 0.2 0.2 0.3 0.3\n")
    captured = {}
    orig_run = appmod.runner.run_cli_stage
    orig_root = appmod._composite_root
    appmod._composite_root = lambda: comp_root
    appmod.runner.run_cli_stage = (
        lambda step, cmd, cwd=None, env_extra=None: captured.update(cmd=cmd, env=dict(env_extra or {})) or {"pid": 1})
    appmod.current_project = state
    client = appmod.create_app().test_client()

    def cleanup():
        appmod.runner.run_cli_stage = orig_run
        appmod._composite_root = orig_root
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(comp_root, ignore_errors=True)
    return {"client": client, "state": state, "export": exp, "composite": comp,
            "captured": captured, "cleanup": cleanup}


def test_list_classes_follows_source_dir():
    fx = _h2_fixture()
    try:
        c = fx["client"]
        r = c.get("/api/step/6/list_classes")
        d = r.get_json() or {}
        check(r.status_code == 200, f"export listing: {r.status_code} {d}")
        check([(x["id"], x["name"]) for x in d["classes"]] == [(0, "OFAV"), (1, "PA")], f"export ids: {d['classes']}")
        check(d.get("include_classes_source") == "", "blank source echoes ''")
        r = c.get("/api/step/6/list_classes", query_string={"source_dir": fx["composite"]})
        d = r.get_json() or {}
        check(r.status_code == 200, f"composite listing: {r.status_code} {d}")
        check([(x["id"], x["name"]) for x in d["classes"]] == [(0, "MC"), (1, "OFAV"), (2, "PA")],
              f"composite ids must come from the composite data.yaml: {d['classes']}")
        pa = [x for x in d["classes"] if x["name"] == "PA"][0]
        check(pa["instance_count"] == 1 and pa["image_count"] == 1, f"counts from the composite labels: {pa}")
        check(d.get("src_dir") == os.path.realpath(fx["composite"]), f"src_dir is the composite: {d.get('src_dir')}")
        check(d.get("include_classes_source") == fx["composite"], "picker source echoed for collectConfig")
        r = c.get("/api/step/6/list_classes", query_string={"source_dir": fx["export"]})
        check(r.status_code == 400, f"a source outside _composite is refused: {r.status_code} {r.get_json()}")
        check(fx["export"] in (r.get_json() or {}).get("error", ""), "refusal names the path")
    finally:
        fx["cleanup"]()


def test_list_classes_translates_saved_selection_by_name():
    fx = _h2_fixture()
    try:
        c, cfg = fx["client"], fx["state"]["steps"]["6"]["config"]
        # Saved against the export (legacy config: no include_classes_source): PA is id 1.
        cfg["include_classes"] = [1]
        d = c.get("/api/step/6/list_classes", query_string={"source_dir": fx["composite"]}).get_json()
        check(d.get("include_classes") == [2], f"PA must pre-tick as id 2 in the composite: {d.get('include_classes')}")
        # Saved against the composite: PA is id 2; the export shows it as 1.
        cfg["include_classes"] = [2]
        cfg["include_classes_source"] = fx["composite"]
        d = c.get("/api/step/6/list_classes").get_json()
        check(d.get("include_classes") == [1], f"PA must pre-tick as id 1 in the export: {d.get('include_classes')}")
        # Same source as saved: ids pass through untouched.
        d = c.get("/api/step/6/list_classes", query_string={"source_dir": fx["composite"]}).get_json()
        check(d.get("include_classes") == [2], f"same source -> unchanged: {d.get('include_classes')}")
        # MC (composite id 0) has no counterpart in the export: reported, not guessed.
        cfg["include_classes"] = [0, 2]
        d = c.get("/api/step/6/list_classes").get_json()
        check(d.get("include_classes") == [1] and d.get("saved_classes_dropped") == ["MC"],
              f"absent name dropped and reported: {d.get('include_classes')} {d.get('saved_classes_dropped')}")
        # Saved source vanished: every class ticked (None), never a stale id list.
        cfg["include_classes_source"] = os.path.join(os.path.dirname(fx["composite"]), "gone")
        d = c.get("/api/step/6/list_classes").get_json()
        check(d.get("include_classes") is None, f"vanished saved source -> all: {d.get('include_classes')}")
    finally:
        fx["cleanup"]()


def test_run_step6_maps_export_ids_into_composite():
    """Picker loaded from the export (PA = id 1), training source = composite
    (PA = id 2): the trainer must receive TCRMP_STEP6_INCLUDE_CLASSES=2."""
    for legacy in (True, False):
        fx = _h2_fixture()
        try:
            cfg = fx["state"]["steps"]["6"]["config"]
            cfg["source_dir"] = fx["composite"]
            cfg["include_classes"] = [1]
            if not legacy:
                cfg["include_classes_source"] = ""
            r = fx["client"].post("/api/step/6/run")
            d = r.get_json() or {}
            check(r.status_code == 200, f"run (legacy={legacy}): {r.status_code} {d}")
            env = fx["captured"].get("env") or {}
            check(env.get("TCRMP_STEP6_INCLUDE_CLASSES") == "2",
                  f"PA must map to composite id 2 (legacy={legacy}): {env.get('TCRMP_STEP6_INCLUDE_CLASSES')!r}")
            check(d.get("include_classes") == [2] and not d.get("dropped_classes"), f"response echoes the mapped ids: {d}")
            check(cfg.get("include_classes") == [2] and cfg.get("include_classes_source") == fx["composite"],
                  f"saved ids are re-based on the training source: {cfg.get('include_classes')} {cfg.get('include_classes_source')}")
        finally:
            fx["cleanup"]()


def test_run_step6_composite_picker_ids_pass_through():
    fx = _h2_fixture()
    try:
        cfg = fx["state"]["steps"]["6"]["config"]
        cfg["source_dir"] = fx["composite"]
        cfg["include_classes"] = [2]
        cfg["include_classes_source"] = fx["composite"]
        r = fx["client"].post("/api/step/6/run")
        check(r.status_code == 200, f"run: {r.status_code} {r.get_json()}")
        env = fx["captured"].get("env") or {}
        check(env.get("TCRMP_STEP6_INCLUDE_CLASSES") == "2", f"ids unchanged: {env.get('TCRMP_STEP6_INCLUDE_CLASSES')!r}")
    finally:
        fx["cleanup"]()
    # And the export picker with the export as the training source: unchanged too.
    fx = _h2_fixture()
    try:
        cfg = fx["state"]["steps"]["6"]["config"]
        cfg["source_dir"] = ""
        cfg["include_classes"] = [1]
        r = fx["client"].post("/api/step/6/run")
        check(r.status_code == 200, f"run: {r.status_code} {r.get_json()}")
        env = fx["captured"].get("env") or {}
        check(env.get("TCRMP_STEP6_INCLUDE_CLASSES") == "1", f"export ids unchanged: {env}")
    finally:
        fx["cleanup"]()


def test_run_step6_reports_dropped_names_and_refuses_empty():
    fx = _h2_fixture(export_names={0: "OFAV", 1: "PA", 2: "SSID"})
    try:
        cfg = fx["state"]["steps"]["6"]["config"]
        cfg["source_dir"] = fx["composite"]
        cfg["include_classes"] = [1, 2]  # PA + SSID; the composite has no SSID
        r = fx["client"].post("/api/step/6/run")
        d = r.get_json() or {}
        check(r.status_code == 200, f"run: {r.status_code} {d}")
        env = fx["captured"].get("env") or {}
        check(env.get("TCRMP_STEP6_INCLUDE_CLASSES") == "2", f"only PA survives: {env}")
        check(d.get("dropped_classes") == ["SSID"], f"dropped names reported: {d}")
        fx["captured"].clear()
        cfg["include_classes"] = [2]  # SSID only -> nothing left to train on
        cfg["include_classes_source"] = ""  # the run above re-based the saved source; picker is the export again
        r = fx["client"].post("/api/step/6/run")
        d = r.get_json() or {}
        check(r.status_code == 400 and "SSID" in d.get("error", ""), f"all dropped -> 400 naming the class: {r.status_code} {d}")
        check(not fx["captured"], "runner must not be called when no class maps")
    finally:
        fx["cleanup"]()


def test_run_step6_refuses_vanished_picker_source():
    fx = _h2_fixture()
    try:
        cfg = fx["state"]["steps"]["6"]["config"]
        cfg["source_dir"] = fx["composite"]
        cfg["include_classes"] = [1]
        cfg["include_classes_source"] = os.path.join(os.path.dirname(fx["composite"]), "gone")
        r = fx["client"].post("/api/step/6/run")
        d = r.get_json() or {}
        check(r.status_code == 400 and "no longer valid" in d.get("error", ""), f"vanished picker source -> 400: {r.status_code} {d}")
        check(not fx["captured"], "runner must not be called")
    finally:
        fx["cleanup"]()


def test_translate_class_ids_helper():
    import app as appmod
    src = {0: "OFAV", 1: "PA", 2: "ssid"}
    dst = {0: "MC", 1: "ofav", 2: "PA"}
    mapped, dropped = appmod._translate_class_ids([1, 0, 2, 7, "x"], src, dst)
    check(mapped == [2, 1] and dropped == ["ssid", "#7"], f"by-name mapping: {mapped} {dropped}")
    check(appmod._translate_class_ids([], src, dst) == ([], []), "empty in, empty out")
    check(appmod._translate_class_ids([1, 1], src, dst) == ([2], []), "no duplicate ids")


def test_data_yaml_id_names_keeps_gaps():
    import app as appmod
    tmp = tempfile.mkdtemp(prefix="step6_h2_yaml_")
    try:
        _write_yaml(tmp, {0: "MC", 2: "PA"})
        p = os.path.join(tmp, "data.yaml")
        check(appmod._data_yaml_id_names(p) == {0: "MC", 2: "PA"}, f"gapped mapping: {appmod._data_yaml_id_names(p)}")
        with open(p, "w") as f:
            f.write("names: [MC, PA]\n")
        check(appmod._data_yaml_id_names(p) == {0: "MC", 1: "PA"}, "inline list form")
        check(appmod._data_yaml_id_names(os.path.join(tmp, "nope.yaml")) == {}, "unreadable -> {}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TASK 1 — Step 6 trains from 4.test export\n")
    print("BEHAVIOR:")
    run(test_list_classes_reads_4test_export)
    print("SOURCE:")
    run(test_run_step6_source_is_4test_not_step5)
    print("CHAIN BRIDGE:")
    run(test_load_project_unlocks_train_after_step3)
    print("TRAINING SOURCE (composite exports, 2026-08-26):")
    run(test_source_dir_outside_composite_root_is_400)
    run(test_source_dir_missing_data_yaml_is_400)
    run(test_valid_composite_source_is_used_and_saved)
    run(test_blank_source_dir_keeps_project_export)
    print("H2 (2026-08-26): class picker follows the training source:")
    run(test_list_classes_follows_source_dir)
    run(test_list_classes_translates_saved_selection_by_name)
    run(test_run_step6_maps_export_ids_into_composite)
    run(test_run_step6_composite_picker_ids_pass_through)
    run(test_run_step6_reports_dropped_names_and_refuses_empty)
    run(test_run_step6_refuses_vanished_picker_source)
    run(test_translate_class_ids_helper)
    run(test_data_yaml_id_names_keeps_gaps)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
