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
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

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
    tmp = tempfile.mkdtemp()
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
    d = tempfile.mkdtemp()
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
    loaded = pm.load_project(d)
    check(loaded["steps"]["6"]["status"] == "pending",
          f"train (6) should unlock after step 3 done; got {loaded['steps']['6']['status']}")


if __name__ == "__main__":
    print("TASK 1 — Step 6 trains from 4.test export\n")
    print("BEHAVIOR:")
    run(test_list_classes_reads_4test_export)
    print("SOURCE:")
    run(test_run_step6_source_is_4test_not_step5)
    print("CHAIN BRIDGE:")
    run(test_load_project_unlocks_train_after_step3)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
