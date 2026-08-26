"""Task 9 (orchestrator side): rounds ledger read route + champion promotion.

evaluate_run.py's append_round_row / ROUNDS_CSV_HEADER are covered by
TR/tests/test_rounds_ledger.py; this file only exercises the two orchestrator
routes that sit on top of the rounds.csv/champion.json files it produces:
  GET  /api/step/7/rounds   -> parsed rows (numeric cols cast) + champion
  POST /api/step/7/promote  -> validates weights + an eval row, writes
                                champion.json

No-pytest harness: run with env/bin/python <this file>.
"""
import os
import sys
import csv
import json
import shutil
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm

_RESULTS = []

ROUNDS_CSV_HEADER = [
    'round', 'at', 'run_name', 'run_dir', 'base_model', 'split_pinned',
    'n_train', 'n_valid', 'n_test',
    'map50_M', 'map50_95_M', 'recall_M', 'precision_M',
    'map50_B', 'map50_95_B',
    'gate_map', 'gate_class', 'gate_recall',
    'per_class_json',
]


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
    project_dir = tmp
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    state = {
        "id": os.path.basename(project_dir), "name": "t", "project_dir": project_dir,
        "created_at": "x", "updated_at": "x", "current_step": 7,
        "steps": steps,
    }
    return state


def _write_rounds_csv(step6_dir, rows):
    os.makedirs(step6_dir, exist_ok=True)
    path = os.path.join(step6_dir, "rounds.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROUNDS_CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ROUNDS_CSV_HEADER})
    return path


def _make_run(step6_dir, run_name, with_best=True):
    run_dir = os.path.join(step6_dir, "runs", run_name)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    if with_best:
        with open(os.path.join(weights_dir, "best.pt"), "wb") as f:
            f.write(b"\x00" * 8)
    return run_dir


def _client_with_project():
    tmp = tempfile.mkdtemp()
    state = _make_project(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    return client, tmp, state


def test_rounds_empty_when_no_ledger():
    client, tmp, state = _client_with_project()
    try:
        r = client.get("/api/step/7/rounds")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("rounds") == [], f"expected empty rounds list: {data}")
        check(data.get("champion") is None, f"expected no champion: {data}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_rounds_parses_rows_and_casts_numerics():
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run1")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "at": "2026-07-09T10:00:00-04:00", "run_name": "run1",
            "run_dir": run_dir, "base_model": "yolo11m-seg.pt", "split_pinned": "true",
            "n_train": 100, "n_valid": 20, "n_test": 10,
            "map50_M": 0.55, "map50_95_M": 0.42, "recall_M": 0.61, "precision_M": 0.70,
            "map50_B": 0.60, "map50_95_B": 0.45,
            "gate_map": "pass", "gate_class": "pass", "gate_recall": "pass",
            "per_class_json": "rounds/round_1_per_class.json",
        }])
        r = client.get("/api/step/7/rounds")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        rows = data.get("rounds") or []
        check(len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}")
        row = rows[0]
        check(row["round"] == 1 and isinstance(row["round"], int), f"round not cast to int: {row}")
        check(row["n_train"] == 100 and isinstance(row["n_train"], int), f"n_train not cast: {row}")
        check(abs(row["map50_95_M"] - 0.42) < 1e-9 and isinstance(row["map50_95_M"], float),
              f"map50_95_M not cast to float: {row}")
        check(row["gate_map"] == "pass", f"gate_map should pass through as string: {row}")
        check(row["run_dir"] == run_dir, f"run_dir mismatch: {row}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_promote_writes_champion_json():
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_good")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "at": "2026-07-09T10:00:00-04:00", "run_name": "run_good",
            "run_dir": run_dir, "base_model": "yolo11m-seg.pt", "split_pinned": "true",
            "n_train": 100, "n_valid": 20, "n_test": 10,
            "map50_M": 0.55, "map50_95_M": 0.415, "recall_M": 0.61, "precision_M": 0.70,
            "map50_B": 0.60, "map50_95_B": 0.45,
            "gate_map": "pass", "gate_class": "pass", "gate_recall": "pass",
            "per_class_json": "rounds/round_1_per_class.json",
        }])

        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code == 200, f"promote failed: {data}")
        check(data.get("success") is True, f"expected success: {data}")
        champ = data.get("champion") or {}
        check(champ.get("run_dir") == os.path.abspath(run_dir), f"champion run_dir mismatch: {champ}")
        check(abs(champ.get("map50_95_M") - 0.415) < 1e-9, f"champion map50_95_M mismatch: {champ}")
        check("promoted_at" in champ, f"champion missing promoted_at: {champ}")
        check(champ.get("note") == "", f"champion note should default to empty string: {champ}")

        champion_path = os.path.join(step6_dir, "champion.json")
        check(os.path.isfile(champion_path), "champion.json was not written")
        with open(champion_path) as f:
            on_disk = json.load(f)
        check(on_disk == champ, f"on-disk champion.json {on_disk} != response {champ}")

        # GET /rounds must now surface the same champion.
        r2 = client.get("/api/step/7/rounds")
        data2 = r2.get_json()
        check(data2.get("champion", {}).get("run_dir") == champ["run_dir"],
              f"GET /rounds champion mismatch: {data2}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_promote_rejects_run_without_weights():
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_no_weights", with_best=False)
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_no_weights", "run_dir": run_dir,
            "map50_95_M": 0.5, "gate_map": "pass",
        }])
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code >= 400, f"expected error status, got {r.status_code}: {data}")
        check("weights" in str(data.get("error", "")).lower(), f"expected a weights-related error: {data}")
        check(not os.path.isfile(os.path.join(step6_dir, "champion.json")),
              "champion.json must not be written when weights are missing")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_promote_rejects_run_without_eval_row():
    """A run with weights but no rounds.csv row at all (never evaluated) must
    be rejected — promotion needs a basis for "the best"."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_unevaluated", with_best=True)
        # No rounds.csv written at all.
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code >= 400, f"expected error status, got {r.status_code}: {data}")
        check("evaluat" in str(data.get("error", "")).lower(),
              f"expected an 'evaluate first' style error: {data}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_list_runs_reports_champion_run_dir():
    """Task 9's step-8 run-picker default: /api/step/6/list_runs surfaces
    champion_run_dir + per-run is_champion once a champion is promoted."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_good")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_good", "run_dir": run_dir,
            "map50_95_M": 0.415, "gate_map": "pass",
        }])
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        check(r.status_code == 200, f"promote failed: {r.get_json()}")

        r2 = client.get("/api/step/6/list_runs")
        data2 = r2.get_json()
        check(r2.status_code == 200, f"list_runs failed: {data2}")
        check(data2.get("champion_run_dir") == os.path.abspath(run_dir),
              f"champion_run_dir mismatch: {data2}")
        matching = [x for x in data2.get("runs", []) if x["name"] == "run_good"]
        check(matching and matching[0]["is_champion"] is True,
              f"run_good should be flagged is_champion: {data2}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("TASK 9 -- rounds ledger routes + champion promotion (orchestrator)\n")
    print("GET /rounds:")
    run(test_rounds_empty_when_no_ledger)
    run(test_rounds_parses_rows_and_casts_numerics)
    print("POST /promote:")
    run(test_promote_writes_champion_json)
    run(test_promote_rejects_run_without_weights)
    run(test_promote_rejects_run_without_eval_row)
    print("STEP 8 PICKER DEFAULT:")
    run(test_list_runs_reports_champion_run_dir)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
