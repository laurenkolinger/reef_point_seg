"""Task T9, Step 7 (internal step 8): the inference run list, the gallery
manifest and image routes keyed by output folder, and the launch command.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step8_runs.py

Routes under test (plan docs/superpowers/plans/2026-09-04-vega-training-foldin.md,
"API contract for wave 2"):
  GET  /api/step/8/runs              every inference run from the open project's
                                     step8_inference/* and from
                                     inprocess/_models/_inference/*, newest first,
                                     with the counts read from each manifest.json
  GET  /api/step/8/manifest?out_dir  the manifest.json object plus ok and
                                     image_base; the project's saved pointer
                                     when no out_dir is given
  GET  /api/step/8/image?out_dir&rel a file inside that output folder only, the
                                     containment checked with commonpath so a
                                     sibling folder whose name extends the
                                     allowed one is refused
  POST /api/step/8/run               the command carries --device from the panel
                                     and --codes_csv from pipeline.yaml, and no
                                     longer carries --save_crops

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the VICARIUS bridge is replaced by a recorder, and
the stage runner's launch is replaced by a capture. Every fixture lives under
one temp folder that is removed at the end, and the removal is verified.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
_FIXTURES = []


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


class _FakeVic:
    """The VICARIUS bridge with every call recorded and nothing written."""
    notes = []

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None

    @classmethod
    def note(cls, message, *_a, **_k):
        cls.notes.append(message)


appmod.vic = _FakeVic()

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
RUN_KEYS = {"out_dir", "name", "created_at", "model_run_name", "n_images", "n_detections", "source_label"}
JPG_BYTES = b"\xff\xd8\xff\xe0fixture"
ISO_OFFSET = "-04:00"


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function and record PASS or FAIL with its traceback."""
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

def _write(path, data):
    """Write text or bytes to path, creating the folder."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(data)


def _make_project_state(project_dir, name="fx"):
    """A loaded-project dict of the shape app.py keeps, with every step dir made."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    return {"id": os.path.basename(project_dir), "name": name, "project_dir": project_dir,
            "created_at": "x", "updated_at": "x", "current_step": 8, "steps": steps}


def _write_inference(out_dir, run_dir, generated_at, n_images, n_detections, source_type, source_dir=""):
    """One inference output folder: manifest.json plus one overlay per image."""
    items = []
    for i in range(n_images):
        rel = f"overlays/img{i}.jpg"
        _write(os.path.join(out_dir, rel), JPG_BYTES + str(i).encode())
        items.append({"filename": f"img{i}.jpg", "raw": f"/src/img{i}.jpg", "overlay": rel,
                      "n_detections": 1 if i < n_detections else 0, "max_conf": 0.5,
                      "detections": [{"class": "PA", "confidence": 0.5}] if i < n_detections else []})
    _write(os.path.join(out_dir, "manifest.json"), json.dumps({
        "items": items, "total_predictions": n_detections, "with_any": n_detections,
        "class_names": {"0": "PA"},
        "args": {"run_dir": run_dir, "out_dir": out_dir, "source_type": source_type,
                 "source_dir": source_dir, "device": "0"},
        "generated_at": generated_at,
    }))
    return out_dir


def _make_fixture():
    """A project with two inference runs and one lab inference run under _models."""
    tmp = tempfile.mkdtemp(prefix="t9_step8_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    project_dir = os.path.join(root, "run_26-09-05_LO_train_fx")
    state = _make_project_state(project_dir)
    step8 = state["steps"]["8"]["dir"]
    run_dir = os.path.join(state["steps"]["6"]["dir"], "runs", "det_run")
    _write(os.path.join(run_dir, "args.yaml"), "task: detect\nimgsz: 256\n")
    _write(os.path.join(run_dir, "weights", "best.pt"), "weights")
    old = _write_inference(os.path.join(step8, "run_a"), run_dir, "2026-09-05T08:00:00-04:00",
                           2, 1, "unused")
    new = _write_inference(os.path.join(step8, "run_ab"), run_dir, "2026-09-05T10:00:00-04:00",
                           3, 3, "dir", "/data/frames")
    lab = _write_inference(os.path.join(root, "_models", "_inference", "20260905_090000_note"),
                           os.path.join(root, "_models", "lab", "runs", "lab_run"),
                           "2026-09-05T09:00:00.500000", 1, 0, "full")
    now = time.time()
    for path, age in ((old, 7200), (lab, 3600), (new, 0)):
        os.utime(os.path.join(path, "manifest.json"), (now - age, now - age))
        os.utime(path, (now - age, now - age))
    return {"tmp": tmp, "root": root, "state": state, "project_dir": project_dir, "step8": step8,
            "run_dir": run_dir, "old": old, "new": new, "lab": lab}


def _client(fx, open_project=True):
    """A test client with the inprocess root pointed at the fixture."""
    appmod._inprocess_root = lambda: fx["root"]
    appmod.current_project = fx["state"] if open_project else None
    return appmod.create_app().test_client()


def _close(fx):
    """Restore the module state and remove the fixture."""
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    shutil.rmtree(fx["tmp"], ignore_errors=True)


_REAL_INPROCESS_ROOT = appmod._inprocess_root


def _query(url):
    """The query parameters of a url as a flat dict."""
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


# ---------------------------------------------------------------------------
# GET /api/step/8/runs
# ---------------------------------------------------------------------------

def test_step8_runs_newest_first_across_both_roots():
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/8/runs")
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"runs: {r.status_code} {data}")
        runs = data.get("runs") or []
        check([x["name"] for x in runs] == ["run_ab", "20260905_090000_note", "run_a"],
              f"newest first across the project and the lab root: {[x['name'] for x in runs]}")
        for row in runs:
            check(set(row) == RUN_KEYS, f"row keys: {sorted(set(row) ^ RUN_KEYS)}")
        newest = runs[0]
        check(newest["out_dir"] == fx["new"] and newest["n_images"] == 3 and newest["n_detections"] == 3,
              f"counts from the manifest: {newest}")
        check(newest["model_run_name"] == "det_run", f"the model run is named: {newest}")
        check(newest["created_at"] == "2026-09-05T10:00:00-04:00", f"created_at ISO AST: {newest}")
        check(newest["source_label"] == "folder /data/frames", f"a directory source names the folder: {newest}")
        lab = runs[1]
        check(lab["created_at"] == "2026-09-05T09:00:00-04:00" and lab["model_run_name"] == "lab_run",
              f"a naive generated_at is stamped AST: {lab}")
        check(lab["source_label"] == "full TCRMP set" and runs[2]["source_label"] == "unused TCRMP frames",
              f"the sampling sources are labelled: {lab} {runs[2]}")
    finally:
        _close(fx)


def test_step8_runs_without_a_project_lists_the_lab_root_only():
    fx = _make_fixture()
    try:
        r = _client(fx, open_project=False).get("/api/step/8/runs")
        data = r.get_json() or {}
        check(r.status_code == 200 and [x["name"] for x in data.get("runs") or []] == ["20260905_090000_note"],
              f"project-free listing: {r.status_code} {data}")
    finally:
        _close(fx)


def test_step8_runs_skips_a_folder_without_a_manifest():
    fx = _make_fixture()
    try:
        os.makedirs(os.path.join(fx["step8"], "half_written"), exist_ok=True)
        _write(os.path.join(fx["step8"], "broken", "manifest.json"), "{not json")
        runs = (_client(fx).get("/api/step/8/runs").get_json() or {}).get("runs") or []
        names = [x["name"] for x in runs]
        check("half_written" not in names and "broken" not in names, f"only readable manifests are listed: {names}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/step/8/manifest
# ---------------------------------------------------------------------------

def test_step8_manifest_accepts_an_out_dir():
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/8/manifest", query_string={"out_dir": fx["lab"]})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"manifest: {r.status_code} {data}")
        check(len(data.get("items") or []) == 1 and data.get("total_predictions") == 0,
              f"the manifest object comes back whole: {data}")
        base = data.get("image_base") or ""
        check(base.startswith("/api/step/8/image?out_dir=") and base.endswith("&rel="),
              f"image_base shape: {base!r}")
        check(_query(base + "x").get("out_dir") == fx["lab"], f"image_base carries the out_dir: {base!r}")
        check(data.get("verdicts") == {}, f"no verdicts file reads as an empty object: {data.get('verdicts')}")
    finally:
        _close(fx)


def test_step8_manifest_carries_the_recorded_verdicts():
    fx = _make_fixture()
    try:
        client = _client(fx)
        for filename, verdict in (("img0.jpg", "yes"), ("img1.jpg", "skip"), ("img0.jpg", "no")):
            r = client.post("/api/step/8/verify", json={"out_dir": fx["old"], "filename": filename, "verdict": verdict})
            check(r.status_code == 200, f"verify {filename} {verdict}: {r.status_code} {r.get_json()}")
        with open(os.path.join(fx["old"], "verifications.csv"), "a", newline="") as fh:
            fh.write("img9.jpg,maybe,,2026-09-05T12:00:00-04:00\n")
        data = client.get("/api/step/8/manifest", query_string={"out_dir": fx["old"]}).get_json() or {}
        check(data.get("verdicts") == {"img0.jpg": "no", "img1.jpg": "skip"},
              f"last row per filename wins and an unknown verdict is skipped: {data.get('verdicts')}")
        check((client.get("/api/step/8/manifest", query_string={"out_dir": fx["new"]}).get_json() or {}).get("verdicts") == {},
              "another run's verdicts do not leak")
    finally:
        _close(fx)


def test_step8_manifest_falls_back_to_the_project_pointer():
    fx = _make_fixture()
    try:
        fx["state"]["steps"]["8"]["outputs"]["manifest"] = os.path.join(fx["old"], "manifest.json")
        fx["state"]["steps"]["8"]["outputs"]["out_dir"] = fx["old"]
        r = _client(fx).get("/api/step/8/manifest")
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True and len(data.get("items") or []) == 2,
              f"pointer fallback: {r.status_code} {data}")
        check(_query(data["image_base"] + "x").get("out_dir") == fx["old"], f"image_base: {data['image_base']}")
        fx["state"]["steps"]["8"]["outputs"] = {}
        r = _client(fx).get("/api/step/8/manifest")
        check(r.status_code == 404, f"no pointer and no out_dir answers 404: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


def test_step8_manifest_refuses_outside_roots_and_reports_a_missing_manifest():
    fx = _make_fixture()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        _write(os.path.join(outside, "manifest.json"), "{}")
        client = _client(fx)
        r = client.get("/api/step/8/manifest", query_string={"out_dir": outside})
        check(r.status_code == 403, f"outside both roots answers 403: {r.status_code} {r.get_json()}")
        empty = os.path.join(fx["step8"], "empty")
        os.makedirs(empty)
        r = client.get("/api/step/8/manifest", query_string={"out_dir": empty})
        data = r.get_json() or {}
        check(r.status_code == 404 and "manifest.json" in str(data.get("error", "")),
              f"a folder without a manifest answers 404 naming the file: {r.status_code} {data}")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# GET /api/step/8/image
# ---------------------------------------------------------------------------

def test_step8_image_serves_an_overlay_by_out_dir():
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/8/image", query_string={"out_dir": fx["lab"], "rel": "overlays/img0.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"0", f"overlay bytes: {r.status_code}")
    finally:
        _close(fx)


def test_step8_image_refuses_the_sibling_prefix_folder():
    """run_a is allowed; run_ab extends its name. The old startswith guard let a
    path under run_ab through; commonpath refuses it."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        os.symlink(os.path.join(fx["new"], "overlays", "img0.jpg"), os.path.join(fx["old"], "overlays", "sib.jpg"))
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "rel": "overlays/sib.jpg"})
        check(r.status_code == 403, f"a symlink into the sibling-prefix folder is refused: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "rel": "../run_ab/overlays/img0.jpg"})
        check(r.status_code == 403, f"a dotdot path into the sibling is refused: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"] + "b", "rel": "overlays/img0.jpg"})
        check(r.status_code == 200, f"the sibling itself is a valid out_dir: {r.status_code}")
    finally:
        _close(fx)


def test_step8_image_refuses_absolute_outside_and_missing():
    fx = _make_fixture()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        client = _client(fx)
        secret = os.path.join(outside, "secret.jpg")
        _write(secret, JPG_BYTES)
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "rel": secret})
        check(r.status_code == 403, f"an absolute rel is refused: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": outside, "rel": "secret.jpg"})
        check(r.status_code == 403, f"an out_dir outside both roots is refused: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "rel": "overlays/nope.jpg"})
        check(r.status_code == 404, f"a missing file answers 404: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"]})
        check(r.status_code == 400, f"a missing rel answers 400: {r.status_code}")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


def test_step8_image_falls_back_to_the_project_pointer():
    fx = _make_fixture()
    try:
        fx["state"]["steps"]["8"]["outputs"]["out_dir"] = fx["old"]
        r = _client(fx).get("/api/step/8/image", query_string={"rel": "overlays/img1.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"1", f"pointer fallback: {r.status_code}")
    finally:
        _close(fx)


def test_step8_image_serves_a_raw_frame_by_manifest_filename():
    """The raw source of a tile lives outside out_dir, so it is never taken
    from the client: the filename is looked up in the run's own manifest and
    the recorded raw path is served."""
    fx = _make_fixture()
    try:
        frame = os.path.join(fx["tmp"], "frames", "img0.jpg")
        _write(frame, JPG_BYTES + b"raw0")
        manifest_path = os.path.join(fx["old"], "manifest.json")
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        manifest["items"][0]["raw"] = frame
        manifest["items"][1]["raw"] = "overlays/img1.jpg"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh)
        client = _client(fx)
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "raw": "1", "filename": "img0.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"raw0", f"the recorded raw path is served: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "raw": "1", "filename": "img1.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"1", f"a raw path relative to out_dir resolves: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "raw": "1", "filename": "not_there.jpg"})
        check(r.status_code == 404, f"a filename the manifest does not list answers 404: {r.status_code} {r.get_json()}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["new"], "raw": "1", "filename": "img0.jpg"})
        check(r.status_code == 403, f"a recorded raw path that is not a readable file answers 403: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "raw": "1"})
        check(r.status_code == 400, f"raw=1 without a filename answers 400: {r.status_code}")
        r = client.get("/api/step/8/image", query_string={"out_dir": fx["old"], "raw": "1", "filename": frame})
        check(r.status_code == 404, f"a client-supplied absolute path is never resolved: {r.status_code}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POST /api/step/8/verify
# ---------------------------------------------------------------------------

def _read_verdicts(out_dir):
    """Rows of <out_dir>/verifications.csv as dicts, [] when absent."""
    import csv
    path = os.path.join(out_dir, "verifications.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_step8_verify_writes_to_the_given_out_dir():
    fx = _make_fixture()
    try:
        fx["state"]["steps"]["8"]["outputs"]["out_dir"] = fx["new"]
        client = _client(fx)
        r = client.post("/api/step/8/verify", json={"out_dir": fx["old"], "filename": "img0.jpg", "verdict": "yes"})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"verify: {r.status_code} {data}")
        rows = _read_verdicts(fx["old"])
        check(len(rows) == 1 and rows[0]["filename"] == "img0.jpg" and rows[0]["verdict"] == "yes",
              f"the verdict goes to the named run: {rows}")
        check(rows[0]["recorded_at"].endswith(ISO_OFFSET), f"recorded_at carries the AST offset: {rows[0]}")
        check(not _read_verdicts(fx["new"]), "the project's newest run is untouched")
        r = client.post("/api/step/8/verify", json={"filename": "img2.jpg", "verdict": "no"})
        check(r.status_code == 200 and _read_verdicts(fx["new"])[0]["filename"] == "img2.jpg",
              f"without out_dir the saved pointer is used: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


def test_step8_verify_refuses_outside_roots_and_a_bad_verdict():
    fx = _make_fixture()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        client = _client(fx)
        r = client.post("/api/step/8/verify", json={"out_dir": outside, "filename": "x.jpg", "verdict": "yes"})
        check(r.status_code == 403, f"outside both roots answers 403: {r.status_code} {r.get_json()}")
        check(not os.path.isfile(os.path.join(outside, "verifications.csv")), "nothing written outside")
        r = client.post("/api/step/8/verify", json={"out_dir": fx["old"], "filename": "x.jpg", "verdict": "maybe"})
        check(r.status_code == 400, f"an unknown verdict answers 400: {r.status_code} {r.get_json()}")
        r = client.post("/api/step/8/verify", json={"out_dir": fx["old"], "verdict": "yes"})
        check(r.status_code == 400, f"a missing filename answers 400: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


# ---------------------------------------------------------------------------
# POST /api/step/8/run
# ---------------------------------------------------------------------------

def _infer(client, config):
    """PUT the Step 8 config, POST the run with the runner stubbed; (response, argv)."""
    captured = {}
    real = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        return {"pid": 8}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/8", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        return client.post("/api/step/8/run"), captured.get("cmd") or []
    finally:
        appmod.runner.run_cli_stage = real


def _argv_value(cmd, flag):
    """The value following flag in an argv list, or None."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_step8_launch_carries_device_and_codes_csv_and_no_save_crops():
    fx = _make_fixture()
    codes = os.path.join(fx["tmp"], "codes.csv")
    _write(codes, "code,category,name\nPA,coral,Porites astreoides\n")
    real_codes = appmod.PATHS.get("master_codes_csv")
    appmod.PATHS["master_codes_csv"] = codes
    try:
        r, cmd = _infer(_client(fx), {"run_dir": fx["run_dir"], "source_type": "dir", "source_dir": "/data/x",
                                      "device": "1", "save_crops": True, "save_overlays": True})
        body = r.get_json() or {}
        check(r.status_code == 200, f"launch: {r.status_code} {body}")
        check(_argv_value(cmd, "--device") == "1", f"--device from the panel: {cmd}")
        check(_argv_value(cmd, "--codes_csv") == codes, f"--codes_csv from pipeline.yaml: {cmd}")
        check("--save_crops" not in cmd, f"--save_crops is gone even when a stale config still says so: {cmd}")
        check("--save_predictions" in cmd and "--save_overlays" in cmd, f"the other flags still ride: {cmd}")
    finally:
        appmod.PATHS["master_codes_csv"] = real_codes
        _close(fx)


def test_step8_launch_omits_device_when_blank_and_warns_on_a_missing_codes_file():
    fx = _make_fixture()
    real_codes = appmod.PATHS.get("master_codes_csv")
    appmod.PATHS["master_codes_csv"] = os.path.join(fx["tmp"], "absent.csv")
    try:
        r, cmd = _infer(_client(fx), {"run_dir": fx["run_dir"], "source_type": "dir", "source_dir": "/data/x",
                                      "device": ""})
        body = r.get_json() or {}
        check(r.status_code == 200, f"launch: {r.status_code} {body}")
        check("--device" not in cmd, f"a blank device leaves the engine default: {cmd}")
        check("--codes_csv" not in cmd, f"a missing codes file is not forwarded: {cmd}")
        check(any("absent.csv" in w for w in body.get("warnings") or []), f"the missing file is reported: {body}")
    finally:
        appmod.PATHS["master_codes_csv"] = real_codes
        _close(fx)


if __name__ == "__main__":
    print("T9 -- inference runs, gallery manifest and image by folder, launch command\n")
    print("GET /api/step/8/runs:")
    run(test_step8_runs_newest_first_across_both_roots)
    run(test_step8_runs_without_a_project_lists_the_lab_root_only)
    run(test_step8_runs_skips_a_folder_without_a_manifest)
    print("GET /api/step/8/manifest:")
    run(test_step8_manifest_accepts_an_out_dir)
    run(test_step8_manifest_carries_the_recorded_verdicts)
    run(test_step8_manifest_falls_back_to_the_project_pointer)
    run(test_step8_manifest_refuses_outside_roots_and_reports_a_missing_manifest)
    print("GET /api/step/8/image:")
    run(test_step8_image_serves_an_overlay_by_out_dir)
    run(test_step8_image_refuses_the_sibling_prefix_folder)
    run(test_step8_image_refuses_absolute_outside_and_missing)
    run(test_step8_image_falls_back_to_the_project_pointer)
    run(test_step8_image_serves_a_raw_frame_by_manifest_filename)
    print("POST /api/step/8/verify:")
    run(test_step8_verify_writes_to_the_given_out_dir)
    run(test_step8_verify_refuses_outside_roots_and_a_bad_verdict)
    print("POST /api/step/8/run:")
    run(test_step8_launch_carries_device_and_codes_csv_and_no_save_crops)
    run(test_step8_launch_omits_device_when_blank_and_warns_on_a_missing_codes_file)

    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    if leftovers:
        print(f"\nFIXTURES NOT REMOVED: {leftovers}")
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
    sys.exit(1 if failed or leftovers else 0)
