"""Task T9, Step 6 (internal step 7): the evaluation report keyed by run.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step7_report_by_run.py

Routes under test (plan docs/superpowers/plans/2026-09-04-vega-training-foldin.md,
"API contract for wave 2"):
  GET  /api/step/7/report?run_dir   the newest evaluation folder for that run,
                                    searched in the run's project
                                    step7_evaluateModel/eval_* and in
                                    inprocess/_models/_eval/<run>_*; falls back
                                    to the project's saved pointer when no
                                    run_dir is given
  GET  /api/step/7/image?run_dir&rel a file inside that evaluation folder only
  GET  /api/step/7/pdf?run_dir       the PDF as an attachment, or 404
  POST /api/step/7/copy_pdf          copies the PDF for a run to a folder
  POST /api/step/7/run               a blank image size auto-fills from the
                                    run's args.yaml; a non-integer is 400
  GET  /api/step/7/run_info?path     the run's trained image size plus the
                                    recommended measurement settings behind
                                    the panel's Use recommended defaults

Also pinned here: the Evaluate step measures at confidence 0.001 while the
Inference step deploys at 0.25, the rendered panel says why, and the button
and the stored default carry the same numbers.

Nothing here reaches a real process: subprocess.run and subprocess.Popen are
stubbed at module level, the VICARIUS bridge is replaced by a recorder, and
the stage runner's launch is replaced by a capture. Every fixture lives under
one temp folder that is removed at the end, and the removal is verified.
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

import io
import json
import os
import re
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
REPORT_KEYS = {"ok", "run_name", "run_dir", "out_dir", "evaluated_at", "report_md",
               "metrics", "previews", "figures", "pdf_url"}
METRICS_KEYS = {"overall", "per_class", "per_class_instances", "absent_classes",
                "metric_family", "class_names", "class_full_names"}
# What a staged dataset's names table writes into metrics.json, and what the report
# card shows beside the class token.
FISH_TOKEN = "EPST_Dark"
FISH_FULL_NAME = "Epinephelus striatus, Nassau grouper, dark phase"
FIGURE_KEYS = {"box_pr", "box_f1", "mask_pr", "mask_f1", "confusion_norm", "confusion_abs", "results"}
PREVIEW_KEYS = {"filename", "url", "n_gt", "n_pred"}
PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"
JPG_BYTES = b"\xff\xd8\xff\xe0fixture"
PDF_BYTES = b"%PDF-1.4\nfixture\n%%EOF\n"
ISO_OFFSET = "-04:00"
# What the Evaluate step measures with, and what the Inference step deploys
# with. Average precision is the area under the whole precision-recall curve,
# so the evaluation keeps almost every prediction; inference keeps only the
# confident ones.
EVAL_CONF = 0.001
EVAL_IOU = 0.6
EVAL_PREVIEWS = 8
INFERENCE_CONF = 0.25
ORCHESTRATOR_JS = os.path.join(_ORCH, "static", "orchestrator.js")
# The phrases the Evaluate confidence tooltip owes a reader, lower-cased. The
# contrast with the Inference step sits in the grey helper line under the
# field instead, where it is read without hovering.
CONF_TOOLTIP_PHRASES = ("0.001", "understate")  # 2026-09-17: "precision-recall curve" dropped as jargon under the no-paragraph conversion
CONF_HELPER_PHRASE = "Model Inference step uses a high threshold"
# The confidence a project saved before 2026-09-15, which the panel replaces
# with the measurement default when the project is opened.
RETIRED_CONF = 0.25


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
            "created_at": "x", "updated_at": "x", "current_step": 7, "steps": steps}


def _write_eval(out_dir, run_dir, generated_at, previews, val_run_files, pdf=False, family="box"):
    """One evaluation folder: metrics.json, report.md, previews/, val_run/, optional report.pdf."""
    preview_records = []
    for name, n_gt, n_pred in previews:
        overlay = os.path.join(out_dir, "previews", f"{name}_sbs.jpg")
        _write(overlay, JPG_BYTES + name.encode())
        preview_records.append({"source": f"/src/{name}.jpg", "overlay": overlay,
                                "filename": f"{name}.jpg", "n_gt": n_gt, "n_pred": n_pred})
    for fn in val_run_files:
        _write(os.path.join(out_dir, "val_run", fn), PNG_BYTES + fn.encode())
    _write(os.path.join(out_dir, "report.md"), f"# Report for {os.path.basename(run_dir)}\n")
    _write(os.path.join(out_dir, "metrics.json"), json.dumps({
        "metrics": {
            "wall_seconds": 1.5, "weights": os.path.join(run_dir, "weights", "best.pt"),
            "class_names": {"0": "Acropora palmata"},
            "overall": {"box_map50": 0.5, "box_map50_95": 0.4, "mask_map50": None if family == "box" else 0.3},
            "per_class": {"Acropora palmata": {"box_map50": 0.5}},
            "per_class_instances": {"Acropora palmata": 3},
            "absent_classes": ["Porites astreoides"],
            "metric_family": family,
        },
        "context": {"run_dir": run_dir},
        "previews": preview_records,
        "args": {"imgsz": 256},
        "generated_at": generated_at,
    }))
    if pdf:
        _write(os.path.join(out_dir, "report.pdf"), PDF_BYTES)
    return out_dir


def _make_fixture():
    """A project run with an older evaluation in its own project and a newer one
    under _models/_eval, plus a run that was never evaluated."""
    tmp = tempfile.mkdtemp(prefix="t9_report_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    root = os.path.join(tmp, "inprocess")
    project_dir = os.path.join(root, "run_26-09-05_LO_train_fx")
    state = _make_project_state(project_dir)
    step6 = state["steps"]["6"]["dir"]
    step7 = state["steps"]["7"]["dir"]
    run_dir = os.path.join(step6, "runs", "seg_run")
    _write(os.path.join(run_dir, "args.yaml"), "task: detect\nimgsz: 640\n")
    _write(os.path.join(run_dir, "weights", "best.pt"), "weights")
    _write(os.path.join(run_dir, "results.png"), PNG_BYTES + b"results")
    unscored = os.path.join(step6, "runs", "unscored")
    _write(os.path.join(unscored, "args.yaml"), "task: detect\nimgsz: 320\n")
    _write(os.path.join(unscored, "weights", "best.pt"), "weights")
    os.makedirs(os.path.join(step6, "dataset"), exist_ok=True)

    older = _write_eval(os.path.join(step7, "eval_seg_run"), run_dir,
                        "2026-09-05T08:00:00-04:00", [("a", 2, 1)], ["BoxPR_curve.png"])
    newer = _write_eval(os.path.join(root, "_models", "_eval", "seg_run_20260905_100000"), run_dir,
                        "2026-09-05T10:00:00.123456", [("b", 3, 4)],
                        ["BoxPR_curve.png", "confusion_matrix.png"], pdf=True)
    now = time.time()
    for path, age in ((older, 7200), (newer, 0)):
        for name in ("metrics.json", "report.md"):
            os.utime(os.path.join(path, name), (now - age, now - age))
        os.utime(path, (now - age, now - age))
    return {"tmp": tmp, "root": root, "state": state, "project_dir": project_dir,
            "step7": step7, "run_dir": run_dir, "unscored": unscored, "older": older, "newer": newer}


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
# GET /api/step/7/report
# ---------------------------------------------------------------------------

def test_report_forwards_the_species_names_the_evaluation_wrote():
    """class_full_names reaches the card intact, and a metrics.json without the key forwards an empty map."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        plain = (client.get("/api/step/7/report", query_string={"run_dir": fx["run_dir"]}).get_json() or {})
        check(plain.get("metrics", {}).get("class_full_names") == {},
              f"an evaluation from before the names table forwards an empty map: {plain.get('metrics', {}).get('class_full_names')}")
        path = os.path.join(fx["newer"], "metrics.json")
        with open(path) as fh:
            doc = json.load(fh)
        doc["metrics"]["class_names"] = {"0": FISH_TOKEN}
        doc["metrics"]["class_full_names"] = {"0": FISH_FULL_NAME}
        _write(path, json.dumps(doc))
        named = (client.get("/api/step/7/report", query_string={"run_dir": fx["run_dir"]}).get_json() or {})
        check(named.get("metrics", {}).get("class_full_names") == {"0": FISH_FULL_NAME},
              f"the dataset's species names reach the card: {named.get('metrics', {}).get('class_full_names')}")
        check(named["metrics"]["class_names"] == {"0": FISH_TOKEN},
              f"the class tokens ride along beside the names: {named['metrics']['class_names']}")
        doc["metrics"]["class_full_names"] = "not a map"
        _write(path, json.dumps(doc))
        hostile = (client.get("/api/step/7/report", query_string={"run_dir": fx["run_dir"]}).get_json() or {})
        check(hostile.get("metrics", {}).get("class_full_names") == {},
              f"a names map of the wrong type is forwarded as empty, not as text: {hostile.get('metrics', {}).get('class_full_names')}")
    finally:
        _close(fx)


def test_report_resolves_by_run_dir_newest_wins():
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/7/report", query_string={"run_dir": fx["run_dir"]})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"report: {r.status_code} {data}")
        check(set(data) == REPORT_KEYS, f"response keys: {sorted(set(data) ^ REPORT_KEYS)}")
        check(data["out_dir"] == fx["newer"], f"the newest evaluation wins: {data['out_dir']}")
        check(data["run_name"] == "seg_run" and data["run_dir"] == fx["run_dir"], f"run identity: {data}")
        check(data["report_md"].startswith("# Report for seg_run"), f"report_md: {data['report_md']!r}")
        check(data["evaluated_at"] == "2026-09-05T10:00:00-04:00",
              f"a naive generated_at is stamped AST at second precision: {data['evaluated_at']!r}")
        check(set(data["metrics"]) == METRICS_KEYS, f"metrics keys: {sorted(set(data['metrics']) ^ METRICS_KEYS)}")
        check(data["metrics"]["metric_family"] == "box" and data["metrics"]["absent_classes"] == ["Porites astreoides"],
              f"metrics pass through: {data['metrics']}")
        previews = data["previews"]
        check(len(previews) == 1 and set(previews[0]) == PREVIEW_KEYS, f"previews: {previews}")
        check(previews[0]["filename"] == "b.jpg" and previews[0]["n_gt"] == 3 and previews[0]["n_pred"] == 4,
              f"the evaluator's counts ride along: {previews[0]}")
        q = _query(previews[0]["url"])
        check(previews[0]["url"].startswith("/api/step/7/image?") and q.get("run_dir") == fx["run_dir"]
              and q.get("rel") == "previews/b_sbs.jpg", f"preview url: {previews[0]['url']}")
        figures = data["figures"]
        check(set(figures) == FIGURE_KEYS, f"figure keys: {sorted(set(figures) ^ FIGURE_KEYS)}")
        check(_query(figures["box_pr"]).get("rel") == "val_run/BoxPR_curve.png", f"box_pr: {figures['box_pr']}")
        check(_query(figures["confusion_abs"]).get("rel") == "val_run/confusion_matrix.png", f"{figures}")
        check(figures["mask_pr"] is None and figures["mask_f1"] is None and figures["box_f1"] is None
              and figures["confusion_norm"] is None, f"missing figures are null: {figures}")
        check(_query(figures["results"]).get("rel") == "run/results.png", f"results from the run: {figures['results']}")
        check(_query(data["pdf_url"]).get("run_dir") == fx["run_dir"] and data["pdf_url"].startswith("/api/step/7/pdf?"),
              f"pdf_url present when report.pdf exists: {data['pdf_url']}")
    finally:
        _close(fx)


def test_report_falls_back_to_the_old_pointer_without_run_dir():
    fx = _make_fixture()
    try:
        fx["state"]["steps"]["7"]["outputs"]["out_dir"] = fx["older"]
        fx["state"]["steps"]["7"]["config"]["run_dir"] = fx["run_dir"]
        r = _client(fx).get("/api/step/7/report")
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"fallback report: {r.status_code} {data}")
        check(data["out_dir"] == fx["older"], f"the saved pointer is used when no run_dir is given: {data['out_dir']}")
        check(data["pdf_url"] is None, f"no PDF in the older folder: {data['pdf_url']}")
        check(data["evaluated_at"] == "2026-09-05T08:00:00-04:00", f"evaluated_at: {data['evaluated_at']}")
        check(data["previews"][0]["filename"] == "a.jpg", f"previews from that folder: {data['previews']}")
    finally:
        _close(fx)


def test_report_404_when_no_evaluation_exists():
    fx = _make_fixture()
    try:
        client = _client(fx)
        r = client.get("/api/step/7/report", query_string={"run_dir": fx["unscored"]})
        data = r.get_json() or {}
        check(r.status_code == 404 and "unscored" in str(data.get("error", "")),
              f"a never-evaluated run answers 404 naming the run: {r.status_code} {data}")
        check(data.get("ready") is False, f"the empty-state marker rides along: {data}")
        r = client.get("/api/step/7/report")
        data = r.get_json() or {}
        check(r.status_code == 404 and "evaluat" in str(data.get("error", "")).lower(),
              f"no pointer and no run_dir answers 404: {r.status_code} {data}")
        check(data.get("ready") is False, f"the empty-state marker rides along: {data}")
    finally:
        _close(fx)


def test_report_refuses_a_run_outside_both_roots():
    fx = _make_fixture()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        stray = os.path.join(outside, "runs", "stray")
        _write(os.path.join(stray, "weights", "best.pt"), "w")
        r = _client(fx).get("/api/step/7/report", query_string={"run_dir": stray})
        check(r.status_code == 403, f"outside both roots answers 403: {r.status_code} {r.get_json()}")
        r = _client(fx).get("/api/step/7/report", query_string={"run_dir": os.path.join(fx["root"], "nope")})
        data = r.get_json() or {}
        check(r.status_code == 404 and "ready" not in data,
              f"a missing run folder inside the roots answers 404 without the empty-state marker: {r.status_code} {data}")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


def test_figures_are_all_null_without_a_val_run_folder():
    fx = _make_fixture()
    try:
        shutil.rmtree(os.path.join(fx["newer"], "val_run"))
        os.remove(os.path.join(fx["run_dir"], "results.png"))
        data = _client(fx).get("/api/step/7/report", query_string={"run_dir": fx["run_dir"]}).get_json() or {}
        check(data.get("out_dir") == fx["newer"], f"still the newest folder: {data.get('out_dir')}")
        check(all(v is None for v in data["figures"].values()), f"every figure null: {data['figures']}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/step/7/pdf and POST /api/step/7/copy_pdf
# ---------------------------------------------------------------------------

def test_pdf_route_serves_the_newest_pdf_or_404():
    fx = _make_fixture()
    try:
        client = _client(fx)
        r = client.get("/api/step/7/pdf", query_string={"run_dir": fx["run_dir"]})
        check(r.status_code == 200 and r.data == PDF_BYTES, f"pdf bytes: {r.status_code} {r.data[:20]!r}")
        check("seg_run" in (r.headers.get("Content-Disposition") or ""),
              f"the download is named after the run: {r.headers.get('Content-Disposition')}")
        os.remove(os.path.join(fx["newer"], "report.pdf"))
        r = client.get("/api/step/7/pdf", query_string={"run_dir": fx["run_dir"]})
        data = r.get_json() or {}
        check(r.status_code == 404 and "report.pdf" in str(data.get("error", "")),
              f"no PDF answers 404 JSON naming the file: {r.status_code} {data}")
    finally:
        _close(fx)


def test_copy_pdf_by_run_dir():
    fx = _make_fixture()
    try:
        dst = os.path.join(fx["tmp"], "exports")
        r = _client(fx).post("/api/step/7/copy_pdf", json={"run_dir": fx["run_dir"], "dst_dir": dst})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("ok") is True, f"copy: {r.status_code} {data}")
        check(data.get("dst") == os.path.join(dst, "seg_run_report.pdf") and os.path.isfile(data["dst"]),
              f"the copy is named after the run: {data}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# GET /api/step/7/image
# ---------------------------------------------------------------------------

def test_image_serves_a_preview_and_a_run_figure():
    fx = _make_fixture()
    try:
        client = _client(fx)
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "previews/b_sbs.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"b", f"preview bytes: {r.status_code}")
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "val_run/BoxPR_curve.png"})
        check(r.status_code == 200 and r.data.startswith(PNG_BYTES), f"curve bytes: {r.status_code}")
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "run/results.png"})
        check(r.status_code == 200 and r.data == PNG_BYTES + b"results", f"results.png from the run: {r.status_code}")
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "run/args.yaml"})
        check(r.status_code == 403, f"only results.png may be read from the run folder: {r.status_code}")
    finally:
        _close(fx)


def test_image_refuses_traversal_absolute_symlink_and_outside():
    fx = _make_fixture()
    outside = tempfile.mkdtemp(prefix="t9_outside_", dir=_TMP_ROOT)
    _FIXTURES.append(outside)
    try:
        client = _client(fx)
        secret = os.path.join(outside, "secret.jpg")
        _write(secret, JPG_BYTES)
        cases = {
            "traversal": {"run_dir": fx["run_dir"], "rel": "../../../../" + os.path.relpath(secret, "/")},
            "dotdot inside": {"run_dir": fx["run_dir"], "rel": "previews/../../eval_other/x.jpg"},
            "absolute": {"run_dir": fx["run_dir"], "rel": secret},
            "run outside roots": {"run_dir": outside, "rel": "secret.jpg"},
        }
        for label, query in cases.items():
            r = client.get("/api/step/7/image", query_string=query)
            check(r.status_code == 403, f"{label} answers 403: {r.status_code} {r.get_json()}")
        # A symlink inside the evaluation folder that points out of it.
        os.symlink(secret, os.path.join(fx["newer"], "previews", "escape.jpg"))
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "previews/escape.jpg"})
        check(r.status_code == 403, f"a symlink out of the folder answers 403: {r.status_code}")
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "previews/missing.jpg"})
        check(r.status_code == 404, f"a missing file answers 404: {r.status_code}")
        r = client.get("/api/step/7/image", query_string={"run_dir": fx["run_dir"]})
        check(r.status_code == 400, f"a missing rel answers 400: {r.status_code}")
    finally:
        _close(fx)
        shutil.rmtree(outside, ignore_errors=True)


def test_step7_image_refuses_a_sibling_prefix_path():
    """A sibling evaluation folder whose name extends the allowed one passed the
    old startswith guard; commonpath refuses it."""
    fx = _make_fixture()
    try:
        sibling = fx["newer"] + "_extra"
        _write(os.path.join(sibling, "previews", "x.jpg"), JPG_BYTES)
        os.symlink(os.path.join(sibling, "previews", "x.jpg"), os.path.join(fx["newer"], "previews", "sib.jpg"))
        r = _client(fx).get("/api/step/7/image", query_string={"run_dir": fx["run_dir"], "rel": "previews/sib.jpg"})
        check(r.status_code == 403, f"the sibling-prefix folder is refused: {r.status_code} {r.get_json()}")
    finally:
        _close(fx)


def test_image_falls_back_to_the_pointer_without_run_dir():
    fx = _make_fixture()
    try:
        fx["state"]["steps"]["7"]["outputs"]["out_dir"] = fx["older"]
        r = _client(fx).get("/api/step/7/image", query_string={"rel": "previews/a_sbs.jpg"})
        check(r.status_code == 200 and r.data == JPG_BYTES + b"a", f"pointer fallback: {r.status_code}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# POST /api/step/7/run: image size
# ---------------------------------------------------------------------------

def _evaluate(client, config):
    """PUT the Step 7 config, POST the run with the runner stubbed; (response, argv)."""
    captured = {}
    real = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = list(cmd)
        return {"pid": 7}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/7", json=config)
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        return client.post("/api/step/7/run"), captured.get("cmd") or []
    finally:
        appmod.runner.run_cli_stage = real


def _argv_value(cmd, flag):
    """The value following flag in an argv list, or None."""
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


def test_blank_imgsz_auto_fills_from_the_runs_args_yaml():
    fx = _make_fixture()
    try:
        client = _client(fx)
        for blank in ("", None, 0, "0"):
            r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": blank})
            body = r.get_json() or {}
            check(r.status_code == 200, f"blank {blank!r} launches: {r.status_code} {body}")
            check(_argv_value(cmd, "--imgsz") == "640", f"blank {blank!r} auto-fills 640: {cmd}")
            check(any("640" in w for w in body.get("warnings") or []), f"the auto-fill is reported: {body}")
        check(appmod.current_project["steps"]["7"]["config"]["imgsz"] == 640, "the chosen size is saved")
    finally:
        _close(fx)


def test_blank_imgsz_without_a_recorded_size_falls_back_and_says_so():
    fx = _make_fixture()
    try:
        _write(os.path.join(fx["unscored"], "args.yaml"), "task: detect\n")
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["unscored"], "imgsz": ""})
        body = r.get_json() or {}
        check(r.status_code == 200 and _argv_value(cmd, "--imgsz") == "1024", f"fallback 1024: {cmd} {body}")
        check(any("1024" in w for w in body.get("warnings") or []), f"the fallback is reported: {body}")
    finally:
        _close(fx)


def test_non_integer_imgsz_is_400():
    fx = _make_fixture()
    try:
        client = _client(fx)
        for bad in ("abc", "512px", "12.5", -1, "-8"):
            r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": bad})
            body = r.get_json() or {}
            check(r.status_code == 400, f"{bad!r} is refused with 400: {r.status_code} {body}")
            check("image size" in str(body.get("error", "")).lower(), f"the error names the field: {body}")
            check(not cmd, f"nothing launched for {bad!r}")
    finally:
        _close(fx)


def test_typed_imgsz_is_used_as_given():
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["run_dir"], "imgsz": 320})
        check(r.status_code == 200 and _argv_value(cmd, "--imgsz") == "320", f"typed size wins: {cmd}")
    finally:
        _close(fx)


def test_detector_note_names_the_box_family():
    fx = _make_fixture()
    try:
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["run_dir"], "imgsz": 640})
        warnings = " ".join((r.get_json() or {}).get("warnings") or [])
        check("box" in warnings.lower() and "mask" in warnings.lower(),
              f"a detector evaluation says it is scored on box metrics and why the mask columns are blank: {warnings!r}")
        check("nothing is wrong" not in warnings.lower(), f"the old reassurance is gone: {warnings!r}")
    finally:
        _close(fx)


# ---------------------------------------------------------------------------
# What a score was measured with: confidence, overlap, image size
# ---------------------------------------------------------------------------

def _rendered_page():
    """The orchestrator page as HTML, rendered with no project open."""
    appmod.current_project = None
    resp = appmod.create_app().test_client().get("/")
    check(resp.status_code == 200, f"GET / answered {resp.status_code}")
    return resp.get_data(as_text=True)


def _input_tag(html, element_id):
    """The whole <input> tag carrying element_id, or '' when it is absent."""
    found = re.search(r'<input[^>]*id="%s"[^>]*>' % re.escape(element_id), html)
    return found.group(0) if found else ""


def _title_of(tag):
    """The title attribute of one tag, or '' when it carries none."""
    found = re.search(r'title="([^"]*)"', tag)
    return found.group(1) if found else ""


def test_evaluate_defaults_keep_the_low_confidence_tail():
    """Evaluate measures at 0.001; Inference keeps its 0.25 deployment threshold."""
    step7 = appmod.DEFAULT_STEP_CONFIGS["7"]
    step8 = appmod.DEFAULT_STEP_CONFIGS["8"]
    check(float(step7["conf_threshold"]) == EVAL_CONF,
          f"the Evaluate default keeps the low-confidence tail: {step7['conf_threshold']}")
    check(appmod.EVAL_CONF_DEFAULT == EVAL_CONF,
          f"app.py reads its fallback from the same config: {appmod.EVAL_CONF_DEFAULT}")
    check(float(step8["conf_threshold"]) == INFERENCE_CONF,
          f"the Inference default is a deployment threshold and stays: {step8['conf_threshold']}")
    check(float(step7["iou_threshold"]) == EVAL_IOU and appmod.EVAL_IOU_DEFAULT == EVAL_IOU,
          f"the Evaluate overlap default is unchanged: {step7['iou_threshold']}")


def test_a_config_that_cannot_be_read_is_refused_by_name():
    """A pipeline.yaml whose Evaluate default is missing or not a number fails
    loudly, naming the file and the key, rather than measuring at a number
    nobody chose."""
    real = appmod.DEFAULT_STEP_CONFIGS
    for broken in ({"7": {}}, {"7": {"conf_threshold": "very low"}}, {"7": {"conf_threshold": None}}):
        appmod.DEFAULT_STEP_CONFIGS = broken
        try:
            appmod._eval_default("conf_threshold", float)
            check(False, f"{broken} must be refused")
        except RuntimeError as exc:
            check("conf_threshold" in str(exc) and "pipeline.yaml" in str(exc),
                  f"the refusal names the key and the file: {exc}")
        finally:
            appmod.DEFAULT_STEP_CONFIGS = real
    check(appmod._eval_default("conf_threshold", float) == EVAL_CONF,
          "the real config still reads back after the broken ones")


def test_run_info_offers_the_recommended_measurement_settings():
    """The answer behind Use recommended defaults carries the same numbers the config holds."""
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/7/run_info", query_string={"path": fx["run_dir"]})
        data = r.get_json() or {}
        check(r.status_code == 200, f"run_info answered {r.status_code}: {data}")
        rec = data.get("recommended") or {}
        check(rec.get("conf_threshold") == EVAL_CONF,
              f"the recommended answer carries the measurement confidence: {rec}")
        check(rec.get("iou_threshold") == EVAL_IOU and rec.get("preview_count") == EVAL_PREVIEWS,
              f"the rest of the recommended answer: {rec}")
        check(rec.get("split") == "test", f"the recommended split: {rec}")
        check(data.get("imgsz") == 640, f"the run's own trained size still rides along: {data}")
    finally:
        _close(fx)


def test_evaluation_forwards_the_confidence_the_panel_holds():
    """A typed confidence reaches the evaluator; an untouched panel launches at 0.001."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": 640})
        check(r.status_code == 200, f"the untouched panel launches: {r.status_code} {r.get_json()}")
        check(_argv_value(cmd, "--conf") == str(EVAL_CONF),
              f"a config that names no confidence measures at {EVAL_CONF}: {cmd}")
        check(_argv_value(cmd, "--iou") == str(EVAL_IOU), f"and at the overlap default: {cmd}")
        r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": 640,
                                    "conf_threshold": 0.4, "iou_threshold": 0.55})
        check(r.status_code == 200, f"the typed panel launches: {r.status_code} {r.get_json()}")
        check(_argv_value(cmd, "--conf") == "0.4", f"the typed confidence is forwarded: {cmd}")
        check(_argv_value(cmd, "--iou") == "0.55", f"the typed overlap is forwarded: {cmd}")
    finally:
        _close(fx)


def test_a_project_still_holding_the_old_cut_off_is_told():
    """A project created before the fix kept 0.25 in its own config; the launch
    says the score will read low and how to fix it, and stays quiet at 0.001."""
    fx = _make_fixture()
    try:
        client = _client(fx)
        r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": 640, "conf_threshold": 0.25})
        warnings = " ".join((r.get_json() or {}).get("warnings") or [])
        check(_argv_value(cmd, "--conf") == "0.25",
              f"the saved value is still honoured, never silently rewritten: {cmd}")
        check("0.25" in warnings and "Use recommended defaults" in warnings,
              f"the warning names the value and the way out: {warnings!r}")
        check("lower than these weights earn" in warnings,
              f"and says what it costs: {warnings!r}")
        r, cmd = _evaluate(client, {"run_dir": fx["run_dir"], "imgsz": 640,
                                    "conf_threshold": EVAL_CONF})
        warnings = " ".join((r.get_json() or {}).get("warnings") or [])
        check("Use recommended defaults" not in warnings,
              f"a measurement at {EVAL_CONF} draws no warning: {warnings!r}")
        # A hand-edited project.json can hold anything; the warning never fires
        # on a value that is not a number, which the evaluator itself refuses.
        for junk in ("high", None, [0.25]):
            check(appmod._above_measurement_ceiling(junk) is False,
                  f"{junk!r} is not a number and draws no warning")
    finally:
        _close(fx)


def test_the_evaluate_panel_says_why_the_confidence_is_so_low():
    """The rendered Evaluate panel offers 0.001 and explains it; Inference still shows 0.25."""
    html = _rendered_page()
    conf_tag = _input_tag(html, "s7-conf")
    check('value="0.001"' in conf_tag, f"the Evaluate confidence field opens on 0.001: {conf_tag!r}")
    tooltip = _title_of(conf_tag).lower()
    missing = [p for p in CONF_TOOLTIP_PHRASES if p not in tooltip]
    check(not missing, f"the confidence tooltip owes the reader {missing}: {tooltip!r}")
    check(not tooltip.startswith("you"), f"a tooltip never opens with You: {tooltip!r}")
    check(tooltip.count(". ") <= 1,
          f"the confidence tooltip stays within two sentences: {tooltip!r}")
    longest = max((len(s.split()) for s in tooltip.split(". ")), default=0)
    check(longest <= 30,
          f"no tooltip sentence does two jobs at once ({longest} words): {tooltip!r}")
    # 2026-09-17: the design spec (v2, section 6) forbids explanatory prose on screen, so the
    # contrast with Inference now lives in the field's tooltip rather than under the field.
    on_screen = re.sub(r'title="[^"]*"', "", html)
    check(CONF_HELPER_PHRASE.lower() in tooltip and CONF_HELPER_PHRASE not in on_screen,
          f"the contrast with Inference is in the confidence tooltip and nowhere on screen: {CONF_HELPER_PHRASE!r}")
    iou_tooltip = _title_of(_input_tag(html, "s7-iou")).lower()
    check("0.6" in iou_tooltip and "duplicate" in iou_tooltip,
          f"the overlap tooltip states its default and what it does: {iou_tooltip!r}")
    check('value="0.25"' in _input_tag(html, "s8-conf"),
          "the Inference confidence stays a deployment threshold")
    # 2026-09-17: the on-screen "conf=0.001, iou=0.6" banner went into the button's tooltip
    # under the no-paragraph rule, so the recommended values are pinned there.
    m = re.search(r'<button[^>]*title="([^"]*)"[^>]*>\s*Use recommended defaults\s*</button>', html, re.S)
    rec = (m.group(1) if m else "").lower()
    check(m is not None and "0.001" in rec and "0.25" not in rec,
          f"the recommended-defaults button offers the same number the field holds: {rec!r}")


def test_the_shipped_evaluation_preset_offers_the_same_confidence():
    """Import on the shipped preset must not put the understated cut-off back."""
    fx = _make_fixture()
    try:
        r = _client(fx).get("/api/step/7/presets/standard.yaml")
        data = r.get_json() or {}
        check(r.status_code == 200, f"the shipped preset loads: {r.status_code} {data}")
        params = data.get("params") or {}
        check(float(params.get("conf_threshold")) == EVAL_CONF,
              f"the preset measures at the same confidence as the config: {params}")
        check(float(params.get("iou_threshold")) == EVAL_IOU, f"and the same overlap: {params}")
    finally:
        _close(fx)


def test_the_recommended_defaults_button_offers_the_same_confidence():
    """applyStep7Defaults() fills the field from EVAL_DEFAULTS, which must agree with the config."""
    with io.open(ORCHESTRATOR_JS, encoding="utf-8") as fh:
        js = fh.read()
    found = re.search(r"const EVAL_DEFAULTS = Object\.freeze\(\{([^}]*)\}\)", js)
    check(found is not None, f"EVAL_DEFAULTS not found in {ORCHESTRATOR_JS}")
    block = found.group(1) if found else ""
    check(f"conf_threshold: {EVAL_CONF}" in block,
          f"the button offers the measurement confidence: {block!r}")
    check(f"iou_threshold: {EVAL_IOU}" in block, f"and the overlap default: {block!r}")


def _step7_block_of(function_marker, end_marker):
    """The lines of orchestrator.js between two markers, as one string.

    Parameters:
        function_marker: the comment or line that opens the block.
        end_marker: the comment or line that closes it.
    Returns:
        The text between them.
    Raises:
        AssertionError naming the marker that could not be found.
    """
    with io.open(ORCHESTRATOR_JS, encoding="utf-8") as fh:
        js = fh.read()
    start = js.find(function_marker)
    check(start >= 0, f"{function_marker!r} not found in {ORCHESTRATOR_JS}")
    end = js.find(end_marker, start)
    check(end > start, f"{end_marker!r} not found after {function_marker!r}")
    return js[start:end]


def test_the_panel_fills_its_evaluate_fields_from_one_place():
    """populateConfigs() holds no second copy of the Evaluate defaults: a
    project whose step 7 block names no confidence must still open on 0.001."""
    block = _step7_block_of("// Step 7: evaluation", "// Step 8: inference")
    check("EVAL_DEFAULTS.conf_threshold" in block,
          f"the confidence fallback reads the frozen constant: {block!r}")
    check("EVAL_DEFAULTS.iou_threshold" in block and "EVAL_DEFAULTS.preview_count" in block,
          f"so do the overlap and preview-count fallbacks: {block!r}")
    stray = [lit for lit in (str(RETIRED_CONF), "0.6", ": 8") if lit in block]
    check(not stray, f"no Evaluate default is repeated as a literal here: {stray} in {block!r}")


def test_the_recommended_answer_from_the_server_fills_the_form():
    """applyStep7Defaults() prefers the run_info answer and falls back to the
    frozen constant only when that call fails, so one source of truth wins."""
    block = _step7_block_of("function applyStep7Defaults", "\n}\n")
    check("step7RecommendedDefaults(" in block,
          f"the button asks the server what it recommends: {block!r}")
    source = _step7_block_of("async function step7RecommendedDefaults", "\n}\n")
    check(".recommended" in source and "run_info" in source,
          f"that answer comes from the run_info route: {source!r}")
    check("EVAL_DEFAULTS" in source,
          f"and the frozen constant is the fallback when the call fails: {source!r}")


def test_the_rounds_table_never_ranks_two_measurements():
    """The Rounds table draws a delta only between rows measured the same way,
    and every score cell says what it was measured with."""
    block = _step7_block_of("async function refreshStep7Rounds", "\n}\n")
    check("roundsComparable(" in block,
          f"the delta is guarded by the comparability rule: {block!r}")
    check("roundsMeasuredAt(r)" in block,
          f"and each score cell carries what the row was measured with: {block!r}")
    rule = _step7_block_of("function roundsComparable", "\n}\n")
    check("prevConf == null && conf == null" in rule,
          f"two rows that both record nothing compare as they did before: {rule!r}")
    check("ROUNDS_CONF_TOLERANCE" in rule,
          f"and a CSV round trip does not count as a different measurement: {rule!r}")


def test_a_project_that_saved_the_old_cut_off_is_upgraded_when_it_opens():
    """Five of the six runs on disk saved 0.25 into their own config before the
    measurement default was fixed. Opening one replaces exactly that value,
    saves it, and says so; a deliberate other value is never touched."""
    tmp = tempfile.mkdtemp(prefix="t9_upgrade_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    try:
        def _open_with(conf, name):
            """Write a project.json holding conf, open it, return (loaded, on-disk config)."""
            project_dir = os.path.join(tmp, name)
            state = _make_project_state(project_dir, name=name)
            state["steps"]["7"]["config"] = {"conf_threshold": conf, "iou_threshold": EVAL_IOU}
            _write(os.path.join(project_dir, "project.json"), json.dumps(state))
            loaded = appmod._load_project(project_dir)
            with open(os.path.join(project_dir, "project.json")) as fh:
                saved = json.load(fh)["steps"]["7"]["config"]
            return loaded, saved

        loaded, saved = _open_with(RETIRED_CONF, "run_26-09-02_LO_train_coral_intel")
        cfg = loaded["steps"]["7"]["config"]
        check(cfg["conf_threshold"] == EVAL_CONF,
              f"the retired cut-off is replaced in memory: {cfg}")
        check(saved["conf_threshold"] == EVAL_CONF,
              f"and written back to project.json, so the next open is clean: {saved}")
        payload = appmod._state_payload(loaded)
        check(payload.get("eval_conf_upgraded_from") == RETIRED_CONF,
              f"the payload says what was replaced: {payload.get('eval_conf_upgraded_from')!r}")
        check("project.json" not in json.dumps(loaded.get("steps")),
              "the notice is never written into the project's own steps")

        for keep, label in ((0.4, "run_deliberate"), (EVAL_CONF, "run_already_fixed"),
                            ("high", "run_hand_edited")):
            loaded, saved = _open_with(keep, label)
            check(loaded["steps"]["7"]["config"]["conf_threshold"] == keep,
                  f"{label}: a value nobody retired is left alone: {loaded['steps']['7']['config']}")
            check(appmod._state_payload(loaded).get("eval_conf_upgraded_from") is None,
                  f"{label}: and draws no notice")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_the_launch_takes_its_split_from_the_config():
    """A panel that names no split measures on the split config/pipeline.yaml
    seeds, not on a literal repeated in the launch."""
    fx = _make_fixture()
    real = appmod.EVAL_SPLIT_DEFAULT
    try:
        appmod.EVAL_SPLIT_DEFAULT = "val"
        r, cmd = _evaluate(_client(fx), {"run_dir": fx["run_dir"], "imgsz": 640})
        check(r.status_code == 200, f"the launch answered {r.status_code}: {r.get_json()}")
        check(_argv_value(cmd, "--split") == "val",
              f"the launch follows the config's split default: {cmd}")
    finally:
        appmod.EVAL_SPLIT_DEFAULT = real
        _close(fx)


def test_the_rounds_route_hands_back_what_a_score_was_measured_with():
    """conf, iou and imgsz reach the Rounds table as numbers, and a row written
    before those columns existed reads back as null rather than as text."""
    fx = _make_fixture()
    try:
        step6 = fx["state"]["steps"]["6"]["dir"]
        header = ("round,run_name,run_dir,map50_95_B,metric_family,conf,iou,imgsz\n")
        _write(os.path.join(step6, "rounds.csv"),
               header
               + f"1,old,/r/old,0.535,box,,,\n"
               + f"2,new,/r/new,0.587,box,{EVAL_CONF},{EVAL_IOU},512\n")
        rows = (_client(fx).get("/api/step/7/rounds").get_json() or {}).get("rounds") or []
        check(len(rows) == 2, f"both rows reach the table: {rows}")
        check(rows[0]["conf"] is None and rows[0]["imgsz"] is None,
              f"a row from before the columns existed reads back empty: {rows[0]}")
        check(rows[1]["conf"] == EVAL_CONF and rows[1]["iou"] == EVAL_IOU,
              f"a measured row hands back numbers, not strings: {rows[1]}")
        check(rows[1]["imgsz"] == 512 and isinstance(rows[1]["imgsz"], int),
              f"the image size is a whole number: {rows[1]}")
    finally:
        _close(fx)


if __name__ == "__main__":
    print("T9 -- evaluation report keyed by run\n")
    print("GET /api/step/7/report:")
    run(test_report_resolves_by_run_dir_newest_wins)
    run(test_report_forwards_the_species_names_the_evaluation_wrote)
    run(test_report_falls_back_to_the_old_pointer_without_run_dir)
    run(test_report_404_when_no_evaluation_exists)
    run(test_report_refuses_a_run_outside_both_roots)
    run(test_figures_are_all_null_without_a_val_run_folder)
    print("GET /api/step/7/pdf, POST /api/step/7/copy_pdf:")
    run(test_pdf_route_serves_the_newest_pdf_or_404)
    run(test_copy_pdf_by_run_dir)
    print("GET /api/step/7/image:")
    run(test_image_serves_a_preview_and_a_run_figure)
    run(test_image_refuses_traversal_absolute_symlink_and_outside)
    run(test_step7_image_refuses_a_sibling_prefix_path)
    run(test_image_falls_back_to_the_pointer_without_run_dir)
    print("POST /api/step/7/run image size:")
    run(test_blank_imgsz_auto_fills_from_the_runs_args_yaml)
    run(test_blank_imgsz_without_a_recorded_size_falls_back_and_says_so)
    run(test_non_integer_imgsz_is_400)
    run(test_typed_imgsz_is_used_as_given)
    run(test_detector_note_names_the_box_family)
    print("What a score was measured with:")
    run(test_evaluate_defaults_keep_the_low_confidence_tail)
    run(test_a_config_that_cannot_be_read_is_refused_by_name)
    run(test_run_info_offers_the_recommended_measurement_settings)
    run(test_evaluation_forwards_the_confidence_the_panel_holds)
    run(test_a_project_still_holding_the_old_cut_off_is_told)
    run(test_the_evaluate_panel_says_why_the_confidence_is_so_low)
    run(test_the_recommended_defaults_button_offers_the_same_confidence)
    run(test_the_shipped_evaluation_preset_offers_the_same_confidence)
    run(test_the_panel_fills_its_evaluate_fields_from_one_place)
    run(test_the_rounds_table_never_ranks_two_measurements)
    run(test_the_recommended_answer_from_the_server_fills_the_form)
    run(test_a_project_that_saved_the_old_cut_off_is_upgraded_when_it_opens)
    run(test_the_launch_takes_its_split_from_the_config)
    run(test_the_rounds_route_hands_back_what_a_score_was_measured_with)

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
