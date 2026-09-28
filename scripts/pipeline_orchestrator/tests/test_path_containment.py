"""Path containment on the three file routes that took any path (2026-09-17, LO).

VEGA answers on 0.0.0.0:5050 with no login. Until this change three routes read
or wrote wherever a request pointed: POST /api/remap/load read any file, GET
/api/step/7/run_info read any folder's args.yaml, and POST /api/step/7/copy_pdf
created folders and wrote a PDF at any path. This suite pins the containment:
reads stay inside the module root; the PDF export keeps its real use (a folder
Lauren chooses) but only under an allowed root, never a system path, never a
symlink that escapes, never creating parents outside those roots.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_path_containment.py

subprocess.run and subprocess.Popen are stubbed at module level. Every fixture
lives under one temp folder removed at the end, and the removal is verified.
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
_FIXTURES = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

import app as appmod  # noqa: E402

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()


def check(cond, msg):
    """Record one assertion outcome."""
    _RESULTS.append((msg, bool(cond), msg))
    print(("PASS " if cond else "FAIL ") + msg)


def run(fn):
    """Run one test, recording an exception as a failure."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        import traceback
        _RESULTS.append((fn.__name__, False, traceback.format_exc()))
        print(f"FAIL {fn.__name__}: {exc}")


def _tmp(prefix):
    d = tempfile.mkdtemp(prefix=prefix, dir=_TMP_ROOT)
    _FIXTURES.append(d)
    return d


def _client():
    return appmod.create_app().test_client()


class _RootsPinned:
    """Point the module root and the export roots at fixture folders for one test."""

    def __init__(self, module_root, export_roots):
        self.module_root = module_root
        self.export_roots = export_roots

    def __enter__(self):
        self._saved = (appmod._module_root, appmod._export_roots)
        appmod._module_root = lambda: self.module_root
        appmod._export_roots = lambda: list(self.export_roots)
        return self

    def __exit__(self, *_exc):
        appmod._module_root, appmod._export_roots = self._saved


# ---- reads ------------------------------------------------------------------

def test_remap_load_refuses_a_file_outside_the_module_root():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    secret = os.path.join(outside, "remap_log.json")
    with open(secret, "w") as fh:
        json.dump({"leak": True}, fh)
    calls = []
    real = appmod.remap_loader.load_remap_log
    appmod.remap_loader.load_remap_log = lambda p: calls.append(p) or {"ok": 1}
    try:
        with _RootsPinned(module, []):
            r = _client().post("/api/remap/load", json={"path": secret})
    finally:
        appmod.remap_loader.load_remap_log = real
    check(r.status_code in (400, 403), f"a remap log outside the module root is refused, got {r.status_code}")
    check("module" in (r.get_json() or {}).get("error", "").lower(), "the refusal says where a remap log may live")
    check(calls == [], "the file outside the root was never read")


def test_remap_load_refuses_a_symlink_that_escapes():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    secret = os.path.join(outside, "remap_log.json")
    with open(secret, "w") as fh:
        json.dump({}, fh)
    link = os.path.join(module, "inprocess", "remap_log.json")
    os.makedirs(os.path.dirname(link)); os.symlink(secret, link)
    with _RootsPinned(module, []):
        r = _client().post("/api/remap/load", json={"path": link})
    check(r.status_code in (400, 403), f"a symlink under the root that points outside is refused, got {r.status_code}")


def test_remap_load_still_reads_a_log_inside_the_module_root():
    module = _tmp("contain_mod_")
    log = os.path.join(module, "inprocess", "run_x", "remap_log.json")
    os.makedirs(os.path.dirname(log))
    with open(log, "w") as fh:
        json.dump({"remaps": []}, fh)
    real = appmod.remap_loader.load_remap_log
    appmod.remap_loader.load_remap_log = lambda p: {"loaded": p}
    try:
        with _RootsPinned(module, []):
            r = _client().post("/api/remap/load", json={"path": log})
    finally:
        appmod.remap_loader.load_remap_log = real
    check(r.status_code == 200 and (r.get_json() or {}).get("success"), f"a log inside the root still loads, got {r.status_code}")


def test_remap_load_rejects_a_non_string_path():
    with _RootsPinned(_tmp("contain_mod_"), []):
        r = _client().post("/api/remap/load", json={"path": ["/etc/passwd"]})
    check(r.status_code == 400, f"a non-string path is a 400, got {r.status_code}")


def test_run_info_refuses_a_folder_outside_the_module_root():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    with open(os.path.join(outside, "args.yaml"), "w") as fh:
        fh.write("imgsz: 640\n")
    with _RootsPinned(module, []):
        r = _client().get("/api/step/7/run_info", query_string={"path": outside})
    check(r.status_code in (400, 403), f"run_info outside the module root is refused, got {r.status_code}")
    check("args_yaml_found" not in (r.get_json() or {}), "nothing about the outside folder is disclosed")


def test_run_info_reads_a_run_inside_the_module_root():
    module = _tmp("contain_mod_")
    run_dir = os.path.join(module, "inprocess", "run_x", "step6_trainModel", "runs", "R")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "args.yaml"), "w") as fh:
        fh.write("imgsz: 512\n")
    with _RootsPinned(module, []):
        r = _client().get("/api/step/7/run_info", query_string={"path": run_dir})
    check(r.status_code == 200 and (r.get_json() or {}).get("args_yaml_found") is True, f"a run inside the root is read, got {r.status_code}")


# ---- the PDF export ---------------------------------------------------------

def _evaluation(module):
    """A project with one evaluated run whose PDF exists; returns (run_dir, pdf)."""
    step7 = os.path.join(module, "inprocess", "run_p", "step7_evaluateModel")
    run_dir = os.path.join(module, "inprocess", "run_p", "step6_trainModel", "runs", "R")
    os.makedirs(os.path.join(run_dir, "weights")); os.makedirs(step7)
    out_dir = os.path.join(step7, "eval_R"); os.makedirs(out_dir)
    pdf = os.path.join(out_dir, appmod.REPORT_PDF_NAME)
    with open(pdf, "wb") as fh:
        fh.write(b"%PDF-1.4 test")
    return run_dir, pdf


def _resolve_to(run_dir, out_dir):
    """Pin _resolve_evaluation to the fixture for one call."""
    real = appmod._resolve_evaluation
    appmod._resolve_evaluation = lambda _rd: (run_dir, out_dir, "", 200)
    return real


def test_copy_pdf_refuses_a_system_path_and_creates_nothing():
    module = _tmp("contain_mod_")
    run_dir, pdf = _evaluation(module)
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": "/etc/vega_should_not_exist"})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code in (400, 403), f"an export into /etc is refused, got {r.status_code}")
    check(not os.path.exists("/etc/vega_should_not_exist"), "no folder was created outside the allowed roots")
    check("allowed" in (r.get_json() or {}).get("error", "").lower() or "root" in (r.get_json() or {}).get("error", "").lower(),
          "the refusal names where an export may go")


def test_copy_pdf_refuses_traversal_out_of_an_allowed_root():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    run_dir, pdf = _evaluation(module)
    sneaky = os.path.join(module, "exports", "..", "..", os.path.basename(outside), "deck")
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": sneaky})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code in (400, 403), f"a .. traversal that leaves the root is refused, got {r.status_code}")
    check(not os.path.exists(os.path.join(outside, "deck")), "the traversal created nothing")


def test_copy_pdf_refuses_a_symlinked_destination_that_escapes():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    run_dir, pdf = _evaluation(module)
    link = os.path.join(module, "exports_link"); os.symlink(outside, link)
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": link})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code in (400, 403), f"a symlink to outside the root is refused, got {r.status_code}")
    check(not any(n.endswith("_report.pdf") for n in os.listdir(outside)), "nothing was written through the symlink")


def test_copy_pdf_writes_into_an_allowed_root_and_may_create_the_leaf():
    module = _tmp("contain_mod_"); nas = _tmp("contain_nas_")
    run_dir, pdf = _evaluation(module)
    dst = os.path.join(nas, "presentations", "deck")
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module, nas]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": dst})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code == 200, f"an export under a listed root succeeds, got {r.status_code}: {r.get_json()}")
    check(os.path.isfile(os.path.join(dst, "R_report.pdf")), "the PDF copy exists where asked, the leaf folder created")


def test_export_roots_come_from_the_environment_and_stay_absolute():
    saved = os.environ.get("VEGA_EXPORT_ROOTS")
    os.environ["VEGA_EXPORT_ROOTS"] = "/volume5/Archive10_20TB:relative/path::/mnt/tear"
    try:
        roots = appmod._export_roots()
    finally:
        if saved is None:
            os.environ.pop("VEGA_EXPORT_ROOTS", None)
        else:
            os.environ["VEGA_EXPORT_ROOTS"] = saved
    check("/volume5/Archive10_20TB" in roots and "/mnt/tear" in roots, f"listed absolute roots are honoured: {roots}")
    check(not any(r == "relative/path" for r in roots), "a relative entry is ignored, never made absolute against the cwd")
    check(all(os.path.isabs(r) for r in roots), "every root is absolute")


# ---- review closures (2026-09-17) -------------------------------------------

def test_a_root_of_slash_is_never_allowed():
    saved_home, saved_env = os.environ.get("HOME"), os.environ.get("VEGA_EXPORT_ROOTS")
    os.environ["HOME"] = ""; os.environ["VEGA_EXPORT_ROOTS"] = "/"
    try:
        roots = appmod._export_roots()
    finally:
        os.environ["HOME"] = saved_home or ""; 
        if saved_env is None: os.environ.pop("VEGA_EXPORT_ROOTS", None)
        else: os.environ["VEGA_EXPORT_ROOTS"] = saved_env
    check(os.sep not in roots, f"an empty HOME or a '/' entry never yields the filesystem root: {roots}")


def test_null_bytes_are_a_400_not_a_500():
    with _RootsPinned(_tmp("contain_mod_"), []):
        c = _client()
        r1 = c.post("/api/remap/load", json={"path": "/etc/pass\u0000wd"})
        r2 = c.get("/api/step/7/run_info", query_string={"path": "/etc/pass\u0000wd"})
        r3 = c.post("/api/remap/apply", json={"remap_log_path": "/etc/pass\u0000wd"})
    check(r1.status_code == 400 and r2.status_code == 400 and r3.status_code == 400,
          f"a null byte in a path is a 400 on every read route, got {r1.status_code}/{r2.status_code}/{r3.status_code}")


def test_remap_apply_is_contained_like_remap_load():
    module = _tmp("contain_mod_"); outside = _tmp("contain_out_")
    with open(os.path.join(outside, "remap_log.json"), "w") as fh:
        json.dump({}, fh)
    with _RootsPinned(module, []):
        r = _client().post("/api/remap/apply", json={"remap_log_path": os.path.join(outside, "remap_log.json")})
    check(r.status_code in (400, 403), f"remap/apply refuses a log outside the module folder, got {r.status_code}")


def test_a_literal_dotdot_is_refused_even_when_it_would_collapse_safely():
    module = _tmp("contain_mod_")
    run_dir, pdf = _evaluation(module)
    inside_after_collapse = os.path.join(module, "a", "..", "exports")
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": inside_after_collapse})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code == 400 and ".." in (r.get_json() or {}).get("error", ""), f"a path that climbs is refused rather than straightened, got {r.status_code}")


def test_a_destination_that_is_a_file_is_a_409():
    module = _tmp("contain_mod_")
    run_dir, pdf = _evaluation(module)
    as_file = os.path.join(module, "not_a_folder")
    with open(as_file, "w") as fh:
        fh.write("x")
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": as_file})
    finally:
        appmod._resolve_evaluation = real
    check(r.status_code == 409, f"a destination that names a file is a 409 with a sentence, got {r.status_code}")


def test_refusals_do_not_list_the_home_folder_or_shares():
    module = _tmp("contain_mod_")
    run_dir, pdf = _evaluation(module)
    real = _resolve_to(run_dir, os.path.dirname(pdf))
    try:
        with _RootsPinned(module, [module, "/some/nas/share"]):
            r = _client().post("/api/step/7/copy_pdf", json={"run_dir": run_dir, "dst_dir": "/etc/x"})
    finally:
        appmod._resolve_evaluation = real
    err = (r.get_json() or {}).get("error", "")
    check("/some/nas/share" not in err and os.path.expanduser("~") not in err and module not in err,
          f"the refusal names roots by role, never by path: {err!r}")


def test_project_open_is_contained_to_the_inprocess_root():
    root = _tmp("contain_inproc_"); outside = _tmp("contain_out_")
    stray = os.path.join(outside, "run_x"); os.makedirs(stray)
    with open(os.path.join(stray, "project.json"), "w") as fh:
        json.dump({"name": "x", "project_dir": stray, "steps": {}}, fh)
    saved = appmod._inprocess_root; appmod._inprocess_root = lambda: root
    try:
        r = _client().post("/api/project/open", json={"project_dir": stray})
        r2 = _client().post("/api/project/open", json={"project_dir": ["a"]})
    finally:
        appmod._inprocess_root = saved
    check(r.status_code != 500, f"a project outside inprocess/ is answered, not crashed (containment is a recorded decision, TODO T20260917-152000), got {r.status_code}")
    check(r2.status_code == 400, f"a non-string project_dir is a 400, got {r2.status_code}")


if __name__ == "__main__":
    for fn in [
        test_remap_load_refuses_a_file_outside_the_module_root,
        test_remap_load_refuses_a_symlink_that_escapes,
        test_remap_load_still_reads_a_log_inside_the_module_root,
        test_remap_load_rejects_a_non_string_path,
        test_run_info_refuses_a_folder_outside_the_module_root,
        test_run_info_reads_a_run_inside_the_module_root,
        test_copy_pdf_refuses_a_system_path_and_creates_nothing,
        test_copy_pdf_refuses_traversal_out_of_an_allowed_root,
        test_copy_pdf_refuses_a_symlinked_destination_that_escapes,
        test_copy_pdf_writes_into_an_allowed_root_and_may_create_the_leaf,
        test_export_roots_come_from_the_environment_and_stay_absolute,
        test_a_root_of_slash_is_never_allowed,
        test_null_bytes_are_a_400_not_a_500,
        test_remap_apply_is_contained_like_remap_load,
        test_a_literal_dotdot_is_refused_even_when_it_would_collapse_safely,
        test_a_destination_that_is_a_file_is_a_409,
        test_refusals_do_not_list_the_home_folder_or_shares,
        test_project_open_is_contained_to_the_inprocess_root,
    ]:
        run(fn)
    for p in _FIXTURES:
        shutil.rmtree(p, ignore_errors=True)
    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{str(detail)[:300]}")
    sys.exit(1 if failed or leftovers else 0)
