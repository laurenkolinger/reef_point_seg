"""An annotated project's images and selection are protected.

Incident (2026-10-05): Step 3 was re-run on a project that already had
annotated frames. On the next Open, the automatic re-route saw
selected_frames.csv newer than the cached click prompts and cleared every
subfolder of routed_input/, including raw/, the folder every all_images link
and every segmentations entry's image_path_abs points into. 556 annotated
frames lost their images.

Covered here:
  COUNT   count_annotated_frames: reviewed or exported frames; a missing file
          is zero; a malformed, empty or wrong-shaped file is UNKNOWN (None),
          never zero.
  STEP 3  the run route answers 409 with a one-sentence message naming the
          count when any frame is annotated or the count is unknown, and runs
          when the count is zero.
  OPEN    an annotated or unknown project never re-routes and deletes nothing,
          even with a newer selection; an unannotated project still re-routes.
  CLEAR   _clear_routed_input never deletes raw/ (source check and behaviour).
  RESET   the 4.test reset refuses while frames are exported unless the
          request carries confirm_exported equal to the exported count.

No pytest. Run with the unified env python and no card visible:
    CUDA_VISIBLE_DEVICES="" env/bin/python \
        scripts/pipeline_orchestrator/tests/test_annotated_project_guard.py

Nothing reaches a real process, port or card: subprocess.run and Popen are
stubbed at module level, sockets to 5050 and 5080-5089 raise, and the runner
methods that would spawn are replaced with recorders. Fixtures live in a temp
directory and their removal is verified at the end.
"""

import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen

_REAL_SOCKET = socket.socket
_GUARDED_PORTS = {5050} | set(range(5080, 5090))


class _NoLivePortSocket(_REAL_SOCKET):
    def connect(self, address, *a, **k):
        if (isinstance(address, tuple) and len(address) >= 2
                and address[1] in _GUARDED_PORTS):
            raise AssertionError(f"test tried to reach guarded port {address}")
        return super().connect(address, *a, **k)


socket.socket = _NoLivePortSocket

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402


class _FakeVic:
    notes = []

    @classmethod
    def note(cls, msg, *_a, **_k):
        cls.notes.append(str(msg))

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None


appmod.vic = _FakeVic()
appmod._flush_annotator = lambda port: (True, None)

_RESULTS = []
_FIXTURES = []
_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
FILES = ("a.jpg", "b.jpg", "c.jpg", "d.jpg")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - the harness reports every failure
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------

def _seg(reviewed=False, exported=False):
    return {"image_path": "raw/x.jpg", "image_width": 10, "image_height": 10,
            "masks": [], "reviewed": reviewed, "exported": exported}


def _project(segs=None, raw_text=None, stale=False):
    """A project with a Step 3 selection, a routed_input holding raw/ images and
    prompts, and an optional segmentations.json. `segs` is {name: seg dict};
    `raw_text` writes that text verbatim instead. `stale` makes the selection
    newer than the cached prompts, the state that triggered the incident."""
    tmp = tempfile.mkdtemp(prefix="annot_guard_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    steps["3"]["config"] = {"target_species": "PA"}
    sel = os.path.join(steps["3"]["dir"], "selected_frames.csv")
    with open(sel, "w") as fh:
        fh.write("basename,year\nTCRMP20180225_clip_BIX_T111,2018\n")
    steps["3"]["outputs"]["selected_frames"] = sel
    export = os.path.join(tmp, pm.STEP_DIRS["step4test"])
    routed = os.path.join(export, "routed_input")
    os.makedirs(os.path.join(routed, "raw"))
    os.makedirs(os.path.join(routed, "ids"))
    for name in FILES:
        with open(os.path.join(routed, "raw", name), "w") as fh:
            fh.write("img")
    prompts = os.path.join(routed, "ids", "sam_click_prompts.json")
    with open(prompts, "w") as fh:
        fh.write("{}")
    seg_dir = os.path.join(export, "segmentations")
    os.makedirs(seg_dir)
    seg_path = os.path.join(seg_dir, "segmentations.json")
    if raw_text is not None:
        with open(seg_path, "w") as fh:
            fh.write(raw_text)
    elif segs is not None:
        with open(seg_path, "w") as fh:
            json.dump(segs, fh)
    now = time.time()
    os.utime(prompts, (now - 1000, now - 1000))
    os.utime(sel, (now - 2000 + (3000 if stale else 0),) * 2)
    state = {"id": os.path.basename(tmp), "name": "guard fixture",
             "project_dir": tmp, "created_at": "2026-10-15T10:00:00-04:00",
             "updated_at": "2026-10-15T10:00:00-04:00", "current_step": 3,
             "steps": steps}
    pm.save_project(state)
    return state, {"tmp": tmp, "export": export, "routed": routed,
                   "raw": os.path.join(routed, "raw"), "prompts": prompts,
                   "seg": seg_path}


class _Recorder:
    """Stands in for runner.run_cli_stage / run_flask_stage / kill."""

    def __init__(self, result):
        self.calls = []
        self.result = result

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.result


def _client(state, cli_result=None, flask_result=None):
    appmod.current_project = state
    appmod.step4test_cfg = {}
    appmod.route_thread = None
    cli = _Recorder(cli_result if cli_result is not None else {"pid": 4242})
    flask_rec = _Recorder(flask_result if flask_result is not None
                          else {"error": "stubbed launch"})
    appmod.runner.run_cli_stage = cli
    appmod.runner.run_flask_stage = flask_rec
    appmod.runner.kill = _Recorder(None)
    appmod.runner.health_check = lambda port: False
    return appmod.create_app().test_client(), cli, flask_rec


def _close():
    appmod.current_project = None
    appmod._step_launches.clear()


def _raw_intact(paths):
    return sorted(os.listdir(paths["raw"])) == sorted(FILES)


# ---------------------------------------------------------------------------
# COUNT
# ---------------------------------------------------------------------------

def test_count_reviewed_and_exported():
    _, p = _project({"a": _seg(reviewed=True), "b": _seg(exported=True),
                     "c": _seg(), "d": _seg(reviewed=True, exported=True)})
    got = appmod.count_annotated_frames(p["export"])
    check(got == {"annotated": 3, "exported": 2}, f"counts: {got}")


def test_count_missing_file_is_zero():
    _, p = _project(None)
    got = appmod.count_annotated_frames(p["export"])
    check(got == {"annotated": 0, "exported": 0}, f"missing file is zero: {got}")
    check(appmod.count_annotated_frames("") == {"annotated": 0, "exported": 0},
          "no export dir is zero")


def test_count_malformed_is_unknown_never_zero():
    for label, text in (("garbage", "{not json"), ("empty file", ""),
                        ("a list", "[]"), ("a string", '"x"'),
                        ("entry not a dict", '{"a": 5}'),
                        ("null", "null")):
        _, p = _project(raw_text=text)
        got = appmod.count_annotated_frames(p["export"])
        check(got is None, f"{label} must be unknown (None), got {got}")


def test_count_hostile_values_do_not_crash():
    _, p = _project({"a": {"reviewed": "yes"}, "b": {"exported": 1},
                     "c": {"reviewed": None}, "d": {}})
    got = appmod.count_annotated_frames(p["export"])
    check(got == {"annotated": 0, "exported": 0}
          or got is None, f"only literal True counts, or unknown: {got}")
    check(got != {"annotated": 2, "exported": 1}, "truthy strings are not review marks")


# ---------------------------------------------------------------------------
# STEP 3
# ---------------------------------------------------------------------------

def test_step3_run_blocked_when_annotated():
    segs = {f"f{i}.jpg": _seg(reviewed=True) for i in range(691)}
    state, p = _project(segs)
    try:
        client, cli, _ = _client(state)
        r = client.post("/api/step/3/run")
        body = r.get_json() or {}
        check(r.status_code == 409, f"annotated project blocks Step 3: {r.status_code} {body}")
        err = str(body.get("error") or "")
        check("691" in err, f"the message names the count: {err}")
        check("already annotated" in err, f"and says why: {err}")
        check(err.count(".") <= 4 and "\n" not in err, f"plain sentences: {err}")
        check(not cli.calls, "Step 3 was never started")
        check(state["steps"]["3"]["status"] == "pending", "the step was not stamped running")
        check(_raw_intact(p), "images untouched")
    finally:
        _close()


def test_step3_run_blocked_when_count_unknown():
    state, p = _project(raw_text="{broken")
    try:
        client, cli, _ = _client(state)
        r = client.post("/api/step/3/run")
        body = r.get_json() or {}
        check(r.status_code == 409, f"unknown count blocks Step 3: {r.status_code} {body}")
        check("could not be read" in str(body.get("error")), f"says it could not tell: {body}")
        check(not cli.calls, "Step 3 was never started")
    finally:
        _close()


def test_step3_run_allowed_when_zero():
    state, p = _project({"a": _seg(), "b": _seg()})
    try:
        client, cli, _ = _client(state)
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"zero annotated runs Step 3: {r.status_code} {r.get_json()}")
        check(len(cli.calls) == 1, "Step 3 was started once")
    finally:
        _close()


def test_step3_run_allowed_when_no_segmentations_file():
    state, p = _project(None)
    try:
        client, cli, _ = _client(state)
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"a fresh project runs Step 3: {r.status_code} {r.get_json()}")
    finally:
        _close()


def test_step3_passes_the_projects_remap_log():
    state, p = _project({"a": _seg()})
    log = os.path.join(state["steps"]["2"]["dir"], "remap_log.json")
    with open(log, "w") as fh:
        fh.write('{"remaps": []}')
    try:
        client, cli, _ = _client(state)
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"runs: {r.status_code}")
        cmd = cli.calls[0][0][1]
        check("--remap-log" in cmd and cmd[cmd.index("--remap-log") + 1] == log,
              f"the project's recode log is handed to Step 3: {cmd}")
    finally:
        _close()


def test_step3_without_a_remap_log_passes_no_flag():
    state, p = _project({"a": _seg()})
    try:
        client, cli, _ = _client(state)
        client.post("/api/step/3/run")
        check("--remap-log" not in cli.calls[0][0][1], "no log, no flag")
    finally:
        _close()


# ---------------------------------------------------------------------------
# OPEN
# ---------------------------------------------------------------------------

def _open(client):
    return client.post("/api/step/step4test/run", json={})


def test_open_annotated_with_newer_selection_does_not_reroute_or_delete():
    state, p = _project({"a": _seg(reviewed=True), "b": _seg()}, stale=True)
    try:
        check(appmod._routed_prompts_stale(p["routed"], state["steps"]["3"]["outputs"]["selected_frames"]),
              "fixture: the selection really is newer than the prompts")
        _FakeVic.notes.clear()
        client, _, flask_rec = _client(state)
        _open(client)
        keys = [c[0][0] for c in flask_rec.calls]
        check("step4test_route" not in keys, f"no routing pass started: {keys}")
        check(_raw_intact(p), "raw/ images all still there")
        check(os.path.isfile(p["prompts"]), "the existing routing is kept")
        check(any("older than" in n or "newer" in n or "selection" in n.lower()
                  for n in _FakeVic.notes), f"a warning line is logged: {_FakeVic.notes}")
        check(any("annotated" in n for n in _FakeVic.notes), "and it names the reason")
    finally:
        _close()


def test_open_unknown_count_does_not_reroute_or_delete():
    state, p = _project(raw_text="oops", stale=True)
    try:
        client, _, flask_rec = _client(state)
        _open(client)
        keys = [c[0][0] for c in flask_rec.calls]
        check("step4test_route" not in keys, f"no routing pass started: {keys}")
        check(_raw_intact(p) and os.path.isfile(p["prompts"]), "nothing deleted")
    finally:
        _close()


def test_open_unannotated_stale_reroutes_and_clears_unreferenced_raw():
    """D5: a re-route writes its images fresh. An old image that no entry names
    would otherwise be skipped as a name collision and paired with rescaled
    points (Low res only toggled), so unreferenced raw files are removed; one
    an entry names (by key or by image_path) stays."""
    segs = {"a.jpg": _seg(), "k2": dict(_seg(), image_path="raw/b.jpg")}
    state, p = _project(segs, stale=True)
    try:
        client, _, flask_rec = _client(state)
        _open(client)
        keys = [c[0][0] for c in flask_rec.calls]
        check("step4test_route" in keys, f"an unannotated project still re-routes: {keys}")
        check(not os.path.isfile(p["prompts"]), "the stale prompts were cleared")
        check(sorted(os.listdir(p["raw"])) == ["a.jpg", "b.jpg"],
              f"only referenced images stay: {sorted(os.listdir(p['raw']))}")
    finally:
        _close()


def test_open_unannotated_without_entries_clears_all_raw():
    state, p = _project(None, stale=True)
    try:
        client, _, flask_rec = _client(state)
        _open(client)
        check(os.listdir(p["raw"]) == [], f"zero entries: raw emptied: {os.listdir(p['raw'])}")
        check(os.path.isdir(p["raw"]), "but the folder stays")
    finally:
        _close()


def test_open_lores_toggle_on_unannotated_clears_old_images():
    state, p = _project(None)
    try:
        client, _, flask_rec = _client(state)
        appmod._write_routed_lores_marker(p["routed"], False)
        appmod.step4test_cfg = {}
        r = client.post("/api/step/step4test/run", json={"config": {"lores_mode": True}})
        keys = [c[0][0] for c in flask_rec.calls]
        check("step4test_route" in keys, f"toggle re-routes an unannotated project: {keys}")
        check(os.listdir(p["raw"]) == [], "old full-res images are gone before routing")
    finally:
        _close()


def test_open_annotated_without_prompts_refuses_to_route():
    for label, kwargs in (("annotated", {"segs": {"a.jpg": _seg(reviewed=True)}}),
                          ("unknown", {"raw_text": "{oops"})):
        state, p = _project(**{("segs" if "segs" in kwargs else "raw_text"): list(kwargs.values())[0]})
        try:
            os.remove(p["prompts"])
            client, _, flask_rec = _client(state)
            r = _open(client)
            body = r.get_json() or {}
            keys = [c[0][0] for c in flask_rec.calls]
            check("step4test_route" not in keys, f"{label}: no routing pass: {keys}")
            check(body.get("error") or body.get("phase") == "error",
                  f"{label}: the refusal is reported: {body}")
            check(_raw_intact(p), f"{label}: raw untouched")
        finally:
            _close()


def test_open_force_reroute_refused_when_annotated():
    state, p = _project({"a": _seg(exported=True)}, stale=True)
    try:
        appmod.current_project = state
        appmod.route_thread = None
        # _run_step4test is nested in create_app; reach it through the route's
        # module-level counterpart: a force flag has no HTTP caller, so the
        # guard is proved by the shared decision helper.
        msg = appmod.reroute_block_reason(p["export"])
        check(msg and "1" in msg, f"the decision helper blocks and names the count: {msg}")
        check(appmod.reroute_block_reason(_project({"a": _seg()})[1]["export"]) is None,
              "and allows a re-route on an unannotated project")
    finally:
        _close()


# ---------------------------------------------------------------------------
# CLEAR
# ---------------------------------------------------------------------------

def test_clear_routed_input_keeps_raw_behaviour():
    state, p = _project({"a": _seg()})
    for y in ("2018", "2022"):
        os.makedirs(os.path.join(p["routed"], y, "ids"))
        os.makedirs(os.path.join(p["routed"], y, "raw"))
    n = appmod._clear_routed_input(p["routed"])
    check(_raw_intact(p), "flat raw/ and every image in it survive")
    check(not os.path.exists(os.path.join(p["routed"], "2018")), "year dirs are cleared")
    check(not os.path.exists(os.path.join(p["routed"], "ids")), "stale flat ids/ is cleared")
    check(n == 3, f"counts the 2 year dirs and ids/: {n}")


def test_clear_routed_input_keeps_raw_static():
    src = open(os.path.join(_ORCH, "app.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_clear_routed_input")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    consts = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)}
    check("ROUTED_RAW_DIR_NAME" in names or "raw" in consts,
          "the function names the raw folder it must keep")
    check(appmod.ROUTED_RAW_DIR_NAME == "raw", "the kept folder is raw")


# ---------------------------------------------------------------------------
# RESET
# ---------------------------------------------------------------------------

def test_reset_refused_without_confirm_when_exported():
    state, p = _project({"a": _seg(exported=True), "b": _seg(exported=True),
                         "c": _seg(reviewed=True)})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={})
        body = r.get_json() or {}
        check(r.status_code == 409, f"export present refuses: {r.status_code} {body}")
        check(body.get("annotated_count") == 3, f"server reports reviewed or exported: {body}")
        check("3" in str(body.get("error")) and "reviewed or exported" in str(body.get("error")),
              f"and says so: {body}")
        check(os.path.isdir(p["export"]) and _raw_intact(p), "nothing deleted")
        for wrong in (1, 2, "x", None, True):
            r = client.post("/api/step/step4test/reset", json={"confirm_exported": wrong})
            check(r.status_code == 409, f"wrong confirm {wrong!r} still refuses: {r.status_code}")
        check(os.path.isdir(p["export"]), "still nothing deleted")
    finally:
        _close()


def test_reset_proceeds_with_exact_confirm():
    state, p = _project({"a": _seg(exported=True), "b": _seg(exported=True)})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={"confirm_exported": 2})
        check(r.status_code == 200, f"exact count proceeds: {r.status_code} {r.get_json()}")
        check(not os.path.exists(p["export"]), "the export dir was removed")
    finally:
        _close()


def test_reset_proceeds_with_string_count():
    state, p = _project({"a": _seg(exported=True)})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={"confirm_exported": "1"})
        check(r.status_code == 200, f"the count as text proceeds: {r.status_code}")
    finally:
        _close()


def test_reset_with_nothing_annotated_needs_no_confirm():
    state, p = _project({"a": _seg(), "b": _seg()})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={})
        check(r.status_code == 200, f"unannotated reset proceeds: {r.status_code} {r.get_json()}")
        check(not os.path.exists(p["export"]), "removed")
    finally:
        _close()


def test_reset_reviewed_but_unexported_needs_the_count():
    state, p = _project({"a": _seg(reviewed=True), "b": _seg()})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={})
        body = r.get_json() or {}
        check(r.status_code == 409 and body.get("annotated_count") == 1,
              f"a reviewed outline needs the confirm too: {r.status_code} {body}")
        check(os.path.isdir(p["export"]), "nothing deleted")
        r = client.post("/api/step/step4test/reset", json={"confirm_exported": 1})
        check(r.status_code == 200, f"the count proceeds: {r.status_code}")
    finally:
        _close()


def test_reset_unknown_count_needs_the_word_unknown():
    state, p = _project(raw_text="{broken")
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/step4test/reset", json={})
        body = r.get_json() or {}
        check(r.status_code == 409, f"unreadable record refuses: {r.status_code} {body}")
        check(body.get("annotated_count") is None and body.get("confirm_token") == "unknown",
              f"the server asks for the token: {body}")
        check(os.path.isdir(p["export"]), "nothing deleted")
        r = client.post("/api/step/step4test/reset", json={"confirm_exported": "unknown"})
        check(r.status_code == 200, f"the token proceeds: {r.status_code}")
    finally:
        _close()


def _step_dir_files(state, n):
    d = state["steps"][str(n)]["dir"]
    f = os.path.join(d, "keep.txt")
    with open(f, "w") as fh:
        fh.write("x")
    return f


def test_generic_reset_1_2_3_refused_when_annotated():
    for n in (1, 2, 3):
        state, p = _project({f"f{i}.jpg": _seg(reviewed=True) for i in range(691)})
        try:
            keep = _step_dir_files(state, n)
            client, _, _ = _client(state)
            r = client.post(f"/api/step/{n}/reset")
            body = r.get_json() or {}
            check(r.status_code == 409, f"step {n} reset refused: {r.status_code} {body}")
            err = str(body.get("error") or "")
            check("691" in err and "already annotated" in err, f"names the count: {err}")
            check(f"Step {n} cannot be reset" in err or "cannot be reset" in err, f"says so: {err}")
            check(os.path.isfile(keep) and _raw_intact(p), "nothing deleted")
        finally:
            _close()


def test_generic_reset_1_2_3_refused_when_unknown():
    for n in (1, 2, 3):
        state, p = _project(raw_text="{broken")
        try:
            keep = _step_dir_files(state, n)
            client, _, _ = _client(state)
            r = client.post(f"/api/step/{n}/reset")
            check(r.status_code == 409, f"step {n}: {r.status_code}")
            check("could not be read" in str((r.get_json() or {}).get("error")), "says why")
            check(os.path.isfile(keep), "nothing deleted")
        finally:
            _close()


def test_generic_reset_1_2_3_allowed_when_zero():
    for n in (1, 2, 3):
        state, p = _project({"a": _seg()})
        try:
            keep = _step_dir_files(state, n)
            client, _, _ = _client(state)
            r = client.post(f"/api/step/{n}/reset")
            check(r.status_code == 200, f"step {n}: {r.status_code} {r.get_json()}")
            check(not os.path.exists(keep), "the step was cleared")
        finally:
            _close()


def test_generic_reset_4_to_8_unchanged_when_annotated():
    for n in (4, 5, 7, 8):
        state, p = _project({"a": _seg(reviewed=True)})
        try:
            keep = _step_dir_files(state, n)
            client, _, _ = _client(state)
            r = client.post(f"/api/step/{n}/reset")
            check(r.status_code == 200, f"step {n} still resets: {r.status_code} {r.get_json()}")
            check(not os.path.exists(keep), "cleared as before")
            check(_raw_intact(p), "and the 4.test export is untouched")
        finally:
            _close()


def _write_step5(state, segs, year="2024", raw_text=None):
    """Step 5 keeps one segmentations.json per year under step5_segmentImages."""
    d = os.path.join(state["steps"]["5"]["dir"], "segmentations", year)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "segmentations.json"), "w") as fh:
        if raw_text is not None:
            fh.write(raw_text)
        else:
            json.dump(segs, fh)


def test_count_includes_step5_frames_once():
    state, p = _project({"a.jpg": _seg(reviewed=True)})
    _write_step5(state, {"s1.jpg": _seg(exported=True), "s2.jpg": _seg(exported=True),
                         "a.jpg": _seg(exported=True)}, year="2024")
    _write_step5(state, {"s3.jpg": _seg(reviewed=True), "s4.jpg": _seg()}, year="2023")
    got = appmod.count_annotated_frames(p["export"])
    check(got == {"annotated": 4, "exported": 3},
          f"step 4.test and step 5 frames, a shared name counted once: {got}")


def test_count_unknown_when_a_step5_file_is_broken():
    state, p = _project({"a.jpg": _seg()})
    _write_step5(state, None, raw_text="{broken")
    check(appmod.count_annotated_frames(p["export"]) is None, "unreadable step 5 record is unknown")


def test_step5_work_protects_step3_resets_and_open():
    state, p = _project(None, stale=True)
    _write_step5(state, {f"s{i}.jpg": _seg(exported=True) for i in range(400)})
    try:
        client, cli, flask_rec = _client(state)
        r = client.post("/api/step/3/run")
        check(r.status_code == 409 and "400" in str((r.get_json() or {}).get("error")),
              f"step 5 work blocks Step 3: {r.status_code}")
        r = client.post("/api/step/2/reset")
        check(r.status_code == 409, f"and the numbered resets: {r.status_code}")
        _open(client)
        check("step4test_route" not in [c[0][0] for c in flask_rec.calls], "and a re-route")
        check(_raw_intact(p) and os.path.isfile(p["prompts"]), "nothing deleted")
    finally:
        _close()


def test_steps_1_and_2_run_link_and_remap_apply_refused_when_annotated():
    state, p = _project({"a.jpg": _seg(reviewed=True), "b.jpg": _seg(exported=True)})
    try:
        client, cli, _ = _client(state)
        for label, call in (
                ("step 1 run", lambda: client.post("/api/step/1/run")),
                ("step 2 run", lambda: client.post("/api/step/2/run")),
                ("link", lambda: client.post("/api/step/1/link", json={"all_points": "/nonexistent.csv"})),
                ("remap apply", lambda: client.post("/api/remap/apply", json={"remap_log_path": "x.json"}))):
            r = call()
            body = r.get_json() or {}
            err = str(body.get("error") or "")
            check(r.status_code == 409, f"{label} refused: {r.status_code} {body}")
            check("2" in err and "already annotated" in err, f"{label} names the count: {err}")
        check(not cli.calls, "nothing was started")
    finally:
        _close()


def test_steps_1_and_2_run_link_and_remap_apply_refused_when_unknown():
    state, p = _project(raw_text="{broken")
    try:
        client, cli, _ = _client(state)
        for call in (lambda: client.post("/api/step/1/run"),
                     lambda: client.post("/api/step/1/link", json={"all_points": "/nonexistent.csv"}),
                     lambda: client.post("/api/remap/apply", json={"remap_log_path": "x.json"})):
            check(call().status_code == 409, "unknown count refuses")
    finally:
        _close()


def test_link_and_remap_apply_reach_their_own_checks_when_zero():
    state, p = _project({"a.jpg": _seg()})
    try:
        client, _, _ = _client(state)
        r = client.post("/api/step/1/link", json={"all_points": "/nonexistent.csv"})
        check(r.status_code == 400, f"link gets past the guard to its own error: {r.status_code}")
        r = client.post("/api/remap/apply", json={"remap_log_path": "x.json"})
        check(r.status_code != 409, f"remap apply gets past the guard: {r.status_code}")
    finally:
        _close()


def test_reset_js_sends_the_count():
    js = open(os.path.join(_ORCH, "static", "orchestrator.js"), encoding="utf-8").read()
    check("confirm_exported" in js, "the page sends confirm_exported")
    check("annotated_count" in js and "confirm_token" in js,
          "and reads the count the server reports")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            run(fn)
    for t in _FIXTURES:
        shutil.rmtree(t, ignore_errors=True)
    still = [t for t in _FIXTURES if os.path.exists(t)]
    check_clean = not still
    print("fixtures removed" if check_clean else f"fixtures NOT removed: {still}")
    ok = sum(1 for _, passed, _ in _RESULTS if passed)
    for name, passed, detail in _RESULTS:
        if not passed:
            print(f"\n--- {name}\n{detail}")
    print(f"\n{ok}/{len(_RESULTS)} passed")
    sys.exit(0 if ok == len(_RESULTS) and check_clean else 1)
