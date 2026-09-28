"""Reviewer identity wiring + composite listing (Owner O, spec 2026-08-26).

Covers:
  reviewer_for_state  - latest access_log entry wins; fallback to the creator
                        initials; empty log -> ''; always uppercased.
  _post_reviewer      - one urllib POST to /api/reviewer with a 2 s timeout;
                        never raises (a failing socket answers False).
  _push_reviewer_after_health - posts once after health_fn says yes; gives up
                        silently at the deadline.
  launch env          - the editmasks launch (same _launch_step4test_ui path as
                        step4test / step4loop) carries TCRMP_REVIEWER and
                        schedules the post-health push; an "already running"
                        answer pushes too; the legacy step 5 launch env carries
                        TCRMP_REVIEWER as well.
  /api/composites     - lists composite exports under _composite_root().

No-pytest harness: run with env/bin/python <this file>. subprocess.run is
stubbed module-wide (standing rule: never reach a live annotator); the mask
registry is redirected to a throwaway dir; every temp dir is removed at exit.
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")
_TMP_ROOT = tempfile.mkdtemp(prefix="reviewer_env_")
os.environ["TCRMP_MASK_REGISTRY_DIR"] = os.path.join(_TMP_ROOT, "_mask_registry")
os.makedirs(os.environ["TCRMP_MASK_REGISTRY_DIR"], exist_ok=True)


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402

# Test isolation (2026-09-28 review): _kill_all_annotators' flush-before-kill
# (POST /api/flush, pipeline_orchestrator/app.py) reaches the network even
# when subprocess.run/Popen are stubbed, since it goes through urllib, not
# subprocess. Every editmasks/run call in this file drives it, and only ONE
# test below (test_post_reviewer_is_one_urllib_post_with_two_second_timeout)
# stubs appmod.urlopen, and only for its own duration. Stub
# appmod._flush_annotator here at module level (never a real save, never a
# real socket) and guard socket.socket so any code path that still tries to
# reach the orchestrator (5050) or the annotator port range (5080-5089)
# fails the test loudly instead of making a live network call.
appmod._flush_annotator = lambda port: (True, None)
import socket as _socket
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


def _make_project(tmp, access_log=None, initials=None):
    project_dir = os.path.join(tmp, "run_p1")
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    export_dir = os.path.join(project_dir, pm.STEP_DIRS["step4test"])
    os.makedirs(os.path.join(export_dir, "segmentations"), exist_ok=True)
    with open(os.path.join(export_dir, "segmentations", "segmentations.json"), "w") as f:
        json.dump({"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}}, f)
    state = {"id": "p1", "name": "p1", "project_dir": project_dir,
             "created_at": "x", "updated_at": "x", "current_step": 4, "steps": steps,
             "initials": initials, "access_log": access_log or []}
    return state, export_dir


def _reset_states():
    for st in (appmod.step4test_state, appmod.step4loop_state, appmod.editmasks_state):
        st.update(running=False, port=None, started_at=None, out_dir=None,
                  project_id=None, input_dir=None)


# ---- reviewer_for_state ---------------------------------------------------

def test_reviewer_latest_access_entry_wins():
    st = {"initials": "zz", "access_log": [
        {"ts": "2026-08-01T09:00:00-04:00", "initials": "lo", "action": "create"},
        {"ts": "2026-08-20T10:00:00-04:00", "initials": "ab", "action": "open"},
    ]}
    check(pm.reviewer_for_state(st) == "AB", f"latest entry must win: {pm.reviewer_for_state(st)!r}")
    st["access_log"].append({"ts": "2026-08-26T12:00:00-04:00", "action": "open"})  # no initials
    check(pm.reviewer_for_state(st) == "AB", "an entry without initials is skipped")
    st["access_log"].append({"ts": "2026-08-26T13:00:00-04:00", "initials": " cd ", "action": "open"})
    check(pm.reviewer_for_state(st) == "CD", "stripped + uppercased")


def test_reviewer_falls_back_to_creator():
    check(pm.reviewer_for_state({"initials": "lo"}) == "LO", "project.json initials fallback")
    st = {"access_log": [{"initials": "", "action": "open"}], "initials": None}
    check(pm.reviewer_for_state(st) == "", "blank everywhere -> ''")
    check(pm.reviewer_for_state({}) == "", "empty state -> ''")
    check(pm.reviewer_for_state({"access_log": None}) == "", "None log -> ''")


# ---- _post_reviewer / _push_reviewer_after_health ---------------------------

class _FakeResp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_post_reviewer_is_one_urllib_post_with_two_second_timeout():
    calls = []
    orig = appmod.urlopen

    def _fake_urlopen(req, timeout=None):
        calls.append({"url": req.full_url, "method": req.get_method(),
                      "body": json.loads(req.data.decode("utf-8")), "timeout": timeout})
        return _FakeResp()

    appmod.urlopen = _fake_urlopen
    try:
        ok = appmod._post_reviewer(5085, "AB")
        check(ok is True, "2xx -> True")
        check(len(calls) == 1 and calls[0]["url"] == "http://127.0.0.1:5085/api/reviewer",
              f"wrong target: {calls}")
        check(calls[0]["method"] == "POST" and calls[0]["body"] == {"reviewer": "AB"},
              f"wrong POST body: {calls}")
        check(calls[0]["timeout"] == 2.0, f"timeout must be 2 s: {calls[0]['timeout']}")
        check(appmod._post_reviewer(None, "AB") is False and appmod._post_reviewer(5085, "") is False,
              "no port / no reviewer -> False without a request")
        check(len(calls) == 1, "no request when port or reviewer is blank")

        def _boom(req, timeout=None):
            raise OSError("connection refused")
        appmod.urlopen = _boom
        check(appmod._post_reviewer(5085, "AB") is False, "a failing socket answers False, never raises")
    finally:
        appmod.urlopen = orig


def test_push_after_health_posts_once_then_stops():
    posted = []
    orig = appmod._post_reviewer
    appmod._post_reviewer = lambda port, rev, timeout=2.0: posted.append((port, rev)) or True
    try:
        seen = {"n": 0}

        def _health(port):
            seen["n"] += 1
            return seen["n"] >= 2  # first poll not ready, second ready
        appmod._push_reviewer_after_health(5080, "LO", _health, interval_s=0.01, background=False)
        check(posted == [(5080, "LO")], f"one post after health: {posted}")
        check(seen["n"] == 2, f"polls until healthy then stops: {seen['n']}")
        posted.clear()
        appmod._push_reviewer_after_health(5080, "LO", lambda p: False,
                                           deadline_s=0.02, interval_s=0.01, background=False)
        check(posted == [], "gives up silently at the deadline")
        check(appmod._push_reviewer_after_health(None, "LO", lambda p: True, background=False) is None
              and posted == [], "no port -> nothing")
    finally:
        appmod._post_reviewer = orig


# ---- launch env + push wiring ----------------------------------------------

def _stub_launch(captured, pushes, port=5083, result=None):
    orig = {"flask": appmod.runner.run_flask_stage, "ffp": appmod.find_free_port,
            "push": appmod._push_reviewer_after_health}

    def _fake_flask_stage(runner_key, cmd, prt, cwd=None, env_extra=None):
        captured.update(runner_key=runner_key, cmd=cmd, port=prt, env_extra=env_extra or {})
        return result if result is not None else {"pid": 99999, "port": prt}

    appmod.runner.run_flask_stage = _fake_flask_stage
    appmod.find_free_port = lambda preferred: preferred
    appmod._push_reviewer_after_health = (
        lambda prt, rev, health_fn, **kw: pushes.append((prt, rev, health_fn == appmod.runner.health_check)))
    return orig


def _restore_launch(orig):
    appmod.runner.run_flask_stage = orig["flask"]
    appmod.find_free_port = orig["ffp"]
    appmod._push_reviewer_after_health = orig["push"]


def test_editmasks_launch_env_carries_reviewer_and_pushes_after_health():
    tmp = tempfile.mkdtemp(prefix="reviewer_env_launch_", dir=_TMP_ROOT)
    captured, pushes = {}, []
    orig = _stub_launch(captured, pushes)
    try:
        _reset_states()
        state, export_dir = _make_project(tmp, access_log=[
            {"ts": "2026-08-01T09:00:00-04:00", "initials": "lo", "action": "create"},
            {"ts": "2026-08-26T10:00:00-04:00", "initials": "ab", "action": "open"}])
        client = appmod.create_app().test_client()
        appmod.current_project = state
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("success") is True, f"launch failed: {r.status_code} {data}")
        env = captured.get("env_extra") or {}
        check(env.get("TCRMP_REVIEWER") == "AB", f"launch env must carry TCRMP_REVIEWER=AB: {env.get('TCRMP_REVIEWER')!r}")
        check(env.get("TCRMP_EXPORT_DIR") == export_dir, "export dir unchanged by the reviewer wiring")
        check(pushes == [(5083, "AB", True)],
              f"post-health push must be scheduled with the runner health check: {pushes}")
    finally:
        _restore_launch(orig)
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


def test_editmasks_already_running_pushes_reviewer():
    tmp = tempfile.mkdtemp(prefix="reviewer_env_reuse_", dir=_TMP_ROOT)
    captured, pushes = {}, []
    orig = _stub_launch(captured, pushes, result={"error": "Step editmasks is already running"})
    try:
        _reset_states()
        state, _ = _make_project(tmp, initials="lo")
        client = appmod.create_app().test_client()
        appmod.current_project = state
        appmod.runner.ports["editmasks"] = 5083
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json() or {}
        check(r.status_code == 200 and data.get("already_running") is True, f"expected reuse answer: {data}")
        check(pushes == [(5083, "LO", True)], f"a reused process still gets the reviewer pushed: {pushes}")
    finally:
        appmod.runner.ports.pop("editmasks", None)
        _restore_launch(orig)
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_reviewer_recorded_means_no_env_and_no_push():
    tmp = tempfile.mkdtemp(prefix="reviewer_env_blank_", dir=_TMP_ROOT)
    captured, pushes = {}, []
    orig = _stub_launch(captured, pushes)
    try:
        _reset_states()
        state, _ = _make_project(tmp)  # no initials, empty access_log
        client = appmod.create_app().test_client()
        appmod.current_project = state
        r = client.post("/api/step/editmasks/run", json={})
        check(r.status_code == 200, f"launch failed: {r.get_json()}")
        env = captured.get("env_extra") or {}
        check("TCRMP_REVIEWER" not in env, "nothing recorded -> no TCRMP_REVIEWER (never inferred)")
        check(pushes == [(5083, "", True)], f"push is scheduled with '' and is a no-op: {pushes}")
    finally:
        _restore_launch(orig)
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


def test_step5_launch_env_carries_reviewer():
    tmp = tempfile.mkdtemp(prefix="reviewer_env_step5_", dir=_TMP_ROOT)
    captured, pushes = {}, []
    orig = _stub_launch(captured, pushes, port=5070)
    try:
        _reset_states()
        state, _ = _make_project(tmp, access_log=[{"initials": "cd", "action": "open"}])
        client = appmod.create_app().test_client()
        appmod.current_project = state
        r = client.post("/api/step/5/run")
        data = r.get_json() or {}
        check(r.status_code == 200, f"step 5 launch failed: {r.status_code} {data}")
        env = captured.get("env_extra") or {}
        check(env.get("TCRMP_REVIEWER") == "CD", f"step 5 env must carry TCRMP_REVIEWER: {env}")
        check(pushes and pushes[0][1] == "CD", f"step 5 push scheduled: {pushes}")
    finally:
        _restore_launch(orig)
        appmod.current_project = None
        _reset_states()
        shutil.rmtree(tmp, ignore_errors=True)


# ---- /api/composites --------------------------------------------------------

def test_api_composites_lists_exports_under_composite_root():
    root = tempfile.mkdtemp(prefix="reviewer_env_composite_", dir=_TMP_ROOT)
    orig_root = appmod._composite_root
    orig_bbox_root = appmod._bbox_root
    appmod._composite_root = lambda: root
    # The list also carries staged box datasets; keep this box's own out.
    appmod._bbox_root = lambda: os.path.join(root, "no_bbox_datasets")
    try:
        good = os.path.join(root, "composite_26-08-26_1200")
        os.makedirs(os.path.join(good, "all_images"))
        os.makedirs(os.path.join(good, "all_labels"))
        for n in ("a.jpeg", "b.jpeg"):
            open(os.path.join(good, "all_images", n), "w").close()
        with open(os.path.join(good, "all_labels", "a.txt"), "w") as f:
            f.write("0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.1 0.1 0.2 0.2 0.3 0.3\n")
        with open(os.path.join(good, "all_labels", "b.txt"), "w") as f:
            f.write("")  # negative image
        with open(os.path.join(good, "data.yaml"), "w") as f:
            f.write("path: x\ntrain: all_images\nval: all_images\nnc: 2\nnames:\n  0: MC\n  1: PA\n")
        os.makedirs(os.path.join(root, "not_a_composite"))  # no data.yaml -> skipped
        client = appmod.create_app().test_client()
        r = client.get("/api/composites")
        data = r.get_json() or {}
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        comps = data.get("composites") or []
        check(len(comps) == 1 and comps[0]["name"] == "composite_26-08-26_1200", f"listing: {comps}")
        c = comps[0]
        check(c["path"] == good and c["n_images"] == 2 and c["n_masks"] == 2, f"counts: {c}")
        check(c["labels"] == ["MC", "PA"], f"labels from data.yaml: {c['labels']}")
        import re as _re
        check(bool(_re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", c["created_at"] or "")),
              f"created_at must be AST 'YYYY-MM-DD HH:MM:SS' like the matrix list: {c['created_at']!r}")
        check(data.get("root") == root, "root echoed")
        shutil.rmtree(root)
        r = client.get("/api/composites")
        check((r.get_json() or {}).get("composites") == [], "missing root -> empty list")
    finally:
        appmod._composite_root = orig_root
        appmod._bbox_root = orig_bbox_root
        shutil.rmtree(root, ignore_errors=True)


def test_api_composites_skips_entries_outside_the_root():
    """L13: a symlink under _composite that resolves outside the root would be
    offered in the Step 6 select and then refused by _validate_composite_source;
    the listing must skip it, mirroring the guard."""
    base = tempfile.mkdtemp(prefix="reviewer_env_symlink_", dir=_TMP_ROOT)
    root = os.path.join(base, "_composite")
    outside = os.path.join(base, "outside_export")
    for d in (root, os.path.join(outside, "all_images")):
        os.makedirs(d)
    with open(os.path.join(outside, "data.yaml"), "w") as f:
        f.write("names:\n  0: MC\n")
    real = os.path.join(root, "composite_real")
    os.makedirs(os.path.join(real, "all_images"))
    with open(os.path.join(real, "data.yaml"), "w") as f:
        f.write("names:\n  0: MC\n")
    os.symlink(outside, os.path.join(root, "sneaky"))
    orig_root = appmod._composite_root
    orig_bbox_root = appmod._bbox_root
    appmod._composite_root = lambda: root
    appmod._bbox_root = lambda: os.path.join(base, "no_bbox_datasets")
    try:
        names = [c["name"] for c in appmod._list_composites()]
        check(names == ["composite_real"], f"symlink out of the root must be skipped: {names}")
        _, err = appmod._validate_composite_source(os.path.join(root, "sneaky"))
        check(bool(err), "the guard refuses the same entry, so listing and guard agree")
        r = appmod.create_app().test_client().get("/api/composites")
        check([c["name"] for c in (r.get_json() or {}).get("composites", [])] == ["composite_real"], "route agrees")
    finally:
        appmod._composite_root = orig_root
        appmod._bbox_root = orig_bbox_root
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    print("reviewer identity + composite listing (orchestrator wiring)\n")
    try:
        for fn in (test_reviewer_latest_access_entry_wins,
                   test_reviewer_falls_back_to_creator,
                   test_post_reviewer_is_one_urllib_post_with_two_second_timeout,
                   test_push_after_health_posts_once_then_stops,
                   test_editmasks_launch_env_carries_reviewer_and_pushes_after_health,
                   test_editmasks_already_running_pushes_reviewer,
                   test_no_reviewer_recorded_means_no_env_and_no_push,
                   test_step5_launch_env_carries_reviewer,
                   test_api_composites_lists_exports_under_composite_root,
                   test_api_composites_skips_entries_outside_the_root):
            run(fn)
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
        if os.path.exists(_TMP_ROOT):
            print(f"  WARN temp root still present: {_TMP_ROOT}")
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    for name, _, tb in failed:
        print(f"--- {name}\n{tb}")
    sys.exit(1 if failed else 0)
