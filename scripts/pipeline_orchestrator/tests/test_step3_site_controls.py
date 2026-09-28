"""Task 11: Step 3 site-balance controls (checkbox + max-share field) surfaced
through app.py's _run_step3 command building and completion status, plus the
editMasks preferred_port move off the colliding 5085.

No-pytest harness: run with env/bin/python <this file>.

Covers:
  config round-trip  - site_balance (bool) and max_site_frac (float) survive
                        a /api/project/config/3 PUT and read back from
                        /api/project/state unchanged.
  _run_step3 command - --no-site-balance is passed ONLY when site_balance is
                        explicitly False; --max-site-frac is always passed,
                        carrying the configured (or defaulted) value.
  completion payload - /api/step/3/status attaches `site_summary` (parsed
                        from output/selection_summary.json) on the
                        running -> completed transition when the file
                        exists; omits the key cleanly for a legacy run with
                        no JSON, and never attaches it for other steps.
  port move          - editMasks launches on 5083, never 5085 (vicariusLLM's
                        live autostart port, TODO T20260826-223636).
  template markers   - the two new #panel-3 controls exist, default correctly,
                        and carry plain-language title tooltips (checkbox
                        cites the Coral Bay example).
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

# Registry safety (standing rule for this batch): redirect the mask registry
# and clip index to throwaway scratch dirs before anything imports app.py, so
# a stray route reachable from these tests can never touch real project data.
_TMP_ROOT = tempfile.mkdtemp(prefix="step3_site_controls_")
os.environ["TCRMP_MASK_REGISTRY_DIR"] = os.path.join(_TMP_ROOT, "_mask_registry")
os.environ["TCRMP_CLIP_INDEX_DIR"] = os.path.join(_TMP_ROOT, "_clip_index")
os.makedirs(os.environ["TCRMP_MASK_REGISTRY_DIR"], exist_ok=True)
os.makedirs(os.environ["TCRMP_CLIP_INDEX_DIR"], exist_ok=True)


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


# Stub subprocess.run for the WHOLE module before app.py is imported: the
# editmasks launch route reaches _kill_all_annotators(), which shells out to
# `ss`/`pgrep` against the box's real live processes (standing rule: never
# reach a live annotator or vicariusLLM from a test).
subprocess.run = lambda *a, **k: _FakeCompleted()

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402

# Test isolation (2026-09-28 review): _kill_all_annotators' flush-before-kill
# (POST /api/flush, pipeline_orchestrator/app.py) reaches the network even
# when subprocess.run/Popen are stubbed, since it goes through urllib, not
# subprocess. This file's editmasks/run call drives it, so
# appmod._flush_annotator is stubbed here (never a real save, never a real
# socket) and socket.socket is guarded so any code path that still tries to
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
        "created_at": "x", "updated_at": "x", "current_step": 3,
        "steps": steps,
    }
    return state


def _client_with_project():
    tmp = tempfile.mkdtemp(dir=_TMP_ROOT)
    state = _make_project(tmp)
    client = appmod.create_app().test_client()
    appmod.current_project = state
    return client, tmp, state


# ── config round-trip ────────────────────────────────────────────────────

def test_config_round_trip_site_balance_and_max_site_frac():
    client, tmp, state = _client_with_project()
    try:
        r = client.put("/api/project/config/3", json={
            "target_species": "PA, OFRA", "site_balance": False, "max_site_frac": 0.4,
        })
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")

        r = client.get("/api/project/state")
        cfg = r.get_json()["state"]["steps"]["3"]["config"]
        check(cfg.get("site_balance") is False, f"site_balance should round-trip as False: {cfg}")
        check(cfg.get("max_site_frac") == 0.4, f"max_site_frac should round-trip as 0.4: {cfg}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_round_trip_defaults():
    client, tmp, state = _client_with_project()
    try:
        r = client.put("/api/project/config/3", json={
            "target_species": "PA", "site_balance": True, "max_site_frac": 0.25,
        })
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")

        r = client.get("/api/project/state")
        cfg = r.get_json()["state"]["steps"]["3"]["config"]
        check(cfg.get("site_balance") is True, f"site_balance should round-trip as True: {cfg}")
        check(cfg.get("max_site_frac") == 0.25, f"max_site_frac should round-trip as 0.25: {cfg}")
    finally:
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


# ── _run_step3 command building ──────────────────────────────────────────

def test_run_step3_passes_no_site_balance_only_when_unchecked():
    client, tmp, state = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/3", json={
            "target_species": "PA", "site_balance": False, "max_site_frac": 0.3,
        })
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")

        cmd = captured.get("cmd") or []
        check("--no-site-balance" in cmd, f"--no-site-balance must be passed when unchecked: {cmd}")
        check("--max-site-frac" in cmd, f"--max-site-frac must always be passed: {cmd}")
        i = cmd.index("--max-site-frac")
        check(cmd[i + 1] == "0.3", f"--max-site-frac value should be 0.3: {cmd[i + 1]!r}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_step3_omits_no_site_balance_when_checked():
    client, tmp, state = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/3", json={
            "target_species": "PA", "site_balance": True, "max_site_frac": 0.25,
        })
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")

        cmd = captured.get("cmd") or []
        check("--no-site-balance" not in cmd,
              f"--no-site-balance must be absent when site balance is checked: {cmd}")
        check("--max-site-frac" in cmd, f"--max-site-frac must always be passed: {cmd}")
        i = cmd.index("--max-site-frac")
        check(cmd[i + 1] == "0.25", f"--max-site-frac value should be 0.25: {cmd[i + 1]!r}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_step3_defaults_when_config_never_set():
    """A project whose config PUT never mentioned the new keys (an older
    saved project, or a fresh one before the user touches the new controls)
    must still run with site balance ON and the 0.25 cap, matching
    select_images.py's own argparse defaults (site_balance true unless
    --no-site-balance; --max-site-frac default 0.25)."""
    client, tmp, state = _client_with_project()
    captured = {}
    orig = appmod.runner.run_cli_stage

    def _fake(step, cmd, cwd=None, env_extra=None):
        captured["cmd"] = cmd
        return {"pid": 1}

    appmod.runner.run_cli_stage = _fake
    try:
        r = client.put("/api/project/config/3", json={"target_species": "PA"})
        check(r.status_code == 200, f"config PUT failed: {r.get_json()}")
        r = client.post("/api/step/3/run")
        check(r.status_code == 200, f"run failed: {r.get_json()}")

        cmd = captured.get("cmd") or []
        check("--no-site-balance" not in cmd,
              f"no config keys set -> site balance stays on (no disable flag): {cmd}")
        i = cmd.index("--max-site-frac")
        check(cmd[i + 1] == "0.25", f"--max-site-frac should default to 0.25: {cmd[i + 1]!r}")
    finally:
        appmod.runner.run_cli_stage = orig
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_step3_clamps_max_site_frac_server_side():
    """/api/project/config PUT accepts any JSON value with no server-side
    check (the HTML min/max attributes are cosmetic; collectConfig's isNaN
    guard is client-only), so a bad value can still reach _run_step3 via a
    manual PUT or a corrupted saved project. _resolve_max_site_frac must
    catch it before it is str()'d into the CLI: non-numeric/NaN -> 0.25
    (select_images.py's own default); <=0 or >1 -> the nearest bound of
    [0.05, 1.0]; anything already in range passes through unchanged."""
    cases = [
        ("abc", "0.25"),   # non-numeric -> fallback to the script's own default
        (-1, "0.05"),      # <=0 -> clamp to the lower bound
        (5, "1.0"),        # >1 -> clamp to the upper bound
        (0.3, "0.3"),      # already in range -> passes through unchanged
    ]
    for raw, expected in cases:
        client, tmp, state = _client_with_project()
        captured = {}
        orig = appmod.runner.run_cli_stage

        def _fake(step, cmd, cwd=None, env_extra=None):
            captured["cmd"] = cmd
            return {"pid": 1}

        appmod.runner.run_cli_stage = _fake
        try:
            r = client.put("/api/project/config/3", json={
                "target_species": "PA", "max_site_frac": raw,
            })
            check(r.status_code == 200, f"config PUT failed for {raw!r}: {r.get_json()}")
            r = client.post("/api/step/3/run")
            check(r.status_code == 200, f"run failed for {raw!r}: {r.get_json()}")

            cmd = captured.get("cmd") or []
            i = cmd.index("--max-site-frac")
            check(cmd[i + 1] == expected,
                  f"max_site_frac={raw!r} should compose --max-site-frac {expected}, got {cmd[i + 1]!r}")
        finally:
            appmod.runner.run_cli_stage = orig
            appmod.current_project = None
            shutil.rmtree(tmp, ignore_errors=True)


# ── completion payload: site_summary ─────────────────────────────────────

def _write_selection_summary(step_dir, **overrides):
    doc = {
        "per_site": {"CB": 40, "TG": 30, "SR": 30},
        "per_species": {"PA": {"target": 100, "achieved": 100, "reason": "ok"}},
        "cap_frac": 0.4,
        "cap_ok": True,
        "cap_satisfiable": True,
        "n_selected": 100,
        "shortfalls": [],
    }
    doc.update(overrides)
    with open(os.path.join(step_dir, "selection_summary.json"), "w") as f:
        json.dump(doc, f)
    return doc


def test_status_completion_carries_site_summary_when_json_exists():
    client, tmp, state = _client_with_project()
    step_dir = state["steps"]["3"]["dir"]
    doc = _write_selection_summary(step_dir)
    state["steps"]["3"]["status"] = "running"
    state["steps"]["3"]["started_at"] = "2026-09-01T00:00:00"
    orig_poll = appmod.runner.poll_status
    appmod.runner.poll_status = lambda step: {"running": False, "exit_code": 0}
    try:
        r = client.get("/api/step/3/status")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("completed") is True, f"expected completed=true: {data}")
        check(data.get("site_summary") == doc, f"site_summary must match selection_summary.json: {data}")
    finally:
        appmod.runner.poll_status = orig_poll
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_status_completion_omits_site_summary_when_absent():
    """A legacy run (select_images.py invoked before this batch, or the JSON
    was never written) must still complete cleanly: no site_summary key at
    all, never a null placeholder or a 500."""
    client, tmp, state = _client_with_project()
    state["steps"]["3"]["status"] = "running"
    state["steps"]["3"]["started_at"] = "2026-09-01T00:00:00"
    orig_poll = appmod.runner.poll_status
    appmod.runner.poll_status = lambda step: {"running": False, "exit_code": 0}
    try:
        r = client.get("/api/step/3/status")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("completed") is True, f"expected completed=true: {data}")
        check("site_summary" not in data, f"no JSON on disk -> no site_summary key at all: {data}")
    finally:
        appmod.runner.poll_status = orig_poll
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


def test_status_no_site_summary_for_other_steps():
    """The site_summary attachment is step-3-specific; a step-1 completion
    must never carry it even if a stray selection_summary.json exists in its
    dir (defensive; not expected in practice)."""
    client, tmp, state = _client_with_project()
    step1_dir = state["steps"]["1"]["dir"]
    _write_selection_summary(step1_dir)
    state["steps"]["1"]["status"] = "running"
    state["steps"]["1"]["started_at"] = "2026-09-01T00:00:00"
    orig_poll = appmod.runner.poll_status
    appmod.runner.poll_status = lambda step: {"running": False, "exit_code": 0}
    try:
        r = client.get("/api/step/1/status")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check("site_summary" not in data, f"step 1 completion must never carry site_summary: {data}")
    finally:
        appmod.runner.poll_status = orig_poll
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


# ── port move: editMasks preferred_port == 5083 ──────────────────────────

def _write_segmentations(export_dir):
    seg_dir = os.path.join(export_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as f:
        json.dump({"img0.jpeg": {"masks": [], "reviewed": False, "exported": False}}, f)


def test_editmasks_preferred_port_is_5083_not_5085():
    client, tmp, state = _client_with_project()
    export_dir = os.path.join(state["project_dir"], pm.STEP_DIRS["step4test"])
    os.makedirs(export_dir, exist_ok=True)
    _write_segmentations(export_dir)
    appmod.editmasks_state = {"running": False, "port": None, "started_at": None,
                              "out_dir": None, "project_id": None, "input_dir": None}
    orig_flask_stage = appmod.runner.run_flask_stage
    orig_find_free_port = appmod.find_free_port
    captured = {}

    def _fake_flask_stage(runner_key, cmd, port, cwd=None, env_extra=None):
        captured["port"] = port
        return {"pid": 1, "port": port}

    appmod.runner.run_flask_stage = _fake_flask_stage
    # Identity find_free_port exposes the real preferred_port constant _run
    # uses, exactly the pattern test_editmasks_routes.py's happy path uses.
    appmod.find_free_port = lambda preferred: preferred
    try:
        r = client.post("/api/step/editmasks/run", json={})
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("ui_port") == 5083, f"editMasks must default to port 5083, not 5085: {data}")
        check(captured.get("port") == 5083, f"the launched port must be 5083: {captured}")
        check(data.get("ui_port") != 5085, "5085 stays vicariusLLM's live autostart port")
    finally:
        appmod.runner.run_flask_stage = orig_flask_stage
        appmod.find_free_port = orig_find_free_port
        appmod.current_project = None
        shutil.rmtree(tmp, ignore_errors=True)


# ── template: new controls carry plain-language tooltips ────────────────

def _rendered_page():
    """The window as the browser receives it.

    Every tooltip is now set once into a Jinja variable and handed to both the
    "?" glyph and the control it explains, so the raw template holds the
    literal "{{ t }}" where the sentence used to be. A test that checks tooltip
    wording has to read the rendered page.

    Returns:
        The rendered page as text.
    """
    return appmod.create_app().test_client().get("/").get_data(as_text=True)


def test_template_has_site_balance_checkbox_with_tooltip():
    html = _rendered_page()
    m = re.search(r'id="s3-site-balance"[^>]*title="([^"]+)"', html)
    check(m is not None, "#s3-site-balance checkbox with a title tooltip not found")
    tip = m.group(1)
    check("Coral Bay" in tip, f"checkbox tooltip must cite the Coral Bay overrepresentation example: {tip!r}")
    check("--no-site-balance" in tip, f"checkbox tooltip should name the flag it controls: {tip!r}")
    tag_start = html.index('id="s3-site-balance"')
    tag = html[max(0, tag_start - 60):tag_start + 40]
    check("checked" in tag, f"checkbox must default to checked (site balance on by default): {tag!r}")


def test_template_has_max_site_frac_field_with_tooltip():
    html = _rendered_page()
    m = re.search(r'id="s3-max-site-frac"[^>]*value="([^"]+)"', html)
    check(m is not None, "#s3-max-site-frac field not found")
    check(m.group(1) == "0.25", f"field must default to 0.25: {m.group(1)!r}")
    m2 = re.search(r'id="s3-max-site-frac"[^>]*title="([^"]+)"', html)
    check(m2 is not None, "#s3-max-site-frac field must carry a title tooltip")
    check("--max-site-frac" in m2.group(1), f"tooltip should name the flag it controls: {m2.group(1)!r}")
    # The Wave 1 task's existing "no evenly across" claim (test_ui_sweep_
    # 20260901.py::test_step3_progress_and_flag_copy) must still hold now
    # that these controls exist inside #panel-3.
    i = html.index('id="panel-3"')
    panel = html[i:html.index('id="panel-', i + 20)]
    check("evenly across" not in panel,
          "new Step 3 copy must not reintroduce the banned 'evenly across' phrase")


if __name__ == "__main__":
    print("TASK 11 -- Step 3 site-balance controls + editMasks port move\n")
    try:
        print("CONFIG ROUND-TRIP:")
        run(test_config_round_trip_site_balance_and_max_site_frac)
        run(test_config_round_trip_defaults)
        print("_RUN_STEP3 COMMAND:")
        run(test_run_step3_passes_no_site_balance_only_when_unchecked)
        run(test_run_step3_omits_no_site_balance_when_checked)
        run(test_run_step3_defaults_when_config_never_set)
        run(test_run_step3_clamps_max_site_frac_server_side)
        print("COMPLETION PAYLOAD:")
        run(test_status_completion_carries_site_summary_when_json_exists)
        run(test_status_completion_omits_site_summary_when_absent)
        run(test_status_no_site_summary_for_other_steps)
        print("PORT MOVE:")
        run(test_editmasks_preferred_port_is_5083_not_5085)
        print("TEMPLATE MARKERS:")
        run(test_template_has_site_balance_checkbox_with_tooltip)
        run(test_template_has_max_site_frac_field_with_tooltip)
    finally:
        shutil.rmtree(_TMP_ROOT, ignore_errors=True)
        if os.path.exists(_TMP_ROOT):
            print(f"  WARN temp root still present: {_TMP_ROOT}")

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
    sys.exit(1 if failed else 0)
