"""GPU right-of-way gate on the VEGA stage runner (docs/superpowers/plans/
2026-09-17-gpu-right-of-way.md, Task 9): StageRunner.run_cli_stage and
run_flask_stage ask gpu_claim.check() before every Popen for a runner key
listed in module.yaml's gpu.stages block, claim with the child's own pid
immediately after, and release when _capture_output sees the process exit.
A runner key absent from that map is never gated: it never calls the helper
at all.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_gpu_gate.py

Standing rule for every orchestrator suite: subprocess.Popen is stubbed at
module level before stage_runner is imported, so nothing here can ever spawn
a real process (relevant right now: a real VOYAGER reconstruction is running
on this box and must never be touched by a test).
"""

import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
# Every fixture file/dir this run created, checked for removal at the end.
_FIXTURES = []


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, print PASS or FAIL, and record the outcome."""
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# Standing rule: no orchestrator test may reach a real subprocess. Popen is
# blocked at module level, before stage_runner (which calls subprocess.Popen
# by attribute lookup on the subprocess module) is even imported.
# ---------------------------------------------------------------------------
class _FakeProc:
    """Stands in for a Popen handle. stdout/stderr are real OS pipes, kept
    open until finish() closes their write ends, so _capture_output's
    blocking readline() reaches end-of-stream (and releases the GPU claim)
    only when a test calls finish() - the same two-phase lifecycle a real
    child process has (pipes open while it runs, EOF exactly at exit). An
    already-EOF stream (io.StringIO("")) cannot simulate that: it would race
    the release ahead of every assertion a test makes about the moment
    right after spawn, before the process has "exited"."""

    def __init__(self, pid):
        self.pid = pid
        out_r, self._out_w = os.pipe()
        err_r, self._err_w = os.pipe()
        self.stdout = os.fdopen(out_r, "r")
        self.stderr = os.fdopen(err_r, "r")
        self._exit_code = None

    def poll(self):
        """None while "alive" (finish() not yet called), else its exit code."""
        return self._exit_code

    def finish(self, exit_code=0):
        """End the stand-in: close the pipes' write ends so
        _capture_output's readline() loop reaches end-of-stream and this
        step's GPU claim, if any, is released. Safe to call more than once."""
        if self._exit_code is not None:
            return
        self._exit_code = exit_code
        os.close(self._out_w)
        os.close(self._err_w)


_NEXT_FAKE_PID = [420000]
_ALL_FAKE_PROCS = []


def _fake_popen(cmd, **kwargs):
    """subprocess.Popen stand-in: hands out a fresh fake pid per call and
    spawns nothing. Recorded on _POPEN_CALLS for the never-spawns assertions
    and on _ALL_FAKE_PROCS so every pipe opened during the run is closed at
    the end even if the test that created it never called finish()."""
    _NEXT_FAKE_PID[0] += 1
    proc = _FakeProc(_NEXT_FAKE_PID[0])
    _POPEN_CALLS.append((cmd, kwargs))
    _ALL_FAKE_PROCS.append(proc)
    return proc


_POPEN_CALLS = []
subprocess.Popen = _fake_popen

import stage_runner  # noqa: E402
import gpu_claim  # noqa: E402


# ---------------------------------------------------------------------------
# gpu_claim stand-ins: deterministic decisions with no network call, no real
# 5090, recorded so a test can assert exactly what was asked and claimed.
# ---------------------------------------------------------------------------
_CHECK_CALLS = []
_CLAIM_CALLS = []
_RELEASE_CALLS = []


def _install_gpu_stub(check_decision, claim_decision="granted"):
    """Replace gpu_claim.check/claim/release with recording stand-ins.

    Parameters:
        check_decision: the "decision" value gpu_claim.check() returns
            every call, one of "granted", "refused", "needs_pin",
            "unregistered".
        claim_decision: the "decision" value gpu_claim.claim() returns.

    Returns: None. Call _restore_gpu() afterwards to put the real functions
        back (harmless here since they only reach the network, but keeps
        every test's stub scoped to itself).
    """
    _CHECK_CALLS.clear()
    _CLAIM_CALLS.clear()
    _RELEASE_CALLS.clear()

    def _check(module, tier=None, by="LO"):
        _CHECK_CALLS.append({"module": module, "tier": tier, "by": by})
        result = {"decision": check_decision}
        if check_decision in ("refused", "needs_pin"):
            result["message"] = f"GPU right of way: {tier} is busy for testing."
        return result

    def _claim(module, tier=None, run_id="", pid=None, by="LO", bypass=False, reason=""):
        _CLAIM_CALLS.append({"module": module, "tier": tier, "run_id": run_id,
                              "pid": pid, "by": by})
        result = {"decision": claim_decision}
        if claim_decision == "granted":
            result["claim_id"] = f"c-test-{pid}"
        return result

    def _release(claim_id=None, pid=None):
        _RELEASE_CALLS.append({"claim_id": claim_id, "pid": pid})
        return {"released": True}

    gpu_claim.check = _check
    gpu_claim.claim = _claim
    gpu_claim.release = _release


def _restore_gpu():
    """Put the real gpu_claim.check/claim/release back after a stub test."""
    import importlib
    importlib.reload(gpu_claim)


def _write_module_yaml(stages):
    """A throwaway module.yaml fixture with only a gpu.stages block.

    Parameters:
        stages: dict to write under gpu.stages (may be empty, for "no stage
            is gated").

    Returns: the fixture file's path. Tracked in _FIXTURES for cleanup.
    """
    fd, path = tempfile.mkstemp(prefix="gpu_gate_module_", suffix=".yaml")
    _FIXTURES.append(path)
    with os.fdopen(fd, "w") as fh:
        fh.write("gpu:\n  stages:\n")
        for key, tier in stages.items():
            fh.write(f'    "{key}": {tier}\n')
    return path


def _new_runner(stages):
    """A StageRunner reading gpu.stages from a fresh fixture file."""
    stage_runner.MODULE_YAML_PATH = __import__("pathlib").Path(_write_module_yaml(stages))
    return stage_runner.StageRunner()


def _drain(runner, step, timeout=2.0):
    """Wait for step's log thread (and its _gpu_release call) to finish."""
    thread = runner.log_threads.get(step)
    if thread is not None:
        thread.join(timeout)


# ---------------------------------------------------------------------------
# GATE: a gated stage refused before Popen
# ---------------------------------------------------------------------------

def test_gated_stage_refused_never_spawns():
    _install_gpu_stub(check_decision="refused")
    runner = _new_runner({"6": "vega"})
    before = len(_POPEN_CALLS)
    result = runner.run_cli_stage(6, ["true"])
    check(result == {"error": "GPU right of way: vega is busy for testing.", "busy": True},
          f"a refused check returns the busy-result shape: {result}")
    check(len(_POPEN_CALLS) == before, "Popen was never called for a refused stage")
    check(_CLAIM_CALLS == [], "claim() is never reached when check() refuses")
    check(6 not in runner.processes, "no process was registered for the refused step")


def test_gated_stage_refused_needs_pin_also_never_spawns():
    # StageRunner has no admin-PIN concept of its own: needs_pin is a
    # refusal here too (a human-tier stage over a model-tier holder).
    _install_gpu_stub(check_decision="needs_pin")
    runner = _new_runner({"step4test": "human"})
    before = len(_POPEN_CALLS)
    result = runner.run_flask_stage("step4test", ["true"], 5080)
    check(result.get("busy") is True, f"needs_pin is refused like a refusal: {result}")
    check(len(_POPEN_CALLS) == before, "Popen was never called on needs_pin")


# ---------------------------------------------------------------------------
# GATE: a gated stage granted, spawns, and claims with the child pid
# ---------------------------------------------------------------------------

def test_gated_stage_granted_spawns_and_claims_child_pid():
    _install_gpu_stub(check_decision="granted", claim_decision="granted")
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"])
    try:
        check("pid" in result and "error" not in result,
              f"a granted check lets the stage spawn: {result}")
        check(len(_CHECK_CALLS) == 1 and _CHECK_CALLS[0]["tier"] == "vega",
              f"check() was asked once at the mapped tier: {_CHECK_CALLS}")
        check(len(_CLAIM_CALLS) == 1, f"claim() was called exactly once: {_CLAIM_CALLS}")
        check(_CLAIM_CALLS[0]["pid"] == result["pid"],
              f"the claim carries the CHILD's pid, not the orchestrator's: "
              f"claimed {_CLAIM_CALLS[0]['pid']!r} vs spawned {result['pid']!r}")
        check(_CLAIM_CALLS[0]["module"] == "reef_point_seg", "the claim names this module")
        # The fake process's pipes are still open (finish() not called yet),
        # so its log-reader thread is blocked and cannot have released this
        # claim already: it is genuinely still held right after spawn.
        check(runner.gpu_claims.get(6) == f"c-test-{result['pid']}",
              f"the granted claim id is remembered against the step: {runner.gpu_claims}")
    finally:
        runner.processes[6].finish()


def test_gated_flask_stage_granted_spawns_and_claims_child_pid():
    _install_gpu_stub(check_decision="granted", claim_decision="granted")
    runner = _new_runner({"step4test": "human"})
    result = runner.run_flask_stage("step4test", ["true"], 5080)
    try:
        check("pid" in result and result.get("port") == 5080,
              f"a granted flask stage spawns and keeps its port: {result}")
        check(_CLAIM_CALLS[0]["tier"] == "human" and _CLAIM_CALLS[0]["pid"] == result["pid"],
              f"the flask stage claims at its mapped tier with the child pid: {_CLAIM_CALLS}")
    finally:
        runner.processes["step4test"].finish()


# ---------------------------------------------------------------------------
# UNGATED: a runner key absent from gpu.stages never calls the helper
# ---------------------------------------------------------------------------

def test_ungated_stage_never_calls_the_helper():
    _install_gpu_stub(check_decision="refused")  # would refuse everything if asked
    runner = _new_runner({"6": "vega"})  # step 1 deliberately absent
    result = runner.run_cli_stage(1, ["true"])
    check("pid" in result, f"an ungated stage always spawns: {result}")
    check(_CHECK_CALLS == [], f"check() was never called for an ungated stage: {_CHECK_CALLS}")
    check(_CLAIM_CALLS == [], f"claim() was never called for an ungated stage: {_CLAIM_CALLS}")


def test_empty_gpu_stages_gates_nothing():
    _install_gpu_stub(check_decision="refused")
    runner = _new_runner({})
    result = runner.run_cli_stage(6, ["true"])
    check("pid" in result, f"no gpu.stages entries means nothing is gated: {result}")
    check(_CHECK_CALLS == [], "check() is never called when gpu.stages is empty")


# ---------------------------------------------------------------------------
# UNREGISTERED: 5090 unreachable lets the stage run
# ---------------------------------------------------------------------------

def test_unregistered_helper_lets_the_stage_run():
    _install_gpu_stub(check_decision="unregistered", claim_decision="unregistered")
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"])
    check("pid" in result, f"an unregistered check proceeds rather than refusing: {result}")
    check(len(_CHECK_CALLS) == 1, "check() was still asked once")
    check(6 not in runner.gpu_claims,
          "an unregistered claim leaves nothing to release later")


def test_malformed_check_reply_fails_open_rather_than_crashing():
    # A defensive edge case: gpu_claim never raises per its own contract, but
    # an unexpected reply shape (missing "decision") must not crash a launch.
    _install_gpu_stub(check_decision="granted")
    gpu_claim.check = lambda module, tier=None, by="LO": {}
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"])
    check("pid" in result, f"a malformed check reply fails open, not closed: {result}")


# ---------------------------------------------------------------------------
# RELEASE: the claim is released when _capture_output sees the process exit
# ---------------------------------------------------------------------------

def test_claim_is_released_when_the_process_exits():
    _install_gpu_stub(check_decision="granted", claim_decision="granted")
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"])
    pid = result["pid"]
    check(runner.gpu_claims.get(6) == f"c-test-{pid}",
          "the claim is recorded and held while the process's pipes stay open")
    check(_RELEASE_CALLS == [], "nothing is released before the process exits")
    # "The process exits": close its pipes' write ends, the same signal a
    # real child's exit sends its parent's readers.
    runner.processes[6].finish()
    _drain(runner, 6)
    check(_RELEASE_CALLS == [{"claim_id": f"c-test-{pid}", "pid": None}],
          f"release() is called with the claim id once the process exits: {_RELEASE_CALLS}")
    check(6 not in runner.gpu_claims, "the claim is forgotten once released")


def test_ungated_stage_release_is_a_no_op():
    _install_gpu_stub(check_decision="refused")  # never reached; step 1 is ungated
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(1, ["true"])
    runner.processes[1].finish()
    _drain(runner, 1)
    check(_RELEASE_CALLS == [], "an ungated stage releases nothing on exit")


# ---------------------------------------------------------------------------
# MODULE.YAML PARSING: missing / corrupt fails toward "no gate"
# ---------------------------------------------------------------------------

def test_missing_module_yaml_gates_nothing():
    import pathlib
    stage_runner.MODULE_YAML_PATH = pathlib.Path(
        tempfile.mkdtemp(prefix="gpu_gate_missing_")) / "absent.yaml"
    _FIXTURES.append(str(stage_runner.MODULE_YAML_PATH.parent))
    stages = stage_runner._load_gpu_stages()
    check(stages == {}, f"a missing module.yaml reads as no gated stages: {stages}")


def test_corrupt_module_yaml_gates_nothing():
    import pathlib
    fd, path = tempfile.mkstemp(prefix="gpu_gate_corrupt_", suffix=".yaml")
    _FIXTURES.append(path)
    with os.fdopen(fd, "w") as fh:
        fh.write("{{{not yaml: [")
    stage_runner.MODULE_YAML_PATH = pathlib.Path(path)
    stages = stage_runner._load_gpu_stages()
    check(stages == {}, f"corrupt YAML reads as no gated stages, not a crash: {stages}")


def test_gpu_stages_block_missing_or_wrong_shape_gates_nothing():
    import pathlib
    fd, path = tempfile.mkstemp(prefix="gpu_gate_noblock_", suffix=".yaml")
    _FIXTURES.append(path)
    with os.fdopen(fd, "w") as fh:
        fh.write("name: reef_point_seg\n")  # no gpu: key at all
    stage_runner.MODULE_YAML_PATH = pathlib.Path(path)
    check(stage_runner._load_gpu_stages() == {}, "a module.yaml with no gpu: block gates nothing")

    fd2, path2 = tempfile.mkstemp(prefix="gpu_gate_wrongshape_", suffix=".yaml")
    _FIXTURES.append(path2)
    with os.fdopen(fd2, "w") as fh:
        fh.write("gpu:\n  stages: not-a-mapping\n")  # wrong shape
    stage_runner.MODULE_YAML_PATH = pathlib.Path(path2)
    check(stage_runner._load_gpu_stages() == {},
          "a gpu.stages that is not a mapping gates nothing rather than raising")


# ---------------------------------------------------------------------------
# REAL MODULE.YAML: the corrected stage map, pinned so it cannot regress
# ---------------------------------------------------------------------------

def test_the_real_module_yaml_carries_the_corrected_stage_map():
    """Pins the map this task corrected (2026-09-17): the plan's first draft
    named only "4", "5", "step4test_route" and "annotate" as human-tier and
    never gated "step4test", "step4loop" or "editmasks", even though all
    three launch through the same call site as step4test_route (app.py's
    _launch_step4test_ui, line ~5065) and every one of them loads a SAM3
    model (TCRMPclip_combinedAnnotate/src/app.py and TCRMPclip_editMasks/
    src/app.py both call SAM3Engine(cfg) at boot). This test reads the real,
    live module.yaml, not a fixture, so a future edit that drops one of
    these keys fails here rather than silently ungating a GPU-holding stage.
    """
    import pathlib
    # The real file: REPO_DIR/module.yaml (github_repo/module.yaml), not a
    # fixture, so this test proves the shipped map is correct right now.
    stage_runner.MODULE_YAML_PATH = pathlib.Path(stage_runner.REPO_DIR) / "module.yaml"
    stages = stage_runner._load_gpu_stages()

    human_stages = {"4", "5", "step4test_route", "step4test", "step4loop",
                     "editmasks", "annotate"}
    vega_stages = {"6", "7", "8"}
    for key in human_stages:
        check(stages.get(key) == "human", f"{key!r} must be tier 'human' (got {stages.get(key)!r})")
    for key in vega_stages:
        check(stages.get(key) == "vega", f"{key!r} must be tier 'vega' (got {stages.get(key)!r})")
    for key in ("1", "2", "3"):
        check(key not in stages, f"{key!r} (frame extraction/recode/choose images) must stay ungated")
    check(set(stages.keys()) == human_stages | vega_stages,
          f"no extra or missing keys in the real gpu.stages map: {sorted(stages.keys())}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("GATE:")
    run(test_gated_stage_refused_never_spawns)
    run(test_gated_stage_refused_needs_pin_also_never_spawns)
    run(test_gated_stage_granted_spawns_and_claims_child_pid)
    run(test_gated_flask_stage_granted_spawns_and_claims_child_pid)
    print("UNGATED:")
    run(test_ungated_stage_never_calls_the_helper)
    run(test_empty_gpu_stages_gates_nothing)
    print("UNREGISTERED:")
    run(test_unregistered_helper_lets_the_stage_run)
    run(test_malformed_check_reply_fails_open_rather_than_crashing)
    print("RELEASE:")
    run(test_claim_is_released_when_the_process_exits)
    run(test_ungated_stage_release_is_a_no_op)
    print("MODULE.YAML PARSING:")
    run(test_missing_module_yaml_gates_nothing)
    run(test_corrupt_module_yaml_gates_nothing)
    run(test_gpu_stages_block_missing_or_wrong_shape_gates_nothing)
    print("REAL MODULE.YAML:")
    run(test_the_real_module_yaml_carries_the_corrected_stage_map)

    # Close every fake process's pipes so no background reader thread is
    # left blocked on one (harmless at interpreter exit either way, since
    # every log_threads entry is a daemon thread, but leaving nothing open
    # is the tidier proof that this suite spawned no real process and leaked
    # no real resource).
    for proc in _ALL_FAKE_PROCS:
        proc.finish()

    _restore_gpu()

    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    for p in leftovers:
        try:
            if os.path.isdir(p):
                import shutil
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except OSError:
            pass
    still_there = [p for p in _FIXTURES if os.path.exists(p)]
    if still_there:
        print(f"  FAIL fixture cleanup: {still_there}")
        _RESULTS.append(("fixture_cleanup", False, str(still_there)))

    failed = [n for n, ok, _ in _RESULTS if not ok]
    print()
    for name, ok, detail in _RESULTS:
        if not ok:
            print(f"FAILED {name}:\n{detail}")
    print("PASS" if not failed else f"{len(failed)} FAILED")
    sys.exit(1 if failed else 0)
