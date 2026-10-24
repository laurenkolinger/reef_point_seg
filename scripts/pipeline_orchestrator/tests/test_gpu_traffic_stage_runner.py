"""GPU traffic per-card assignment on the VEGA stage runner (docs/superpowers/
plans/2026-09-28-gpu-traffic.md, Task T5; spec docs/superpowers/specs/
2026-09-28-gpu-traffic-design.md, sections 3 and 7).

Covers what test_gpu_gate.py does not: the per-card fields on the answer
gpu_claim.check() returns (row/cards/env/device/notice), and what
stage_runner.py does with them before and after a stage's Popen:

  cards_wanted   Steps 4 and step4test_route (human tier) always ask for one
                 card ("one"); every other human-tier gated step asks for
                 every card its row allows. A vega-tier step (6, 7, 8) asks
                 from _vega_gpu_plan, which reads the stage's OWN requested
                 --device (from its cmd, or from env_extra for Step 6's
                 TCRMP_STEP6_DEVICE): one card names one card, two or more
                 names every allowed card, "cpu" or an unreadable device
                 default to one card. This is the fix for a real crash: a
                 fixed "every allowed card" for Step 6 once handed a plain
                 single-card or CPU panel request a two-card
                 VICARIUS_GPU_DEVICE under the "no split" preset, and
                 Ultralytics refuses AutoBatch (batch<1, the panel's own
                 default) on Multi-GPU.
  human tier     gets gpu_claim.launch_env(answer) verbatim
                 (CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER)
  vega tier      gets CUDA_DEVICE_ORDER and VICARIUS_GPU_DEVICE, and never
                 CUDA_VISIBLE_DEVICES (Ultralytics derives its own visible-
                 device list from --device / VICARIUS_GPU_DEVICE and would
                 overwrite it) -- except a "cpu" request, which gets neither:
                 the stage never touches a card.
  claim          the post-spawn claim carries cards=card_uuids(answer): the
                 cards the pre-spawn check actually found, not every
                 allowed card; never an empty list ([]) when the answer
                 named none -- None instead, since the claims registry
                 reads the two differently.
  no per-card fields  an "unregistered" decision, or a "granted" decision
                 that carries no cards/env/device (an older VICARIUS
                 build), changes nothing.
  "wait"         blocks a launch exactly like "refused" or "needs_pin", with
                 its own message (ADR 0008 / gpu_claims.py's own
                 DECISION_WAIT: only "granted" ever means proceed).
  allow-list     _GPU_STARTABLE_DECISIONS names the two decisions that ever
                 start a stage ("granted", "unregistered"); anything else,
                 including a decision word this file has never seen and an
                 answer with no "decision" key at all, blocks by default
                 with the answer's own message or GPU_NO_USABLE_ANSWER_
                 MESSAGE. This file used to check a block-list instead,
                 which would have started a stage on any word it had not
                 been taught to recognise.

No pytest. gpu_claim.check/claim/launch_env/card_uuids/device_string are
replaced at module level with local stand-ins implementing the shared
contract (docs/superpowers/plans/2026-09-28-gpu-traffic.md, "Shared
contract"), never the real network call, so this suite is independent of
whether gpu_claim.py has been updated with those functions yet.

Run: env/bin/python scripts/pipeline_orchestrator/tests/test_gpu_traffic_stage_runner.py

Standing rule for every orchestrator suite: subprocess.Popen is stubbed at
module level before stage_runner is imported, so nothing here can ever spawn
a real process (relevant right now: a real Coral Intel training is running
on this box and must never be touched by a test).
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

import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")

_RESULTS = []
_FIXTURES = []


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


# ---------------------------------------------------------------------------
# Popen stand-in, identical shape to test_gpu_gate.py's: real pipes so
# _capture_output's blocking readline() only reaches end-of-stream once a
# test calls finish(), the same two-phase lifecycle a real child has.
# ---------------------------------------------------------------------------
class _FakeProc:
    def __init__(self, pid):
        self.pid = pid
        out_r, self._out_w = os.pipe()
        err_r, self._err_w = os.pipe()
        self.stdout = os.fdopen(out_r, "r")
        self.stderr = os.fdopen(err_r, "r")
        self._exit_code = None

    def poll(self):
        return self._exit_code

    def finish(self, exit_code=0):
        if self._exit_code is not None:
            return
        self._exit_code = exit_code
        os.close(self._out_w)
        os.close(self._err_w)


_NEXT_FAKE_PID = [520000]
_ALL_FAKE_PROCS = []
_POPEN_CALLS = []


def _fake_popen(cmd, **kwargs):
    _NEXT_FAKE_PID[0] += 1
    proc = _FakeProc(_NEXT_FAKE_PID[0])
    _POPEN_CALLS.append((cmd, kwargs))
    _ALL_FAKE_PROCS.append(proc)
    return proc


subprocess.Popen = _fake_popen

import stage_runner  # noqa: E402
import gpu_claim  # noqa: E402


# ---------------------------------------------------------------------------
# gpu_claim stand-ins implementing the shared contract's answer shape and
# helper functions, recorded for assertions.
# ---------------------------------------------------------------------------
_CHECK_CALLS = []
_CLAIM_CALLS = []


def answer(decision, cards=None, device=None, message=None):
    """One gpu_claim.check()-shaped answer, per the shared contract.

    Parameters:
        decision: "granted", "refused", "needs_pin", "wait" or
            "unregistered".
        cards: list of {"index", "uuid", "bus"} dicts the row was granted;
            None or [] means the answer carries no per-card fields at all
            (the "unregistered", or an older-build "granted", case).
        device: the comma-joined nvidia-smi indices string, when the
            caller wants it distinct from what `cards` would derive (a
            granted decision always sets it from cards otherwise).
        message: the refusal text, for "refused" / "needs_pin" / "wait".
    """
    ans = {"decision": decision}
    if message:
        ans["message"] = message
    if cards:
        ans["cards"] = cards
        ans["env"] = {
            "CUDA_VISIBLE_DEVICES": ",".join(c["uuid"] for c in cards),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        ans["device"] = device if device is not None else ",".join(
            str(c["index"]) for c in cards)
    elif device is not None:
        ans["device"] = device
    return ans


def _install_gpu_stub(check_answer, claim_decision="granted"):
    """Replace gpu_claim.check/claim/launch_env/card_uuids/device_string
    with fixed-answer stand-ins (every call gets the same check_answer).

    Parameters:
        check_answer: the dict gpu_claim.check() returns every call (see
            answer()).
        claim_decision: the "decision" value gpu_claim.claim() returns.
    """
    _CHECK_CALLS.clear()
    _CLAIM_CALLS.clear()

    def _check(module, tier=None, by="LO", cards_wanted=None):
        _CHECK_CALLS.append({"module": module, "tier": tier, "by": by,
                              "cards_wanted": cards_wanted})
        return check_answer

    _install_common(_check, claim_decision)


def _install_gpu_stub_responsive(cards_for_wanted, claim_decision="granted"):
    """Replace gpu_claim.check with a stand-in whose granted cards depend on
    the cards_wanted it was asked, the way a real server under a permissive
    preset actually behaves (it grants exactly what was asked, not always
    every allowed card).

    Parameters:
        cards_for_wanted: callable(cards_wanted) -> list of card dicts.
        claim_decision: the "decision" value gpu_claim.claim() returns.
    """
    _CHECK_CALLS.clear()
    _CLAIM_CALLS.clear()

    def _check(module, tier=None, by="LO", cards_wanted=None):
        _CHECK_CALLS.append({"module": module, "tier": tier, "by": by,
                              "cards_wanted": cards_wanted})
        return answer("granted", cards=cards_for_wanted(cards_wanted))

    _install_common(_check, claim_decision)


def _install_common(check_fn, claim_decision):
    def _claim(module, tier=None, run_id="", pid=None, by="LO", bypass=False,
               reason="", cards=None):
        _CLAIM_CALLS.append({"module": module, "tier": tier, "run_id": run_id,
                              "pid": pid, "by": by, "cards": cards})
        result = {"decision": claim_decision}
        if claim_decision == "granted":
            result["claim_id"] = f"c-test-{pid}"
        return result

    def _launch_env(ans):
        env = (ans or {}).get("env")
        return dict(env) if isinstance(env, dict) else {}

    def _card_uuids(ans):
        cards = (ans or {}).get("cards")
        if not isinstance(cards, list):
            return []
        return [c.get("uuid") for c in cards if isinstance(c, dict) and c.get("uuid")]

    def _device_string(ans):
        device = (ans or {}).get("device")
        return device if isinstance(device, str) and device else None

    gpu_claim.check = check_fn
    gpu_claim.claim = _claim
    gpu_claim.launch_env = _launch_env
    gpu_claim.card_uuids = _card_uuids
    gpu_claim.device_string = _device_string


def _restore_gpu():
    import importlib
    try:
        importlib.reload(gpu_claim)
    except Exception as exc:  # pragma: no cover - best-effort cleanup only
        print(f"  (cleanup) could not reload gpu_claim: {exc}")


def _write_module_yaml(stages):
    fd, path = tempfile.mkstemp(prefix="gpu_traffic_sr_module_", suffix=".yaml")
    _FIXTURES.append(path)
    with os.fdopen(fd, "w") as fh:
        fh.write("gpu:\n  stages:\n")
        for key, tier in stages.items():
            fh.write(f'    "{key}": {tier}\n')
    return path


def _new_runner(stages):
    stage_runner.MODULE_YAML_PATH = __import__("pathlib").Path(_write_module_yaml(stages))
    return stage_runner.StageRunner()


def _drain(runner, step, timeout=2.0):
    thread = runner.log_threads.get(step)
    if thread is not None:
        thread.join(timeout)


CARD_A = {"index": 0, "uuid": "GPU-A", "bus": "0000:41:00.0"}
CARD_B = {"index": 1, "uuid": "GPU-B", "bus": "0000:a1:00.0"}


# ---------------------------------------------------------------------------
# cards_wanted, human tier: a fixed set of runner keys always asks for one
# card; every other human-tier gated step asks for every allowed card.
# ---------------------------------------------------------------------------

def test_human_tier_cards_wanted_one_for_named_steps():
    for step in ("4", "step4test_route"):
        _install_gpu_stub(answer("granted", cards=[CARD_A]))
        runner = _new_runner({step: "human"})
        runner._gpu_refusal(step)
        check(_CHECK_CALLS[-1]["cards_wanted"] == "one",
              f"step {step!r} asks cards_wanted='one': {_CHECK_CALLS[-1]}")


def test_human_tier_cards_wanted_all_for_other_steps():
    for step in ("5", "step4test", "step4loop", "editmasks", "annotate"):
        _install_gpu_stub(answer("granted", cards=[CARD_A]))
        runner = _new_runner({step: "human"})
        runner._gpu_refusal(step)
        check(_CHECK_CALLS[-1]["cards_wanted"] is None,
              f"step {step!r} asks for every allowed card: {_CHECK_CALLS[-1]}")


# ---------------------------------------------------------------------------
# cards_wanted, vega tier: derived from the stage's OWN requested --device
# (stage_runner._requested_device / _vega_gpu_plan), never a fixed step-name
# set. This is the fix for the Step 6 AutoBatch/Multi-GPU crash.
# ---------------------------------------------------------------------------

def test_requested_device_reads_cmd_then_env_extra():
    RD = stage_runner._requested_device
    check(RD("7", ["--device", "0"], None) == "0",
          "a --device pair in cmd is read for any vega step")
    check(RD("8", ["x", "--device", "0,1", "y"], None) == "0,1",
          "--device is found anywhere in cmd, not only at a fixed index")
    check(RD("8", ["--device"], None) is None,
          "a --device flag with nothing after it is not misread")
    check(RD("6", [], {"TCRMP_STEP6_DEVICE": "cpu"}) == "cpu",
          "Step 6's device travels as TCRMP_STEP6_DEVICE in env_extra")
    check(RD("6", [], {}) is None,
          "Step 6 with no TCRMP_STEP6_DEVICE at all is unreadable")
    check(RD("6", [], None) is None,
          "Step 6 with env_extra=None is unreadable, not a crash")
    check(RD("7", [], {"TCRMP_STEP6_DEVICE": "0"}) is None,
          "the env-var fallback is scoped to Step 6 only, never Step 7 or 8")


def test_vega_gpu_plan_single_card():
    for step, cmd, env_extra in (
        ("7", ["--device", "0"], None),
        ("8", ["--device", "1"], None),
        ("6", [], {"TCRMP_STEP6_DEVICE": "0"}),
    ):
        cards_wanted, apply_override = stage_runner._vega_gpu_plan(step, cmd, env_extra)
        check(cards_wanted == "one" and apply_override is True,
              f"step {step!r} device from {cmd or env_extra!r}: single card asks "
              f"cards_wanted='one' and applies the override "
              f"(got {(cards_wanted, apply_override)})")


def test_vega_gpu_plan_two_or_more_cards():
    for step, cmd, env_extra in (
        ("7", ["--device", "0,1"], None),
        ("6", [], {"TCRMP_STEP6_DEVICE": "0,1"}),
    ):
        cards_wanted, apply_override = stage_runner._vega_gpu_plan(step, cmd, env_extra)
        check(cards_wanted is None and apply_override is True,
              f"step {step!r} device from {cmd or env_extra!r}: two or more cards "
              f"asks cards_wanted=None (every allowed card) and applies the "
              f"override (got {(cards_wanted, apply_override)})")


def test_vega_gpu_plan_cpu_asks_one_card_but_skips_the_override():
    for step, cmd, env_extra in (
        ("7", ["--device", "cpu"], None),
        ("8", ["--device", "CPU"], None),
        ("6", [], {"TCRMP_STEP6_DEVICE": " cpu "}),
    ):
        cards_wanted, apply_override = stage_runner._vega_gpu_plan(step, cmd, env_extra)
        check(cards_wanted == "one" and apply_override is False,
              f"step {step!r} device from {cmd or env_extra!r}: cpu asks for one "
              f"card (a minimal, harmless ask) but never applies the override "
              f"(got {(cards_wanted, apply_override)})")


def test_vega_gpu_plan_unreadable_device_defaults_to_one_card():
    # The bug this fixes: Step 6 with no device information at all (the
    # panel left it blank; run_step6.sh's own default is a single card) must
    # ask for one card, not every allowed card, so a "no split" preset never
    # hands it a two-card VICARIUS_GPU_DEVICE it never asked for.
    for step, cmd, env_extra in (("6", [], {}), ("6", [], None),
                                  ("7", [], None), ("8", [], {})):
        cards_wanted, apply_override = stage_runner._vega_gpu_plan(step, cmd, env_extra)
        check(cards_wanted == "one" and apply_override is True,
              f"step {step!r} with no readable device: defaults to a single-card "
              f"ask (got {(cards_wanted, apply_override)})")


def test_gpu_refusal_wires_vega_plan_through_cards_wanted():
    """Integration: _gpu_refusal's cards_wanted for a vega step comes from
    the cmd/env_extra it was actually called with, not a fixed step set."""
    cases = [
        ("7", ["--device", "0"], None, "one"),
        ("8", ["--device", "0,1"], None, None),
        ("6", [], {"TCRMP_STEP6_DEVICE": "cpu"}, "one"),
        ("6", [], {}, "one"),
    ]
    for step, cmd, env_extra, expected in cases:
        _install_gpu_stub(answer("granted", cards=[CARD_A, CARD_B]))
        runner = _new_runner({step: "vega"})
        runner._gpu_refusal(step, cmd, env_extra)
        check(_CHECK_CALLS[-1]["cards_wanted"] == expected,
              f"step {step!r} cmd={cmd!r} env_extra={env_extra!r}: expected "
              f"cards_wanted={expected!r}, got {_CHECK_CALLS[-1]['cards_wanted']!r}")


# ---------------------------------------------------------------------------
# The crash this fixes, end to end: a single-card or CPU Step 6 request
# never receives a two-card override, even when the row grants both cards
# (a permissive "no split" preset); only an explicit two-card request does.
# ---------------------------------------------------------------------------

def test_step6_single_card_request_never_widens_to_two_cards():
    # A real server under "no split" grants exactly what cards_wanted asked
    # for: one card when asked for one, both when asked for none (every
    # allowed card). Modelling that relationship here is what makes this
    # test able to catch the original bug (a fixed cards_wanted=None for
    # Step 6 would have made this stub hand back two cards regardless).
    _install_gpu_stub_responsive(
        lambda wanted: [CARD_A, CARD_B] if wanted is None else [CARD_A])
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"], env_extra={"TCRMP_STEP6_DEVICE": "0"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check(_CHECK_CALLS[-1]["cards_wanted"] == "one",
              f"a single-card panel request asks for one card: {_CHECK_CALLS[-1]}")
        check(env.get("VICARIUS_GPU_DEVICE") == "0",
              f"the child never receives a two-card override for a single-card "
              f"request, even though the row would grant both (got "
              f"{env.get('VICARIUS_GPU_DEVICE')!r})")
    finally:
        runner.processes[6].finish()
        _drain(runner, 6)


def test_step6_two_card_request_gets_the_two_card_override():
    _install_gpu_stub_responsive(
        lambda wanted: [CARD_A, CARD_B] if wanted is None else [CARD_A])
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"], env_extra={"TCRMP_STEP6_DEVICE": "0,1"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check(_CHECK_CALLS[-1]["cards_wanted"] is None,
              f"an explicit two-card panel request asks for every allowed card: "
              f"{_CHECK_CALLS[-1]}")
        check(env.get("VICARIUS_GPU_DEVICE") == "0,1",
              f"the child receives the two-card override it explicitly asked "
              f"for (got {env.get('VICARIUS_GPU_DEVICE')!r})")
    finally:
        runner.processes[6].finish()
        _drain(runner, 6)


def test_step6_cpu_request_gets_no_device_override_even_if_granted_both():
    # The row grants both cards (a permissive preset, or an operator's PIN
    # bypass), but the stage itself asked for "cpu": it must not be handed a
    # device index of any kind.
    _install_gpu_stub(answer("granted", cards=[CARD_A, CARD_B], device="0,1"))
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"], env_extra={"TCRMP_STEP6_DEVICE": "cpu"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check("VICARIUS_GPU_DEVICE" not in env,
              f"a cpu request gets no device override at all: {env}")
        check("CUDA_DEVICE_ORDER" not in env,
              f"a cpu request gets no CUDA_DEVICE_ORDER either: {env}")
    finally:
        runner.processes[6].finish()
        _drain(runner, 6)


# ---------------------------------------------------------------------------
# Per-tier environment, tested directly against _apply_gpu_env so no ambient
# os.environ entry can produce a false pass or a false CUDA_VISIBLE_DEVICES
# absence.
# ---------------------------------------------------------------------------

def test_human_tier_gets_launch_env_verbatim_after_env_extra():
    _install_gpu_stub(answer("granted", cards=[CARD_B]))
    runner = _new_runner({"4": "human"})
    runner._gpu_refusal("4")
    env = {"FOO": "bar", "CUDA_VISIBLE_DEVICES": "stale-guess"}
    runner._apply_gpu_env("4", env)
    check(env["CUDA_VISIBLE_DEVICES"] == "GPU-B",
          f"the GPU answer's env overrides a caller's own guess: {env}")
    check(env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID", f"CUDA_DEVICE_ORDER is set: {env}")
    check(env["FOO"] == "bar", "env_extra's unrelated entries survive")


def test_vega_tier_gets_device_and_order_never_cvd():
    _install_gpu_stub(answer("granted", cards=[CARD_A], device="0"))
    runner = _new_runner({"7": "vega"})
    runner._gpu_refusal("7", ["--device", "0"])
    env = {"FOO": "bar"}
    runner._apply_gpu_env("7", env)
    check(env["VICARIUS_GPU_DEVICE"] == "0", f"VICARIUS_GPU_DEVICE is set: {env}")
    check(env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID", f"CUDA_DEVICE_ORDER is set: {env}")
    check("CUDA_VISIBLE_DEVICES" not in env,
          f"vega tier never sets CUDA_VISIBLE_DEVICES, even though the answer "
          f"carried an env with one (Ultralytics would overwrite it anyway): {env}")
    check(env["FOO"] == "bar", "env_extra's unrelated entries survive")


def test_vega_tier_ddp_device_string_passes_through():
    _install_gpu_stub(answer("granted", cards=[CARD_A, CARD_B], device="0,1"))
    runner = _new_runner({"6": "vega"})
    runner._gpu_refusal("6", [], {"TCRMP_STEP6_DEVICE": "0,1"})
    env = {}
    runner._apply_gpu_env("6", env)
    check(env["VICARIUS_GPU_DEVICE"] == "0,1", f"a two-card device string passes through: {env}")


def test_unregistered_answer_changes_nothing():
    _install_gpu_stub({"decision": "unregistered", "reason": "5090 down"},
                       claim_decision="unregistered")
    for step, tier, cmd, env_extra in (("4", "human", None, None),
                                        ("7", "vega", ["--device", "0"], None)):
        runner = _new_runner({step: tier})
        runner._gpu_refusal(step, cmd, env_extra)
        env = {"FOO": "bar"}
        runner._apply_gpu_env(step, env)
        check(env == {"FOO": "bar"},
              f"an unregistered answer changes nothing for tier {tier!r}: {env}")


def test_granted_answer_missing_percard_fields_changes_nothing():
    # An older VICARIUS build: "granted" with none of cards/env/device.
    _install_gpu_stub({"decision": "granted"})
    for step, tier, cmd, env_extra in (("4", "human", None, None),
                                        ("7", "vega", ["--device", "0"], None)):
        runner = _new_runner({step: tier})
        refusal = runner._gpu_refusal(step, cmd, env_extra)
        check(refusal is None, f"a granted decision never refuses: {refusal}")
        env = {"FOO": "bar"}
        runner._apply_gpu_env(step, env)
        check(env == {"FOO": "bar"},
              f"a granted answer with no per-card fields changes nothing "
              f"for tier {tier!r}: {env}")


def test_ungated_step_apply_gpu_env_is_a_no_op():
    _install_gpu_stub(answer("granted", cards=[CARD_A]))
    runner = _new_runner({"6": "vega"})  # step 1 deliberately absent
    env = {"FOO": "bar"}
    runner._apply_gpu_env(1, env)
    check(env == {"FOO": "bar"}, "an ungated step's env is left untouched")


# ---------------------------------------------------------------------------
# "wait" blocks a launch exactly like "refused" or "needs_pin" (ADR 0008).
# ---------------------------------------------------------------------------

def test_wait_decision_blocks_like_refused():
    _install_gpu_stub(answer(
        "wait", message="GPU right of way: wait, a manual edit window may free the card soon."))
    for step, tier, cmd in (("6", "vega", ["true"]), ("4", "human", ["true"])):
        runner = _new_runner({step: tier})
        before = len(_POPEN_CALLS)
        result = runner.run_cli_stage(step, cmd)
        check(result == {"error": "GPU right of way: wait, a manual edit window may "
                                   "free the card soon.", "busy": True},
              f"a wait decision refuses with its own message for tier {tier!r}: {result}")
        check(len(_POPEN_CALLS) == before, f"a wait decision never spawns ({tier})")
        check(_CLAIM_CALLS == [], f"a wait decision never claims ({tier})")


# ---------------------------------------------------------------------------
# _GPU_STARTABLE_DECISIONS is an allow-list, never a block-list: a decision
# word this file has never seen, and an answer with no "decision" key at
# all, both refuse by default rather than starting the stage.
# ---------------------------------------------------------------------------

def test_unknown_decision_word_blocks_by_default():
    # A future right-of-way decision this file predates: no entry in either
    # _GPU_STARTABLE_DECISIONS or the old block-list vocabulary. An
    # allow-list refuses it; a block-list would have started the stage.
    _install_gpu_stub({"decision": "quarantined",
                        "message": "GPU right of way: quarantined pending review."})
    runner = _new_runner({"6": "vega"})
    before = len(_POPEN_CALLS)
    result = runner.run_cli_stage(6, ["true"])
    check(result == {"error": "GPU right of way: quarantined pending review.", "busy": True},
          f"an unrecognised decision word refuses with its own message: {result}")
    check(len(_POPEN_CALLS) == before, "an unrecognised decision word never spawns")
    check(_CLAIM_CALLS == [], "an unrecognised decision word never claims")


def test_missing_decision_key_blocks_with_the_fallback_message():
    # A malformed reply: no "decision" key at all. decision.get("decision")
    # reads None, which must never appear in _GPU_STARTABLE_DECISIONS.
    _install_gpu_stub({"reason": "server returned an unexpected shape"})
    runner = _new_runner({"4": "human"})
    before = len(_POPEN_CALLS)
    result = runner.run_cli_stage("4", ["true"])
    check(result == {"error": "GPU right of way returned no usable answer.", "busy": True},
          f"a missing decision key refuses with the generic fallback message: {result}")
    check(len(_POPEN_CALLS) == before, "a missing decision key never spawns")
    check(_CLAIM_CALLS == [], "a missing decision key never claims")


def test_startable_decisions_is_an_allow_list_not_the_old_block_list():
    # Pins the allow-list itself: exactly granted and unregistered start a
    # stage; nothing else, however this file used to spell "blocking".
    check(stage_runner._GPU_STARTABLE_DECISIONS == frozenset({"granted", "unregistered"}),
          f"the startable set is exactly granted and unregistered "
          f"(got {stage_runner._GPU_STARTABLE_DECISIONS})")
    for word in ("refused", "needs_pin", "wait", "quarantined", "", None):
        check(word not in stage_runner._GPU_STARTABLE_DECISIONS,
              f"{word!r} is not in the startable set")


# ---------------------------------------------------------------------------
# The post-spawn claim never sends an empty card list: None or the list.
# ---------------------------------------------------------------------------

def test_claim_never_sends_an_empty_card_list():
    _install_gpu_stub({"decision": "granted"}, claim_decision="granted")  # no cards field
    runner = _new_runner({"4": "human"})
    result = runner.run_cli_stage("4", ["true"])
    try:
        check(_CLAIM_CALLS[0]["cards"] is None,
              f"an answer naming no cards sends cards=None, never []: {_CLAIM_CALLS}")
    finally:
        runner.processes["4"].finish()
        _drain(runner, "4")


# ---------------------------------------------------------------------------
# End to end through Popen: env_extra and the GPU answer both reach the
# child; the post-spawn claim carries the checked cards; the stashed answer
# is popped once claimed.
# ---------------------------------------------------------------------------

def test_human_tier_end_to_end_env_and_claim():
    _install_gpu_stub(answer("granted", cards=[CARD_B]), claim_decision="granted")
    runner = _new_runner({"4": "human"})
    result = runner.run_cli_stage("4", ["true"], env_extra={"FOO": "bar"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check(env.get("FOO") == "bar", "env_extra reaches the child")
        check(env.get("CUDA_VISIBLE_DEVICES") == "GPU-B",
              f"the checked card reaches the child's environment: {env}")
        check(env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER reaches the child")
        check(_CLAIM_CALLS[0]["cards"] == ["GPU-B"],
              f"the post-spawn claim carries the checked cards: {_CLAIM_CALLS}")
        check(_CLAIM_CALLS[0]["pid"] == result["pid"], "the claim carries the child's pid")
        check("4" not in runner._gpu_answers, "the stashed answer is popped once claimed")
    finally:
        runner.processes["4"].finish()
        _drain(runner, "4")


def test_vega_tier_end_to_end_env_and_claim():
    _install_gpu_stub(answer("granted", cards=[CARD_A, CARD_B], device="0,1"),
                       claim_decision="granted")
    runner = _new_runner({"6": "vega"})
    result = runner.run_cli_stage(6, ["true"], env_extra={"TCRMP_STEP6_DEVICE": "0,1",
                                                           "FOO": "bar"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check(env.get("FOO") == "bar", "env_extra reaches the child")
        check(_CHECK_CALLS[-1]["cards_wanted"] is None,
              f"an explicit two-card request asks for every allowed card: {_CHECK_CALLS[-1]}")
        check(env.get("VICARIUS_GPU_DEVICE") == "0,1", f"the device string reaches the child: {env}")
        check(env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "CUDA_DEVICE_ORDER reaches the child")
        check(_CLAIM_CALLS[0]["cards"] == ["GPU-A", "GPU-B"],
              f"the post-spawn claim carries both checked cards: {_CLAIM_CALLS}")
    finally:
        runner.processes[6].finish()
        _drain(runner, 6)


def test_flask_stage_gets_env_too():
    _install_gpu_stub(answer("granted", cards=[CARD_A]), claim_decision="granted")
    runner = _new_runner({"step4test": "human"})
    result = runner.run_flask_stage("step4test", ["true"], 5080, env_extra={"FOO": "bar"})
    try:
        cmd, kwargs = _POPEN_CALLS[-1]
        env = kwargs["env"]
        check(env.get("FOO") == "bar", "env_extra reaches a flask stage's child")
        check(env.get("CUDA_VISIBLE_DEVICES") == "GPU-A", f"the checked card reaches it: {env}")
    finally:
        runner.processes["step4test"].finish()
        _drain(runner, "step4test")


def test_refusal_never_reaches_apply_env_or_claim():
    _install_gpu_stub(answer("refused", message="GPU right of way: vega is busy for testing."))
    runner = _new_runner({"6": "vega"})
    before = len(_POPEN_CALLS)
    result = runner.run_cli_stage(6, ["true"])
    check(result == {"error": "GPU right of way: vega is busy for testing.", "busy": True},
          f"a refused check still returns the busy-result shape: {result}")
    check(len(_POPEN_CALLS) == before, "a refused check never spawns")
    check(_CLAIM_CALLS == [], "a refused check never claims")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("CARDS_WANTED (human tier):")
    run(test_human_tier_cards_wanted_one_for_named_steps)
    run(test_human_tier_cards_wanted_all_for_other_steps)
    print("CARDS_WANTED (vega tier, device-derived):")
    run(test_requested_device_reads_cmd_then_env_extra)
    run(test_vega_gpu_plan_single_card)
    run(test_vega_gpu_plan_two_or_more_cards)
    run(test_vega_gpu_plan_cpu_asks_one_card_but_skips_the_override)
    run(test_vega_gpu_plan_unreadable_device_defaults_to_one_card)
    run(test_gpu_refusal_wires_vega_plan_through_cards_wanted)
    print("STEP 6 CRASH REGRESSION:")
    run(test_step6_single_card_request_never_widens_to_two_cards)
    run(test_step6_two_card_request_gets_the_two_card_override)
    run(test_step6_cpu_request_gets_no_device_override_even_if_granted_both)
    print("PER-TIER ENV:")
    run(test_human_tier_gets_launch_env_verbatim_after_env_extra)
    run(test_vega_tier_gets_device_and_order_never_cvd)
    run(test_vega_tier_ddp_device_string_passes_through)
    run(test_unregistered_answer_changes_nothing)
    run(test_granted_answer_missing_percard_fields_changes_nothing)
    run(test_ungated_step_apply_gpu_env_is_a_no_op)
    print("WAIT BLOCKS LIKE REFUSED:")
    run(test_wait_decision_blocks_like_refused)
    print("ALLOW-LIST, NEVER A BLOCK-LIST:")
    run(test_unknown_decision_word_blocks_by_default)
    run(test_missing_decision_key_blocks_with_the_fallback_message)
    run(test_startable_decisions_is_an_allow_list_not_the_old_block_list)
    print("CLAIM CARDS, NEVER []:")
    run(test_claim_never_sends_an_empty_card_list)
    print("END TO END:")
    run(test_human_tier_end_to_end_env_and_claim)
    run(test_vega_tier_end_to_end_env_and_claim)
    run(test_flask_stage_gets_env_too)
    run(test_refusal_never_reaches_apply_env_or_claim)

    for proc in _ALL_FAKE_PROCS:
        proc.finish()

    _restore_gpu()

    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    for p in leftovers:
        try:
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
