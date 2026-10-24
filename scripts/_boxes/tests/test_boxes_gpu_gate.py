#!/usr/bin/env python
"""GPU right-of-way gate on the Bounding Boxes conversion job (_boxes/blueprint.py).

bbox_to_seg.py (POST /boxes/api/convert) loads a SAM3 model onto a card, the
same way every tier="human" reef_point_seg stage does, but its Popen sits
outside stage_runner.py's runner-key gate: this blueprint asks the platform's
GPU right-of-way / GPU traffic gate directly before it starts
(docs/superpowers/specs/2026-09-28-gpu-traffic-design.md, section 7). This
suite is the dedicated proof for that gate, added alongside it; the larger
test_boxes_routes.py suite stubs gpu_claim to a plain "granted, no per-card
fields" answer so its own, unrelated assertions about the launched command
are unaffected by the gate's addition.

No-pytest house harness, matching test_boxes_routes.py: subprocess.Popen is
replaced at module level before the blueprint is imported, so no test here
can start a real SAM3 conversion. gpu_claim.check/claim/launch_env/
card_uuids/device_string are replaced on the blueprint module's own
`gpu_claim` reference (never the shared module object), so this suite never
calls out to a real (or absent) VICARIUS on port 5090 and never writes a
claim into the live GPU claims registry.

Run with:
  env/bin/python scripts/_boxes/tests/test_boxes_gpu_gate.py
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
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_LAUNCHES = []
_IMPOSSIBLE_PID = 2 ** 22 + 9999


class _FakeProc:
    """Stands in for a launched conversion process; records cmd/cwd/env."""

    def __init__(self, cmd, cwd=None, env=None, **_kw):
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = dict(env or {})
        self.pid = _IMPOSSIBLE_PID
        self.returncode = None
        self.stdout = io.StringIO("")
        self.stderr = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -15


def _fake_popen(cmd, **kwargs):
    proc = _FakeProc(cmd, **kwargs)
    _LAUNCHES.append(proc)
    return proc


_REAL_POPEN = subprocess.Popen
subprocess.Popen = _fake_popen

from flask import Flask  # noqa: E402

from _boxes import blueprint as bpmod  # noqa: E402
from _boxes import ingest_coral_intel as ing  # noqa: E402
from _boxes.tests.test_ingest_coral_intel import build_source, write  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


# ---------------------------------------------------------------------------
# gpu_claim stand-ins: deterministic decisions, no network call, recorded so
# a test can assert exactly what was asked and claimed.
# ---------------------------------------------------------------------------
_CHECK_CALLS = []
_CLAIM_CALLS = []


class _StubGpuClaim:
    def __init__(self, check_decision=None, message="", cards=None, device=None,
                 claim_decision="granted", omit_decision_key=False):
        self.check_decision = check_decision
        self.message = message
        self.cards = cards or []
        self.device = device
        self.claim_decision = claim_decision
        # A malformed reply carries no "decision" key at all, distinct from
        # any string check_decision could name (including "" or None).
        self.omit_decision_key = omit_decision_key

    def check(self, module, tier=None, by="LO"):
        _CHECK_CALLS.append({"module": module, "tier": tier, "by": by})
        answer = {} if self.omit_decision_key else {"decision": self.check_decision}
        if self.message:
            answer["message"] = self.message
        if self.cards:
            answer["cards"] = self.cards
            answer["env"] = {
                "CUDA_VISIBLE_DEVICES": ",".join(c["uuid"] for c in self.cards),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        if self.device is not None:
            answer["device"] = self.device
        return answer

    def claim(self, module, tier=None, run_id="", pid=None, by="LO",
              bypass=False, reason="", cards=None):
        _CLAIM_CALLS.append({"module": module, "tier": tier, "run_id": run_id,
                              "pid": pid, "by": by, "cards": cards})
        result = {"decision": self.claim_decision}
        if self.claim_decision == "granted":
            result["claim_id"] = f"c-test-{pid}"
        return result

    @staticmethod
    def launch_env(answer):
        env = (answer or {}).get("env")
        return dict(env) if isinstance(env, dict) else {}

    @staticmethod
    def card_uuids(answer):
        cards = (answer or {}).get("cards")
        if not isinstance(cards, list):
            return []
        return [c.get("uuid") for c in cards if isinstance(c, dict) and c.get("uuid")]

    @staticmethod
    def device_string(answer):
        device = (answer or {}).get("device")
        return device if isinstance(device, str) and device else None


def _install_gpu_stub(**kwargs):
    _CHECK_CALLS.clear()
    _CLAIM_CALLS.clear()
    bpmod.gpu_claim = _StubGpuClaim(**kwargs)


# ---------------------------------------------------------------------------
# Fixture: one staged dataset, one Flask test client per test (so runner
# state and _LAUNCHES never leak between scenarios).
# ---------------------------------------------------------------------------

def _client(bbox_root, seg_root, scripts_dir, py, source_path=""):
    app = Flask("boxes_gpu_gate_test")
    app.register_blueprint(
        bpmod.make_blueprint(
            lambda: {"source_path": source_path, "bbox_root": bbox_root,
                     "seg_root": seg_root, "scripts_dir": scripts_dir, "python": py,
                     "inprocess_root": ""},
            log_fn=lambda m: None),
        url_prefix="/boxes")
    return app.test_client()


def main():
    global _fail
    src = tempfile.mkdtemp(prefix="boxes_gpu_gate_src_")
    bbox_root = tempfile.mkdtemp(prefix="boxes_gpu_gate_bbox_")
    seg_root = tempfile.mkdtemp(prefix="boxes_gpu_gate_seg_")
    scripts_dir = tempfile.mkdtemp(prefix="boxes_gpu_gate_scripts_")
    try:
        build_source(src)
        dataset = os.path.join(bbox_root, "coral_intel")
        ing.ingest(src, dataset, log=lambda m: None)
        write(os.path.join(scripts_dir, "_boxes", "bbox_to_seg.py"), "")
        py = os.path.join(scripts_dir, "python")
        write(py, "")

        # ---- refused: never spawns, never claims -------------------------
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="refused",
                          message="GPU right of way: card 0 is busy for testing.")
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"a refused GPU check is 409 (got {r.status_code})")
        check(r.get_json().get("error") == "GPU right of way: card 0 is busy for testing.",
              f"the refusal carries the gate's own message (got {r.get_json()})")
        check(not _LAUNCHES, "a refused GPU check never spawns bbox_to_seg.py")
        check(_CLAIM_CALLS == [], "claim() is never reached when check() refuses")
        check(len(_CHECK_CALLS) == 1 and _CHECK_CALLS[0]["module"] == "reef_point_seg"
              and _CHECK_CALLS[0]["tier"] == "human",
              f"the check names this module and the human tier (got {_CHECK_CALLS})")

        # ---- needs_pin: refused the same way as an outright refusal ------
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="needs_pin",
                          message="GPU right of way: a VOYAGER run holds every card.")
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"needs_pin is refused like a refusal (got {r.status_code})")
        check(not _LAUNCHES, "needs_pin never spawns bbox_to_seg.py")

        # ---- wait: refused the same way, per ADR 0008 (only "granted" ever
        #      means proceed; a manual edit window or a driver-started
        #      VOYAGER turn may free the card soon, but not yet) -----------
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="wait",
                          message="GPU right of way: wait, a manual edit window may free the card soon.")
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"wait is refused like a refusal (got {r.status_code})")
        check(r.get_json().get("error")
              == "GPU right of way: wait, a manual edit window may free the card soon.",
              f"the wait refusal carries the gate's own message (got {r.get_json()})")
        check(not _LAUNCHES, "wait never spawns bbox_to_seg.py")
        check(_CLAIM_CALLS == [], "claim() is never reached when check() answers wait")

        # ---- an unrecognised decision word blocks by default (allow-list,
        #      never a block-list): a future right-of-way decision this
        #      blueprint predates must not start the conversion job -------
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="quarantined",
                          message="GPU right of way: quarantined pending review.")
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"an unrecognised decision word is refused (got {r.status_code})")
        check(r.get_json().get("error") == "GPU right of way: quarantined pending review.",
              f"the refusal carries the gate's own message (got {r.get_json()})")
        check(not _LAUNCHES, "an unrecognised decision word never spawns bbox_to_seg.py")

        # ---- a malformed reply with no "decision" key at all blocks with
        #      the generic fallback message ---------------------------------
        del _LAUNCHES[:]
        _install_gpu_stub(omit_decision_key=True)
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"a missing decision key is refused (got {r.status_code})")
        check(r.get_json().get("error") == "GPU right of way returned no usable answer.",
              f"the refusal falls back to the generic message (got {r.get_json()})")
        check(not _LAUNCHES, "a missing decision key never spawns bbox_to_seg.py")

        # ---- granted with cards: spawns, env carries the answer, claims
        #      with the child pid and the checked cards -------------------
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="granted", claim_decision="granted",
                          cards=[{"index": 1, "uuid": "GPU-B", "bus": "b1"}])
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert",
                   json={"dataset": dataset, "limit": 5, "name": "smoke_gpu"})
        body = r.get_json()
        check(r.status_code == 200 and body.get("ok") is True,
              f"a granted GPU check lets the conversion start (got {r.status_code} {body})")
        check(len(_LAUNCHES) == 1, "exactly one process was spawned")
        proc = _LAUNCHES[0]
        check(proc.env.get("CUDA_VISIBLE_DEVICES") == "GPU-B",
              f"the child's environment carries the checked card (got {proc.env.get('CUDA_VISIBLE_DEVICES')!r})")
        check(proc.env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
              f"the child's environment carries CUDA_DEVICE_ORDER (got {proc.env.get('CUDA_DEVICE_ORDER')!r})")
        check(len(_CLAIM_CALLS) == 1 and _CLAIM_CALLS[0]["pid"] == proc.pid,
              f"the claim carries the CHILD's pid (got {_CLAIM_CALLS})")
        check(_CLAIM_CALLS[0]["cards"] == ["GPU-B"],
              f"the claim carries the cards the pre-spawn check found (got {_CLAIM_CALLS[0]['cards']})")
        check(_CLAIM_CALLS[0]["tier"] == "human", "the claim is made at the human tier")

        # ---- granted with no cards named: the claim sends None, never an
        #      empty list (the claims registry reads the two differently) --
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="granted", claim_decision="granted")  # no cards=
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert",
                   json={"dataset": dataset, "limit": 5, "name": "smoke_nocards"})
        check(r.status_code == 200, f"a granted check with no cards named still starts (got {r.status_code})")
        check(_CLAIM_CALLS[-1]["cards"] is None,
              f"an answer naming no cards sends cards=None, never [] (got {_CLAIM_CALLS[-1]['cards']!r})")

        # ---- unregistered: proceeds, env unaffected, no crash on claim ---
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="unregistered", claim_decision="unregistered")
        c = _client(bbox_root, seg_root, scripts_dir, py)
        r = c.post("/boxes/api/convert",
                   json={"dataset": dataset, "limit": 5, "name": "smoke_unreg"})
        body = r.get_json()
        check(r.status_code == 200 and body.get("ok") is True,
              f"an unregistered GPU check still lets the conversion start (got {r.status_code})")
        check("CUDA_VISIBLE_DEVICES" not in _LAUNCHES[0].env
              or _LAUNCHES[0].env.get("CUDA_VISIBLE_DEVICES") != "GPU-B",
              "an unregistered answer adds no GPU device restriction")

        # ---- busy guard still wins: the GPU gate is never even asked -----
        del _LAUNCHES[:]
        _install_gpu_stub(check_decision="refused")  # would refuse the gate if asked
        c = _client(bbox_root, seg_root, scripts_dir, py, source_path=src)
        r = c.post("/boxes/api/ingest", json={})
        check(r.status_code == 200, f"ingest starts, making the runner busy (got {r.status_code})")
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"convert while the runner is busy is 409 (got {r.status_code})")
        check(_CHECK_CALLS == [],
              f"the GPU gate is never asked when the runner is already busy: {_CHECK_CALLS}")

        print("PASS" if _fail == 0 else f"{_fail} FAILED")
    finally:
        subprocess.Popen = _REAL_POPEN
        for path in (src, bbox_root, seg_root, scripts_dir):
            shutil.rmtree(path, ignore_errors=True)
        leftover = [p for p in (src, bbox_root, seg_root, scripts_dir) if os.path.exists(p)]
        if leftover:
            print(f"  FAIL fixture cleanup: {leftover}")
            _fail += 1

    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
