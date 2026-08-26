"""A cached ui_ready/review_ready phase must never outlive the process it
describes. Regression for 2026-08-13: the annotator died at SAM3 load after
route_status was set to ui_ready, and the UI polled a dead port for hours
while the API kept answering "Combined annotator ready".

No-pytest harness: run with github_repo/env/bin/python <this file>.
"""
import json
import os
import shutil
import sys
import subprocess
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)


# _kill_all_annotators() reaches OUTSIDE the runner: it pkills every
# combinedAnnotate/editMasks process on the box by entry-script match and runs
# `fuser -k` over ports 5080-5089. Neuter subprocess.run for the whole module
# (the same guard test_step4test_route_inflight.py uses) so this suite can
# never reach off its own process.
class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()

import app as appmod

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


client = appmod.create_app().test_client()
_orig_runner = appmod.runner


class _DeadRunner:
    def poll_status(self, key):
        return {"running": False, "exit_code": -9,
                "log_tail": ["boom line %d" % i for i in range(20)]}

    def is_running(self, key):
        return False

    def kill(self, key):
        pass


class _LiveRunner:
    def poll_status(self, key):
        return {"running": True, "exit_code": None, "log_tail": []}

    def is_running(self, key):
        return True

    def kill(self, key):
        pass


class _KeyedRunner:
    """Live for some runner keys, dead for the rest: models the real box where
    _kill_all_annotators kills step4test before an Edit Masks/Refine launch."""

    def __init__(self, live_keys):
        self.live_keys = set(live_keys)

    def is_running(self, key):
        return key in self.live_keys

    def poll_status(self, key):
        if key in self.live_keys:
            return {"running": True, "exit_code": None, "log_tail": []}
        return {"running": False, "exit_code": -15, "log_tail": ["killed"]}

    def kill(self, key):
        pass


# ui_ready + dead process -> error with exit code and log tail.
appmod.runner = _DeadRunner()
with appmod.route_lock:
    appmod.route_status.update(phase="ui_ready", ui_port=5080, error=None,
                               message="Combined annotator ready.")
r = client.get("/api/step/step4test/route_status").get_json()
check(r["phase"] == "error", "dead annotator flips ui_ready to error")
check("exited" in (r.get("error") or ""), "error names the process exit")
check(r.get("died") is True, "died marker set for the frontend")
check(r.get("ui_port") is None, "dead port is not advertised")
check(len(r.get("log_tail") or []) == 12, "last 12 log lines attached")

# The flip is persisted: a second poll reports error even if the reconcile
# were skipped (state written back under the lock).
r2 = client.get("/api/step/step4test/route_status").get_json()
check(r2["phase"] == "error", "flip is persisted (second poll still error)")

# ui_ready + live process -> untouched.
appmod.runner = _LiveRunner()
with appmod.route_lock:
    appmod.route_status.update(phase="ui_ready", ui_port=5080, error=None,
                               message="Combined annotator ready.",
                               log_tail=[])
r3 = client.get("/api/step/step4test/route_status").get_json()
check(r3["phase"] == "ui_ready", "live annotator stays ui_ready")
check(r3.get("ui_port") == 5080, "live port still advertised")

# Phases that do not claim a live UI are never reconciled (a routing pass has
# no annotator process yet; its own driver thread reports its errors).
appmod.runner = _DeadRunner()
with appmod.route_lock:
    appmod.route_status.update(phase="routing_ocr", ui_port=None, error=None)
r4 = client.get("/api/step/step4test/route_status").get_json()
check(r4["phase"] == "routing_ocr", "routing phases untouched by liveness check")

# sam3_status: review_ready with a dead step-5 sub-app -> error.
appmod.runner = _DeadRunner()
with appmod.sam3_lock:
    appmod.sam3_status.update(phase="review_ready", running=False, error=None)
r5 = client.get("/api/step/5/sam3_status").get_json()
check(r5["phase"] == "error", "dead step-5 sub-app flips review_ready to error")

# sam3_status idle is never reconciled (nothing was ever launched).
with appmod.sam3_lock:
    appmod.sam3_status.update(phase="idle", running=False, error=None)
r6 = client.get("/api/step/5/sam3_status").get_json()
check(r6["phase"] == "idle", "idle sam3 status untouched")

# A deliberate Stop must never read as a crash: stopping a ui_ready annotator
# rewrites the phase, so the next poll reports "stopped", not "process exited".
appmod.runner = _DeadRunner()
with appmod.route_lock:
    appmod.route_status.update(phase="ui_ready", ui_port=5080, error=None,
                               message="Combined annotator ready.")
client.post("/api/step/step4test/stop")
r7 = client.get("/api/step/step4test/route_status").get_json()
check(r7["phase"] == "stopped", "deliberate Stop reads as stopped, not crash")
check(not r7.get("died"), "died marker cleared by the Stop transition")

# A fresh launch phase clears a previous death marker.
appmod._route_set(phase="routing", error=None)
r8 = client.get("/api/step/step4test/route_status").get_json()
check(not r8.get("died"), "new launch lifecycle clears the died marker")

# OWNERSHIP: route_status is shared by step4test/step4loop/editmasks. An
# Edit Masks ui_ready must be verified against the editmasks process, not the
# step4test one that _kill_all_annotators just killed (2026-08-14 review
# defect: every Edit Masks launch false-alarmed the tile).
appmod.runner = _KeyedRunner(live_keys=["editmasks"])
appmod._route_set(phase="ui_ready", ui_port=5085, error=None,
                  runner_key="editmasks", runner_label="Edit Masks app",
                  message="Edit Masks ready.")
r9 = client.get("/api/step/step4test/route_status").get_json()
check(r9["phase"] == "ui_ready",
      "editmasks-owned ui_ready survives a dead step4test key")

# And when the OWNER dies, the error names the owner.
appmod.runner = _KeyedRunner(live_keys=[])
r10 = client.get("/api/step/step4test/route_status").get_json()
check(r10["phase"] == "error" and "Edit Masks app" in (r10.get("error") or ""),
      "dead owner reported under its own label")

# step4test's Stop must NOT claim an Edit Masks session stopped: it only
# kills its own keys, so a foreign-owned ui_ready stays untouched.
appmod.runner = _KeyedRunner(live_keys=["editmasks"])
appmod._route_set(phase="ui_ready", ui_port=5085, error=None,
                  runner_key="editmasks", runner_label="Edit Masks app",
                  message="Edit Masks ready.")
client.post("/api/step/step4test/stop")
r11 = client.get("/api/step/step4test/route_status").get_json()
check(r11["phase"] == "ui_ready",
      "step4test Stop leaves an editmasks-owned ui_ready alone")

# editmasks' own stop marks its owned ui_ready stopped, never a crash.
client.post("/api/step/editmasks/stop")
r12 = client.get("/api/step/step4test/route_status").get_json()
check(r12["phase"] == "stopped" and not r12.get("died"),
      "editmasks Stop reads as stopped for its own session")

# Step-5 deliberate stop: /api/step/5/stop clears review_ready so the kill
# never reads as a crash on the next sam3_status poll.
appmod.runner = _DeadRunner()
with appmod.sam3_lock:
    appmod.sam3_status.update(phase="review_ready", running=False, error=None)
client.post("/api/step/5/stop")
r13 = client.get("/api/step/5/sam3_status").get_json()
check(r13["phase"] == "stopped" and not r13.get("died"),
      "step-5 Stop reads as stopped, not crash")

# Recovery hygiene: a healthy relaunch clears stale error text and log tail.
appmod.runner = _LiveRunner()
appmod._route_set(phase="error", error="old crash", log_tail=["x"], died=True)
appmod._route_set(phase="ui_ready", ui_port=5080, runner_key="step4test",
                  runner_label="Combined annotator", message="ready")
r14 = client.get("/api/step/step4test/route_status").get_json()
check(r14.get("error") is None and not r14.get("log_tail"),
      "healthy relaunch clears stale error and log tail")

# The original incident window: the step-5 sub-app dying DURING model load
# (waiting_for_sub_app) must flip, not wait for the driver's 10-min timeout.
appmod.runner = _DeadRunner()
with appmod.sam3_lock:
    appmod.sam3_status.update(phase="waiting_for_sub_app", running=True,
                              error=None)
r15 = client.get("/api/step/5/sam3_status").get_json()
check(r15["phase"] == "error" and r15.get("died") is True,
      "step-5 death during model load (waiting_for_sub_app) flips to error")

# A routing pass whose DRIVER THREAD died must not stick at routing_ocr
# forever: a started-and-finished thread with a routing phase means a wedge.
_t = threading.Thread(target=lambda: None)
_t.start(); _t.join()
appmod.route_thread = _t
appmod._route_set(phase="routing_ocr", processed=3, total=10, error=None)
r16 = client.get("/api/step/step4test/route_status").get_json()
check(r16["phase"] == "error" and "driver thread" in (r16.get("error") or ""),
      "dead routing driver thread flips routing phase to error")
appmod.route_thread = None

# step_reset at or below 5 kills step 5, so its cached sam3 phase must read
# stopped, never a crash. Needs a loaded project (minimal launcher-style stub).
def _stub_project(tmp):
    stub = {"id": os.path.basename(tmp), "name": "t", "project_dir": tmp,
            "created_at": "2026-08-14T00:00:00", "updated_at": "2026-08-14T00:00:00",
            "current_step": 1, "steps": {}}
    with open(os.path.join(tmp, "project.json"), "w") as f:
        json.dump(stub, f)
    return tmp

_tmpA = _stub_project(tempfile.mkdtemp(prefix="liveness_A_"))
_tmpB = _stub_project(tempfile.mkdtemp(prefix="liveness_B_"))

appmod.runner = _DeadRunner()
ropen = client.post("/api/project/open", json={"project_dir": _tmpA}).get_json()
check(ropen.get("success") is True, "stub project A opens")
with appmod.sam3_lock:
    appmod.sam3_status.update(phase="review_ready", running=False, error=None)
client.post("/api/step/4/reset")
r17 = client.get("/api/step/5/sam3_status").get_json()
check(r17["phase"] == "stopped" and not r17.get("died"),
      "step reset at/below 5 reads as stopped, not crash")

# Switching projects must not carry project A's ui_ready into project B;
# reopening the SAME project must keep it.
appmod.runner = _LiveRunner()
appmod._route_set(phase="ui_ready", ui_port=5080, runner_key="step4test",
                  runner_label="Combined annotator", message="ready")
client.post("/api/project/open", json={"project_dir": _tmpA})
r18 = client.get("/api/step/step4test/route_status").get_json()
check(r18["phase"] == "ui_ready", "reopening the same project keeps ui_ready")
client.post("/api/project/open", json={"project_dir": _tmpB})
r19 = client.get("/api/step/step4test/route_status").get_json()
check(r19["phase"] == "idle", "switching projects resets stale ui_ready")

appmod.current_project = None
shutil.rmtree(_tmpA, ignore_errors=True)
shutil.rmtree(_tmpB, ignore_errors=True)

appmod.runner = _orig_runner
print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
