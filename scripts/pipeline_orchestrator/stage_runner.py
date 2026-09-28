"""
Stage runner: launch, monitor, and kill subprocesses for pipeline stages.
"""

import os
import sys
import signal
import atexit
import socket
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import yaml

import gpu_claim
from orchestrator_config import PYTHON_PATHS, ENTRY_POINTS, WORKING_DIRS, REPO_DIR


# Hold ~1 hour of heavy ultralytics-verbose output. Older lines roll off;
# the client offset logic below maps stale offsets into the surviving window
# so the stream never stops flowing.
_LOG_MAXLEN = 20000

# The wording every launch path answers when a step's process is still alive.
# Routes match on it (app.py narrows on "step4test is already running"), so
# the label-free form never changes.
STEP_BUSY_ERROR = "{label} is already running"

# GPU right-of-way (docs/superpowers/specs/2026-09-17-gpu-right-of-way-
# design.md). module.yaml lives at the repo root (github_repo/module.yaml);
# REPO_DIR is resolved once by orchestrator_config.py from pipeline.yaml's
# own location. TCRMP_MODULE_YAML_PATH overrides it for tests, the same
# override pattern orchestrator_config.py uses for its own config path.
MODULE_YAML_PATH = Path(
    os.environ.get("TCRMP_MODULE_YAML_PATH") or os.path.join(REPO_DIR, "module.yaml"))

# The module slug this runner claims GPU work under: module.yaml's own
# `name:` and the VICARIUS module registry slug for this module.
GPU_CLAIM_MODULE = "reef_point_seg"

# Every automated backend claim is signed with this VICARIUS-wide agent
# initials (platform rule: "Agent actions are signed LO"); the stage runner
# has no per-request operator identity to attach a claim to instead.
GPU_CLAIM_BY = "LO"

# gpu_claim.check() decisions that mean "do not spawn". A human-tier stage
# asked to bypass a model-tier holder ("needs_pin") is treated the same as an
# outright refusal here: StageRunner has no admin-session concept to grant a
# PIN bypass with, so the only two decisions that let a stage proceed are
# "granted" and "unregistered" (port 5090 down; the caller proceeds and the
# GPU monitor will show it as unclaimed until the UI returns).
_GPU_BLOCKING_DECISIONS = frozenset({"refused", "needs_pin"})


def _load_gpu_stages():
    """Read the `gpu.stages` map from module.yaml (MODULE_YAML_PATH).

    Returns:
        dict mapping a runner key (str, e.g. "4" or "step4test") to a GPU
        right-of-way tier name (str, e.g. "human" or "vega"). A runner key
        absent from this map is never gated: Step 1 (frame extraction),
        Step 2 (species recode) and Step 3 (choose images) hold no card and
        are correctly absent from module.yaml's gpu.stages block.

        Any problem reading or parsing module.yaml (missing file, corrupt
        YAML, a gpu.stages block that is not a mapping) returns {} rather
        than raising, with one warning on stderr: a module.yaml the
        orchestrator cannot read must not crash stage launches, so this
        fails toward "no stage is gated" rather than toward "every stage is
        refused". This is the opposite failure direction from the claims
        registry itself (gpu_claims.py), which fails toward "no claims", and
        that asymmetry is deliberate: an unreadable local config file should
        never be able to make VICARIUS refuse everything.
    """
    try:
        data = yaml.safe_load(MODULE_YAML_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        print(f"[stage_runner] warning: could not read {MODULE_YAML_PATH} for "
              f"gpu.stages ({exc}); no stage will be GPU-gated", file=sys.stderr)
        return {}
    stages = ((data or {}).get("gpu") or {}).get("stages")
    if not isinstance(stages, dict):
        return {}
    return {str(key): str(tier) for key, tier in stages.items()}


def busy_result(step, label=None, advice=None):
    """Result dict for a launch refused because `step` is still running.

    Parameters:
        step: the runner key whose process is alive (an int step number or
            a string key such as "step4test").
        label: what to call the step in the message; defaults to
            "Step <step>". A route passes the sidebar name so the reader
            recognises the tile.
        advice: an optional sentence appended after the refusal, telling the
            reader what to do.
    Returns:
        {"error": "<label> is already running[. <advice>]", "busy": True}.
        The "busy" flag lets a caller tell this refusal apart from a launch
        that started and then failed: nothing was spawned, so nothing about
        the step's recorded state may change.
    """
    message = STEP_BUSY_ERROR.format(label=label or f"Step {step}")
    if advice:
        message = f"{message}. {advice}"
    return {"error": message, "busy": True}


class StageRunner:
    def __init__(self):
        self.processes = {}       # step -> Popen
        self.logs = {}            # step -> deque of log lines (capped)
        self.log_totals = {}      # step -> monotonic count of all lines ever produced
        self.log_threads = {}     # step -> Thread
        self.ports = {}           # step -> port (for Flask stages)
        self._lock = threading.Lock()

        # GPU right-of-way: which runner keys touch a graphics card, and at
        # what tier, read once from module.yaml's gpu.stages block (see
        # module.yaml for the verified stage-by-stage rationale). A runner
        # key absent from self.gpu_stages is never gated and never claims.
        self.gpu_stages = _load_gpu_stages()
        self.gpu_claims = {}      # step -> claim_id, for gated stages only

        atexit.register(self.kill_all)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _gpu_refusal(self, step):
        """Ask the GPU right-of-way gate before `step`'s process is spawned.

        Parameters:
            step: the runner key about to launch (an int step number or a
                string key such as "step4test").

        Returns:
            None when `step` is not in self.gpu_stages (never gated) or the
            gate answers "granted" or "unregistered" (port 5090 down; the
            caller proceeds). Otherwise a busy_result-shaped
            {"error": <right-of-way message>, "busy": True}, so every route
            that already handles a busy stage handles a right-of-way refusal
            with no change (spec: "Gates (where each tier asks)").
        """
        tier = self.gpu_stages.get(str(step))
        if tier is None:
            return None
        decision = gpu_claim.check(GPU_CLAIM_MODULE, tier=tier, by=GPU_CLAIM_BY)
        if decision.get("decision") in _GPU_BLOCKING_DECISIONS:
            message = decision.get("message") or (
                f"GPU right of way: another run has the cards {step} needs.")
            return {"error": message, "busy": True}
        return None

    def _gpu_claim_after_spawn(self, step, pid):
        """Claim the card for `step`'s freshly spawned child, when gated.

        Called immediately after Popen returns, with the child's own pid, so
        the registry protects the process that actually holds the card
        rather than the orchestrator's own pid.

        Parameters:
            step: the runner key that just spawned.
            pid: the child process's pid.

        Returns:
            None always. A claim that comes back anything other than
            "granted" after the pre-spawn check already passed (a race with
            another claim, or 5090 going down between the check and the
            claim) is logged on stderr and left running: the process already
            exists, so refusing the claim after the fact cannot un-spawn it,
            and StageRunner has no policy left to enforce at this point.
        """
        tier = self.gpu_stages.get(str(step))
        if tier is None:
            return
        result = gpu_claim.claim(GPU_CLAIM_MODULE, tier=tier, run_id=str(step),
                                  pid=pid, by=GPU_CLAIM_BY)
        decision = result.get("decision")
        if decision == "granted":
            self.gpu_claims[step] = result.get("claim_id")
        elif decision != "unregistered":
            print(f"[stage_runner] warning: GPU claim for step {step} (pid {pid}) "
                  f"was not granted after spawn ({decision!r}); the process is "
                  f"already running and proceeds without a claim", file=sys.stderr)

    def _gpu_release(self, step):
        """Release `step`'s GPU claim once its process has exited.

        Parameters:
            step: the runner key whose process just finished (its
                stdout/stderr reader in _capture_output reached end of
                stream).

        Returns:
            None always. A step that was never gated, or whose claim never
            made it into self.gpu_claims (an unregistered helper, or a lost
            race in _gpu_claim_after_spawn), has nothing to release and this
            is a no-op.
        """
        claim_id = self.gpu_claims.pop(step, None)
        if claim_id is not None:
            gpu_claim.release(claim_id=claim_id)

    def _signal_handler(self, signum, frame):
        self.kill_all()
        sys.exit(0)

    def _capture_output(self, step, proc):
        """Background thread: read stdout/stderr line by line."""
        log = self.logs[step]
        for stream in [proc.stdout, proc.stderr]:
            if stream is None:
                continue
            try:
                for line in iter(stream.readline, ""):
                    if not line:
                        break
                    log.append(line.rstrip("\n"))
                    # Track total-ever-produced so the client can follow the
                    # stream even after old lines fall off the capped deque.
                    self.log_totals[step] = self.log_totals.get(step, 0) + 1
            except (ValueError, OSError):
                pass
        self._gpu_release(step)

    def _resolve_python(self, step):
        """Find the Python interpreter for a stage."""
        py = PYTHON_PATHS.get(step, "python3")
        if os.path.isfile(py):
            return py
        # Fallback
        return "python3"

    def run_cli_stage(self, step, cmd, cwd=None, env_extra=None):
        """Launch a CLI stage (makeAllPoints, chooseImages)."""
        with self._lock:
            if step in self.processes and self.processes[step].poll() is None:
                return busy_result(step)

        gpu_refusal = self._gpu_refusal(step)
        if gpu_refusal is not None:
            return gpu_refusal

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        self.logs[step] = deque(maxlen=_LOG_MAXLEN)
        self.log_totals[step] = 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or WORKING_DIRS.get(step),
            env=env,
            text=True,
            start_new_session=True,
        )
        self._gpu_claim_after_spawn(step, proc.pid)

        with self._lock:
            self.processes[step] = proc

        t = threading.Thread(target=self._capture_output, args=(step, proc), daemon=True)
        t.start()
        self.log_threads[step] = t

        return {"pid": proc.pid}

    def run_flask_stage(self, step, cmd, port, cwd=None, env_extra=None):
        """Launch a Flask sub-app stage."""
        with self._lock:
            if step in self.processes and self.processes[step].poll() is None:
                return busy_result(step)

        gpu_refusal = self._gpu_refusal(step)
        if gpu_refusal is not None:
            return gpu_refusal

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        self.logs[step] = deque(maxlen=_LOG_MAXLEN)
        self.log_totals[step] = 0
        self.ports[step] = port

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd or WORKING_DIRS.get(step),
            env=env,
            text=True,
            start_new_session=True,
        )
        self._gpu_claim_after_spawn(step, proc.pid)

        with self._lock:
            self.processes[step] = proc

        t = threading.Thread(target=self._capture_output, args=(step, proc), daemon=True)
        t.start()
        self.log_threads[step] = t

        return {"pid": proc.pid, "port": port}

    def poll_status(self, step):
        """Check if step is running, get exit code and log tail."""
        with self._lock:
            proc = self.processes.get(step)

        if proc is None:
            return {"running": False, "exit_code": None, "log_tail": []}

        exit_code = proc.poll()
        log = list(self.logs.get(step, []))
        # Return last 100 lines
        tail = log[-100:]

        return {
            "running": exit_code is None,
            "exit_code": exit_code,
            "log_tail": tail,
        }

    def get_log(self, step, offset=0):
        """
        Return lines since the given client offset plus bookkeeping fields.

        The deque is capped (see _LOG_MAXLEN), so the oldest lines are rolled
        off when the child is very chatty. We still track `total`, the
        monotonic count of all lines ever produced. The client's `offset` is
        interpreted against that total. If the client has fallen off the
        back of the surviving window, we hand them what's still in memory
        and record how many lines were dropped, so the UI can show a marker
        and resynchronize rather than freezing.

        Returns (lines, new_offset, dropped) where:
            lines       : the list of lines from the client's position forward
            new_offset  : total-ever-produced (the value the client should send next)
            dropped     : how many lines fell off the front since the client
                          last polled (non-zero means the client had to skip)
        """
        with self._lock:
            window = list(self.logs.get(step, []))
            total = self.log_totals.get(step, 0)
        window_size = len(window)
        window_start = max(0, total - window_size)
        dropped = 0
        if offset < window_start:
            # Client fell behind; skip forward to the oldest surviving line.
            dropped = window_start - offset
            start_in_window = 0
        else:
            start_in_window = offset - window_start
            if start_in_window > window_size:
                start_in_window = window_size
        lines = window[start_in_window:]
        return lines, total, dropped

    def health_check(self, port):
        """Check if a Flask app is responding on a port."""
        try:
            r = urlopen(f"http://localhost:{port}/", timeout=2)
            return r.status == 200
        except (URLError, OSError, Exception):
            return False

    def kill(self, step):
        """Kill the subprocess for a step."""
        with self._lock:
            proc = self.processes.get(step)
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

    def kill_all(self):
        """Kill all running subprocesses."""
        for step in list(self.processes.keys()):
            self.kill(step)

    def is_running(self, step):
        """Cheap liveness probe: no log-deque copy, just the process poll.
        The status endpoints reconcile cached phases against liveness on
        every UI poll (2.5s), so the common alive path must not pay
        poll_status's full-log copy."""
        with self._lock:
            proc = self.processes.get(step)
        return proc is not None and proc.poll() is None


def find_free_port(preferred):
    """Try preferred port first, then find any free one."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
