"""
Stage runner: launch, monitor, and kill subprocesses for pipeline stages.
"""

import os
import re
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

# gpu_claim.check() decisions that mean "spawn". ADR 0008 / gpu_claims.py's
# own contract is that only "granted" ever means proceed; "unregistered" (port
# 5090 down) is the sole exception, so the caller proceeds and the GPU
# monitor shows it as unclaimed until the UI returns. This is deliberately an
# allow-list, not a list of the decisions that block: a word this file has
# never seen ("wait", a right-of-way decision added after this file was
# written, or any future one), and an answer with no "decision" key at all
# (a malformed reply), must both block by default, the same as "refused" and
# "needs_pin" (a human-tier stage asked to bypass a model-tier holder, which
# StageRunner has no admin-session concept to grant a PIN bypass for). A
# block-list would do the opposite: it would start the stage on any word it
# had not been told to recognise.
_GPU_STARTABLE_DECISIONS = frozenset({"granted", "unregistered"})

# The refusal shown when gpu_claim.check() answers a decision this file must
# not start on, and that answer carries no "message" of its own (an unknown
# decision word, or a reply with no "decision" key at all): a person or a
# monitor reading the refusal still learns that the gate, not a step's own
# process, is why nothing spawned.
GPU_NO_USABLE_ANSWER_MESSAGE = "GPU right of way returned no usable answer."

# gpu_claim.check(cards_wanted=...) value for a stage that always uses one
# card, however many cards its row allows on the current traffic preset
# (docs/superpowers/specs/2026-09-28-gpu-traffic-design.md, section 3).
GPU_CARDS_WANTED_ONE = "one"

# Runner keys, outside the vega tier, whose stage always uses exactly one
# card and so ask for GPU_CARDS_WANTED_ONE rather than every allowed card:
# Step 4 (placePoints, single-card OCR detection) and the headless routing
# pass that runs the same OCR detector (step4test_route). A vega-tier step
# (6, 7, 8) never reads this set: _vega_gpu_plan below derives its
# cards_wanted from the device the stage's own command line or env_extra
# asks for, because a fixed answer here once sent a plain single-card or
# CPU Step 6 run a two-card VICARIUS_GPU_DEVICE override it never asked for
# (Ultralytics refuses AutoBatch, batch<1, on Multi-GPU). A key absent from
# both this set and self.gpu_stages is never gated at all; a human-tier key
# in self.gpu_stages but absent from this set asks for every allowed card
# (cards_wanted=None).
GPU_CARDS_WANTED_ONE_STEPS = frozenset({"4", "step4test_route"})

# module.yaml's own env var name for the device the operator picked for Step
# 6, keyed by runner key: run_step6.sh reads it and forwards --device to
# train_wrapper.py only when it is set (blank means the wrapper's own
# default, a single card). Steps 7 and 8 (evaluate_run.py, run_inference.py)
# are invoked directly with an explicit "--device" pair in their own cmd
# list when the operator picked one, so they need no entry here:
# _requested_device reads their cmd directly.
_VEGA_DEVICE_ENV_VAR = {"6": "TCRMP_STEP6_DEVICE"}

# A device string naming two or more cards ("0,1", a DDP list): anything
# else this function is asked about is one card, "cpu", or unreadable.
_MULTI_CARD_DEVICE_RE = re.compile(r"^\d+(,\d+)+$")

# The device value Ultralytics (and app.py's own comment beside
# TCRMP_STEP6_DEVICE) accepts to mean "no card at all". Compared
# case-insensitively, since operators and stray whitespace vary.
_CPU_DEVICE_VALUE = "cpu"


def _requested_device(step, cmd, env_extra):
    """The --device value `step`'s own command or env_extra asks for.

    Parameters:
        step: the runner key about to launch.
        cmd: the argv list about to be spawned (a "--device" flag and its
            value, when the caller built one directly into the command,
            as app.py does for Steps 7 and 8).
        env_extra: the caller-supplied environment additions, or None (Step
            6's own device travels as TCRMP_STEP6_DEVICE, since run_step6.sh
            is a shell wrapper and decides for itself whether to forward
            --device to train_wrapper.py).

    Returns:
        The device string exactly as the caller wrote it ("0", "0,1",
        "cpu", ...), or None when neither source names one: a blank or
        absent value means "the stage's own script default", which this
        function does not guess at, and _vega_gpu_plan treats the same as
        a single card.
    """
    for i, arg in enumerate(cmd or []):
        if arg == "--device" and i + 1 < len(cmd):
            return str(cmd[i + 1])
    env_var = _VEGA_DEVICE_ENV_VAR.get(str(step))
    if env_var:
        value = (env_extra or {}).get(env_var)
        if value:
            return str(value)
    return None


def _vega_gpu_plan(step, cmd, env_extra):
    """cards_wanted and whether to apply VICARIUS_GPU_DEVICE, for a vega-tier
    stage, from the --device value it actually asks for (see
    _requested_device).

    Ultralytics refuses to auto-batch (batch<1, the panel's own default)
    across more than one card, so VICARIUS must never widen a single-card or
    CPU request into a two-card claim on its own: only a stage that itself
    names two or more cards is checked, and later overridden, for more than
    one.

    Parameters:
        step: the runner key about to launch.
        cmd: passed through to _requested_device.
        env_extra: passed through to _requested_device.

    Returns:
        (cards_wanted, apply_override).

        cards_wanted is GPU_CARDS_WANTED_ONE unless the stage's own device
        names two or more cards, in which case it is None (every allowed
        card, the traffic row's own default).

        apply_override is False only when the stage names "cpu" (any case,
        with surrounding whitespace): _apply_gpu_env must then leave
        VICARIUS_GPU_DEVICE (and CUDA_DEVICE_ORDER) unset even though the
        tier is gated, because a CPU run never touches a card. Every other
        case, including a device that could not be read at all, is True.
    """
    device = _requested_device(step, cmd, env_extra)
    if device is not None:
        normalized = device.strip().lower()
        if normalized == _CPU_DEVICE_VALUE:
            return GPU_CARDS_WANTED_ONE, False
        if _MULTI_CARD_DEVICE_RE.match(normalized):
            return None, True
    return GPU_CARDS_WANTED_ONE, True


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
        # step -> the full decision dict gpu_claim.check() returned before
        # spawn (GPU traffic's per-card fields: cards/env/device/notice),
        # read by _apply_gpu_env (before Popen) and _gpu_claim_after_spawn
        # (after Popen, to claim the same cards); popped once claimed.
        self._gpu_answers = {}
        # step -> whether _apply_gpu_env may set VICARIUS_GPU_DEVICE at all
        # (False only for a vega-tier stage whose own --device names "cpu");
        # popped alongside self._gpu_answers.
        self._gpu_apply_override = {}

        atexit.register(self.kill_all)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _gpu_refusal(self, step, cmd=None, env_extra=None):
        """Ask the GPU right-of-way gate before `step`'s process is spawned.

        Parameters:
            step: the runner key about to launch (an int step number or a
                string key such as "step4test").
            cmd: the argv list about to be spawned; read for a vega-tier
                step's own "--device" (see _vega_gpu_plan). Unused for a
                human-tier step.
            env_extra: the caller's own environment additions; read for a
                vega-tier step whose device travels as an env var (Step 6's
                TCRMP_STEP6_DEVICE) rather than a "--device" cmd flag.

        Returns:
            None when `step` is not in self.gpu_stages (never gated) or the
            gate answers "granted" or "unregistered" (port 5090 down; the
            caller proceeds): _GPU_STARTABLE_DECISIONS is checked as an
            allow-list, never a block-list, so a decision word this file has
            never seen ("wait", a right-of-way decision added later, or any
            other), and an answer with no "decision" key at all, both refuse
            by default rather than starting the stage on a word nobody
            taught this file to recognise. Otherwise a busy_result-shaped
            {"error": <right-of-way message>, "busy": True}, so every route
            that already handles a busy stage handles a right-of-way refusal
            with no change (spec: "Gates (where each tier asks)"). The
            message is the answer's own when it has one, else
            GPU_NO_USABLE_ANSWER_MESSAGE.

            Either way, when `step` is gated, the full decision dict is
            stashed in self._gpu_answers[step] (GPU traffic's per-card
            fields: cards/env/device/notice) for _apply_gpu_env to build the
            child's environment from and for _gpu_claim_after_spawn to claim
            the same cards with; self._gpu_apply_override[step] records
            whether _apply_gpu_env may act at all (see _vega_gpu_plan). A
            human-tier step whose stage only ever uses one card
            (GPU_CARDS_WANTED_ONE_STEPS) asks with cards_wanted="one"; every
            other human-tier gated step asks for every card its row allows.
            A vega-tier step's cards_wanted and apply_override both come
            from _vega_gpu_plan(step, cmd, env_extra).
        """
        tier = self.gpu_stages.get(str(step))
        if tier is None:
            return None
        if tier == "vega":
            cards_wanted, apply_override = _vega_gpu_plan(step, cmd, env_extra)
        else:
            cards_wanted = (GPU_CARDS_WANTED_ONE if str(step) in GPU_CARDS_WANTED_ONE_STEPS
                             else None)
            apply_override = True
        self._gpu_apply_override[step] = apply_override
        decision = gpu_claim.check(GPU_CLAIM_MODULE, tier=tier, by=GPU_CLAIM_BY,
                                    cards_wanted=cards_wanted)
        self._gpu_answers[step] = decision
        if decision.get("decision") not in _GPU_STARTABLE_DECISIONS:
            message = decision.get("message") or GPU_NO_USABLE_ANSWER_MESSAGE
            return {"error": message, "busy": True}
        return None

    def _apply_gpu_env(self, step, env):
        """Add `step`'s GPU environment variables, after env_extra is applied.

        Parameters:
            step: the runner key about to launch.
            env: the dict about to become the child's environment; mutated
                in place. Anything this sets overrides a same-named variable
                env_extra already put there, because the GPU right-of-way
                answer is authoritative over a caller-supplied guess at
                which card to use.

        Returns:
            None. A step outside self.gpu_stages (never gated), or one whose
            pre-spawn answer is missing (only possible if _gpu_refusal was
            never called first, which every caller in this file does before
            this method), leaves env unchanged.

            A tier "human" step gets gpu_claim.launch_env(answer)
            (CUDA_VISIBLE_DEVICES plus CUDA_DEVICE_ORDER). A tier "vega" step
            gets CUDA_DEVICE_ORDER and VICARIUS_GPU_DEVICE, never
            CUDA_VISIBLE_DEVICES: Ultralytics (train_wrapper.py,
            evaluate_run.py, run_inference.py, this tier's only callers)
            derives its own visible-device list from --device /
            VICARIUS_GPU_DEVICE and overwrites CUDA_VISIBLE_DEVICES from
            that, so setting the platform's copy here would only be
            overwritten a moment later. An "unregistered" decision, or an
            answer missing the per-card fields (an older VICARIUS build),
            carries no cards/env/device: gpu_claim.launch_env and
            gpu_claim.device_string both read such an answer as "nothing to
            add", so this method changes nothing for it, matching the
            shared contract ("an answer without these fields changes
            nothing"). Nor does a vega-tier step whose own device named
            "cpu" (self._gpu_apply_override[step] is False): the answer may
            carry a real card, but the stage that is about to run never
            touches one.
        """
        tier = self.gpu_stages.get(str(step))
        answer = self._gpu_answers.get(step)
        if tier is None or answer is None or not self._gpu_apply_override.get(step, True):
            return
        if tier == "human":
            env.update(gpu_claim.launch_env(answer))
        elif tier == "vega":
            device = gpu_claim.device_string(answer)
            if device:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["VICARIUS_GPU_DEVICE"] = device

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

            Pops self._gpu_answers[step] (set by _gpu_refusal) and sends its
            cards on the claim (cards=), so the claim is bound to exactly
            the cards the pre-spawn check found, not to every card its row
            allows. Sends None, never an empty list, when the answer named
            no cards at all (an unregistered decision, or one from an older
            VICARIUS build): the claims registry reads an empty list and
            "no restriction" as two different things, and this is the
            latter.
        """
        tier = self.gpu_stages.get(str(step))
        if tier is None:
            return
        answer = self._gpu_answers.pop(step, None)
        self._gpu_apply_override.pop(step, None)
        cards = gpu_claim.card_uuids(answer) if answer else []
        cards = cards or None  # never send an empty list; see docstring
        result = gpu_claim.claim(GPU_CLAIM_MODULE, tier=tier, run_id=str(step),
                                  pid=pid, by=GPU_CLAIM_BY, cards=cards)
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

        gpu_refusal = self._gpu_refusal(step, cmd, env_extra)
        if gpu_refusal is not None:
            return gpu_refusal

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        self._apply_gpu_env(step, env)

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

        gpu_refusal = self._gpu_refusal(step, cmd, env_extra)
        if gpu_refusal is not None:
            return gpu_refusal

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        self._apply_gpu_env(step, env)

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
