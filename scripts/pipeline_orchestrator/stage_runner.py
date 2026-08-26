"""
Stage runner — launch, monitor, and kill subprocesses for pipeline stages.
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
from urllib.request import urlopen
from urllib.error import URLError

from orchestrator_config import PYTHON_PATHS, ENTRY_POINTS, WORKING_DIRS, REPO_DIR


# Hold ~1 hour of heavy ultralytics-verbose output. Older lines roll off;
# the client offset logic below maps stale offsets into the surviving window
# so the stream never stops flowing.
_LOG_MAXLEN = 20000


class StageRunner:
    def __init__(self):
        self.processes = {}       # step -> Popen
        self.logs = {}            # step -> deque of log lines (capped)
        self.log_totals = {}      # step -> monotonic count of all lines ever produced
        self.log_threads = {}     # step -> Thread
        self.ports = {}           # step -> port (for Flask stages)
        self._lock = threading.Lock()

        atexit.register(self.kill_all)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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
                return {"error": f"Step {step} is already running"}

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
                return {"error": f"Step {step} is already running"}

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
        off when the child is very chatty. We still track `total` — the
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
