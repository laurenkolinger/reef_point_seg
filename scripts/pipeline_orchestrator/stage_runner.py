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


class StageRunner:
    def __init__(self):
        self.processes = {}       # step -> Popen
        self.logs = {}            # step -> deque of log lines
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

        self.logs[step] = deque(maxlen=5000)
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

        self.logs[step] = deque(maxlen=5000)
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
        """Get log lines from offset."""
        log = list(self.logs.get(step, []))
        return log[offset:]

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
