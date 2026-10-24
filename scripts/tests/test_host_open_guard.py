"""Host-open guard (2026-09-02 hotfix): no test may pop a file-manager window.

pipeline_orchestrator/host_open.open_in_file_manager() is the one spawn path
behind every "Open folder" control in the orchestrator and the coverage
matrix. Test fixtures used to reach the real xdg-open because launch-route
suites stubbed subprocess.run but not subprocess.Popen, and each such spawn
opened a file-manager window on the operator's desktop.

host_open.host_open_refusal() now refuses, before any spawn, when
  (a) VICARIUS_HOST_OPEN=0 (run_all_tests.sh exports it), or
  (b) the path sits under the system temp dir (only fixtures live there).
This suite pins both refusals, proves the enabled path goes through
subprocess.Popen with the detached GUI env (c), and finally drives the REAL
subprocess.Popen against a PATH-shim xdg-open (d) so the whole spawn path is
exercised end to end while only the shim ever executes.

Why (d) wraps gui_env(): gui_env() pins /usr/bin first on PATH by design (the
system xdg-open must beat conda's), and Popen resolves the executable through
the PATH of the env it is handed. On any host with /usr/bin/xdg-open a shim
placed on os.environ PATH can therefore never win. The test wraps gui_env()
to prepend the shim dir to gui_env()'s own output, asserts the shim is first
and is what "xdg-open" resolves to, and only then spawns.

subprocess.Popen is blocked at module level (records the call, then raises).
open_in_file_manager() swallows exceptions from Popen into (False, text), so
every check also asserts the recorded-reach list stays empty.

Run: env/bin/python scripts/tests/test_host_open_guard.py
     Safe with or without VICARIUS_HOST_OPEN set; it manages the var itself.
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

import atexit
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
REPO = os.path.dirname(SCRIPTS)            # github_repo: a real, non-temp directory
sys.path.insert(0, os.path.join(SCRIPTS, "pipeline_orchestrator"))

_REAL_POPEN = subprocess.Popen
_REACHED = []      # every call that hit the blocked Popen (must stay empty)
_SPAWNS = []       # calls recorded by the fake Popen in (c)


def _blocked_popen(*a, **k):
    argv = a[0] if a else k.get("args")
    _REACHED.append(argv)
    raise AssertionError(f"subprocess.Popen reached: argv={argv!r}")


# Module level: host_open references subprocess.Popen through the module, so
# this rebinding is what every spawn sees.
subprocess.Popen = _blocked_popen

import host_open  # noqa: E402

_pass = 0
_fail = 0


def check(c, m):
    global _pass, _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if c:
        _pass += 1
    else:
        _fail += 1


class _HostOpenEnv:
    """Set (or unset, value None) VICARIUS_HOST_OPEN for one block; restores on exit."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.saved = os.environ.get("VICARIUS_HOST_OPEN")
        if self.value is None:
            os.environ.pop("VICARIUS_HOST_OPEN", None)
        else:
            os.environ["VICARIUS_HOST_OPEN"] = self.value
        return self

    def __exit__(self, *exc):
        if self.saved is None:
            os.environ.pop("VICARIUS_HOST_OPEN", None)
        else:
            os.environ["VICARIUS_HOST_OPEN"] = self.saved
        return False


_TMP = tempfile.mkdtemp(prefix="host_open_guard_")
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))

check(os.path.isdir(REPO) and not host_open.host_open_refusal(REPO) == "refusing to open a temporary folder on the host desktop",
      f"REPO is a real non-temp directory: {REPO}")

# ── (a) VICARIUS_HOST_OPEN=0 refuses before any spawn ────────────────────
print("(a) VICARIUS_HOST_OPEN=0")
for value in ("0", " 0 "):
    with _HostOpenEnv(value):
        ok, err = host_open.open_in_file_manager(REPO)
        check(ok is False and "VICARIUS_HOST_OPEN=0" in err,
              f"VICARIUS_HOST_OPEN={value!r} -> (False, refusal): {ok} {err!r}")
check(not _REACHED, f"no Popen under VICARIUS_HOST_OPEN=0 (reached: {_REACHED})")

# ── (b) temp-dir paths are refused with the var unset ────────────────────
print("(b) temp-dir refusal with VICARIUS_HOST_OPEN unset")
sub = os.path.join(_TMP, "fixture", "run_X", "step4test_combinedAnnotate")
os.makedirs(sub)
with _HostOpenEnv(None):
    for p in (_TMP, sub, tempfile.gettempdir()):
        ok, err = host_open.open_in_file_manager(p)
        check(ok is False and "temporary" in err, f"temp path refused: {p} -> {ok} {err!r}")
    check(host_open.host_open_refusal(REPO) is None, "a non-temp path is not refused when the var is unset")
check(not _REACHED, f"no Popen for temp paths (reached: {_REACHED})")

# ── (c) enabled + non-temp path spawns through subprocess.Popen ──────────
print("(c) non-temp path spawns through subprocess.Popen (fake proc)")


class _FakeProc:
    def __init__(self, argv, **kw):
        _SPAWNS.append({"argv": argv, "kw": kw})
        self.returncode = None

    def communicate(self, timeout=None):
        raise subprocess.TimeoutExpired(cmd=["xdg-open"], timeout=timeout)


subprocess.Popen = _FakeProc
try:
    with _HostOpenEnv(None):
        ok, err = host_open.open_in_file_manager(REPO)
        check((ok, err) == (True, ""), f"still running after the timeout counts as success: {ok} {err!r}")
        check(len(_SPAWNS) == 1 and _SPAWNS[0]["argv"] == ["xdg-open", REPO],
              f"argv is ['xdg-open', REPO]: {[s['argv'] for s in _SPAWNS]}")
        kw = _SPAWNS[0]["kw"] if _SPAWNS else {}
        check(kw.get("start_new_session") is True, "spawn is detached (start_new_session=True)")
        spawn_env = kw.get("env") or {}
        check(isinstance(spawn_env, dict) and spawn_env.get("PATH", "").split(os.pathsep)[0] == "/usr/bin",
              f"spawn env is gui_env() (PATH pins /usr/bin first): {spawn_env.get('PATH', '')[:80]!r}")
    with _HostOpenEnv("1"):
        ok, err = host_open.open_in_file_manager(REPO)
        check(ok is True and len(_SPAWNS) == 2, f"VICARIUS_HOST_OPEN=1 leaves opening enabled: {ok} {err!r}")
finally:
    subprocess.Popen = _blocked_popen
check(not _REACHED, f"blocked Popen untouched through (c) (reached: {_REACHED})")

# ── (d) real Popen against a PATH-shim xdg-open ──────────────────────────
print("(d) PATH-shim xdg-open through the real subprocess.Popen")
shim_dir = os.path.join(_TMP, "shim_bin")
os.makedirs(shim_dir)
record = os.path.join(_TMP, "shim_argv.txt")
shim = os.path.join(shim_dir, "xdg-open")
with open(shim, "w") as f:
    f.write("#!/bin/sh\n"
            f"printf '%s\\n' \"$@\" > '{record}'\n"
            "exit 0\n")
os.chmod(shim, 0o755)

_real_gui_env = host_open.gui_env


def _shimmed_gui_env():
    env = _real_gui_env()
    env["PATH"] = shim_dir + os.pathsep + env["PATH"]
    return env


real_env = _real_gui_env()
check(real_env["PATH"].split(os.pathsep)[0] == "/usr/bin",
      "real gui_env() pins /usr/bin first (why the shim needs the wrapper)")
spawn_path = _shimmed_gui_env()["PATH"]
parts = spawn_path.split(os.pathsep)
resolved = shutil.which("xdg-open", path=spawn_path)
check(parts[0] == shim_dir, f"shim dir is first on the spawn PATH: {parts[:2]}")
check(parts[1:] == real_env["PATH"].split(os.pathsep),
      "the rest of the spawn PATH is gui_env()'s own output, untouched")
check(resolved == shim, f"'xdg-open' resolves to the shim on that PATH: {resolved}")

if parts[0] == shim_dir and resolved == shim and os.access(shim, os.X_OK):
    host_open.gui_env = _shimmed_gui_env
    subprocess.Popen = _REAL_POPEN
    try:
        with _HostOpenEnv(None):
            ok, err = host_open.open_in_file_manager(REPO)
    finally:
        subprocess.Popen = _blocked_popen
        host_open.gui_env = _real_gui_env
    check((ok, err) == (True, ""), f"shim exits 0 -> (True, ''): {ok} {err!r}")
    check(os.path.isfile(record), "the shim ran and recorded its argv")
    argv = open(record).read().splitlines() if os.path.isfile(record) else []
    check(argv == [REPO], f"the shim received exactly the folder path: {argv}")
else:
    check(False, "shim is not what 'xdg-open' resolves to; spawn skipped so nothing real can open")

check(host_open.gui_env is _real_gui_env and subprocess.Popen is _blocked_popen,
      "gui_env and the Popen block are restored")
check(not _REACHED, f"blocked Popen never reached across the suite (reached: {_REACHED})")

shutil.rmtree(_TMP, ignore_errors=True)
check(not os.path.exists(_TMP), "temp fixture removed")

print(f"\n{_pass} passed, {_fail} failed")
sys.exit(1 if _fail else 0)
