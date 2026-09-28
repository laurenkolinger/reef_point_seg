"""
Host desktop spawns for the reef_point_seg orchestrator (2026-08-26).

Two helpers that used to live inline in app.py (added the same day for the
"I can not open folder" fix) and are now shared by the orchestrator's own
folder routes AND the native coverage-matrix blueprint, which receives
``open_in_file_manager`` as its ``open_folder_fn`` at registration (spec D8).
Keeping them in one module means every "Open folder" control on the host
goes through the same environment scrub and the same detached spawn.

    gui_env()                        -> dict, the environment for host GUI apps
    open_in_file_manager(path)       -> (ok: bool, err: str)

``subprocess`` is referenced through the module (``subprocess.Popen``) so a
test that stubs ``subprocess.Popen`` module-wide sees every spawn.
"""

import os
import subprocess
import tempfile


def gui_env():
    """Environment for spawning host desktop apps (xdg-open / the file manager).

    Mirrors vicarius_ui_os/runner.py gui_subprocess_env(): when the server was
    started from a VS Code snap shell the inherited env carries snap GTK/GIO
    paths that make a host GTK app die before it shows a window. Also puts
    /usr/bin first so xdg-open resolves the system gio, not conda's."""
    env = dict(os.environ)
    for key in [k for k in env if k.endswith("_VSCODE_SNAP_ORIG")]:
        env[key[: -len("_VSCODE_SNAP_ORIG")]] = env.pop(key)
    snap_root = env.get("SNAP")
    if snap_root:
        snap_name = env.get("SNAP_INSTANCE_NAME") or env.get("SNAP_NAME") or ""
        prefixes = [p for p in (
            snap_root, env.get("SNAP_USER_DATA"), env.get("SNAP_USER_COMMON"),
            env.get("SNAP_DATA"), env.get("SNAP_COMMON"),
            f"/snap/{snap_name}/" if snap_name else None) if p]
        for key in [k for k, v in env.items()
                    if k == "SNAP" or k.startswith("SNAP_")
                    or any(p in v for p in prefixes)]:
            env.pop(key, None)
    path = env.get("PATH", "")
    parts = [p for p in path.split(os.pathsep) if p and p != "/usr/bin"]
    env["PATH"] = os.pathsep.join(["/usr/bin"] + parts)
    return env


def host_open_refusal(path):
    """Return a refusal message when a host window must NOT be opened for
    ``path``, else None. Two guards, both independent of any test stubbing:
    VICARIUS_HOST_OPEN=0 (exported by run_all_tests.sh and any headless run)
    disables host opening entirely, and a path under the system temp dir is
    always refused because no real project, composite, or dataset lives
    there; only test fixtures do, and each such spawn used to open a real
    file-manager window on the operator's desktop."""
    if os.environ.get("VICARIUS_HOST_OPEN", "1").strip() == "0":
        return "host folder opening is disabled in this environment (VICARIUS_HOST_OPEN=0)"
    try:
        real = os.path.realpath(path or "")
        tmp_root = os.path.realpath(tempfile.gettempdir())
    except Exception:
        return "cannot resolve the folder path"
    if real == tmp_root or real.startswith(tmp_root + os.sep):
        return "refusing to open a temporary folder on the host desktop"
    return None


def open_in_file_manager(path):
    """Spawn ``xdg-open <path>`` detached. Returns (ok, error_text). A launch
    that is still running after 3 s counts as success (the file manager keeps
    the process alive); a fast non-zero exit surfaces its stderr so a failure
    is legible instead of vanishing into DEVNULL."""
    refusal = host_open_refusal(path)
    if refusal:
        return False, refusal
    try:
        proc = subprocess.Popen(
            ["xdg-open", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=gui_env(),
        )
    except Exception as exc:
        return False, str(exc)
    try:
        _out, err = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        return True, ""
    if proc.returncode not in (0, None):
        text = (err or b"").decode("utf-8", "replace").strip()
        return False, text[:400] or f"xdg-open exited with code {proc.returncode}"
    return True, ""
