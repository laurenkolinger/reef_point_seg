"""Module: scripts/tests/_isolation.py

Purpose: keep every test suite that can reach the shared label stores off the
live copies when the suite is run alone (outside scripts/run_all_tests.sh).

The three live stores:
  * the mask registry   <module>/inprocess/_mask_registry/
  * the expert library  <module>/inprocess/_expert_id_library/
  * the clip index      <repo>/supporting_data/clip_index/

``install()`` does two things, in this order, before the code under test is
imported:

  1. Points TCRMP_MASK_REGISTRY_DIR, TCRMP_EXPERT_LIBRARY_DIR and
     TCRMP_CLIP_INDEX_DIR at fresh temp directories (the same variables the
     board sets), removed when the process exits. A value the board already
     set is kept, unless it points inside a live store.
  2. Installs a write guard. Any attempt to open a file for writing, or to
     create, rename, replace, remove or touch anything, under a live store
     raises ``LiveStoreWriteError`` before the operation happens.

Why a guard on the file functions and not only a before/after hash: the hash
cannot say which line did it and cannot stop it, and a live annotator or the
UI may legitimately change the stores while a suite runs. The guard stops the
write at the call and names the path and the caller. A before/after
fingerprint of the live stores is kept as a second, independent check for
writes the patched functions cannot see (a C extension, a child process
started without the isolating variables). It only prints a note by default,
because a live annotator or another process may change the stores at the same
moment; set TCRMP_ISOLATION_STRICT_FINGERPRINT=1 to make it fail the run.

``LiveStoreWriteError`` derives from BaseException so a suite's broad
``except Exception`` cannot swallow it. Every violation is also recorded and
turns the process exit code into EXIT_CODE_VIOLATION.

Inputs: none (paths are derived from this file's location).
Outputs: environment variables, a patched ``builtins.open`` / ``io.open`` /
``os`` surface, an atexit hook.
"""
import atexit
import builtins
import hashlib
import io
import os
import shutil
import sys
import tempfile
import traceback

EXIT_CODE_VIOLATION = 97
ENV_REGISTRY = "TCRMP_MASK_REGISTRY_DIR"
ENV_LIBRARY = "TCRMP_EXPERT_LIBRARY_DIR"
ENV_CLIP_INDEX = "TCRMP_CLIP_INDEX_DIR"
ENV_VIOLATION_LOG = "TCRMP_ISOLATION_VIOLATION_LOG"
ENV_STRICT_FINGERPRINT = "TCRMP_ISOLATION_STRICT_FINGERPRINT"   # "1": a changed live file fails the run
_WRITE_MODE_CHARS = frozenset("wax+")
_WRITE_OS_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_FINGERPRINT_HASH_LIMIT = 4 * 1024 * 1024   # hash files up to this size

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
_REPO = os.path.dirname(_SCRIPTS)            # github_repo
_MODULE_ROOT = os.path.dirname(_REPO)        # reef_point_seg


class LiveStoreWriteError(BaseException):
    """A suite tried to write to a live label store."""


def live_store_paths():
    """Return the three live store directories (real paths, as strings)."""
    return [
        os.path.realpath(os.path.join(_MODULE_ROOT, "inprocess", "_mask_registry")),
        os.path.realpath(os.path.join(_MODULE_ROOT, "inprocess", "_expert_id_library")),
        os.path.realpath(os.path.join(_REPO, "supporting_data", "clip_index")),
    ]


def _inside(path, roots):
    """True when ``path`` resolves to one of ``roots`` or below it."""
    try:
        real = os.path.realpath(os.fspath(path))
    except (TypeError, ValueError):
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)


def _fingerprint(roots):
    """Map every file under ``roots`` to (size, mtime_ns, sha256 or None)."""
    prints = {}
    for root in roots:
        for base, _dirs, names in os.walk(root):
            for name in names:
                path = os.path.join(base, name)
                try:
                    st = os.stat(path)
                    digest = None
                    if st.st_size <= _FINGERPRINT_HASH_LIMIT:
                        with _REAL_OPEN(path, "rb") as fh:
                            digest = hashlib.sha256(fh.read()).hexdigest()
                    prints[path] = (st.st_size, st.st_mtime_ns, digest)
                except OSError as exc:
                    prints[path] = ("unreadable", str(exc), None)
    return prints


_REAL_OPEN = builtins.open
_state = {"violations": [], "installed": False, "originals": {}, "fingerprint": None,
          "roots": [], "tmp_dirs": []}


def _violation(action, path):
    """Record, log and raise for a blocked write."""
    caller = "".join(traceback.format_stack(limit=6)[:-2])
    msg = f"LIVE STORE WRITE BLOCKED: {action} {os.fspath(path)!r}\n{caller}"
    _state["violations"].append(msg)
    sys.stderr.write(msg + "\n")
    log = os.environ.get(ENV_VIOLATION_LOG)
    if log:
        with _REAL_OPEN(log, "a") as fh:
            fh.write(f"{os.getpid()}\t{action}\t{os.fspath(path)}\n")
    raise LiveStoreWriteError(msg)


def _is_path_arg(value):
    """True for str/bytes/PathLike (not an int file descriptor)."""
    return isinstance(value, (str, bytes, os.PathLike))


def _wrap_open(real):
    """Wrap open/io.open: a write mode on a live path raises."""
    def guarded(file, mode="r", *args, **kwargs):
        if _is_path_arg(file) and _WRITE_MODE_CHARS & set(str(mode)) \
                and _inside(file, _state["roots"]):
            _violation(f"open(mode={mode!r})", file)
        return real(file, mode, *args, **kwargs)
    return guarded


def _wrap_os_open(real):
    """Wrap os.open: write flags on a live path raise."""
    def guarded(path, flags, *args, **kwargs):
        if _is_path_arg(path) and (flags & _WRITE_OS_FLAGS) \
                and _inside(path, _state["roots"]):
            _violation(f"os.open(flags={flags})", path)
        return real(path, flags, *args, **kwargs)
    return guarded


def _resolve_against_fd(path, dir_fd):
    """Absolute form of ``path`` when it is relative to the open directory ``dir_fd``."""
    if dir_fd is None or not _is_path_arg(path) or os.path.isabs(os.fsdecode(path)):
        return path
    try:
        base = os.readlink("/proc/self/fd/%d" % dir_fd)
    except OSError:
        return path
    return os.path.join(base, os.fsdecode(path))


def _wrap_path_op(name, real, path_args=(0,)):
    """Wrap an os function whose path arguments (by index) must stay off live roots.

    Paths given relative to an open directory (``dir_fd``, ``src_dir_fd``,
    ``dst_dir_fd``) are resolved against that directory first, which is how
    ``shutil.rmtree`` deletes entries.
    """
    fd_kwargs = {0: ("dir_fd", "src_dir_fd"), 1: ("dst_dir_fd", "dir_fd")}

    def guarded(*args, **kwargs):
        for i in path_args:
            if i >= len(args) or not _is_path_arg(args[i]):
                continue
            fd = next((kwargs[k] for k in fd_kwargs.get(i, ("dir_fd",)) if kwargs.get(k) is not None), None)
            if _inside(_resolve_against_fd(args[i], fd), _state["roots"]):
                _violation(f"os.{name}", args[i])
        return real(*args, **kwargs)
    guarded.__name__ = name
    return guarded


def _wrap_rmtree(real):
    """Wrap shutil.rmtree: removing a live path (or a tree inside one) raises up front."""
    def guarded(path, *args, **kwargs):
        if _is_path_arg(path) and _inside(path, _state["roots"]):
            _violation("shutil.rmtree", path)
        return real(path, *args, **kwargs)
    return guarded


def _wrap_mkdir(real):
    """Wrap os.mkdir: creating a missing directory in a live store raises."""
    def guarded(path, *args, **kwargs):
        if _is_path_arg(path) and not os.path.exists(path) \
                and _inside(path, _state["roots"]):
            _violation("os.mkdir", path)
        return real(path, *args, **kwargs)
    return guarded


def install_guard(live_roots=None):
    """Install the write guard (idempotent). ``live_roots`` overrides the stores.

    Also takes the before-fingerprint used by the exit check.
    """
    if _state["installed"]:
        return
    roots = [os.path.realpath(r) for r in (live_roots or live_store_paths())]
    _state["roots"] = roots
    _state["fingerprint"] = _fingerprint([r for r in roots if os.path.isdir(r)])
    orig = _state["originals"]
    orig["builtins.open"], orig["io.open"] = builtins.open, io.open
    orig["os.open"] = os.open
    guarded_open = _wrap_open(builtins.open)
    builtins.open, io.open = guarded_open, guarded_open
    os.open = _wrap_os_open(os.open)
    for name, idx in (("replace", (0, 1)), ("rename", (0, 1)), ("remove", (0,)),
                      ("unlink", (0,)), ("rmdir", (0,)), ("utime", (0,)),
                      ("chmod", (0,)), ("truncate", (0,))):
        if hasattr(os, name):
            orig["os." + name] = getattr(os, name)
            setattr(os, name, _wrap_path_op(name, getattr(os, name), idx))
    orig["os.mkdir"] = os.mkdir
    os.mkdir = _wrap_mkdir(os.mkdir)
    orig["shutil.rmtree"] = shutil.rmtree
    shutil.rmtree = _wrap_rmtree(shutil.rmtree)
    _state["installed"] = True


def uninstall_guard():
    """Remove the guard (used by the guard's own test only)."""
    for key, real in _state["originals"].items():
        mod, attr = key.split(".", 1)
        setattr({"builtins": builtins, "io": io, "os": os, "shutil": shutil}[mod], attr, real)
    _state["originals"].clear()
    _state["installed"] = False
    _state["roots"] = []


def fingerprint_changes():
    """Return one line per live-store file that differs from the install-time fingerprint."""
    if _state["fingerprint"] is None:
        return []
    now = _fingerprint([r for r in _state["roots"] if os.path.isdir(r)])
    before = _state["fingerprint"]
    return [f"live store file changed during the run: {path}"
            for path in sorted(set(before) | set(now))
            if before.get(path) != now.get(path)]


def check_unchanged():
    """Return the recorded guard violations (each one is a blocked live write)."""
    return list(_state["violations"])


def _exit_check():
    """atexit hook: fail the process when the live stores were touched."""
    problems = check_unchanged()
    changed = fingerprint_changes()
    strict = os.environ.get(ENV_STRICT_FINGERPRINT) == "1"
    if changed and not problems:
        sys.stderr.write(
            "\nISOLATION NOTE: a live label store changed while this suite ran, but the "
            "guard saw no write from this process (a live annotator, the UI or another "
            "suite run at the same time can do this):\n")
        for line in changed:
            sys.stderr.write("  " + line + "\n")
    if problems or (changed and strict):
        sys.stderr.write("\nISOLATION FAILURE: this suite touched a live label store:\n")
        for p in problems + (changed if strict else []):
            sys.stderr.write("  " + p.splitlines()[0] + "\n")
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(EXIT_CODE_VIOLATION)


def _remove_tmp_dirs():
    """atexit hook: delete the temp stores this module created."""
    for path in _state["tmp_dirs"]:
        shutil.rmtree(path, ignore_errors=True)


def _isolate_env(var, prefix, roots):
    """Point ``var`` at a fresh temp dir unless it already names a non-live dir."""
    current = os.environ.get(var)
    if current and not _inside(current, roots):
        return current
    path = tempfile.mkdtemp(prefix=prefix)
    _state["tmp_dirs"].append(path)
    os.environ[var] = path
    return path


def install():
    """Isolate the three stores and install the guard. Call before importing app code.

    Returns a dict of the three isolating directories.
    """
    roots = live_store_paths()
    # Registered first so it runs last: the cleanup below must not look like a write.
    atexit.register(_exit_check)
    atexit.register(_remove_tmp_dirs)
    dirs = {
        ENV_REGISTRY: _isolate_env(ENV_REGISTRY, "mask_registry_iso_", roots),
        ENV_LIBRARY: _isolate_env(ENV_LIBRARY, "expert_library_iso_", roots),
        ENV_CLIP_INDEX: _isolate_env(ENV_CLIP_INDEX, "clip_index_iso_", roots),
    }
    install_guard(roots)
    return dirs


def _install_from_sitecustomize():
    """Guard-only install for the discovery sweep (no env changes)."""
    install_guard()
    atexit.register(_exit_check)
