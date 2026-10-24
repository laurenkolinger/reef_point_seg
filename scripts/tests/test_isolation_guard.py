"""Module: scripts/tests/test_isolation_guard.py

Purpose: prove the live-store write guard in scripts/tests/_isolation.py fires.

Run: env/bin/python scripts/tests/test_isolation_guard.py

Every write probe aims at the real live store paths with a file name that does
not exist, so a guard that failed to fire would create a stray file; the suite
checks none appears. Reads of live files must keep working.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _isolation as ISO

_fail = 0


def check(cond, msg):
    """Print PASS/FAIL for one assertion and count failures."""
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def raises_violation(fn):
    """True when ``fn`` raises LiveStoreWriteError (and nothing else)."""
    try:
        fn()
    except ISO.LiveStoreWriteError:
        return True
    except BaseException:
        return False
    return False


def test_guard_on_the_real_live_paths():
    """Every kind of write to a real live path raises before touching disk."""
    registry, library, clip_index = ISO.live_store_paths()
    probe = "zz_isolation_probe_never_created"
    paths = {"registry": os.path.join(registry, probe),
             "library": os.path.join(library, probe),
             "clip index": os.path.join(clip_index, probe)}
    ISO.install_guard()
    try:
        for label, path in paths.items():
            check(raises_violation(lambda p=path: open(p, "w")), f"open(w) on the {label} is blocked")
            check(raises_violation(lambda p=path: open(p, "a")), f"open(a) on the {label} is blocked")
            check(raises_violation(lambda p=path: open(p, "r+")), f"open(r+) on the {label} is blocked")
            check(raises_violation(lambda p=path: os.open(p, os.O_WRONLY | os.O_CREAT)),
                  f"os.open(write) on the {label} is blocked")
            check(raises_violation(lambda p=path: os.replace(p + ".tmp", p)),
                  f"os.replace into the {label} is blocked")
            check(raises_violation(lambda p=path: os.remove(p)), f"os.remove in the {label} is blocked")
            check(raises_violation(lambda p=path: os.mkdir(p)), f"os.mkdir in the {label} is blocked")
        check(raises_violation(lambda: open(os.path.join(registry, "..", "_mask_registry", probe), "w")),
              "a '..' detour into the registry is blocked")
        manifest = os.path.join(registry, "manifest.csv")
        with open(manifest, "rb") as fh:
            check(len(fh.read(16)) > 0, "reading the live manifest still works")
        check(len(ISO.check_unchanged()) >= 22, "every blocked write is recorded as a violation")
    finally:
        ISO.uninstall_guard()
        ISO._state["violations"].clear()
    for label, path in paths.items():
        check(not os.path.exists(path), f"no stray probe file appeared in the {label}")


def test_rmtree_and_dir_fd_deletes_are_blocked():
    """shutil.rmtree on a live path, and deletes relative to an open live folder, raise."""
    import shutil
    registry = ISO.live_store_paths()[0]
    probe = "zz_isolation_probe_never_created"
    ISO.install_guard()
    try:
        check(raises_violation(lambda: shutil.rmtree(registry)), "shutil.rmtree of the live registry is blocked")
        check(raises_violation(lambda: shutil.rmtree(os.path.join(registry, probe))),
              "shutil.rmtree inside the live registry is blocked")
        fd = os.open(registry, os.O_RDONLY)
        try:
            check(raises_violation(lambda: os.unlink(probe, dir_fd=fd)), "os.unlink relative to an open live folder is blocked")
            check(raises_violation(lambda: os.rmdir(probe, dir_fd=fd)), "os.rmdir relative to an open live folder is blocked")
        finally:
            os.close(fd)
        check(os.path.isdir(registry) and os.path.isfile(os.path.join(registry, "manifest.csv")),
              "the live registry is intact")
        tmp = tempfile.mkdtemp(prefix="isolation_rm_")
        os.makedirs(os.path.join(tmp, "a", "b"))
        shutil.rmtree(tmp)
        check(not os.path.exists(tmp), "shutil.rmtree of a temp tree still works")
    finally:
        ISO.uninstall_guard()
        ISO._state["violations"].clear()


def test_guard_ignores_other_paths_and_is_not_an_exception():
    """Temp-dir writes pass, and the error cannot be swallowed by 'except Exception'."""
    tmp = tempfile.mkdtemp(prefix="isolation_guard_")
    live = os.path.join(tmp, "live")
    os.makedirs(live)
    ISO.install_guard([live])
    try:
        other = os.path.join(tmp, "other.txt")
        with open(other, "w") as fh:
            fh.write("ok")
        check(os.path.exists(other), "a write outside the live store succeeds")
        swallowed = False
        try:
            try:
                open(os.path.join(live, "x"), "w")
            except Exception:
                swallowed = True
        except ISO.LiveStoreWriteError:
            pass
        check(not swallowed, "a broad 'except Exception' does not swallow the guard error")
        check(isinstance(ISO.LiveStoreWriteError("x"), BaseException)
              and not isinstance(ISO.LiveStoreWriteError("x"), Exception),
              "LiveStoreWriteError is a BaseException, not an Exception")
    finally:
        ISO.uninstall_guard()
        ISO._state["violations"].clear()
        shutil.rmtree(tmp, ignore_errors=True)


def test_install_isolates_env_and_exits_nonzero_on_violation():
    """A child that installs, then writes to the live store, exits 97; a clean child exits 0."""
    registry = ISO.live_store_paths()[0]
    body = ("import sys; sys.path.insert(0, %r); import _isolation as I; d = I.install();"
            "import os; assert all(os.path.isdir(v) for v in d.values()), d;"
            "assert not any(v.startswith(%r) for v in d.values());" % (HERE, os.path.dirname(registry)))
    clean = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True)
    check(clean.returncode == 0, f"a clean child exits 0 (got {clean.returncode}: {clean.stderr[-200:]})")
    swallow = body + ("\ntry:\n open(%r, 'w')\nexcept BaseException:\n pass\n"
                      % os.path.join(registry, "zz_isolation_probe_never_created"))
    bad = subprocess.run([sys.executable, "-c", swallow], capture_output=True, text=True)
    check(bad.returncode == ISO.EXIT_CODE_VIOLATION,
          f"a child that wrote and swallowed the error still exits {ISO.EXIT_CODE_VIOLATION} (got {bad.returncode})")
    check("ISOLATION FAILURE" in bad.stderr, "the failure is announced on stderr")
    check(not os.path.exists(os.path.join(registry, "zz_isolation_probe_never_created")),
          "the child created no file in the live registry")


def test_install_keeps_a_board_provided_dir_and_replaces_a_live_one():
    """A non-live value set by the board is kept; a value pointing into a live store is replaced."""
    registry = ISO.live_store_paths()[0]
    tmp = tempfile.mkdtemp(prefix="isolation_keep_")
    body = ("import sys, os; sys.path.insert(0, %r); import _isolation as I; d = I.install();"
            "print(d['TCRMP_MASK_REGISTRY_DIR']); print(d['TCRMP_CLIP_INDEX_DIR'])" % HERE)
    env = dict(os.environ, TCRMP_MASK_REGISTRY_DIR=tmp, TCRMP_CLIP_INDEX_DIR=registry)
    out = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, env=env)
    lines = out.stdout.split()
    check(lines[:1] == [tmp], "a board-provided temp dir is kept")
    check(len(lines) == 2 and lines[1] != registry and not os.path.exists(lines[1]),
          "a value pointing at a live store is replaced, and the temp dir is removed at exit")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_guard_on_the_real_live_paths()
    test_rmtree_and_dir_fd_deletes_are_blocked()
    test_guard_ignores_other_paths_and_is_not_an_exception()
    test_install_isolates_env_and_exits_nonzero_on_violation()
    test_install_keeps_a_board_provided_dir_and_replaces_a_live_one()
    print("FAILED: %d" % _fail if _fail else "ALL PASS")
    sys.exit(1 if _fail else 0)
