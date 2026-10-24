"""project.json is written atomically and under an exclusive lock.

project.json carries a project's whole state (every step's status, config and
outputs, current_step, access_log, the training source). The orchestrator is a
threaded Flask server and the VICARIUS launcher writes the same file from
another process, so two students on one orchestrator, or one student clicking
twice, used to be able to interleave two writes through one fixed temporary
name and destroy the project.

This suite proves the contract in project_manager: a unique temporary file in
the same directory, flush and fsync, os.replace, an exclusive lock shared by
threads and by processes, a stale-temp sweep, and a read path whose errors name
the file instead of raising a bare JSONDecodeError.

It also proves what the contract does when the lock itself misbehaves, which is
what a project folder on a network share does: a filesystem that cannot lock
saves anyway and says so once, a failure while releasing the file lock never
leaves the in-process lock held, and a project.json save needs no write
permission on the lock file. update_project_file, the entry point a writer
outside the orchestrator uses, is proven against threads and against real child
processes.

Nothing here imports app.py and nothing reaches a launch route, so subprocess
stays real: the crash, the cross-process lock and the outside-writer checks need
genuine child processes. No child touches a port, a project outside its own temp
directory, or any running process.

Run with github_repo/env/bin/python.
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

import errno
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH_DIR = os.path.dirname(HERE)
sys.path.insert(0, ORCH_DIR)

import project_manager as pm

_fail = 0


def check(condition, message):
    """Print one PASS/FAIL line and count the failures."""
    global _fail
    print(("  PASS " if condition else "  FAIL ") + message)
    if not condition:
        _fail += 1


def skip(message):
    """Print one SKIP line for a check this environment cannot make."""
    print("  SKIP " + message)


# Concurrency sizes: enough overlap to interleave, small enough to stay quick.
WRITER_THREADS = 16
SAVES_PER_THREAD = 8
ACCESS_LOG_THREADS = 32
# Writers outside the orchestrator: threads in this process, then real child
# processes, each appending to the same access_log through update_project_file.
EXTERNAL_THREADS = 32
EXTERNAL_CHILDREN = 4
APPENDS_PER_CHILD = 5
# How much deeper than the interpreter allows a state has to nest before
# json.dumps gives up on it.
OVER_RECURSION_DEPTH = 100
# The Atlantic Standard Time offset every project.json timestamp carries.
AST_OFFSET = timedelta(hours=-4)
# Saves made before the descriptor count is taken, so a lazily created lock or
# log descriptor is already open, then saves made while it is watched.
FD_LEAK_WARMUP_SAVES = 5
FD_LEAK_SAVES = 50
# The child process holds the lock this long while the parent tries to save.
LOCK_HOLD_SECONDS = 0.6
# The parent's blocked save must take at least this long to prove exclusion.
LOCK_BLOCK_FLOOR_SECONDS = 0.3
# How long to wait for a child or a thread before calling it wedged.
WAIT_TIMEOUT_SECONDS = 20
# A large access_log, to prove a multi-megabyte state round-trips.
HUGE_LOG_ENTRIES = 5000

base = tempfile.mkdtemp(prefix="project_atomic_")
PY = sys.executable


def make_project(name, **extra):
    """Create a project directory with a saved project.json and return its state."""
    state = pm.create_project(name, base)
    if extra:
        state.update(extra)
        pm.save_project(state)
    return state


def read_raw(project_dir):
    """The exact bytes of a project's project.json, decoded as text."""
    with open(pm.project_file_path(project_dir), encoding="utf-8") as fh:
        return fh.read()


def temp_files(project_dir):
    """Every project.json temporary file sitting in a project directory."""
    return sorted(
        n for n in os.listdir(project_dir)
        if (n.startswith(pm.TEMP_PREFIX) and n.endswith(pm.TEMP_SUFFIX))
        or n == pm.LEGACY_TEMP_NAME)


def write_child(name, body):
    """Write a child script into the temp base and return its path."""
    path = os.path.join(base, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def backdate(path, seconds):
    """Set a file's mtime that many seconds into the past."""
    old = time.time() - seconds
    os.utime(path, (old, old))


def finished(fn, label):
    """Run fn on a thread and report whether it returned inside the timeout.

    A save that hangs on a lock nobody will ever release cannot be caught with
    try/except: the call simply never comes back. Every check for "no permanent
    hang" therefore runs on a thread and asks whether it is still alive.

    fn: the callable to run. label: how the case reads in the PASS/FAIL line.
    Returns: (done, result_or_error) where done is False when it is still
    running after WAIT_TIMEOUT_SECONDS.
    """
    box = []

    def run():
        try:
            box.append(("ok", fn()))
        except BaseException as exc:        # noqa: BLE001 - reported by caller
            box.append(("raised", exc))

    thread = threading.Thread(target=run, daemon=True, name=label)
    thread.start()
    thread.join(WAIT_TIMEOUT_SECONDS)
    if thread.is_alive():
        return False, None
    return True, box[0]


def stale_temp_in(project_dir, label):
    """Plant an abandoned temporary file in a project directory.

    project_dir: the project folder. label: a name fragment, so two planted
    files never collide.
    Returns: the path planted, already old enough to count as abandoned.
    """
    path = os.path.join(project_dir, f"{pm.TEMP_PREFIX}{label}{pm.TEMP_SUFFIX}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{partial")
    backdate(path, pm.TEMP_FILE_STALE_SECONDS + 60)
    return path


def log_entry(purpose):
    """One access_log entry, shaped the way every VICARIUS writer shapes it."""
    return {"ts": "2026-09-16T10:00:00-04:00", "initials": "LO",
            "action": "open", "purpose": purpose}


try:
    # ---------------------------------------------------------------- basics
    print("save and read back")
    st = make_project("atomic-basics")
    pdir = st["project_dir"]
    written = pm.save_project(st)
    check(written == pm.project_file_path(pdir),
          "save_project returns the path it wrote")
    stored = json.loads(read_raw(pdir))
    check(stored["id"] == st["id"], "project.json holds the saved state")
    check(temp_files(pdir) == [], "no temporary file is left after a save")
    check(os.path.isfile(os.path.join(pdir, pm.PROJECT_LOCK_NAME)),
          "the lock file sits beside project.json")
    check(pm.read_project_file(pdir)["id"] == st["id"],
          "read_project_file parses the saved file")

    mode = stat.S_IMODE(os.stat(pm.project_file_path(pdir)).st_mode)
    check(mode == pm.PROJECT_FILE_MODE,
          f"a new project.json is readable ({oct(mode)}), not mkstemp's 0o600")
    os.chmod(pm.project_file_path(pdir), 0o640)
    pm.save_project(st)
    mode = stat.S_IMODE(os.stat(pm.project_file_path(pdir)).st_mode)
    check(mode == 0o640, f"a save keeps the file's existing mode (got {oct(mode)})")

    # ------------------------------------------------- unique temporary name
    print("each writer gets its own temporary file")
    seen_temps = []
    real_replace = os.replace

    def recording_replace(src, dst):
        seen_temps.append(src)
        return real_replace(src, dst)

    os.replace = recording_replace
    try:
        pm.save_project(st)
        pm.save_project(st)
    finally:
        os.replace = real_replace
    check(len(seen_temps) == 2 and seen_temps[0] != seen_temps[1],
          "two saves use two different temporary files")
    check(all(os.path.dirname(t) == os.path.abspath(pdir) for t in seen_temps),
          "the temporary file sits in the project directory, so the move cannot "
          "cross a filesystem")
    check(all(os.path.basename(t) != pm.LEGACY_TEMP_NAME for t in seen_temps),
          "the fixed name project.json.tmp is never reused")

    # --------------------------------------------------- concurrent writers
    print("concurrent writers")
    conc = make_project("atomic-concurrent")
    conc_dir = conc["project_dir"]
    errors = []
    bad_reads = []
    stop_reading = threading.Event()

    def writer(worker):
        try:
            for i in range(SAVES_PER_THREAD):
                conc["steps"]["1"]["config"][f"worker_{worker}"] = i
                pm.save_project(conc)
        except BaseException as exc:            # noqa: BLE001 - reported below
            errors.append(f"writer {worker}: {exc!r}")

    def reader():
        while not stop_reading.is_set():
            try:
                text = read_raw(conc_dir)
                if not text.strip():
                    bad_reads.append("empty file")
                else:
                    json.loads(text)
            except FileNotFoundError:
                bad_reads.append("project.json missing mid-write")
            except ValueError as exc:
                bad_reads.append(f"unparsable: {exc}")

    watcher = threading.Thread(target=reader, daemon=True)
    watcher.start()
    threads = [threading.Thread(target=writer, args=(w,)) for w in range(WRITER_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT_TIMEOUT_SECONDS)
    stop_reading.set()
    watcher.join(WAIT_TIMEOUT_SECONDS)

    check(not any(t.is_alive() for t in threads),
          f"{WRITER_THREADS} writer threads all finished (no deadlock)")
    check(errors == [], f"no writer raised (first: {errors[:1]})")
    check(bad_reads == [],
          f"a concurrent reader never saw a truncated or empty file "
          f"({len(bad_reads)} bad reads, first: {bad_reads[:1]})")
    final = json.loads(read_raw(conc_dir))
    check(isinstance(final, dict) and final["id"] == conc["id"],
          "project.json is still valid JSON after "
          f"{WRITER_THREADS * SAVES_PER_THREAD} overlapping saves")
    stored_config = final.get("steps", {}).get("1", {}).get("config", {})
    every_worker = all(stored_config.get(f"worker_{w}") == SAVES_PER_THREAD - 1
                       for w in range(WRITER_THREADS))
    check(every_worker,
          f"every writer's last value is on disk, none mangled by an "
          f"interleaved write (stored {len(stored_config)} keys)")
    check(temp_files(conc_dir) == [],
          f"no temporary files survive the run (found {temp_files(conc_dir)})")

    # ------------------------------------------------ access_log under load
    print("access_log survives concurrency")
    logged = make_project("atomic-access-log")
    log_dir = logged["project_dir"]
    logged["access_log"] = []
    log_errors = []
    barrier = threading.Barrier(ACCESS_LOG_THREADS)

    def log_writer(worker):
        try:
            logged["access_log"].append(
                {"ts": "2026-09-16T10:00:00-04:00", "initials": "LO",
                 "action": "open", "purpose": f"worker {worker}"})
            barrier.wait(WAIT_TIMEOUT_SECONDS)
            pm.save_project(logged)
        except BaseException as exc:            # noqa: BLE001 - reported below
            log_errors.append(f"worker {worker}: {exc!r}")

    log_threads = [threading.Thread(target=log_writer, args=(w,))
                   for w in range(ACCESS_LOG_THREADS)]
    for t in log_threads:
        t.start()
    for t in log_threads:
        t.join(WAIT_TIMEOUT_SECONDS)
    check(log_errors == [], f"no access_log writer raised (first: {log_errors[:1]})")
    on_disk = json.loads(read_raw(log_dir))["access_log"]
    purposes = sorted(e["purpose"] for e in on_disk)
    check(len(on_disk) == ACCESS_LOG_THREADS,
          f"all {ACCESS_LOG_THREADS} access_log entries are on disk "
          f"(found {len(on_disk)})")
    check(purposes == sorted(f"worker {w}" for w in range(ACCESS_LOG_THREADS)),
          "every entry is present exactly once, none overwritten by an "
          "interleaved write")

    # --------------------------------------------- crash between write+replace
    print("a crash between the write and the replace")
    crashed = make_project("atomic-crash")
    crash_dir = crashed["project_dir"]
    good_text = read_raw(crash_dir)
    crash_script = write_child("crash_child.py", f'''"""Die between writing the
temporary file and moving it onto project.json, the way a killed orchestrator
would."""
import os
import sys
sys.path.insert(0, {ORCH_DIR!r})
import project_manager as pm

state = pm.read_project_file({crash_dir!r})
state["project_dir"] = {crash_dir!r}
state["name"] = "never-written"


def die_before_replace(src, dst):
    os._exit(9)


os.replace = die_before_replace
pm.save_project(state)
''')
    proc = subprocess.run([PY, crash_script], capture_output=True, text=True,
                          timeout=WAIT_TIMEOUT_SECONDS)
    check(proc.returncode == 9,
          f"the child died mid-write on purpose (rc={proc.returncode}, "
          f"stderr={proc.stderr.strip()[:120]!r})")
    check(read_raw(crash_dir) == good_text,
          "the previous project.json is byte-identical after the crash")
    check(json.loads(read_raw(crash_dir))["name"] == crashed["name"],
          "the crashed write never reached project.json")
    leftovers = temp_files(crash_dir)
    check(len(leftovers) == 1,
          f"the crash left its temporary file behind, as expected ({leftovers})")

    backdate(os.path.join(crash_dir, leftovers[0]), pm.TEMP_FILE_STALE_SECONDS + 60)
    pm.save_project(crashed)
    check(temp_files(crash_dir) == [],
          "the next save sweeps the temporary file the crash left")
    check(json.loads(read_raw(crash_dir))["name"] == crashed["name"],
          "the project still saves normally after a crashed writer")

    # ------------------------------------------- replace fails, nothing lost
    print("a failed replace leaves the good file in place")
    failing = make_project("atomic-replace-fails")
    fail_dir = failing["project_dir"]
    before = read_raw(fail_dir)

    def refuse_replace(src, dst):
        raise OSError(28, "No space left on device")

    os.replace = refuse_replace
    raised = None
    try:
        pm.save_project(failing)
    except OSError as exc:
        raised = exc
    finally:
        os.replace = real_replace
    check(isinstance(raised, OSError), f"a failed replace raises OSError (got {raised!r})")
    check(pm.PROJECT_FILE_NAME in str(raised) and fail_dir in str(raised),
          f"the error names the file it could not write ({str(raised)[:90]!r})")
    check(read_raw(fail_dir) == before, "project.json is untouched after the failure")
    check(temp_files(fail_dir) == [],
          f"the failed write cleaned up its temporary file ({temp_files(fail_dir)})")
    pm.save_project(failing)
    check(json.loads(read_raw(fail_dir))["id"] == failing["id"],
          "the project saves again once the failure clears")

    # --------------------------------------------------------- stale sweep
    print("stale temporary files")
    swept = make_project("atomic-sweep")
    sweep_dir = swept["project_dir"]
    stale = os.path.join(sweep_dir, f"{pm.TEMP_PREFIX}abandoned{pm.TEMP_SUFFIX}")
    legacy = os.path.join(sweep_dir, pm.LEGACY_TEMP_NAME)
    fresh = os.path.join(sweep_dir, f"{pm.TEMP_PREFIX}inflight{pm.TEMP_SUFFIX}")
    for path in (stale, legacy, fresh):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{partial")
    backdate(stale, pm.TEMP_FILE_STALE_SECONDS + 60)
    backdate(legacy, pm.TEMP_FILE_STALE_SECONDS + 60)
    pm.save_project(swept)
    check(not os.path.exists(stale), "an abandoned temporary file is swept")
    check(not os.path.exists(legacy),
          "the legacy fixed-name temporary file is swept too")
    check(os.path.isfile(fresh),
          "a temporary file young enough to belong to a live writer is kept")
    os.unlink(fresh)

    # --------------------------------------------------- corrupt read paths
    print("a corrupt project.json gives a clear error")
    corrupt = make_project("atomic-corrupt")
    corrupt_dir = corrupt["project_dir"]
    corrupt_path = pm.project_file_path(corrupt_dir)

    def expect_value_error(contents, needle, label):
        """Write contents over project.json and check the error it produces.

        contents: bytes for the file. needle: a phrase the message must carry.
        label: how the case reads in the PASS/FAIL line.
        """
        with open(corrupt_path, "wb") as fh:
            fh.write(contents)
        for fn, how in ((pm.read_project_file, "read_project_file"),
                        (pm.load_project, "load_project")):
            try:
                fn(corrupt_dir)
                check(False, f"{how}: {label} should raise")
            except ValueError as exc:
                msg = str(exc)
                check(type(exc) is ValueError,
                      f"{how}: {label} raises a plain ValueError rather than a "
                      f"raw decoder error (got {type(exc).__name__})")
                check(corrupt_path in msg,
                      f"{how}: {label} error names the file")
                check(needle in msg,
                      f"{how}: {label} error says what is wrong "
                      f"({msg[:80]!r})")

    expect_value_error(b'{"id": "half", "steps": {', "not valid JSON",
                       "truncated JSON")
    expect_value_error(b"", "is empty", "an empty file")
    expect_value_error(b"   \n\t ", "is empty", "a whitespace-only file")
    expect_value_error(b'["not", "an", "object"]', "not the object", "a JSON list")
    expect_value_error(b"\xff\xfe\x00\x01binary", "not readable as text",
                       "binary rubbish")

    with open(corrupt_path, "w", encoding="utf-8") as fh:
        fh.write('{"id": "half", "steps": {')
    try:
        pm.load_project(corrupt_dir)
    except ValueError:
        pass
    check(read_raw(corrupt_dir) == '{"id": "half", "steps": {',
          "a failed load never writes over the corrupt file")

    missing_dir = os.path.join(base, "no-such-project")
    os.makedirs(missing_dir, exist_ok=True)
    try:
        pm.read_project_file(missing_dir)
        check(False, "a missing project.json should raise FileNotFoundError")
    except FileNotFoundError as exc:
        check(missing_dir in str(exc),
              f"a missing project.json names the folder ({str(exc)[:70]!r})")

    # -------------------------------------------------- read-only directory
    print("a read-only project directory")
    if os.geteuid() == 0:
        skip("running as root: file modes cannot make a directory read-only")
    else:
        locked_existing = make_project("atomic-readonly-existing")
        existing_dir = locked_existing["project_dir"]
        existing_before = read_raw(existing_dir)
        os.chmod(existing_dir, 0o500)
        try:
            raised = None
            try:
                pm.save_project(locked_existing)
            except OSError as exc:
                raised = exc
            check(isinstance(raised, OSError),
                  f"a read-only directory raises OSError (got {raised!r})")
            check(pm.PROJECT_FILE_NAME in str(raised)
                  and os.path.abspath(existing_dir) in str(raised),
                  f"the error names the file and the directory "
                  f"({str(raised)[:110]!r})")
            check(read_raw(existing_dir) == existing_before,
                  "the read-only project's project.json is untouched")
        finally:
            os.chmod(existing_dir, 0o700)
        pm.save_project(locked_existing)
        check(json.loads(read_raw(existing_dir))["id"] == locked_existing["id"],
              "the project saves again once the directory is writable")

        virgin_dir = os.path.join(base, "atomic-readonly-virgin")
        os.makedirs(virgin_dir)
        virgin = {"id": "virgin", "name": "virgin", "project_dir": virgin_dir,
                  "steps": {}, "current_step": 1}
        with open(os.path.join(virgin_dir, pm.PROJECT_FILE_NAME), "w",
                  encoding="utf-8") as fh:
            json.dump(virgin, fh)
        virgin_before = read_raw(virgin_dir)
        os.chmod(virgin_dir, 0o500)
        try:
            raised = None
            try:
                pm.save_project(virgin)
            except OSError as exc:
                raised = exc
            check(isinstance(raised, OSError),
                  f"a read-only directory with no lock file raises OSError "
                  f"(got {raised!r})")
            check(os.path.abspath(virgin_dir) in str(raised),
                  f"that error names the directory ({str(raised)[:110]!r})")
            check(read_raw(virgin_dir) == virgin_before,
                  "its project.json is untouched")
        finally:
            os.chmod(virgin_dir, 0o700)

    # ------------------------------------------------- cross-process locking
    print("the lock excludes another process")
    shared = make_project("atomic-cross-process")
    shared_dir = shared["project_dir"]
    ready_marker = os.path.join(base, "child_holds_lock")
    hold_script = write_child("hold_lock_child.py", f'''"""Hold one project's
exclusive lock for a moment, so the parent's save has to wait for it."""
import sys
import time
sys.path.insert(0, {ORCH_DIR!r})
import project_manager as pm

with pm.exclusive_project_lock({shared_dir!r}):
    with open({ready_marker!r}, "w", encoding="utf-8") as fh:
        fh.write("held")
    time.sleep({LOCK_HOLD_SECONDS})
''')
    holder = subprocess.Popen([PY, hold_script])
    try:
        waited = 0.0
        while not os.path.exists(ready_marker) and waited < WAIT_TIMEOUT_SECONDS:
            time.sleep(0.01)
            waited += 0.01
        check(os.path.exists(ready_marker),
              "the child process took the lock")
        started = time.monotonic()
        pm.save_project(shared)
        elapsed = time.monotonic() - started
        check(elapsed >= LOCK_BLOCK_FLOOR_SECONDS,
              f"the parent's save waited for the child's lock "
              f"({elapsed:.2f}s, floor {LOCK_BLOCK_FLOOR_SECONDS}s)")
        check(json.loads(read_raw(shared_dir))["id"] == shared["id"],
              "the save went through once the lock was free")
    finally:
        holder.wait(timeout=WAIT_TIMEOUT_SECONDS)
    check(holder.returncode == 0, f"the lock-holding child exited cleanly "
                                  f"(rc={holder.returncode})")

    # ------------------------------------------- one holder at a time
    print("two threads never hold one project's lock at once")
    mutual = make_project("atomic-mutual-exclusion")
    inside = []
    overlaps = []

    def hold_briefly():
        with pm.exclusive_project_lock(mutual["project_dir"]):
            inside.append(1)
            if len(inside) > 1:
                overlaps.append(len(inside))
            time.sleep(0.01)
            inside.pop()

    holders = [threading.Thread(target=hold_briefly) for _ in range(8)]
    for t in holders:
        t.start()
    for t in holders:
        t.join(WAIT_TIMEOUT_SECONDS)
    check(not any(t.is_alive() for t in holders),
          "8 threads all took and released the lock")
    check(overlaps == [],
          f"writers serialise instead of interleaving ({len(overlaps)} overlaps)")

    # ---------------------------------------------------------- reentrancy
    print("a save nested inside the lock does not deadlock")
    nested = make_project("atomic-nested")
    nested_result = []

    def nested_save():
        try:
            with pm.exclusive_project_lock(nested["project_dir"]):
                pm.save_project(nested)
            nested_result.append("ok")
        except BaseException as exc:            # noqa: BLE001 - reported below
            nested_result.append(f"raised {exc!r}")

    nested_thread = threading.Thread(target=nested_save, daemon=True)
    nested_thread.start()
    nested_thread.join(WAIT_TIMEOUT_SECONDS)
    check(not nested_thread.is_alive(),
          "a save inside an already held lock returns instead of hanging")
    check(nested_result == ["ok"], f"the nested save succeeded ({nested_result})")

    # ---------------------------------------------------- input validation
    print("input validation")
    for bad, expected, label in (
            ("not a dict", TypeError, "a string"),
            (None, TypeError, "None"),
            ([], TypeError, "a list")):
        try:
            pm.save_project(bad)
            check(False, f"save_project({label}) should raise")
        except expected as exc:
            check("project state dict" in str(exc),
                  f"save_project({label}) says it needs a state dict "
                  f"({str(exc)[:70]!r})")

    try:
        pm.save_project({"id": "orphan"})
        check(False, "a state with no project_dir should raise")
    except ValueError as exc:
        check("project_dir" in str(exc),
              f"a state with no project_dir names the missing key "
              f"({str(exc)[:80]!r})")

    gone = os.path.join(base, "deleted-project")
    try:
        pm.save_project({"id": "gone", "project_dir": gone})
        check(False, "a vanished project directory should raise")
    except FileNotFoundError as exc:
        check(gone in str(exc),
              f"a vanished project directory is named in the error "
              f"({str(exc)[:80]!r})")

    unserialisable = make_project("atomic-unserialisable")
    unserial_dir = unserialisable["project_dir"]
    before = read_raw(unserial_dir)
    unserialisable["steps"]["1"]["config"]["sites"] = {"BPT", "CAS"}
    try:
        pm.save_project(unserialisable)
        check(False, "a state holding a set should raise")
    except ValueError as exc:
        check(unserial_dir in str(exc) and "JSON" in str(exc),
              f"an unserialisable state names the file ({str(exc)[:90]!r})")
    check(read_raw(unserial_dir) == before,
          "an unserialisable state never overwrites the good file")
    check(temp_files(unserial_dir) == [],
          "an unserialisable state leaves no temporary file")
    unserialisable["steps"]["1"]["config"].pop("sites")

    # ------------------------------------- state that moves while it is dumped
    print("state mutated by another thread mid-dump")

    class ShiftingDict(dict):
        """A dict that raises the way a dict mutated during iteration does."""

        def __init__(self, payload, failures):
            super().__init__(payload)
            self.failures_left = failures

        def items(self):
            if self.failures_left > 0:
                self.failures_left -= 1
                raise RuntimeError("dictionary changed size during iteration")
            return super().items()

    shifting = make_project("atomic-shifting")
    shift_dir = shifting["project_dir"]
    shifting["steps"]["1"]["config"] = ShiftingDict({"all_points": "x.csv"}, 1)
    pm.save_project(shifting)
    check(json.loads(read_raw(shift_dir))["steps"]["1"]["config"]["all_points"] == "x.csv",
          "a dump that hits one mid-dump mutation retries and succeeds")

    before = read_raw(shift_dir)
    shifting["steps"]["1"]["config"] = ShiftingDict({"all_points": "y.csv"},
                                                    pm.SERIALIZE_ATTEMPTS + 1)
    try:
        pm.save_project(shifting)
        check(False, "a state that never settles should raise")
    except ValueError as exc:
        check(shift_dir in str(exc) and "kept changing" in str(exc),
              f"a state that never settles gives a clear error ({str(exc)[:90]!r})")
    check(read_raw(shift_dir) == before,
          "a state that never settles never overwrites the good file")
    check(temp_files(shift_dir) == [],
          "a state that never settles leaves no temporary file")

    # --------------------------------------------------------- huge payload
    print("a very large state")
    huge = make_project("atomic-huge")
    huge_dir = huge["project_dir"]
    huge["access_log"] = [
        {"ts": "2026-09-16T10:00:00-04:00", "initials": "LO", "action": "open",
         "purpose": f"entry {i} with some padding text to make the file large"}
        for i in range(HUGE_LOG_ENTRIES)]
    pm.save_project(huge)
    reloaded = json.loads(read_raw(huge_dir))
    size_mb = os.path.getsize(pm.project_file_path(huge_dir)) / (1024 * 1024)
    check(len(reloaded["access_log"]) == HUGE_LOG_ENTRIES,
          f"a {size_mb:.1f} MB project.json round-trips whole "
          f"({len(reloaded['access_log'])} entries)")
    check(temp_files(huge_dir) == [], "the large save leaves no temporary file")

    # ------------------------------------------------- hostile file contents
    print("hostile content round-trips")
    hostile = make_project("atomic-hostile")
    hostile_dir = hostile["project_dir"]
    nasty = 'quote" brace} newline\n tab\t backslash\\ unicode é中 null-ish \\u0000'
    hostile["description"] = nasty
    hostile["steps"]["1"]["config"]["note"] = nasty
    pm.save_project(hostile)
    back = json.loads(read_raw(hostile_dir))
    check(back["description"] == nasty and back["steps"]["1"]["config"]["note"] == nasty,
          "quotes, braces, newlines and non-ASCII survive the write unchanged")
    check(pm.load_project(hostile_dir)["description"] == nasty,
          "load_project reads that content back unchanged")

    # ------------------------------------------- listing tolerates a bad file
    print("a corrupt project does not break the listing")
    listing_root = os.path.join(base, "listing")
    os.makedirs(listing_root, exist_ok=True)
    good = pm.create_project("listing-good", listing_root)
    broken_dir = os.path.join(listing_root, "26-09-16_listing-broken")
    os.makedirs(broken_dir, exist_ok=True)
    with open(os.path.join(broken_dir, pm.PROJECT_FILE_NAME), "w",
              encoding="utf-8") as fh:
        fh.write("{not json")
    found = dict((pid, pdir) for pid, pdir in pm.find_projects(listing_root))
    check(good["id"] in found, "the good project is listed")
    check("26-09-16_listing-broken" in found,
          "the corrupt project is listed under its folder name, not skipped")

    # ------------------------------------------- no leaked descriptors
    print("the lock file descriptor is always closed")
    FD_DIR = "/proc/self/fd"
    if not os.path.isdir(FD_DIR):
        skip("this kernel has no /proc/self/fd, so descriptors cannot be counted")
    else:
        leaky = make_project("atomic-fd-leak")

        def open_fds():
            """How many file descriptors this process holds right now."""
            return len(os.listdir(FD_DIR))

        for _ in range(FD_LEAK_WARMUP_SAVES):
            pm.save_project(leaky)
        before_fds = open_fds()
        for _ in range(FD_LEAK_SAVES):
            pm.save_project(leaky)
        after_fds = open_fds()
        check(after_fds <= before_fds,
              f"{FD_LEAK_SAVES} saves leak no descriptor "
              f"(held {before_fds}, now {after_fds})")

    # ------------------------------------- a filesystem that cannot lock
    print("a filesystem that cannot lock saves anyway")
    if not pm._HAVE_FCNTL:
        skip("this platform has no fcntl, so there is no flock to refuse")
    else:
        nolock = make_project("atomic-no-flock")
        nolock_dir = nolock["project_dir"]
        nolock_lock = pm._lock_file_path(nolock_dir)
        pm._NO_CROSS_PROCESS_LOCK.discard(nolock_lock)
        real_flock = pm.fcntl.flock

        def refuse_flock(fd, operation):
            """What an NFS mount with no lock daemon answers."""
            raise OSError(errno.ENOLCK, "No locks available")

        captured = io.StringIO()
        real_stderr = sys.stderr
        pm.fcntl.flock = refuse_flock
        sys.stderr = captured
        try:
            done, outcome = finished(lambda: pm.save_project(nolock),
                                     "save-without-flock")
            done_twice, outcome_twice = finished(lambda: pm.save_project(nolock),
                                                 "save-without-flock-again")
        finally:
            pm.fcntl.flock = real_flock
            sys.stderr = real_stderr
        check(done and outcome[0] == "ok",
              f"a save on a filesystem that cannot lock returns instead of "
              f"failing or hanging (got {outcome})")
        check(done_twice and outcome_twice[0] == "ok",
              f"and so does the next one (got {outcome_twice})")
        check(json.loads(read_raw(nolock_dir))["id"] == nolock["id"],
              "the state reached project.json without the cross-process lock")
        warning = captured.getvalue()
        check(warning.count("cannot be locked") == 1,
              f"the lost guarantee is reported once, not once per save "
              f"({warning.count('cannot be locked')} times)")
        check(nolock_lock in warning and "second process" in warning,
              f"the report names the file and says what is at risk "
              f"({warning.strip()[:110]!r})")
        check(not pm._THREAD_LOCKS[nolock_lock].locked(),
              "the in-process lock is released, not left held for ever")
        done_after, outcome_after = finished(lambda: pm.save_project(nolock),
                                             "save-after-flock-recovers")
        check(done_after and outcome_after[0] == "ok",
              f"a later save on a healthy filesystem returns "
              f"(got {outcome_after})")

    # -------------------------------------- a failure releasing the lock
    print("a failure while releasing the lock")
    brittle = make_project("atomic-release-fails")
    brittle_dir = brittle["project_dir"]
    brittle_lock = pm._lock_file_path(brittle_dir)
    real_release = pm._release_lock_file

    def refuse_release(lock_fd, held):
        """Close the descriptor, then fail the way a dying mount would."""
        os.close(lock_fd)
        raise OSError(errno.EIO, "Input/output error")

    pm._release_lock_file = refuse_release
    try:
        done, outcome = finished(lambda: pm.save_project(brittle),
                                 "save-with-failing-release")
    finally:
        pm._release_lock_file = real_release
    check(done, "a save whose lock release fails still returns")
    check(outcome is not None and outcome[0] == "raised"
          and isinstance(outcome[1], OSError),
          f"and the failure is reported rather than swallowed (got {outcome})")
    check(not pm._THREAD_LOCKS[brittle_lock].locked(),
          "the in-process lock is released even when the release path raises")
    done_after, outcome_after = finished(lambda: pm.save_project(brittle),
                                         "save-after-failing-release")
    check(done_after and outcome_after[0] == "ok",
          f"the project still saves afterwards, with no permanent hang "
          f"(got {outcome_after})")

    # --------------------------------------------- a read-only lock file
    print("a read-only lock file")
    if os.geteuid() == 0:
        skip("running as root: file modes cannot make a lock file read-only")
    else:
        ro_lock = make_project("atomic-readonly-lockfile")
        ro_dir = ro_lock["project_dir"]
        lock_file = pm._lock_file_path(ro_dir)
        os.chmod(lock_file, 0o444)
        try:
            raised = None
            try:
                pm.save_project(ro_lock)
            except OSError as exc:
                raised = exc
            check(raised is None,
                  f"a save needs no write permission on the lock file "
                  f"(got {raised!r})")
            check(json.loads(read_raw(ro_dir))["id"] == ro_lock["id"],
                  "and the state is on disk")
        finally:
            os.chmod(lock_file, 0o644)

    # ------------------------------- a project folder with a glob character
    print("a project folder holding a glob metacharacter")
    tricky = pm.create_project("atomic-glob[a-z]*set", base)
    tricky_dir = tricky["project_dir"]
    check("[" in os.path.basename(tricky_dir) and "*" in os.path.basename(tricky_dir),
          f"the folder really carries the characters "
          f"({os.path.basename(tricky_dir)!r})")
    tricky_stale = stale_temp_in(tricky_dir, "abandoned")
    pm.save_project(tricky)
    check(not os.path.exists(tricky_stale),
          "the stale sweep still clears an abandoned temporary file there")
    with open(os.path.join(tricky["steps"]["1"]["dir"], "all_points.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("image,x,y\n")
    pm.complete_step(tricky, 1)
    check(tricky["steps"]["2"]["config"].get("all_points", "").endswith("all_points.csv"),
          f"step 1's output still auto-links into step 2 "
          f"(got {tricky['steps']['2']['config'].get('all_points')!r})")

    # -------------------------------------------- a state nested too deeply
    print("a state nested too deeply to dump")
    deep = make_project("atomic-deep")
    deep_dir = deep["project_dir"]
    before = read_raw(deep_dir)
    nest = {}
    cursor = nest
    for _ in range(sys.getrecursionlimit() + OVER_RECURSION_DEPTH):
        cursor["child"] = {}
        cursor = cursor["child"]
    deep["steps"]["1"]["config"]["deep"] = nest
    try:
        pm.save_project(deep)
        check(False, "a state nested too deeply should raise")
    except ValueError as exc:
        msg = str(exc)
        check("nested too deeply" in msg and deep_dir in msg,
              f"a state nested too deeply says so and names the file "
              f"({msg[:90]!r})")
        check("kept changing" not in msg,
              "and does not blame a concurrent writer that does not exist")
    check(read_raw(deep_dir) == before,
          "the good file survives a state that cannot be dumped")
    deep["steps"]["1"]["config"].pop("deep")
    del nest, cursor

    # --------------------------------- a writer outside the orchestrator
    print("a writer outside the orchestrator")
    external = make_project("atomic-external-writer", access_log=[])
    ext_dir = external["project_dir"]
    written = pm.update_project_file(
        ext_dir, lambda state: state["access_log"].append(log_entry("first")))
    on_disk = json.loads(read_raw(ext_dir))
    check(on_disk["access_log"][-1]["purpose"] == "first",
          "an outside writer's entry reaches project.json")
    check(written["access_log"] == on_disk["access_log"],
          "update_project_file returns the state it wrote")
    check(datetime.fromisoformat(on_disk["updated_at"]).utcoffset() == AST_OFFSET,
          f"it stamps updated_at in Atlantic Standard Time "
          f"(got {on_disk['updated_at']!r})")
    check(temp_files(ext_dir) == [], "it leaves no temporary file")

    try:
        pm.update_project_file(ext_dir, {"not": "callable"})
        check(False, "a non-callable mutate should raise")
    except TypeError as exc:
        check("callable" in str(exc),
              f"a non-callable mutate says what it needed ({str(exc)[:70]!r})")
    try:
        pm.update_project_file(os.path.join(base, "no-such-folder"),
                               lambda state: None)
        check(False, "a vanished folder should raise")
    except FileNotFoundError as exc:
        check("does not exist" in str(exc),
              f"a vanished folder is named ({str(exc)[:80]!r})")
    empty_dir = os.path.join(base, "external-no-project-json")
    os.makedirs(empty_dir, exist_ok=True)
    try:
        pm.update_project_file(empty_dir, lambda state: None)
        check(False, "a folder with no project.json should raise")
    except FileNotFoundError as exc:
        check(empty_dir in str(exc),
              f"a folder with no project.json is named ({str(exc)[:80]!r})")

    ext_errors = []

    def external_appender(worker):
        try:
            pm.update_project_file(
                ext_dir,
                lambda state: state["access_log"].append(
                    log_entry(f"thread {worker}")))
        except BaseException as exc:        # noqa: BLE001 - reported below
            ext_errors.append(f"worker {worker}: {exc!r}")

    ext_threads = [threading.Thread(target=external_appender, args=(w,))
                   for w in range(EXTERNAL_THREADS)]
    for t in ext_threads:
        t.start()
    for t in ext_threads:
        t.join(WAIT_TIMEOUT_SECONDS)
    check(not any(t.is_alive() for t in ext_threads),
          f"{EXTERNAL_THREADS} outside writers all finished (no deadlock)")
    check(ext_errors == [], f"none of them raised (first: {ext_errors[:1]})")
    thread_log = json.loads(read_raw(ext_dir))["access_log"]
    thread_purposes = sorted(e["purpose"] for e in thread_log if e["purpose"] != "first")
    check(thread_purposes == sorted(f"thread {w}" for w in range(EXTERNAL_THREADS)),
          f"every outside writer's entry survived, none lost to a read-modify-"
          f"write race ({len(thread_purposes)} of {EXTERNAL_THREADS})")

    # ----------------------------- outside writers in separate processes
    print("outside writers in separate processes")
    cross = make_project("atomic-external-processes", access_log=[])
    cross_dir = cross["project_dir"]
    appender_script = write_child("external_appender_child.py", f'''"""Append to
one project's access_log the way a writer outside the orchestrator does, through
the shared lock, so several processes can be proven not to lose each other's
entries."""
import sys
sys.path.insert(0, {ORCH_DIR!r})
import project_manager as pm

label = sys.argv[1]
for i in range({APPENDS_PER_CHILD}):
    pm.update_project_file(
        {cross_dir!r},
        lambda state, n=i: state.setdefault("access_log", []).append(
            {{"ts": "2026-09-16T10:00:00-04:00", "initials": "LO",
              "action": "open", "purpose": "%s %d" % (label, n)}}))
''')
    children = [subprocess.Popen([PY, appender_script, f"child{c}"])
                for c in range(EXTERNAL_CHILDREN)]
    parent_errors = []
    try:
        for i in range(APPENDS_PER_CHILD):
            try:
                pm.update_project_file(
                    cross_dir,
                    lambda state, n=i: state["access_log"].append(
                        log_entry(f"parent {n}")))
            except BaseException as exc:    # noqa: BLE001 - reported below
                parent_errors.append(repr(exc))
    finally:
        for child in children:
            child.wait(timeout=WAIT_TIMEOUT_SECONDS)
    check(parent_errors == [],
          f"the parent's own appends all went through (first: {parent_errors[:1]})")
    check(all(c.returncode == 0 for c in children),
          f"every child exited cleanly (rc={[c.returncode for c in children]})")
    cross_log = json.loads(read_raw(cross_dir))["access_log"]
    expected = sorted([f"child{c} {i}" for c in range(EXTERNAL_CHILDREN)
                       for i in range(APPENDS_PER_CHILD)]
                      + [f"parent {i}" for i in range(APPENDS_PER_CHILD)])
    check(sorted(e["purpose"] for e in cross_log) == expected,
          f"all {len(expected)} entries from {EXTERNAL_CHILDREN} processes plus "
          f"this one are on disk, none overwritten ({len(cross_log)} found)")
    check(temp_files(cross_dir) == [],
          f"no temporary file survives them ({temp_files(cross_dir)})")

    # --------------------------------------------------------- timestamps
    print("timestamps carry the Atlantic Standard Time offset")
    stamped = pm.create_project("atomic-timestamps", base)
    stamped_dir = stamped["project_dir"]
    for field in ("created_at", "updated_at"):
        try:
            offset = datetime.fromisoformat(stamped[field]).utcoffset()
        except (TypeError, ValueError):
            offset = None
        check(offset == AST_OFFSET,
              f"{field} is ISO 8601 in AST, not a naive local time "
              f"(got {stamped[field]!r})")
    with open(os.path.join(stamped["steps"]["1"]["dir"], "all_points.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("image,x,y\n")
    pm.complete_step(stamped, 1)
    completed = stamped["steps"]["1"].get("completed_at", "")
    try:
        completed_offset = datetime.fromisoformat(completed).utcoffset()
    except (TypeError, ValueError):
        completed_offset = None
    check(completed_offset == AST_OFFSET,
          f"completed_at carries the offset too (got {completed!r})")
    check(json.loads(read_raw(stamped_dir))["updated_at"] == stamped["updated_at"],
          "the stored timestamp is the one the state carries")
    check(pm._created_date_str(stamped["created_at"])
          == datetime.now(pm.AST).strftime("%y-%m-%d"),
          "the auto name still reads the date out of the stored created_at")

    # ------------------------------------------------ existing flow still works
    print("the normal project flow still works")
    flow = pm.create_project("atomic-flow", base)
    # load_project demotes a completed step whose output directory is empty, so
    # give Step 1 an output before completing it.
    with open(os.path.join(flow["steps"]["1"]["dir"], "all_points.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("image,x,y\n")
    pm.complete_step(flow, 1)
    reloaded = pm.load_project(flow["project_dir"])
    check(reloaded["steps"]["1"]["status"] == "completed",
          "complete_step still persists through the new write path")
    check(reloaded["current_step"] == 2, "current_step still reconciles on load")
    pm.reset_step(reloaded, 1)
    check(json.loads(read_raw(flow["project_dir"]))["steps"]["1"]["status"] == "pending",
          "reset_step still persists through the new write path")
    check(temp_files(flow["project_dir"]) == [],
          "the whole flow leaves no temporary files")

finally:
    shutil.rmtree(base, ignore_errors=True)

check(not os.path.exists(base), "temp dir removed")
print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
