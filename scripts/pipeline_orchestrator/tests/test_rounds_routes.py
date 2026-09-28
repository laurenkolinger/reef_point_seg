"""Task 9 (orchestrator side): rounds ledger read route + champion promotion.

evaluate_run.py's append_round_row / ROUNDS_CSV_HEADER are covered by
TR/tests/test_rounds_ledger.py; this file only exercises the orchestrator
routes that sit on top of the rounds.csv/champion.json files it produces:
  GET  /api/step/7/rounds    -> parsed rows (numeric cols cast) + champion
  POST /api/step/7/promote   -> validates weights + an eval row, writes
                                champion.json beside the run's own ledger AND
                                at <inprocess>/_models/champion.json
  GET  /api/step/6/list_runs -> the run picker's champion default + its task
  GET  /api/step/6/all_runs  -> the runs card, including what it says when the
                                promoted run's folder is no longer on disk

EVERY test here points app._inprocess_root() at its own fixture. The promote
route writes the champion record twice, and the second write is under the
inprocess root: a test that leaves that function pointing at the real module
tree writes a champion record into
<repo>/../inprocess/_models/champion.json naming a temp folder that no longer
exists, which is exactly what happened on 2026-09-05 and made the live window
report four detector runs as "not comparable: different metric family".
test_promote_writes_champion_json proves both writes are inside the fixture
and that no champion record in the real module tree moved.

That guard is scoped to the paths this route can create: files named
champion.json or champion.json.tmp (the only names the route writes, through
_write_json_atomic) and the fixture project's own folder name. Everything else
under the real inprocess root is ignored on purpose, because a training job
writes its own logs, weights and metrics there for hours at a time and none of
that can have come from the route under test.
test_module_tree_guard_ignores_a_concurrent_job and
test_module_tree_guard_reports_every_champion_write prove both halves of that
on a throwaway folder, so the guard is never quietly vacuous, and
test_write_surface_pin_sees_a_file_the_name_guard_ignores proves the wider scan
the escape test pins the route's write surface with is not blind either.

The VICARIUS event bridge is replaced with a recorder before any route runs.
POST /api/step/7/promote records a user note on every success, and with the
real bridge live that note goes into the shared platform event database at
/mnt/rip/vicarius_drive/vicarius/_logging/db/vicarius_events.db, so every run
of this suite filed promotions of fixture runs that never existed.

No-pytest harness: run with env/bin/python <this file>.
"""
import os
import sys
import csv
import json
import shutil
import signal
import stat
import hashlib
import tempfile
import time
from collections.abc import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
sys.path.insert(0, _ORCH)

import app as appmod
import project_manager as pm


class _FakeVic:
    """The VICARIUS event bridge with every call kept in memory.

    The promote route records a user note on success. The real bridge writes
    that note into the platform event database, where it reads as a promotion
    that really happened, so this stand-in replaces it for the whole suite.
    """

    notes: list[str] = []

    @staticmethod
    def process_start(*_args, **_kwargs) -> None:
        """Stand in for the bridge's process_start. Returns None, which is
        what the real bridge returns when event logging is switched off."""
        return None

    @staticmethod
    def process_end(*_args, **_kwargs) -> None:
        """Stand in for the bridge's process_end, recording nothing."""
        return None

    @classmethod
    def note(cls, message: str, *_args, **_kwargs) -> None:
        """Keep one user note in memory instead of writing it to the shared
        platform event database."""
        cls.notes.append(message)


appmod.vic = _FakeVic()

_RESULTS = []

# The real functions, captured before any test redirects them, and the real
# module inprocess folder they resolve to. _MODULE_INPROCESS is only ever
# scanned, never written, by this file.
_REAL_INPROCESS_ROOT = appmod._inprocess_root
_MODULE_INPROCESS = os.path.abspath(_REAL_INPROCESS_ROOT())

# The fixture project's folder name. It starts with "run_" because
# labruns.list_all_runs only scans inprocess children with that prefix.
FIXTURE_PROJECT_NAME = "run_26-09-05_LO_rounds_fixture"
MODELS_DIR_NAME = "_models"
# A run folder name for the guard fixture only, standing in for whatever job
# happens to be writing into the module tree while this suite runs.
CONCURRENT_JOB_RUN_NAME = "run_26-09-16_LO_concurrent_training_job"
CHAMPION_FILE_NAME = "champion.json"

# _write_json_atomic writes <path>.tmp and renames it over <path>, so both
# names belong to the route.
ATOMIC_TMP_SUFFIX = ".tmp"
# The only file names POST /api/step/7/promote can create anywhere. A
# concurrent training job never writes either one, which is what keeps the
# module-tree guard below precise.
ROUTE_WRITTEN_FILE_NAMES = (CHAMPION_FILE_NAME, CHAMPION_FILE_NAME + ATOMIC_TMP_SUFFIX)
# A file name POST /api/step/7/promote never writes. It stands in for a write
# the route could grow, so the wider scan behind the escape test's pin can be
# shown to report what the name-scoped guard skips.
UNWATCHED_FILE_NAME = "extra_manifest.json"
# How much of a watched file is hashed. A champion record is a few hundred
# bytes; the cap only bounds the read if something unexpected takes the name,
# and size plus modification time already separate those cases.
FINGERPRINT_MAX_BYTES = 1 << 20
# What _fingerprint calls each kind of watched path. Only a regular file is
# opened: the guard scans a folder every process on this box can write to, and
# opening a FIFO there blocks until a writer arrives.
FINGERPRINT_KIND_DIR = "dir"
FINGERPRINT_KIND_LINK = "link"
FINGERPRINT_KIND_FILE = "file"
FINGERPRINT_KIND_OTHER = "other"
# How long one call may run under _call_within_deadline before the suite calls
# it a hang. Hashing a few hundred bytes is instant, so only a blocking open
# reaches this bound.
FINGERPRINT_DEADLINE_SECONDS = 5.0
# The bound the deadline's own test uses, and the sleep it outruns. Short
# enough that proving the alarm fires costs the board a twentieth of a second.
DEADLINE_SELF_TEST_SECONDS = 0.05
DEADLINE_SELF_TEST_SLEEP_SECONDS = 5.0
# How many escaped paths a failure message quotes before it stops.
MAX_ESCAPED_PATHS_REPORTED = 10
# Stands for "this path was not in the mapping at all", so a path that appears
# and then vanishes before it can be stat-ed is still reported as a change.
_MISSING = object()

ROUNDS_CSV_HEADER = [
    'round', 'at', 'run_name', 'run_dir', 'base_model', 'split_pinned',
    'n_train', 'n_valid', 'n_test',
    'map50_M', 'map50_95_M', 'recall_M', 'precision_M',
    'map50_B', 'map50_95_B',
    'gate_map', 'gate_class', 'gate_recall',
    'per_class_json',
]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# One watched path as this suite compares it: its kind, its size (None for a
# folder), its modification time in nanoseconds, and a detail that depends on
# the kind. None means the path was not there at all.
Fingerprint = tuple[str, int | None, int, str] | None


class _DeadlineExceeded(Exception):
    """A call under _call_within_deadline outran its wall-clock bound."""


def _call_within_deadline(call: Callable[[], object], seconds: float) -> object:
    """Run call() under a wall-clock bound, so a read that blocks fails the
    suite instead of hanging the board for whoever runs it next.

    Parameters:
        call: a function taking no arguments.
        seconds: the bound in seconds, greater than zero.
    Returns:
        whatever call() returns.
    Raises:
        ValueError when seconds is not a positive number.
        _DeadlineExceeded when call() is still running at the bound. The alarm
        interrupts the blocking system call, which is the only way a test can
        prove that a read never waits on another process.
    """
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        raise ValueError(f"seconds must be a positive number, got {seconds!r}")

    def _fire(_signum, _frame):
        """Turn the alarm into the exception the caller reports as a hang."""
        raise _DeadlineExceeded(f"the call was still running after {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _require_scan_root(root: object) -> str:
    """Reject a scan root that is not a usable path, before any walk starts.

    Parameters:
        root: the value handed to one of the two scan helpers below.
    Returns:
        root unchanged, once it is a non-empty string.
    Raises:
        ValueError naming what arrived. os.walk yields nothing for a blank or
        non-string root, so a guard built on one would pass while watching no
        path at all.
    """
    if not isinstance(root, str) or not root:
        raise ValueError(f"root must be a non-empty path string, got {root!r}")
    return root


def _fingerprint(path: str) -> Fingerprint:
    """What one watched path looks like right now, or None when it is absent.

    Parameters:
        path: absolute path. Stat-ed without following a symlink, so a link's
            target is never read.
    Returns:
        (kind, size, modification time in nanoseconds, detail), or None when
        the path does not exist. kind is FINGERPRINT_KIND_DIR (size None,
        detail ""), FINGERPRINT_KIND_LINK (detail the link target),
        FINGERPRINT_KIND_FILE (detail the sha256 hex of the first
        FINGERPRINT_MAX_BYTES bytes) or FINGERPRINT_KIND_OTHER (detail "") for
        a FIFO, socket or device node. Content is hashed as well as stat-ed so
        a rewrite is caught even if the clock does not move.

        Only a regular file is opened. This guard scans the live module
        inprocess folder, where any process can give any name any kind, and
        opening a FIFO there blocks until a writer arrives, which would hang
        the suite rather than fail it.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        try:
            return (FINGERPRINT_KIND_LINK, info.st_size, info.st_mtime_ns, os.readlink(path))
        except OSError:
            return (FINGERPRINT_KIND_LINK, info.st_size, info.st_mtime_ns, "")
    if stat.S_ISDIR(info.st_mode):
        return (FINGERPRINT_KIND_DIR, None, info.st_mtime_ns, "")
    if not stat.S_ISREG(info.st_mode):
        return (FINGERPRINT_KIND_OTHER, info.st_size, info.st_mtime_ns, "")
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read(FINGERPRINT_MAX_BYTES)).hexdigest()
    except OSError:
        digest = ""
    return (FINGERPRINT_KIND_FILE, info.st_size, info.st_mtime_ns, digest)


def _route_write_fingerprints(root: str) -> dict[str, Fingerprint]:
    """{path: _fingerprint(path)} for every path under root that the promote
    route could have created.

    Watched: any file named champion.json or champion.json.tmp, and any entry
    named FIXTURE_PROJECT_NAME. A champion record is the only thing POST
    /api/step/7/promote writes into an inprocess root, and the only folder it
    creates is the _models folder holding that record, whose arrival the
    record itself proves. So anything else under root belongs to another
    writer and is skipped, which is what lets this guard run against the live
    module tree while a training job writes into it.

    The FIXTURE_PROJECT_NAME arm is a forward guard, not a live one. Every
    fixture this suite builds sits inside its own temporary folder, so nothing
    here can put that name in the real module tree today; the arm is watched
    so that a fixture built under the real root later reads as an escape on
    its first run instead of passing quietly.
    test_module_tree_guard_reports_every_champion_write exercises it on a
    throwaway folder.

    What this name scoping skips is pinned by
    test_module_tree_guard_catches_a_promote_that_escaped_its_fixture, at both
    roots the route writes into: the inprocess root it is handed and the run's
    own ledger folder inside the project. A promote that grows a third write
    fails that pin instead of slipping past unwatched.

    Parameters:
        root: absolute path of the folder to scan. Read only, never written.
            A root that does not exist yields an empty mapping.
    Returns:
        The mapping described above, empty when root holds no watched path.
        os.walk skips folders that vanish or cannot be read mid-scan.
    Raises:
        ValueError when root is not a non-empty path string.
    """
    _require_scan_root(root)
    watched = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            if name not in ROUTE_WRITTEN_FILE_NAMES and name != FIXTURE_PROJECT_NAME:
                continue
            path = os.path.join(dirpath, name)
            watched[path] = _fingerprint(path)
    return watched


def _every_path_under(root: str) -> set[str]:
    """Every path under root, relative to it, as a set.

    Only ever pointed at a private temporary folder: on a shared tree such as
    the real module inprocess folder this cannot tell one writer from another,
    which is the whole reason the guard above is scoped by name instead.

    Parameters:
        root: absolute path of the folder to scan. Read only, never written.
    Returns:
        A set of paths relative to root, using os.sep, empty when root holds
        nothing.
    Raises:
        ValueError when root is not a non-empty path string, on the same terms
        as _route_write_fingerprints. A scan helper that returned an empty set
        for a bad root would make the write-surface pin built on it vacuous.
    """
    _require_scan_root(root)
    found = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            found.add(os.path.relpath(os.path.join(dirpath, name), root))
    return found


def _fingerprint_diff(before: dict[str, Fingerprint],
                      after: dict[str, Fingerprint]) -> list[str]:
    """Sorted paths that appeared, vanished or changed between two mappings.

    Parameters:
        before, after: mappings from _route_write_fingerprints.
    Returns:
        A sorted list of absolute paths, empty when nothing watched moved.
    """
    changed = [p for p in after if before.get(p, _MISSING) != after[p]]
    changed += [p for p in before if p not in after]
    return sorted(set(changed))


def _make_project(project_dir):
    """Project state with every step folder created under project_dir."""
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(project_dir, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    state = {
        "id": os.path.basename(project_dir), "name": "t", "project_dir": project_dir,
        "created_at": "x", "updated_at": "x", "current_step": 7,
        "steps": steps,
    }
    return state


def _write_rounds_csv(step6_dir, rows):
    os.makedirs(step6_dir, exist_ok=True)
    path = os.path.join(step6_dir, "rounds.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROUNDS_CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ROUNDS_CSV_HEADER})
    return path


def _make_run(step6_dir, run_name, with_best=True, task="segment"):
    """A run folder with weights/best.pt and the args.yaml line labruns reads
    to tell a segmenter from a detector."""
    run_dir = os.path.join(step6_dir, "runs", run_name)
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    if with_best:
        with open(os.path.join(weights_dir, "best.pt"), "wb") as f:
            f.write(b"\x00" * 8)
    with open(os.path.join(run_dir, "args.yaml"), "w") as f:
        f.write(f"task: {task}\nimgsz: 512\nmodel: yolo11m-seg.pt\n")
    return run_dir


def _client_with_project():
    """A test client whose project AND inprocess root are both inside one
    temporary folder. Returns (client, tmp, state); every caller closes with
    _close(tmp) so the redirected root is restored."""
    tmp = tempfile.mkdtemp(prefix="rounds_routes_")
    root = os.path.join(tmp, "inprocess")
    project_dir = os.path.join(root, FIXTURE_PROJECT_NAME)
    os.makedirs(project_dir, exist_ok=True)
    state = _make_project(project_dir)
    appmod._inprocess_root = lambda: root
    client = appmod.create_app().test_client()
    appmod.current_project = state
    return client, tmp, state


def _close(tmp):
    """Restore the module state this suite redirects and remove the fixture."""
    appmod.current_project = None
    appmod._inprocess_root = _REAL_INPROCESS_ROOT
    shutil.rmtree(tmp, ignore_errors=True)


def _fixture_root(tmp):
    """The fixture's inprocess root."""
    return os.path.join(tmp, "inprocess")


def _module_champion_path(tmp):
    """The module-wide champion record inside the fixture."""
    return os.path.join(_fixture_root(tmp), MODELS_DIR_NAME, CHAMPION_FILE_NAME)


def test_the_hang_bound_fires_on_a_call_that_outruns_it():
    """The bound the FIFO test below leans on, proved first. Without this, a
    _call_within_deadline that had quietly stopped arming its alarm would let
    a blocking read hang the board rather than fail it."""
    started = time.monotonic()
    try:
        _call_within_deadline(lambda: time.sleep(DEADLINE_SELF_TEST_SLEEP_SECONDS),
                              DEADLINE_SELF_TEST_SECONDS)
    except _DeadlineExceeded:
        waited = time.monotonic() - started
        check(waited < DEADLINE_SELF_TEST_SLEEP_SECONDS,
              f"the bound must cut the call short, it waited {waited:.2f}s")
    else:
        check(False, "a call that outruns its bound must raise _DeadlineExceeded")

    try:
        _call_within_deadline(lambda: None, 0)
    except ValueError as exc:
        check("positive" in str(exc), f"a bad bound must say what is wrong: {exc}")
    else:
        check(False, "a bound of zero must be refused")


def test_fingerprint_never_opens_a_path_that_is_not_a_regular_file():
    """The guard scans the live module inprocess folder, where any process can
    give any name any kind. A FIFO named champion.json is the kind that
    matters: opening one blocks until a writer arrives, so a guard that opened
    it would hang every run of this suite instead of failing it. _fingerprint
    takes the kind from lstat and opens nothing but a regular file, while the
    name stays watched so its arrival still reads as a change."""
    tmp = tempfile.mkdtemp(prefix="rounds_fifo_")
    try:
        fifo_path = os.path.join(tmp, CHAMPION_FILE_NAME)
        os.mkfifo(fifo_path)

        seen = _call_within_deadline(lambda: _fingerprint(fifo_path),
                                     FINGERPRINT_DEADLINE_SECONDS)
        check(seen is not None and seen[0] == FINGERPRINT_KIND_OTHER,
              f"a FIFO must fingerprint as {FINGERPRINT_KIND_OTHER!r}, got {seen}")

        watched = _call_within_deadline(lambda: _route_write_fingerprints(tmp),
                                        FINGERPRINT_DEADLINE_SECONDS)
        check(list(watched) == [fifo_path],
              f"the guard must watch the name whatever kind takes it: {sorted(watched)}")
        check(_fingerprint_diff({}, watched) == [fifo_path],
              "a FIFO appearing under a watched name must read as a change")

        plain = os.path.join(tmp, MODELS_DIR_NAME + CHAMPION_FILE_NAME)
        with open(plain, "w") as fh:
            fh.write("{}")
        check(_fingerprint(plain)[0] == FINGERPRINT_KIND_FILE,
              "a regular file must still fingerprint as a hashed file")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_both_scan_helpers_refuse_a_root_that_is_not_a_path():
    """Neither scan helper may quietly return nothing for a bad root: os.walk
    yields nothing there, so a guard or a pin built on one would pass while
    watching no path at all. Both raise ValueError naming what arrived."""
    for helper in (_route_write_fingerprints, _every_path_under):
        for bad in ("", None, 7, b"/tmp"):
            try:
                helper(bad)
            except ValueError as exc:
                check(repr(bad) in str(exc),
                      f"{helper.__name__} must name what arrived: {exc}")
            else:
                check(False, f"{helper.__name__} accepted {bad!r}")


def test_write_surface_pin_sees_a_file_the_name_guard_ignores():
    """The write-surface pin in
    test_module_tree_guard_catches_a_promote_that_escaped_its_fixture is only
    worth its line if the scan behind it reports a write the name-scoped guard
    skips. Proved here on a throwaway folder: a file the promote route never
    writes shows up in _every_path_under and not in _route_write_fingerprints,
    so a promote that grew that write would fail the pin instead of passing
    it."""
    tmp, root, champion_path, _log_path = _guard_fixture()
    try:
        unwatched = os.path.join(os.path.dirname(champion_path), UNWATCHED_FILE_NAME)
        with open(unwatched, "w") as fh:
            fh.write("{}")

        everything = _every_path_under(root)
        check(os.path.relpath(unwatched, root) in everything,
              f"the wider scan must report an unwatched file: {sorted(everything)}")
        check(os.path.relpath(champion_path, root) in everything,
              f"the wider scan must report the champion record too: {sorted(everything)}")

        watched = _route_write_fingerprints(root)
        check(unwatched not in watched,
              f"the name-scoped guard skips it, which is what the pin covers: {sorted(watched)}")
        check(champion_path in watched,
              f"the name-scoped guard must still hold the champion record: {sorted(watched)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_root_the_promote_route_writes_is_inside_the_fixture():
    """The redirect itself, asserted before any route runs: both champion
    paths the promote route builds must resolve inside the fixture, and the
    real module inprocess folder must not be one of them."""
    client, tmp, state = _client_with_project()
    try:
        root = os.path.abspath(appmod._inprocess_root())
        check(root == os.path.abspath(_fixture_root(tmp)),
              f"_inprocess_root must point at the fixture, got {root}")
        check(os.path.commonpath([os.path.abspath(tmp), root]) == os.path.abspath(tmp),
              f"the fixture root escaped the temporary folder: {root}")
        check(os.path.commonpath([_MODULE_INPROCESS, root]) != _MODULE_INPROCESS,
              f"the fixture root sits inside the real module tree: {root}")
    finally:
        _close(tmp)


def _guard_fixture():
    """A throwaway folder shaped like the inprocess root, so the guard can be
    exercised without writing a byte into the real module tree.

    Returns:
        (tmp, root, champion_path, log_path): the temporary folder the caller
        removes, the inprocess root inside it, the module-wide champion record
        under _models, and a log file under a run folder standing in for a
        training job that writes while the suite runs.
    """
    tmp = tempfile.mkdtemp(prefix="rounds_guard_")
    root = os.path.join(tmp, "inprocess")
    models_dir = os.path.join(root, MODELS_DIR_NAME)
    os.makedirs(models_dir, exist_ok=True)
    champion_path = os.path.join(models_dir, CHAMPION_FILE_NAME)
    with open(champion_path, "w") as fh:
        json.dump({"run_dir": "/somewhere/run_already_promoted"}, fh)
    logs_dir = os.path.join(root, CONCURRENT_JOB_RUN_NAME, "ablation", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "A1_replicate_train.log")
    with open(log_path, "w") as fh:
        fh.write("epoch 40\n")
    return tmp, root, champion_path, log_path


def test_module_tree_guard_ignores_a_concurrent_job():
    """Half one of the guard: a training job writing its own logs, weights and
    folders into the inprocess tree is not the promote route, so none of it may
    read as an escaped write. This is the false positive that failed the suite
    on 2026-09-16, when the guard compared the whole tree."""
    tmp, root, champion_path, log_path = _guard_fixture()
    try:
        before = _route_write_fingerprints(root)
        check(champion_path in before,
              f"the guard must watch the module-wide champion record: {sorted(before)}")

        with open(log_path, "a") as fh:
            fh.write("epoch 41\n")
        with open(os.path.join(os.path.dirname(log_path), "A2_replicate_train.log"), "w") as fh:
            fh.write("epoch 1\n")
        weights_dir = os.path.join(root, CONCURRENT_JOB_RUN_NAME, "ablation", "weights")
        os.makedirs(weights_dir, exist_ok=True)
        with open(os.path.join(weights_dir, "last.pt"), "wb") as fh:
            fh.write(b"\x00" * 8)

        noise = _fingerprint_diff(before, _route_write_fingerprints(root))
        check(not noise, f"a concurrent job's own files must not read as an escaped write: {noise}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_module_tree_guard_reports_every_champion_write():
    """Half two of the guard, so the scoping above can never make it vacuous:
    an overwritten champion record (the 2026-09-05 bug), the temp file the
    atomic write goes through, a fixture project folder appearing where no
    fixture belongs, and a removed record are each reported."""
    tmp, root, champion_path, _log_path = _guard_fixture()
    try:
        before = _route_write_fingerprints(root)
        with open(champion_path, "w") as fh:
            json.dump({"run_dir": "/tmp/a_folder_that_is_gone"}, fh)
        overwritten = _fingerprint_diff(before, _route_write_fingerprints(root))
        check(overwritten == [champion_path],
              f"an overwritten champion record must be reported: {overwritten}")

        before = _route_write_fingerprints(root)
        temp_record = champion_path + ATOMIC_TMP_SUFFIX
        with open(temp_record, "w") as fh:
            fh.write("{}")
        stray_project = os.path.join(root, FIXTURE_PROJECT_NAME)
        os.makedirs(stray_project, exist_ok=True)
        appeared = _fingerprint_diff(before, _route_write_fingerprints(root))
        check(appeared == sorted([temp_record, stray_project]),
              f"the temp record and a stray fixture project must both be reported: {appeared}")

        before = _route_write_fingerprints(root)
        os.remove(temp_record)
        vanished = _fingerprint_diff(before, _route_write_fingerprints(root))
        check(vanished == [temp_record], f"a removed champion record must be reported: {vanished}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_module_tree_guard_catches_a_promote_that_escaped_its_fixture():
    """The whole point of the guard, proved end to end: with _inprocess_root
    left pointing at a folder that is not the fixture (the 2026-09-05 defect),
    POST /api/step/7/promote writes its module-wide champion record there and
    _route_write_fingerprints names exactly that file. Both roots are
    temporary folders, so the real module tree is never written to."""
    tmp = tempfile.mkdtemp(prefix="rounds_escape_")
    try:
        stand_in_root = os.path.join(tmp, "stand_in_module_inprocess")
        os.makedirs(stand_in_root, exist_ok=True)
        project_dir = os.path.join(tmp, "fixture", FIXTURE_PROJECT_NAME)
        os.makedirs(project_dir, exist_ok=True)
        state = _make_project(project_dir)
        # The defect itself: the redirect points somewhere other than the
        # fixture, so the route's second write leaves the project.
        appmod._inprocess_root = lambda: stand_in_root
        client = appmod.create_app().test_client()
        appmod.current_project = state

        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_escape")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_escape", "run_dir": run_dir,
            "map50_95_M": 0.415, "gate_map": "pass",
        }])

        before = _route_write_fingerprints(stand_in_root)
        project_before = _every_path_under(project_dir)
        response = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        check(response.status_code == 200, f"promote failed: {response.get_json()}")
        escaped = _fingerprint_diff(before, _route_write_fingerprints(stand_in_root))
        expected = [os.path.join(stand_in_root, MODELS_DIR_NAME, CHAMPION_FILE_NAME)]
        check(escaped == expected,
              f"the guard must name the escaped champion record, got {escaped}")

        # The name-scoped guard is only as wide as the route's write surface,
        # so pin that surface here, on folders no other process touches. The
        # route writes at two roots and both are pinned, because pinning only
        # the inprocess root leaves the other write unwatched: a promote that
        # starts writing something else fails one of these checks and forces
        # ROUTE_WRITTEN_FILE_NAMES to be widened with it.
        # Root one, the inprocess root the route was handed.
        written = _every_path_under(stand_in_root)
        surface = {MODELS_DIR_NAME, os.path.join(MODELS_DIR_NAME, CHAMPION_FILE_NAME)}
        check(written == surface,
              f"promote wrote more into the inprocess root than the guard watches: "
              f"{sorted(written - surface)}")
        # Root two, the run's own ledger folder, which for a project run is
        # the project's step 6 folder.
        ledger_record = os.path.join(os.path.relpath(step6_dir, project_dir),
                                     CHAMPION_FILE_NAME)
        appeared = _every_path_under(project_dir) - project_before
        check(appeared == {ledger_record},
              f"promote wrote more into the project than its own champion record: "
              f"{sorted(appeared - {ledger_record})}")
    finally:
        appmod.current_project = None
        appmod._inprocess_root = _REAL_INPROCESS_ROOT
        shutil.rmtree(tmp, ignore_errors=True)


def test_rounds_empty_when_no_ledger():
    client, tmp, state = _client_with_project()
    try:
        r = client.get("/api/step/7/rounds")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        check(data.get("rounds") == [], f"expected empty rounds list: {data}")
        check(data.get("champion") is None, f"expected no champion: {data}")
    finally:
        _close(tmp)


def test_rounds_parses_rows_and_casts_numerics():
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run1")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "at": "2026-07-09T10:00:00-04:00", "run_name": "run1",
            "run_dir": run_dir, "base_model": "yolo11m-seg.pt", "split_pinned": "true",
            "n_train": 100, "n_valid": 20, "n_test": 10,
            "map50_M": 0.55, "map50_95_M": 0.42, "recall_M": 0.61, "precision_M": 0.70,
            "map50_B": 0.60, "map50_95_B": 0.45,
            "gate_map": "pass", "gate_class": "pass", "gate_recall": "pass",
            "per_class_json": "rounds/round_1_per_class.json",
        }])
        r = client.get("/api/step/7/rounds")
        data = r.get_json()
        check(r.status_code == 200, f"status {r.status_code}: {data}")
        rows = data.get("rounds") or []
        check(len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}")
        row = rows[0]
        check(row["round"] == 1 and isinstance(row["round"], int), f"round not cast to int: {row}")
        check(row["n_train"] == 100 and isinstance(row["n_train"], int), f"n_train not cast: {row}")
        check(abs(row["map50_95_M"] - 0.42) < 1e-9 and isinstance(row["map50_95_M"], float),
              f"map50_95_M not cast to float: {row}")
        check(row["gate_map"] == "pass", f"gate_map should pass through as string: {row}")
        check(row["run_dir"] == run_dir, f"run_dir mismatch: {row}")
    finally:
        _close(tmp)


def test_promote_writes_champion_json():
    """Both writes (the run's own ledger folder and the module-wide record)
    happen inside the fixture, and no champion record anywhere in the real
    module inprocess folder appeared, vanished or changed while they did."""
    real_before = _route_write_fingerprints(_MODULE_INPROCESS)
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_good")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "at": "2026-07-09T10:00:00-04:00", "run_name": "run_good",
            "run_dir": run_dir, "base_model": "yolo11m-seg.pt", "split_pinned": "true",
            "n_train": 100, "n_valid": 20, "n_test": 10,
            "map50_M": 0.55, "map50_95_M": 0.415, "recall_M": 0.61, "precision_M": 0.70,
            "map50_B": 0.60, "map50_95_B": 0.45,
            "gate_map": "pass", "gate_class": "pass", "gate_recall": "pass",
            "per_class_json": "rounds/round_1_per_class.json",
        }])

        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code == 200, f"promote failed: {data}")
        check(data.get("ok") is True, f"expected ok: {data}")
        champ = data.get("champion") or {}
        check(champ.get("run_dir") == os.path.abspath(run_dir), f"champion run_dir mismatch: {champ}")
        check(champ.get("run_name") == "run_good", f"champion run_name mismatch: {champ}")
        check(abs(champ.get("score") - 0.415) < 1e-9 and champ.get("family") == "mask",
              f"a segmenter's champion score is its mask mAP50-95: {champ}")
        check(champ.get("missing") is False, f"the promoted run is on disk: {champ}")
        check(str(champ.get("promoted_at") or "").endswith("-04:00"), f"promoted_at is ISO AST: {champ}")

        # Write 1: beside the run's own rounds ledger.
        champion_path = os.path.join(step6_dir, CHAMPION_FILE_NAME)
        check(os.path.isfile(champion_path), "champion.json was not written beside the ledger")
        with open(champion_path) as f:
            on_disk = json.load(f)
        check(on_disk.get("run_dir") == champ["run_dir"] and on_disk.get("metric_family") == "mask"
              and abs(on_disk.get("map50_95_M") - 0.415) < 1e-9 and on_disk.get("note") == ""
              and on_disk.get("promoted_at") == champ["promoted_at"],
              f"on-disk champion.json is the family-aware record: {on_disk}")

        # Write 2: the module-wide record the runs card reads. It must be the
        # fixture's own _models folder, never the real module tree.
        module_path = _module_champion_path(tmp)
        check(os.path.isfile(module_path), f"module-wide champion.json not written at {module_path}")
        with open(module_path) as f:
            module_record = json.load(f)
        check(module_record == on_disk, f"the two records must be identical: {module_record} vs {on_disk}")

        # GET /rounds must now surface the same champion.
        r2 = client.get("/api/step/7/rounds")
        data2 = r2.get_json()
        check(data2.get("champion", {}).get("run_dir") == champ["run_dir"],
              f"GET /rounds champion mismatch: {data2}")
    finally:
        _close(tmp)
    real_after = _route_write_fingerprints(_MODULE_INPROCESS)
    escaped = _fingerprint_diff(real_before, real_after)
    check(not escaped,
          "the promote route wrote a champion record into the real module "
          f"tree at {_MODULE_INPROCESS}: {escaped[:MAX_ESCAPED_PATHS_REPORTED]}")


def test_rounds_route_labels_both_champion_scopes():
    """The Rounds block is handed two records under two names: the module
    champion (the promoted model) and this project's own gate baseline (what
    evaluate_run.py measured its gate columns against). Before 2026-09-16 it
    was handed the second and called it the champion, which is how one window
    could show two."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_good")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_good", "run_dir": run_dir,
            "map50_95_M": 0.415, "gate_map": "pass",
        }])

        before = client.get("/api/step/7/rounds").get_json() or {}
        check(before.get("champion") is None and before.get("gate_baseline") is None,
              f"nothing is promoted yet: {before}")
        check(before.get("champion_scope") == "module"
              and before.get("gate_baseline_scope") == "project",
              f"both scopes are named even when both records are empty: {before}")

        check(client.post("/api/step/7/promote", json={"run_dir": run_dir}).status_code == 200,
              "promote failed")
        after = client.get("/api/step/7/rounds").get_json() or {}
        check((after.get("champion") or {}).get("run_dir") == os.path.abspath(run_dir),
              f"the module champion: {after.get('champion')}")
        check((after.get("gate_baseline") or {}).get("run_dir") == os.path.abspath(run_dir),
              f"the project's own gate baseline: {after.get('gate_baseline')}")
        check((after.get("champion") or {}).get("metric_family") == "mask",
              f"the record is normalised, so the family and score keys are always present: "
              f"{after.get('champion')}")
    finally:
        _close(tmp)


def test_promote_over_a_failing_gate_writes_nothing_until_it_is_confirmed():
    """A run whose own evaluation did not pass its gates can still be
    promoted, but only as a stated decision. Neither champion record may be
    written while the confirmation is missing, and no record anywhere in the
    real module tree may move."""
    real_before = _route_write_fingerprints(_MODULE_INPROCESS)
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_regressed")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_regressed", "run_dir": run_dir,
            "map50_95_M": 0.31, "gate_map": "fail", "gate_class": "pass",
            "gate_recall": "pass",
        }])

        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        body = r.get_json() or {}
        check(r.status_code == 409, f"the refusal is a conflict, not a server error: {r.status_code} {body}")
        check(body.get("needs_confirmation") is True, f"and says what is missing: {body}")
        check("says fail" in str(body.get("error") or ""), f"the verdict is stated: {body}")
        check(not os.path.isfile(os.path.join(step6_dir, CHAMPION_FILE_NAME)),
              "no ledger record is written while the confirmation is missing")
        check(not os.path.isfile(_module_champion_path(tmp)),
              "and no module-wide record either")

        r2 = client.post("/api/step/7/promote",
                         json={"run_dir": run_dir, "confirm_gate": True})
        body2 = r2.get_json() or {}
        check(r2.status_code == 200, f"the confirmed promotion goes through: {r2.status_code} {body2}")
        check(body2.get("scopes_written") == ["project", "module"],
              f"and says which records it wrote: {body2.get('scopes_written')}")
        with open(_module_champion_path(tmp)) as fh:
            record = json.load(fh)
        check(record.get("gate", {}).get("gate_map") == "fail",
              f"the record keeps the verdict it was promoted over: {record}")
    finally:
        _close(tmp)
    escaped = _fingerprint_diff(real_before, _route_write_fingerprints(_MODULE_INPROCESS))
    check(not escaped,
          "the promote route wrote a champion record into the real module "
          f"tree at {_MODULE_INPROCESS}: {escaped[:MAX_ESCAPED_PATHS_REPORTED]}")


def test_promote_rejects_run_without_weights():
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_no_weights", with_best=False)
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_no_weights", "run_dir": run_dir,
            "map50_95_M": 0.5, "gate_map": "pass",
        }])
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code >= 400, f"expected error status, got {r.status_code}: {data}")
        check("weights" in str(data.get("error", "")).lower(), f"expected a weights-related error: {data}")
        check(not os.path.isfile(os.path.join(step6_dir, CHAMPION_FILE_NAME)),
              "champion.json must not be written when weights are missing")
        check(not os.path.isfile(_module_champion_path(tmp)),
              "the module-wide champion.json must not be written when weights are missing")
    finally:
        _close(tmp)


def test_promote_rejects_run_without_eval_row():
    """A run with weights but no rounds.csv row at all (never evaluated) must
    be rejected: promotion needs a basis for "the best"."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_unevaluated", with_best=True)
        # No rounds.csv written at all.
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        data = r.get_json()
        check(r.status_code >= 400, f"expected error status, got {r.status_code}: {data}")
        check("evaluat" in str(data.get("error", "")).lower(),
              f"expected an 'evaluate first' style error: {data}")
        check(not os.path.isfile(_module_champion_path(tmp)),
              "the module-wide champion.json must not be written for an unevaluated run")
    finally:
        _close(tmp)


def test_list_runs_reports_champion_run_dir():
    """Task 9's step-8 run-picker default: /api/step/6/list_runs surfaces
    champion_run_dir + per-run is_champion once a champion is promoted."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_good")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_good", "run_dir": run_dir,
            "map50_95_M": 0.415, "gate_map": "pass",
        }])
        r = client.post("/api/step/7/promote", json={"run_dir": run_dir})
        check(r.status_code == 200, f"promote failed: {r.get_json()}")

        r2 = client.get("/api/step/6/list_runs")
        data2 = r2.get_json()
        check(r2.status_code == 200, f"list_runs failed: {data2}")
        check(data2.get("champion_run_dir") == os.path.abspath(run_dir),
              f"champion_run_dir mismatch: {data2}")
        matching = [x for x in data2.get("runs", []) if x["name"] == "run_good"]
        check(matching and matching[0]["is_champion"] is True,
              f"run_good should be flagged is_champion: {data2}")
    finally:
        _close(tmp)


def test_list_runs_reports_the_champion_task():
    """Defect 17: the "Continue from a previous run" group is filtered by the
    training task, and the champion option needs the champion's own task to be
    filtered with the rest. A champion whose folder is gone reports no task."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_det", task="detect")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_det", "run_dir": run_dir,
            "map50_95_B": 0.21, "gate_map": "pass",
        }])
        check(client.post("/api/step/7/promote", json={"run_dir": run_dir}).status_code == 200,
              "promote failed")

        data = client.get("/api/step/6/list_runs").get_json()
        check(data.get("champion_task") == "detect",
              f"the champion's task must ride with champion_run_dir: {data}")
        rows = {x["name"]: x for x in data.get("runs", [])}
        check(rows["run_det"]["task"] == "detect" and rows["run_det"]["family"] == "box",
              f"each run carries its own task and family: {rows}")

        shutil.rmtree(run_dir)
        gone = client.get("/api/step/6/list_runs").get_json()
        check(gone.get("champion_task") == "",
              f"a champion whose folder is gone reports no task, not a guess: {gone}")
    finally:
        _close(tmp)


def _promote_then_delete(client, state, tmp):
    """Promote a detector run, then remove its folder, leaving a champion
    record that names a run nobody can read. Returns the vanished run_dir."""
    step6_dir = state["steps"]["6"]["dir"]
    run_dir = _make_run(step6_dir, "run_det", task="detect")
    _write_rounds_csv(step6_dir, [{
        "round": 1, "run_name": "run_det", "run_dir": run_dir,
        "map50_95_B": 0.21, "gate_map": "pass",
    }])
    check(client.post("/api/step/7/promote", json={"run_dir": run_dir}).status_code == 200,
          "promote failed")
    shutil.rmtree(run_dir)
    return run_dir


def test_all_runs_champion_whose_folder_is_gone_reads_as_missing():
    """A champion record naming a folder that no longer exists must say so.
    Before this, the record read as a plain champion, task_for_run_dir fell
    back to its default and every detector run in the table was labelled
    "not comparable: different metric family".

    The key set below is the payload contract the runs card reads: the five
    record keys, missing, and scope. scope joined the view on 2026-09-16, when
    the module-wide champion and a project's own gate baseline stopped sharing
    the name "champion", and its value is asserted as well as its presence
    because this route reads the module-wide record and a card handed the
    project one would rank every run against the wrong baseline."""
    client, tmp, state = _client_with_project()
    try:
        gone_run = _promote_then_delete(client, state, tmp)
        data = client.get("/api/step/6/all_runs").get_json() or {}
        champ = data.get("champion") or {}
        check(champ.get("run_dir") == os.path.abspath(gone_run), f"champion record still read: {champ}")
        check(champ.get("missing") is True, f"the champion must report itself missing: {champ}")
        check(set(champ) == {"run_dir", "run_name", "promoted_at", "score", "family", "missing",
                             "scope"},
              f"champion keys are the contract plus missing and scope: {sorted(champ)}")
        check(champ.get("scope") == "module",
              f"the runs card is told it is showing the module champion: {champ}")
    finally:
        _close(tmp)


def test_all_runs_comparability_is_unknown_against_a_missing_champion():
    """Comparability against a champion nobody can read is null (unknown),
    never false with a reason, because the reason would be a guess."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        _make_run(step6_dir, "run_seg", task="segment")
        _promote_then_delete(client, state, tmp)
        rows = (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []
        check(rows, "the surviving run must still be listed")
        for row in rows:
            check(row["comparable_to_champion"] is None,
                  f"comparability against a missing champion is unknown: {row}")
            check(row["comparability_reason"] == "",
                  f"a missing champion is never a stated reason: {row}")
    finally:
        _close(tmp)


def test_all_runs_champion_on_disk_is_not_missing():
    """The other side of the same flag: a champion whose folder is there
    reports missing False, and its own row stays comparable to itself."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_det", task="detect")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "run_name": "run_det", "run_dir": run_dir,
            "map50_95_B": 0.21, "gate_map": "pass",
        }])
        check(client.post("/api/step/7/promote", json={"run_dir": run_dir}).status_code == 200,
              "promote failed")
        data = client.get("/api/step/6/all_runs").get_json() or {}
        check((data.get("champion") or {}).get("missing") is False,
              f"a champion on disk is not missing: {data.get('champion')}")
        rows = {row["run_name"]: row for row in data.get("runs") or []}
        check(rows["run_det"]["champion"] is True and rows["run_det"]["comparable_to_champion"] is True,
              f"the champion row is comparable to itself: {rows}")
    finally:
        _close(tmp)


def _write_evaluation(state, run_name, family="mask", map50=0.51, map50_95=0.33):
    """An evaluation folder on disk for one run, the way evaluate_run.py
    leaves it: eval_<run_name> under the project's step 7 folder, holding a
    metrics.json with the family's own overall scores."""
    out_dir = os.path.join(state["steps"]["7"]["dir"], f"eval_{run_name}")
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "generated_at": "2026-09-05T11:22:33-04:00",
        "metrics": {
            "metric_family": family,
            "overall": {f"{family}_map50": map50, f"{family}_map50_95": map50_95},
            "per_class": {}, "per_class_instances": {}, "absent_classes": [], "class_names": {},
        },
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(doc, fh)
    return out_dir


def test_all_runs_reads_a_score_from_the_evaluation_folder_without_a_ledger_row():
    """S1: a run evaluated before the ledger existed (no rounds.csv row)
    still has metrics.json on disk. The row carries those numbers and says
    the score came from the evaluation, not the ledger, so the card can
    explain why the run cannot be promoted."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        _make_run(step6_dir, "run_unledgered", task="segment")
        _write_evaluation(state, "run_unledgered")
        rows = {r["run_name"]: r for r in (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []}
        row = rows["run_unledgered"]
        check(abs(row["map50"] - 0.51) < 1e-9 and abs(row["map50_95"] - 0.33) < 1e-9,
              f"the evaluation's own numbers must show: {row}")
        check(row["evaluated_at"] == "2026-09-05T11:22:33-04:00",
              f"evaluated_at comes from the evaluation: {row}")
        check(row["score_source"] == "evaluation",
              f"the row must say where the score came from: {row}")
    finally:
        _close(tmp)


def test_all_runs_prefers_the_ledger_row_over_the_evaluation_folder():
    """The ledger stays the authority: a run with both reads its ledger row,
    which is the record promotion is allowed on."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        run_dir = _make_run(step6_dir, "run_both", task="segment")
        _write_rounds_csv(step6_dir, [{
            "round": 1, "at": "2026-09-04T09:00:00-04:00", "run_name": "run_both",
            "run_dir": run_dir, "map50_M": 0.61, "map50_95_M": 0.42, "gate_map": "pass",
        }])
        _write_evaluation(state, "run_both", map50=0.51, map50_95=0.33)
        rows = {r["run_name"]: r for r in (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []}
        row = rows["run_both"]
        check(abs(row["map50"] - 0.61) < 1e-9 and abs(row["map50_95"] - 0.42) < 1e-9,
              f"the ledger row wins: {row}")
        check(row["evaluated_at"] == "2026-09-04T09:00:00-04:00", f"the ledger stamp wins: {row}")
        check(row["score_source"] == "ledger", f"the row must name the ledger as the source: {row}")
    finally:
        _close(tmp)


def test_all_runs_reports_no_score_source_for_a_run_nobody_evaluated():
    """A run with neither a ledger row nor an evaluation folder keeps null
    scores and names no source, and a damaged metrics.json reads the same
    way rather than raising."""
    client, tmp, state = _client_with_project()
    try:
        step6_dir = state["steps"]["6"]["dir"]
        _make_run(step6_dir, "run_fresh", task="segment")
        _make_run(step6_dir, "run_damaged", task="segment")
        damaged = os.path.join(state["steps"]["7"]["dir"], "eval_run_damaged")
        os.makedirs(damaged, exist_ok=True)
        with open(os.path.join(damaged, "metrics.json"), "w") as fh:
            fh.write("{not json at all")
        rows = {r["run_name"]: r for r in (client.get("/api/step/6/all_runs").get_json() or {}).get("runs") or []}
        for name in ("run_fresh", "run_damaged"):
            check(rows[name]["map50"] is None and rows[name]["map50_95"] is None,
                  f"{name} has no score: {rows[name]}")
            check(rows[name]["evaluated_at"] is None, f"{name} has no evaluation stamp: {rows[name]}")
            check(rows[name]["score_source"] == "", f"{name} names no score source: {rows[name]}")
    finally:
        _close(tmp)


if __name__ == "__main__":
    print("TASK 9 -- rounds ledger routes + champion promotion (orchestrator)\n")
    print("GUARD HELPERS:")
    run(test_the_hang_bound_fires_on_a_call_that_outruns_it)
    run(test_fingerprint_never_opens_a_path_that_is_not_a_regular_file)
    run(test_both_scan_helpers_refuse_a_root_that_is_not_a_path)
    run(test_write_surface_pin_sees_a_file_the_name_guard_ignores)
    print("FIXTURE ISOLATION:")
    run(test_every_root_the_promote_route_writes_is_inside_the_fixture)
    run(test_module_tree_guard_ignores_a_concurrent_job)
    run(test_module_tree_guard_reports_every_champion_write)
    run(test_module_tree_guard_catches_a_promote_that_escaped_its_fixture)
    print("GET /rounds:")
    run(test_rounds_empty_when_no_ledger)
    run(test_rounds_parses_rows_and_casts_numerics)
    print("POST /promote:")
    run(test_promote_writes_champion_json)
    run(test_rounds_route_labels_both_champion_scopes)
    run(test_promote_over_a_failing_gate_writes_nothing_until_it_is_confirmed)
    run(test_promote_rejects_run_without_weights)
    run(test_promote_rejects_run_without_eval_row)
    print("STEP 8 PICKER DEFAULT:")
    run(test_list_runs_reports_champion_run_dir)
    run(test_list_runs_reports_the_champion_task)
    print("MISSING CHAMPION:")
    run(test_all_runs_champion_whose_folder_is_gone_reads_as_missing)
    run(test_all_runs_comparability_is_unknown_against_a_missing_champion)
    run(test_all_runs_champion_on_disk_is_not_missing)
    print("SCORE SOURCE:")
    run(test_all_runs_reads_a_score_from_the_evaluation_folder_without_a_ledger_row)
    run(test_all_runs_prefers_the_ledger_row_over_the_evaluation_folder)
    run(test_all_runs_reports_no_score_source_for_a_run_nobody_evaluated)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
