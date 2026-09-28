"""
Module: project_manager
Purpose: create, load, save, auto-link and reset pipeline projects, and own the
    single contract for how a project's state file reaches disk. Every write of
    project.json made through this module is serialised by an exclusive lock,
    written to a uniquely named temporary file in the same directory, flushed
    and fsynced, then moved onto project.json with os.replace. A reader
    therefore sees either the whole previous file or the whole new one, never a
    truncated or empty one, even when two students click at the same moment or
    a process is killed mid-write.
Inputs: a project directory holding project.json (or a base directory for a new
    project), plus the in-memory project state dict the orchestrator mutates.
Outputs: <project_dir>/project.json, the numbered step directories, and an
    empty lock file .project.json.lock beside project.json.
Limits of the lock (2026-09-16): fcntl.flock is advisory, so it constrains only
    the writers that take it. Every write inside this module takes it, and a
    writer in any other process can take it through exclusive_project_lock or
    update_project_file. Two writers outside this module do not take it yet, so
    a save can still be lost to one of them, and nothing in this module can
    prevent that from its own side:
      /mnt/rip/vicarius_drive/vicarius_ui_os/module_views.py, function
          _stamp_project_access (line 1674), which rewrites project.json in
          place on every project open: no lock, and no atomic replace either,
          so a process killed mid-write leaves the file empty or half written;
      /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/
          scripts/TCRMPclip_combinedAnnotate/src/custom_imports.py, function
          record_in_project (line 257), which appends an import batch: atomic
          by os.replace, but with no lock and a fixed temporary name.
    Each is one line of work once it imports this module: wrap its
    read-modify-write in update_project_file. Until both do, the cross-process
    guarantee covers only the writers listed above as taking the lock.
"""

import copy
import csv
import glob
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

try:
    import fcntl  # POSIX advisory file locking (this box is Linux)
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
    _HAVE_FCNTL = False

from orchestrator_config import REPO_DIR, DEFAULT_STEP_CONFIGS

STEP_DIRS = {
    "1": "step1_makeAllPoints",
    "2": "step2_recodeSpecies",
    "3": "step3_chooseImages",
    "4": "step4_routeChosenImages",
    "5": "step5_segmentImages",
    "6": "step6_trainModel",
    "7": "step7_evaluateModel",
    "8": "step8_inference",
    # The 4.test combined annotator (replaces Steps 4+5) writes its YOLO export
    # and segmentations/ here. Resolvable via resolve_step_dir(step="step4test").
    # NOT a numbered chain step (absent from STEP_KEYS), so it is not auto-created
    # at project setup; 4.test creates it on first routing/export.
    "step4test": "step4test_combinedAnnotate",
}

STEP_NAMES = {
    "1": "Make All Points",
    "2": "Recode Species",
    "3": "Choose Images",
    "4": "Place Points",
    "5": "Segment Images (SAM3)",
    "6": "Train Model",
    "7": "Evaluate Model",
    "8": "Model Inference",
}

# Ordered list of step keys (used in multiple places to avoid hardcoding 5/6 bumps).
STEP_KEYS = ["1", "2", "3", "4", "5", "6", "7", "8"]

# Atlantic Standard Time, the one clock every VICARIUS surface writes in.
AST = timezone(timedelta(hours=-4))
# Timestamps stored in project.json are ISO 8601 with the AST offset, to the
# second: 2026-09-16T01:32:33-04:00. Sub-second precision says nothing useful
# about a student clicking a button.
TIMESTAMP_PRECISION = "seconds"


def now_ast_iso() -> str:
    """The current time as ISO 8601 in Atlantic Standard Time, to the second.

    Every timestamp this module stamps into project.json (created_at,
    updated_at, completed_at) comes from here, so a stored time always says
    which zone it is in. Before 2026-09-16 this module wrote a naive local
    timestamp, which left every reader guessing: the Bounding Boxes page
    already documents created_at as "ISO 8601 with offset" and converts it to
    AST, and the training-project minter already writes the offset.

    Returns: the timestamp string, for example 2026-09-16T01:32:33-04:00.
    """
    return datetime.now(AST).isoformat(timespec=TIMESTAMP_PRECISION)

# ---------------------------------------------------------------------------
# The project.json write contract (2026-09-16)
# ---------------------------------------------------------------------------
# project.json holds a project's whole state: every step's status, config and
# outputs, current_step, the access log and the training source. The
# orchestrator runs on a threaded Flask server and the VICARIUS launcher writes
# the same file from another process, so two writers overlap whenever two
# students work on one orchestrator or one student clicks twice. Writing in
# place, or through a fixed temporary name, lets those writers interleave and
# destroy the project. Every write therefore goes through save_project (for the
# orchestrator's own in-memory state) or update_project_file (for any other
# writer), both of which hold the exclusive lock and replace the file
# atomically. The lock is advisory, so it protects the project only against
# writers that take it; the module docstring lists the two that do not yet.

# Name of the state file inside every project directory.
PROJECT_FILE_NAME = "project.json"
# Zero-byte lock file beside it; the exclusive lock every writer takes.
PROJECT_LOCK_NAME = ".project.json.lock"
# Temporary files are <TEMP_PREFIX><random><TEMP_SUFFIX> in the project dir.
# Unique per writer, so two concurrent saves cannot share one scratch file.
TEMP_PREFIX = ".project.json."
TEMP_SUFFIX = ".tmp"
# The fixed temporary name this module wrote before 2026-09-16. Swept on the
# same rule as the new ones so an old crash leaves no litter behind.
LEGACY_TEMP_NAME = "project.json.tmp"
# A save takes milliseconds, so a temporary file older than this belongs to a
# process that died mid-write, never to a writer still in flight.
TEMP_FILE_STALE_SECONDS = 3600
# Permission bits for a project.json this module creates. mkstemp makes its
# temporary file 0600, which would otherwise leave the file unreadable to
# anyone but the writer once it is moved into place.
PROJECT_FILE_MODE = 0o644
# Permission bits for the lock file this module creates. It holds no content;
# every writer only needs to open it, so the bits match project.json's.
LOCK_FILE_MODE = 0o644
# Indent for the stored JSON (the file is read by people as well as by code).
JSON_INDENT = 2
# How many times to re-serialise state when another request thread mutates the
# same dict mid-dump. See _serialize_state.
SERIALIZE_ATTEMPTS = 3

# One threading.Lock per project lock file, so two request threads inside the
# orchestrator serialise before they ever reach the cross-process flock. The
# registry holds one small lock per project this process has saved.
_THREAD_LOCKS = {}
# Lock files this process has already reported as unlockable, so the warning is
# printed once per project rather than once per save. See
# _warn_no_cross_process_lock.
_NO_CROSS_PROCESS_LOCK = set()
# Guards both module-level registries above.
_THREAD_LOCKS_GUARD = threading.Lock()
# Per-thread lock depth, so a save nested inside an already held lock on the
# SAME thread proceeds instead of deadlocking on its own flock.
_LOCAL = threading.local()


def _abs_dir(project_dir: str | os.PathLike[str]) -> Path:
    """One project directory as an absolute, normalised path.

    os.path.abspath normalises "." and ".." without resolving symlinks, which
    is what the rest of the orchestrator means by a project_dir; Path.resolve
    would follow links and hand app.py a different string than the one the
    student opened.

    project_dir: the project folder, absolute or relative.
    Returns: that folder as an absolute Path.
    """
    return Path(os.path.abspath(os.fspath(project_dir)))


def project_file_path(project_dir: str | os.PathLike[str]) -> str:
    """Absolute path of a project directory's project.json.

    project_dir: the project folder (a relative path is made absolute).
    Returns: <project_dir>/project.json as a string, because every caller
    (app.py, the annotators, the matrix, the review site) joins and compares
    plain strings.
    """
    return str(_abs_dir(project_dir) / PROJECT_FILE_NAME)


def _lock_file_path(project_dir: str | os.PathLike[str]) -> str:
    """Absolute path of the lock file guarding one project's project.json.

    project_dir: the project folder.
    Returns: <project_dir>/.project.json.lock as a string. It keys the
    process-wide registry of threading locks, so one project must always spell
    it one way.
    """
    return str(_abs_dir(project_dir) / PROJECT_LOCK_NAME)


def _thread_lock_for(lock_path: str) -> threading.Lock:
    """The process-wide threading.Lock for one project lock file.

    lock_path: absolute path from _lock_file_path.
    Returns: the threading.Lock every thread in this process shares for that
    project, created on first use.
    """
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(lock_path)
        if lock is None:
            lock = _THREAD_LOCKS[lock_path] = threading.Lock()
        return lock


def _held_depths() -> dict[str, int]:
    """This thread's {lock_path: depth} map, created on first use.

    Returns: the dict recording which project locks this thread already holds.
    """
    depths = getattr(_LOCAL, "depths", None)
    if depths is None:
        depths = _LOCAL.depths = {}
    return depths


def _open_lock_file(lock_path: str) -> int:
    """Open (creating) one project's lock file, ready for an exclusive flock.

    Opened read-only, because flock takes an exclusive lock through a read-only
    descriptor: a save then needs no more permission on the lock file than a
    read, so a project whose lock file somebody made read-only still saves.
    O_CREAT never truncates, so a writer creating the file at the same moment
    as another cannot wipe a lock somebody is holding.

    lock_path: absolute path from _lock_file_path.
    Returns: the open file descriptor; the caller closes it.
    Raises OSError naming the lock file, what a save does with it, and the
    directory it sits in, which is what fails when the lock file is unreadable
    or the project directory is not writable and the file is not there yet.
    """
    try:
        return os.open(lock_path, os.O_RDONLY | os.O_CREAT, LOCK_FILE_MODE)
    except OSError as exc:
        raise OSError(
            f"Cannot open the project lock file {lock_path}: "
            f"{exc.strerror or exc}. Saving {PROJECT_FILE_NAME} reads that "
            f"file, and creates it when it is missing, so it must be readable "
            f"and {os.path.dirname(lock_path)} must be writable the first "
            f"time.") from exc


def _warn_no_cross_process_lock(lock_path: str, reason: OSError) -> None:
    """Report once, on stderr, that one project has no cross-process lock.

    Whether a filesystem can lock is a property of where the project sits, not
    of this save, so the message is printed once per project instead of once
    per write.

    lock_path: the lock file that could not be locked. reason: the OSError the
    filesystem gave.
    Returns: None.
    """
    with _THREAD_LOCKS_GUARD:
        if lock_path in _NO_CROSS_PROCESS_LOCK:
            return
        _NO_CROSS_PROCESS_LOCK.add(lock_path)
    print(f"project_manager: {lock_path} cannot be locked "
          f"({reason.strerror or reason}), so writes of {PROJECT_FILE_NAME} in "
          f"{os.path.dirname(lock_path)} are serialised inside this process "
          f"only and a second process writing the same project could overwrite "
          f"a save. Move the project to a filesystem that supports file "
          f"locking to get that guarantee back.", file=sys.stderr)


def _take_cross_process_lock(lock_fd: int, lock_path: str) -> bool:
    """Take the exclusive flock that keeps other PROCESSES out.

    A filesystem that cannot lock (an NFS mount with no lock daemon, and some
    SMB and FUSE mounts, which answer ENOLCK) must not fail a student's save:
    the threading.Lock still serialises this orchestrator's own writers and the
    write is still atomic, so the only thing lost is exclusion against a second
    process. That loss is reported once and the save goes on.

    lock_fd: an open descriptor on the lock file. lock_path: its path, for the
    report.
    Returns: True when the cross-process lock is held, False when this
    filesystem cannot provide one.
    """
    if not _HAVE_FCNTL:
        return False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except OSError as exc:
        _warn_no_cross_process_lock(lock_path, exc)
        return False
    return True


def _release_lock_file(lock_fd: int, held: bool) -> None:
    """Drop one project's cross-process lock and close its descriptor.

    Closing the descriptor releases the flock by itself, so a failing LOCK_UN
    is not worth raising over: the close in the finally does the same job, and
    an exception raised here would travel out of a finally block and hide
    whatever the caller was already handling.

    lock_fd: the descriptor from _open_lock_file. held: True when
    _take_cross_process_lock actually took the lock.
    Returns: None.
    """
    try:
        if held and _HAVE_FCNTL:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    except OSError:
        pass  # closing the descriptor below releases the lock anyway
    finally:
        os.close(lock_fd)


@contextmanager
def exclusive_project_lock(project_dir: str | os.PathLike[str]) -> Iterator[None]:
    """Hold the exclusive lock on one project's project.json for the block.

    Two locks, always taken in this order: a threading.Lock, so request threads
    inside one orchestrator serialise, then an fcntl.flock on
    .project.json.lock, so a second process serialises too. Public, because
    flock binds only the writers that take it: a writer outside this module
    (the VICARIUS launcher, the combined annotator, a maintenance script) takes
    the same lock by entering this block, or by calling update_project_file.
    Reentrant per thread: a save nested inside a block that already holds the
    lock proceeds under it rather than blocking on itself.

    The threading.Lock is released whatever the body does and whatever the
    release of the file lock does. A threading.Lock left held would hang every
    later save of that project, with no error and no way back short of
    restarting the orchestrator, which is worse than any failure it could be
    covering.

    project_dir: the project folder. Must exist, and be writable the first time
    so the lock file can be created.
    Yields: nothing.
    Raises OSError naming the lock file when it cannot be opened.
    """
    lock_path = _lock_file_path(project_dir)
    depths = _held_depths()
    if depths.get(lock_path):
        depths[lock_path] += 1
        try:
            yield
        finally:
            depths[lock_path] -= 1
        return

    thread_lock = _thread_lock_for(lock_path)
    thread_lock.acquire()
    try:
        depths[lock_path] = 1
        lock_fd = None
        held = False
        try:
            lock_fd = _open_lock_file(lock_path)
            held = _take_cross_process_lock(lock_fd, lock_path)
            yield
        finally:
            depths[lock_path] = 0
            if lock_fd is not None:
                _release_lock_file(lock_fd, held)
    finally:
        thread_lock.release()


def _remove_quietly(path: str | os.PathLike[str]) -> bool:
    """Delete a temporary file this module owns.

    Used on a failure path, where the error being raised is the one that
    matters, and in the stale sweep, where another writer may have removed the
    same file first.

    path: the file to delete.
    Returns: True when the path is gone afterwards, False when it could not be
    removed (a permission problem in the project directory).
    """
    try:
        os.unlink(path)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _fsync_dir(directory: str | os.PathLike[str]) -> None:
    """Flush a directory entry so a completed os.replace survives a power cut.

    Best effort by design: a filesystem that refuses to fsync a directory is
    not a reason to fail a save whose data is already on disk and in place.

    directory: the directory whose entries changed.
    Returns: None.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # the rename already succeeded; durability is the only loss
    finally:
        os.close(fd)


def _replacement_mode(path: str | os.PathLike[str]) -> int:
    """Permission bits to give the replacement file.

    path: the file about to be replaced.
    Returns: the existing file's bits when it exists, so a save never tightens
    or loosens what the project already had, else PROJECT_FILE_MODE.
    """
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return PROJECT_FILE_MODE


def _write_json_atomic(path: str, text: str) -> None:
    """Replace a file's whole content with text, atomically.

    Writes a uniquely named temporary file in the SAME directory (so the rename
    cannot cross a filesystem), flushes and fsyncs it, moves it onto path with
    os.replace, then fsyncs the directory. A reader sees the old file or the new
    one. A process killed at any point leaves the old file untouched and at
    worst a temporary file, which _sweep_stale_temp_files clears later.

    path: the file to replace. text: its complete new content.
    Returns: None.
    Raises OSError naming path when the write fails; the temporary file is
    removed first, so a failure leaves nothing behind.
    """
    directory = str(Path(os.path.abspath(path)).parent)
    try:
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=TEMP_PREFIX,
                                   suffix=TEMP_SUFFIX)
    except OSError as exc:
        raise OSError(
            f"Failed to write {path}: no temporary file could be created in "
            f"{directory} ({exc.strerror or exc}). The project directory must "
            f"be writable.") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, _replacement_mode(path))
        os.replace(tmp, path)
    except OSError as exc:
        _remove_quietly(tmp)
        raise OSError(f"Failed to write {path}: {exc.strerror or exc}") from exc
    except BaseException:
        # Anything else (a KeyboardInterrupt, a stubbed os.replace in a test)
        # still clears the scratch file before the failure travels on.
        _remove_quietly(tmp)
        raise
    _fsync_dir(directory)


def _is_temp_file_name(name: str) -> bool:
    """True for a name this module uses for a project.json temporary file.

    Matched by name rather than by a glob pattern, because a project folder may
    legitimately hold a glob metacharacter (a dataset called ``PA[2024]`` makes
    one), and a pattern built from such a path matches nothing or, worse, some
    other folder.

    name: one entry name from the project directory, not a path.
    Returns: True for <TEMP_PREFIX><anything><TEMP_SUFFIX> and for the fixed
    legacy name; False for project.json and for the lock file, which share the
    prefix but not the suffix.
    """
    if name == LEGACY_TEMP_NAME:
        return True
    return name.startswith(TEMP_PREFIX) and name.endswith(TEMP_SUFFIX)


def _sweep_stale_temp_files(project_dir: str | os.PathLike[str],
                            max_age_seconds: float = TEMP_FILE_STALE_SECONDS) -> int:
    """Delete project.json temporary files a killed process left behind.

    A save takes milliseconds, so a temporary file older than max_age_seconds
    belongs to a process that died; a live writer's file is never touched. The
    fixed-name legacy temporary file is swept on the same rule. Only regular
    files and symbolic links are ever removed, so a directory wearing one of
    those names is left alone.

    project_dir: the project folder. max_age_seconds: the age at which a
    temporary file counts as abandoned.
    Returns: how many files were removed.
    """
    directory = _abs_dir(project_dir)
    try:
        entries = list(directory.iterdir())
    except OSError:
        return 0  # the project folder vanished or is unreadable: nothing to do
    cutoff = time.time() - max_age_seconds
    removed = 0
    for entry in entries:
        if not _is_temp_file_name(entry.name):
            continue
        try:
            info = entry.lstat()
        except OSError:
            continue  # it vanished under us, or cannot be read: leave it alone
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            continue
        if info.st_mtime > cutoff:
            continue  # young enough to belong to a writer still in flight
        if _remove_quietly(entry):
            removed += 1
    return removed


def _serialize_state(state: Any, path: str) -> str:
    """JSON text for a project state, retried when the state moves mid-dump.

    The exclusive lock serialises WRITERS; it cannot stop another orchestrator
    request thread from adding a key to the same in-memory dict while it is
    being serialised, which raises RuntimeError("dictionary changed size during
    iteration"). Re-dumping takes the next consistent snapshot instead of
    failing a student's save. Every other RuntimeError json.dumps raises says
    the same thing about a different container ("Set changed size during
    iteration"), so the retry covers them all; RecursionError is a RuntimeError
    too but means the opposite, a state too deeply nested to dump at all, and
    is reported as itself rather than retried into a misleading message.

    state: the project dict. path: the file being written, named in errors.
    Returns: the JSON text.
    Raises ValueError naming path when the state cannot be serialised, is
    nested too deeply, or was still changing after SERIALIZE_ATTEMPTS attempts.
    """
    last_error = None
    for _ in range(SERIALIZE_ATTEMPTS):
        try:
            return json.dumps(state, indent=JSON_INDENT)
        except RecursionError as exc:
            raise ValueError(
                f"The project state for {path} is nested too deeply to be "
                f"written as JSON (the limit is {sys.getrecursionlimit()} "
                f"levels): {exc}") from exc
        except RuntimeError as exc:
            last_error = exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The project state for {path} cannot be written as JSON: "
                f"{exc}") from exc
    raise ValueError(
        f"The project state for {path} kept changing while it was being "
        f"written ({SERIALIZE_ATTEMPTS} attempts): {last_error}") from last_error


# ---------------------------------------------------------------------------
# Auto project names (2026-08-26)
# ---------------------------------------------------------------------------
# Projects created by the VICARIUS launcher in auto-naming mode carry
# ``"name_auto": true`` plus ``"initials"`` in project.json. For those projects
# the display name is rebuilt on EVERY save from
#     {INITIALS}_{SPECIES}_{N}img_{YY-MM-DD}
# so the name always reflects who made it, which target labels were chosen
# (Step 3 config.target_species), how many images Step 3 selected, and the
# creation date. The run FOLDER is never renamed (every step stores absolute
# paths); only project.json ``name`` moves. Hand-named legacy projects (no
# name_auto flag) are never touched. This module is the single source of
# truth for the format: the launcher's initial name must equal
# auto_project_name(initials, [], None, created_at), and the orchestrator UI
# asks GET /api/project/name-preview rather than formatting anything itself.

# Species codes shown in full before collapsing to "+{remaining}".
AUTO_NAME_MAX_SPECIES = 3
# Any character outside this set becomes "_" inside a name part.
_AUTO_NAME_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]")
# Step 3 output whose data-row count is the {N}img part.
SELECTED_FRAMES_CSV = "selected_frames.csv"


def _sanitize_name_part(part):
    return _AUTO_NAME_UNSAFE.sub("_", str(part))


def _created_date_str(created_at):
    """YY-MM-DD for an ISO string or datetime; anything unparsable -> today."""
    if isinstance(created_at, datetime):
        dt = created_at
    else:
        try:
            dt = datetime.fromisoformat(str(created_at).strip())
        except (TypeError, ValueError):
            dt = datetime.now(AST)
    return dt.strftime("%y-%m-%d")


def species_from_config(target_species_str):
    """Split a Step 3 ``target_species`` value ("PA, SS,,PA") into a list of
    codes: comma-split, stripped, empties dropped, order preserved, deduped.
    A list/tuple is accepted too (same cleaning, no split)."""
    if target_species_str is None:
        return []
    if isinstance(target_species_str, (list, tuple, set)):
        raw = [str(s) for s in target_species_str]
    else:
        raw = str(target_species_str).split(",")
    out = []
    seen = set()
    for code in (s.strip() for s in raw):
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def count_selected_frames(step3_dir):
    """Number of data rows (header excluded, blank lines ignored) in
    ``{step3_dir}/selected_frames.csv``. None when the file is missing (Step 3
    not run yet, or reset) or cannot be read."""
    if not step3_dir:
        return None
    path = os.path.join(step3_dir, SELECTED_FRAMES_CSV)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, newline="") as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    except (OSError, csv.Error, UnicodeDecodeError):
        # An unreadable or malformed selection csv only costs the name its
        # {N}img part; the count is decoration, never a reason to fail a save.
        return None
    return max(len(rows) - 1, 0)


def auto_project_name(initials, species, n_images, created_at):
    """``{INITIALS}_{SPECIES}_{N}img_{YY-MM-DD}``.

    initials   -> uppercased. species -> iterable of codes joined with "-";
    more than AUTO_NAME_MAX_SPECIES codes shows the first three then
    "+{remaining}" (PA-SS-MC+2); omitted when empty. n_images -> "{N}img",
    omitted when None. created_at -> ISO string or datetime (garbage -> today).
    Each part is sanitized to [A-Za-z0-9._+-]; parts are joined with "_".
    Examples: LO_26-08-26, LO_PA-SS_26-08-26, LO_PA-SS_714img_26-08-26."""
    codes = species_from_config(list(species or []))
    parts = [str(initials or "").strip().upper()]
    if codes:
        if len(codes) > AUTO_NAME_MAX_SPECIES:
            shown = "-".join(codes[:AUTO_NAME_MAX_SPECIES])
            parts.append(f"{shown}+{len(codes) - AUTO_NAME_MAX_SPECIES}")
        else:
            parts.append("-".join(codes))
    if n_images is not None:
        parts.append(f"{int(n_images)}img")
    parts.append(_created_date_str(created_at))
    return "_".join(_sanitize_name_part(p) for p in parts if p)


def initials_for_state(state):
    """Uppercased initials for a project: project.json ``initials``, else the
    initials of the first access_log entry whose action is "create". Empty
    string when neither exists (auto naming is then skipped)."""
    ini = str(state.get("initials") or "").strip()
    if not ini:
        for entry in state.get("access_log") or []:
            if isinstance(entry, dict) and entry.get("action") == "create":
                ini = str(entry.get("initials") or "").strip()
                break
    return ini.upper()


def reviewer_for_state(state):
    """Reviewer identity for annotator launches (spec D2, 2026-08-26): the
    initials of the most recent access_log entry (the launcher appends one
    on create and on every open, so the last entry is whoever opened the
    project last), uppercased. Falls back to initials_for_state (project.json
    initials, else the create entry). '' when nothing is recorded; nothing
    is ever inferred."""
    latest = ""
    for entry in state.get("access_log") or []:
        if not isinstance(entry, dict):
            continue
        ini = str(entry.get("initials") or "").strip()
        if ini:
            latest = ini
    if not latest:
        latest = initials_for_state(state)
    return str(latest or "").strip().upper()


def step3_dir_for_state(state):
    """Step 3 directory: steps.3.dir when recorded, else derived from
    project_dir + STEP_DIRS["3"]; "" when neither is available."""
    step3 = (state.get("steps") or {}).get("3") or {}
    d = step3.get("dir")
    if not d and state.get("project_dir"):
        d = os.path.join(state["project_dir"], STEP_DIRS["3"])
    return d or ""


def n_images_for_state(state):
    """Image count for the auto name: the data-row count of Step 3's
    selected_frames.csv, but ONLY while Step 3 is completed. A running or
    reset Step 3 (or a stale csv next to a re-run) never lends its count to
    the name, so the {N}img part always belongs to the labels Step 3 actually
    finished with."""
    step3 = (state.get("steps") or {}).get("3") or {}
    if step3.get("status") != "completed":
        return None
    return count_selected_frames(step3_dir_for_state(state))


def refresh_auto_name(state):
    """Rebuild ``state["name"]`` for auto-named projects. Returns True when the
    name changed. No-op (False) unless ``state["name_auto"]`` is truthy and the
    project has initials (see initials_for_state)."""
    if not state.get("name_auto"):
        return False
    initials = initials_for_state(state)
    if not initials:
        return False
    step3 = (state.get("steps") or {}).get("3") or {}
    species = species_from_config((step3.get("config") or {}).get("target_species"))
    n_images = n_images_for_state(state)
    new_name = auto_project_name(initials, species, n_images, state.get("created_at"))
    if state.get("name") == new_name:
        return False
    state["name"] = new_name
    return True


def find_projects(projects_root):
    """[(project_id, project_dir)] for every <projects_root>/*/project.json.

    project_id is the project.json 'id' (falls back to the dir name). Used to
    resolve a returned expert CSV's project back to its on-disk step dirs, and
    to enumerate projects for the multi-project review-site dropdown."""
    out = []
    try:
        entries = sorted(os.listdir(projects_root))
    except (FileNotFoundError, NotADirectoryError):
        return out
    for entry in entries:
        project_dir = os.path.join(projects_root, entry)
        if not os.path.isfile(os.path.join(project_dir, PROJECT_FILE_NAME)):
            continue
        try:
            pid = read_project_file(project_dir).get('id') or entry
        except (OSError, ValueError):
            # A project whose project.json is unreadable or corrupt still
            # belongs in the listing, under its folder name: this is an
            # enumeration, and failing it would hide every other project.
            pid = entry
        out.append((pid, project_dir))
    return out


def resolve_step_dir(project_id, projects_root, step="5"):
    """Map a canonical project_id to its <project_dir>/<step dir>, or '' if no
    project under projects_root has that id."""
    if not project_id:
        return ""
    for pid, pdir in find_projects(projects_root):
        if pid == project_id:
            return os.path.join(pdir, STEP_DIRS.get(str(step), ""))
    return ""


def create_project(name, base_dir):
    """Create a new project directory with project.json and step subdirs.

    One folder per project: the folder is named ``YY-MM-DD_{name}`` with NO
    random hash suffix, so a project has a single, stable, predictable home.
    If a folder of that name already exists, raise: the caller should Open the
    existing project instead of minting a duplicate (this is what produced the
    pile of ``{name}_{date}_{hash}`` folders and sent exports to the wrong
    one)."""
    name = name.strip().replace(" ", "_")
    if not name:
        raise ValueError("Project name cannot be empty")

    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    date_str = datetime.now(AST).strftime("%y-%m-%d")
    dir_name = f"{date_str}_{name}"
    project_dir = os.path.join(base_dir, dir_name)
    if os.path.exists(project_dir):
        raise ValueError(
            f"A project folder already exists: {dir_name}. "
            f"Open it instead of creating a new one.")
    os.makedirs(project_dir)

    # Create step subdirs (numbered chain steps only; step4test_combinedAnnotate
    # is created lazily by the 4.test routing/export, not at project setup).
    for s in STEP_KEYS:
        os.makedirs(os.path.join(project_dir, STEP_DIRS[s]), exist_ok=True)

    now = now_ast_iso()
    state = {
        "id": dir_name,
        "name": name,
        "description": "",
        "created_at": now,
        "updated_at": now,
        "project_dir": project_dir,
        "repo_dir": REPO_DIR,
        "current_step": 1,
        "steps": {},
    }

    for s in STEP_KEYS:
        state["steps"][s] = {
            "status": "pending" if s == "1" else "locked",
            "name": STEP_NAMES[s],
            "dir": os.path.join(project_dir, STEP_DIRS[s]),
            "config": copy.deepcopy(DEFAULT_STEP_CONFIGS.get(s, {})),
            "outputs": {},
        }

    save_project(state)
    return state


def read_project_file(project_dir):
    """Read and parse a project directory's project.json, with no reconciliation.

    Every failure names the file in a sentence a student can act on, instead of
    a bare JSONDecodeError traceback out of the orchestrator.

    project_dir: the project folder.
    Returns: the parsed state dict exactly as stored.
    Raises FileNotFoundError when there is no project.json, OSError when it
    cannot be read, and ValueError naming the file when it is empty, not valid
    JSON, or not a JSON object.
    """
    directory = os.path.abspath(str(project_dir))
    path = project_file_path(directory)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No {PROJECT_FILE_NAME} in {directory}")
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON: it is not readable as text "
            f"({exc.reason} at byte {exc.start}). Restore the file from a "
            f"backup or create the project again.") from exc
    except OSError as exc:
        raise OSError(f"Cannot read {path}: {exc.strerror or exc}") from exc

    if not raw.strip():
        raise ValueError(
            f"{path} is empty, so the project's state is gone. Restore the "
            f"file from a backup or create the project again.")
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}). Restore the file from a backup or create the project "
            f"again.") from exc
    if not isinstance(state, dict):
        raise ValueError(
            f"{path} holds a JSON {type(state).__name__}, not the object a "
            f"project's state must be. Restore the file from a backup or "
            f"create the project again.")
    return state


def load_project(project_dir):
    """Load project.json from a project directory, reconcile it, and save it.

    Reading needs no lock: save_project replaces the file atomically, so a
    reader always sees one whole version. The reconciled state is then written
    back through save_project, which takes the lock itself.

    project_dir: the project folder.
    Returns: the reconciled state dict, with project_dir set to the folder it
    was read from.
    Raises FileNotFoundError when there is no project.json, ValueError naming
    the file when it is corrupt, and OSError naming the file when the
    reconciled state cannot be written back.
    """
    project_dir = os.path.abspath(project_dir)
    state = read_project_file(project_dir)

    state["project_dir"] = project_dir

    # Back-compat: older projects (pre-step-6) don't have a steps.6 entry.
    # Graft one on so the UI + orchestrator logic can find it.
    for s in STEP_KEYS:
        if s not in state.get("steps", {}):
            state.setdefault("steps", {})[s] = {
                "status": "locked",
                "name": STEP_NAMES[s],
                "dir": os.path.join(project_dir, STEP_DIRS[s]),
                "config": copy.deepcopy(DEFAULT_STEP_CONFIGS.get(s, {})),
                "outputs": {},
            }

    # Validate completed steps still have outputs
    for s in STEP_KEYS:
        step = state["steps"].get(s, {})
        if step.get("status") == "completed":
            step_dir = os.path.join(project_dir, STEP_DIRS[s])
            if not os.path.isdir(step_dir) or not os.listdir(step_dir):
                step["status"] = "pending"

    # Unlock any step whose predecessor is completed. This catches the case
    # where a step was added later (e.g. step 6 grafted onto a pre-6 project):
    # if step 5 is completed, step 6 should be pending, not stuck locked.
    for i, s in enumerate(STEP_KEYS):
        if i == 0:
            continue
        prev = STEP_KEYS[i - 1]
        if (state["steps"][prev]["status"] == "completed"
                and state["steps"][s]["status"] == "locked"):
            state["steps"][s]["status"] = "pending"

    # Step 8 (inference) only requires a trained model from step 6; evaluating
    # (step 7) is optional. If step 6 is done, unlock step 8 regardless of 7.
    if (state["steps"].get("6", {}).get("status") == "completed"
            and state["steps"].get("8", {}).get("status") == "locked"):
        state["steps"]["8"]["status"] = "pending"

    # 4.test promotion bridge: the combined annotator (non-chain "step4test")
    # replaces Steps 4 + 5, so those chain slots never complete and would leave
    # Step 6 (train) permanently locked. Once Step 3 is done, unlock Step 6
    # directly so the promoted flow can reach it; _run_step6's preflight is the
    # real gate on the 4.test export contents.
    if (state["steps"].get("3", {}).get("status") == "completed"
            and state["steps"].get("6", {}).get("status") == "locked"):
        state["steps"]["6"]["status"] = "pending"

    # Reconcile current_step
    for s in STEP_KEYS:
        if state["steps"][s]["status"] not in ("completed",):
            state["current_step"] = int(s)
            break
    else:
        state["current_step"] = int(STEP_KEYS[-1])

    save_project(state)
    return state


def save_project(state):
    """Write a project's project.json atomically, under the exclusive lock.

    Auto-named projects (name_auto) get their display name rebuilt first, so
    every save path (config PUT, complete_step, reset_step, load_project, quit)
    keeps the name current. The lock serialises concurrent writers, the
    temporary-file-then-os.replace write means a crash can never leave a
    truncated or empty project.json, and any temporary file an earlier crash
    left behind is swept once it is stale.

    state: the in-memory project dict. Must carry "project_dir".
    Returns: the path written.
    Raises TypeError when state is not a dict, ValueError when project_dir is
    missing or the state cannot be serialised, FileNotFoundError when the
    project directory is gone, and OSError naming the file when the lock or the
    write fails (a read-only project directory).
    """
    if not isinstance(state, dict):
        raise TypeError(
            f"save_project needs a project state dict, got "
            f"{type(state).__name__}")
    project_dir = str(state.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError(
            f'Cannot save {PROJECT_FILE_NAME}: the project state has no '
            f'"project_dir", so nothing says which folder it belongs in.')
    if not os.path.isdir(project_dir):
        raise FileNotFoundError(
            f"Cannot save {PROJECT_FILE_NAME}: the project directory "
            f"{project_dir} does not exist.")

    refresh_auto_name(state)
    state["updated_at"] = now_ast_iso()
    path = project_file_path(project_dir)
    with exclusive_project_lock(project_dir):
        _write_json_atomic(path, _serialize_state(state, path))
        _sweep_stale_temp_files(project_dir)
    return path


def update_project_file(project_dir: str | os.PathLike[str],
                        mutate: Callable[[dict], Any]) -> dict:
    """Read, change and rewrite one project.json under the exclusive lock.

    The entry point for a writer that does NOT hold the project's state in
    memory the way the orchestrator does: the VICARIUS launcher stamping an
    access_log entry when somebody opens a project, the combined annotator
    recording an import batch, a maintenance script. It reads the file inside
    the lock, hands the parsed state to mutate, stamps updated_at and writes it
    back atomically, so the change cannot interleave with an orchestrator save,
    cannot lose one, and cannot leave a half-written file behind if the process
    is killed.

    Importing this module from another process needs two things: this folder
    (<clone>/scripts/pipeline_orchestrator) on sys.path, and the orchestrator
    config it imports, which reads <clone>/config/pipeline.yaml and needs PyYAML
    in that process's interpreter.

    project_dir: the project folder. mutate: a callable taking the parsed state
    dict and changing it in place; its return value is ignored.
    Returns: the state as written.
    Raises TypeError when mutate is not callable, FileNotFoundError when the
    folder or its project.json is gone, ValueError naming the file when it is
    corrupt or cannot be serialised, and OSError naming the file when the lock
    or the write fails.
    """
    if not callable(mutate):
        raise TypeError(
            f"update_project_file needs a callable to change the state, got "
            f"{type(mutate).__name__}")
    directory = str(_abs_dir(project_dir))
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"Cannot update {PROJECT_FILE_NAME}: the project directory "
            f"{directory} does not exist.")
    path = project_file_path(directory)
    with exclusive_project_lock(directory):
        state = read_project_file(directory)
        mutate(state)
        state["updated_at"] = now_ast_iso()
        _write_json_atomic(path, _serialize_state(state, path))
        _sweep_stale_temp_files(directory)
    return state


def _step_files(step_dir: str, pattern: str) -> list[str]:
    """Sorted paths in one step directory matching a file-name pattern.

    The directory is escaped before the pattern is joined on, so a project
    folder holding a glob metacharacter (a training set called ``PA[2024]``
    makes one) matches its own files instead of nothing. The wildcard in
    pattern still applies to the file name, which is the point of the call.

    step_dir: the step directory to look in. pattern: a file-name glob such as
    "all_points*.csv".
    Returns: the matching absolute paths, sorted, empty when none match.
    """
    return sorted(glob.glob(os.path.join(glob.escape(step_dir), pattern)))


def auto_link_outputs(state, completed_step):
    """After step N completes, wire its outputs into step N+1 config."""
    cs = str(completed_step)
    ns = str(completed_step + 1)
    if ns not in state["steps"]:
        return

    step_dir = os.path.join(state["project_dir"], STEP_DIRS[cs])
    next_step = state["steps"][ns]

    if cs == "1":
        # Find all_points CSV (may have date suffix)
        ap_files = _step_files(step_dir, "all_points*.csv")
        # Match both the legacy `master_codes*.csv` name and the VICARIUS
        # metadata-library name `tcrmp_species_codes*.csv`. Either may appear
        # in step_dir depending on whether step 1 ran end-to-end or was
        # linked via /api/step/1/link.
        mc_files = _step_files(step_dir, "master_codes*.csv")
        if not mc_files:
            mc_files = _step_files(step_dir, "tcrmp_species_codes*.csv")
        # `step1_link` writes the resolved master_codes path directly into
        # step1.outputs before calling complete_step, which calls
        # auto_link_outputs.
        # Prefer that explicit value when the on-disk glob misses.
        mc_from_outputs = state["steps"]["1"].get("outputs", {}).get("master_codes")
        if ap_files:
            next_step["config"]["all_points"] = ap_files[-1]
            state["steps"][cs]["outputs"]["all_points"] = ap_files[-1]
        mc_path = mc_files[-1] if mc_files else mc_from_outputs
        if mc_path:
            next_step["config"]["master_codes"] = mc_path
            state["steps"][cs]["outputs"]["master_codes"] = mc_path

    elif cs == "2":
        ap_files = _step_files(step_dir, "all_points*.csv")
        mc_files = _step_files(step_dir, "master_codes*.csv")
        if ap_files:
            next_step["config"]["all_points"] = ap_files[-1]
            state["steps"][cs]["outputs"]["all_points_recoded"] = ap_files[-1]
        if mc_files:
            next_step["config"]["master_codes"] = mc_files[-1]
            state["steps"][cs]["outputs"]["master_codes_recoded"] = mc_files[-1]

    elif cs == "3":
        sf = os.path.join(step_dir, "selected_frames.csv")
        if os.path.isfile(sf):
            next_step["config"]["selected_frames"] = sf
            state["steps"][cs]["outputs"]["selected_frames"] = sf
        # Also pass all_points for species enrichment
        ap = state["steps"]["2"].get("outputs", {}).get("all_points_recoded")
        if ap:
            next_step["config"]["all_points"] = ap
        # 4.test replaces Steps 4+5 (non-chain "step4test"), so completing Step 3
        # should make train (Step 6) reachable directly. _run_step6 preflights the
        # 4.test export, so unlocking here cannot train on missing data.
        s6 = state["steps"].get("6")
        if s6 is not None and s6.get("status") == "locked":
            s6["status"] = "pending"

    elif cs == "4":
        next_step["config"]["input_dir"] = step_dir
        state["steps"][cs]["outputs"]["export_dir"] = step_dir

    elif cs == "5":
        # Step 5's export dir contains all_images/, all_labels/, data.yaml,
        # which is exactly what step 6 needs as input.
        next_step["config"]["input_dir"] = step_dir
        state["steps"][cs]["outputs"]["export_dir"] = step_dir

    elif cs == "6":
        # Step 6's runs/ dir holds every training run; both step 7 (evaluate)
        # and step 8 (inference) need it. Step 8 doesn't require step 7 to
        # have run (inference works without evaluating), so unlock both
        # here.
        runs_dir = os.path.join(step_dir, "runs")
        state["steps"][cs]["outputs"]["runs_dir"] = runs_dir
        next_step["config"]["runs_dir"] = runs_dir
        s8 = state["steps"].get("8")
        if s8 is not None:
            s8.setdefault("config", {})["runs_dir"] = runs_dir
            if s8.get("status") == "locked":
                s8["status"] = "pending"

    elif cs == "7":
        # Step 7 doesn't strictly feed step 8 (step 8 picks a model from step
        # 6's runs/ directly), but the path is still wired here.
        next_step["config"]["runs_dir"] = (
            state["steps"].get("6", {}).get("outputs", {}).get("runs_dir", "")
        )

    # Unlock next step
    if next_step["status"] == "locked":
        next_step["status"] = "pending"
    state["current_step"] = completed_step + 1

    save_project(state)


def complete_step(state, step_num):
    """Mark a step as completed, auto-link outputs, unlock next."""
    s = str(step_num)
    state["steps"][s]["status"] = "completed"
    state["steps"][s]["completed_at"] = now_ast_iso()
    auto_link_outputs(state, step_num)
    save_project(state)


def reset_step(state, step_num):
    """Clear step outputs and lock it + all subsequent steps."""
    last = int(STEP_KEYS[-1])
    for s in range(step_num, last + 1):
        ss = str(s)
        if ss not in state["steps"]:
            continue
        step_dir = os.path.join(state["project_dir"], STEP_DIRS[ss])
        # Clear directory contents but keep the dir
        if os.path.isdir(step_dir):
            for item in os.listdir(step_dir):
                path = os.path.join(step_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        state["steps"][ss]["status"] = "locked" if s > step_num else "pending"
        state["steps"][ss]["outputs"] = {}
        state["steps"][ss].pop("completed_at", None)
        state["steps"][ss].pop("started_at", None)

    state["current_step"] = step_num
    save_project(state)
