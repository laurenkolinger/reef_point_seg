"""Reading and writing a project's label ledger (label_provenance.csv).

Module: _reefreview.ledger_io
Purpose: the one implementation of how the ledger is locked, replaced and
    checked, shared by the combined annotator, Edit Masks, Step 5 and the
    expert-ID importer, so no tool writes it a different way.
Inputs: an export folder, per-frame outcome dicts (see provenance.py).
Outputs: <export_dir>/label_provenance.csv, plus the lock file
    <export_dir>/label_provenance.csv.lock beside it.

Rules:
  * Every read-modify-write runs inside ledger_lock(): a thread lock plus an
    fcntl.flock on the lock file, so two threads, or the annotator and the
    importer in another process, never lose each other's rows.
  * The file is swapped in atomically: a unique temp file in the same folder,
    fsync, chmod 0664, os.replace.
  * Owner group. step4test, step4loop, step5 and edit all describe the same
    segmentations.json and recompute every found and pending label on the
    frame (expert ones included, from the masks' expert_id and review fields),
    so an owner-group write drops ALL of the frame's rows, whatever source
    wrote them, before the new rows go in. A label that left the frame
    therefore leaves the ledger. Any other writer (the importer's
    `expertids`) replaces only the labels it writes.
  * The lock wait is bounded (LEDGER_LOCK_TIMEOUT_S). A stuck holder makes
    the writer raise LedgerLockTimeout; its caller keeps the work queued.
  * A blank label is never written.
"""

import contextlib
import csv
import fcntl
import os
import tempfile
import threading
import time

LEDGER_FILE_NAME = 'label_provenance.csv'
LEDGER_LOCK_NAME = 'label_provenance.csv.lock'
CSV_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']

# Group writable, like the rest of a project's files (the importer and the
# annotators run as different processes).
LEDGER_FILE_MODE = 0o664

# The tools that all edit one project's segmentations.json. One owner group:
# a label removed by any of them must leave the ledger whichever tool wrote it.
OWNER_GROUP_SOURCES = frozenset({'step4test', 'step4loop', 'step5', 'edit'})

# Longest a writer waits for the ledger lock before giving up, so a stuck
# holder can never freeze editing; the caller keeps its frames queued.
LEDGER_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.05


class LedgerLockTimeout(OSError):
    """The ledger lock was not free within LEDGER_LOCK_TIMEOUT_S."""


_THREAD_LOCK = threading.RLock()
_held = threading.local()


@contextlib.contextmanager
def ledger_lock(export_dir):
    """Hold the ledger's write lock for this folder (re-entrant per thread).

    Parameters:
        export_dir: the project's export folder.

    Raises:
        OSError: the folder or lock file cannot be created or locked.
    """
    depth = getattr(_held, 'depth', 0)
    if depth:
        _held.depth = depth + 1
        try:
            yield
        finally:
            _held.depth -= 1
        return
    os.makedirs(export_dir, exist_ok=True)
    lock_path = os.path.join(export_dir, LEDGER_LOCK_NAME)
    deadline = time.monotonic() + LEDGER_LOCK_TIMEOUT_S
    if not _THREAD_LOCK.acquire(timeout=LEDGER_LOCK_TIMEOUT_S):
        raise LedgerLockTimeout(f'ledger lock for {export_dir} not free within '
                                f'{LEDGER_LOCK_TIMEOUT_S:g}s (another thread holds it)')
    try:
        with open(lock_path, 'a') as lock_file:
            try:
                os.chmod(lock_path, LEDGER_FILE_MODE)
            except OSError:
                pass  # a lock file another user created: it is already usable
            while True:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LedgerLockTimeout(
                            f'ledger lock {lock_path} not free within '
                            f'{LEDGER_LOCK_TIMEOUT_S:g}s (another process holds it)')
                    time.sleep(_LOCK_POLL_S)
            _held.depth = 1
            try:
                yield
            finally:
                _held.depth = 0
                fcntl.flock(lock_file, fcntl.LOCK_UN)
    finally:
        _THREAD_LOCK.release()


def read_rows(export_dir):
    """Every row of the ledger as a list of dicts; [] when there is no file.

    Raises:
        OSError / csv.Error: the file exists but cannot be read; callers
        report it rather than overwrite a ledger they could not read.
    """
    path = os.path.join(export_dir, LEDGER_FILE_NAME)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_rows_atomically(export_dir, rows):
    """Write `rows` as the whole ledger through a unique temp file.

    The temp file is flushed to disk before os.replace swaps it in, so a crash
    leaves the old ledger or the new one, never half of one. The finished file
    is chmod 0664. On any failure the temp file is removed and the error
    propagates. Callers hold ledger_lock().
    """
    path = os.path.join(export_dir, LEDGER_FILE_NAME)
    fd, tmp = tempfile.mkstemp(dir=export_dir, prefix='.label_provenance.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, '') for k in CSV_HEADER})
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, LEDGER_FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # nothing left to clean; the original error is re-raised
        raise


def _is_owner_write(source, outcomes):
    """True when the writer is one of the owner-group tools."""
    if source is None:
        source = next((o.get('source') for o in outcomes.values() if o.get('source')), None)
    return source in OWNER_GROUP_SOURCES


def write_frames(export_dir, frames, project_id, source=None,
                 protect_sources=frozenset()):
    """Replace several frames' rows in ONE read-modify-write of the ledger.

    Parameters:
        export_dir: the project's export folder.
        frames: {basename: {label: outcome dict}}; an empty outcome dict
            clears the frame's rows when the writer is an owner-group tool.
        project_id: written to every new row.
        source: the writer ('step4test', 'edit', 'expertids', ...). Defaults
            to the source of the first outcome of each frame. An owner-group
            writer replaces every row of the frame, whatever source wrote it;
            any other writer replaces only the labels it writes.
        protect_sources: sources whose rows are never dropped or overwritten
            (the self-repair passes 'expertids', so it cannot undo an import).
            A label that already has a protected row is not rewritten.

    Returns:
        True when the file changed on disk, False when it already held
        exactly these rows (nothing is written then).

    Raises:
        LedgerLockTimeout: the ledger lock stayed busy past the bound.
    """
    with ledger_lock(export_dir):
        old = read_rows(export_dir)
        rows = list(old)
        for basename, outcomes in frames.items():
            protected = {r.get('label') for r in rows
                         if r.get('basename') == basename and r.get('source') in protect_sources}
            incoming = {c for c in outcomes if str(c or '').strip() and c not in protected}
            owner = _is_owner_write(source, outcomes)
            rows = [r for r in rows
                    if not (r.get('basename') == basename
                            and r.get('source') not in protect_sources
                            and (owner or r.get('label') in incoming))]
            for code in sorted(incoming):
                o = outcomes[code]
                rows.append({
                    'basename': basename, 'label': code, 'outcome': o['outcome'],
                    'source': o['source'], 'reviewer': o.get('reviewer', ''),
                    'at': o['at'], 'project_id': project_id,
                })
        rows.sort(key=lambda r: (r.get('basename', ''), r.get('label', '')))
        if _normalized(rows) == _normalized(old):
            return False
        write_rows_atomically(export_dir, rows)
        return True


def _normalized(rows):
    """Rows reduced to the header columns, for comparing two ledgers."""
    return [{k: r.get(k, '') for k in CSV_HEADER} for r in rows]


def frames_with_drift(export_dir, expected, agree_sources=frozenset()):
    """Basenames whose ledger rows disagree with the outcomes stored for them.

    Parameters:
        export_dir: the project's export folder.
        expected: {basename: {label: outcome dict}} as segmentations.json
            holds them (an empty dict for a scrapped frame).

    Returns:
        Sorted basenames where a stored label has no ledger row with the same
        outcome, or an owner-group row exists for a label that is no longer
        stored. A row from another source (the importer's) that already
        carries the label and outcome counts as agreement.
    """
    by_frame = {}
    for r in read_rows(export_dir):
        by_frame.setdefault(r.get('basename'), []).append(r)
    drifted = []
    for basename, outcomes in expected.items():
        rows = by_frame.get(basename, [])
        have = {(r.get('label'), r.get('outcome')) for r in rows}
        stored = {c for c in outcomes if str(c or '').strip()}
        agreed = {r.get('label') for r in rows if r.get('source') in agree_sources}
        missing = any((c, outcomes[c].get('outcome')) not in have and c not in agreed
                      for c in stored)
        stale = any(r.get('source') in OWNER_GROUP_SOURCES and r.get('label') not in stored
                    for r in rows)
        if missing or stale:
            drifted.append(basename)
    return sorted(drifted)
