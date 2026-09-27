"""Tests for snapshot_runs.py (the nightly copy of inprocess/ run data).

Every fixture lives under a fresh tempfile.TemporaryDirectory(); nothing
here ever reads or writes the real inprocess/ tree or the real
/mnt/tear/vega_run_snapshots/. Covers: happy path, hard-linking of
unchanged files across two nights, retention pruning, a mismatch that
clears on retry still producing a finished snapshot, a mismatch on every
retry exhausting the attempts (pruning nothing, exiting nonzero), lock
contention, hostile names (spaces, quotes, unicode, a decoy stamp-named
entry outside dest, a symlink inside dest that pruning must not follow),
a missing source, a dest whose parent (mount) does not exist, an empty
source, rsync exit 24 (vanished source files) logged as a warning rather
than failing, a temp file present on attempt 1 and gone by the retry
recovering once --delete is in the rsync command, --dest name refusal
(and its --allow-dest override), the PROTECTED_SNAPSHOT_NAMES set never
being pruned, and manifest.json surviving a simulated crash between being
written and the rename to the final name. Every retry test passes
verify_retry_wait_seconds=0 so a forced mismatch does not really sleep.

Run: /usr/bin/python3 scripts/tests/test_snapshot_runs.py
"""
import fcntl
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
import snapshot_runs as M  # noqa: E402

_pass = 0
_fail = 0


def check(condition, message):
    global _pass, _fail
    print(("  PASS " if condition else "  FAIL ") + message)
    if condition:
        _pass += 1
    else:
        _fail += 1


def fixed_now(moment):
    """A zero-arg callable snapshot_runs._now_ast can be monkeypatched to,
    so a test controls the stamp instead of racing the wall clock."""
    return lambda: moment


def make_run(root, name, files):
    """Create inprocess/<name>/ with the given {relative_path: bytes}
    files, returning its Path."""
    run_dir = Path(root) / name
    for rel, content in files.items():
        target = run_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return run_dir


# ---------------------------------------------------------------------
# 1. Happy path: a fresh source snapshots cleanly and verifies.
# ---------------------------------------------------------------------
def test_happy_path():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_alpha", {
            "step4test_combinedAnnotate/segmentations/segmentations.json": b'{"a": 1}',
            "project.json": b'{"name": "alpha"}',
        })
        (source / "_mask_registry").mkdir(parents=True)
        (source / "_mask_registry" / "manifest.csv").write_text("id,label\n1,coral\n")

        M._now_ast = fixed_now(datetime(2026, 9, 27, 2, 30, 0, tzinfo=M.AST))
        result = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        check(not result["dry_run"], "happy path is not a dry run")
        check(result["stamp"] == "20260927-023000", "stamp matches the fixed AST moment")
        final = result["final_dir"]
        check(final.is_dir(), "final snapshot directory exists")
        check((final / "run_alpha" / "project.json").read_bytes() == b'{"name": "alpha"}',
              "copied file content matches source")
        check((final / "_mask_registry" / "manifest.csv").exists(), "fixed subdir _mask_registry copied")
        check((final / M.MANIFEST_FILENAME).is_file(), "manifest.json written")
        # project.json and segmentations.json from run_alpha, plus manifest.csv
        # from _mask_registry: all three names are in VERIFY_FILENAMES.
        check(set(result["verified_files"]) == {
            "run_alpha/project.json",
            "run_alpha/step4test_combinedAnnotate/segmentations/segmentations.json",
            "_mask_registry/manifest.csv",
        }, "every VERIFY_FILENAMES match across both directories was content-checked")
        check(not os.access(final / "run_alpha" / "project.json", os.W_OK),
              "copied file is read-only")
        check(not os.access(final, os.W_OK), "final snapshot directory itself is read-only")
        check(not (dest / f"{result['stamp']}{M.PARTIAL_SUFFIX}").exists(),
              ".partial directory renamed away, none left behind")


# ---------------------------------------------------------------------
# 2. Second night: unchanged file hard-links, changed file copies fresh.
# ---------------------------------------------------------------------
def test_second_run_hardlinks_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_beta", {
            "unchanged.txt": b"same both nights",
            "changed.txt": b"night one",
        })

        M._now_ast = fixed_now(datetime(2026, 9, 25, 2, 30, 0, tzinfo=M.AST))
        r1 = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        (source / "run_beta" / "changed.txt").write_bytes(b"night two, much longer content")

        M._now_ast = fixed_now(datetime(2026, 9, 26, 2, 30, 0, tzinfo=M.AST))
        r2 = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        unchanged1 = r1["final_dir"] / "run_beta" / "unchanged.txt"
        unchanged2 = r2["final_dir"] / "run_beta" / "unchanged.txt"
        changed1 = r1["final_dir"] / "run_beta" / "changed.txt"
        changed2 = r2["final_dir"] / "run_beta" / "changed.txt"

        check(unchanged1.stat().st_ino == unchanged2.stat().st_ino,
              "unchanged file is the same inode across two nights (hard-linked)")
        check(changed1.stat().st_ino != changed2.stat().st_ino,
              "changed file got a fresh copy, not a hard link")
        check(changed2.read_bytes() == b"night two, much longer content",
              "changed file's new content is what was copied")


# ---------------------------------------------------------------------
# 3. Retention: 16 finished snapshots, keep=14 -> 14 remain, oldest 2 gone.
# ---------------------------------------------------------------------
def test_retention_keeps_newest_n():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "dest"
        dest.mkdir()
        stamps = [f"202609{day:02d}-023000" for day in range(1, 17)]  # 16 stamps
        for stamp in stamps:
            snap = dest / stamp
            snap.mkdir()
            (snap / "placeholder.txt").write_text("x")

        removed = M.prune_old_snapshots(dest, keep=14)

        remaining = sorted(p.name for p in M.list_finished_snapshots(dest))
        check(len(remaining) == 14, f"14 snapshots remain, found {len(remaining)}")
        check(remaining == stamps[2:], "the two oldest were the ones removed")
        check([p.name for p in removed] == stamps[:2], "prune_old_snapshots reports the removed names")
        for stamp in stamps[:2]:
            check(not (dest / stamp).exists(), f"removed snapshot {stamp} is actually gone")


# ---------------------------------------------------------------------
# 4. Unverified run: a mismatch on EVERY attempt exhausts the retries,
#    prunes nothing, and exits nonzero.
# ---------------------------------------------------------------------
def test_unverified_run_prunes_nothing_and_fails():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_gamma", {"project.json": b'{"name": "gamma"}'})

        # A first good snapshot, so there is something to (wrongly) prune if the
        # bug being guarded against were present.
        M._now_ast = fixed_now(datetime(2026, 9, 20, 2, 30, 0, tzinfo=M.AST))
        M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
        before = sorted(p.name for p in M.list_finished_snapshots(dest))

        # Force every verify attempt to mismatch: the copy's hash always
        # comes back different from the source's, simulating a file that
        # stays corrupted (or a live writer that never settles) across
        # every retry, without depending on a timing race. The .partial
        # suffix is only ever present on the destination side of a
        # comparison.
        real_sha256 = M._sha256_of
        real_run_rsync = M.run_rsync
        rsync_calls = {"n": 0}

        def mismatching_sha256(path):
            return "dest-side-hash" if M.PARTIAL_SUFFIX in str(path) else "source-side-hash"

        def counting_run_rsync(paths, dest_partial, link_dest, log=print):
            rsync_calls["n"] += 1
            return real_run_rsync(paths, dest_partial, link_dest, log=log)

        M._sha256_of = mismatching_sha256
        M.run_rsync = counting_run_rsync

        log_lines = []
        M._now_ast = fixed_now(datetime(2026, 9, 21, 2, 30, 0, tzinfo=M.AST))
        raised = False
        try:
            M.take_snapshot(source, dest, keep=14, verify_retry_wait_seconds=0,
                             log=log_lines.append)
        except M.SnapshotVerificationError:
            raised = True
        finally:
            M._sha256_of = real_sha256
            M.run_rsync = real_run_rsync

        check(raised, "a content mismatch on every attempt raises SnapshotVerificationError")
        check(rsync_calls["n"] == M.VERIFY_MAX_ATTEMPTS,
              f"rsync ran once per attempt ({M.VERIFY_MAX_ATTEMPTS}), ran {rsync_calls['n']} times")
        attempt_lines = [line for line in log_lines if "found a mismatch" in line]
        check(len(attempt_lines) == M.VERIFY_MAX_ATTEMPTS,
              f"every attempt logged its own mismatch line, found {len(attempt_lines)}")
        check(all("project.json" in line for line in attempt_lines),
              "each mismatch line names the file that mismatched")
        after = sorted(p.name for p in M.list_finished_snapshots(dest))
        check(before == after, "the earlier finished snapshot was not pruned")
        check((dest / "20260921-023000.partial").is_dir(),
              "the failed run's .partial directory is left in place for inspection")
        check(not (dest / "20260921-023000").exists(), "no final snapshot was created for the failed run")


# ---------------------------------------------------------------------
# 4b. A mismatch on attempt 1 that clears on attempt 2 still produces a
#     finished snapshot (the retry actually re-copies and re-verifies).
# ---------------------------------------------------------------------
def test_mismatch_clears_on_retry():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_delta2", {"project.json": b'{"name": "delta2"}'})

        real_sha256 = M._sha256_of
        real_run_rsync = M.run_rsync
        rsync_calls = {"n": 0}

        def counting_run_rsync(paths, dest_partial, link_dest, log=print):
            rsync_calls["n"] += 1
            return real_run_rsync(paths, dest_partial, link_dest, log=log)

        def flaky_sha256(path):
            # Lie about the destination-side hash only while rsync has run
            # exactly once (attempt 1's copy); by the time a retry has run
            # rsync a second time, report the real (matching) hash, as a
            # live writer's own rewrite would have settled by then.
            if M.PARTIAL_SUFFIX in str(path) and rsync_calls["n"] < 2:
                return "forced-mismatch-before-retry"
            return real_sha256(path)

        M.run_rsync = counting_run_rsync
        M._sha256_of = flaky_sha256

        log_lines = []
        M._now_ast = fixed_now(datetime(2026, 9, 22, 2, 30, 0, tzinfo=M.AST))
        try:
            result = M.take_snapshot(source, dest, keep=14, verify_retry_wait_seconds=0,
                                      log=log_lines.append)
        finally:
            M._sha256_of = real_sha256
            M.run_rsync = real_run_rsync

        check(not result["dry_run"] and result["final_dir"].is_dir(),
              "a mismatch that clears on retry still produces a finished snapshot")
        check(rsync_calls["n"] == 2, f"rsync ran twice (initial copy + one retry), ran {rsync_calls['n']} times")
        check((result["final_dir"] / "run_delta2" / "project.json").read_bytes()
              == b'{"name": "delta2"}',
              "the finished snapshot has the correct, verified content")
        check(any("attempt 1/3 found a mismatch" in line for line in log_lines),
              "attempt 1's mismatch was logged")
        check(any("project.json" in line for line in log_lines if "found a mismatch" in line),
              "the logged mismatch names the file")
        check(not (dest / "20260922-023000.partial").exists(),
              "no .partial left behind once the retry verified clean")


# ---------------------------------------------------------------------
# 5. Lock contention: a second run refuses while the lock is held.
# ---------------------------------------------------------------------
def test_lock_contention():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_delta", {"project.json": b"{}"})
        dest.mkdir()

        lock_path = dest / M.LOCK_FILENAME
        holder = open(lock_path, "a+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            M._now_ast = fixed_now(datetime(2026, 9, 22, 2, 30, 0, tzinfo=M.AST))
            raised = False
            try:
                M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
            except M.SnapshotLockError:
                raised = True
            check(raised, "a concurrent run refuses with SnapshotLockError")
            check(list_dest_entries_excluding_lock(dest) == [],
                  "the contending run wrote nothing into dest")
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()

        # Lock released: a normal run now succeeds.
        result = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
        check(not result["dry_run"] and result["final_dir"].is_dir(),
              "a run after the lock is released succeeds normally")


def list_dest_entries_excluding_lock(dest):
    return [p.name for p in dest.iterdir() if p.name != M.LOCK_FILENAME]


# ---------------------------------------------------------------------
# 6. Hostile names: spaces, quotes, unicode; a decoy stamp-named entry
#    outside dest; a symlink inside dest that pruning must not follow.
# ---------------------------------------------------------------------
def test_hostile_names():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        hostile = {
            "weird file 'quoted' name.txt": b"space and quote",
            "unicode_éè中文.json": b'{"ok": true}',
            'double"quote.csv': b"a,b\n1,2\n",
        }
        make_run(source, "run_epsilon", hostile)

        # A decoy directory named exactly like a stamp, but living OUTSIDE
        # dest (a sibling). Pruning must never look here, let alone delete it.
        decoy_root = Path(tmp)
        decoy = decoy_root / "20200101-000000"
        decoy.mkdir()
        (decoy / "do_not_touch.txt").write_text("outside dest")

        M._now_ast = fixed_now(datetime(2026, 9, 23, 2, 30, 0, tzinfo=M.AST))
        result = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        for name in hostile:
            copied = result["final_dir"] / "run_epsilon" / name
            check(copied.is_file(), f"hostile filename survived the copy: {name!r}")
            check(copied.read_bytes() == hostile[name], f"hostile filename content intact: {name!r}")

        check(decoy.exists() and (decoy / "do_not_touch.txt").exists(),
              "a stamp-named directory outside dest is untouched")

        # A symlink inside dest, at the top level, that LOOKS like a finished
        # snapshot name but is actually a symlink pointing outside dest.
        outside_target = Path(tmp) / "outside_target"
        outside_target.mkdir()
        (outside_target / "marker.txt").write_text("must survive")
        trap = dest / "20190101-000000"
        trap.symlink_to(outside_target, target_is_directory=True)

        # Sixteen more finished snapshots so retention has something to prune,
        # with the trap symlink sitting among names that sort older than all
        # of them (it would be pruned first if it were mistakenly treated as
        # a real, deletable snapshot).
        for day in range(1, 17):
            stamp = f"202601{day:02d}-000000"
            snap = dest / stamp
            snap.mkdir()
            (snap / "x.txt").write_text("x")

        pruned = M.prune_old_snapshots(dest, keep=14)

        check(trap.is_symlink(), "the trap symlink still exists as a symlink")
        check(os.path.realpath(trap) == os.path.realpath(outside_target),
              "the trap symlink still points at its original target")
        check((outside_target / "marker.txt").exists(), "the symlink's target directory is untouched")
        check(all(p.name != "20190101-000000" for p in pruned),
              "the trap symlink was never selected for pruning")


# ---------------------------------------------------------------------
# 7. Missing source fails clearly.
# ---------------------------------------------------------------------
def test_missing_source():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "does_not_exist"
        dest = Path(tmp) / "dest"
        raised = False
        try:
            M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
        except M.SnapshotError as exc:
            raised = True
            check("does not exist" in str(exc), "error message names the missing source")
        check(raised, "a missing source raises SnapshotError")
        check(not dest.exists(), "nothing was created at dest")


# ---------------------------------------------------------------------
# 8. Dest on a missing mount (nonexistent parent) fails clearly.
# ---------------------------------------------------------------------
def test_dest_parent_missing():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        make_run(source, "run_zeta", {"project.json": b"{}"})
        dest = Path(tmp) / "unmounted_drive" / "sub" / "daily"
        raised = False
        try:
            M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
        except M.SnapshotError as exc:
            raised = True
            check("does not exist" in str(exc) and "mounted" in str(exc),
                  "error message names the missing parent and asks about the mount")
        check(raised, "a dest whose parent is missing raises SnapshotError")


# ---------------------------------------------------------------------
# 9. Empty source: no run_* dirs, no fixed subdirs -> succeeds, empty copy.
# ---------------------------------------------------------------------
def test_empty_source():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        source.mkdir()
        dest = Path(tmp) / "dest"

        M._now_ast = fixed_now(datetime(2026, 9, 24, 2, 30, 0, tzinfo=M.AST))
        result = M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        check(not result["dry_run"], "empty source is still a real (non-dry) run")
        check(result["final_dir"].is_dir(), "an empty snapshot directory was still created")
        check(result["counts"] == {}, "no directories were copied or counted")
        check(result["verified_files"] == [], "nothing to content-verify")


# ---------------------------------------------------------------------
# 10. --keep must be at least 1.
# ---------------------------------------------------------------------
def test_keep_must_be_positive():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "dest"
        dest.mkdir()
        (dest / "20260101-000000").mkdir()
        raised = False
        try:
            M.prune_old_snapshots(dest, keep=0)
        except M.SnapshotError:
            raised = True
        check(raised, "prune_old_snapshots refuses --keep=0")


# ---------------------------------------------------------------------
# 11. discover_source_paths respects include_models.
# ---------------------------------------------------------------------
def test_include_models_flag():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        make_run(source, "run_eta", {"project.json": b"{}"})
        models_dir = source / M.MODELS_SUBDIR
        models_dir.mkdir(parents=True)
        (models_dir / "weights.pt").write_bytes(b"fake weights")

        without = M.discover_source_paths(source, include_models=False)
        with_models = M.discover_source_paths(source, include_models=True)

        check(M.MODELS_SUBDIR not in [p.name for p in without],
              "_models excluded by default")
        check(M.MODELS_SUBDIR in [p.name for p in with_models],
              "_models included when --include-models is passed")


# ---------------------------------------------------------------------
# 12. --dry-run touches nothing.
# ---------------------------------------------------------------------
def test_dry_run_touches_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_theta", {"project.json": b"{}"})

        result = M.take_snapshot(source, dest, keep=14, dry_run=True, log=lambda *_: None)

        check(result["dry_run"], "result reports dry_run True")
        check(not dest.exists(), "dry run created no destination directory at all")


# ---------------------------------------------------------------------
# 13. Stale .partial from a crashed earlier run is swept on the next run.
# ---------------------------------------------------------------------
def test_stale_partial_is_swept():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_iota", {"project.json": b"{}"})
        dest.mkdir()
        stale = dest / "20260101-010000.partial"
        stale.mkdir()
        (stale / "half_copied.txt").write_text("leftover from a crash")
        not_ours = dest / "not_a_stamp_at_all.partial"
        not_ours.mkdir()

        M._now_ast = fixed_now(datetime(2026, 9, 28, 2, 30, 0, tzinfo=M.AST))
        M.take_snapshot(source, dest, keep=14, log=lambda *_: None)

        check(not stale.exists(), "the stale, stamp-shaped .partial was swept away")
        check(not_ours.exists(), "a .partial-suffixed name that is not stamp-shaped was left alone")


# ---------------------------------------------------------------------
# 14. rsync exit 24 (source files vanished mid-transfer) is a logged
#     warning, not a failure; the run proceeds to verification and
#     finishes normally.
# ---------------------------------------------------------------------
def test_vanished_source_exit_24_is_a_warning():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_kappa", {"project.json": b'{"name": "kappa"}'})

        real_subprocess_run = M.subprocess.run

        def rsync_reports_vanished_source(cmd, **kwargs):
            # Let the real rsync actually run (this fixture has nothing to
            # make vanish), then report the exit code rsync itself would
            # give if a source file had disappeared mid-transfer.
            result = real_subprocess_run(cmd, **kwargs)
            return subprocess.CompletedProcess(
                cmd, returncode=M.RSYNC_VANISHED_SOURCE_EXIT_CODE, stdout=result.stdout,
                stderr="rsync: some files vanished before they could be transferred (code 24)",
            )

        M.subprocess.run = rsync_reports_vanished_source
        log_lines = []
        M._now_ast = fixed_now(datetime(2026, 9, 23, 3, 0, 0, tzinfo=M.AST))
        try:
            result = M.take_snapshot(source, dest, keep=14, log=log_lines.append)
        finally:
            M.subprocess.run = real_subprocess_run

        check(not result["dry_run"] and result["final_dir"].is_dir(),
              "exit 24 does not fail the run; a finished snapshot is still produced")
        check(any("exit 24" in line and "vanished" in line for line in log_lines),
              "the vanished-source warning was logged, naming exit code 24")
        check((result["final_dir"] / "run_kappa" / "project.json").is_file(),
              "the file was actually copied despite the reported exit code")


# ---------------------------------------------------------------------
# 15. A temp file present in source on attempt 1, gone by the retry:
#     --delete on the retry rsync removes the stale copy from the
#     .partial so counts match again (reproduces, then clears, the
#     reviewer's "run_a: source has 1, copy has 2").
# ---------------------------------------------------------------------
def test_vanished_temp_file_retry_recovers_with_delete():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        run_dir = make_run(source, "run_a", {
            "real.json": b'{"ok": true}',
            "temp.tmp": b"annotator temp save file",
        })

        real_run_rsync = M.run_rsync
        calls = {"n": 0}

        def wrapper(paths, dest_partial, link_dest, log=print):
            calls["n"] += 1
            result = real_run_rsync(paths, dest_partial, link_dest, log=log)
            if calls["n"] == 1:
                # The annotator cleans up its own temp save file in the
                # window between pass 1's copy and the retry.
                (run_dir / "temp.tmp").unlink()
            return result

        M.run_rsync = wrapper
        log_lines = []
        M._now_ast = fixed_now(datetime(2026, 9, 29, 2, 30, 0, tzinfo=M.AST))
        try:
            result = M.take_snapshot(source, dest, keep=14, verify_retry_wait_seconds=0,
                                      log=log_lines.append)
        finally:
            M.run_rsync = real_run_rsync

        check(not result["dry_run"] and result["final_dir"].is_dir(),
              "the vanished-temp-file case still produces a finished snapshot")
        check(calls["n"] == 2, f"rsync ran twice (initial copy + one retry), ran {calls['n']} times")
        check(any("run_a: source has 1, copy has 2" in line for line in log_lines),
              "attempt 1 reproduced the reviewer's exact count mismatch")
        check(not (result["final_dir"] / "run_a" / "temp.tmp").exists(),
              "--delete removed the stale temp file the retry rsync left behind from pass 1")
        check((result["final_dir"] / "run_a" / "real.json").is_file(),
              "the real file is still present and correct")


# ---------------------------------------------------------------------
# 16. --dest must end in a folder named "daily" unless --allow-dest is
#     passed.
# ---------------------------------------------------------------------
def test_dest_name_refusal():
    with tempfile.TemporaryDirectory() as tmp:
        good_dest = Path(tmp) / "daily"
        bad_dest = Path(tmp) / "not_daily"

        raised = False
        try:
            M.validate_dest_name(bad_dest, allow_dest=False)
        except M.SnapshotError as exc:
            raised = True
            check("daily" in str(exc), "the refusal message names the required folder name")
        check(raised, "a --dest not ending in 'daily' is refused by default")

        no_raise = True
        try:
            M.validate_dest_name(bad_dest, allow_dest=True)
        except M.SnapshotError:
            no_raise = False
        check(no_raise, "--allow-dest lets a non-standard dest name through")

        no_raise2 = True
        try:
            M.validate_dest_name(good_dest, allow_dest=False)
        except M.SnapshotError:
            no_raise2 = False
        check(no_raise2, "a --dest actually named 'daily' is never refused")

        # End to end through main(): refused without --allow-dest (exit 1,
        # nothing written), succeeds with --allow-dest.
        source = Path(tmp) / "inprocess"
        make_run(source, "run_lambda", {"project.json": b"{}"})

        M._now_ast = fixed_now(datetime(2026, 9, 24, 2, 30, 0, tzinfo=M.AST))
        rc_refused = M.main(["--source", str(source), "--dest", str(bad_dest)])
        check(rc_refused == 1, "main() exits 1 for a non-'daily' dest without --allow-dest")
        check(not bad_dest.exists(), "nothing was written when the dest name was refused")

        rc_allowed = M.main(["--source", str(source), "--dest", str(bad_dest), "--allow-dest"])
        check(rc_allowed == 0, "main() succeeds for a non-'daily' dest with --allow-dest")
        check(bad_dest.is_dir(), "the non-standard dest was actually used once allowed")


# ---------------------------------------------------------------------
# 17. A name in PROTECTED_SNAPSHOT_NAMES is never pruned and never counts
#     toward --keep.
# ---------------------------------------------------------------------
def test_protected_snapshot_name_never_pruned():
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "dest"
        dest.mkdir()
        protected_name = next(iter(M.PROTECTED_SNAPSHOT_NAMES))
        protected = dest / protected_name
        protected.mkdir()
        (protected / "placeholder.txt").write_text("made by a different tool, not this script")

        # 15 ordinary finished snapshots plus the protected one; keep=14
        # should remove exactly the single oldest ordinary snapshot and
        # never touch the protected name, which does not count toward keep.
        stamps = [f"202608{day:02d}-023000" for day in range(1, 16)]  # 15 stamps
        for stamp in stamps:
            snap = dest / stamp
            snap.mkdir()
            (snap / "placeholder.txt").write_text("x")

        removed = M.prune_old_snapshots(dest, keep=14)

        check(protected.exists(), "the protected snapshot name still exists")
        check(all(p.name != protected_name for p in removed),
              "the protected name was never a removal candidate")
        remaining_ordinary = sorted(
            p.name for p in M.list_finished_snapshots(dest) if p.name != protected_name
        )
        check(len(remaining_ordinary) == 14,
              f"14 ordinary snapshots remain (protected one is separate), found {len(remaining_ordinary)}")
        check(remaining_ordinary == stamps[1:], "only the single oldest ordinary snapshot was pruned")
        check(len(removed) == 1 and removed[0].name == stamps[0],
              "prune_old_snapshots removed exactly the oldest ordinary snapshot")


# ---------------------------------------------------------------------
# 18. manifest.json is written before the rename to the final name: a
#     crash between the two leaves a .partial that already has its own
#     manifest, never a manifest-less "finished" snapshot. A finished
#     snapshot without a manifest is skipped as a link-dest base.
# ---------------------------------------------------------------------
def test_manifest_written_before_rename_survives_simulated_crash():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "inprocess"
        dest = Path(tmp) / "dest"
        make_run(source, "run_crash", {"project.json": b'{"name": "crash"}'})

        real_rename = Path.rename

        def crashing_rename(self, target):
            raise OSError("simulated crash between write_manifest and rename")

        M._now_ast = fixed_now(datetime(2026, 9, 30, 2, 30, 0, tzinfo=M.AST))
        Path.rename = crashing_rename
        raised = False
        try:
            M.take_snapshot(source, dest, keep=14, log=lambda *_: None)
        except OSError:
            raised = True
        finally:
            Path.rename = real_rename

        check(raised, "the simulated crash during rename propagates")
        partial = dest / "20260930-023000.partial"
        final = dest / "20260930-023000"
        check(partial.is_dir(), "the .partial survives the simulated crash")
        check((partial / M.MANIFEST_FILENAME).is_file(),
              "manifest.json was already written into the .partial before the rename was attempted")
        check(not final.exists(), "no finished (non-.partial) snapshot exists after the crash")

        # A finished-looking directory without a manifest (a crash under
        # some older build of this script, or the one-off snapshot made by
        # another tool) is never chosen as a link-dest base.
        manifestless = dest / "20260929-000000"
        manifestless.mkdir()
        (manifestless / "placeholder.txt").write_text("no manifest here")
        chosen = M.latest_finished_snapshot(dest)
        check(chosen is None,
              "a manifest-less finished snapshot is skipped; none other qualifies as a link-dest base")


def main():
    test_happy_path()
    test_second_run_hardlinks_unchanged()
    test_retention_keeps_newest_n()
    test_unverified_run_prunes_nothing_and_fails()
    test_mismatch_clears_on_retry()
    test_lock_contention()
    test_hostile_names()
    test_missing_source()
    test_dest_parent_missing()
    test_empty_source()
    test_keep_must_be_positive()
    test_include_models_flag()
    test_dry_run_touches_nothing()
    test_stale_partial_is_swept()
    test_vanished_source_exit_24_is_a_warning()
    test_vanished_temp_file_retry_recovers_with_delete()
    test_dest_name_refusal()
    test_protected_snapshot_name_never_pruned()
    test_manifest_written_before_rename_survives_simulated_crash()

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
