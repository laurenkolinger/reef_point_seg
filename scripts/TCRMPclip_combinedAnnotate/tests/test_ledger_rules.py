"""The label ledger (label_provenance.csv) rules in provenance.py.

Plan 2026-10-15-vega-project-and-label-integrity.md, task L:
  3. A mask with a blank species never makes a ledger row.
  4. Writing a frame replaces every row that frame has under the same source,
     so a label that left the frame disappears; other frames and other
     sources are untouched.
  6. The manual sources come from the one shared list, `merged` included.
  8. The ledger temp file has a unique name in the same folder, is flushed
     before the swap, and the finished file is group writable (0664); many
     concurrent writers never crash or leave a temp file behind.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_ledger_rules.py
"""
import csv
import os
import shutil
import stat
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import provenance as P

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def mask(species, status="accepted", source_type="manual_click", **extra):
    m = {"species": species, "status": status, "source_type": source_type}
    m.update(extra)
    return m


def rows(d):
    with open(os.path.join(d, "label_provenance.csv"), newline="") as f:
        return list(csv.DictReader(f))


def leftovers(d):
    """Files in the folder other than the lock file kept beside the ledger."""
    return sorted(n for n in os.listdir(d) if not n.endswith(".lock"))


def out(label, outcome="found_manual", source="step4test"):
    return {label: {"outcome": outcome, "reviewed": True, "source": source,
                    "reviewer": "LO", "at": "2026-10-15T10:00:00-04:00"}}


# ── 3. blank species ───────────────────────────────────────────────────────
res = P.compute_label_outcomes({"masks": [mask(""), mask("  "), mask(None)]}, [], "step4test", "LO")
check(res == {}, f"blank / whitespace / None species make no outcome (got {res})")
res = P.compute_label_outcomes({"masks": [{"status": "accepted", "source_type": "auto"}]}, [], "step4test")
check(res == {}, f"a mask with no species key makes no outcome (got {res})")
res = P.compute_label_outcomes({"masks": [mask(""), mask("PA")]}, ["PA"], "step4test", "LO")
check(set(res) == {"PA"}, f"only the labelled mask yields a row (got {set(res)})")
d = tempfile.mkdtemp(prefix="ledger_blank_")
try:
    P.write_provenance_csv(d, "F1", {"": out("")[""], "PA": out("PA")["PA"]}, "p")
    check([r["label"] for r in rows(d)] == ["PA"], "the writer refuses a blank label row")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ── 4. replace every row of the frame under this source ────────────────────
d = tempfile.mkdtemp(prefix="ledger_replace_")
try:
    both = {**out("PA"), **out("OFRA")}
    P.write_provenance_csv(d, "F1", both, "p")
    P.write_provenance_csv(d, "F2", out("PA"), "p")
    P.write_provenance_csv(d, "F1", out("SS", source="expertids"), "p")
    P.write_provenance_csv(d, "F1", out("PA"), "p", source="step4test")
    got = sorted((r["basename"], r["label"], r["source"]) for r in rows(d))
    check(got == [("F1", "PA", "step4test"), ("F2", "PA", "step4test")],
          f"OFRA left F1 and is gone; F2 stays; the owner write also drops F1's expertids row (got {got})")
    P.write_provenance_csv(d, "F1", {}, "p", source="step4test")
    got = sorted((r["basename"], r["label"]) for r in rows(d))
    check(got == [("F2", "PA")],
          f"an empty outcome set clears the frame's rows for that source (got {got})")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ── 6. one shared list of manual sources ───────────────────────────────────
from _reefreview.manual_sources import MANUAL_SOURCE_TYPES
check("merged" in MANUAL_SOURCE_TYPES and {"manual_box", "manual_click"} <= MANUAL_SOURCE_TYPES,
      f"shared constant holds box, click and merged (got {sorted(MANUAL_SOURCE_TYPES)})")
check(P._MANUAL_SOURCES is MANUAL_SOURCE_TYPES or P._MANUAL_SOURCES == MANUAL_SOURCE_TYPES,
      "provenance.py uses the shared list")

# ── 8. unique temp name, fsync, mode, concurrency ──────────────────────────
d = tempfile.mkdtemp(prefix="ledger_tmp_")
try:
    P.write_provenance_csv(d, "F1", out("PA"), "p")
    mode = stat.S_IMODE(os.stat(os.path.join(d, "label_provenance.csv")).st_mode)
    check(mode == 0o664, f"ledger mode is 0664 (got {oct(mode)})")
    check(leftovers(d) == ["label_provenance.csv"], f"no temp file remains (got {leftovers(d)})")

    errors = []

    def writer(i):
        try:
            for k in range(15):
                P.write_provenance_csv(d, f"T{i}_{k}", out("PA"), "p")
        except Exception as e:  # noqa: BLE001 - the test reports every failure
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(errors == [], f"concurrent writers raise nothing (got {errors[:2]})")
    check(len(rows(d)) == 8 * 15 + 1,
          f"no concurrent writer loses another's row (got {len(rows(d))} of {8 * 15 + 1})")
    check(leftovers(d) == ["label_provenance.csv"],
          f"no temp file left after concurrent writes (got {leftovers(d)})")
    with open(os.path.join(d, "label_provenance.csv"), newline="") as f:
        header = next(csv.reader(f))
    check(header == P.CSV_HEADER, "the ledger is still a valid CSV with its header")

    # A failure while writing leaves the old ledger and no temp file.
    before = open(os.path.join(d, "label_provenance.csv")).read()
    orig_replace = os.replace

    def boom(*a, **k):
        raise OSError("disk full")
    os.replace = boom
    try:
        raised = False
        try:
            P.write_provenance_csv(d, "NEW", out("PA"), "p")
        except OSError:
            raised = True
    finally:
        os.replace = orig_replace
    check(raised, "a failed swap raises a real error")
    check(open(os.path.join(d, "label_provenance.csv")).read() == before,
          "a failed swap leaves the old ledger intact")
    check(leftovers(d) == ["label_provenance.csv"], "a failed swap leaves no temp file")
finally:
    shutil.rmtree(d, ignore_errors=True)


# ── D1. one owner group: a label removed by any tool leaves the ledger ──────
d = tempfile.mkdtemp(prefix="ledger_group_")
try:
    P.write_provenance_csv(d, "F1", {**out("PA"), **out("XYZ")}, "p")
    P.write_provenance_csv(d, "F1", out("PA", source="edit"), "p")
    got = sorted((r["label"], r["source"]) for r in rows(d))
    check(got == [("PA", "edit")],
          f"Edit Masks relabelling XYZ to PA removes the step4test XYZ row (got {got})")
    P.write_provenance_csv(d, "F1", out("SS", source="expertids"), "p")
    P.write_provenance_csv(d, "F1", out("PA", source="step5"), "p")
    got = sorted((r["label"], r["source"]) for r in rows(d))
    check(got == [("PA", "step5")],
          f"an owner-group write drops the frame's expertids rows too, it recomputes every expert label (got {got})")
    P.write_provenance_csv(d, "F1", out("PA", source="expertids", outcome="found_expert"), "p")
    P.write_provenance_csv(d, "F1", out("SS", source="expertids", outcome="pending_expert"), "p")
    got = sorted((r["label"], r["source"]) for r in rows(d))
    check(got == [("PA", "expertids"), ("SS", "expertids")] or got == [("PA", "step5"), ("SS", "expertids")],
          f"an importer write replaces only the labels it writes (got {got})")
    P.write_provenance_csv(d, "F1", out("PA", source="edit"), "p")
    P.write_provenance_csv(d, "F1", out("XYZ", source="expertids", outcome="found_expert"), "p")
    P.write_provenance_csv(d, "F1", out("PA", source="edit"), "p")
    got = sorted((r["label"], r["source"]) for r in rows(d))
    check(got == [("PA", "edit")],
          f"an expert-identified label that left the frame leaves the ledger (got {got})")
    # the heal protects an importer row
    P.write_provenance_csv(d, "F2", out("XYZ", source="expertids", outcome="found_expert"), "p")
    from _reefreview.ledger_io import write_frames, frames_with_drift
    write_frames(d, {"F2": out("XYZ")}, "p", protect_sources=frozenset({"expertids"}))
    got = sorted((r["basename"], r["label"], r["source"]) for r in rows(d) if r["basename"] == "F2")
    check(got == [("F2", "XYZ", "expertids")], f"a protected expertids row is not overwritten (got {got})")
    check(frames_with_drift(d, {"F2": out("XYZ")}, agree_sources=frozenset({"expertids"})) == [],
          "an expertids row for a stored label counts as agreement")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ── D2. a whole queue is one read-modify-write ──────────────────────────────
d = tempfile.mkdtemp(prefix="ledger_batch_")
try:
    calls = []
    real_replace = os.replace

    def counting(a, b):
        calls.append(b)
        return real_replace(a, b)
    os.replace = counting
    try:
        frames = {f"F{i}": out("PA") for i in range(20)}
        changed = P.write_provenance_frames(d, frames, "p")
        again = P.write_provenance_frames(d, frames, "p")
    finally:
        os.replace = real_replace
    check(changed is True and len(rows(d)) == 20, "20 frames written")
    check(len(calls) == 1, f"20 frames cost one swap, and an identical rewrite none (swaps: {len(calls)})")
    check(again is False, "rewriting identical rows reports no change")
    exp = {f"F{i}": out("PA") for i in range(20)}
    exp["F3"] = out("OFRA")
    exp["F4"] = {}
    exp["NEW"] = out("PA")
    check(P.frames_with_ledger_drift(d, exp) == ["F3", "F4", "NEW"],
          f"drift finds changed, cleared and unwritten frames (got {P.frames_with_ledger_drift(d, exp)})")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ── D3. a file lock beside the ledger holds other processes out ─────────────
import subprocess, time
from _reefreview.ledger_io import ledger_lock
d = tempfile.mkdtemp(prefix="ledger_flock_")
try:
    P.write_provenance_csv(d, "F0", out("PA"), "p")
    check(os.path.exists(os.path.join(d, "label_provenance.csv.lock")),
          "the lock file sits beside the ledger")
    hold = subprocess.Popen([sys.executable, "-c", (
        "import fcntl,sys,time;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);"
        "print('held',flush=True);time.sleep(1.5)"), os.path.join(d, "label_provenance.csv.lock")],
        stdout=subprocess.PIPE, text=True)
    hold.stdout.readline()
    t0 = time.time()
    P.write_provenance_csv(d, "F1", out("PA"), "p")
    waited = time.time() - t0
    hold.wait()
    check(waited >= 1.0, f"a writer waits for another process's lock (waited {waited:.2f}s)")
    check(sorted(r["basename"] for r in rows(d)) == ["F0", "F1"], "both rows present after the wait")

    # the importer's prune takes the same lock
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "_expertids"))
    from _expertids import importer as IMP
    P.write_provenance_csv(d, "F2", out("ZZZ", outcome="pending_expert", source="expertids"), "p")
    hold = subprocess.Popen([sys.executable, "-c", (
        "import fcntl,sys,time;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);"
        "print('held',flush=True);time.sleep(1.5)"), os.path.join(d, "label_provenance.csv.lock")],
        stdout=subprocess.PIPE, text=True)
    hold.stdout.readline()
    t0 = time.time()
    IMP._prune_stale_pending_rows(d, {"F2": set()}, P.CSV_HEADER)
    waited = time.time() - t0
    hold.wait()
    check(waited >= 1.0, f"the importer's prune waits for the same lock (waited {waited:.2f}s)")
    check("F2" not in [r["basename"] for r in rows(d)], "the prune still removes the stale pending row")
finally:
    shutil.rmtree(d, ignore_errors=True)


# ── bounded lock wait ───────────────────────────────────────────────────────
import _reefreview.ledger_io as LIO
d = tempfile.mkdtemp(prefix="ledger_timeout_")
try:
    P.write_provenance_csv(d, "F0", out("PA"), "p")
    check(isinstance(LIO.LEDGER_LOCK_TIMEOUT_S, (int, float)) and LIO.LEDGER_LOCK_TIMEOUT_S <= 30,
          "the lock wait is bounded by a named constant")
    old_timeout = LIO.LEDGER_LOCK_TIMEOUT_S
    LIO.LEDGER_LOCK_TIMEOUT_S = 1.0
    hold = subprocess.Popen([sys.executable, "-c", (
        "import fcntl,sys,time;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);"
        "print('held',flush=True);time.sleep(3)"), os.path.join(d, "label_provenance.csv.lock")],
        stdout=subprocess.PIPE, text=True)
    hold.stdout.readline()
    t0 = time.time()
    raised = None
    try:
        P.write_provenance_csv(d, "F1", out("PA"), "p")
    except LIO.LedgerLockTimeout as e:
        raised = e
    waited = time.time() - t0
    hold.wait()
    LIO.LEDGER_LOCK_TIMEOUT_S = old_timeout
    check(raised is not None and isinstance(raised, OSError), "a stuck holder raises LedgerLockTimeout (an OSError)")
    check(0.8 <= waited < 2.5, f"it gives up after the bound, not forever (waited {waited:.2f}s)")
    check([r["basename"] for r in rows(d)] == ["F0"], "nothing was written on timeout")
finally:
    LIO.LEDGER_LOCK_TIMEOUT_S = old_timeout
    shutil.rmtree(d, ignore_errors=True)

print(f"\n{'FAILED' if _fail else 'ALL PASSED'} ({_fail} failure(s))")
sys.exit(1 if _fail else 0)
