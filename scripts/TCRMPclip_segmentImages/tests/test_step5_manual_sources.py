"""Step 5 (segmentImages) counts a merged mask as manual, from the shared list.

Plan 2026-10-15-vega-project-and-label-integrity.md, task L, item 6. Its
provenance.py carried its own manual list without `merged`, so a frame whose
accepted mask was hand-merged read found_ai.

Run: env/bin/python scripts/TCRMPclip_segmentImages/tests/test_step5_manual_sources.py
"""
import csv
import os
import shutil
import stat
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
import provenance as P
from _reefreview.manual_sources import MANUAL_SOURCE_TYPES

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


check(P._MANUAL_SOURCES == MANUAL_SOURCE_TYPES and "merged" in P._MANUAL_SOURCES,
      f"Step 5 provenance uses the shared manual list (got {sorted(P._MANUAL_SOURCES)})")
seg = {"masks": [{"species": "PA", "status": "accepted", "source_type": "merged"}]}
out = P.compute_label_outcomes(seg, ["PA"], "step5", "LO")
check(out.get("PA", {}).get("outcome") == "found_manual",
      f"a merged-only accepted mask is found_manual (got {out})")


# Step 5 writes the ledger like the shared copies: no blank label, one owner
# group, unique temp file, file lock, mode 0664.
seg = {"masks": [{"species": "", "status": "accepted", "source_type": "auto"},
                 {"species": "PA", "status": "accepted", "source_type": "auto"}]}
out = P.compute_label_outcomes(seg, [], "step5", "LO")
check(set(out) == {"PA"}, f"Step 5 skips a blank-species mask (got {set(out)})")
d = tempfile.mkdtemp(prefix="si_ledger_")
try:
    def o(label, source):
        return {label: {"outcome": "found_manual", "source": source, "reviewer": "", "at": "t"}}
    P.write_provenance_csv(d, "F1", {**o("PA", "step4test"), **o("XYZ", "step4test")}, "p")
    P.write_provenance_csv(d, "F1", o("PA", "step5"), "p")
    with open(os.path.join(d, "label_provenance.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    check([(r["label"], r["source"]) for r in rows] == [("PA", "step5")],
          f"a Step 5 write replaces the frame's step4test rows (got {rows})")
    P.write_provenance_csv(d, "F2", {"": o("", "step5")[""]}, "p")
    check(all(r["label"] for r in csv.DictReader(open(os.path.join(d, "label_provenance.csv")))),
          "a blank label is never written")
    mode = stat.S_IMODE(os.stat(os.path.join(d, "label_provenance.csv")).st_mode)
    check(mode == 0o664, f"ledger mode 0664 (got {oct(mode)})")
    check(sorted(os.listdir(d)) == ["label_provenance.csv", "label_provenance.csv.lock"],
          f"a lock file beside the ledger and no fixed .tmp left (got {sorted(os.listdir(d))})")
finally:
    shutil.rmtree(d, ignore_errors=True)

print(f"\n{'FAILED' if _fail else 'ALL PASSED'} ({_fail} failure(s))")
sys.exit(1 if _fail else 0)
