"""A mask whose source is 'merged' records found_manual, not not_found.

VEGA annotation safety net, 2026-09-27 (LO): merge_many (see
_merge_ids in src/app.py) only ever produces a source_type='merged' survivor
from a human operator's own POST /api/image/<fn>/merge_many call, gated on a
named reviewer's initials (_stamp_reviewer). It is always the product of
manual merging of reviewed masks, so provenance.py must count it as manual
work, the same as manual_box and manual_click. Before this fix, an accepted
mask with source_type='merged' fell through compute_label_outcomes' found_*
checks (neither in _MANUAL_SOURCES nor _AI_SOURCES) and, absent an expert ID,
was scored not_found for a target species whose only accepted mask was that
merge.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_merged_outcome_found.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import provenance as P

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def mask(species, status, source_type, **extra):
    m = {"species": species, "status": status, "source_type": source_type}
    m.update(extra)
    return m


# A target species whose ONLY accepted mask is a merge: found_manual, not
# not_found (the bug this test locks against).
seg = {"masks": [mask("PA", "accepted", "merged", merged_from=[1, 2])]}
out = P.compute_label_outcomes(seg, ["PA"], "step4test", "LO")
check(out["PA"]["outcome"] == "found_manual",
      f"merged-only accepted mask -> found_manual, got {out.get('PA')}")

# A merged mask still loses to an expert ID on the SAME code (unchanged
# priority: found_expert beats found_manual regardless of source_type).
em = mask("PA", "accepted", "merged")
em["expert_id"] = {"code": "PA", "mode": "EXPERT", "labeler": "XY"}
out = P.compute_label_outcomes({"masks": [em]}, ["PA"], "step5")
check(out["PA"]["outcome"] == "found_expert",
      "an expert-ID'd merged mask is still found_expert, not found_manual")

# A merged mask sitting alongside an auto/exemplar mask of the same code
# still reads as manual (merge wins over AI-only, same rule as manual_click).
seg = {"masks": [mask("PA", "accepted", "auto"),
                 mask("PA", "accepted", "merged")]}
out = P.compute_label_outcomes(seg, ["PA"], "step5")
check(out["PA"]["outcome"] == "found_manual",
      f"merged wins over a sibling auto mask of the same code, got {out.get('PA')}")

# A rejected merged mask contributes nothing (same rule as every other
# source_type): a target species with no other accepted mask is not_found.
seg = {"masks": [mask("PA", "rejected", "merged")]}
out = P.compute_label_outcomes(seg, ["PA"], "step5")
check(out["PA"]["outcome"] == "not_found",
      f"a rejected merged mask does not count, got {out.get('PA')}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
