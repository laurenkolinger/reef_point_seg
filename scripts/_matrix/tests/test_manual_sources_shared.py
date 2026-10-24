#!/usr/bin/env python
"""A merged mask is hand work everywhere: one shared list of manual sources.

Plan 2026-10-15-vega-project-and-label-integrity.md, task L, item 6. Before
this, only the annotator's provenance.py counted `merged` as manual; the
coverage matrix, the composite builder and its source filter, and the Step 5
provenance each carried their own list without it, so the same accepted merge
read found_manual in the ledger and found_ai in the grid.

Run: env/bin/python scripts/_matrix/tests/test_manual_sources_shared.py
"""
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _reefreview.manual_sources import MANUAL_SOURCE_TYPES  # noqa: E402
from _matrix import blueprint, builder, composite  # noqa: E402
from _matrix.tests._fixture_cell import (  # noqa: E402
    T101, make_image, make_project, mask, seg_entry, write,
)
import json  # noqa: E402

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


check(MANUAL_SOURCE_TYPES >= {"manual_box", "manual_click", "merged"},
      f"shared list holds box, click and merged (got {sorted(MANUAL_SOURCE_TYPES)})")
check(builder._MANUAL_SOURCE_TYPES == MANUAL_SOURCE_TYPES,
      "builder uses the shared list")
check(composite._MANUAL_SOURCE_TYPES == MANUAL_SOURCE_TYPES,
      "composite builder uses the shared list")
check(MANUAL_SOURCE_TYPES <= composite.SOURCE_TYPES,
      f"the composite source filter accepts every manual source (got {sorted(composite.SOURCE_TYPES)})")
check(blueprint.SOURCE_TYPES == composite.SOURCE_TYPES and "merged" in blueprint.SOURCE_TYPES,
      "the blueprint validates a request against the same list, merged included")
check(composite._derived_outcome([{"source_type": "merged"}]) == "found_manual",
      "a merged mask derives found_manual in the composite builder")
check(composite._derived_outcome([{"source_type": "auto"}]) == "found_ai",
      "an auto mask still derives found_ai")

# End to end through the grid: a project with no ledger and no registry rows,
# whose only accepted mask is a merge.
root = tempfile.mkdtemp(prefix="manual_shared_")
try:
    proj = make_project(root, "run_projM_20261015_aaaaaa", "projM", "Project M", target_species="PA")
    raw = os.path.join(proj, "step4test_combinedAnnotate", "routed_input", "raw")
    img = make_image(raw, T101)
    seg = {T101: seg_entry(img, [mask(0, "PA", source_type="merged")])}
    write(os.path.join(proj, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
          json.dumps(seg))
    m = builder.build_matrix(root)
    cell = m["cells"].get(os.path.splitext(T101)[0], {}).get("PA", {})
    check(cell.get("outcome") == "found_manual",
          f"grid cell for a merged-only frame is found_manual (got {cell.get('outcome')})")
finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n{'FAILED' if _fail else 'ALL PASSED'} ({_fail} failure(s))")
sys.exit(1 if _fail else 0)
