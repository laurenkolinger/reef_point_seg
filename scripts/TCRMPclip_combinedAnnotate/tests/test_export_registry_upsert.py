# CA/tests/test_export_registry_upsert.py
"""export_batch upserts every mask of an exported frame (accepted AND
rejected) into the cross-project canonical mask registry, non-fatally.
Run with env/bin/python.
"""
import os, sys, tempfile
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SCRIPTS)

import export_yolo as EY
from _reefreview.mask_registry import MaskRegistry

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1


def _img(d, name):
    p = os.path.join(d, name)
    Image.new("RGB", (64, 48), (1, 2, 3)).save(p)
    return p


d = tempfile.mkdtemp(prefix="exp_reg_")
reg_dir = tempfile.mkdtemp(prefix="exp_reg_manifest_")
os.environ['TCRMP_MASK_REGISTRY_DIR'] = reg_dir
# The orchestrator stamps TCRMP_PROJECT_ID (project.json's UNPREFIXED 'id')
# into os.environ for the annotator apps; the hook must prefer that over the
# run_-prefixed dir-derived id so it agrees with the label-coverage matrix's
# lookup key (_matrix/builder.py groups registry rows by project.json's id).
os.environ['TCRMP_PROJECT_ID'] = '20260101_demo'

fn = "TCRMP20201022_clip_SCP_T101.jpeg"
img_path = _img(d, fn)
# run_* segment in the export_dir path so project_id derives deterministically
# in the no-env fallback case (see the second block below).
export_dir = os.path.join(d, "run_20260101_demo", "step5_segmentImages")
os.makedirs(export_dir, exist_ok=True)

segs = {
    fn: {
        "image_path_abs": img_path, "image_width": 64, "image_height": 48,
        "masks": [
            {
                "id": 0, "status": "accepted", "species": "PA", "category": "Coral",
                "source_type": "manual_click", "source_x": 12, "source_y": 34,
                "review": False,
                "rle": {"counts": "abc123", "size": [48, 64]},
                "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]],
                "image_path_abs": img_path,
            },
            {
                "id": 1, "status": "rejected", "species": "OFRA", "category": "Coral",
                "source_type": "manual_click", "source_x": 40, "source_y": 20,
                "review": False,
                "rle": {"counts": "def456", "size": [48, 64]},
                "polygon_norm": [[0.5, 0.5, 0.6, 0.5, 0.6, 0.6]],
                "image_path_abs": img_path,
            },
        ],
    },
}

class_map = {}
stats = EY.export_batch(segs, export_dir, class_map, symlink=False)

check(isinstance(stats, dict), "export_batch still returns a stats dict")
check(stats.get("exported_images") == 1, f"exported_images=1, got {stats.get('exported_images')}")
check(stats.get("exported_masks") == 1, f"exported_masks=1 (only the accepted mask), got {stats.get('exported_masks')}")

reg = MaskRegistry(root=reg_dir)
rows = reg.rows()
by_species_status = {(r.get('species'), r.get('status')): r for r in rows}

accepted_row = by_species_status.get(('PA', 'accepted'))
check(accepted_row is not None, f"registry gained accepted PA row (rows={rows})")
if accepted_row:
    # UN-PREFIXED: must equal TCRMP_PROJECT_ID exactly (no run_ prefix), since
    # that is the same unprefixed id the matrix keys registry rows by (it
    # reads project.json's 'id' field, which never carries the run_ prefix).
    check(accepted_row.get('project_id') == '20260101_demo',
          f"accepted row project_id == TCRMP_PROJECT_ID, un-prefixed (got {accepted_row.get('project_id')!r})")
    check(accepted_row.get('project_name') == '20260101_demo',
          f"accepted row project_name defaults to project_id (got {accepted_row.get('project_name')!r})")

rejected_row = by_species_status.get(('OFRA', 'rejected'))
check(rejected_row is not None, f"registry gained rejected OFRA row (rows={rows})")

check(len(rows) == 2, f"exactly 2 registry rows (accepted + rejected), got {len(rows)}")

# --- Rejected-only frame: no accepted masks at all on this frame. This mask
# is also review-flagged, so it is a "review-only" frame in combinedAnnotate
# terms and previously hit `if has_review: continue` (or, in segmentImages,
# the unconditional `if not accepted: continue`), returning to the top of the
# loop BEFORE reaching _upsert_registry_for_frame. Before the fix, a frame
# like this never got upserted and a rejected mask silently vanished from the
# registry. The hook must now run before that guard so this frame's rejected
# mask still lands in the registry with status='rejected'.
#
# This block also doubles as the no-env fallback test: TCRMP_PROJECT_ID is
# deliberately absent here, so the hook must fall back to
# _project_id_from_dir(export_dir), which must strip the run_ prefix off the
# matched dir segment (run_20260101_rejonly -> 20260101_rejonly) to match
# project.json's unprefixed 'id' convention.
os.environ.pop('TCRMP_PROJECT_ID', None)
d2 = tempfile.mkdtemp(prefix="exp_reg_rejonly_")
reg_dir2 = tempfile.mkdtemp(prefix="exp_reg_rejonly_manifest_")
os.environ['TCRMP_MASK_REGISTRY_DIR'] = reg_dir2

fn2 = "TCRMP20201022_clip_SCP_T202.jpeg"
img_path2 = _img(d2, fn2)
export_dir2 = os.path.join(d2, "run_20260101_rejonly", "step5_segmentImages")
os.makedirs(export_dir2, exist_ok=True)

segs2 = {
    fn2: {
        "image_path_abs": img_path2, "image_width": 64, "image_height": 48,
        "masks": [
            {
                "id": 0, "status": "rejected", "species": "MCAV", "category": "Coral",
                "source_type": "manual_click", "source_x": 5, "source_y": 6,
                "review": True,
                "rle": {"counts": "ghi789", "size": [48, 64]},
                "polygon_norm": [[0.2, 0.2, 0.4, 0.2, 0.4, 0.4]],
                "image_path_abs": img_path2,
            },
        ],
    },
}

class_map2 = {}
stats2 = EY.export_batch(segs2, export_dir2, class_map2, symlink=False)

check(isinstance(stats2, dict), "export_batch still returns a stats dict (rejected-only case)")
check(stats2.get("exported_masks", 0) == 0,
      f"exported_masks=0 (no accepted masks were written to YOLO labels), got {stats2.get('exported_masks')}")

reg2 = MaskRegistry(root=reg_dir2)
rows2 = reg2.rows()
by_species_status2 = {(r.get('species'), r.get('status')): r for r in rows2}

rejected_only_row = by_species_status2.get(('MCAV', 'rejected'))
check(rejected_only_row is not None,
      f"rejected-only frame's mask still reaches the registry with status='rejected' (rows={rows2})")
if rejected_only_row:
    check(rejected_only_row.get('project_id') == '20260101_rejonly',
          "no-env fallback: run_20260101_rejonly -> 20260101_rejonly, un-prefixed "
          f"(got {rejected_only_row.get('project_id')!r})")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
