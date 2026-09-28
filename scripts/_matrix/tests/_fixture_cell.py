"""Shared fixture builder for the matrix cell / preferences / composite /
stats / routes suites. Not a suite itself: nothing to register.

build_root(root) creates, under a throwaway inprocess root:

  run_projA_20260826_aaaaaa   id projA, flat segmentations.json, creator LO,
                              latest access_log entry AB (opened_by AB),
                              registry rows seeded for its masks
  run_projB_20260826_bbbbbb   id projB, nested 2014/segmentations.json,
                              no registry rows (legacy derivation), creator CD
  run_projC_20260826_cccccc   id projC, ledger only (no segmentations)
  run_junk                    no project.json (ignored)
  _mask_registry/manifest.csv projA rows + test residue + one unknown id

Images (all 64x48, real files under each project's routed_input/raw/):
  T101  PA (projA manual, projB expert) + OFAV (both auto, projB later)
  T102  review-only in projA (pending REVIEW mask)
  T403  reviewed, no masks (negative)
  T404  never reviewed, no masks (skipped)
  T105  accepted mask with no species (blocked, then negative)
"""

import json
import os
import sys

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _reefreview.mask_registry import MaskRegistry, build_registry_record  # noqa: E402

T101 = "TCRMP20140930_clip_SCP_T101.jpeg"
T102 = "TCRMP20140930_clip_SCP_T102.jpeg"
T105 = "TCRMP20140930_clip_SCP_T105.jpeg"
T403 = "TCRMP20141003_clip_SSJ_T403.jpeg"
T404 = "TCRMP20141003_clip_SSJ_T404.jpeg"
T2015 = "TCRMP20151001_clip_BIT_T201.jpeg"

RING_A = [0.10, 0.10, 0.30, 0.10, 0.30, 0.30, 0.10, 0.30]
RING_B = [0.50, 0.50, 0.70, 0.50, 0.70, 0.70]
RING_C = [0.80, 0.80, 0.90, 0.80, 0.90, 0.90]


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def make_image(d, name):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    Image.new("RGB", (64, 48), (3, 2, 1)).save(p)
    return p


def mask(mid, species, status="accepted", source_type="manual_click", sx=10, sy=20,
         rings=None, review=False, reviewer="", expert=False, reviews=None, rle="r"):
    m = {
        "id": mid,
        "species": species,
        "status": status,
        "source_type": source_type,
        "source_x": sx,
        "source_y": sy,
        "rle": {"counts": rle + str(mid) + species, "size": [48, 64]},
        "bbox": [1, 2, 3, 4],
        "area": 100 + mid,
        "polygon_norm": rings if rings is not None else [RING_A],
        "polygon_px": [[int(v * 64) if i % 2 == 0 else int(v * 48) for i, v in enumerate(r)]
                       for r in (rings if rings is not None else [RING_A])],
        "review": review,
    }
    if reviewer:
        m["reviewer"] = reviewer
    if expert:
        m["expert_id"] = {"code": species, "mode": "EXPERT", "labeler": "XY"}
    if reviews:
        m["reviews"] = reviews
    return m


def seg_entry(img_path, masks, reviewed=True, exported=False, processed_at="2026-08-14T10:00:00",
              label_outcomes=None):
    return {
        "image_path": "raw/" + os.path.basename(img_path),
        "image_path_abs": img_path,
        "image_width": 64,
        "image_height": 48,
        "masks": masks,
        "processed_at": processed_at,
        "reviewed": reviewed,
        "exported": exported,
        "label_outcomes": label_outcomes or {},
    }


def make_project(root, run_dir, pid, name, target_species="PA, OFAV", initials="LO",
                 access_log=None, created_at="2026-08-14T09:00:00"):
    project_dir = os.path.join(root, run_dir)
    meta = {
        "id": pid,
        "name": name,
        "created_at": created_at,
        "project_dir": project_dir,
        "initials": initials,
        "access_log": access_log if access_log is not None else [
            {"ts": created_at, "initials": initials, "action": "create", "purpose": "t"},
        ],
        "steps": {"3": {"config": {"target_species": target_species}}},
    }
    write(os.path.join(project_dir, "project.json"), json.dumps(meta, indent=2))
    return project_dir


def build_root(root, seed_registry=True):
    """Build the fixture tree; returns a dict of useful paths and ids."""
    # projA: flat segmentations, opened most recently by AB.
    proj_a = make_project(
        root, "run_projA_20260826_aaaaaa", "projA", "Project A",
        access_log=[
            {"ts": "2026-08-14T09:00:00-04:00", "initials": "lo", "action": "create"},
            {"ts": "2026-08-20T09:00:00-04:00", "initials": "ab", "action": "open"},
        ],
    )
    raw_a = os.path.join(proj_a, "step4test_combinedAnnotate", "routed_input", "raw")
    img_a = {n: make_image(raw_a, n) for n in (T101, T102, T105, T403, T404)}
    seg_a = {
        T101: seg_entry(img_a[T101], [
            mask(0, "PA", source_type="manual_click", sx=10, sy=20, reviewer="LO"),
            mask(1, "OFAV", source_type="auto", sx=30, sy=40, rings=[RING_B, RING_C]),
            mask(2, "PA", status="rejected", source_type="auto", sx=50, sy=60),
        ], processed_at="2026-08-14T10:00:00",
            label_outcomes={"PA": {"outcome": "found_manual", "reviewed": True, "source": "step4test",
                                   "reviewer": "LO", "at": "2026-08-14T10:05:00-04:00"}}),
        T102: seg_entry(img_a[T102], [
            mask(0, "REVIEW", status="pending", review=True, sx=11, sy=21,
                 reviews=[{"reviewer": "AB", "code": "DSTO", "confidence": "high"}]),
        ]),
        T105: seg_entry(img_a[T105], [mask(0, "", status="accepted", sx=12, sy=22)]),
        T403: seg_entry(img_a[T403], []),
        T404: seg_entry(img_a[T404], [], reviewed=False),
    }
    write(os.path.join(proj_a, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
          json.dumps(seg_a, indent=1))

    # projB: nested segmentations, expert PA on T101, later OFAV, plus a 2015 BIT image.
    proj_b = make_project(root, "run_projB_20260826_bbbbbb", "projB", "Project B", initials="CD",
                          created_at="2026-08-15T09:00:00")
    raw_b = os.path.join(proj_b, "step5_segmentImages", "routed_input", "raw")
    img_b = {n: make_image(raw_b, n) for n in (T101, T2015)}
    seg_b = {
        T101: seg_entry(img_b[T101], [
            mask(0, "PA", source_type="manual_click", sx=15, sy=25, expert=True, reviewer="CD"),
            mask(1, "OFAV", source_type="auto", sx=35, sy=45, rings=[RING_B]),
        ], processed_at="2026-08-20T10:00:00", exported=True),
        T2015: seg_entry(img_b[T2015], [
            mask(0, "PA", source_type="auto", sx=5, sy=6, reviewer="CD"),
            mask(1, "MC", source_type="manual_box", sx=7, sy=8, status="rejected"),
        ], processed_at="2026-08-20T11:00:00"),
    }
    write(os.path.join(proj_b, "step5_segmentImages", "segmentations", "2014", "segmentations.json"),
          json.dumps(seg_b, indent=1))

    # projC: ledger only.
    proj_c = make_project(root, "run_projC_20260826_cccccc", "projC", "Project C", initials="EF")
    write(os.path.join(proj_c, "step4_routeChosenImages", "label_provenance.csv"),
          "basename,label,outcome,source,reviewer,at,project_id\n"
          "IMG_ledger,PA,found_manual,step4test,EF,2026-08-01T10:00:00,projC\n")

    # junk dir without project.json.
    os.makedirs(os.path.join(root, "run_junk"), exist_ok=True)

    reg = MaskRegistry(os.path.join(root, "_mask_registry"))
    uids_a = {}
    if seed_registry:
        for fn, seg in seg_a.items():
            for m in seg["masks"]:
                rec = build_registry_record(fn, m, "projA", "stale name never shown")
                if rec is None:
                    continue
                rec["reviewer"] = m.get("reviewer", "")
                reg.upsert(rec)
                uids_a[(fn, m["id"])] = rec["uid"]
        for pid, uid in (("exp_foo", "EXP-x0y0"), ("tmp", "TMP-x0y0"),
                         ("e2e_bar", "E2E-x0y0"), ("ghost", "GHOST-x0y0")):
            reg.upsert({
                "uid": uid, "source_image": "POS.jpg", "basename": "POS", "species": "PA",
                "category": "", "source_type": "manual_click", "review": "", "expert_mode": "",
                "status": "accepted", "project_id": pid, "project_name": pid,
            })
    return {
        "proj_a": proj_a, "proj_b": proj_b, "proj_c": proj_c,
        "img_a": img_a, "img_b": img_b, "seg_a": seg_a, "seg_b": seg_b,
        "uids_a": uids_a, "registry_dir": reg.root,
    }
