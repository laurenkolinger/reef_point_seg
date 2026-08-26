"""
Task 2: the expert-id round trip relabels the 4.test combined-annotator export
(step4test_combinedAnnotate/segmentations), not only the legacy Step-5 dir.

NO-pytest harness (mirrors test_expertids.py). Run with the unified env python:
    env/bin/python scripts/_expertids/tests/test_step4test_relabel.py

Reuses test_expertids.py fixtures (numpy/PIL image, REVIEW mask, _env, _seg_mask,
review_export, importer). Covers:
  UNIT        a 4.test-annotated project (segmentations under
              step4test_combinedAnnotate) round-trips: import lands the tentative
              review there, and accept stamps expert_id there.
  REGRESSION  a project with ONLY a Step-5 segmentations tree still resolves to
              the Step-5 dir (prefer-4test, fall-back-to-5).
"""

import os
import sys
import json
import tempfile

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)               # so we can import the sibling test module
import test_expertids as te             # noqa: E402 — reuse its fixtures
import project_manager                  # noqa: E402 — already on path via te

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    import traceback
    try:
        fn(); _RESULTS.append((fn.__name__, True, "")); print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _mk_project_4test(projects_root, pid, fn, mask):
    """Like te._mk_project but the export dir is step4test_combinedAnnotate."""
    pdir = os.path.join(projects_root, pid)
    export_dir = os.path.join(pdir, "step4test_combinedAnnotate")
    segdir = os.path.join(export_dir, "segmentations", "2020"); os.makedirs(segdir, exist_ok=True)
    raw = os.path.join(export_dir, "raw"); os.makedirs(raw, exist_ok=True)
    with open(os.path.join(pdir, "project.json"), "w") as f:
        json.dump({"id": pid, "name": pid.split("_")[0]}, f)
    img = os.path.join(raw, fn)
    Image.fromarray((np.random.rand(200, 200, 3) * 255).astype("uint8")).save(img)
    seg = {"image_path": "raw/" + fn, "image_path_abs": img, "image_width": 200,
           "image_height": 200, "masks": [mask], "reviewed": False, "exported": False}
    return export_dir, segdir, {fn: seg}


def _export_4test(projects_root, pid, fn, *, review_dir, library_dir, sq=(60, 140, 60, 140)):
    y0, y1, x0, x1 = sq
    mask = te._review_mask(te.mask_geom.rle_encode(te._square_mask(200, 200, y0, y1, x0, x1)),
                           [x0, y0, x1 - 1, y1 - 1],
                           [[x0, y0, x1, y0, x1, y1, x0, y1]], (x0 + x1) // 2, (y0 + y1) // 2)
    export_dir, segdir, segmap = _mk_project_4test(projects_root, pid, fn, mask)
    st = te.review_export.export_flagged_masks(
        segmap, [fn], export_dir=export_dir, review_dir=review_dir, repo_url="",
        library_dir=library_dir, master_codes=None, contacts=[], featured_codes=["PA"],
        project_id=pid, project_name="", git_push=False, log_fn=lambda m: None)
    with open(os.path.join(segdir, "segmentations.json"), "w") as f:
        json.dump(segmap, f)
    return export_dir, st["uids"][0]


def test_round_trip_relabels_4test_segmentations():
    with tempfile.TemporaryDirectory() as t:
        e = te._env(t)
        fn = "TCRMP20201022_clip_SCP_T101.jpeg"
        pid = "d4test_20260626_ddd"
        ed4, uid = _export_4test(e["projects_root"], pid, fn,
                                 review_dir=e["review_dir"], library_dir=e["library_dir"])
        check(ed4.endswith("step4test_combinedAnnotate"), ed4)
        # resolver (default) prefers the 4.test dir for this project
        from _expertids import importer as imp
        resolved = imp._resolve_step_dir(pid, e["projects_root"])
        check(resolved == ed4, f"resolver picked {resolved}, want {ed4}")
        base = dict(review_dir=e["review_dir"], repo_url="", library_dir=e["library_dir"],
                    master_codes=None, projects_root=e["projects_root"], git_push=False,
                    log_fn=lambda m: None)
        # import: tentative review must land in the 4.test segmentations
        st = te.importer.import_rows(
            [{"uid": uid, "code": "PA", "reviewer": "jane", "project_id": pid}],
            export_dir="", **base)
        check(st["unrouted"] == [], f"nothing unrouted: {st}")
        m = te._seg_mask(ed4, fn)
        check(m is not None and m["reviews"][0]["code"] == "PA",
              f"tentative PA in 4.test seg: {m}")
        # accept: expert_id must be stamped in the 4.test segmentations
        res = te.importer.accept_uid(uid, "PA", **base)
        check(res["ok"] and res["relabeled_seg"], f"accept relabel: {res}")
        check(res["export_dir"].endswith("step4test_combinedAnnotate"),
              f"accept resolved 4.test dir: {res['export_dir']}")
        m2 = te._seg_mask(ed4, fn)
        check(isinstance(m2.get("expert_id"), dict) and m2["expert_id"].get("code") == "PA",
              f"expert_id stamped in 4.test seg: {m2.get('expert_id')}")
        check(m2.get("review") is False, "mask review flag cleared after accept")


def test_step5_only_still_resolves_step5():
    """Regression: a project annotated only in the old Step-5 UI still resolves
    to its Step-5 dir (prefer-4.test, fall back to 5)."""
    with tempfile.TemporaryDirectory() as t:
        e = te._env(t)
        fn = "TCRMP20201022_clip_BWR_T201.jpeg"
        pid = "legacy5_20200101_eee"
        ed5, uid = te._export(e["projects_root"], pid, fn,
                              review_dir=e["review_dir"], library_dir=e["library_dir"])
        check(ed5.endswith("step5_segmentImages"), ed5)
        from _expertids import importer as imp
        resolved = imp._resolve_step_dir(pid, e["projects_root"])
        check(resolved == ed5, f"legacy resolver picked {resolved}, want {ed5}")


def test_empty_step4test_segmentations_does_not_shadow_step5():
    """Adversarial regression (review finding): an EMPTY/aborted
    step4test_combinedAnnotate/segmentations/ dir must NOT shadow a real Step-5
    annotation. The content-aware guard globs for an actual segmentations.json,
    so a bare empty dir falls back to Step 5."""
    with tempfile.TemporaryDirectory() as t:
        e = te._env(t)
        fn = "TCRMP20201022_clip_SCP_T101.jpeg"
        pid = "both_20260626_fff"
        # Real annotation lives in the Step-5 dir.
        ed5, uid = te._export(e["projects_root"], pid, fn,
                              review_dir=e["review_dir"], library_dir=e["library_dir"])
        check(ed5.endswith("step5_segmentImages"), ed5)
        # Aborted 4.test routing left an EMPTY segmentations/ dir (no year JSON).
        empty_seg = os.path.join(e["projects_root"], pid,
                                 "step4test_combinedAnnotate", "segmentations")
        os.makedirs(empty_seg, exist_ok=True)
        from _expertids import importer as imp
        resolved = imp._resolve_step_dir(pid, e["projects_root"])
        check(resolved == ed5,
              f"empty step4test must not shadow step5; got {resolved}")
        # And the round trip still relabels the real Step-5 mask.
        base = dict(review_dir=e["review_dir"], repo_url="", library_dir=e["library_dir"],
                    master_codes=None, projects_root=e["projects_root"], git_push=False,
                    log_fn=lambda m: None)
        te.importer.import_rows(
            [{"uid": uid, "code": "PA", "reviewer": "jane", "project_id": pid}],
            export_dir="", **base)
        res = te.importer.accept_uid(uid, "PA", **base)
        check(res["ok"] and res["relabeled_seg"], f"accept must relabel step5: {res}")
        check(res["export_dir"].endswith("step5_segmentImages"),
              f"accept resolved {res['export_dir']}, want step5")


if __name__ == "__main__":
    print("TASK 2 — expert-id round trip relabels 4.test segmentations\n")
    print("UNIT:")
    run(test_round_trip_relabels_4test_segmentations)
    print("REGRESSION:")
    run(test_step5_only_still_resolves_step5)
    run(test_empty_step4test_segmentations_does_not_shadow_step5)
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
