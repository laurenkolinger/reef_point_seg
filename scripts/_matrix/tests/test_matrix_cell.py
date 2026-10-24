#!/usr/bin/env python
"""No-pytest harness for the matrix cell drill-down (cellinfo.py) and the
builder's identity / per-source additions.

Run with:
  env/bin/python scripts/_matrix/tests/test_matrix_cell.py

Exits nonzero on any failed assertion.
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix, project_identity, scan_projects  # noqa: E402
from _matrix.cellinfo import cell_info, image_path_for, load_project_segmentations  # noqa: E402
from _matrix.preferences import Preferences  # noqa: E402
from _matrix.tests._fixture_cell import T101, T102, T403, build_root  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def main():
    root = tempfile.mkdtemp(prefix="matrix_cell_test_")
    try:
        fx = build_root(root)

        # --- project_identity ---
        ini, opened = project_identity({"initials": "lo", "access_log": [
            {"initials": "lo", "action": "create"}, {"initials": "ab", "action": "open"}]})
        check((ini, opened) == ("LO", "AB"), f"identity: latest access_log entry wins (got {ini}, {opened})")
        ini, opened = project_identity({"access_log": [{"initials": "cd", "action": "create"}]})
        check((ini, opened) == ("CD", "CD"), f"identity: creator from access_log create, opened_by falls back (got {ini}, {opened})")
        ini, opened = project_identity({"initials": "EF", "access_log": []})
        check((ini, opened) == ("EF", "EF"), f"identity: empty log falls back to project initials (got {ini}, {opened})")
        ini, opened = project_identity({})
        check((ini, opened) == ("", ""), f"identity: nothing recorded -> blank, nothing inferred (got {ini!r}, {opened!r})")

        # --- scan_projects ---
        scanned = scan_projects(root)
        ids = [p["project_id"] for p in scanned]
        check(ids == ["projA", "projB", "projC"], f"scan_projects lists the three real projects, junk ignored (got {ids})")
        pa = scanned[0]
        check(pa["initials"] == "LO" and pa["opened_by"] == "AB",
              f"scan_projects: projA creator LO, opened_by AB (got {pa['initials']}, {pa['opened_by']})")
        check(pa["run_dir"] == "run_projA_20260826_aaaaaa" and pa["project_dir"] == fx["proj_a"],
              f"scan_projects: run_dir + project_dir (got {pa['run_dir']}, {pa['project_dir']})")
        check(pa["name"] == "Project A", f"scan_projects: name from project.json, never the registry (got {pa['name']})")

        # --- build_matrix additions ---
        m = build_matrix(root)
        proj_a = next(p for p in m["projects"] if p["project_id"] == "projA")
        for key in ("initials", "opened_by", "run_dir", "project_dir", "created_at"):
            check(key in proj_a, f"/api/data project carries {key}")
        check(proj_a.get("opened_by") == "AB" and proj_a.get("created_at") == "2026-08-14T09:00:00",
              f"/api/data project identity values (got {proj_a.get('opened_by')}, {proj_a.get('created_at')})")

        cell = m["cells"][_stem(T101)]["PA"]
        check(cell["outcome"] == "found_expert", f"T101/PA outcome found_expert from projB (got {cell['outcome']})")
        src_a = next(s for s in cell["sources"] if s["project_id"] == "projA")
        src_b = next(s for s in cell["sources"] if s["project_id"] == "projB")
        check(src_a["source"] == "registry" and src_a["uid"] == fx["uids_a"][(T101, 0)],
              f"registry source carries the mask uid (got {src_a})")
        check(src_a["status"] == "accepted" and src_a["source_type"] == "manual_click",
              f"registry source carries status + source_type (got {src_a})")
        check(src_a["reviewer"] in ("LO", ""),
              f"registry source reviewer read with .get (column may not exist yet) (got {src_a['reviewer']!r})")
        check(src_b["source"] == "derived" and src_b["uid"] == "" and src_b["reviewer"] == "CD",
              f"legacy derived source: uid '' and reviewer from the mask (got {src_b})")
        ledger = m["cells"]["IMG_ledger"]["PA"]["sources"][0]
        check(ledger["uid"] == "" and ledger["status"] == "" and ledger["reviewer"] == "EF",
              f"ledger source: uid '' (got {ledger})")
        for name in m["projects"]:
            check("stale name" not in name["name"], f"registry project_name never displayed ({name['name']})")

        # --- segmentation loading: flat and nested ---
        segs_a = load_project_segmentations(fx["proj_a"])
        segs_b = load_project_segmentations(fx["proj_b"])
        check(_stem(T101) in segs_a and segs_a[_stem(T101)]["filename"] == T101,
              "flat segmentations/segmentations.json resolved")
        check(_stem(T101) in segs_b and "2014" in segs_b[_stem(T101)]["seg_file"],
              "nested segmentations/<batch>/segmentations.json resolved")

        # --- cell_info: T101 / PA ---
        info = cell_info(root, _stem(T101), "PA", image_url_base="/matrix/img")
        check(info is not None and info["image"] == _stem(T101) and info["label"] == "PA", "cell_info returns the cell")
        check(info["outcome"] == "found_expert", f"cell_info outcome (got {info['outcome']})")
        check(len(info["sources"]) == 2, f"cell_info sources from both projects (got {len(info['sources'])})")
        check(info["preference"] is None, "cell_info preference None when nothing chosen")
        pids = [p["project_id"] for p in info["projects"]]
        check(pids == ["projA", "projB"], f"cell_info projects: flat projA + nested projB (got {pids})")
        p_a = info["projects"][0]
        p_b = info["projects"][1]
        check(p_a["outcome_for_label"] == "found_manual" and p_b["outcome_for_label"] == "found_expert",
              f"outcome_for_label per project (got {p_a['outcome_for_label']}, {p_b['outcome_for_label']})")
        check(p_a["image_url"] == f"/matrix/img/projA/{_stem(T101)}", f"image_url shape (got {p_a['image_url']})")
        check(p_a["image_width"] == 64 and p_a["image_height"] == 48, "image size from segmentations")
        check(p_a["opened_by"] == "AB" and p_a["initials"] == "LO" and p_a["run_dir"] == "run_projA_20260826_aaaaaa",
              "cell_info project identity fields")
        check(len(p_a["masks"]) == 3, f"every mask of the image is listed, not only the label's (got {len(p_a['masks'])})")
        m0 = p_a["masks"][0]
        check(m0["uid"] == fx["uids_a"][(T101, 0)], f"mask uid matches the registry uid (got {m0['uid']})")
        check(m0["is_label"] is True and p_a["masks"][1]["is_label"] is False,
              "is_label true for the PA mask, false for the OFAV mask")
        check(m0["reviewer"] == "LO" and m0["status"] == "accepted" and m0["source_type"] == "manual_click",
              f"mask record fields (got {m0})")
        check(m0["polygon_px"] and m0["bbox"] == [1, 2, 3, 4] and m0["area"] == 100,
              "mask record carries polygon_px, bbox, area")
        check(m0["at"] == "2026-08-14T10:05:00-04:00", f"mask at from label_outcomes (got {m0['at']})")
        check(p_b["masks"][0]["reviewer"] == "CD" and p_b["masks"][0]["expert"] is True,
              "projB expert mask reviewer CD")

        # --- cell_info: review mask with tentative code counts as the label ---
        info2 = cell_info(root, _stem(T102), "DSTO")
        check(info2 is not None and [p["project_id"] for p in info2["projects"]] == ["projA"],
              "review-only image resolves to its project")
        check(info2["projects"][0]["masks"][0]["is_label"] is True and info2["projects"][0]["masks"][0]["review"] is True,
              "pending review mask with tentative code DSTO is_label for DSTO")

        # --- cell_info: ledger-only image, unknown image ---
        info3 = cell_info(root, "IMG_ledger", "PA")
        check(info3 is not None and info3["projects"] == [] and info3["outcome"] == "found_manual",
              "ledger-only cell: no project holds the image but the cell answers")
        check(cell_info(root, "nope", "PA") is None, "unknown image -> None (route 404)")
        check(cell_info(root, _stem(T403), "PA") is not None, "reviewed image with no masks still resolves")

        # --- cell_info: preference.stale (L3) ---
        Preferences(root).set(_stem(T101), "PA", "projA", [], by="LO")
        info4 = cell_info(root, _stem(T101), "PA")
        check(info4["preference"]["project_id"] == "projA" and info4["preference"]["stale"] is False,
              f"preference for a project holding the image is not stale (got {info4['preference']})")
        Preferences(root).set(_stem(T101), "PA", "gone_project", [], by="LO")
        info5 = cell_info(root, _stem(T101), "PA")
        check(info5["preference"]["project_id"] == "gone_project" and info5["preference"]["stale"] is True,
              f"preference whose project holds no copy of the image is stale (got {info5['preference']})")
        Preferences(root).set("IMG_ledger", "PA", "projC", [], by="LO")
        info6 = cell_info(root, "IMG_ledger", "PA")
        check(info6["preference"]["stale"] is True,
              "ledger-only cell: a preference is stale because no project holds the image in segmentations")
        check("stale" not in Preferences(root).get(_stem(T101), "PA"),
              "stale is a cell_info annotation, never written to preferences.csv")
        Preferences(root).clear(_stem(T101), "PA")
        Preferences(root).clear("IMG_ledger", "PA")
        check(cell_info(root, _stem(T101), "PA")["preference"] is None, "preference None again after clear")

        # --- image_path_for ---
        check(image_path_for(root, "projA", _stem(T101)) == fx["img_a"][T101], "image_path_for by stem")
        check(image_path_for(root, "projA", T101) == fx["img_a"][T101], "image_path_for by filename")
        check(image_path_for(root, "run_projA_20260826_aaaaaa", T101) == fx["img_a"][T101], "image_path_for by run_dir")
        check(image_path_for(root, "projB", _stem(T403)) is None, "project without that image -> None")
        check(image_path_for(root, "zzz", T101) is None, "unknown project -> None")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check(not os.path.exists(root), "temp fixture removed")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
