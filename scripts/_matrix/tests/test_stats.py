#!/usr/bin/env python
"""No-pytest harness for the STATS aggregation (stats.py): species / year /
site counts, unknown-project exclusion, legacy derivation, the status
filter and the test-residue purge with backup.

Run with:
  env/bin/python scripts/_matrix/tests/test_stats.py
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

import glob
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.stats import compute_stats, is_test_residue_id, purge_test_rows  # noqa: E402
from _matrix.tests._fixture_cell import T101, T2015, build_root  # noqa: E402
from _reefreview.mask_registry import MaskRegistry  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def _species(stats, code):
    return next((s for s in stats["species"] if s["code"] == code), None)


def main():
    for pid, expect in (("tmp", True), ("exp_neg_9mb", True), ("e2e_x", True), ("combined_smoke_1", True),
                        ("label_guard_a", True), ("expert_lock_test_b", True), ("26-08-14_lo2", False),
                        ("tmp2", False), ("ghost", False)):
        check(is_test_residue_id(pid) is expect, f"is_test_residue_id({pid!r}) is {expect}")

    root = tempfile.mkdtemp(prefix="matrix_stats_test_")
    try:
        build_root(root)

        s = compute_stats(root)
        check(s["status"] == "accepted" and s["generated_at"].endswith("-04:00"), "default status accepted, AST stamp")
        t = s["totals"]
        check(t["masks"] == 5, f"totals.masks: 2 projA registry + 3 projB legacy accepted (got {t['masks']})")
        check(t["images"] == 2 and t["projects"] == 2 and t["labels"] == 2,
              f"totals images/projects/labels (got {t})")
        pa = _species(s, "PA")
        ofav = _species(s, "OFAV")
        check(pa == {"code": "PA", "n_masks": 3, "n_images": 2, "n_projects": 2}, f"species PA (got {pa})")
        check(ofav == {"code": "OFAV", "n_masks": 2, "n_images": 1, "n_projects": 2}, f"species OFAV (got {ofav})")
        check([x["code"] for x in s["species"]] == ["PA", "OFAV"], "species sorted by n_masks desc")
        check(s["years"] == [2014, 2015] and s["sites"] == ["BIT", "SCP"], f"years/sites sorted (got {s['years']}, {s['sites']})")
        check(s["by_year"]["PA"] == {"2014": 2, "2015": 1} and s["by_year"]["OFAV"] == {"2014": 2},
              f"by_year (got {s['by_year']})")
        check(s["by_site"]["PA"] == {"SCP": 2, "BIT": 1} and s["by_site"]["OFAV"] == {"SCP": 2},
              f"by_site (got {s['by_site']})")
        check(s["images_by_year"] == {"2014": 1, "2015": 1} and s["images_by_site"] == {"SCP": 1, "BIT": 1},
              f"images per year/site (got {s['images_by_year']}, {s['images_by_site']})")
        cd = next((r for r in s["reviewers"] if r["initials"] == "CD"), None)
        check(cd is not None and cd["n_masks"] == 2, f"reviewer CD from legacy masks (got {s['reviewers']})")
        check(sum(r["n_masks"] for r in s["reviewers"]) == 5, "reviewer counts sum to the mask total")
        check(all(r["initials"] for r in s["reviewers"]), "blank reviewer rendered as 'unrecorded', never blank")
        check(s["legacy_projects"] == ["projB"], f"projB (no registry rows) derived from segmentations (got {s['legacy_projects']})")
        ex = s["excluded"]
        check(ex["unknown_project_rows"] == 4 and ex["unknown_project_ids"] == ["e2e_bar", "exp_foo", "ghost", "tmp"],
              f"unknown-project rows counted and named, never dropped silently (got {ex})")
        check(ex["test_residue_rows"] == 3, f"test residue rows (got {ex['test_residue_rows']})")
        # L5: projA's T105 accepted mask has no species: reported, never vanishing.
        check(ex["unlabeled_rows"] == 1 and ex["unlabeled_by_project"] == {"projA": 1},
              f"blank-species live rows counted into excluded.unlabeled_rows and per project (got {ex})")
        check(t["masks"] == 5 and "" not in {x["code"] for x in s["species"]},
              "unlabeled rows stay out of the totals and never appear as a blank species")

        s_all = compute_stats(root, "all")
        check(s_all["status"] == "all" and s_all["totals"]["masks"] == 8,
              f"status=all counts rejected + review masks too (got {s_all['totals']})")
        codes = {x["code"] for x in s_all["species"]}
        check(codes == {"PA", "OFAV", "REVIEW", "MC"}, f"status=all labels (got {codes})")
        check(compute_stats(root, "ALL")["status"] == "all" and compute_stats(root, "bogus")["status"] == "accepted",
              "status normalizes: ALL -> all, anything else -> accepted")

        # L5: the status filter applies to unlabeled rows too; unknown-project
        # blanks stay in the unknown bucket, not the unlabeled one.
        reg_l5 = MaskRegistry(os.path.join(root, "_mask_registry"))
        reg_l5.upsert({
            "uid": "BLANK-REJ-x0y0", "source_image": T101, "basename": _stem(T101), "species": "",
            "category": "", "source_type": "manual_click", "review": "", "expert_mode": "",
            "status": "rejected", "project_id": "projA", "project_name": "projA",
        })
        reg_l5.upsert({
            "uid": "BLANK-GHOST-x0y0", "source_image": T101, "basename": _stem(T101), "species": "",
            "category": "", "source_type": "manual_click", "review": "", "expert_mode": "",
            "status": "accepted", "project_id": "ghost", "project_name": "ghost",
        })
        ex_acc = compute_stats(root)["excluded"]
        ex_all = compute_stats(root, "all")["excluded"]
        check(ex_acc["unlabeled_rows"] == 1 and ex_all["unlabeled_rows"] == 2
              and ex_all["unlabeled_by_project"] == {"projA": 2},
              f"rejected blank-species row counted only under status=all (got {ex_acc['unlabeled_rows']}, {ex_all['unlabeled_rows']})")
        check(ex_acc["unknown_project_rows"] == 5 and "ghost" not in ex_acc["unlabeled_by_project"],
              f"blank species on an unknown project stays an unknown-project row (got {ex_acc})")
        check(compute_stats(root, "all")["totals"]["masks"] == 8, "status=all totals unchanged by unlabeled rows")
        with reg_l5._locked():
            recs = reg_l5.load()
            recs.pop("BLANK-REJ-x0y0", None)
            recs.pop("BLANK-GHOST-x0y0", None)
            reg_l5._write_all(recs)
        check(compute_stats(root)["excluded"]["unknown_project_rows"] == 4, "L5 probe rows removed")

        # ---- purge with backup ----
        reg = MaskRegistry(os.path.join(root, "_mask_registry"))
        before = len(reg.rows())
        removed, backup = purge_test_rows(root)
        check(removed == 3, f"purge removes the 3 residue rows (got {removed})")
        check(backup and os.path.isfile(backup) and os.path.basename(backup).startswith("manifest.csv.bak-"),
              f"backup written before the purge (got {backup})")
        after = reg.rows()
        check(len(after) == before - 3, f"manifest shrank by 3 (got {before} -> {len(after)})")
        pids = {r["project_id"] for r in after}
        check("ghost" in pids and "projA" in pids and not (pids & {"exp_foo", "tmp", "e2e_bar"}),
              f"unknown non-residue 'ghost' and live projA rows survive (got {sorted(pids)})")
        removed2, _b = purge_test_rows(root)
        check(removed2 == 0, f"second purge finds nothing (got {removed2})")
        check(len(glob.glob(os.path.join(root, "_mask_registry", "manifest.csv.bak-*"))) >= 1, "backup file present")

        # after purge the stats no longer report residue
        s2 = compute_stats(root)
        check(s2["excluded"]["test_residue_rows"] == 0 and s2["excluded"]["unknown_project_rows"] == 1,
              f"stats after purge: only ghost remains excluded (got {s2['excluded']})")
        check(s2["totals"]["masks"] == 5, "live counts unchanged by the purge")

        # legacy uid minted with mask_uid: T2015 PA row present with a site/year parsed from the name
        check(s2["by_site"]["PA"].get("BIT") == 1 and s2["by_year"]["PA"].get("2015") == 1,
              f"legacy {_stem(T2015)} contributes BIT/2015 (got {s2['by_site']['PA']}, {s2['by_year']['PA']})")
        check(s2["by_site"]["PA"].get("SCP") == 2, f"{_stem(T101)} counted from both projects")
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
