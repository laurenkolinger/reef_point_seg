#!/usr/bin/env python
"""No-pytest harness for the per-cell registry merge (data fidelity).

A project that carries SOME mask-registry rows used to have its entire
ledger/legacy contribution thrown away, so a project with 1547 masks on disk
and 5 registry rows rendered as 5 masks. The contract this suite pins:

  * build_matrix starts from the ledger/legacy rows and upserts the registry
    rows over their own (basename, label) cells. Registry wins its own cell;
    every other cell survives.
  * not_found keeps coming from the ledger/legacy chain only.
  * A fully registered project behaves exactly as before: one registry source
    per cell, no duplicate derived source.
  * Each project entry in the /api/data payload carries partially_registered.
  * stats.collect_records merges at mask (uid) granularity, so the grid and
    the STATS tab report the same (image, label) population for a project.
  * The legacy fallback is disabled per processing directory, not per
    project: a directory with its own label_provenance.csv stays ledger
    driven, a sibling directory with segmentations and no ledger derives.

Run with:
  env/bin/python scripts/_matrix/tests/test_partial_registry_merge.py

Exits nonzero on any failed assertion.
"""

import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix  # noqa: E402
from _matrix.stats import collect_records, compute_stats  # noqa: E402
from _matrix.tests._fixture_cell import make_project, mask, seg_entry, write  # noqa: E402
from _reefreview.mask_registry import MaskRegistry, build_registry_record  # noqa: E402

_FAILURES = []

# Image names parse into site/date/transect/frame so the minted uid is safe
# and matches between the registry writer and the stats re-derivation.
P1 = "TCRMP20140930_clip_SCP_T301.jpeg"
P2 = "TCRMP20140930_clip_SCP_T302.jpeg"
P3 = "TCRMP20140930_clip_SCP_T303.jpeg"
F1 = "TCRMP20140930_clip_SSJ_T311.jpeg"
F2 = "TCRMP20140930_clip_SSJ_T312.jpeg"
F3 = "TCRMP20140930_clip_SSJ_T313.jpeg"
L1 = "TCRMP20140930_clip_SCP_T321.jpeg"
M2 = "TCRMP20140930_clip_SCP_T332.jpeg"
S2 = "TCRMP20140930_clip_SCP_T342.jpeg"

TARGETS = "PA, OFAV, MISS"


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def _seg_doc(pairs):
    """{filename: seg_entry} for a list of (filename, [masks])."""
    return {fn: seg_entry(fn, masks) for fn, masks in pairs}


def _write_segs(project_dir, step_dir, pairs, batch=None):
    import json
    parts = [project_dir, step_dir, "segmentations"]
    if batch:
        parts.append(batch)
    parts.append("segmentations.json")
    write(os.path.join(*parts), json.dumps(_seg_doc(pairs), indent=1))


def _cells_of(matrix, project_id, include_not_found=False):
    """{(basename, label)} the grid credits to one project."""
    out = set()
    for basename, by_label in (matrix.get("cells") or {}).items():
        for label, cell in by_label.items():
            for src in cell.get("sources") or []:
                if src.get("project_id") != project_id:
                    continue
                if not include_not_found and src.get("outcome") == "not_found":
                    continue
                out.add((basename, label))
    return out


def _project(matrix, project_id):
    return next((p for p in matrix["projects"] if p["project_id"] == project_id), None)


def build_fixture(root):
    """Five projects covering every branch of the merge."""
    # --- projPartial: legacy chain, ONE of its three cells registered. ---
    partial = make_project(root, "run_projPartial_20260901_p1p1p1", "projPartial",
                           "Project Partial", target_species=TARGETS)
    m_p1 = mask(0, "PA", source_type="auto", sx=10, sy=20)
    _write_segs(partial, "step5_segmentImages", [
        (P1, [m_p1]),
        (P2, [mask(0, "OFAV", source_type="manual_click", sx=30, sy=40)]),
        (P3, [mask(0, "PA", source_type="manual_box", sx=50, sy=60)]),
    ], batch="2014")

    # --- projFull: legacy chain, ALL three cells registered. ---
    full = make_project(root, "run_projFull_20260901_f1f1f1", "projFull",
                        "Project Full", target_species=TARGETS)
    m_f1 = mask(0, "PA", source_type="manual_click", sx=11, sy=21)
    m_f2 = mask(0, "OFAV", source_type="auto", sx=31, sy=41)
    m_f3 = mask(0, "PA", source_type="manual_box", sx=51, sy=61)
    _write_segs(full, "step5_segmentImages", [(F1, [m_f1]), (F2, [m_f2]), (F3, [m_f3])])

    # --- projLegacy: no registry rows at all. ---
    legacy = make_project(root, "run_projLegacy_20260901_l1l1l1", "projLegacy",
                          "Project Legacy", target_species=TARGETS)
    _write_segs(legacy, "step5_segmentImages", [
        (L1, [mask(0, "PA", source_type="manual_click", sx=12, sy=22)]),
    ])

    # --- projMixedDirs: ledger in one processing dir, segmentations with no
    # ledger in a sibling dir. The sibling must still derive. ---
    mixed = make_project(root, "run_projMixedDirs_20260901_m1m1m1", "projMixedDirs",
                         "Project Mixed Dirs", target_species=TARGETS)
    write(os.path.join(mixed, "step4test_combinedAnnotate", "label_provenance.csv"),
          "basename,label,outcome,source,reviewer,at,project_id\n"
          "IMG_mixed_ledger,PA,found_manual,step4test,LO,2026-09-01T10:00:00,projMixedDirs\n")
    _write_segs(mixed, "step5_segmentImages", [
        (M2, [mask(0, "OFAV", source_type="manual_click", sx=13, sy=23)]),
    ], batch="2014")

    # --- projLedgerSameDir: ledger AND segmentations in the SAME dir. The
    # ledger stays authoritative for that dir (no derivation). ---
    same = make_project(root, "run_projSameDir_20260901_s1s1s1", "projSameDir",
                        "Project Same Dir", target_species=TARGETS)
    write(os.path.join(same, "step4test_combinedAnnotate", "label_provenance.csv"),
          "basename,label,outcome,source,reviewer,at,project_id\n"
          "IMG_same_ledger,PA,found_manual,step4test,LO,2026-09-01T10:00:00,projSameDir\n")
    _write_segs(same, "step4test_combinedAnnotate", [
        (S2, [mask(0, "OFAV", source_type="manual_click", sx=14, sy=24)]),
    ])

    # --- registry: projPartial's P1 mask (stamped EXPERT so the registry
    # outcome differs visibly from the legacy found_ai) + all of projFull. ---
    reg = MaskRegistry(os.path.join(root, "_mask_registry"))
    rec = build_registry_record(P1, m_p1, "projPartial", "Project Partial")
    rec["expert_mode"] = "1"
    reg.upsert(rec)
    for fn, m in ((F1, m_f1), (F2, m_f2), (F3, m_f3)):
        reg.upsert(build_registry_record(fn, m, "projFull", "Project Full"))
    return {"partial": partial, "full": full, "reg": reg}


def main():
    root = tempfile.mkdtemp(prefix="matrix_partial_merge_")
    try:
        build_fixture(root)
        m = build_matrix(root)

        # ---- partially registered project: every cell survives ----
        found_partial = _cells_of(m, "projPartial")
        expect_partial = {(_stem(P1), "PA"), (_stem(P2), "OFAV"), (_stem(P3), "PA")}
        check(found_partial == expect_partial,
              f"partial project keeps all three found cells, not just the registered one (got {sorted(found_partial)})")

        c_p1 = m["cells"].get(_stem(P1), {}).get("PA", {})
        check(c_p1.get("outcome") == "found_expert",
              f"registry wins its own cell: P1/PA found_expert over the legacy found_ai (got {c_p1.get('outcome')})")
        srcs_p1 = c_p1.get("sources") or []
        check(len(srcs_p1) == 1 and srcs_p1[0].get("source") == "registry",
              f"registered cell carries the registry source only, no duplicate derived row (got {srcs_p1})")

        c_p2 = m["cells"].get(_stem(P2), {}).get("OFAV", {})
        srcs_p2 = c_p2.get("sources") or []
        check(c_p2.get("outcome") == "found_manual" and len(srcs_p2) == 1
              and srcs_p2[0].get("source") == "derived",
              f"unregistered cell keeps its legacy row (got {c_p2.get('outcome')}, {srcs_p2})")
        c_p3 = m["cells"].get(_stem(P3), {}).get("PA", {})
        check(c_p3.get("outcome") == "found_manual",
              f"second unregistered cell keeps its legacy row (got {c_p3.get('outcome')})")

        # ---- not_found still comes from the ledger/legacy chain ----
        c_miss = m["cells"].get(_stem(P1), {}).get("MISS", {})
        check(c_miss.get("outcome") == "not_found",
              f"target species with no mask stays not_found on a registry-bearing project (got {c_miss.get('outcome')})")
        c_p1_ofav = m["cells"].get(_stem(P1), {}).get("OFAV", {})
        check(c_p1_ofav.get("outcome") == "not_found",
              f"P1/OFAV not_found survives the merge (got {c_p1_ofav.get('outcome')})")

        # ---- fully registered project behaves exactly as before ----
        found_full = _cells_of(m, "projFull")
        check(found_full == {(_stem(F1), "PA"), (_stem(F2), "OFAV"), (_stem(F3), "PA")},
              f"fully registered project keeps its three cells (got {sorted(found_full)})")
        for fn, expect in ((F1, "found_manual"), (F2, "found_ai"), (F3, "found_manual")):
            cell = m["cells"].get(_stem(fn), {})
            label = "OFAV" if fn == F2 else "PA"
            cell = cell.get(label, {})
            srcs = cell.get("sources") or []
            check(cell.get("outcome") == expect and len(srcs) == 1
                  and srcs[0].get("source") == "registry",
                  f"fully registered {_stem(fn)}/{label}: one registry source, {expect} (got {cell.get('outcome')}, {srcs})")
        check(m["cells"].get(_stem(F1), {}).get("MISS", {}).get("outcome") == "not_found",
              "fully registered project keeps its legacy not_found cells")

        # ---- partially_registered flag ----
        for pid, expect in (("projPartial", True), ("projFull", False),
                            ("projLegacy", False), ("projMixedDirs", False),
                            ("projSameDir", False)):
            entry = _project(m, pid)
            check(entry is not None and entry.get("partially_registered") is expect,
                  f"{pid}.partially_registered is {expect} (got {entry and entry.get('partially_registered')})")

        # ---- legacy-only project untouched by a sibling's registry rows ----
        check(_cells_of(m, "projLegacy") == {(_stem(L1), "PA")},
              f"legacy-only project keeps its cell (got {sorted(_cells_of(m, 'projLegacy'))})")

        # ---- per-directory ledger rule ----
        mixed_cells = _cells_of(m, "projMixedDirs")
        check(("IMG_mixed_ledger", "PA") in mixed_cells,
              f"ledger dir still contributes its rows (got {sorted(mixed_cells)})")
        check((_stem(M2), "OFAV") in mixed_cells,
              f"sibling dir with segmentations and no ledger derives its cells (got {sorted(mixed_cells)})")
        same_cells = _cells_of(m, "projSameDir")
        check(same_cells == {("IMG_same_ledger", "PA")},
              f"ledger stays authoritative for its own dir, no derivation there (got {sorted(same_cells)})")

        # ---- STATS agrees with the grid on the partial project ----
        records, _excluded, legacy_projects = collect_records(root)
        stats_cells = {(r["basename"], r["species"]) for r in records
                       if r["project_id"] == "projPartial" and r["species"]}
        check(stats_cells == expect_partial,
              f"STATS reports the same (image, label) population as the grid (got {sorted(stats_cells)})")
        partial_records = [r for r in records if r["project_id"] == "projPartial"]
        check(len(partial_records) == 3,
              f"STATS counts all three masks of the partial project, not just the registered one (got {len(partial_records)})")
        origins = sorted({r["origin"] for r in partial_records})
        check(origins == ["legacy", "registry"],
              f"partial project mixes one registry record with two derived records (got {origins})")
        check(len([r for r in partial_records if r["origin"] == "registry"]) == 1,
              "exactly one registry-origin record for the partial project")
        check(len([r for r in records if r["project_id"] == "projFull"]) == 3
              and {r["origin"] for r in records if r["project_id"] == "projFull"} == {"registry"},
              "fully registered project still counts registry records only, no double count")
        check("projPartial" in legacy_projects and "projFull" not in legacy_projects,
              f"legacy_projects names the projects that contributed derived records (got {legacy_projects})")

        s = compute_stats(root)
        pa = next((x for x in s["species"] if x["code"] == "PA"), None)
        # projPartial P1 + P3, projFull F1 + F3, projLegacy L1, projSameDir S2
        # is OFAV, projMixedDirs M2 is OFAV.
        check(pa is not None and pa["n_masks"] == 5,
              f"STATS species PA counts every accepted PA mask across projects (got {pa})")
        ofav = next((x for x in s["species"] if x["code"] == "OFAV"), None)
        check(ofav is not None and ofav["n_masks"] == 4,
              f"STATS species OFAV counts every accepted OFAV mask (got {ofav})")
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
