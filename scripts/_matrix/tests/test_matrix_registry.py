#!/usr/bin/env python
"""No-pytest harness for the matrix's registry-preferred build path.

Run with:
  env/bin/python scripts/_matrix/tests/test_matrix_registry.py

Exits nonzero on any failed assertion.
"""

import json
import os
import shutil
import sys
import tempfile

# Make the packages importable: scripts/_matrix/tests/ -> scripts/ on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix  # noqa: E402
from _reefreview.mask_registry import MaskRegistry  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _make_project_json(run_dir, project_id, name, target_species):
    meta = {
        "id": project_id,
        "name": name,
        "steps": {"3": {"config": {"target_species": target_species}}},
    }
    _write(os.path.join(run_dir, "project.json"), json.dumps(meta, indent=2))


def build_fixture(root):
    """Two projects, each with a project.json but NO ledger and NO legacy
    segmentations.json -- their found-* cells must come entirely from the
    registry once it is seeded."""
    proj_a = os.path.join(root, "run_projA_20260710_aaaaaa")
    _make_project_json(proj_a, "projA", "Project A", "PA")

    proj_b = os.path.join(root, "run_projB_20260710_bbbbbb")
    _make_project_json(proj_b, "projB", "Project B", "PA")

    return proj_a, proj_b


def main():
    root = tempfile.mkdtemp(prefix="matrix_registry_test_")
    try:
        proj_a, proj_b = build_fixture(root)

        # --- Registry EMPTY: build_matrix must fall back without error. ---
        m_empty = build_matrix(root)
        check(m_empty is not None, "build_matrix returns with an empty/absent registry")
        check(m_empty["stats"]["n_projects"] == 2,
              f"empty-registry: stats.n_projects == 2 (got {m_empty['stats']['n_projects']})")
        # No ledger, no legacy segmentations.json -> no found rows, only the
        # (empty, since no reviewed images exist) not_found derivation. This
        # must not raise.
        check(m_empty["cells"] == {}, f"empty-registry: no cells produced (got {m_empty['cells']})")

        # --- Seed the canonical registry with accepted manual PA masks on
        # two basenames across the two project_ids. ---
        registry_dir = os.path.join(root, "_mask_registry")
        reg = MaskRegistry(registry_dir)
        reg.upsert({
            "uid": "SCP-20260710-T101-x10y20-aaaa",
            "source_image": "IMG_one.jpeg",
            "basename": "IMG_one",
            "species": "PA",
            "category": "",
            "source_type": "manual_click",
            "review": "",
            "expert_mode": "",
            "status": "accepted",
            "project_id": "projA",
            "project_name": "Project A",
        })
        reg.upsert({
            "uid": "SCP-20260710-T102-x30y40-bbbb",
            "source_image": "IMG_two.jpeg",
            "basename": "IMG_two",
            "species": "PA",
            "category": "",
            "source_type": "manual_box",
            "review": "",
            "expert_mode": "",
            "status": "accepted",
            "project_id": "projB",
            "project_name": "Project B",
        })

        m = build_matrix(root)

        c_one_pa = m["cells"].get("IMG_one", {}).get("PA", {})
        check(c_one_pa.get("outcome") == "found_manual",
              f"registry-seeded: cells[IMG_one][PA].outcome == found_manual (got {c_one_pa.get('outcome')})")

        c_two_pa = m["cells"].get("IMG_two", {}).get("PA", {})
        check(c_two_pa.get("outcome") == "found_manual",
              f"registry-seeded: cells[IMG_two][PA].outcome == found_manual (got {c_two_pa.get('outcome')})")

        # Sources trace back to the registry, not derived/ledger.
        sources_one = c_one_pa.get("sources", [])
        check(len(sources_one) == 1 and sources_one[0].get("source") == "registry",
              f"registry-seeded: cells[IMG_one][PA] source == registry (got {sources_one})")

        check(m["stats"]["n_projects"] == 2,
              f"registry-seeded: stats.n_projects == 2 (got {m['stats']['n_projects']})")
        bo = m["stats"]["by_outcome"]
        check(bo.get("found_manual") == 2,
              f"registry-seeded: stats.by_outcome.found_manual == 2 (got {bo.get('found_manual')})")

        # --- A pending (review) mask must surface as pending_expert even
        # though its status is not 'accepted'. ---
        reg.upsert({
            "uid": "SCP-20260710-T103-x50y60-cccc",
            "source_image": "IMG_three.jpeg",
            "basename": "IMG_three",
            "species": "DSTO",
            "category": "",
            "source_type": "manual_click",
            "review": "1",
            "expert_mode": "",
            "status": "pending",
            "project_id": "projA",
            "project_name": "Project A",
        })
        m2 = build_matrix(root)
        c_three_dsto = m2["cells"].get("IMG_three", {}).get("DSTO", {})
        check(c_three_dsto.get("outcome") == "pending_expert",
              f"registry pending mask -> pending_expert (got {c_three_dsto.get('outcome')})")

        # --- A non-accepted, non-review mask contributes nothing. ---
        reg.upsert({
            "uid": "SCP-20260710-T104-x70y80-dddd",
            "source_image": "IMG_four.jpeg",
            "basename": "IMG_four",
            "species": "MC",
            "category": "",
            "source_type": "auto",
            "review": "",
            "expert_mode": "",
            "status": "rejected",
            "project_id": "projA",
            "project_name": "Project A",
        })
        m3 = build_matrix(root)
        check("IMG_four" not in m3["cells"],
              f"registry rejected non-review mask excluded (got {m3['cells'].get('IMG_four')})")

        # --- Per-project registry preference: a project with ONLY a real
        # ledger (no registry rows of its own) must keep its ledger-derived
        # cells even though a sibling project has registry rows. Regression
        # test for the global `if registry_rows:` switch bug, which replaced
        # EVERY project's rows with that project's (possibly empty) registry
        # slice the moment ANY project had registry rows. ---
        mixed_root = tempfile.mkdtemp(prefix="matrix_registry_mixed_")
        try:
            # Project A: real ledger row only, NO registry rows.
            proj_ledger = os.path.join(mixed_root, "run_projLedgerOnly_20260710_eeeeee")
            _make_project_json(proj_ledger, "projLedgerOnly", "Project Ledger Only", "PA")
            ledger_path = os.path.join(
                proj_ledger, "step4_routeChosenImages", "label_provenance.csv"
            )
            _write(
                ledger_path,
                "basename,label,outcome,source,reviewer,at,project_id\n"
                "IMG_a,PA,found_manual,step4test,LO,2026-07-10T10:00:00,projLedgerOnly\n",
            )

            # Project B: registry row only (via MaskRegistry.upsert), no ledger.
            proj_registry = os.path.join(mixed_root, "run_projRegistryOnly_20260710_ffffff")
            _make_project_json(proj_registry, "projRegistryOnly", "Project Registry Only", "PA")
            mixed_registry_dir = os.path.join(mixed_root, "_mask_registry")
            mixed_reg = MaskRegistry(mixed_registry_dir)
            mixed_reg.upsert({
                "uid": "SCP-20260710-T105-x90y10-eeee",
                "source_image": "IMG_b.jpeg",
                "basename": "IMG_b",
                "species": "PA",
                "category": "",
                "source_type": "manual_click",
                "review": "",
                "expert_mode": "",
                "status": "accepted",
                "project_id": "projRegistryOnly",
                "project_name": "Project Registry Only",
            })

            m_mixed = build_matrix(mixed_root)

            c_a_pa = m_mixed["cells"].get("IMG_a", {}).get("PA", {})
            check(
                c_a_pa.get("outcome") == "found_manual",
                "mixed-project: ledger-only project's IMG_a/PA found_manual "
                f"cell survives a sibling project's registry rows (got {c_a_pa.get('outcome')})",
            )
            sources_a = c_a_pa.get("sources", [])
            check(
                bool(sources_a) and sources_a[0].get("source") == "step4test",
                f"mixed-project: IMG_a/PA source stays ledger-derived (got {sources_a})",
            )

            c_b_pa = m_mixed["cells"].get("IMG_b", {}).get("PA", {})
            check(
                c_b_pa.get("outcome") == "found_manual",
                f"mixed-project: registry-only project's IMG_b/PA found_manual cell present (got {c_b_pa.get('outcome')})",
            )
        finally:
            shutil.rmtree(mixed_root, ignore_errors=True)

    finally:
        shutil.rmtree(root, ignore_errors=True)

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
