#!/usr/bin/env python
"""No-pytest harness for the matrix data core.

Run with:
  env/bin/python scripts/_matrix/tests/test_matrix.py

Exits nonzero on any failed assertion.
"""

import json
import os
import shutil
import sys
import tempfile

# Make the package importable: scripts/_matrix/tests/ -> scripts/ on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix  # noqa: E402

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
    ts = "2026-06-25T10:00:00"

    # projA: targets PA, has a label_provenance.csv ledger.
    proj_a = os.path.join(root, "run_projA_20260625_aaaaaa")
    _make_project_json(proj_a, "projA", "Project A", "PA")
    ledger = os.path.join(proj_a, "step4_routeChosenImages", "label_provenance.csv")
    _write(
        ledger,
        "basename,label,outcome,source,reviewer,at,project_id\n"
        f"IMG_one,PA,found_manual,step4test,LO,{ts},projA\n"
        f"IMG_two,PA,not_found,step5,,{ts},projA\n"
        f"IMG_two,DLAB,found_ai,step5,,{ts},projA\n",
    )

    # projC: ledger with the expert-review outcomes; conflicts with projA on
    # IMG_one/PA (found_expert vs found_manual), IMG_two/PA (pending_expert vs
    # not_found) and IMG_two/DLAB (pending_expert vs found_ai).
    proj_c = os.path.join(root, "run_projC_20260625_cccccc")
    _make_project_json(proj_c, "projC", "Project C", "PA")
    ledger_c = os.path.join(proj_c, "step4_routeChosenImages", "label_provenance.csv")
    _write(
        ledger_c,
        "basename,label,outcome,source,reviewer,at,project_id\n"
        f"IMG_one,PA,found_expert,step4test,LO,{ts},projC\n"
        f"IMG_two,PA,pending_expert,step4test,LO,{ts},projC\n"
        f"IMG_two,DLAB,pending_expert,step4test,LO,{ts},projC\n",
    )

    # projB: targets OFAV, NO ledger, legacy segmentations.json fallback.
    proj_b = os.path.join(root, "run_projB_20260625_bbbbbb")
    _make_project_json(proj_b, "projB", "Project B", "OFAV")
    seg_doc = {
        "IMG_one.jpeg": {
            "image_path": "IMG_one.jpeg",
            "processed_at": ts,
            "reviewed": True,
            "exported": False,
            "masks": [
                {
                    "id": 0,
                    "label": "O",
                    "species": "OFAV",
                    "status": "accepted",
                    "source_type": "auto",
                    "review": None,
                },
            ],
        },
        # Legacy expert-review states: an accepted expert-ID mask, a review
        # mask with tentative codes (the 'overlap' duplicate excluded), a
        # review mask with no tentative code (label REVIEW), and a rejected
        # review mask that must emit nothing.
        "IMG_three.jpeg": {
            "image_path": "IMG_three.jpeg",
            "processed_at": ts,
            "reviewed": True,
            "exported": False,
            "masks": [
                {
                    "id": 0,
                    "species": "AGA",
                    "status": "accepted",
                    "source_type": "manual_click",
                    "review": False,
                    "expert_id": {"code": "AGA", "mode": "EXPERT", "labeler": "XY"},
                },
                {
                    "id": 1,
                    "species": "REVIEW",
                    "status": "accepted",
                    "source_type": "manual_click",
                    "review": True,
                    "reviews": [
                        {"reviewer": "AB", "code": "DSTO", "confidence": "high"},
                        {"reviewer": "overlap", "code": "ZZZ", "confidence": ""},
                        {"reviewer": "CD", "code": "", "confidence": ""},
                    ],
                },
                {
                    "id": 2,
                    "species": "REVIEW",
                    "status": "accepted",
                    "source_type": "manual_click",
                    "review": True,
                },
                {
                    "id": 3,
                    "species": "REVIEW",
                    "status": "rejected",
                    "source_type": "manual_click",
                    "review": True,
                    "reviews": [{"reviewer": "EF", "code": "XYZ", "confidence": ""}],
                },
            ],
        },
    }
    seg_path = os.path.join(
        proj_b, "step5_segmentImages", "segmentations", "2014", "segmentations.json"
    )
    _write(seg_path, json.dumps(seg_doc, indent=2))

    # junk dir: no project.json -> must be ignored.
    junk = os.path.join(root, "run_demo_try1")
    os.makedirs(junk, exist_ok=True)
    _write(os.path.join(junk, "notes.txt"), "i am not a project")

    return proj_b


def main():
    root = tempfile.mkdtemp(prefix="matrix_test_")
    try:
        proj_b = build_fixture(root)

        m = build_matrix(root)

        check(m["images"] == sorted(["IMG_one", "IMG_two", "IMG_three"]),
              f"images union == [IMG_one, IMG_three, IMG_two] (got {m['images']})")
        check(m["labels"] == sorted(["AGA", "DLAB", "DSTO", "OFAV", "PA", "REVIEW"]),
              f"labels union == [AGA, DLAB, DSTO, OFAV, PA, REVIEW] (got {m['labels']})")

        # Cross-project ranking: found_expert beats found_manual.
        c_one_pa = m["cells"].get("IMG_one", {}).get("PA", {})
        check(c_one_pa.get("outcome") == "found_expert",
              f"cells[IMG_one][PA].outcome == found_expert, beats found_manual (got {c_one_pa.get('outcome')})")

        c_one_ofav = m["cells"].get("IMG_one", {}).get("OFAV", {})
        check(c_one_ofav.get("outcome") == "found_ai",
              f"cells[IMG_one][OFAV].outcome == found_ai (got {c_one_ofav.get('outcome')})")
        ofav_sources = c_one_ofav.get("sources", [])
        check(len(ofav_sources) == 1 and ofav_sources[0].get("source") == "derived",
              f"cells[IMG_one][OFAV] source == derived (got {ofav_sources})")

        # Cross-project ranking: pending_expert beats not_found only.
        c_two_pa = m["cells"].get("IMG_two", {}).get("PA", {})
        check(c_two_pa.get("outcome") == "pending_expert",
              f"cells[IMG_two][PA].outcome == pending_expert, beats not_found (got {c_two_pa.get('outcome')})")
        c_two_dlab = m["cells"].get("IMG_two", {}).get("DLAB", {})
        check(c_two_dlab.get("outcome") == "found_ai",
              f"cells[IMG_two][DLAB].outcome == found_ai, pending loses to found (got {c_two_dlab.get('outcome')})")

        # Legacy fallback: accepted expert-ID mask -> found_expert.
        c_three_aga = m["cells"].get("IMG_three", {}).get("AGA", {})
        check(c_three_aga.get("outcome") == "found_expert",
              f"legacy cells[IMG_three][AGA].outcome == found_expert (got {c_three_aga.get('outcome')})")

        # Legacy fallback: review mask -> pending_expert labeled by tentative
        # codes; 'overlap' duplicates and blank codes excluded; bare review
        # mask labeled REVIEW; rejected review mask emits nothing.
        c_three_dsto = m["cells"].get("IMG_three", {}).get("DSTO", {})
        check(c_three_dsto.get("outcome") == "pending_expert",
              f"legacy cells[IMG_three][DSTO].outcome == pending_expert (got {c_three_dsto.get('outcome')})")
        c_three_rev = m["cells"].get("IMG_three", {}).get("REVIEW", {})
        check(c_three_rev.get("outcome") == "pending_expert",
              f"legacy cells[IMG_three][REVIEW].outcome == pending_expert (got {c_three_rev.get('outcome')})")
        check("ZZZ" not in m["labels"],
              f"legacy: 'overlap' reviewer code ZZZ excluded (got {m['labels']})")
        check("XYZ" not in m["labels"],
              f"legacy: rejected review mask code XYZ excluded (got {m['labels']})")

        # Legacy fallback: target with no accepted or pending mask -> not_found.
        c_three_ofav = m["cells"].get("IMG_three", {}).get("OFAV", {})
        check(c_three_ofav.get("outcome") == "not_found",
              f"legacy cells[IMG_three][OFAV].outcome == not_found (got {c_three_ofav.get('outcome')})")

        check(m["stats"]["n_projects"] == 3,
              f"stats.n_projects == 3 (got {m['stats']['n_projects']})")
        bo = m["stats"]["by_outcome"]
        check(bo.get("found_expert") == 2,
              f"stats.by_outcome.found_expert == 2 (got {bo.get('found_expert')})")
        check(bo.get("pending_expert") == 3,
              f"stats.by_outcome.pending_expert == 3 (got {bo.get('pending_expert')})")
        check(bo.get("found_manual") == 0,
              f"stats.by_outcome.found_manual == 0, expert overrides (got {bo.get('found_manual')})")

        # Purge projB and rebuild: OFAV column/cell must disappear.
        shutil.rmtree(proj_b)
        m2 = build_matrix(root)

        check("OFAV" not in m2["labels"],
              f"after purge: OFAV not in labels (got {m2['labels']})")
        check("OFAV" not in m2["cells"].get("IMG_one", {}),
              "after purge: cells[IMG_one][OFAV] gone")
        check("REVIEW" not in m2["labels"],
              f"after purge: REVIEW not in labels (got {m2['labels']})")
        check(m2["stats"]["n_projects"] == 2,
              f"after purge: stats.n_projects == 2 (got {m2['stats']['n_projects']})")

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
