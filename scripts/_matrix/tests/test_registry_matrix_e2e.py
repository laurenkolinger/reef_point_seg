#!/usr/bin/env python
"""End-to-end seam test: export_yolo's registry hook -> canonical mask
registry -> _matrix.builder.build_matrix.

This is the integration test that would have caught the project_id key
mismatch bug: the export registry hook used to store project_id as the
run_-PREFIXED export_dir segment (e.g. "run_test_5june2026_..."), while the
matrix grouped/looked up registry rows by project.json's un-prefixed 'id'
field (e.g. "test_5june2026_..."). The two never matched, so
`build_matrix` always fell back to the ledger/legacy path and the
registry-derived branch in `_matrix/builder.py` (the `if project_id in
registry_by_project:` branch, whose rows are tagged source == "registry")
never fired in production.

This test builds a real inprocess/run_<id>/ project directory with a
project.json, sets TCRMP_PROJECT_ID to that SAME un-prefixed id (mirroring
what the orchestrator stamps into os.environ for the annotator apps before
calling export_yolo.export_batch), exports one accepted manual PA mask
through the real hook, and then asserts build_matrix's PA cell for that
image traces back to source == "registry" -- proving the hook -> registry
-> matrix seam actually connects end to end with matching project_ids.

Run with:
  env/bin/python scripts/_matrix/tests/test_registry_matrix_e2e.py

Exits nonzero on any failed assertion.
"""

import json
import os
import shutil
import sys
import tempfile

from PIL import Image

# Make the packages importable: scripts/_matrix/tests/ -> scripts/ on path.
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


def _img(d, name):
    p = os.path.join(d, name)
    os.makedirs(d, exist_ok=True)
    Image.new("RGB", (64, 48), (1, 2, 3)).save(p)
    return p


def main():
    root = tempfile.mkdtemp(prefix="registry_matrix_e2e_")
    try:
        # --- Build the inprocess tree: one run_<id>/ project with a
        # project.json whose 'id' is UN-PREFIXED, exactly like the real
        # orchestrator writes it. ---
        project_id = "test_5june2026_e2e"
        run_dir = os.path.join(root, f"run_{project_id}")
        _write(
            os.path.join(run_dir, "project.json"),
            json.dumps(
                {
                    "id": project_id,
                    "name": "E2E Registry Matrix Project",
                    "steps": {"3": {"config": {"target_species": "PA"}}},
                },
                indent=2,
            ),
        )

        # A step4test export dir under the run_ directory, mirroring where
        # the real annotator apps call export_yolo.export_batch from.
        export_dir = os.path.join(run_dir, "step4test_exportImages")
        os.makedirs(export_dir, exist_ok=True)

        fn = "TCRMP20260605_clip_SCP_T101.jpeg"
        img_path = _img(export_dir, fn)

        segs = {
            fn: {
                "image_path_abs": img_path,
                "image_width": 64,
                "image_height": 48,
                "masks": [
                    {
                        "id": 0,
                        "status": "accepted",
                        "species": "PA",
                        "category": "Coral",
                        "source_type": "manual_click",
                        "source_x": 12,
                        "source_y": 34,
                        "review": False,
                        "rle": {"counts": "abc123", "size": [48, 64]},
                        "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]],
                        "image_path_abs": img_path,
                    },
                ],
            },
        }

        # Point the registry at the SAME _mask_registry the matrix will
        # scan under inprocess_root (root here), and set TCRMP_PROJECT_ID to
        # the SAME un-prefixed id project.json declares -- exactly what the
        # orchestrator does for the annotator apps before they export.
        registry_dir = os.path.join(root, "_mask_registry")
        os.environ["TCRMP_MASK_REGISTRY_DIR"] = registry_dir
        os.environ["TCRMP_PROJECT_ID"] = project_id

        # Import export_yolo fresh (combinedAnnotate's copy; the hook is
        # identical across all three sub-tools) and run the real hook via
        # export_batch, exactly as production does.
        _ca_src = os.path.abspath(
            os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate", "src")
        )
        if _ca_src not in sys.path:
            sys.path.insert(0, _ca_src)
        import export_yolo as EY  # noqa: E402

        class_map = {}
        stats = EY.export_batch(segs, export_dir, class_map, symlink=False)
        check(isinstance(stats, dict), "export_batch returns a stats dict")
        check(
            stats.get("exported_masks") == 1,
            f"export_batch exported the accepted PA mask (got {stats.get('exported_masks')})",
        )

        # --- Now build the matrix over the SAME inprocess root and confirm
        # the PA cell for this image traces back to the registry. ---
        m = build_matrix(root)

        check(
            m["stats"]["n_projects"] == 1,
            f"build_matrix sees the one project (got {m['stats']['n_projects']})",
        )

        basename = "TCRMP20260605_clip_SCP_T101"
        cell = m["cells"].get(basename, {}).get("PA", {})
        check(
            cell.get("outcome") == "found_manual",
            f"cells[{basename}][PA].outcome == found_manual (got {cell.get('outcome')})",
        )

        sources = cell.get("sources", [])
        check(
            bool(sources) and sources[0].get("source") == "registry",
            "the PA cell's source is tagged 'registry' -- proving the hook's "
            "write and the matrix's read agreed on the SAME (unprefixed) "
            f"project_id and the registry branch actually fired (got {sources})",
        )
        check(
            bool(sources) and sources[0].get("project_id") == project_id,
            f"the registry-derived source carries the matching project_id (got {sources})",
        )

        # Belt-and-suspenders: confirm the registry row itself was written
        # with the un-prefixed project_id (not the run_-prefixed dir name),
        # which is the root-cause assertion for the whole bug.
        from _reefreview.mask_registry import MaskRegistry  # noqa: E402

        reg = MaskRegistry(root=registry_dir)
        rows = reg.rows()
        pa_rows = [r for r in rows if r.get("species") == "PA"]
        check(len(pa_rows) == 1, f"exactly one PA registry row written (got {pa_rows})")
        if pa_rows:
            check(
                pa_rows[0].get("project_id") == project_id,
                "registry row's project_id is the UN-PREFIXED id matching "
                f"project.json (got {pa_rows[0].get('project_id')!r}, expected {project_id!r})",
            )

    finally:
        os.environ.pop("TCRMP_MASK_REGISTRY_DIR", None)
        os.environ.pop("TCRMP_PROJECT_ID", None)
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
