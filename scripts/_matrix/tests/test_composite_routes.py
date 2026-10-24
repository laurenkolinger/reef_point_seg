#!/usr/bin/env python
"""No-pytest harness for the /matrix/api/composite route additions from
Task 6: source_types validation (400 on an unknown value) and rebuild (bool
validation, atomic swap, 409 preserved without it). The build_composite
resolution/filter logic itself is covered by test_composite.py; this file
only exercises the Flask route layer (blueprint.py) via the test client,
mirroring the stubbing pattern in test_matrix_routes.py so nothing is ever
spawned.

Run with:
  env/bin/python scripts/_matrix/tests/test_composite_routes.py
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

import json
import os
import shutil
import subprocess
import sys
import tempfile


def _blocked(*_a, **_k):
    raise RuntimeError("subprocess blocked in tests")


subprocess.run = _blocked
subprocess.Popen = _blocked

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from flask import Flask  # noqa: E402

from _matrix import blueprint as bpmod  # noqa: E402
from _matrix.composite import SOURCE_TYPES  # noqa: E402
from _matrix.tests._fixture_cell import T2015, build_root  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def main():
    root = tempfile.mkdtemp(prefix="matrix_composite_routes_test_")
    try:
        build_root(root)
        app = Flask("matrix_composite_routes_test")
        app.register_blueprint(
            bpmod.make_blueprint(lambda: {"inprocess_root": root}, log_fn=lambda m: None),
            url_prefix="/matrix")
        c = app.test_client()

        # ---- source_types validation ----
        r = c.post("/matrix/api/composite", json={"name": "bad_st", "source_types": "manual_click"})
        check(r.status_code == 400 and "list" in r.get_json()["error"],
              f"source_types not a list -> 400 (got {r.status_code} {r.get_json()})")

        r = c.post("/matrix/api/composite", json={"name": "bad_st2", "source_types": ["manual_click", "bogus"]})
        body = r.get_json()
        check(r.status_code == 400 and "bogus" in body["error"] and body.get("allowed") == sorted(SOURCE_TYPES),
              f"unknown source_type -> 400 with the allowed set (got {r.status_code} {body})")

        r = c.post("/matrix/api/composite", json={"name": "bad_rebuild", "rebuild": "yes"})
        check(r.status_code == 400 and "rebuild" in r.get_json()["error"],
              f"rebuild not a boolean -> 400 (got {r.status_code} {r.get_json()})")

        # ---- a valid source_types list actually filters ----
        r_all = c.post("/matrix/api/composite", json={"name": "routes_all", "by": "lo"})
        body_all = r_all.get_json()
        check(r_all.status_code == 200 and body_all["source_types"] is None and body_all["by_label"].get("OFAV") == 1,
              f"no source_types -> null, every source type included (got {r_all.status_code} {body_all})")

        r_manual = c.post("/matrix/api/composite", json={
            "name": "routes_manual", "by": "lo", "source_types": ["manual_box", "manual_click"],
        })
        body_manual = r_manual.get_json()
        check(r_manual.status_code == 200 and body_manual["source_types"] == ["manual_box", "manual_click"],
              f"valid source_types recorded sorted in the response (got {r_manual.status_code} {body_manual})")
        check("OFAV" not in body_manual["by_label"],
              f"manual-only filter drops the accepted auto OFAV mask over the route (got {body_manual['by_label']})")
        check(_stem(T2015) in body_manual["negatives"],
              f"T2015's only PA candidate is auto, excluded, becomes a negative (got {body_manual['negatives']})")

        # ---- rebuild: 409 without it, 200 + history with it ----
        r_first = c.post("/matrix/api/composite", json={"name": "routes_rebuild", "by": "lo"})
        check(r_first.status_code == 200, f"first build of routes_rebuild -> 200 (got {r_first.status_code})")

        r_conflict = c.post("/matrix/api/composite", json={"name": "routes_rebuild", "by": "lo"})
        check(r_conflict.status_code == 409 and r_conflict.get_json()["error"],
              f"same name without rebuild -> 409 (got {r_conflict.status_code} {r_conflict.get_json()})")

        r_rebuild = c.post("/matrix/api/composite", json={
            "name": "routes_rebuild", "by": "ab", "rebuild": True,
            "source_types": ["manual_box", "manual_click"],
        })
        body_rebuild = r_rebuild.get_json()
        check(r_rebuild.status_code == 200 and body_rebuild["by"] == "AB",
              f"rebuild: true replaces the existing composite -> 200 (got {r_rebuild.status_code} {body_rebuild})")
        check("OFAV" not in body_rebuild["by_label"],
              f"rebuilt content reflects the new source_types filter (got {body_rebuild['by_label']})")
        with open(os.path.join(body_rebuild["out_dir"], "composite.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        check(isinstance(meta.get("history"), list) and len(meta["history"]) == 1 and meta["history"][0]["by"] == "LO",
              f"composite.json history carries the previous build after a route-triggered rebuild (got {meta.get('history')})")
        check(not os.path.exists(body_rebuild["out_dir"] + ".rebuild-tmp")
              and not os.path.exists(body_rebuild["out_dir"] + ".rebuild-backup"),
              "no leftover swap directories after a route-triggered rebuild")

        r_listing = c.get("/matrix/api/composites")
        names = [x["name"] for x in r_listing.get_json()["composites"]]
        check(names.count("routes_rebuild") == 1,
              f"a rebuilt composite still lists once, not as a duplicate (got count={names.count('routes_rebuild')})")

        # ---- project_ids validation + pass-through (Task H) ----
        r = c.post("/matrix/api/composite", json={"name": "bad_pids", "project_ids": "projA"})
        check(r.status_code == 400 and "list" in r.get_json()["error"],
              f"project_ids not a list -> 400 (got {r.status_code} {r.get_json()})")

        r = c.post("/matrix/api/composite", json={"name": "bad_pids2", "project_ids": ["projA", "nope"]})
        check(r.status_code == 400 and "nope" in r.get_json()["error"],
              f"an unknown project id -> 400 naming it (got {r.status_code} {r.get_json()})")

        r = c.post("/matrix/api/composite", json={"name": "routes_filtered", "by": "lo",
                                                  "project_ids": ["projA"]})
        body = r.get_json()
        check(r.status_code == 200 and body.get("project_filter") == ["projA"],
              f"project_ids reaches build_composite (got {r.status_code} {body.get('project_filter')})")
        check([s["project_id"] for s in body.get("source_projects") or []] == ["projA"],
              f"the result carries source_projects (got {body.get('source_projects')})")

        # ---- create-project ----
        r = c.post("/matrix/api/composite/create-project", json={"name": "", "by": "LO"})
        check(r.status_code == 400 and "name" in (r.get_json() or {}).get("error", ""),
              f"an empty name -> 400 (got {r.status_code} {r.get_json()})")
        r = c.post("/matrix/api/composite/create-project", json={"by": "LO"})
        check(r.status_code == 400, f"a missing name -> 400 (got {r.status_code} {r.get_json()})")
        r = c.post("/matrix/api/composite/create-project", json={"name": "routes_filtered", "by": "9"})
        check(r.status_code == 400, f"bad initials -> 400 (got {r.status_code} {r.get_json()})")
        r = c.post("/matrix/api/composite/create-project", json={"name": "ghost_set", "by": "LO"})
        check(r.status_code == 404, f"unknown composite -> 404 (got {r.status_code} {r.get_json()})")

        r = c.post("/matrix/api/composite/create-project", json={"name": "routes_filtered", "by": "lo"})
        body = r.get_json()
        check(r.status_code == 200 and body.get("display_name") == "TRAIN_routes_filtered",
              f"create-project mints the training project (got {r.status_code} {body})")
        check(os.path.islink(os.path.join(body.get("run_dir", ""), "step4test_combinedAnnotate")),
              f"the minted project carries the export symlink (got {body})")
        r_listing = c.get("/matrix/api/composites")
        entry = next(x for x in r_listing.get_json()["composites"] if x["name"] == "routes_filtered")
        check(entry.get("linked_projects") == [body["run_dir_name"]],
              f"the listing reports the linked training project (got {entry.get('linked_projects')})")
        check(entry.get("linked_project_dirs") == [os.path.join(root, rd) for rd in entry["linked_projects"]],
              "listing carries linked_project_dirs as absolute run dirs in linked_projects order")
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
