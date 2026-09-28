#!/usr/bin/env python
"""No-pytest harness for every matrix blueprint route (blueprint.py) via
the Flask test client with a stubbed open_folder_fn. subprocess.run and
subprocess.Popen are stubbed at module level so nothing is ever spawned.

Run with:
  env/bin/python scripts/_matrix/tests/test_matrix_routes.py
"""

import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _blocked(*_a, **_k):
    raise RuntimeError("subprocess blocked in tests")


subprocess.run = _blocked
_REAL_POPEN = subprocess.Popen
subprocess.Popen = _blocked

from flask import Flask  # noqa: E402

from _matrix import blueprint as bpmod  # noqa: E402
from _matrix.preferences import Preferences  # noqa: E402
from _matrix.tests._fixture_cell import T101, T102, T403, build_root, write  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


class _FakeProc:
    def __init__(self, returncode, err=b"", hang=False):
        self.returncode = returncode
        self._err = err
        self._hang = hang

    def communicate(self, timeout=None):
        if self._hang:
            raise subprocess.TimeoutExpired("xdg-open", timeout)
        return b"", self._err


def main():
    root = tempfile.mkdtemp(prefix="matrix_routes_test_")
    outside = tempfile.mkdtemp(prefix="matrix_routes_outside_")
    # Name resolution (Task 2): a master_codes.csv the paths_provider points
    # at so /api/data and /api/stats can resolve PA to a full name.
    codes_dir = tempfile.mkdtemp(prefix="matrix_routes_codes_")
    try:
        fx = build_root(root)

        # A run_* symlink whose target lives outside inprocess: must be 403.
        ext = os.path.join(outside, "run_linked_20260826_zzzzzz")
        write(os.path.join(ext, "project.json"), '{"id": "linked", "name": "Linked"}')
        os.symlink(ext, os.path.join(root, "run_linked_20260826_zzzzzz"))

        master_codes_csv = os.path.join(codes_dir, "master_codes.csv")
        write(master_codes_csv, "code,category,name\nPA,Coral,Porites astreoides\n")

        calls = []
        outcome = {"ok": True, "err": ""}

        def open_folder_stub(path):
            calls.append(path)
            return outcome["ok"], outcome["err"]

        app = Flask("matrix_routes_test")
        app.register_blueprint(
            bpmod.make_blueprint(lambda: {"inprocess_root": root, "master_codes_csv": master_codes_csv},
                                 log_fn=lambda m: None, open_folder_fn=open_folder_stub),
            url_prefix="/matrix")
        c = app.test_client()

        # ---- /api/data ----
        r = c.get("/matrix/api/data")
        data = r.get_json()
        check(r.status_code == 200 and data["stats"]["n_projects"] == 4, f"/api/data 200 (got {r.status_code})")
        pa = next(p for p in data["projects"] if p["project_id"] == "projA")
        check(pa["opened_by"] == "AB" and pa["initials"] == "LO" and pa["run_dir"] == "run_projA_20260826_aaaaaa"
              and pa["project_dir"] == fx["proj_a"] and pa["created_at"], f"/api/data project identity fields (got {pa})")
        src = data["cells"][_stem(T101)]["PA"]["sources"][0]
        check({"uid", "status", "source_type", "reviewer"} <= set(src), f"/api/data source fields (got {sorted(src)})")
        check("label_names" in data and isinstance(data["label_names"], dict), "/api/data carries label_names")
        check("site_names" in data and isinstance(data["site_names"], dict), "/api/data carries site_names")
        check(data["label_names"].get("PA") == "Porites astreoides",
              f"/api/data label_names resolves PA via master_codes_csv (got {data['label_names']})")

        # ---- /api/cell ----
        r = c.get(f"/matrix/api/cell?image={_stem(T101)}&label=PA")
        cell = r.get_json()
        check(r.status_code == 200 and [p["project_id"] for p in cell["projects"]] == ["projA", "projB"],
              f"/api/cell 200 with both projects (got {r.status_code})")
        check(cell["projects"][0]["image_url"] == f"/matrix/img/projA/{_stem(T101)}",
              f"/api/cell image_url mount-aware (got {cell['projects'][0]['image_url']})")
        check(cell["preference"] is None and cell["outcome"] == "found_expert", "/api/cell outcome + no preference")
        r = c.get("/matrix/api/cell?image=&label=PA")
        check(r.status_code == 400 and "error" in r.get_json(), "/api/cell without image -> 400 JSON")
        r = c.get("/matrix/api/cell?image=nope&label=PA")
        check(r.status_code == 404 and r.get_json()["error"], "/api/cell unknown image -> 404 JSON")

        # ---- /img ----
        r = c.get(f"/matrix/img/projA/{_stem(T101)}")
        with open(fx["img_a"][T101], "rb") as fh:
            raw = fh.read()
        check(r.status_code == 200 and r.data == raw, "/img serves the bytes of image_path_abs")
        r = c.get(f"/matrix/img/projA/{T101}")
        check(r.status_code == 200, "/img accepts the filename form too")
        r = c.get(f"/matrix/img/projB/{_stem(T403)}")
        check(r.status_code == 404 and r.get_json()["error"], "/img project without that image -> 404 JSON")
        r = c.get("/matrix/img/nope/x")
        check(r.status_code == 404 and r.get_json()["error"], "/img unknown project -> 404 JSON")
        r = c.get("/matrix/img/projA/..%2F..%2Fetc%2Fpasswd")
        check(r.status_code == 404, "/img never serves a path from the query")

        # ---- /api/open-folder ----
        r = c.post("/matrix/api/open-folder", json={"project_id": "projA"})
        body = r.get_json()
        check(r.status_code == 200 and body["success"] is True and body["path"] == fx["proj_a"],
              f"/api/open-folder 200 with path (got {r.status_code} {body})")
        check(calls == [fx["proj_a"]], f"open_folder_fn called with the run dir (got {calls})")
        r = c.post("/matrix/api/open-folder", json={"project_id": "run_projB_20260826_bbbbbb"})
        check(r.status_code == 200 and r.get_json()["path"] == fx["proj_b"], "/api/open-folder accepts the run_dir name")
        r = c.post("/matrix/api/open-folder", json={})
        check(r.status_code == 400 and "path" in r.get_json(), "/api/open-folder missing project_id -> 400 with path key")
        r = c.post("/matrix/api/open-folder", json={"project_id": "nope"})
        check(r.status_code == 400 and "path" in r.get_json(), "/api/open-folder unknown project -> 400 with path key")
        r = c.post("/matrix/api/open-folder", json={"project_id": "linked"})
        check(r.status_code == 403 and r.get_json()["path"], "/api/open-folder run_* symlink outside inprocess -> 403 with path")
        n_calls = len(calls)
        outcome["ok"], outcome["err"] = False, "no display"
        r = c.post("/matrix/api/open-folder", json={"project_id": "projA"})
        body = r.get_json()
        check(r.status_code == 500 and "no display" in body["error"] and body["path"] == fx["proj_a"],
              f"/api/open-folder spawn failure -> 500 carrying the error and path (got {body})")
        outcome["ok"], outcome["err"] = True, ""
        check(len(calls) == n_calls + 1, "the 403 case never reached open_folder_fn")

        # ---- default_open_folder with a stubbed Popen ----
        bpmod.subprocess.Popen = lambda *a, **k: _FakeProc(None, hang=True)
        check(bpmod.default_open_folder("/x") == (True, ""), "default_open_folder: still running after 3 s counts as success")
        bpmod.subprocess.Popen = lambda *a, **k: _FakeProc(1, err=b"gio: nope")
        ok, err = bpmod.default_open_folder("/x")
        check(ok is False and "gio: nope" in err, "default_open_folder: fast non-zero exit surfaces stderr")
        bpmod.subprocess.Popen = _blocked
        ok, err = bpmod.default_open_folder("/x")
        check(ok is False and "blocked" in err, "default_open_folder: spawn exception -> (False, text)")

        # ---- preferences ----
        r = c.get("/matrix/api/preferences")
        check(r.status_code == 200 and r.get_json() == {"preferences": []}, "/api/preferences empty")
        r = c.post("/matrix/api/prefer", json={"image": _stem(T101), "label": "PA", "project_id": "projB", "by": ""})
        check(r.status_code == 400, "/api/prefer blank by -> 400")
        r = c.post("/matrix/api/prefer", json={"image": _stem(T102), "label": "DSTO", "project_id": "projA", "by": "LO"})
        check(r.status_code == 400 and "no accepted" in r.get_json()["error"],
              "/api/prefer project with no accepted mask of that label -> 400")
        r = c.post("/matrix/api/prefer", json={"image": _stem(T101), "label": "PA", "project_id": "projC", "by": "LO"})
        check(r.status_code == 400, "/api/prefer project that does not hold the image -> 400")
        r = c.post("/matrix/api/prefer", json={"image": _stem(T101), "label": "PA", "project_id": "projA",
                                                "uids": ["not-a-uid"], "by": "LO"})
        check(r.status_code == 400 and "uid" in r.get_json()["error"], "/api/prefer unknown uid -> 400")
        r = c.post("/matrix/api/prefer", json={"image": T101, "label": "PA", "project_id": "projA",
                                                "uids": [fx["uids_a"][(T101, 0)]], "by": "lo", "note": "n"})
        body = r.get_json()
        check(r.status_code == 200 and body["success"] is True and body["preference"]["chosen_by"] == "LO"
              and body["preference"]["basename"] == _stem(T101) and body["preference"]["uids"] == [fx["uids_a"][(T101, 0)]],
              f"/api/prefer valid -> 200 with the row (got {body})")
        r = c.get("/matrix/api/preferences")
        check(len(r.get_json()["preferences"]) == 1, "/api/preferences lists the row")
        r = c.get(f"/matrix/api/cell?image={_stem(T101)}&label=PA")
        check(r.get_json()["preference"]["project_id"] == "projA", "/api/cell now carries the preference")
        r = c.post("/matrix/api/prefer/clear", json={"image": _stem(T101), "label": "PA"})
        check(r.status_code == 200 and r.get_json() == {"success": True, "removed": True}, "/api/prefer/clear -> success")
        r = c.post("/matrix/api/prefer/clear", json={"image": _stem(T101)})
        check(r.status_code == 400, "/api/prefer/clear missing label -> 400")
        check(c.get("/matrix/api/preferences").get_json()["preferences"] == [], "preferences empty after clear")

        # L15: a run_dir name resolves like /api/open-folder; the row keeps the project.json id.
        r = c.post("/matrix/api/prefer", json={"image": _stem(T101), "label": "PA",
                                                "project_id": "run_projA_20260826_aaaaaa", "by": "LO"})
        body = r.get_json()
        check(r.status_code == 200 and body["preference"]["project_id"] == "projA",
              f"/api/prefer accepts the run_dir name and stores the project id (got {r.status_code} {body})")
        r = c.get(f"/matrix/api/cell?image={_stem(T101)}&label=PA")
        pref = r.get_json()["preference"]
        check(pref["project_id"] == "projA" and pref["stale"] is False,
              f"/api/cell preference.stale false while the project holds the image (got {pref})")
        r = c.post("/matrix/api/prefer", json={"image": _stem(T101), "label": "PA", "project_id": "run_nope", "by": "LO"})
        check(r.status_code == 400 and "does not hold" in r.get_json()["error"],
              "/api/prefer unknown run_dir -> 400 does not hold")
        # L3: a saved choice whose project vanished is flagged stale.
        Preferences(root).set(_stem(T101), "PA", "gone_project", [], by="LO")
        r = c.get(f"/matrix/api/cell?image={_stem(T101)}&label=PA")
        pref = r.get_json()["preference"]
        check(pref["project_id"] == "gone_project" and pref["stale"] is True,
              f"/api/cell preference.stale true when no project holds the image for that id (got {pref})")
        Preferences(root).clear(_stem(T101), "PA")
        check(c.get("/matrix/api/preferences").get_json()["preferences"] == [], "preferences empty again")

        # ---- stats ----
        r = c.get("/matrix/api/stats?status=bogus")
        check(r.status_code == 400, "/api/stats bad status -> 400")
        r = c.get("/matrix/api/stats")
        s = r.get_json()
        check(r.status_code == 200 and s["status"] == "accepted" and s["totals"]["masks"] == 5
              and s["excluded"]["test_residue_rows"] == 3, f"/api/stats default (got {s.get('totals')})")
        check("label_names" in s and isinstance(s["label_names"], dict), "/api/stats carries label_names")
        check("site_names" in s and isinstance(s["site_names"], dict), "/api/stats carries site_names")
        r = c.get("/matrix/api/stats?status=all")
        check(r.get_json()["totals"]["masks"] == 8, "/api/stats?status=all")

        # ---- composites ----
        r = c.get("/matrix/api/composites")
        check(r.status_code == 200 and r.get_json() == {"composites": []}, "/api/composites empty")
        r = c.post("/matrix/api/composite", json={"name": "routes test", "labels": None, "by": "lo"})
        body = r.get_json()
        check(r.status_code == 200 and body["success"] is True and body["n_masks"] == 3 and body["n_images"] == 4
              and body["n_negatives"] == 2 and body["skipped_review_only"] == 1 and body["by_label"] == {"OFAV": 1, "PA": 2}
              and body["out_dir"] == os.path.join(root, "_composite", "routes_test"),
              f"/api/composite builds (got {r.status_code} {body})")
        comp_dir = body["out_dir"]
        r = c.post("/matrix/api/composite", json={"name": "routes test"})
        check(r.status_code == 409 and r.get_json()["error"], "/api/composite existing name -> 409")
        body = r.get_json()
        check(body.get("path") == comp_dir and body.get("out_dir") == comp_dir,
              f"L4: 409 body carries both path and out_dir (got {body})")
        r = c.post("/matrix/api/composite", json={"name": "bad", "labels": "PA"})
        check(r.status_code == 400, "/api/composite labels not a list -> 400")
        r = c.post("/matrix/api/composite", json={"labels": ["PA"], "by": "LO"})
        check(r.status_code == 200 and r.get_json()["name"].startswith("composite_"), "/api/composite default name")
        r = c.get("/matrix/api/composites")
        names = [x["name"] for x in r.get_json()["composites"]]
        check("routes_test" in names and len(names) == 2, f"/api/composites lists both (got {names})")
        entry = next(x for x in r.get_json()["composites"] if x["name"] == "routes_test")
        check({"name", "path", "created_at", "n_images", "n_masks", "labels"} <= set(entry), "composite listing shape")
        check(len(entry["created_at"]) == 19 and entry["created_at"][10] == " ",
              f"L13: listing created_at is 'YYYY-MM-DD HH:MM:SS' (got {entry['created_at']!r})")

        # L13: a symlink under _composite that resolves outside is never listed.
        ext_comp = os.path.join(outside, "ext_composite")
        write(os.path.join(ext_comp, "data.yaml"), "nc: 1\n")
        sneaky = os.path.join(root, "_composite", "sneaky")
        os.symlink(ext_comp, sneaky)
        names = [x["name"] for x in c.get("/matrix/api/composites").get_json()["composites"]]
        check("sneaky" not in names and len(names) == 2, f"/api/composites skips a symlink out of _composite (got {names})")

        # ---- M1: /api/open-folder {path} for a composite dir ----
        n_calls = len(calls)
        r = c.post("/matrix/api/open-folder", json={"path": comp_dir})
        body = r.get_json()
        check(r.status_code == 200 and body["success"] is True and body["path"] == comp_dir,
              f"/api/open-folder {{path}} composite dir -> 200 with path (got {r.status_code} {body})")
        check(calls[-1] == comp_dir and len(calls) == n_calls + 1, f"open_folder_fn called with the composite dir (got {calls[-1:]})")
        r = c.post("/matrix/api/open-folder", json={"path": comp_dir + os.sep})
        check(r.status_code == 200 and r.get_json()["path"] == comp_dir, "/api/open-folder {path} trailing slash normalized")
        r = c.post("/matrix/api/open-folder", json={"path": os.path.join(root, "_composite")})
        check(r.status_code == 403 and r.get_json()["path"], "/api/open-folder {path} _composite root itself -> 403 with path")
        r = c.post("/matrix/api/open-folder", json={"path": os.path.join(comp_dir, "all_images")})
        check(r.status_code == 403 and r.get_json()["path"], "/api/open-folder {path} two levels down -> 403 with path")
        r = c.post("/matrix/api/open-folder", json={"path": fx["proj_a"]})
        check(r.status_code == 403 and r.get_json()["path"] == fx["proj_a"], "/api/open-folder {path} run dir -> 403 with path")
        r = c.post("/matrix/api/open-folder", json={"path": sneaky})
        check(r.status_code == 403 and r.get_json()["path"], "/api/open-folder {path} symlink out of _composite -> 403 with path")
        r = c.post("/matrix/api/open-folder", json={"path": os.path.join(root, "_composite", "missing")})
        body = r.get_json()
        check(r.status_code == 400 and body["path"].endswith("missing"), f"/api/open-folder {{path}} missing dir -> 400 with path (got {body})")
        r = c.post("/matrix/api/open-folder", json={"path": "/"})
        check(r.status_code == 403 and r.get_json()["path"] == "/", "/api/open-folder {path} filesystem root -> 403")
        r = c.post("/matrix/api/open-folder", json={"project_id": "projA", "path": comp_dir})
        check(r.status_code == 200 and r.get_json()["path"] == fx["proj_a"], "/api/open-folder project_id wins when both are sent")
        outcome["ok"], outcome["err"] = False, "no display"
        r = c.post("/matrix/api/open-folder", json={"path": comp_dir})
        body = r.get_json()
        check(r.status_code == 500 and "no display" in body["error"] and body["path"] == comp_dir,
              f"/api/open-folder {{path}} spawn failure -> 500 carrying error and path (got {body})")
        outcome["ok"], outcome["err"] = True, ""
        check(len(calls) == n_calls + 4, f"only the accepted {{path}} / project_id cases reached open_folder_fn (got {len(calls) - n_calls})")
        os.unlink(sneaky)

        # ---- L6: nothing to export -> 400, nothing written ----
        empty = tempfile.mkdtemp(prefix="matrix_routes_empty_")
        try:
            app2 = Flask("matrix_routes_empty")
            app2.register_blueprint(
                bpmod.make_blueprint(lambda: {"inprocess_root": empty}, log_fn=lambda m: None,
                                     open_folder_fn=open_folder_stub),
                url_prefix="/matrix")
            r = app2.test_client().post("/matrix/api/composite", json={"name": "void", "labels": None, "by": "LO"})
            body = r.get_json()
            check(r.status_code == 400 and "nothing to export" in body["error"],
                  f"/api/composite with no labeled mask anywhere -> 400 (got {r.status_code} {body})")
            check(not os.path.exists(os.path.join(empty, "_composite")), "empty-label composite writes nothing")
        finally:
            shutil.rmtree(empty, ignore_errors=True)
        check(not os.path.exists(empty), "empty root removed")

        # ---- purge ----
        r = c.post("/matrix/api/registry/purge-test-rows", json={})
        check(r.status_code == 400, "purge without confirm -> 400")
        r = c.post("/matrix/api/registry/purge-test-rows", json={"confirm": True})
        body = r.get_json()
        check(r.status_code == 200 and body["success"] is True and body["removed"] == 3 and os.path.isfile(body["backup"]),
              f"purge with confirm -> removed 3, backup path (got {body})")
        check(c.get("/matrix/api/stats").get_json()["excluded"]["test_residue_rows"] == 0, "stats clean after purge")
    finally:
        subprocess.Popen = _REAL_POPEN
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)
        shutil.rmtree(codes_dir, ignore_errors=True)
    check(not os.path.exists(root) and not os.path.exists(outside) and not os.path.exists(codes_dir),
          "temp fixtures removed")

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
