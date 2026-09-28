"""Annotate Images sessions as a training source (decision record 0006, 2026-09-17, LO).

A standalone annotation session under inprocess/_standalone/<session>/ writes a
real YOLO export (all_images/, all_labels/, data.yaml). Until this change the
Train step's source list never offered it, so a student's annotation had no
route to a model. This suite pins: sessions with an export are listed by
/api/composites with kind "standalone", task seg and honest counts; a session
without an export is not; the launch accepts a standalone source and refuses
one that escapes the root; hostile session names pass through as data.

No pytest. Run with the unified env python. subprocess is stubbed at module
level. Fixtures live under one temp folder removed at the end, verified.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)
os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")
_RESULTS = []
_FIXTURES = []


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


subprocess.run = lambda *a, **k: _FakeCompleted()


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen
import app as appmod  # noqa: E402

_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()


def check(cond, msg):
    _RESULTS.append((msg, bool(cond), msg)); print(("PASS " if cond else "FAIL ") + msg)


def run(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        import traceback
        _RESULTS.append((fn.__name__, False, traceback.format_exc())); print(f"FAIL {fn.__name__}: {exc}")


def _root():
    tmp = tempfile.mkdtemp(prefix="standalone_src_", dir=_TMP_ROOT); _FIXTURES.append(tmp)
    root = os.path.join(tmp, "_standalone"); os.makedirs(root); return root


def _session(root, name, images=3, labels=2, with_yaml=True):
    d = os.path.join(root, name); os.makedirs(os.path.join(d, "all_images")); os.makedirs(os.path.join(d, "all_labels"))
    for i in range(images):
        open(os.path.join(d, "all_images", f"img_{i}.jpg"), "wb").write(b"\xff\xd8")
    for i in range(labels):
        open(os.path.join(d, "all_labels", f"img_{i}.txt"), "w").write("0 0.1 0.1 0.2 0.2 0.3 0.3\n0 0.5 0.5 0.6 0.6 0.7 0.7\n")
    if with_yaml:
        open(os.path.join(d, "data.yaml"), "w").write("names:\n  0: Porites astreoides\n")
    return d


class _Pinned:
    def __init__(self, root):
        self.root = root
    def __enter__(self):
        self._s = (appmod._standalone_root, appmod._composite_root)
        appmod._standalone_root = lambda: self.root
        appmod._composite_root = lambda: os.path.join(os.path.dirname(self.root), "_composite")
        return self
    def __exit__(self, *_):
        appmod._standalone_root, appmod._composite_root = self._s


def test_sessions_with_an_export_are_listed_with_honest_counts():
    root = _root(); _session(root, "26-09-17_LO_reef_photos", images=3, labels=2)
    with _Pinned(root):
        rows = [r for r in appmod._list_standalone_exports()]
    check(len(rows) == 1, f"one session listed, got {len(rows)}")
    r = rows[0] if rows else {}
    check(r.get("kind") == "standalone" and r.get("task") == "seg", f"kind standalone, task seg: {r}")
    check(r.get("name") == "26-09-17_LO_reef_photos", "named by the session folder")
    check(r.get("n_images") == 3 and r.get("n_masks") == 4, f"counts from all_images and the polygon lines in all_labels: {r.get('n_images')}/{r.get('n_masks')}")
    check(os.path.isfile(os.path.join(r.get("path", ""), "data.yaml")), "path is the export folder holding data.yaml")


def test_a_session_without_an_export_is_not_offered():
    root = _root(); _session(root, "unfinished", with_yaml=False); os.makedirs(os.path.join(root, "empty_dir"))
    open(os.path.join(root, "a_file_not_a_session"), "w").write("x")
    with _Pinned(root):
        rows = appmod._list_standalone_exports()
    check(rows == [], f"no data.yaml, no export: nothing listed, got {[r['name'] for r in rows]}")


def test_api_composites_carries_standalone_rows_beside_the_others():
    root = _root(); _session(root, "s1")
    with _Pinned(root):
        client = appmod.create_app().test_client()
        body = client.get("/api/composites").get_json() or {}
    kinds = [c.get("kind") for c in body.get("composites", [])]
    check("standalone" in kinds, f"/api/composites lists the session: kinds {kinds}")
    row = next((c for c in body["composites"] if c.get("kind") == "standalone"), {})
    check(row.get("usable_here") is True and row.get("refusal") == "", "a session is usable by an outline project")


def test_validate_outline_source_accepts_a_session_and_refuses_an_escape():
    root = _root(); d = _session(root, "s1"); outside = tempfile.mkdtemp(prefix="standalone_out_", dir=_TMP_ROOT); _FIXTURES.append(outside)
    os.makedirs(os.path.join(outside, "all_images")); open(os.path.join(outside, "data.yaml"), "w").write("names: {}\n")
    link = os.path.join(root, "escape"); os.symlink(outside, link)
    with _Pinned(root):
        ok_path, ok_err = appmod._validate_outline_source(d)
        bad_path, bad_err = appmod._validate_outline_source(link)
        no_yaml = _session(root, "s2", with_yaml=False)
        ny_path, ny_err = appmod._validate_outline_source(no_yaml)
    check(ok_path == os.path.realpath(d) and ok_err == "", f"a session export is a valid outline source: {ok_err!r}")
    check(bad_path == "" and bad_err, "a symlink escaping the standalone root is refused")
    check(ny_path == "" and "data.yaml" in ny_err, "a session without data.yaml is refused with the reason")


def test_hostile_session_names_pass_through_as_data():
    root = _root(); hostile = 'x<img src=x onerror=alert(1)>'; _session(root, hostile)
    with _Pinned(root):
        rows = appmod._list_standalone_exports()
    check(rows and rows[0]["name"] == hostile, "the name is returned verbatim; escaping is the renderer's job")


def test_corrupt_yaml_and_unreadable_labels_do_not_take_the_list_down():
    root = _root(); d = _session(root, "s1")
    open(os.path.join(d, "data.yaml"), "wb").write(b"\x00\x01[[[")
    open(os.path.join(d, "all_labels", "img_0.txt"), "wb").write(b"\xff\xfe garbage")
    with _Pinned(root):
        rows = appmod._list_standalone_exports()
    check(len(rows) == 1 and isinstance(rows[0].get("n_masks"), int), "a corrupt yaml or label file leaves the session listed with a best-effort count")


def test_a_symlinked_session_escaping_the_root_is_not_listed():
    root = _root(); _session(root, "real")
    outside = tempfile.mkdtemp(prefix="standalone_out_", dir=_TMP_ROOT); _FIXTURES.append(outside)
    os.makedirs(os.path.join(outside, "all_images")); open(os.path.join(outside, "data.yaml"), "w").write("names: {}\n")
    os.symlink(outside, os.path.join(root, "escape"))
    with _Pinned(root):
        names = [r["name"] for r in appmod._list_standalone_exports()]
    check(names == ["real"], f"the lister never offers a session the launch would refuse: {names}")


def test_rows_share_the_training_set_shape_and_read_session_json():
    root = _root(); d = _session(root, "s1")
    open(os.path.join(d, "session.json"), "w").write(json.dumps({"created": "2026-09-17 09:12:00", "initials": "LO"}))
    with _Pinned(root):
        r = appmod._list_standalone_exports()[0]
    check(r.get("created_at") == "2026-09-17 09:12:00" and r.get("reviewer") == "LO", f"created_at and reviewer come from session.json: {r}")
    check(r.get("labels") == ["Porites astreoides"], f"labels come from data.yaml as a training set's do: {r.get('labels')}")
    check("created" not in r, "no stray created key beside created_at")


if __name__ == "__main__":
    for fn in [test_sessions_with_an_export_are_listed_with_honest_counts, test_a_session_without_an_export_is_not_offered,
               test_api_composites_carries_standalone_rows_beside_the_others, test_validate_outline_source_accepts_a_session_and_refuses_an_escape,
               test_hostile_session_names_pass_through_as_data, test_corrupt_yaml_and_unreadable_labels_do_not_take_the_list_down,
               test_a_symlinked_session_escaping_the_root_is_not_listed, test_rows_share_the_training_set_shape_and_read_session_json]:
        run(fn)
    for p in _FIXTURES: shutil.rmtree(p, ignore_errors=True)
    leftovers = [p for p in _FIXTURES if os.path.exists(p)]
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS)-len(failed)}/{len(_RESULTS)} passed; fixtures removed: {not leftovers}")
    if failed:
        print("\nFAILURES:"); [print(f"--- {n} ---\n{str(d)[:300]}") for n,_,d in failed]
    sys.exit(1 if failed or leftovers else 0)
