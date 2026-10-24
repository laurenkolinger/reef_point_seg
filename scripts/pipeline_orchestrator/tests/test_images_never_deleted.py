"""Images are never deleted (Lauren, 2026-10-15: "IMAGES SHOULD NEVER BE DELETED").

Every code path that used to remove an image folder or file now moves it into
a dated holding folder, <parent>/_superseded/<YYYYMMDD-HHMMSS>/..., that no
code prunes. For each path this suite builds a fixture with real JPEG files,
runs the path, and asserts every original image's bytes (by SHA-256) still
exist somewhere under the fixture, in the original place or the holding folder.

Paths covered:
  prune_unreferenced_raw (Open re-route, Low res toggle), _clear_routed_input,
  _flatten_routed_input (name collision), the step4test reset, the numbered
  resets of steps 1 to 8, and ReviewRepo.remove_item / remove_project.

No pytest. Run with the unified env python and no card visible:
    CUDA_VISIBLE_DEVICES="" env/bin/python \
        scripts/pipeline_orchestrator/tests/test_images_never_deleted.py

Nothing reaches a real process or port: subprocess and sockets are guarded as
in test_annotated_project_guard.py. Fixtures live in a temp directory and their
removal is verified at the end.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_ORCH)
sys.path.insert(0, _ORCH)
sys.path.insert(0, _SCRIPTS)
os.environ.setdefault("VICARIUS_LOCK_BYPASS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Retracted review items are held in the expert library folder; never the live one.
_LIB = tempfile.mkdtemp(prefix="img_keep_lib_")
os.environ["TCRMP_EXPERT_LIBRARY_DIR"] = _LIB


class _FakeCompleted:
    returncode = 0
    stdout = ""
    stderr = ""


_REAL_RUN = subprocess.run
subprocess.run = lambda *a, **k: _FakeCompleted()
_REAL_POPEN = subprocess.Popen


def _blocked_popen(*_a, **_k):
    raise RuntimeError("subprocess.Popen blocked in tests")


subprocess.Popen = _blocked_popen
_REAL_SOCKET = socket.socket
_GUARDED_PORTS = {5050} | set(range(5080, 5090))


class _NoLivePortSocket(_REAL_SOCKET):
    def connect(self, address, *a, **k):
        if (isinstance(address, tuple) and len(address) >= 2
                and address[1] in _GUARDED_PORTS):
            raise AssertionError(f"test tried to reach guarded port {address}")
        return super().connect(address, *a, **k)


socket.socket = _NoLivePortSocket

from PIL import Image  # noqa: E402

import app as appmod  # noqa: E402
import project_manager as pm  # noqa: E402


class _FakeVic:
    @staticmethod
    def note(*_a, **_k):
        return None

    @staticmethod
    def process_start(*_a, **_k):
        return None

    @staticmethod
    def process_end(*_a, **_k):
        return None


appmod.vic = _FakeVic()
appmod._flush_annotator = lambda port: (True, None)

_RESULTS = []
_FIXTURES = [_LIB]
_TMP_ROOT = os.environ.get("TMPDIR") or tempfile.gettempdir()
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
_COUNTER = [0]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _jpeg(path):
    """Write a small real JPEG whose bytes are unique to this call."""
    _COUNTER[0] += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("RGB", (16, 12), (_COUNTER[0] % 256, (_COUNTER[0] * 7) % 256, 90))
    img.putpixel((1, 1), (_COUNTER[0] % 251, 3, 4))
    img.save(path, "JPEG", quality=90)
    return path


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _hashes(root):
    """SHA-256 of every real (non-symlink) image file under root, with its path."""
    out = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.lower().endswith(IMAGE_EXTS) and not os.path.islink(p):
                out[p] = _sha(p)
    return out


def _assert_all_survive(before, root, label):
    after = set(_hashes(root).values())
    lost = sorted(p for p, h in before.items() if h not in after)
    check(not lost, f"{label}: image bytes lost: {lost}")


def _project(images_in=("routed_input/raw",), stale=False, segs=None):
    tmp = tempfile.mkdtemp(prefix="img_keep_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    steps = {}
    for s in pm.STEP_KEYS:
        d = os.path.join(tmp, pm.STEP_DIRS[s])
        os.makedirs(d, exist_ok=True)
        steps[s] = {"status": "pending", "name": pm.STEP_NAMES[s], "dir": d,
                    "config": {}, "outputs": {}}
    sel = os.path.join(steps["3"]["dir"], "selected_frames.csv")
    with open(sel, "w") as fh:
        fh.write("basename,year\nTCRMP20180225_clip_BIX_T111,2018\n")
    steps["3"]["outputs"]["selected_frames"] = sel
    export = os.path.join(tmp, pm.STEP_DIRS["step4test"])
    routed = os.path.join(export, "routed_input")
    os.makedirs(os.path.join(routed, "ids"))
    prompts = os.path.join(routed, "ids", "sam_click_prompts.json")
    with open(prompts, "w") as fh:
        fh.write("{}")
    for sub in images_in:
        for i in range(3):
            _jpeg(os.path.join(export, sub, f"img{i}.jpg"))
    seg_dir = os.path.join(export, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)
    with open(os.path.join(seg_dir, "segmentations.json"), "w") as fh:
        json.dump(segs or {}, fh)
    now = time.time()
    os.utime(prompts, (now - 1000, now - 1000))
    os.utime(sel, (now + (1000 if stale else -2000),) * 2)
    state = {"id": os.path.basename(tmp), "name": "keep", "project_dir": tmp,
             "created_at": "2026-10-15T10:00:00-04:00",
             "updated_at": "2026-10-15T10:00:00-04:00", "current_step": 3,
             "steps": steps}
    pm.save_project(state)
    return state, tmp, export, routed


class _Recorder:
    def __init__(self, result):
        self.result = result

    def __call__(self, *a, **k):
        return self.result


def _client(state):
    appmod.current_project = state
    appmod.step4test_cfg = {}
    appmod.route_thread = None
    appmod.runner.run_cli_stage = _Recorder({"pid": 1})
    appmod.runner.run_flask_stage = _Recorder({"error": "stubbed launch"})
    appmod.runner.kill = _Recorder(None)
    appmod.runner.health_check = lambda port: False
    return appmod.create_app().test_client()


def _close():
    appmod.current_project = None
    appmod._step_launches.clear()


# ---------------------------------------------------------------------------

def test_open_reroute_keeps_unreferenced_raw_images():
    state, tmp, export, routed = _project(stale=True)
    before = _hashes(tmp)
    try:
        _client(state).post("/api/step/step4test/run", json={})
        _assert_all_survive(before, tmp, "re-route prune")
        check(os.listdir(os.path.join(routed, "raw")) == [], "raw is clear for fresh images")
    finally:
        _close()


def test_clear_routed_input_keeps_year_folder_images():
    state, tmp, export, routed = _project(images_in=("routed_input/raw",))
    for y in ("2018", "2022"):
        _jpeg(os.path.join(routed, y, "raw", f"{y}.jpg"))
        _jpeg(os.path.join(routed, y, "test_pts", f"{y}_pts.jpg"))
        os.makedirs(os.path.join(routed, y, "ids"), exist_ok=True)
    before = _hashes(tmp)
    appmod._clear_routed_input(routed)
    _assert_all_survive(before, tmp, "_clear_routed_input")
    check(not os.path.exists(os.path.join(routed, "2018")), "the year folder is gone from routed_input")


def test_flatten_keeps_a_colliding_year_image():
    state, tmp, export, routed = _project(images_in=("routed_input/raw",))
    y = os.path.join(routed, "2018")
    os.makedirs(os.path.join(y, "ids"))
    with open(os.path.join(y, "ids", "sam_click_prompts.json"), "w") as fh:
        fh.write('{"img0.jpg": []}')
    _jpeg(os.path.join(y, "raw", "img0.jpg"))     # same name as a flat image, other bytes
    _jpeg(os.path.join(y, "raw", "new.jpg"))
    _jpeg(os.path.join(y, "test_pts", "new_pts.jpg"))
    before = _hashes(tmp)
    stats = appmod._flatten_routed_input(routed)
    check(stats["collisions"] == 1, f"fixture really collides: {stats}")
    _assert_all_survive(before, tmp, "_flatten_routed_input")


def test_step4test_reset_keeps_every_image():
    segs = {"img0.jpg": {"reviewed": True, "exported": True}}
    state, tmp, export, routed = _project(
        images_in=("routed_input/raw", "all_images", "segmentations/overlays_reviewed"), segs=segs)
    before = _hashes(tmp)
    try:
        client = _client(state)
        r = client.post("/api/step/step4test/reset", json={"confirm_exported": 1})
        check(r.status_code == 200, f"reset ran: {r.status_code} {r.get_json()}")
        check(not os.path.exists(export), "the export folder left its place")
        _assert_all_survive(before, tmp, "step4test reset")
    finally:
        _close()


def test_numbered_resets_keep_every_image():
    for n in range(1, 9):
        state, tmp, export, routed = _project(images_in=())
        for s in pm.STEP_KEYS:
            if int(s) >= n:
                _jpeg(os.path.join(state["steps"][s]["dir"], "gallery", f"s{s}.jpg"))
                _jpeg(os.path.join(state["steps"][s]["dir"], f"top{s}.png".replace(".png", ".jpg")))
        before = _hashes(tmp)
        try:
            client = _client(state)
            appmod._step6_driver_alive_outside_runner = lambda *a, **k: None
            r = client.post(f"/api/step/{n}/reset")
            check(r.status_code == 200, f"step {n} reset ran: {r.status_code} {r.get_json()}")
            _assert_all_survive(before, tmp, f"reset of step {n}")
        finally:
            _close()


def test_review_repo_removal_keeps_item_images():
    from _reefreview.review_repo import ReviewRepo
    root = tempfile.mkdtemp(prefix="img_keep_rr_", dir=_TMP_ROOT)
    _FIXTURES.append(root)
    src = os.path.join(root, "src")
    crop, full = _jpeg(os.path.join(src, "c.jpg")), _jpeg(os.path.join(src, "f.jpg"))
    repo = ReviewRepo(os.path.join(root, "REVIEW"), remote_url="", log_fn=lambda m: None)
    repo.add_item("U1", {"project_id": "P1"}, crop_src=crop, full_src=full)
    crop2, full2 = _jpeg(os.path.join(src, "c2.jpg")), _jpeg(os.path.join(src, "f2.jpg"))
    repo.add_item("U2", {"project_id": "P2"}, crop_src=crop2, full_src=full2)
    shutil.rmtree(src)       # the item copies are now the only bytes
    before = _hashes(root)
    check(len(before) == 4, f"fixture holds the four item images: {before}")
    repo.remove_item("U1")
    after = set(_hashes(root).values()) | set(_hashes(_LIB).values())
    check(set(before.values()) <= after, "remove_item lost an image")
    check(not os.path.exists(os.path.join(repo.items_dir, "U1")), "item left the live items folder")
    repo.remove_project("P2")
    after = set(_hashes(root).values()) | set(_hashes(_LIB).values())
    check(set(before.values()) <= after, "remove_project lost an image")


def _all_files(root):
    return {os.path.join(dp, n): _sha(os.path.join(dp, n))
            for dp, _d, fs in os.walk(root) for n in fs if not os.path.islink(os.path.join(dp, n))}


def test_reset_of_steps_4_and_5_keeps_outline_records():
    """Note 3: a reset never deletes annotation work, even with no image in it."""
    for n in (4, 5):
        state, tmp, export, routed = _project(images_in=())
        d5 = state["steps"]["5"]["dir"]
        recs = []
        for rel, text in (("segmentations/2024/segmentations.json", '{"a.jpg": {"reviewed": true}}'),
                          ("segmentations/segmentations.json", "{}"),
                          ("label_provenance.csv", "basename\n"),
                          ("export_manifest.csv", "x\n"),
                          ("all_labels/a.txt", "0 0.1 0.1 0.2 0.2\n")):
            f = os.path.join(d5, rel)
            os.makedirs(os.path.dirname(f), exist_ok=True)
            open(f, "w").write(text)
            recs.append(f)
        before = {os.path.basename(f) + _sha(f) for f in recs}
        try:
            r = _client(state).post(f"/api/step/{n}/reset")
            check(r.status_code == 200, f"step {n} reset: {r.status_code}")
            after = {os.path.basename(p) + h for p, h in _all_files(tmp).items()}
            check(before <= after, f"step {n} reset lost annotation records: {before - after}")
            check(not any(os.path.exists(f) for f in recs), "and they left the live step folder")
        finally:
            _close()


def test_flatten_puts_test_points_in_one_rolling_folder_and_counts_them():
    state, tmp, export, routed = _project(images_in=("routed_input/raw",))
    for y in ("2018", "2022"):
        os.makedirs(os.path.join(routed, y, "ids"))
        with open(os.path.join(routed, y, "ids", "sam_click_prompts.json"), "w") as fh:
            fh.write("{}")
        _jpeg(os.path.join(routed, y, "raw", f"{y}.jpg"))
        for i in range(3):
            _jpeg(os.path.join(routed, y, "test_pts", f"{y}_{i}_pts.jpg"))
    before = _hashes(tmp)
    stats = appmod._flatten_routed_input(routed)
    _assert_all_survive(before, tmp, "flatten with test points")
    check(stats.get("test_pts_moved") == 6, f"the per-pass count is reported: {stats}")
    holds = [d for d in os.listdir(os.path.join(export, "_superseded"))]
    check(len(holds) == 1, f"one holding folder for the pass: {holds}")
    pts = os.path.join(export, "_superseded", holds[0], "routed_test_pts")
    n = sum(len(fs) for _dp, _d, fs in os.walk(pts))
    check(n == 6, f"all six overlays sit under routed_test_pts/: {n}")
    check(os.path.isfile(os.path.join(routed, "raw", "2018.jpg")), "the routed images are flat in raw/")


def test_moved_aside_records_are_not_live_for_the_guards_or_the_prune():
    segs = {"live.jpg": {"reviewed": True}}
    state, tmp, export, routed = _project(segs=segs)
    # a moved-aside export beside the live one, naming other frames
    aside = os.path.join(tmp, "_superseded", "20261015-101010", "step4test_combinedAnnotate", "segmentations")
    os.makedirs(aside)
    json.dump({f"gone{i}.jpg": {"reviewed": True} for i in range(5)}, open(os.path.join(aside, "segmentations.json"), "w"))
    s5 = os.path.join(state["steps"]["5"]["dir"], "segmentations")
    os.makedirs(os.path.join(s5, "_superseded", "x"))
    json.dump({"old1.jpg": {"exported": True}}, open(os.path.join(s5, "_superseded", "segmentations.json"), "w"))
    json.dump({"s5.jpg": {"exported": True}}, open(os.path.join(s5, "segmentations.json"), "w"))
    got = appmod.count_annotated_frames(export)
    check(got == {"annotated": 2, "exported": 1}, f"only live records count: {got}")
    paths = appmod.annotation_record_paths(export)
    check(not any("_superseded" in p.split(os.sep) for p in paths), f"record paths: {paths}")


def test_prune_is_not_protected_by_a_moved_aside_record():
    state, tmp, export, routed = _project(segs={})
    s5 = os.path.join(state["steps"]["5"]["dir"], "segmentations", "_superseded")
    os.makedirs(s5)
    json.dump({"img0.jpg": {}}, open(os.path.join(s5, "segmentations.json"), "w"))
    before = _hashes(tmp)
    moved = appmod.prune_unreferenced_raw(routed, export)
    check(moved == 3, f"nothing in the moved-aside record protects the live raw images: {moved}")
    _assert_all_survive(before, tmp, "prune")


def test_holding_folder_is_dated_and_never_pruned_by_code():
    import _superseded as sup
    check(sup.SUPERSEDED_DIR_NAME == "_superseded", "holding folder name")
    tmp = tempfile.mkdtemp(prefix="img_keep_h_", dir=_TMP_ROOT)
    _FIXTURES.append(tmp)
    f = _jpeg(os.path.join(tmp, "d", "a.jpg"))
    h = _sha(f)
    dest = sup.supersede(f, base_dir=tmp)
    check(os.path.isfile(dest) and _sha(dest) == h, "moved with identical bytes")
    rel = os.path.relpath(dest, tmp).split(os.sep)
    check(rel[0] == "_superseded" and len(rel[1]) == 15 and rel[1][8] == "-" and rel[2:] == ["d", "a.jpg"],
          f"<base>/_superseded/<stamp>/<relative path>: {rel}")
    f2 = _jpeg(os.path.join(tmp, "d", "a.jpg"))
    dest2 = sup.supersede(f2, base_dir=tmp, stamp=rel[1])
    check(dest2 != dest and os.path.isfile(dest) and os.path.isfile(dest2),
          "a second move in the same second never overwrites the first")
    import inspect
    body = inspect.getsource(sup.supersede)
    check(not any(w in body for w in ("rmtree", "os.remove", "unlink", "os.replace")),
          "supersede itself never deletes or replaces")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            run(fn)
    for t in _FIXTURES:
        shutil.rmtree(t, ignore_errors=True)
    still = [t for t in _FIXTURES if os.path.exists(t)]
    print("fixtures removed" if not still else f"fixtures NOT removed: {still}")
    ok = sum(1 for _, p, _ in _RESULTS if p)
    for n, p, d in _RESULTS:
        if not p:
            print(f"\n--- {n}\n{d}")
    print(f"\n{ok}/{len(_RESULTS)} passed")
    sys.exit(0 if ok == len(_RESULTS) and not still else 1)
