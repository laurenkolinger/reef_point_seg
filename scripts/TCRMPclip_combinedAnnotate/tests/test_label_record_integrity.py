"""Label records stay true to the project they describe: export, merge, ledger
write timing and prior screening, in BOTH annotators.

combinedAnnotate (CA) and Edit Masks (EM) share provenance.py byte for byte and
carry the same session and export machinery in their own app.py, so each check
below runs against both app modules.

Items (plan 2026-10-15-vega-project-and-label-integrity.md, task L):
  1. Export writes no outcome and no registry row for a frame whose image file
     is missing; export_manifest.csv still records image_missing.
  2. A merge takes the one non-blank species among its inputs; it refuses with
     a 400 when two different species meet, or when every input is blank and
     any input is accepted.
  5. The label_provenance.csv ledger is written when segmentations.json is
     written (flush_all), never on the edit itself.
  7. Edit Masks passes screened_elsewhere (reads prior_screening.json).

No pytest, GPU-free (CUDA_VISIBLE_DEVICES is forced empty), the model is never
loaded. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_label_record_integrity.py
"""
import csv
import json
import os
import shutil
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TCRMP_TARGET_SPECIES", "PA")
os.environ.setdefault("TCRMP_MANUAL_ANNOTATE", "1")
os.environ["TCRMP_REVIEWER"] = "LO"
# The export hook upserts the cross-project mask registry: isolate it.
_REGISTRY_DIR = tempfile.mkdtemp(prefix="lri_registry_")
os.environ["TCRMP_MASK_REGISTRY_DIR"] = _REGISTRY_DIR
os.environ["TCRMP_CLIP_INDEX_DIR"] = tempfile.mkdtemp(prefix="lri_clipidx_")

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SCRIPTS)
APPS = {
    "combinedAnnotate": os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate"),
    "editMasks": os.path.join(_SCRIPTS, "TCRMPclip_editMasks"),
}
_MODS = ("app", "config", "mask_ops", "sam_engine", "export_yolo",
         "render_overlay", "provenance", "custom_imports")
LEDGER = "label_provenance.csv"
H, W = 300, 300

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _import_app(app_dir):
    """Import one app module with no other app's modules leaking in."""
    src = os.path.join(app_dir, "src")
    for mod in _MODS:
        sys.modules.pop(mod, None)
    sys.path.insert(0, src)
    try:
        import app as A
    finally:
        sys.path.remove(src)
    return A


def _release():
    for mod in _MODS:
        sys.modules.pop(mod, None)


def _blob(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _mask(mid, species, status, y0, y1, x0, x1, **extra):
    from mask_ops import rle_encode
    m = {"id": mid, "status": status, "species": species, "label": chr(64 + mid),
         "rle": rle_encode(_blob(y0, y1, x0, x1)),
         "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]],
         "refinement_clicks": [], "source_x": 1, "source_y": 1,
         "source_type": "manual_click", "review": False}
    m.update(extra)
    return m


def _seg(path, masks):
    return {"image_path": os.path.basename(path), "image_path_abs": path,
            "image_width": W, "image_height": H, "masks": masks,
            "reference_points": [], "reviewed": False, "exported": False}


def _ledger(export_dir):
    path = os.path.join(export_dir, LEDGER)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _open_session(A, export_dir, segs):
    A.session["export_dir"] = export_dir
    A.session["input_dir"] = export_dir
    A.session["segmentations"] = segs
    A.session["class_map"] = {}
    A.session["review_batch_size"] = 10
    A.session["shuffle"] = False
    A.session["configured"] = True
    A.session["phase"] = "review"
    A.session["contacts"] = []
    A.session["prior_screening"] = {}
    A.session["sam_engine"] = None
    A._build_review_list()


# ── item 1: export with a missing image ───────────────────────────────────
def exercise_export_missing_image(A, label):
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="lri_export_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        good_img = os.path.join(d, "GOOD.jpg")
        Image.new("RGB", (W, H), (10, 20, 30)).save(good_img)
        gone_img = os.path.join(d, "GONE.jpg")  # never written
        segs = {
            "GOOD.jpg": _seg(good_img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)]),
            "GONE.jpg": _seg(gone_img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)]),
        }
        _open_session(A, export_dir, segs)
        r = A.app.test_client().post("/api/export_batch")
        j = r.get_json()
        check(r.status_code == 200 and j and j.get("ok"), f"{label}: export ok ({r.status_code})")
        rows = _ledger(export_dir)
        names = {r_["basename"] for r_ in rows}
        check("GONE" not in names, f"{label}: no ledger row for the image_missing frame (got {sorted(names)})")
        check("GOOD" in names, f"{label}: ledger row kept for the frame whose image exists")
        check(not segs["GONE.jpg"].get("label_outcomes"),
              f"{label}: no outcome stamped on the image_missing frame")
        with open(os.path.join(export_dir, "export_manifest.csv"), newline="") as f:
            man = {r_["basename"]: r_ for r_ in csv.DictReader(f)}
        check(man.get("GONE", {}).get("outcome") == "image_missing",
              f"{label}: export_manifest still records image_missing")
        from _reefreview.mask_registry import MaskRegistry
        reg_rows = MaskRegistry(root=_REGISTRY_DIR).rows()
        check(not any("GONE" in str(r_.get("basename") or r_.get("uid") or r_) for r_ in reg_rows),
              f"{label}: no registry row for the image_missing frame")
        check(any("GOOD" in str(r_) for r_ in reg_rows),
              f"{label}: registry row written for the frame whose image exists")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── item 2: merge species ─────────────────────────────────────────────────
def exercise_merge_species(A, label):
    A.app.config["TESTING"] = True
    client = A.app.test_client()

    def fresh(specs, statuses=None):
        statuses = statuses or ["pending"] * len(specs)
        masks = [_mask(i + 1, sp, st, 10 + 80 * i, 60 + 80 * i, 10, 60)
                 for i, (sp, st) in enumerate(zip(specs, statuses))]
        seg = _seg("x", masks)
        A.session["segmentations"] = {"f.jpg": seg}
        return seg

    seg = fresh(["", "PA"])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [1, 2]})
    check(r.status_code == 200, f"{label}: blank + PA merges ({r.status_code})")
    check(seg["masks"][0]["species"] == "PA",
          f"{label}: survivor takes the non-blank species, got {seg['masks'][0]['species']!r}")

    seg = fresh(["PA", ""])
    r = client.post("/api/image/f.jpg/merge", json={"mask_id_a": 1, "mask_id_b": 2})
    check(r.status_code == 200 and seg["masks"][0]["species"] == "PA",
          f"{label}: pairwise merge keeps PA when the other input is blank")

    seg = fresh(["PA", "OFRA"])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [1, 2]})
    msg = (r.get_json() or {}).get("error", "")
    check(r.status_code == 400 and "PA" in msg and "OFRA" in msg,
          f"{label}: two different species refused with both named ({r.status_code} {msg!r})")
    check(len(seg["masks"]) == 2 and seg["masks"][0]["species"] == "PA",
          f"{label}: a refused merge leaves the frame untouched")

    seg = fresh(["", ""], ["accepted", "pending"])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [1, 2]})
    msg = (r.get_json() or {}).get("error", "")
    check(r.status_code == 400 and msg, f"{label}: all blank with one accepted refused ({r.status_code} {msg!r})")
    check(len(seg["masks"]) == 2, f"{label}: all-blank refusal leaves the frame untouched")

    seg = fresh(["", ""])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [1, 2]})
    check(r.status_code == 200 and seg["masks"][0]["species"] == "",
          f"{label}: all blank and all pending still merges and stays blank ({r.status_code})")

    seg = fresh(["PA", "PA"], ["accepted", "pending"])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [1, 2]})
    check(r.status_code == 200 and seg["masks"][0]["species"] == "PA",
          f"{label}: same species merges as before")

    seg = fresh(["", "", "OFRA"], ["accepted", "pending", "pending"])
    r = client.post("/api/image/f.jpg/merge_many", json={"mask_ids": [3, 2, 1]})
    check(r.status_code == 200 and seg["masks"][0]["species"] == "OFRA",
          f"{label}: three inputs, one non-blank species wins regardless of id order")


# ── item 5: ledger written with the save ──────────────────────────────────
def exercise_ledger_on_flush(A, label):
    d = tempfile.mkdtemp(prefix="lri_flush_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        img = os.path.join(d, "F1.jpg")
        Image.new("RGB", (W, H)).save(img)
        seg = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
        _open_session(A, export_dir, {"F1.jpg": seg})
        A._stamp_provenance(seg, "F1.jpg")
        check(_ledger(export_dir) == [],
              f"{label}: stamping an edit does not write the ledger")
        check("PA" in (seg.get("label_outcomes") or {}),
              f"{label}: the in-memory outcome is stamped on the frame")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        rows = _ledger(export_dir)
        check([(r["basename"], r["label"], r["outcome"]) for r in rows] == [("F1", "PA", "found_manual")],
              f"{label}: flush_all writes the ledger row ({rows})")
        with open(A._seg_path(export_dir)) as f:
            on_disk = json.load(f)
        check(rows and on_disk["F1.jpg"]["label_outcomes"]["PA"]["outcome"] == rows[0]["outcome"],
              f"{label}: ledger and segmentations.json describe the same state")

        # A label that leaves the frame disappears from the ledger on the next flush.
        seg["masks"].append(_mask(2, "OFRA", "accepted", 100, 160, 100, 160))
        A._stamp_provenance(seg, "F1.jpg")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        check(sorted(r["label"] for r in _ledger(export_dir)) == ["OFRA", "PA"],
              "%s: both labels are in the ledger" % label)
        seg["masks"] = [m for m in seg["masks"] if m["species"] == "PA"]
        A._stamp_provenance(seg, "F1.jpg")
        check(sorted(r["label"] for r in _ledger(export_dir)) == ["OFRA", "PA"],
              f"{label}: ledger unchanged until the next save")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        check(sorted(r["label"] for r in _ledger(export_dir)) == ["PA"],
              f"{label}: a label that left the frame is gone after the flush "
              f"({[r['label'] for r in _ledger(export_dir)]})")
        check(not [n for n in os.listdir(export_dir) if n.endswith(".tmp")],
              f"{label}: no temp file left in the export folder")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── item 7: prior screening reaches the stamp ─────────────────────────────
def exercise_prior_screening(A, label):
    d = tempfile.mkdtemp(prefix="lri_prior_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        with open(os.path.join(export_dir, "prior_screening.json"), "w") as f:
            json.dump({"version": 1, "frames": {
                "F1": {"screened": {"PA": {"outcome": "found_manual"}}, "masks": []}}}, f)
        check(hasattr(A, "load_prior_screening"), f"{label}: app defines load_prior_screening")
        if not hasattr(A, "load_prior_screening"):
            return
        loaded = A.load_prior_screening(export_dir)
        check("F1" in loaded and "PA" in loaded["F1"]["screened"],
              f"{label}: loader reads the project's prior_screening.json")
        seg = _seg(os.path.join(d, "F1.jpg"), [])
        _open_session(A, export_dir, {"F1.jpg": seg})
        A.session["prior_screening"] = loaded
        A._stamp_provenance(seg, "F1.jpg")
        check("PA" not in (seg.get("label_outcomes") or {}),
              f"{label}: a label screened elsewhere gets no not_found ({seg.get('label_outcomes')})")
        A.session["prior_screening"] = {}
        A._stamp_provenance(seg, "F1.jpg")
        check((seg.get("label_outcomes") or {}).get("PA", {}).get("outcome") == "not_found",
              f"{label}: without prior screening the same frame is not_found")
        with open(os.path.join(os.path.dirname(A.__file__), "app.py")) as f:
            src = f.read()
        check(src.count("session['prior_screening'] = load_prior_screening(") >= 2,
              f"{label}: the open and resume routes load prior screening")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── D2: the ledger heals only the drift this code created ──────────────────
def exercise_ledger_heals(A, label):
    d = tempfile.mkdtemp(prefix="lri_heal_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        img = os.path.join(d, "F1.jpg")
        Image.new("RGB", (W, H)).save(img)
        segs = {n: _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)]) for n in ("F1.jpg", "F2.jpg")}
        _open_session(A, export_dir, segs)
        for n, seg in segs.items():
            A._stamp_provenance(seg, n)
        check(all(seg.get("ledger_pending") is True for seg in segs.values()),
              f"{label}: a stamped frame carries the ledger_pending marker")
        # The ledger write fails once after the JSON is saved.
        real = A.write_provenance_frames
        A.write_provenance_frames = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        A.write_provenance_frames = real
        check(_ledger(export_dir) == [], f"{label}: the failed write leaves the ledger behind the JSON")
        check(all(seg.get("ledger_pending") for seg in segs.values()),
              f"{label}: a failed write keeps the markers")
        with open(A._seg_path(export_dir)) as f:
            on_disk = json.load(f)
        check(on_disk["F1.jpg"].get("ledger_pending") is True,
              f"{label}: the marker is saved with segmentations.json")
        # A quit and reopen: the queue is gone, the marker is on disk.
        A._ledger_pending.clear()
        A.session["segmentations"] = A.load_segmentations(export_dir)
        A._heal_ledger(export_dir)
        got = sorted(r["basename"] for r in _ledger(export_dir))
        check(got == ["F1", "F2"], f"{label}: reopening writes the marked frames ({got})")
        check(not any(s_.get("ledger_pending") for s_ in A.session["segmentations"].values()),
              f"{label}: the markers clear once the write succeeded")
        check(A._ledger_pending.get(export_dir) in (None, set()), f"{label}: the queue is empty after the heal")
        # An unmarked frame is never rewritten, whatever the disagreement.
        segs2 = A.session["segmentations"]
        segs2["F2.jpg"]["label_outcomes"]["PA"]["outcome"] = "found_ai"
        before = open(os.path.join(export_dir, "label_provenance.csv"), "rb").read()
        A._heal_ledger(export_dir)
        check(open(os.path.join(export_dir, "label_provenance.csv"), "rb").read() == before,
              f"{label}: a frame without the marker is not rewritten")
        with open(os.path.join(os.path.dirname(A.__file__), "app.py")) as f:
            src = f.read()
        check(src.count("_heal_ledger(") >= 3, f"{label}: open and resume heal the ledger")
        # one swap for the whole queue
        calls = []
        real_replace = os.replace
        for n, seg in segs2.items():
            seg["masks"][0]["species"] = "OFRA"
            A._stamp_provenance(seg, n)
        os.replace = lambda a, b: (calls.append(b), real_replace(a, b))[1]
        try:
            A._mark_dirty(filename="F1.jpg")
            A.flush_all()
        finally:
            os.replace = real_replace
        ledger_swaps = [c for c in calls if c.endswith("label_provenance.csv")]
        check(len(ledger_swaps) == 1, f"{label}: a two-frame queue is one ledger swap per save ({len(ledger_swaps)})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── an old store is never rewritten by opening it ──────────────────────────
def exercise_old_store_untouched(A, label):
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="lri_old_")
    try:
        export_dir = os.path.join(d, "run_26-07-06_ZZ_old", "step4test_combinedAnnotate")
        os.makedirs(os.path.join(export_dir, "segmentations"))
        img = os.path.join(d, "F.jpg")
        Image.new("RGB", (W, H)).save(img)
        segs = {}
        for i in range(20):  # historical drift: outcomes stored, no ledger rows
            seg = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
            seg["label_outcomes"] = {"PA": {"outcome": "found_manual", "reviewed": True,
                                            "source": "step4test", "reviewer": "ZZ", "at": "2026-07-20T10:00:00-04:00"}}
            segs[f"OLD{i:02d}.jpg"] = seg
        with open(A._seg_path(export_dir), "w") as f:
            json.dump(segs, f)
        ledger = os.path.join(export_dir, "label_provenance.csv")
        with open(ledger, "w", newline="") as f:  # blank-label rows and a stale row
            f.write("basename,label,outcome,source,reviewer,at,project_id\n")
            for i in range(8):
                f.write(f"BLANK{i},,found_manual,step4test,ZZ,2026-07-06T10:00:00-04:00,old\n")
            f.write("GONE,XYZ,found_ai,step4test,ZZ,2026-07-06T10:00:00-04:00,old\n")
        raw = open(ledger, "rb").read()
        c = A.app.test_client()
        r = c.post("/api/resume", json={"export_dir": export_dir})
        check(r.status_code == 200, f"{label}: resume opens the old project ({r.status_code})")
        check(open(ledger, "rb").read() == raw, f"{label}: resume leaves the ledger byte-identical")
        A._heal_ledger(export_dir)
        check(open(ledger, "rb").read() == raw, f"{label}: a direct heal leaves it byte-identical")
        A.flush_all()
        check(open(ledger, "rb").read() == raw, f"{label}: a no-edit flush leaves it byte-identical")
        A._quit_flushed = False
        A._mark_dirty(filename="OLD00.jpg")
        A.flush_before_exit()
        check(open(ledger, "rb").read() == raw, f"{label}: a quit flush leaves it byte-identical")
        check(not any("ledger_pending" in s_ for s_ in A.session["segmentations"].values()),
              f"{label}: no frame gains a marker by being opened")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── D4 and note 3: export of a frame whose image is missing ────────────────
def exercise_missing_frame_stays_in_queue(A, label):
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="lri_miss_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        good = os.path.join(d, "GOOD.jpg")
        Image.new("RGB", (W, H)).save(good)
        gone = os.path.join(d, "GONE.jpg")
        segs = {"GOOD.jpg": _seg(good, [_mask(1, "PA", "accepted", 10, 60, 10, 60)]),
                "GONE.jpg": _seg(gone, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])}
        _open_session(A, export_dir, segs)
        seen = {}
        real_overlay = A.render_segmentation_overlays

        def spy(seg, fn, ed, stage=None):
            seen["dirty_at_overlay"] = fn in A._dirty_frames
            return real_overlay(seg, fn, ed, stage=stage)
        A.render_segmentation_overlays = spy
        j = A.app.test_client().post("/api/export_batch").get_json()
        A.render_segmentation_overlays = real_overlay
        check(j and j.get("ok"), f"{label}: export ok")
        check(not segs["GONE.jpg"].get("exported"), f"{label}: a missing-image frame is not marked exported")
        check(bool(segs["GONE.jpg"].get("image_missing_at")), f"{label}: it carries a dated image_missing_at mark")
        check("GONE.jpg" not in A.session["review_files"], f"{label}: it stays out of the queue while the image is absent")
        check(j.get("image_missing") == 1 and j.get("image_missing_total") == 1,
              f"{label}: the export answer counts it ({j.get('image_missing')}, {j.get('image_missing_total')})")
        check(A.app.test_client().get("/api/status").get_json()["stats"].get("image_missing") == 1,
              f"{label}: status counts it")
        check(seen.get("dirty_at_overlay") is True, f"{label}: the exported frame is marked dirty in the same locked section as its stamp")
        Image.new("RGB", (W, H)).save(gone)
        A._build_review_list()
        check("GONE.jpg" in A.session["review_files"], f"{label}: it returns to the queue when the image exists again")
        check(not segs["GONE.jpg"].get("image_missing_at"), f"{label}: the mark clears when the image is back")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── scrap removes a frame's rows; unscrap restores them ────────────────────
def exercise_scrap_ledger(A, label):
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="lri_scrap_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        img = os.path.join(d, "F1.jpg")
        Image.new("RGB", (W, H)).save(img)
        seg = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
        _open_session(A, export_dir, {"F1.jpg": seg})
        A._stamp_provenance(seg, "F1.jpg")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        check(len(_ledger(export_dir)) == 1, f"{label}: one row before the scrap")
        c = A.app.test_client()
        c.post("/api/image/F1.jpg/scrap")
        check(_ledger(export_dir) == [], f"{label}: scrapping removes the frame's rows")
        c.post("/api/image/F1.jpg/unscrap")
        check([r["label"] for r in _ledger(export_dir)] == ["PA"], f"{label}: unscrapping restores them on the save")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── E1: a cleared marker is persisted; the heal never undoes an importer row ──
def exercise_marker_persisted_and_expert_row_kept(A, label):
    d = tempfile.mkdtemp(prefix="lri_e1_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        img = os.path.join(d, "F1.jpg")
        Image.new("RGB", (W, H)).save(img)
        seg = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
        _open_session(A, export_dir, {"F1.jpg": seg})
        A._stamp_provenance(seg, "F1.jpg")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        check(not seg.get("ledger_pending"), f"{label}: the marker clears in memory after the write")
        check(A._is_dirty(), f"{label}: clearing marks the frame dirty so the cleared marker is saved")
        A._quit_flushed = False
        A.flush_before_exit()
        with open(A._seg_path(export_dir)) as f:
            on_disk = json.load(f)
        check("ledger_pending" not in on_disk["F1.jpg"], f"{label}: the quit save leaves no marker on disk")

        # Marker left on disk by a failed write, then the importer writes its row.
        seg2 = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
        _open_session(A, export_dir, {"F2.jpg": seg2})
        A._stamp_provenance(seg2, "F2.jpg")
        real = A.write_provenance_frames
        A.write_provenance_frames = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        A._mark_dirty(filename="F2.jpg")
        A.flush_all()
        A.write_provenance_frames = real
        import provenance as P
        P.write_provenance_csv(export_dir, "F2", {"PA": {"outcome": "found_expert", "source": "expertids",
                               "reviewer": "XY", "at": "2026-10-15T12:00:00-04:00"}}, "p")
        A._ledger_pending.clear()
        A.session["segmentations"] = A.load_segmentations(export_dir)
        A._heal_ledger(export_dir)
        rows = [r for r in _ledger(export_dir) if r["basename"] == "F2"]
        check([(r["label"], r["outcome"], r["source"]) for r in rows] == [("PA", "found_expert", "expertids")],
              f"{label}: the heal leaves the importer's row alone ({[(r['label'], r['outcome'], r['source']) for r in rows]})")
        check(not A.session["segmentations"]["F2.jpg"].get("ledger_pending"), f"{label}: the marker clears")

        # The heal recomputes from the frame's current outlines.
        seg3 = _seg(img, [_mask(1, "OFRA", "accepted", 10, 60, 10, 60)])
        seg3["label_outcomes"] = {"PA": {"outcome": "found_manual", "reviewed": True, "source": "step4test",
                                          "reviewer": "", "at": "t"}}
        seg3["ledger_pending"] = True
        _open_session(A, export_dir, {"F3.jpg": seg3})
        A.session["segmentations"]["F3.jpg"]["ledger_pending"] = True
        A._heal_ledger(export_dir)
        got = sorted((r["label"]) for r in _ledger(export_dir) if r["basename"] == "F3")
        check(got == ["OFRA", "PA"] or got == ["OFRA"],
              f"{label}: the heal writes labels computed from the current outlines ({got})")
        check(any(r["label"] == "OFRA" for r in _ledger(export_dir) if r["basename"] == "F3"),
              f"{label}: OFRA, on the frame now, is in the ledger")
        check(not any(r["label"] == "PA" and r["outcome"] == "found_manual" and r["basename"] == "F3"
                      for r in _ledger(export_dir)),
              f"{label}: the stale stored PA found_manual is not restored")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── a stuck lock holder never freezes saving ───────────────────────────────
def exercise_stuck_lock(A, label):
    import subprocess, time
    import _reefreview.ledger_io as LIO
    d = tempfile.mkdtemp(prefix="lri_lock_")
    try:
        export_dir = os.path.join(d, "run_26-10-15_ZZ", "step4test_combinedAnnotate")
        os.makedirs(export_dir)
        img = os.path.join(d, "F1.jpg")
        Image.new("RGB", (W, H)).save(img)
        seg = _seg(img, [_mask(1, "PA", "accepted", 10, 60, 10, 60)])
        _open_session(A, export_dir, {"F1.jpg": seg})
        A._stamp_provenance(seg, "F1.jpg")
        A._mark_dirty(filename="F1.jpg")
        logs = []
        real_log = A.log
        A.log = lambda m: logs.append(m)
        old = LIO.LEDGER_LOCK_TIMEOUT_S
        LIO.LEDGER_LOCK_TIMEOUT_S = 0.5
        hold = subprocess.Popen([sys.executable, "-c", (
            "import fcntl,sys,time;f=open(sys.argv[1],'a');fcntl.flock(f,fcntl.LOCK_EX);"
            "print('held',flush=True);time.sleep(2)"), os.path.join(export_dir, "label_provenance.csv.lock")],
            stdout=subprocess.PIPE, text=True)
        hold.stdout.readline()
        t0 = time.time()
        A.flush_all()
        took = time.time() - t0
        hold.wait()
        LIO.LEDGER_LOCK_TIMEOUT_S = old
        A.log = real_log
        check(took < 1.8, f"{label}: the save returns while another process holds the lock ({took:.2f}s)")
        check(os.path.exists(A._seg_path(export_dir)), f"{label}: segmentations.json was still saved")
        check(seg.get("ledger_pending") is True and "F1.jpg" in A._ledger_pending.get(export_dir, set()),
              f"{label}: the frame stays queued and marked")
        check(len([m for m in logs if "ledger write failed" in m]) == 1, f"{label}: one log line ({len(logs)} lines)")
        A._mark_dirty(filename="F1.jpg")
        A.flush_all()
        check([r["label"] for r in _ledger(export_dir)] == ["PA"], f"{label}: the next save writes it")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def exercise_no_duplicate_helpers(A, label):
    with open(os.path.join(os.path.dirname(A.__file__), "app.py")) as f:
        src = f.read()
    check(src.count("def _count_image_missing(") == 1, f"{label}: _count_image_missing is defined once")


def main():
    for label, app_dir in APPS.items():
        print(f"== {label}")
        for fn in (exercise_export_missing_image, exercise_merge_species,
                   exercise_ledger_on_flush, exercise_prior_screening,
                   exercise_ledger_heals, exercise_old_store_untouched,
                   exercise_missing_frame_stays_in_queue,
                   exercise_marker_persisted_and_expert_row_kept, exercise_stuck_lock,
                   exercise_no_duplicate_helpers,
                   exercise_scrap_ledger):
            A = _import_app(app_dir)
            try:
                fn(A, label)
            except Exception as e:  # a crash is a failed check, not a silent skip
                import traceback
                traceback.print_exc()
                check(False, f"{label}: {fn.__name__} raised {type(e).__name__}: {e}")
            finally:
                _release()
    shutil.rmtree(_REGISTRY_DIR, ignore_errors=True)
    print(f"\n{'FAILED' if _fail else 'ALL PASSED'} ({_fail} failure(s))")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
