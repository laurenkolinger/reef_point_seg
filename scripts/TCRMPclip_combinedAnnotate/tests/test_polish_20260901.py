"""Annotator polish + speed sweep (2026-09-01), combinedAnnotate and editMasks.

Covers the Task 7 fixes, none of which change the interaction model:

  Template (read as text, both apps):
    - doStart no longer reads scanResult.years (neither boot route sends it,
      and the swallowed TypeError killed the manual startup card)
    - scanReviewAccept splices the candidate only on success
    - POST /api/flush is wired: a trailing debounce after mask-status PUTs and
      a synchronous flush on Ctrl+S
    - relics gone: setAutoAdvance, acceptAllMasks, resetMask,
      selectNextPending, aKeyDown
    - reloadSegmentations and mergeSameId have visible, tooltipped buttons
    - the duplicate chk-target-only id is gone; chk-show-reference exists
    - copy: ADD badge, K-sheet lock-in row, "Find all labels", display name
    - quickadd buttons past index 9 show no untypeable key hint
    - the unlabel popup path keeps the mask pending (the server refuses an
      unlabeled accept)
    - the templates still never read .rle / polygon_norm

  app.py (both apps, via the Flask test client):
    - wire_mask() drops rle + polygon_norm and keeps everything else
    - GET /api/image ships wire masks while the seg on disk keeps both fields
    - GET /images/<fn> carries Cache-Control: private, max-age=3600
    - get_image_data kicks a background _ensure_embeddings() warm-up, and a
      second GET for the same frame does not stack a second thread

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_polish_20260901.py
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
import re
import shutil
import sys
import tempfile
import threading
import time

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
CA = os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate")
EM = os.path.join(_SCRIPTS, "TCRMPclip_editMasks")

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def read(p):
    with open(p, "r") as f:
        return f.read()


CA_TPL = read(os.path.join(CA, "src", "templates", "index.html"))
EM_TPL = read(os.path.join(EM, "src", "templates", "index.html"))
CA_APP = read(os.path.join(CA, "src", "app.py"))
EM_APP = read(os.path.join(EM, "src", "app.py"))

TPLS = [("combinedAnnotate", CA_TPL), ("editMasks", EM_TPL)]
APPS = [("combinedAnnotate", CA_APP), ("editMasks", EM_APP)]


# ── 1. Dead manual startup card: no `years` in the doStart info strings ─────
print("\n[1] doStart drops the year phrasing")
for name, tpl in TPLS:
    check("scanResult.years" not in tpl,
          f"{name}: no scanResult.years reference anywhere")
    check("year(s) found" not in tpl and "year(s)." not in tpl,
          f"{name}: no 'year(s)' copy left in the startup info strings")


# ── 2. scanReviewAccept keeps a refused candidate ──────────────────────────
print("\n[2] scanReviewAccept splices only on success")
for name, tpl in TPLS:
    body = tpl.split("async function scanReviewAccept()", 1)[-1].split("\n}\n", 1)[0]
    check("needs_species" in body,
          f"{name}: scanReviewAccept handles the needs_species refusal")
    check("showCatPopup" in body or "pendingAddMaskIdx" in body,
          f"{name}: scanReviewAccept opens the species popup on needs_species")
    # The unconditional splice in `finally` is the bug: the splice must sit
    # behind a success flag instead.
    fin = body.split("} finally {", 1)[-1].split("\n  }", 1)[0] if "finally" in body else ""
    check("scanCandidates.splice" not in fin,
          f"{name}: no unconditional scanCandidates.splice in the finally block")
    check("scanCandidates.splice" in body,
          f"{name}: scanReviewAccept still splices a kept candidate")
    check(re.search(r"if \(kept\) \{\s*\n\s*scanCandidates\.splice", body) is not None,
          f"{name}: the splice runs only when the server committed the candidate")


# ── 3. Persistence: /api/flush is actually called ──────────────────────────
print("\n[3] flush wiring")
for name, tpl in TPLS:
    check("'/api/flush'" in tpl or '"/api/flush"' in tpl,
          f"{name}: the client calls POST /api/flush")
    check("scheduleFlush(" in tpl, f"{name}: scheduleFlush debounce helper exists")
    check("2000" in tpl.split("function scheduleFlush", 1)[-1][:400],
          f"{name}: the flush debounce is a 2s trailing timer")
    persist = tpl.split("async function persistMaskStatus", 1)[-1].split("\n}\n", 1)[0]
    check("scheduleFlush(" in persist,
          f"{name}: persistMaskStatus arms the trailing flush")
    save = tpl.split("async function saveCurrentMasks", 1)[-1].split("\n}\n", 1)[0]
    check("scheduleFlush(" in save,
          f"{name}: saveCurrentMasks arms the trailing flush")
    check("saveFrameNow(" in tpl, f"{name}: Ctrl+S calls the synchronous save helper")
    sfn = tpl.split("async function saveFrameNow", 1)[-1].split("\n}\n", 1)[0]
    check("flushNow(" in sfn, f"{name}: saveFrameNow flushes to disk")
    check("Saved" in sfn, f"{name}: saveFrameNow confirms with a saved status line")
    ctrl_s = re.search(r"case 's': case 'S':(.{0,240})", tpl, re.S)
    check(bool(ctrl_s) and "saveFrameNow(" in ctrl_s.group(1),
          f"{name}: the Ctrl+S key branch routes through saveFrameNow")


# ── 4. Relics ──────────────────────────────────────────────────────────────
print("\n[4] relics removed")
RELICS = ["setAutoAdvance", "acceptAllMasks", "resetMask", "selectNextPending",
          "aKeyDown"]
for name, tpl in TPLS:
    for r in RELICS:
        check(r not in tpl, f"{name}: relic {r} is gone (definition and comments)")
    check("auto-advance-btn" not in tpl,
          f"{name}: the dead auto-advance-btn lookup is gone")


# ── 5. Dead functions wired up, with tooltips ──────────────────────────────
print("\n[5] reloadSegmentations + mergeSameId wired")
for name, tpl in TPLS:
    check('onclick="mergeSameId()"' in tpl, f"{name}: Merge same ID button exists")
    check("Merge same ID" in tpl, f"{name}: Merge same ID button carries its label")
    check('onclick="reloadSegmentations()"' in tpl, f"{name}: Reload IDs button exists")
    check("Reload IDs" in tpl, f"{name}: Reload IDs button carries its label")
    for btn_id in ("btn-merge-same-id", "btn-reload-ids"):
        m = re.search(r'<button[^>]*id="%s"[^>]*>' % btn_id, tpl, re.S)
        check(bool(m) and "title=" in m.group(0), f"{name}: {btn_id} has a tooltip")


# ── 6. Duplicate id + missing reference checkbox ───────────────────────────
print("\n[6] ids: chk-target-only unique, chk-show-reference present")
for name, tpl in TPLS:
    n = len(re.findall(r'id="chk-target-only"', tpl))
    check(n == 1, f"{name}: exactly one element carries id=chk-target-only (got {n})")
    check('id="chk-target-only-setup"' in tpl,
          f"{name}: the startup-card checkbox has its own id")
    setup_ref = re.search(r'const targetOnlyEl = document\.getElementById\(\'([^\']+)\'\)', tpl)
    check(bool(setup_ref) and setup_ref.group(1) == "chk-target-only-setup",
          f"{name}: doStart reads the startup-card checkbox by its new id")
    m = re.search(r'<input[^>]*id="chk-show-reference"[^>]*>', tpl, re.S)
    check(bool(m), f"{name}: a visible chk-show-reference checkbox exists")
    check(bool(m) and "toggleShowReference" in m.group(0),
          f"{name}: chk-show-reference is wired to toggleShowReference")
    check("Ref points (G)" in tpl, f"{name}: the checkbox is labeled Ref points (G)")
    lab = re.search(r'<label[^>]*>\s*<input[^>]*id="chk-show-reference"', tpl, re.S)
    check(bool(lab) and "title=" in lab.group(0),
          f"{name}: the Ref points label carries a tooltip")


# ── 7. Copy ────────────────────────────────────────────────────────────────
print("\n[7] copy fixes")
for name, tpl in TPLS:
    check("hold 1-N" not in tpl, f"{name}: ADD badge no longer claims 'hold 1-N'")
    check("tap 1-N (sticky)" in tpl, f"{name}: ADD badge says 'tap 1-N (sticky)'")
    check("unlabeled is fine" not in tpl,
          f"{name}: K-sheet no longer claims an unlabeled lock-in has no popup")
    check("the label popup opens" in tpl,
          f"{name}: K-sheet says the label popup opens for an unlabeled mask")
    fa = re.search(r'<button[^>]*id="btn-scan-all"[^>]*>', tpl, re.S)
    check(bool(fa) and "pending" in fa.group(0),
          f"{name}: 'Find all labels' tooltip says accepted OR pending")

check("{{ session_label }} | Annotate Images" in CA_TPL and "{% else %}Annotate Images{% endif %}" in CA_TPL,
      "combinedAnnotate: browser tab title names the project or session, falling back to 'Annotate Images' "
      "(2026-09-16: a student with two windows open must be able to tell them apart)")
check(len(re.findall(r"<h1>Annotate Images</h1>", CA_TPL)) == 3,
      "combinedAnnotate: setup, standalone and review h1 all say 'Annotate Images'")
check("TCRMP Segment Images" not in CA_TPL and "Segment Images" not in CA_TPL,
      "combinedAnnotate: no stale 'Segment Images' display name left")
check("<title>Edit Masks</title>" in EM_TPL,
      "editMasks: browser tab title is 'Edit Masks'")
check(len(re.findall(r"<h1>Edit Masks</h1>", EM_TPL)) == 2,
      "editMasks: setup and review h1 both say 'Edit Masks'")
check("TCRMP Segment Images" not in EM_TPL,
      "editMasks: no stale 'TCRMP Segment Images' display name left")
check(CA_APP.splitlines()[1].strip() == "Annotate Images (combinedAnnotate)",
      "combinedAnnotate: app.py docstring names the tool correctly")
check(EM_APP.splitlines()[1].strip() == "Edit Masks (editMasks)",
      "editMasks: app.py docstring names the tool correctly")


# ── 8. Quickadd key hints stop at 9 ────────────────────────────────────────
print("\n[8] quickadd key hints")
for name, tpl in TPLS:
    qa = tpl.split('<div class="quickadd-bar"', 1)[-1].split("</div>", 1)[0]
    check("loop.index <= 9" in qa,
          f"{name}: the quickadd key hint is guarded to indexes 1..9")


# ── 9. Unlabel keeps the mask pending ──────────────────────────────────────
print("\n[9] unlabel popup path")
for name, tpl in TPLS:
    body = tpl.split("function assignNewMaskCategory(category)", 1)[-1].split("\n}\n", 1)[0]
    check("'pending'" in body,
          f"{name}: assignNewMaskCategory keeps an unlabeled mask pending")
    check("persistMaskStatus(mask.id, 'accepted')" not in body.split("if (!mask.species)")[0]
          or "mask.species ? 'accepted' : 'pending'" in body
          or "newStatus" in body,
          f"{name}: the persisted status follows whether a species was assigned")


# ── 10. Templates still never read the wire-stripped fields ────────────────
print("\n[10] template never reads rle / polygon_norm")
for name, tpl in TPLS:
    check(".rle" not in tpl and "'rle'" not in tpl and '"rle"' not in tpl,
          f"{name}: template never reads mask.rle")
    check("polygon_norm" not in tpl, f"{name}: template never reads polygon_norm")


# ── 11. app.py: wire_mask + routes ─────────────────────────────────────────
print("\n[11] wire_mask serializer")


class _StubEngine:
    """Enough of SAM3Engine for _ensure_image_embedded and the warm-up thread."""

    def __init__(self):
        self._prompt_lock = threading.Lock()
        self._current_image_path = None
        self._current_image = None
        self._image_size = None
        self.embed_calls = 0
        self.started = threading.Event()

    def set_image(self, path):
        self._current_image_path = path
        self._current_image = object()
        self._image_size = (64, 48)
        return self._image_size

    def clear_image(self):
        self._current_image_path = None
        self._current_image = None

    def _ensure_embeddings(self):
        self.embed_calls += 1
        self.started.set()
        # Hold the lock briefly so a stacked second thread would be observable.
        time.sleep(0.15)
        return object()


def _seg(img_path):
    return {
        "image_path": "FRAME.jpg",
        "image_path_abs": img_path,
        "image_width": 64,
        "image_height": 48,
        "masks": [{
            "id": 0, "status": "pending", "species": "PA", "label": "1",
            "polygon_px": [[1, 1, 5, 1, 5, 5]],
            "polygon_norm": [[0.1, 0.1, 0.5, 0.1, 0.5, 0.5]],
            "rle": {"counts": "abc", "size": [48, 64]},
            "bbox": [1, 1, 5, 5], "area": 12, "score": 0.9,
            "refinement_clicks": [], "source_type": "auto",
        }],
        "reference_points": [],
        "processed_at": "", "reviewed": False, "exported": False,
    }


def exercise(app_dir, label):
    """Import one app module in a clean sys.path and drive its routes."""
    src = os.path.join(app_dir, "src")
    for mod in ("app", "config", "mask_ops", "sam_engine", "export_yolo",
                "render_overlay", "provenance", "custom_imports"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, src)
    try:
        import app as A
    finally:
        sys.path.remove(src)

    check(hasattr(A, "wire_mask"), f"{label}: app.wire_mask exists")
    if not hasattr(A, "wire_mask"):
        return

    raw = {"id": 3, "species": "PA", "rle": {"counts": "x"},
           "polygon_norm": [[0.1, 0.2]], "polygon_px": [[1, 2]], "area": 9}
    w = A.wire_mask(raw)
    check("rle" not in w and "polygon_norm" not in w,
          f"{label}: wire_mask drops rle and polygon_norm")
    check(w["id"] == 3 and w["species"] == "PA" and w["polygon_px"] == [[1, 2]]
          and w["area"] == 9,
          f"{label}: wire_mask keeps every other field")
    check("rle" in raw and "polygon_norm" in raw,
          f"{label}: wire_mask does not mutate the stored mask")

    d = tempfile.mkdtemp(prefix="polish_%s_" % label)
    try:
        img_path = os.path.join(d, "FRAME.jpg")
        Image.new("RGB", (64, 48), (10, 20, 30)).save(img_path)
        export_dir = os.path.join(d, "out")
        os.makedirs(export_dir, exist_ok=True)

        seg = _seg(img_path)
        A.app.config["TESTING"] = True
        A.session["export_dir"] = export_dir
        A.session["segmentations"] = {"FRAME.jpg": seg}
        eng = _StubEngine()
        A.session["sam_engine"] = eng
        A.session["current_review_image"] = None

        with A.app.test_client() as c:
            r = c.get("/api/image/FRAME.jpg")
            check(r.status_code == 200, f"{label}: GET /api/image 200 (got {r.status_code})")
            body = json.loads(r.data)
            m = body["masks"][0]
            check("rle" not in m, f"{label}: /api/image mask carries no rle")
            check("polygon_norm" not in m,
                  f"{label}: /api/image mask carries no polygon_norm")
            check("polygon_px" in m, f"{label}: /api/image mask keeps polygon_px")
            check("rle" in seg["masks"][0] and "polygon_norm" in seg["masks"][0],
                  f"{label}: the stored seg still carries both fields")

            # The warm-up thread runs after the response is built.
            check(eng.started.wait(timeout=5.0),
                  f"{label}: GET /api/image kicks the embedding warm-up")
            r2 = c.get("/api/image/FRAME.jpg")
            check(r2.status_code == 200, f"{label}: second GET /api/image 200")
            deadline = time.time() + 5.0
            while A._prewarming and time.time() < deadline:
                time.sleep(0.02)
            check(eng.embed_calls == 1,
                  f"{label}: a repeat GET for the same frame stacks no second "
                  f"warm-up thread (calls={eng.embed_calls})")

            ri = c.get("/images/FRAME.jpg")
            check(ri.status_code == 200, f"{label}: GET /images 200 (got {ri.status_code})")
            cc = ri.headers.get("Cache-Control", "")
            check("private" in cc and "max-age=3600" in cc,
                  f"{label}: /images sets Cache-Control private, max-age=3600 (got {cc!r})")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        for mod in ("app", "config", "mask_ops", "sam_engine", "export_yolo",
                    "render_overlay", "provenance", "custom_imports"):
            sys.modules.pop(mod, None)


exercise(CA, "combinedAnnotate")
exercise(EM, "editMasks")


# ── 12. Every mutating route answers with wire masks ───────────────────────
print("\n[12] mutating routes serialize through wire_mask")
for name, src in APPS:
    for route in ("commit_mask", "refine_mask", "rebox_mask", "draw_edit",
                  "undo_refine", "reset_mask", "merge_masks", "merge_many",
                  "merge_same_id", "reload_segmentations", "add_mask",
                  "add_mask_box"):
        body = src.split("def %s(" % route, 1)[-1].split("\n@app.route", 1)[0]
        check("wire_mask(" in body or "wire_masks(" in body,
              f"{name}: {route} serializes its mask payload through wire_mask")
    # The exemplar candidates are round-tripped verbatim to /commit_mask, so
    # they MUST keep rle + polygon_norm.
    ex = src.split("def exemplar_scan(", 1)[-1].split("\n@app.route", 1)[0]
    check("wire_mask" not in ex,
          f"{name}: exemplar candidates keep rle/polygon_norm for the round trip")


# ── 13. Frame switch cannot paint new pixels under old masks ───────────────
# The JPEG is prefetched and served with a cache header, so img.onload can fire
# BEFORE the frame's JSON resolves. render() bails when currentData is missing,
# so unless selectImage repaints after the assignment (or assigns before it
# sets img.src) the canvas sits on the new frame's pixels under the previous
# frame's masks. Structural check on the real statement order, not a word grep.
print("\n[13] frame-switch repaint ordering")
for name, tpl in TPLS:
    start = tpl.index("async function selectImage(")
    body = tpl[start:tpl.index("\n}\n", start)]
    i_src = body.index("img.src = ")
    i_data = body.index("currentData = data;")
    repaint = re.search(r"if \(img && img\.complete\) \{[^}]*render\(\);?\s*\}", body)
    ordered = i_data < i_src
    repaints_after = bool(repaint) and repaint.start() > i_data
    check(ordered or repaints_after,
          f"{name}: selectImage either assigns currentData before img.src "
          f"(assign@{i_data} src@{i_src}) or repaints once the image is decoded")
    check(repaints_after or ordered,
          f"{name}: a decoded-image repaint follows the currentData assignment")
    # A rejected status PUT must not abort the switch after currentIdx moved.
    sp = body[body.index("const savePrev"):body.index("currentIdx = idx;")]
    check(".catch(" in sp,
          f"{name}: the previous frame's save is caught so a failed PUT cannot "
          f"abort the switch after currentIdx and img moved")


# ── 14. Popup honours the "Del to drop it" promise for a scan candidate ────
print("\n[14] popup Delete drops a refused scan candidate")
for name, tpl in TPLS:
    gate = tpl.split("if (pendingAddMaskIdx >= 0 || pendingScanCandidateIdx >= 0) {", 1)[-1]
    gate = gate.split("\n  }\n", 1)[0]
    check("scanReviewReject()" in gate,
          f"{name}: the popup gate has a branch that drops the candidate")
    check("'Delete'" in gate and "'Backspace'" in gate,
          f"{name}: Delete and Backspace reach that branch")
    check("pendingScanCandidateIdx >= 0\n        && (e.key === 'Delete'" in gate,
          f"{name}: the drop branch only fires while labeling a scan candidate")
    # And the status line that promises it is still the one shown.
    check("or Del to drop it" in tpl,
          f"{name}: the scan-review status line still offers Del")


# ── 15. Store writes are serialized and atomic ─────────────────────────────
# The new 2s trailing flush means a timer-driven rewrite can now interleave
# with do_export's flush and the background segmentation loop, all of which
# rewrite segmentations.json. json.dump streams, so an unguarded writer leaves
# a truncated file behind.
print("\n[15] concurrent flush safety")


def exercise_store(app_dir, label):
    src = os.path.join(app_dir, "src")
    for mod in ("app", "config", "mask_ops", "sam_engine", "export_yolo",
                "render_overlay", "provenance", "custom_imports"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, src)
    try:
        import app as A
    finally:
        sys.path.remove(src)

    check(hasattr(A, "_store_write_lock"),
          f"{label}: app holds a module-level store write lock")
    check(hasattr(A, "_atomic_write_json"),
          f"{label}: app has an atomic JSON writer")
    if not (hasattr(A, "_store_write_lock") and hasattr(A, "_atomic_write_json")):
        return
    # Reentrant, or flush_all() would deadlock on its own nested savers.
    check(isinstance(A._store_write_lock, type(threading.RLock())),
          f"{label}: the store lock is reentrant (flush_all nests inside it)")

    d = tempfile.mkdtemp(prefix="store_%s_" % label)
    try:
        export_dir = os.path.join(d, "out")
        path = A._seg_path(export_dir)

        # A big payload so an unguarded json.dump would be observably partial.
        def payload(tag):
            return {f"FRAME_{i:04d}.jpg": {
                "tag": tag, "image_width": 1920, "image_height": 1080,
                "masks": [{"id": j, "species": "PA", "status": "accepted",
                           "polygon_px": [list(range(40))],
                           "rle": {"counts": "z" * 300, "size": [1080, 1920]}}
                          for j in range(12)],
            } for i in range(120)}

        A.save_segmentations(export_dir, payload("seed"))
        check(os.path.exists(path), f"{label}: the seed write produced the store")

        errors = []
        reads = [0]
        stop = threading.Event()

        def writer(tag):
            try:
                for _ in range(12):
                    A.save_segmentations(export_dir, payload(tag))
            except Exception as e:                       # pragma: no cover
                errors.append(("write", tag, repr(e)))

        def reader():
            # Every read must see a COMPLETE file: os.replace swaps the inode,
            # so a reader never observes a half-streamed json.dump.
            while not stop.is_set():
                try:
                    with open(path) as f:
                        doc = json.load(f)
                    if len(doc) != 120:
                        errors.append(("short", len(doc)))
                    reads[0] += 1
                except Exception as e:
                    errors.append(("read", repr(e)))

        rt = threading.Thread(target=reader, daemon=True)
        rt.start()
        ws = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(4)]
        for t in ws:
            t.start()
        for t in ws:
            t.join(timeout=60)
        stop.set()
        rt.join(timeout=10)

        check(not errors,
              f"{label}: 4 concurrent writers + a reader saw no torn or short "
              f"reads (errors={errors[:3]})")
        check(reads[0] > 0, f"{label}: the reader actually read during the writes")
        with open(path) as f:
            final = json.load(f)
        check(len(final) == 120, f"{label}: the final store is complete")

        # A failed serialization must leave the previous file untouched and
        # must not strand a temp file next to it.
        before = open(path).read()
        raised = False
        try:
            A.save_segmentations(export_dir, {"bad": {object()}})
        except Exception:
            raised = True
        check(raised, f"{label}: an unserializable payload still raises")
        check(open(path).read() == before,
              f"{label}: a failed write leaves the previous store byte-identical")
        leftovers = [f for f in os.listdir(os.path.dirname(path))
                     if f.endswith(".tmp")]
        check(not leftovers,
              f"{label}: a failed write strands no temp file (found {leftovers})")

        # flush_all() nests the savers inside the same lock; it must not hang.
        A.session["export_dir"] = export_dir
        A.session["segmentations"] = payload("flush")
        A.session["class_map"] = {}
        done = threading.Event()

        def call_flush():
            A.flush_all()
            done.set()

        ft = threading.Thread(target=call_flush, daemon=True)
        ft.start()
        ft.join(timeout=30)
        check(done.is_set(), f"{label}: flush_all completes (no self-deadlock)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        for mod in ("app", "config", "mask_ops", "sam_engine", "export_yolo",
                    "render_overlay", "provenance", "custom_imports"):
            sys.modules.pop(mod, None)


exercise_store(CA, "combinedAnnotate")
exercise_store(EM, "editMasks")


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
