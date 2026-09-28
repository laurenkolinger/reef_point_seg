"""save_segmentations() merges an outside writer's change into the store
frame by frame instead of refusing the whole write, never loses an edit that
arrives mid-write, and never tells the operator an edit saved when a frame it
touched was also edited outside this process.

VEGA annotation safety net, 2026-09-27/28 (LO). Three review passes on task
T1's persistence layer, in order:

  1. flush_all() wrote segmentations.json from memory unconditionally, with
     no check for a disk-side change first. The expert-ID importer
     (MOD/scripts/_expertids/importer.py, _seg_locked/_save around lines
     192-245) can rewrite that same file while an annotator or Edit Masks
     sits open on the same project.
  2. The first fix diverted a whole-file conflict to a sibling file instead
     of writing, which itself had two defects: a boolean dirty flag could
     silently drop an edit that arrived while a save was still serializing,
     and a stat-only mismatch (a touch, a same-content rewrite) was treated
     as a real conflict.
  3. A CRITICAL flaw in pass 2: Reload IDs (reload_segmentations) adopted
     the disk identity after merging review fields into ONE frame only, so
     the next save silently overwrote every OTHER frame's outside change;
     and every save during an unresolved conflict wrote a brand new,
     unbounded conflict file with no way back.

This suite locks the CURRENT design, a per-frame merge that replaces
divert-on-conflict entirely, in both app.py files:

  - A monotonic dirty counter (_segmentations_dirty_counter /
    _segmentations_saved_version / _is_dirty()) plus a per-frame
    _dirty_frames set, both snapshotted before a save starts writing and
    only advanced once the write completes, so an edit arriving mid-write is
    never silently cleared.
  - save_segmentations() and reload_segmentations() both run
    _merge_store_with_disk(): a frame dirty only locally keeps the local
    version (folding in imported review fields); a frame not dirty locally
    is taken from disk wholesale; a frame on only one side is kept as is; a
    frame dirty on BOTH sides keeps the local edit, folds in review fields,
    and reports the frame as a genuine conflict, with the disk version
    written to ONE conflict-episode sibling file
    (segmentations.conflict-<AST stamp>.json) reused across every save while
    the episode stays unresolved.
  - /api/flush and the equivalent save routes (scrap, unscrap) always answer
    ok: true; a merge with disk-only changes gets {'merged': True,
    'taken_from_disk': N}; a merge with a both-sides frame gets
    {'conflict': True, 'conflict_file': <name>, 'both_sides': N}.
  - configure()/resume()/standalone_start() never mark the session dirty
    (loading is not an edit); scrap/unscrap/export mark dirty themselves,
    before their own flush_all(), instead of relying on the
    @_dirty_on_success wrapper (which fires AFTER the route returns and
    would otherwise re-dirty a session that just saved).
  - configure()/resume() validate the new project BEFORE touching
    session['export_dir'], and save a dirty PREVIOUS session to its own
    folder before switching, so a failed Open can never write the old
    project's store into an unrelated new folder.

No pytest, GPU-free: CUDA_VISIBLE_DEVICES is forced empty below, and nothing
here imports sam_engine or instantiates SAM3Engine (importing app.py as a
module never does either; the model load sits behind `if __name__ ==
'__main__':`).

Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_flush_conflict_guard.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
CA = os.path.join(_SCRIPTS, "TCRMPclip_combinedAnnotate")
EM = os.path.join(_SCRIPTS, "TCRMPclip_editMasks")

_MODS = ("app", "config", "mask_ops", "sam_engine", "export_yolo",
         "render_overlay", "provenance", "custom_imports")

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _import_app(app_dir):
    src = os.path.join(app_dir, "src")
    for mod in _MODS:
        sys.modules.pop(mod, None)
    sys.path.insert(0, src)
    try:
        import app as A
    finally:
        sys.path.remove(src)
    return A


def _release(app_dir):
    for mod in _MODS:
        sys.modules.pop(mod, None)


def _set_dirty(A, filename=None):
    A._mark_dirty(filename=filename)


def _set_clean(A):
    """Force a clean session: the next save has nothing new to capture."""
    A._segmentations_saved_version = A._segmentations_dirty_counter
    A._dirty_frames.clear()


def _bump_mtime(path, seconds_ahead):
    """Force `path`'s mtime detectably forward, simulating a rewrite by
    another process (the importer) without depending on real wall-clock
    elapsed time between two writes in the same test."""
    st = os.stat(path)
    new_ns = st.st_mtime_ns + int(seconds_ahead * 1e9)
    os.utime(path, ns=(new_ns, new_ns))


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def mask(mid, species="PA", status="accepted", **extra):
    m = {"id": mid, "species": species, "status": status}
    m.update(extra)
    return m


# ── Basic save/merge status contract ─────────────────────────────────────

def test_first_save_of_a_session_that_never_loaded_proceeds(app_dir, label):
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="first_save_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            status, info = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "seed"}})
            check(status == "saved", f"{label}: a fresh export_dir's first save is 'saved', got {status!r}")
            check(info is None, f"{label}: no merge info on a plain save")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_a_normal_dirty_save_with_no_disk_change_is_plain_saved(app_dir, label):
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="normal_save_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"FRAME.jpg": {"tag": "original"}})
            A.load_segmentations(export_dir)
            status, info = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "edited"}})
            check(status == "saved", f"{label}: no external change -> 'saved', got {status!r}")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk == {"FRAME.jpg": {"tag": "edited"}}, f"{label}: the edit reached disk, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_mtime_tracking_survives_its_own_saves(app_dir, label):
    """A second save right after the first is not mistaken for a conflict
    with itself."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="mtime_survives_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"FRAME.jpg": {"tag": "seed"}})
            A.load_segmentations(export_dir)
            first, _ = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "edit_1"}})
            check(first == "saved", f"{label}: first save is 'saved', got {first!r}")
            second, _ = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "edit_2"}})
            check(second == "saved", f"{label}: second consecutive save is still 'saved', got {second!r}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_same_content_different_stat_is_not_a_conflict(app_dir, label):
    """A stat mismatch alone is not a conflict: a touch or a same-content
    rewrite adopts the new stat and the save proceeds as a plain save."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="same_content_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            content = {"FRAME.jpg": {"tag": "same"}}
            _write_json(seg_path, content)
            A.load_segmentations(export_dir)
            _write_json(seg_path, content)
            _bump_mtime(seg_path, 5)
            status, info = A.save_segmentations(export_dir, content)
            check(status == "saved", f"{label}: identical content on disk is not a merge, got {status!r}")
            check(info is None, f"{label}: no merge info for a same-content save")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── Per-frame merge ───────────────────────────────────────────────────────

def test_frames_edited_only_outside_are_taken_into_memory(app_dir, label):
    """A frame this session never touched, changed on disk, is taken
    wholesale into the merged store, with no local edit to defend."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="taken_from_disk_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}, "B.jpg": {"tag": "b0"}})
            A.load_segmentations(export_dir)

            # The importer changes B (this process never edits it) and also
            # bumps the whole file's mtime so this save's stat check fires.
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}, "B.jpg": {"tag": "b_from_importer"}})
            _bump_mtime(seg_path, 5)

            mem_store = {"A.jpg": {"tag": "a1_local_edit"}, "B.jpg": {"tag": "b0"}}
            _set_dirty(A, "A.jpg")
            status, info = A.save_segmentations(export_dir, mem_store)

            check(status == "merged", f"{label}: a real disk-side change runs the merge, got {status!r}")
            check(info["both_sides"] == 0, f"{label}: A's own edit does not collide with B's outside change")
            check(info["taken_from_disk"] == 1, f"{label}: B is taken from disk, got {info}")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk["B.jpg"]["tag"] == "b_from_importer",
                  f"{label}: B's outside edit is in the merged file, got {on_disk.get('B.jpg')}")
            check(on_disk["A.jpg"]["tag"] == "a1_local_edit",
                  f"{label}: A's local edit is untouched, got {on_disk.get('A.jpg')}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_both_sides_frame_keeps_local_and_folds_review_fields(app_dir, label):
    """A frame edited here AND on disk keeps the local (accept/reject/
    geometry) version, folds in the disk copy's review/expert fields, and
    the disk version is written to the conflict file."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="both_sides_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            original_mask = mask(1, status="pending")
            _write_json(seg_path, {"F.jpg": {"masks": [original_mask]}})
            A.load_segmentations(export_dir)

            # This process accepts the mask locally...
            local_mask = mask(1, status="accepted")
            mem_store = {"F.jpg": {"masks": [local_mask]}}
            _set_dirty(A, "F.jpg")

            # ...while the expert-ID importer, independently, writes an
            # accepted expert ID onto the SAME mask on disk.
            expert_mask = mask(1, status="pending",
                                expert_id={"mode": "EXPERT", "code": "PA"}, species="PA")
            _write_json(seg_path, {"F.jpg": {"masks": [expert_mask]}})
            _bump_mtime(seg_path, 5)

            status, info = A.save_segmentations(export_dir, mem_store)
            check(status == "merged", f"{label}: status is 'merged', got {status!r}")
            check(info["both_sides"] == 1, f"{label}: F.jpg is reported as a both-sides frame, got {info}")

            merged_mask = mem_store["F.jpg"]["masks"][0]
            check(merged_mask["status"] == "accepted",
                  f"{label}: the local accept survives, got {merged_mask.get('status')}")
            check(merged_mask.get("expert_id", {}).get("mode") == "EXPERT",
                  f"{label}: the disk copy's expert ID is folded in, got {merged_mask.get('expert_id')}")

            conflict_path = info["conflict_path"]
            check(conflict_path is not None and os.path.isfile(conflict_path),
                  f"{label}: the conflict file exists, got {conflict_path!r}")
            with open(conflict_path) as f:
                conflict_content = json.load(f)
            f_versions = conflict_content.get("F.jpg", [])
            check(bool(f_versions) and f_versions[-1]["content"]["masks"][0]["status"] == "pending",
                  f"{label}: the DISK version (still pending, not locally accepted) is in the "
                  f"conflict file, got {f_versions}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_one_episode_yields_at_most_one_conflict_file_across_five_saves(app_dir, label):
    """Five saves in a row, each finding the same both-sides frame, produce
    exactly one conflict-sibling file on disk (its content is appended to,
    never a fresh file per save), so an operator editing through an
    unresolved conflict never fills the project folder with one 24MB backup
    per trailing flush."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="one_episode_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            seg_dir = os.path.dirname(seg_path)
            _write_json(seg_path, {"F.jpg": {"tag": "v0"}})
            A.load_segmentations(export_dir)

            for i in range(5):
                _write_json(seg_path, {"F.jpg": {"tag": f"outside_{i}"}})
                _bump_mtime(seg_path, 5 + i)
                mem_store = {"F.jpg": {"tag": f"local_{i}"}}
                _set_dirty(A, "F.jpg")
                status, info = A.save_segmentations(export_dir, mem_store)
                check(status == "merged" and info["both_sides"] == 1,
                      f"{label}: save {i} finds the both-sides frame, got {status!r} {info}")

            siblings = [f for f in os.listdir(seg_dir)
                        if f.startswith("segmentations.conflict-") and f.endswith(".json")]
            check(len(siblings) == 1,
                  f"{label}: exactly one conflict file survives 5 saves in one episode, found {siblings}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_an_edit_during_the_write_is_not_lost(app_dir, label):
    """A save that snapshots the dirty counter/frames before it starts
    serializing must not clear a NEWER edit that arrived while the write
    itself was in flight (a click ~2s after the last, mid a large write)."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="race_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"FRAME.jpg": {"tag": "v1"}}
            A.session["class_map"] = {}
            _set_dirty(A, "FRAME.jpg")

            _orig_atomic = A._atomic_write_json
            racing_edit_applied = []

            def racing_atomic_write(path, payload, indent):
                A.session["segmentations"]["FRAME.jpg"]["tag"] = "v2_mid_write"
                A._mark_dirty(filename="FRAME.jpg")
                racing_edit_applied.append(1)
                return _orig_atomic(path, payload, indent)

            A._atomic_write_json = racing_atomic_write
            status, _ = A.save_segmentations(export_dir, A.session["segmentations"])
            A._atomic_write_json = _orig_atomic

            check(racing_edit_applied == [1], f"{label}: the racing edit hook actually ran")
            check(status == "saved", f"{label}: the save itself still succeeds, got {status!r}")
            check(A._is_dirty(),
                  f"{label}: still dirty after a save that raced with a new edit "
                  f"(a boolean flag would have wrongly cleared here)")

            second, _ = A.save_segmentations(export_dir, A.session["segmentations"])
            check(second == "saved", f"{label}: the follow-up save writes normally, got {second!r}")
            check(not A._is_dirty(), f"{label}: clean again once that save completes")
            with open(A._seg_path(export_dir)) as f:
                on_disk = json.load(f)
            check(on_disk["FRAME.jpg"]["tag"] == "v2_mid_write",
                  f"{label}: the racing edit's content reached disk, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── Reload IDs: full-store merge, not a one-frame adopt ──────────────────

def test_conflict_then_reload_then_save_keeps_the_other_frames_expert_id(app_dir, label):
    """The exact scenario the CRITICAL review found: a conflict on frame A,
    a Reload IDs while looking at A, then a later save must not erase an
    expert ID the importer wrote to a DIFFERENT frame B that Reload IDs
    never displayed."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="reload_then_save_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            mask_a = mask(1, status="pending")
            mask_b = mask(2, status="accepted")
            _write_json(seg_path, {"A.jpg": {"masks": [mask_a]}, "B.jpg": {"masks": [mask_b]}})
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = A.load_segmentations(export_dir)
            A.session["class_map"] = {}

            # The operator accepts A locally (dirty), never touches B.
            A.session["segmentations"]["A.jpg"]["masks"][0]["status"] = "accepted"
            _set_dirty(A, "A.jpg")

            # The importer, independently, writes an expert ID onto A (a
            # both-sides frame) AND onto B (an outside-only frame).
            expert_a = mask(1, status="pending", expert_id={"mode": "EXPERT", "code": "PA"}, species="PA")
            expert_b = mask(2, status="accepted", expert_id={"mode": "EXPERT", "code": "SS"}, species="SS")
            _write_json(seg_path, {"A.jpg": {"masks": [expert_a]}, "B.jpg": {"masks": [expert_b]}})
            _bump_mtime(seg_path, 5)

            # The operator hits "Reload IDs" while looking at A.
            with A.app.test_client() as c:
                r = c.post("/api/reload_segmentations", json={"filename": "A.jpg"})
                body = r.get_json()
            check(r.status_code == 200, f"{label}: reload_segmentations 200 (got {r.status_code})")
            check(body.get("conflict") is True, f"{label}: reload reports the A/A conflict, got {body}")

            # B's expert ID must already be in memory after the FULL reload,
            # even though the operator never looked at B.
            b_in_memory = A.session["segmentations"]["B.jpg"]["masks"][0]
            check(b_in_memory.get("expert_id", {}).get("code") == "SS",
                  f"{label}: B's outside expert ID reached memory via the full reload, "
                  f"got {b_in_memory.get('expert_id')}")

            # A later save (e.g. an unrelated edit to B, or just a trailing
            # flush) must not lose B's expert ID: nothing on disk changed
            # for B since the reload adopted its identity, so this is a
            # plain, uneventful save.
            _set_dirty(A, "B.jpg")
            status, info = A.save_segmentations(export_dir, A.session["segmentations"])
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk["B.jpg"]["masks"][0].get("expert_id", {}).get("code") == "SS",
                  f"{label}: B's expert ID survives the later save, got {on_disk.get('B.jpg')}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── Session lifecycle: dirty gate, configure/resume safety ───────────────

def test_an_idle_resumed_session_writes_nothing_on_flush_or_quit(app_dir, label):
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="idle_resume_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"FRAME.jpg": {"tag": "v0"}})

            with A.app.test_client() as c:
                r = c.post("/api/resume", json={"export_dir": export_dir})
            check(r.status_code == 200, f"{label}: resume 200 (got {r.status_code})")
            check(not A._is_dirty(), f"{label}: resume never marks the session dirty")

            before_bytes = open(seg_path, "rb").read()
            calls = []
            _orig_save = A.save_segmentations
            A.save_segmentations = lambda *a, **k: calls.append(1)
            with A.app.test_client() as c:
                r2 = c.post("/api/flush")
            A.save_segmentations = _orig_save
            check(calls == [], f"{label}: /api/flush on an idle resumed session never saves")
            check(r2.get_json() == {"ok": True}, f"{label}: /api/flush still answers ok")

            A._quit_flushed = False
            calls2 = []
            A.flush_all = lambda: (calls2.append(1), ("saved", None))[-1]
            A.flush_before_exit()
            check(calls2 == [], f"{label}: quit on an idle resumed session never flushes")
            check(before_bytes == open(seg_path, "rb").read(),
                  f"{label}: the file on disk is untouched")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_a_failed_open_does_not_write_the_old_store_into_the_new_folder(app_dir, label):
    """configure() validates the new project (the Step 4 ids_path check)
    BEFORE touching session['export_dir']: a failed Open with a dirty
    previous session must leave the OLD project's folder holding the save,
    never spill into the new, invalid folder."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="failed_open_%s_" % label)
        try:
            old_export = os.path.join(d, "old_project")
            new_export = os.path.join(d, "new_project")
            bad_input = os.path.join(d, "no_such_input")
            os.makedirs(old_export, exist_ok=True)

            A.session["export_dir"] = old_export
            A.session["segmentations"] = {"FRAME.jpg": {"tag": "old_unsaved"}}
            A.session["class_map"] = {}
            A.session["input_dir"] = ""
            _set_dirty(A, "FRAME.jpg")

            with A.app.test_client() as c:
                r = c.post("/api/configure", json={
                    "input_dir": bad_input, "export_dir": new_export,
                })
            check(r.status_code == 404, f"{label}: a missing Step 4 input 404s (got {r.status_code})")
            check(not os.path.isdir(new_export),
                  f"{label}: the new (invalid) folder was never even created")
            check(A.session["export_dir"] == old_export,
                  f"{label}: session['export_dir'] still points at the OLD project, "
                  f"got {A.session['export_dir']!r}")
            old_seg_path = A._seg_path(old_export)
            check(not os.path.exists(old_seg_path),
                  f"{label}: a REJECTED Open (no switch ever happens) does not even "
                  f"trigger the old-session save; nothing was written anywhere")
            check(A.session["segmentations"] == {"FRAME.jpg": {"tag": "old_unsaved"}},
                  f"{label}: the old project's in-memory edit is untouched, "
                  f"got {A.session['segmentations']}")

            # A SUCCESSFUL switch, by contrast, must save the dirty old
            # session to its own folder first (the second half of the rule).
            good_input = os.path.join(d, "good_input")
            os.makedirs(os.path.join(good_input, "ids"), exist_ok=True)
            os.makedirs(os.path.join(good_input, "raw"), exist_ok=True)
            with open(os.path.join(good_input, "ids", "sam_click_prompts.json"), "w") as f:
                json.dump({}, f)
            with A.app.test_client() as c:
                r2 = c.post("/api/configure", json={
                    "input_dir": good_input, "export_dir": new_export,
                })
            check(r2.status_code == 200, f"{label}: the valid Open succeeds (got {r2.status_code})")
            check(A.session["export_dir"] == os.path.abspath(new_export),
                  f"{label}: the session now points at the NEW project")
            check(os.path.exists(old_seg_path),
                  f"{label}: the old project's dirty session was saved to ITS OWN "
                  f"folder before the switch")
            with open(old_seg_path) as f:
                on_disk = json.load(f)
            check(on_disk == {"FRAME.jpg": {"tag": "old_unsaved"}},
                  f"{label}: the old project's file holds the pre-switch edit, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_configure_and_resume_do_not_mark_dirty(app_dir, label):
    """Loading a project is not an edit: configure()/resume() must never
    increment the dirty counter on their own."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="no_dirty_on_load_%s_" % label)
        try:
            input_dir = os.path.join(d, "input")
            os.makedirs(os.path.join(input_dir, "ids"), exist_ok=True)
            os.makedirs(os.path.join(input_dir, "raw"), exist_ok=True)
            with open(os.path.join(input_dir, "ids", "sam_click_prompts.json"), "w") as f:
                json.dump({}, f)
            export_dir = os.path.join(d, "out")

            before = A._segmentations_dirty_counter
            with A.app.test_client() as c:
                r = c.post("/api/configure", json={"input_dir": input_dir, "export_dir": export_dir})
            check(r.status_code == 200, f"{label}: configure 200 (got {r.status_code})")
            check(A._segmentations_dirty_counter == before,
                  f"{label}: configure() left the dirty counter untouched")
            check(not A._is_dirty(), f"{label}: a freshly configured session is not dirty")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_scrap_unscrap_export_mark_dirty_before_their_own_flush(app_dir, label):
    """scrap/unscrap must not be left dirty right after they just saved: the
    mark has to happen BEFORE their own flush_all(), not via the
    @_dirty_on_success wrapper firing after the route returns."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="scrap_dirty_timing_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"FRAME.jpg": {"masks": [], "scrapped": False, "reviewed": False}}
            A.session["class_map"] = {}
            _set_clean(A)
            with A.app.test_client() as c:
                r = c.post("/api/image/FRAME.jpg/scrap")
            check(r.status_code == 200, f"{label}: /scrap 200 (got {r.status_code})")
            check(not A._is_dirty(),
                  f"{label}: the session is clean right after scrap saved it "
                  f"(no post-hoc re-dirty from the decorator)")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── /api/flush and scrap/unscrap report merges through the test client ────

def test_force_flush_reports_a_merge_via_test_client(app_dir, label):
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="flush_merge_client_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}, "B.jpg": {"tag": "b0"}})
            A.load_segmentations(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}, "B.jpg": {"tag": "b_outside"}})
            _bump_mtime(seg_path, 5)

            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"A.jpg": {"tag": "a1"}, "B.jpg": {"tag": "b0"}}
            A.session["class_map"] = {}
            _set_dirty(A, "A.jpg")

            with A.app.test_client() as c:
                r = c.post("/api/flush")
                body = r.get_json()
            check(r.status_code == 200, f"{label}: /api/flush 200 (got {r.status_code})")
            check(body.get("ok") is True, f"{label}: ok is always true now, got {body}")
            check(body.get("merged") is True, f"{label}: merged is true, got {body}")
            check(body.get("taken_from_disk") == 1, f"{label}: taken_from_disk is 1, got {body}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_force_flush_reports_a_conflict_via_test_client(app_dir, label):
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="flush_conflict_client_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}})
            A.load_segmentations(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a_outside"}})
            _bump_mtime(seg_path, 5)

            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"A.jpg": {"tag": "a_local"}}
            A.session["class_map"] = {}
            _set_dirty(A, "A.jpg")

            with A.app.test_client() as c:
                r = c.post("/api/flush")
                body = r.get_json()
            check(body.get("ok") is True, f"{label}: ok is true even on a conflict, got {body}")
            check(body.get("conflict") is True, f"{label}: conflict is true, got {body}")
            check(body.get("both_sides") == 1, f"{label}: both_sides is 1, got {body}")
            conflict_file = body.get("conflict_file") or ""
            check(conflict_file.startswith("segmentations.conflict-"),
                  f"{label}: conflict_file names the sibling, got {conflict_file!r}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── Static wiring checks (structural, no live route needed) ──────────────

def test_dirty_gate_is_wired_into_flush_before_exit_and_force_flush(app_dir, label):
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    fbe = src.split("def flush_before_exit():", 1)[-1].split("\ndef ", 1)[0]
    check("_is_dirty()" in fbe, f"{label}: flush_before_exit() checks _is_dirty()")
    ff = src.split("def force_flush():", 1)[-1].split("\n\n\n", 1)[0]
    check("_is_dirty()" in ff, f"{label}: force_flush() checks _is_dirty()")


_MUTATING_ROUTES = [
    "update_masks", "accept_all_masks", "commit_mask",
    "refine_mask", "rebox_mask", "draw_edit", "undo_refine", "reset_mask",
    "merge_masks", "merge_many", "merge_same_id", "add_mask", "add_mask_box",
    "set_mask_size",
]

_SELF_FLUSHING_ROUTES = ["scrap_segmentation", "unscrap_segmentation", "do_export"]
_LOAD_ROUTES = ["configure", "resume"]


def test_every_known_mutating_route_is_decorated(app_dir, label, extra_routes=()):
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    for name in list(_MUTATING_ROUTES) + list(extra_routes):
        idx = src.find(f"def {name}(")
        check(idx != -1, f"{label}: {name}() exists")
        if idx == -1:
            continue
        preceding = src[:idx].rstrip().splitlines()
        check(bool(preceding) and preceding[-1].strip() == "@_dirty_on_success",
              f"{label}: {name}() is decorated with @_dirty_on_success")


def test_self_flushing_routes_are_not_decorated(app_dir, label):
    """scrap/unscrap/export mark dirty by hand before their own flush_all();
    @_dirty_on_success on top of that would re-dirty the session right after
    it just saved."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    for name in _SELF_FLUSHING_ROUTES:
        idx = src.find(f"def {name}(")
        check(idx != -1, f"{label}: {name}() exists")
        if idx == -1:
            continue
        preceding = src[:idx].rstrip().splitlines()
        check(bool(preceding) and preceding[-1].strip() != "@_dirty_on_success",
              f"{label}: {name}() is NOT decorated with @_dirty_on_success "
              f"(line above def: {preceding[-1] if preceding else '<none>'!r})")
        body_start = idx
        body = src[body_start:src.find("\n@app.route", body_start + 1)]
        check("_mark_dirty(" in body,
              f"{label}: {name}() marks dirty itself")


def test_load_routes_are_not_decorated(app_dir, label):
    """Loading a project is not an edit."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    for name in _LOAD_ROUTES:
        idx = src.find(f"def {name}(")
        check(idx != -1, f"{label}: {name}() exists")
        if idx == -1:
            continue
        preceding = src[:idx].rstrip().splitlines()
        check(bool(preceding) and preceding[-1].strip() != "@_dirty_on_success",
              f"{label}: {name}() is NOT decorated with @_dirty_on_success")


def test_exemplar_scan_is_not_decorated(app_dir, label):
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    idx = src.find("def exemplar_scan(")
    check(idx != -1, f"{label}: exemplar_scan() exists")
    if idx == -1:
        return
    preceding = src[:idx].rstrip().splitlines()
    check(bool(preceding) and preceding[-1].strip() != "@_dirty_on_success",
          f"{label}: exemplar_scan() carries no @_dirty_on_success")


# ── The background segmentation loop merges, not diverts, per image ──────

def test_background_style_per_image_save_merges_not_diverts(app_dir, label):
    """Simulates the exact pattern _background_process_all() uses (a bare
    save_segmentations() call after each processed image, in a loop): a
    disk-side change on an earlier frame merges in rather than the whole
    batch being refused."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="bg_merge_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            store = {}
            status, _ = A.save_segmentations(export_dir, store)
            check(status == "saved", f"{label}: the seed save is 'saved', got {status!r}")

            # Frame 1 processed.
            store["FRAME_001.jpg"] = {"tag": "processed_1"}
            _set_dirty(A, "FRAME_001.jpg")
            status, _ = A.save_segmentations(export_dir, store)
            check(status == "saved", f"{label}: frame 1's checkpoint save is 'saved', got {status!r}")

            # An outside writer touches frame 1 between checkpoints.
            with open(seg_path) as f:
                disk_now = json.load(f)
            disk_now["FRAME_001.jpg"]["tag"] = "outside_relabel"
            _write_json(seg_path, disk_now)
            _bump_mtime(seg_path, 5)

            # Frame 2 processed; the checkpoint save must merge, not refuse.
            store["FRAME_002.jpg"] = {"tag": "processed_2"}
            _set_dirty(A, "FRAME_002.jpg")
            status, info = A.save_segmentations(export_dir, store)
            check(status == "merged", f"{label}: the checkpoint save after frame 2 merges, got {status!r}")
            check(info["both_sides"] == 0,
                  f"{label}: frame 1's outside relabel is not a both-sides conflict "
                  f"(this loop never re-edits a frame after processing it), got {info}")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk["FRAME_001.jpg"]["tag"] == "outside_relabel",
                  f"{label}: frame 1's outside relabel survives, got {on_disk.get('FRAME_001.jpg')}")
            check(on_disk["FRAME_002.jpg"]["tag"] == "processed_2",
                  f"{label}: frame 2's own result is written, got {on_disk.get('FRAME_002.jpg')}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


# ── Second review pass: defects A (critical), B, C, and the minor fixes ──

def test_reload_then_second_outside_write_then_save_keeps_the_reloaded_edit(app_dir, label):
    """Defect A (CRITICAL), reproduced then fixed: Reload IDs used to clear
    a kept-local frame from _dirty_frames and adopt its identity WITHOUT an
    actual write. Sequence: edit frame a, an outside write, Reload IDs
    (reconciles a in memory only, pre-fix), a SECOND outside write to a
    DIFFERENT frame c, then a save: the pre-fix save saw 'a' as no longer
    dirty and took it from disk (still the pre-edit content, since reload
    never wrote), silently discarding the edit. The fix makes Reload IDs end
    with a real save_segmentations() call, so 'a' only ever leaves the dirty
    set once its edit is actually on disk."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="defectA_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"a.jpg": {"tag": "a0"}, "c.jpg": {"tag": "c0"}})
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = A.load_segmentations(export_dir)
            A.session["class_map"] = {}

            A.session["segmentations"]["a.jpg"]["tag"] = "a_local_edit"
            _set_dirty(A, "a.jpg")

            # An outside write that changes the file (so reload's own stat/
            # hash check has something to reconcile) without touching a's
            # own content.
            _write_json(seg_path, {"a.jpg": {"tag": "a0"}, "c.jpg": {"tag": "c0"}})
            _bump_mtime(seg_path, 5)

            with A.app.test_client() as c:
                r = c.post("/api/reload_segmentations", json={"filename": "a.jpg"})
            check(r.status_code == 200, f"{label}: reload 200 (got {r.status_code})")
            check(A.session["segmentations"]["a.jpg"]["tag"] == "a_local_edit",
                  f"{label}: a's local edit survives the reload itself")

            # A second outside write, to c this time; a is not touched by it.
            with open(seg_path) as f:
                disk_now = json.load(f)
            disk_now["c.jpg"]["tag"] = "c_from_importer"
            _write_json(seg_path, disk_now)
            _bump_mtime(seg_path, 5)

            status, info = A.save_segmentations(export_dir, A.session["segmentations"])
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk["a.jpg"]["tag"] == "a_local_edit",
                  f"{label}: CRITICAL FIX: a's edit survives the save after reload "
                  f"and a second outside write elsewhere, got {on_disk.get('a.jpg')}")
            check(on_disk["c.jpg"]["tag"] == "c_from_importer",
                  f"{label}: c's outside edit is captured too, got {on_disk.get('c.jpg')}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_conflict_file_is_additive_across_different_frames(app_dir, label):
    """Defect B (important): the per-episode conflict file must not be
    overwritten with only the CURRENT save's both-sides frames. Two saves in
    the same episode, each conflicting on a DIFFERENT frame, must both be
    readable from the one conflict file afterward."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="defectB_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}, "B.jpg": {"tag": "b0"}})
            A.load_segmentations(export_dir)

            _write_json(seg_path, {"A.jpg": {"tag": "a_outside"}, "B.jpg": {"tag": "b0"}})
            _bump_mtime(seg_path, 5)
            _set_dirty(A, "A.jpg")
            status1, info1 = A.save_segmentations(
                export_dir, {"A.jpg": {"tag": "a_local"}, "B.jpg": {"tag": "b0"}})
            check(status1 == "merged" and info1["both_sides"] == 1,
                  f"{label}: first save conflicts on A, got {status1!r} {info1}")

            with open(seg_path) as f:
                disk_now = json.load(f)
            disk_now["B.jpg"]["tag"] = "b_outside"
            _write_json(seg_path, disk_now)
            _bump_mtime(seg_path, 5)
            _set_dirty(A, "B.jpg")
            status2, info2 = A.save_segmentations(
                export_dir, {"A.jpg": {"tag": "a_local"}, "B.jpg": {"tag": "b_local"}})
            check(status2 == "merged" and info2["both_sides"] == 1,
                  f"{label}: second save conflicts on B, got {status2!r} {info2}")

            check(info1["conflict_path"] == info2["conflict_path"],
                  f"{label}: both saves reused the same episode file")
            with open(info2["conflict_path"]) as f:
                conflict_content = json.load(f)
            check("A.jpg" in conflict_content and "B.jpg" in conflict_content,
                  f"{label}: BOTH frames' entries survive, got keys {list(conflict_content)}")
            check(conflict_content["A.jpg"][-1]["content"] == {"tag": "a_outside"},
                  f"{label}: A's recorded outside version, got {conflict_content['A.jpg']}")
            check(conflict_content["B.jpg"][-1]["content"] == {"tag": "b_outside"},
                  f"{label}: B's recorded outside version, got {conflict_content['B.jpg']}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_conflict_file_keeps_every_distinct_version_of_one_frame(app_dir, label):
    """Defect B (important), the same-frame case: two saves conflicting on
    the SAME frame with two DIFFERENT outside versions must both be kept, in
    order, each with a captured time."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="defectB2_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}})
            A.load_segmentations(export_dir)

            _write_json(seg_path, {"A.jpg": {"tag": "a_outside_1"}})
            _bump_mtime(seg_path, 5)
            _set_dirty(A, "A.jpg")
            _, info1 = A.save_segmentations(export_dir, {"A.jpg": {"tag": "a_local"}})
            check(info1["both_sides"] == 1, f"{label}: first save conflicts")

            _write_json(seg_path, {"A.jpg": {"tag": "a_outside_2"}})
            _bump_mtime(seg_path, 5)
            _set_dirty(A, "A.jpg")
            _, info2 = A.save_segmentations(export_dir, {"A.jpg": {"tag": "a_local"}})
            check(info2["both_sides"] == 1, f"{label}: second save conflicts again")

            with open(info2["conflict_path"]) as f:
                content = json.load(f)
            versions = content.get("A.jpg", [])
            check(len(versions) == 2,
                  f"{label}: both distinct outside versions of A are kept, got {versions}")
            check(versions[0]["content"] == {"tag": "a_outside_1"}
                  and versions[1]["content"] == {"tag": "a_outside_2"},
                  f"{label}: versions recorded in order, got {versions}")
            check("at" in versions[0] and "at" in versions[1],
                  f"{label}: each version carries a captured time, got {versions}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_reload_handles_truncated_json_without_a_500(app_dir, label):
    """Minor fix: reload_segmentations() must handle a truncated/invalid
    on-disk file the same way save_segmentations() does: a clear error, no
    500, and the in-memory session left exactly as it was."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="defectD_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"A.jpg": {"tag": "a0"}})
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = A.load_segmentations(export_dir)
            A.session["class_map"] = {}
            A.session["segmentations"]["A.jpg"]["tag"] = "a_local"
            _set_dirty(A, "A.jpg")

            with open(seg_path, "w") as f:
                f.write('{"A.jpg": {"tag": ')  # truncated, invalid JSON

            with A.app.test_client() as c:
                r = c.post("/api/reload_segmentations", json={"filename": "A.jpg"})
            check(r.status_code != 500, f"{label}: truncated JSON never 500s, got {r.status_code}")
            body = r.get_json()
            check(bool(body and body.get("error")), f"{label}: a clear error message, got {body}")
            check(A.session["segmentations"]["A.jpg"]["tag"] == "a_local",
                  f"{label}: the in-memory edit is untouched by the failed reload")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_concurrent_route_and_merging_save_do_not_lose_the_route_edit(app_dir, label):
    """Defect C (important): a mutating route's edit must not be lost when
    it races a concurrent save whose merge replaces the store
    (data.clear(); data.update(...)). Proven with two real threads: a save
    is forced to pause mid-merge (a hook on _merge_store_with_disk) while
    STILL HOLDING _store_write_lock; a mutating route fired on a second
    thread during that pause must BLOCK (not run concurrently and touch
    session['segmentations']), and once the save releases the lock, the
    route's edit must reach the FINAL session state, not a dict the
    merge already replaced out from under it."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="defectC_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"FRAME.jpg": {"scrapped": False, "reviewed": False, "masks": []}})
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = A.load_segmentations(export_dir)
            A.session["class_map"] = {}

            # Force the next save into its merge branch.
            _write_json(seg_path, {"FRAME.jpg": {"scrapped": False, "reviewed": False, "masks": []},
                                    "OTHER.jpg": {"tag": "x"}})
            _bump_mtime(seg_path, 5)
            _set_dirty(A, "OTHER.jpg")

            merge_entered = threading.Event()
            release_merge = threading.Event()
            _orig_merge = A._merge_store_with_disk

            def blocking_merge(*args, **kwargs):
                merge_entered.set()
                release_merge.wait(5)
                return _orig_merge(*args, **kwargs)

            A._merge_store_with_disk = blocking_merge

            save_thread = threading.Thread(
                target=lambda: A.save_segmentations(export_dir, A.session["segmentations"]))
            save_thread.start()
            check(merge_entered.wait(2), f"{label}: the save reached its merge")

            route_ran = threading.Event()

            def run_route():
                with A.app.test_client() as c:
                    c.post("/api/image/FRAME.jpg/scrap")
                route_ran.set()

            route_thread = threading.Thread(target=run_route)
            route_thread.start()
            blocked = not route_ran.wait(0.3)
            check(blocked, f"{label}: the route blocks while the save holds the lock mid-merge")

            release_merge.set()
            save_thread.join(5)
            route_thread.join(5)
            A._merge_store_with_disk = _orig_merge

            check(route_ran.is_set(), f"{label}: the route completed once the lock freed")
            check(A.session["segmentations"]["FRAME.jpg"]["scrapped"] is True,
                  f"{label}: the route's edit reached the FINAL session state, not an "
                  f"object the merge's data.clear()/data.update() had already replaced")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_dirty_on_success_holds_store_write_lock(app_dir, label):
    """Static check: _dirty_on_success wraps the view call and the dirty
    mark in the same _store_write_lock save_segmentations() uses."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    body = src.split("def _dirty_on_success(view):", 1)[-1].split("\ndef ", 1)[0]
    check("with _store_write_lock:" in body,
          f"{label}: _dirty_on_success acquires _store_write_lock")
    lock_body = body.split("with _store_write_lock:", 1)[-1]
    check("_mark_dirty(" in lock_body,
          f"{label}: the dirty mark happens inside that lock")


# ── Edit Masks parity: pagehide + trailing flush (template only) ─────────

def test_edit_masks_has_pagehide_and_trailing_flush():
    with open(os.path.join(EM, "src", "templates", "index.html")) as f:
        tpl = f.read()
    check("addEventListener('pagehide'" in tpl,
          "editMasks: a pagehide listener is registered")
    check(tpl.count("addEventListener('pagehide'") == 1,
          "editMasks: exactly one pagehide listener")
    ph_body = tpl.split("addEventListener('pagehide'", 1)[-1].split("});", 1)[0]
    check("sendBeacon(" in ph_body and "/api/flush" in ph_body,
          "editMasks: the pagehide handler sends the flush beacon")
    check("function scheduleFlush()" in tpl, "editMasks: scheduleFlush exists")
    sched_body = tpl.split("function scheduleFlush()", 1)[-1].split("\n}\n", 1)[0]
    check("2000" in sched_body, "editMasks: the trailing flush is a 2s timer")
    check("function setConflictStatus(" in tpl, "editMasks: setConflictStatus exists")
    check(".status.conflict" in tpl, "editMasks: the conflict status color class exists")


for _app_dir, _label in ((CA, "combinedAnnotate"), (EM, "editMasks")):
    print(f"\n[{_label}] basic save/merge status contract")
    test_first_save_of_a_session_that_never_loaded_proceeds(_app_dir, _label)
    test_a_normal_dirty_save_with_no_disk_change_is_plain_saved(_app_dir, _label)
    test_mtime_tracking_survives_its_own_saves(_app_dir, _label)
    test_same_content_different_stat_is_not_a_conflict(_app_dir, _label)
    print(f"\n[{_label}] per-frame merge")
    test_frames_edited_only_outside_are_taken_into_memory(_app_dir, _label)
    test_both_sides_frame_keeps_local_and_folds_review_fields(_app_dir, _label)
    test_one_episode_yields_at_most_one_conflict_file_across_five_saves(_app_dir, _label)
    test_an_edit_during_the_write_is_not_lost(_app_dir, _label)
    print(f"\n[{_label}] Reload IDs full-store merge")
    test_conflict_then_reload_then_save_keeps_the_other_frames_expert_id(_app_dir, _label)
    print(f"\n[{_label}] session lifecycle")
    test_an_idle_resumed_session_writes_nothing_on_flush_or_quit(_app_dir, _label)
    test_a_failed_open_does_not_write_the_old_store_into_the_new_folder(_app_dir, _label)
    test_configure_and_resume_do_not_mark_dirty(_app_dir, _label)
    test_scrap_unscrap_export_mark_dirty_before_their_own_flush(_app_dir, _label)
    print(f"\n[{_label}] /api/flush and scrap/unscrap via the test client")
    test_force_flush_reports_a_merge_via_test_client(_app_dir, _label)
    test_force_flush_reports_a_conflict_via_test_client(_app_dir, _label)
    print(f"\n[{_label}] static wiring")
    test_dirty_gate_is_wired_into_flush_before_exit_and_force_flush(_app_dir, _label)
    test_every_known_mutating_route_is_decorated(_app_dir, _label)
    test_self_flushing_routes_are_not_decorated(_app_dir, _label)
    test_load_routes_are_not_decorated(_app_dir, _label)
    test_exemplar_scan_is_not_decorated(_app_dir, _label)
    print(f"\n[{_label}] background loop merges per image")
    test_background_style_per_image_save_merges_not_diverts(_app_dir, _label)
    print(f"\n[{_label}] second review pass: defect A (critical)")
    test_reload_then_second_outside_write_then_save_keeps_the_reloaded_edit(_app_dir, _label)
    print(f"\n[{_label}] second review pass: defect B (conflict file additive)")
    test_conflict_file_is_additive_across_different_frames(_app_dir, _label)
    test_conflict_file_keeps_every_distinct_version_of_one_frame(_app_dir, _label)
    print(f"\n[{_label}] second review pass: minor fix (truncated JSON on reload)")
    test_reload_handles_truncated_json_without_a_500(_app_dir, _label)
    print(f"\n[{_label}] second review pass: defect C (lock around routes)")
    test_dirty_on_success_holds_store_write_lock(_app_dir, _label)
    test_concurrent_route_and_merging_save_do_not_lose_the_route_edit(_app_dir, _label)

print("\n[editMasks] template parity")
test_edit_masks_has_pagehide_and_trailing_flush()


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
