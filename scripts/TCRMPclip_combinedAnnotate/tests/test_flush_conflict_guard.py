"""save_segmentations() never overwrites a segmentations.json an outside
writer changed, never loses an edit that lands mid-write, and never tells
the operator an edit saved when it did not.

VEGA annotation safety net, 2026-09-27 (LO): independent review of task T1
found that flush_all() wrote segmentations.json from memory unconditionally,
with no check for whether the file changed on disk first. The expert-ID
importer (MOD/scripts/_expertids/importer.py, _seg_locked/_save around
lines 192-245) can rewrite that same file while an annotator or Edit Masks
sits open on the same project; before T1's save-on-quit and pre-kill-flush
paths existed, an idle process never wrote on its own, so this was never a
problem. A second review pass then found two further defects in the first
fix, both covered here too. This suite locks four things in app.py (both
combinedAnnotate and editMasks, same shape in each):

  1. A monotonic dirty counter (_segmentations_dirty_counter /
     _segmentations_saved_version / _is_dirty()), incremented by
     _mark_dirty() from every route that can change
     session['segmentations'] (@_dirty_on_success) plus the one non-route
     mutation point (the background segmentation loop). flush_before_exit()
     and the /api/flush route write only while _is_dirty() is true, so an
     idle session with nothing new never touches the file at all. The
     counter (not a plain boolean) is what makes an edit that lands WHILE a
     save is still writing survive: save_segmentations() snapshots the
     counter before it starts serializing and only advances
     _segmentations_saved_version to that snapshot once the write lands, so
     a later increment (an edit arriving mid-write) is never silently
     cleared by the save it raced with.
  2. A per-path (mtime_ns, size) identity (_seg_disk_identity) plus a sha256
     content hash (_seg_disk_hash), both recorded at load and after every
     successful save. save_segmentations() compares the file's current stat
     against the identity first; a mismatch is then checked against the
     hash before anything is refused, since a touch or a same-content
     rewrite moves the stat without changing the bytes. Real content
     drift diverts the in-memory copy to a sibling
     segmentations.conflict-<AST stamp>.json instead, and the real file is
     left byte-for-byte untouched; a save keeps diverting on every later
     attempt while the mismatch stays unresolved.
  3. /api/flush, and the two other routes whose entire job after mutating is
     to flush (scrap, unscrap), answer {ok: false, conflict: true,
     conflict_file: <name>} instead of ok: true when the underlying save
     diverted, so the client is never told an edit saved when it did not.

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


def _set_dirty(A):
    """Force a dirty session, independent of the counter's current value."""
    A._mark_dirty()


def _set_clean(A):
    """Force a clean session: the next save has nothing new to capture."""
    A._segmentations_saved_version = A._segmentations_dirty_counter


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


def test_idle_process_with_a_newer_disk_file_writes_nothing(app_dir, label):
    """A session with no unsaved change (_is_dirty() False) makes no save
    attempt at all when it quits, even though the disk file changed
    underneath it: the dirty gate, not the conflict check, is what protects
    this case, and it must never even reach save_segmentations()."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="idle_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            original = {"FRAME.jpg": {"masks": [], "tag": "original"}}
            _write_json(seg_path, original)

            loaded = A.load_segmentations(export_dir)
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = loaded
            A.session["class_map"] = {}
            _set_clean(A)
            A._quit_flushed = False

            # The importer rewrites the file while the (idle) process sits open.
            imported = {"FRAME.jpg": {"masks": [], "tag": "from_importer"}}
            _write_json(seg_path, imported)
            _bump_mtime(seg_path, 5)
            before_bytes = open(seg_path, "rb").read()

            calls = []
            _orig_save = A.save_segmentations
            A.save_segmentations = lambda *a, **k: calls.append(1)
            A.flush_before_exit()
            A.save_segmentations = _orig_save

            check(calls == [],
                  f"{label}: an idle quit never even calls save_segmentations")
            check(open(seg_path, "rb").read() == before_bytes,
                  f"{label}: the importer's file on disk is untouched")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_dirty_process_with_a_newer_disk_file_writes_the_conflict_sibling(app_dir, label):
    """A genuinely dirty session whose tracked file changed content on disk
    (an outside writer, in practice the expert-ID importer) diverts its
    save to a conflict sibling and leaves the real file exactly as the
    outside writer left it."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="dirty_conflict_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_dir = os.path.dirname(A._seg_path(export_dir))
            seg_path = A._seg_path(export_dir)
            original = {"FRAME.jpg": {"masks": [], "tag": "original"}}
            _write_json(seg_path, original)
            A.load_segmentations(export_dir)  # records the tracked identity + hash

            imported = {"FRAME.jpg": {"masks": [], "tag": "from_importer"}}
            _write_json(seg_path, imported)
            _bump_mtime(seg_path, 5)
            importer_bytes = open(seg_path, "rb").read()

            in_memory = {"FRAME.jpg": {"masks": [{"id": 1}], "tag": "operator_edit"}}
            result, conflict_path = A.save_segmentations(export_dir, in_memory)

            check(result == "conflict",
                  f"{label}: a save against a changed file reports 'conflict', got {result!r}")
            check(conflict_path is not None and os.path.isfile(conflict_path),
                  f"{label}: the reported conflict path exists, got {conflict_path!r}")
            check(open(seg_path, "rb").read() == importer_bytes,
                  f"{label}: the real file is byte-identical to the importer's version")

            siblings = [f for f in os.listdir(seg_dir)
                        if f.startswith("segmentations.conflict-") and f.endswith(".json")]
            check(len(siblings) == 1,
                  f"{label}: exactly one conflict sibling was written, found {siblings}")
            if siblings:
                with open(os.path.join(seg_dir, siblings[0])) as f:
                    sibling_content = json.load(f)
                check(sibling_content == in_memory,
                      f"{label}: the conflict sibling carries the in-memory copy, "
                      f"got {sibling_content}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_same_content_different_stat_is_not_a_conflict(app_dir, label):
    """A stat mismatch alone is not a conflict. When the disk file's bytes
    are identical to what this process already holds (a touch, or a rewrite
    that reproduced the same content), the sha256 comparison against
    _seg_disk_hash overrules the stat mismatch: the save proceeds normally
    and the stored identity simply adopts the new stat."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="same_content_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            content = {"FRAME.jpg": {"tag": "same"}}
            _write_json(seg_path, content)
            A.load_segmentations(export_dir)

            # An outside process rewrote the file with the IDENTICAL bytes: the
            # mtime moves, the content does not.
            _write_json(seg_path, content)
            _bump_mtime(seg_path, 5)

            result, conflict_path = A.save_segmentations(export_dir, content)
            check(result == "saved",
                  f"{label}: identical content on disk is not a conflict, got {result!r}")
            check(conflict_path is None,
                  f"{label}: no conflict path is reported for a same-content save")

            seg_dir = os.path.dirname(seg_path)
            siblings = [f for f in os.listdir(seg_dir) if "conflict" in f]
            check(siblings == [], f"{label}: no conflict sibling was written, found {siblings}")

            # The adopted identity holds for the NEXT save too (no lingering
            # mismatch from the stat bump).
            second, _ = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "next"}})
            check(second == "saved",
                  f"{label}: the following save is still 'saved', not a false "
                  f"conflict left over from the stat bump, got {second!r}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_conflict_keeps_diverting_until_reconciled(app_dir, label):
    """After a real conflict, later saves keep diverting (never overwrite
    the newer disk file) as long as the mismatch is unresolved: each
    diverted save reports conflict, so a run of edits during an unresolved
    conflict is never quietly reported as saved partway through."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="keep_diverting_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"a": 1})
            A.load_segmentations(export_dir)
            _write_json(seg_path, {"a": "from_importer"})
            _bump_mtime(seg_path, 5)

            first, _ = A.save_segmentations(export_dir, {"a": 2})
            check(first == "conflict", f"{label}: first save after the change is 'conflict'")
            second, _ = A.save_segmentations(export_dir, {"a": 3})
            check(second == "conflict",
                  f"{label}: a second save with the mismatch still unresolved is "
                  f"ALSO 'conflict', not silently 'saved'")

            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk == {"a": "from_importer"},
                  f"{label}: the real file is still exactly what the outside writer "
                  f"left, got {on_disk}")

            # The conflict filename's AST-stamp resolution is one second, so two
            # diversions inside the same second share a name and the later one
            # overwrites the earlier sibling; what matters is that at least one
            # sibling exists and it holds the LATEST diverted attempt, not that
            # every single attempt left behind a distinct file.
            seg_dir = os.path.dirname(seg_path)
            siblings = [f for f in os.listdir(seg_dir) if "conflict" in f]
            check(len(siblings) >= 1,
                  f"{label}: at least one conflict sibling exists, found {siblings}")
            if siblings:
                with open(os.path.join(seg_dir, sorted(siblings)[-1])) as f:
                    latest_sibling = json.load(f)
                check(latest_sibling == {"a": 3},
                      f"{label}: the conflict sibling holds the latest diverted "
                      f"attempt, got {latest_sibling}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_a_conflict_is_logged_naming_both_files(app_dir, label):
    """The conflict path logs exactly one line naming the real path and the
    sibling it wrote instead."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="dirty_conflict_log_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"a": 1})
            A.load_segmentations(export_dir)
            _write_json(seg_path, {"a": 2})
            _bump_mtime(seg_path, 5)

            logged = []
            _orig_log = A.log
            A.log = lambda msg: logged.append(msg)
            A.save_segmentations(export_dir, {"a": 3})
            A.log = _orig_log

            check(len(logged) == 1, f"{label}: exactly one log line, got {logged}")
            if logged:
                check(seg_path in logged[0],
                      f"{label}: the log line names the real path, got {logged[0]!r}")
                check("segmentations.conflict-" in logged[0],
                      f"{label}: the log line names the conflict sibling, "
                      f"got {logged[0]!r}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_a_normal_dirty_save_still_works(app_dir, label):
    """No external change: a dirty session's save proceeds normally, writes
    the real file, and reports 'saved'."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="normal_save_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            seg_path = A._seg_path(export_dir)
            _write_json(seg_path, {"FRAME.jpg": {"tag": "original"}})
            A.load_segmentations(export_dir)

            updated = {"FRAME.jpg": {"tag": "edited"}}
            result, conflict_path = A.save_segmentations(export_dir, updated)

            check(result == "saved", f"{label}: no external change -> 'saved', got {result!r}")
            check(conflict_path is None, f"{label}: no conflict path on a normal save")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk == updated, f"{label}: the edit reached disk, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_mtime_tracking_survives_its_own_saves(app_dir, label):
    """A second save right after the first is not mistaken for a conflict
    with itself: the tracked identity must be refreshed to the file this
    process just wrote, not left pointing at the load-time stat."""
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

            # No external touch in between; a naive design that never refreshed
            # its tracked identity after its own write would see THIS save's
            # future mtime as "changed since load" and wrongly flag a conflict.
            second, _ = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "edit_2"}})
            check(second == "saved",
                  f"{label}: second consecutive save is still 'saved', not a "
                  f"false conflict with its own prior write, got {second!r}")
            with open(seg_path) as f:
                on_disk = json.load(f)
            check(on_disk == {"FRAME.jpg": {"tag": "edit_2"}},
                  f"{label}: the second edit is what's on disk, got {on_disk}")

            seg_dir = os.path.dirname(seg_path)
            siblings = [f for f in os.listdir(seg_dir) if "conflict" in f]
            check(siblings == [], f"{label}: no conflict sibling was ever written, found {siblings}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_first_save_of_a_session_that_never_loaded_proceeds(app_dir, label):
    """A brand-new export_dir with no on-disk file yet (nothing was ever
    loaded) is not a conflict: the first save just writes normally, matching
    every pre-existing test that calls save_segmentations() straight into a
    fresh temp dir with no prior load_segmentations() call."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="first_save_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            result, conflict_path = A.save_segmentations(export_dir, {"FRAME.jpg": {"tag": "seed"}})
            check(result == "saved",
                  f"{label}: a fresh export_dir's first save proceeds, got {result!r}")
            check(conflict_path is None, f"{label}: no conflict path on the first save")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_an_edit_during_the_write_is_not_lost(app_dir, label):
    """A save that snapshots the dirty counter before it starts serializing
    must not clear a NEWER edit that arrived while the write itself was in
    flight (a click ~2s after the last, mid a large write): a plain boolean
    flag would have cleared itself unconditionally at the end of that write,
    silently discarding exactly this edit. Simulated via a hook on
    _atomic_write_json, which save_segmentations() calls AFTER it has
    already snapshotted the counter, so a _mark_dirty() from inside the hook
    reproduces an edit landing mid-write with no real threads or timing
    needed."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="race_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"FRAME.jpg": {"tag": "v1"}}
            A.session["class_map"] = {}
            _set_dirty(A)

            _orig_atomic = A._atomic_write_json
            racing_edit_applied = []

            def racing_atomic_write(path, payload, indent):
                # An edit "arrives" after this save already snapshotted the
                # dirty counter, but before the write (and the version bump
                # that follows it) completes.
                A.session["segmentations"]["FRAME.jpg"]["tag"] = "v2_mid_write"
                A._mark_dirty()
                racing_edit_applied.append(1)
                return _orig_atomic(path, payload, indent)

            A._atomic_write_json = racing_atomic_write
            result, _ = A.save_segmentations(export_dir, A.session["segmentations"])
            A._atomic_write_json = _orig_atomic

            check(racing_edit_applied == [1], f"{label}: the racing edit hook actually ran")
            check(result == "saved", f"{label}: the save itself still succeeds, got {result!r}")
            check(A._is_dirty(),
                  f"{label}: the session is still dirty after a save that raced with a "
                  f"new edit (a boolean flag would have wrongly cleared here)")

            # The next flush must actually write again, capturing the racing edit.
            second, _ = A.save_segmentations(export_dir, A.session["segmentations"])
            check(second == "saved", f"{label}: the follow-up flush writes normally, got {second!r}")
            check(not A._is_dirty(), f"{label}: clean again once that flush completes")
            with open(A._seg_path(export_dir)) as f:
                on_disk = json.load(f)
            check(on_disk["FRAME.jpg"]["tag"] == "v2_mid_write",
                  f"{label}: the racing edit's content reached disk on the follow-up "
                  f"save, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_dirty_counter_survives_two_races_in_the_same_write(app_dir, label):
    """Two edits landing during the same in-flight write both survive: the
    session is dirty after the racing save, and the follow-up save captures
    both (checked by content, not just the dirty flag, to prove nothing was
    dropped between the two racing edits themselves)."""
    A = _import_app(app_dir)
    try:
        d = tempfile.mkdtemp(prefix="race2_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"FRAME.jpg": {"tag": "v1"}, "OTHER.jpg": {"tag": "v1"}}
            A.session["class_map"] = {}
            _set_dirty(A)

            _orig_atomic = A._atomic_write_json

            def racing_atomic_write(path, payload, indent):
                A.session["segmentations"]["FRAME.jpg"]["tag"] = "v2"
                A._mark_dirty()
                A.session["segmentations"]["OTHER.jpg"]["tag"] = "v2"
                A._mark_dirty()
                return _orig_atomic(path, payload, indent)

            A._atomic_write_json = racing_atomic_write
            A.save_segmentations(export_dir, A.session["segmentations"])
            A._atomic_write_json = _orig_atomic

            check(A._is_dirty(), f"{label}: still dirty after two racing edits mid-write")
            A.save_segmentations(export_dir, A.session["segmentations"])
            check(not A._is_dirty(), f"{label}: clean after the follow-up captures both")
            with open(A._seg_path(export_dir)) as f:
                on_disk = json.load(f)
            check(on_disk["FRAME.jpg"]["tag"] == "v2" and on_disk["OTHER.jpg"]["tag"] == "v2",
                  f"{label}: both racing edits reached disk, got {on_disk}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_dirty_gate_is_wired_into_flush_before_exit_and_force_flush(app_dir, label):
    """Static check: both quiet save paths actually reference _is_dirty()
    (locks the wiring in place independent of the behavioural tests above,
    the way test_polish_20260901.py locks route wiring by reading source)."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    fbe = src.split("def flush_before_exit():", 1)[-1].split("\ndef ", 1)[0]
    check("_is_dirty()" in fbe,
          f"{label}: flush_before_exit() checks _is_dirty()")
    ff = src.split("def force_flush():", 1)[-1].split("\n\n\n", 1)[0]
    check("_is_dirty()" in ff,
          f"{label}: the /api/flush route (force_flush) checks _is_dirty()")


_MUTATING_ROUTES = [
    "configure", "resume", "update_masks", "accept_all_masks", "commit_mask",
    "refine_mask", "rebox_mask", "draw_edit", "undo_refine", "reset_mask",
    "merge_masks", "merge_many", "merge_same_id", "reload_segmentations",
    "scrap_segmentation", "unscrap_segmentation", "add_mask", "add_mask_box",
    "do_export", "set_mask_size",
]


def test_every_known_mutating_route_is_decorated(app_dir, label, extra_routes=()):
    """Every route that changes session['segmentations'] carries
    @_dirty_on_success directly above its def line. Reads source text rather
    than driving each route live, mirroring test_polish_20260901.py's own
    structural route checks; combinedAnnotate additionally carries
    import_images and standalone_start (editMasks never runs standalone)."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    for name in list(_MUTATING_ROUTES) + list(extra_routes):
        marker = f"def {name}("
        idx = src.find(marker)
        check(idx != -1, f"{label}: {name}() exists")
        if idx == -1:
            continue
        preceding = src[:idx].rstrip().splitlines()
        check(bool(preceding) and preceding[-1].strip() == "@_dirty_on_success",
              f"{label}: {name}() is decorated with @_dirty_on_success "
              f"(line above def: {preceding[-1] if preceding else '<none>'!r})")


def test_exemplar_scan_is_not_decorated(app_dir, label):
    """exemplar_scan's own docstring says its candidates are NOT persisted to
    seg['masks']; it must not be marked dirty (a false-dirty session would
    just cost an extra harmless save, but the point of the decorator is to
    reflect real mutations, and this route is the one route in the file that
    explicitly is not one)."""
    with open(os.path.join(app_dir, "src", "app.py")) as f:
        src = f.read()
    idx = src.find("def exemplar_scan(")
    check(idx != -1, f"{label}: exemplar_scan() exists")
    if idx == -1:
        return
    preceding = src[:idx].rstrip().splitlines()
    check(bool(preceding) and preceding[-1].strip() != "@_dirty_on_success",
          f"{label}: exemplar_scan() carries no @_dirty_on_success "
          f"(line above def: {preceding[-1] if preceding else '<none>'!r})")


def test_decorator_marks_dirty_only_on_success(app_dir, label):
    """The @_dirty_on_success decorator itself: a 200-ish response marks
    dirty, an error response (>=400) does not."""
    A = _import_app(app_dir)
    try:
        calls = {"count": 0}
        real_mark = A._mark_dirty

        def counting_mark():
            calls["count"] += 1
            real_mark()

        A._mark_dirty = counting_mark

        @A._dirty_on_success
        def ok_view():
            from flask import jsonify
            return jsonify({"ok": True})

        @A._dirty_on_success
        def error_view():
            from flask import jsonify
            return jsonify({"error": "nope"}), 404

        with A.app.test_request_context():
            _set_clean(A)
            ok_view()
            check(A._is_dirty() and calls["count"] == 1,
                  f"{label}: a success response marks dirty")

            _set_clean(A)
            calls["count"] = 0
            error_view()
            check(not A._is_dirty() and calls["count"] == 0,
                  f"{label}: an error response (404) does not mark dirty")
        A._mark_dirty = real_mark
    finally:
        _release(app_dir)


def test_a_mutating_route_through_the_test_client_marks_dirty(app_dir, label):
    """End-to-end: a real mutating request through the Flask test client
    (scrap, the simplest one with no SAM3 engine dependency) sets the dirty
    flag; a request for a frame that does not exist (404) does not."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        A.session["export_dir"] = ""
        A.session["segmentations"] = {
            "FRAME.jpg": {"masks": [], "scrapped": False, "reviewed": False},
        }
        A.session["class_map"] = {}
        _set_clean(A)
        with A.app.test_client() as c:
            r = c.post("/api/image/FRAME.jpg/scrap")
            check(r.status_code == 200, f"{label}: /scrap 200 (got {r.status_code})")
            check(A._is_dirty(),
                  f"{label}: a real mutating request through the test client marks dirty")

            _set_clean(A)
            r = c.post("/api/image/MISSING.jpg/scrap")
            check(r.status_code == 404, f"{label}: /scrap on an unknown frame is 404")
            check(not A._is_dirty(),
                  f"{label}: a 404 that changed nothing does not mark dirty")
    finally:
        _release(app_dir)


def _prime_conflict(A, export_dir):
    """Set up a project whose segmentations.json changes on disk (an outside
    writer) after this process's own load, so the next save diverts.
    Returns nothing; the caller configures session['segmentations'] itself
    and marks the session dirty."""
    seg_path = A._seg_path(export_dir)
    _write_json(seg_path, {"a": 1})
    A.load_segmentations(export_dir)
    _write_json(seg_path, {"a": "from_importer"})
    _bump_mtime(seg_path, 5)


def test_force_flush_reports_conflict_via_test_client(app_dir, label):
    """POST /api/flush answers {ok: false, conflict: true, conflict_file:
    <name>} when the underlying save diverted, instead of claiming ok: true
    for an edit that never reached segmentations.json."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        d = tempfile.mkdtemp(prefix="flush_conflict_client_%s_" % label)
        try:
            export_dir = os.path.join(d, "out")
            _prime_conflict(A, export_dir)
            A.session["export_dir"] = export_dir
            A.session["segmentations"] = {"a": 2}
            A.session["class_map"] = {}
            _set_dirty(A)

            with A.app.test_client() as c:
                r = c.post("/api/flush")
                body = r.get_json()
            check(r.status_code == 200, f"{label}: /api/flush still answers 200 (got {r.status_code})")
            check(body.get("ok") is False, f"{label}: ok is false on a conflict, got {body}")
            check(body.get("conflict") is True, f"{label}: conflict is true, got {body}")
            conflict_file = body.get("conflict_file") or ""
            check(conflict_file.startswith("segmentations.conflict-")
                  and conflict_file.endswith(".json"),
                  f"{label}: conflict_file names the sibling, got {conflict_file!r}")
            seg_dir = os.path.dirname(A._seg_path(export_dir))
            check(os.path.exists(os.path.join(seg_dir, conflict_file)),
                  f"{label}: the named conflict file actually exists on disk")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


def test_scrap_and_unscrap_report_conflict_via_test_client(app_dir, label):
    """POST .../scrap and .../unscrap, whose entire persistence action is
    also a flush_all() call, answer the same {ok: false, conflict: true,
    conflict_file} shape as /api/flush when that flush diverts."""
    A = _import_app(app_dir)
    try:
        A.app.config["TESTING"] = True
        for route, seed in (("scrap", False), ("unscrap", True)):
            d = tempfile.mkdtemp(prefix="scrap_conflict_%s_%s_" % (route, label))
            try:
                export_dir = os.path.join(d, "out")
                _prime_conflict(A, export_dir)
                A.session["export_dir"] = export_dir
                A.session["segmentations"] = {
                    "FRAME.jpg": {"masks": [], "scrapped": seed, "reviewed": False},
                }
                A.session["class_map"] = {}

                with A.app.test_client() as c:
                    r = c.post(f"/api/image/FRAME.jpg/{route}")
                    body = r.get_json()
                check(r.status_code == 200,
                      f"{label}: /{route} still answers 200 (got {r.status_code})")
                check(body.get("ok") is False,
                      f"{label}: /{route} reports ok false on a conflict, got {body}")
                check(body.get("conflict") is True,
                      f"{label}: /{route} reports conflict true, got {body}")
                check((body.get("conflict_file") or "").endswith(".json"),
                      f"{label}: /{route} names the conflict file, got {body}")
                check("remaining" in body,
                      f"{label}: /{route} keeps its own 'remaining' field alongside "
                      f"the conflict fields, got {body}")
            finally:
                shutil.rmtree(d, ignore_errors=True)
    finally:
        _release(app_dir)


for _app_dir, _label in ((CA, "combinedAnnotate"), (EM, "editMasks")):
    print(f"\n[{_label}] idle quit writes nothing")
    test_idle_process_with_a_newer_disk_file_writes_nothing(_app_dir, _label)
    print(f"\n[{_label}] dirty + newer disk file -> conflict sibling")
    test_dirty_process_with_a_newer_disk_file_writes_the_conflict_sibling(_app_dir, _label)
    print(f"\n[{_label}] same content, different stat -> not a conflict")
    test_same_content_different_stat_is_not_a_conflict(_app_dir, _label)
    print(f"\n[{_label}] a conflict keeps diverting until reconciled")
    test_conflict_keeps_diverting_until_reconciled(_app_dir, _label)
    print(f"\n[{_label}] conflict is logged naming both files")
    test_a_conflict_is_logged_naming_both_files(_app_dir, _label)
    print(f"\n[{_label}] normal dirty save still works")
    test_a_normal_dirty_save_still_works(_app_dir, _label)
    print(f"\n[{_label}] mtime tracking survives its own saves")
    test_mtime_tracking_survives_its_own_saves(_app_dir, _label)
    print(f"\n[{_label}] first save of a never-loaded session proceeds")
    test_first_save_of_a_session_that_never_loaded_proceeds(_app_dir, _label)
    print(f"\n[{_label}] an edit during the write is not lost")
    test_an_edit_during_the_write_is_not_lost(_app_dir, _label)
    print(f"\n[{_label}] the dirty counter survives two races in one write")
    test_dirty_counter_survives_two_races_in_the_same_write(_app_dir, _label)
    print(f"\n[{_label}] dirty gate wired into both quiet save paths")
    test_dirty_gate_is_wired_into_flush_before_exit_and_force_flush(_app_dir, _label)
    print(f"\n[{_label}] every known mutating route is decorated")
    extra = ("import_images", "standalone_start") if _app_dir is CA else ()
    test_every_known_mutating_route_is_decorated(_app_dir, _label, extra_routes=extra)
    print(f"\n[{_label}] exemplar_scan is not decorated")
    test_exemplar_scan_is_not_decorated(_app_dir, _label)
    print(f"\n[{_label}] the decorator marks dirty only on success")
    test_decorator_marks_dirty_only_on_success(_app_dir, _label)
    print(f"\n[{_label}] a real mutating request marks dirty end to end")
    test_a_mutating_route_through_the_test_client_marks_dirty(_app_dir, _label)
    print(f"\n[{_label}] POST /api/flush reports a conflict via the test client")
    test_force_flush_reports_conflict_via_test_client(_app_dir, _label)
    print(f"\n[{_label}] scrap/unscrap report a conflict via the test client")
    test_scrap_and_unscrap_report_conflict_via_test_client(_app_dir, _label)


print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
