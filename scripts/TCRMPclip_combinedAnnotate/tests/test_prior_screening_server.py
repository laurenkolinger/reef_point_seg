"""Cross-project no-rescreening data path (VEGA annotator tidy-up plan,
docs/superpowers/plans/2026-09-27-vega-annotator-tidy-and-no-rescreening.md,
task W3): app.py loads <export_dir>/prior_screening.json at configure() and
resume() into session['prior_screening'] (load_prior_screening);
get_image_data() turns a frame's entry into prior_masks (scaled to THIS
frame's own pixel dimensions, since the source polygons are normalized 0..1)
and prior_screened (a pass-through of the screened labels); and
_stamp_provenance() feeds the screened labels to
provenance.compute_label_outcomes()'s new screened_elsewhere argument so a
target species another project already screened this image for never gets a
fresh not_found row here.

Covers, in order:
  1. load_prior_screening(): a valid file, a missing file, malformed JSON, a
     wrong version (both a wrong number and a wrong type), a non-dict top
     level, a non-dict 'frames', per-frame wrong types (screened not a dict,
     masks not a list, an entry that is not a dict at all, individual mask
     entries that are not dicts), and hostile strings/keys (script-like
     content, very long strings, a stem with path-traversal-looking
     characters and an embedded null byte) -- none of this is ever used to
     build a filesystem path, so it is inert as data.
  2. get_image_data(): prior_masks scaled correctly at 1920x1081 and at
     3840x2160, a malformed polygon dropped outright, a mask with one good
     ring and one bad ring surviving on the good ring alone, prior_screened
     passed through verbatim, and prior masks never appended to
     seg['masks'] (so there is nothing for the undo stack to ever see).
  3. compute_label_outcomes(screened_elsewhere=...): backward compatible
     (omitted / None / empty all behave exactly as before), a screened
     label with nothing found locally is left out of the result instead of
     getting not_found, a screened label WITH a local accepted mask still
     reports found_manual, an unscreened target code is unaffected, and a
     pending_expert outcome for a screened code is unaffected.
  4. _stamp_provenance() end to end through PUT /api/image/<f>/masks: the
     same three cases (excluded, found-despite-screened, ordinary not_found)
     wired through the real route and the real session state.
  5. do_export() end to end: a prior-only species with no local mask never
     becomes a class in class_map, a manifest column, or a row in the
     cross-project mask registry.
  6. The loader and a plain read of session['prior_screening'] never touch
     the segmentations dirty counter.

No pytest, GPU-free: CUDA_VISIBLE_DEVICES is forced empty below, and nothing
here imports sam_engine or instantiates SAM3Engine.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_prior_screening_server.py
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_SCRIPTS = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _SRC)
sys.path.insert(0, _SCRIPTS)

from PIL import Image  # noqa: E402

import app as A  # noqa: E402
import provenance as P  # noqa: E402

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def _captured_log():
    """Context-free log capture: swap A.log for a list-appending stub and
    hand back the list plus a restore function."""
    captured = []
    original = A.log
    A.log = lambda msg: captured.append(msg)

    def restore():
        A.log = original

    return captured, restore


# ── 1. load_prior_screening() ───────────────────────────────────────

def test_loader_valid_file_no_log():
    d = tempfile.mkdtemp(prefix="prior_valid_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"), {
            "version": 1, "generated_at": "2026-09-27T00:00:00-04:00",
            "project_id": "current",
            "frames": {"IMG_A": {
                "screened": {"PA": {"outcome": "found_manual", "project_id": "p1",
                                     "project_name": "proj one", "reviewer": "AT",
                                     "at": "2026-09-25T00:00:00-04:00"}},
                "masks": [{"label": "PA", "name": "Porites astreoides",
                           "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}],
            }},
        })
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(logged == [], f"a valid file logs nothing (got {logged})")
        check(set(result) == {"IMG_A"}, f"valid file: one frame loaded (got {set(result)})")
        check(result["IMG_A"]["screened"]["PA"]["outcome"] == "found_manual",
              "valid file: screened block carried through")
        check(len(result["IMG_A"]["masks"]) == 1, "valid file: masks list carried through")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_missing_file():
    d = tempfile.mkdtemp(prefix="prior_missing_")
    try:
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(result == {}, "missing file -> {}")
        check(len(logged) == 1, f"missing file logs exactly one line (got {logged})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_malformed_json():
    d = tempfile.mkdtemp(prefix="prior_badjson_")
    try:
        path = os.path.join(d, "prior_screening.json")
        with open(path, "w") as f:
            f.write("{not valid json,,, [[[")
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(result == {}, "malformed JSON -> {}")
        check(len(logged) == 1, f"malformed JSON logs exactly one line (got {logged})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_wrong_version_number():
    d = tempfile.mkdtemp(prefix="prior_badver_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"),
                    {"version": 2, "frames": {}})
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(result == {}, "version 2 -> {}")
        check(len(logged) == 1, "wrong version number logs exactly one line")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_version_wrong_type():
    d = tempfile.mkdtemp(prefix="prior_verstr_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"),
                    {"version": "1", "frames": {}})
        result = A.load_prior_screening(d)
        check(result == {}, "version as a string '1' (not int 1) -> {}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_toplevel_not_a_dict():
    d = tempfile.mkdtemp(prefix="prior_badtop_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"), [1, 2, 3])
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(result == {}, "top level is a list, not a dict -> {}")
        check(len(logged) == 1, "non-dict top level logs exactly one line")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_frames_not_a_dict():
    d = tempfile.mkdtemp(prefix="prior_badframes_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"),
                    {"version": 1, "frames": "not-a-dict"})
        result = A.load_prior_screening(d)
        check(result == {}, "'frames' not a dict -> {}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_per_frame_wrong_types():
    d = tempfile.mkdtemp(prefix="prior_frametypes_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"), {
            "version": 1,
            "frames": {
                "OK_FRAME": {"screened": {"PA": {"outcome": "found_manual"}},
                             "masks": [{"label": "PA"}]},
                "BAD_SCREENED": {"screened": ["not", "a", "dict"], "masks": []},
                "BAD_MASKS": {"screened": {}, "masks": "not-a-list"},
                "BAD_ENTRY": "not-a-dict-at-all",
                "MIXED_MASKS": {"screened": {}, "masks": [{"label": "PA"}, "garbage", 42, None]},
            },
        })
        logged, restore = _captured_log()
        result = A.load_prior_screening(d)
        restore()
        check(logged == [], f"a per-frame defect alone logs nothing (got {logged})")
        check(set(result) == {"OK_FRAME", "BAD_SCREENED", "BAD_MASKS", "MIXED_MASKS"},
              f"BAD_ENTRY (not a dict) dropped, the other frames kept (got {set(result)})")
        check(result["OK_FRAME"]["screened"] == {"PA": {"outcome": "found_manual"}},
              "well-formed frame's screened block kept as-is")
        check(result["BAD_SCREENED"]["screened"] == {},
              f"screened of the wrong type sanitized to {{}} (got {result['BAD_SCREENED']['screened']})")
        check(result["BAD_MASKS"]["masks"] == [],
              f"masks of the wrong type sanitized to [] (got {result['BAD_MASKS']['masks']})")
        check(result["MIXED_MASKS"]["masks"] == [{"label": "PA"}],
              f"non-dict entries filtered out of a masks list (got {result['MIXED_MASKS']['masks']})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_loader_hostile_strings_and_keys():
    d = tempfile.mkdtemp(prefix="prior_hostile_")
    try:
        hostile_text = "<script>alert(1)</script>\"; DROP TABLE prior--" + ("z" * 5000)
        hostile_stem = "../../../etc/passwd\x00trailer"
        _write_json(os.path.join(d, "prior_screening.json"), {
            "version": 1,
            "frames": {
                hostile_stem: {
                    "screened": {"PA": {"outcome": "found_manual",
                                         "project_name": hostile_text}},
                    "masks": [{"label": "PA", "name": hostile_text,
                               "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}],
                },
            },
        })
        result = A.load_prior_screening(d)
        check(len(result) == 1, f"a hostile stem/string file still loads (got {len(result)} frames)")
        check(hostile_stem in result, "the hostile stem is kept verbatim as a dict key, never touched")
        check(result[hostile_stem]["screened"]["PA"]["project_name"] == hostile_text,
              "a hostile string value is carried through unmodified, never executed or truncated")
        check(result[hostile_stem]["masks"][0]["name"] == hostile_text,
              "the same hostile string in a mask's name field is also carried through unmodified")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── 2. get_image_data(): scaling, dropping, prior_screened, undo surface ──

def test_prior_masks_scaling_and_prior_screened():
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="prior_scale_")
    try:
        img_a = os.path.join(d, "FRAME_A.jpg")
        img_b = os.path.join(d, "FRAME_B.jpg")
        Image.new("RGB", (1920, 1081), (10, 20, 30)).save(img_a, "JPEG")
        Image.new("RGB", (3840, 2160), (10, 20, 30)).save(img_b, "JPEG")

        ring_a = [0.1, 0.13, 0.44, 0.13, 0.44, 0.37]
        ring_b = [0.21, 0.33, 0.61, 0.33, 0.61, 0.62]
        long_name = "Injected\"); DROP TABLE prior--<script>" + ("z" * 300)

        A.session["export_dir"] = os.path.join(d, "out")
        A.session["segmentations"] = {
            "FRAME_A.jpg": {
                "image_path": "FRAME_A.jpg", "image_path_abs": img_a,
                "image_width": 1920, "image_height": 1081,
                "masks": [], "reference_points": [],
                "reviewed": False, "exported": False,
            },
            "FRAME_B.jpg": {
                "image_path": "FRAME_B.jpg", "image_path_abs": img_b,
                "image_width": 3840, "image_height": 2160,
                "masks": [], "reference_points": [],
                "reviewed": False, "exported": False,
            },
        }
        A.session["class_map"] = {}
        A.session["sam_engine"] = None
        A.session["prior_screening"] = {
            "FRAME_A": {
                "screened": {"PA": {"outcome": "found_manual", "project_id": "otherA",
                                     "project_name": long_name, "reviewer": "AT",
                                     "at": "2026-09-25T00:00:00-04:00"}},
                "masks": [
                    {"label": "PA", "name": "Porites astreoides",
                     "polygon_norm": [ring_a], "project_id": "otherA",
                     "project_name": long_name, "reviewer": "AT",
                     "at": "2026-09-25T00:00:00-04:00", "outcome": "found_manual"},
                    # polygon_norm not a list at all: the whole mask is dropped.
                    {"label": "OFRA", "name": "Orbicella franksi",
                     "polygon_norm": "garbage", "project_id": "otherA"},
                    # One malformed (odd-length) ring plus one valid ring: the
                    # mask survives with only the valid ring.
                    {"label": "MCAV", "name": "Montastraea cavernosa",
                     "polygon_norm": [[0.2, 0.2, 0.9],
                                       [0.05, 0.05, 0.15, 0.05, 0.15, 0.15]],
                     "project_id": "otherA"},
                ],
            },
            "FRAME_B": {
                "screened": {},
                "masks": [
                    {"label": "PA", "name": "Porites astreoides",
                     "polygon_norm": [ring_b], "project_id": "otherB",
                     "project_name": "otherproj", "reviewer": "BT",
                     "at": "2026-09-20T00:00:00-04:00", "outcome": "found_ai"},
                ],
            },
        }

        with A.app.test_client() as c:
            r = c.get("/api/image/FRAME_A.jpg")
            check(r.status_code == 200, f"GET FRAME_A -> 200 (got {r.status_code})")
            body = r.get_json()

            expected_ring_a = [round(v * (1920 if i % 2 == 0 else 1081))
                               for i, v in enumerate(ring_a)]
            pm = body.get("prior_masks", [])
            by_label = {m["label"]: m for m in pm}
            check(len(pm) == 2,
                  f"FRAME_A: PA and MCAV survive, OFRA (bad polygon_norm) is dropped (got {len(pm)})")
            check("OFRA" not in by_label, "OFRA (polygon_norm not a list) dropped entirely")
            check("PA" in by_label, "PA prior mask present")
            if "PA" in by_label:
                check(by_label["PA"]["polygon_px"] == [expected_ring_a],
                      f"PA ring scaled to 1920x1081 (got {by_label['PA']['polygon_px']}, "
                      f"want {[expected_ring_a]})")
                check(all(isinstance(v, int) for v in by_label["PA"]["polygon_px"][0]),
                      "scaled coordinates are ints, not floats")
                check(by_label["PA"]["name"] == "Porites astreoides",
                      "full species name carried through for the hover tooltip")
                check(by_label["PA"]["project_name"] == long_name,
                      "a hostile/long project_name is passed through verbatim")
            check("MCAV" in by_label, "MCAV prior mask survives on its one valid ring")
            if "MCAV" in by_label:
                check(len(by_label["MCAV"]["polygon_px"]) == 1,
                      f"MCAV keeps only its one valid ring (got "
                      f"{len(by_label['MCAV']['polygon_px'])} rings)")

            check(body.get("prior_screened") == {
                "PA": {"outcome": "found_manual", "project_id": "otherA",
                       "project_name": long_name, "reviewer": "AT",
                       "at": "2026-09-25T00:00:00-04:00"}},
                  f"prior_screened passes the screened dict through verbatim "
                  f"(got {body.get('prior_screened')})")

            check(A.session["segmentations"]["FRAME_A.jpg"]["masks"] == [],
                  "prior masks are never appended to seg['masks'], so there is "
                  "nothing for the undo stack to ever act on")

            r2 = c.get("/api/image/FRAME_B.jpg")
            check(r2.status_code == 200, f"GET FRAME_B -> 200 (got {r2.status_code})")
            body2 = r2.get_json()
            expected_ring_b = [round(v * (3840 if i % 2 == 0 else 2160))
                               for i, v in enumerate(ring_b)]
            pm2 = body2.get("prior_masks", [])
            check(len(pm2) == 1, f"FRAME_B: 1 prior mask (got {len(pm2)})")
            if pm2:
                check(pm2[0]["polygon_px"] == [expected_ring_b],
                      f"FRAME_B ring scaled to 3840x2160 (got {pm2[0]['polygon_px']}, "
                      f"want {[expected_ring_b]})")
            check(body2.get("prior_screened") == {},
                  "FRAME_B has an empty screened dict (not screened for anything elsewhere)")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_get_image_data_with_no_prior_screening_loaded():
    """The deployment gap the plan calls out: W2's orchestrator half ships
    later, so most projects will have no prior_screening.json for a while.
    session['prior_screening'] simply absent must behave exactly like an
    empty one: no crash, empty prior_masks/prior_screened."""
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="prior_absent_")
    try:
        img = os.path.join(d, "SOLO.jpg")
        Image.new("RGB", (100, 80), (1, 2, 3)).save(img, "JPEG")
        A.session["export_dir"] = os.path.join(d, "out")
        A.session["segmentations"] = {
            "SOLO.jpg": {"image_path": "SOLO.jpg", "image_path_abs": img,
                         "image_width": 100, "image_height": 80,
                         "masks": [], "reference_points": [],
                         "reviewed": False, "exported": False},
        }
        A.session["class_map"] = {}
        A.session["sam_engine"] = None
        A.session.pop("prior_screening", None)

        with A.app.test_client() as c:
            r = c.get("/api/image/SOLO.jpg")
            check(r.status_code == 200, f"GET SOLO -> 200 (got {r.status_code})")
            body = r.get_json()
            check(body.get("prior_masks") == [], "no prior_screening loaded -> prior_masks == []")
            check(body.get("prior_screened") == {}, "no prior_screening loaded -> prior_screened == {}")
    finally:
        A.session["prior_screening"] = {}
        shutil.rmtree(d, ignore_errors=True)


# ── 3. compute_label_outcomes(screened_elsewhere=...) unit tests ────────

def test_compute_label_outcomes_screened_elsewhere():
    seg_empty = {"masks": []}

    out_default = P.compute_label_outcomes(seg_empty, ["PA"], "step5")
    check(out_default.get("PA", {}).get("outcome") == "not_found",
          "no screened_elsewhere argument at all: unaffected (backward compatible)")

    out_none = P.compute_label_outcomes(seg_empty, ["PA"], "step5", screened_elsewhere=None)
    check(out_none.get("PA", {}).get("outcome") == "not_found",
          "screened_elsewhere=None: unaffected")

    out_empty = P.compute_label_outcomes(seg_empty, ["PA"], "step5", screened_elsewhere=set())
    check(out_empty.get("PA", {}).get("outcome") == "not_found",
          "screened_elsewhere=set(): unaffected")

    out_screened = P.compute_label_outcomes(seg_empty, ["PA"], "step5", screened_elsewhere={"PA"})
    check("PA" not in out_screened,
          f"PA screened elsewhere with no local mask -> no PA key at all, not not_found "
          f"(got {out_screened})")

    seg_found = {"masks": [{"species": "PA", "status": "accepted", "source_type": "manual_click"}]}
    out_found = P.compute_label_outcomes(seg_found, ["PA"], "step5", screened_elsewhere={"PA"})
    check(out_found.get("PA", {}).get("outcome") == "found_manual",
          f"PA screened elsewhere but found locally -> found_manual still recorded "
          f"(got {out_found.get('PA')})")

    out_partial = P.compute_label_outcomes({"masks": []}, ["PA", "OFRA"], "step5",
                                            screened_elsewhere={"PA"})
    check("PA" not in out_partial, "PA excluded (screened elsewhere)")
    check(out_partial.get("OFRA", {}).get("outcome") == "not_found",
          f"OFRA (a target NOT screened elsewhere) still gets not_found as usual "
          f"(got {out_partial})")

    pm = {"species": "REVIEW", "status": "accepted", "source_type": "manual_click", "review": True,
          "reviews": [{"reviewer": "AB", "code": "PA", "confidence": "", "at": P.ast_now()}]}
    out_pending = P.compute_label_outcomes({"masks": [pm]}, ["PA"], "step5",
                                            screened_elsewhere={"PA"})
    check(out_pending.get("PA", {}).get("outcome") == "pending_expert",
          f"a pending_expert outcome for a screened-elsewhere code is unaffected "
          f"(got {out_pending.get('PA')})")


# ── 4. _stamp_provenance() end to end via PUT /api/image/<f>/masks ──────

def test_stamp_provenance_excludes_screened_elsewhere():
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="prior_stamp_")
    try:
        img = os.path.join(d, "img.jpg")
        Image.new("RGB", (64, 48), (1, 2, 3)).save(img, "JPEG")

        seg_screened_no_mask = {
            "image_path": "SCREENED_EMPTY.jpg", "image_path_abs": img,
            "image_width": 64, "image_height": 48, "masks": [],
            "reference_points": [], "reviewed": False, "exported": False,
        }
        seg_screened_with_mask = {
            "image_path": "SCREENED_FOUND.jpg", "image_path_abs": img,
            "image_width": 64, "image_height": 48,
            "masks": [{"id": 0, "status": "accepted", "species": "PA",
                       "source_type": "manual_click", "review": False}],
            "reference_points": [], "reviewed": False, "exported": False,
        }
        seg_unscreened = {
            "image_path": "UNSCREENED.jpg", "image_path_abs": img,
            "image_width": 64, "image_height": 48, "masks": [],
            "reference_points": [], "reviewed": False, "exported": False,
        }

        A.session["export_dir"] = d
        A.session["segmentations"] = {
            "SCREENED_EMPTY.jpg": seg_screened_no_mask,
            "SCREENED_FOUND.jpg": seg_screened_with_mask,
            "UNSCREENED.jpg": seg_unscreened,
        }
        A.session["class_map"] = {}
        A.session["reviewer"] = "LO"
        A.session["sam_engine"] = None
        A.session["prior_screening"] = {
            "SCREENED_EMPTY": {
                "screened": {"PA": {"outcome": "found_manual", "project_id": "otherA",
                                     "project_name": "otherrun", "reviewer": "AT",
                                     "at": "2026-09-25T00:00:00-04:00"}},
                "masks": [],
            },
            "SCREENED_FOUND": {
                "screened": {"PA": {"outcome": "found_manual", "project_id": "otherA",
                                     "project_name": "otherrun", "reviewer": "AT",
                                     "at": "2026-09-25T00:00:00-04:00"}},
                "masks": [],
            },
            # UNSCREENED intentionally absent: fully unscreened elsewhere.
        }

        with A.app.test_client() as c:
            for fn in ("SCREENED_EMPTY.jpg", "SCREENED_FOUND.jpg", "UNSCREENED.jpg"):
                r = c.put(f"/api/image/{fn}/masks", json={"updates": {}, "relabel": {}})
                check(r.status_code == 200, f"PUT masks on {fn} -> 200 (got {r.status_code})")

        outcomes_empty = A.session["segmentations"]["SCREENED_EMPTY.jpg"]["label_outcomes"]
        check("PA" not in outcomes_empty,
              f"screened elsewhere + no local mask -> no PA row in label_outcomes "
              f"(got {outcomes_empty})")

        outcomes_found = A.session["segmentations"]["SCREENED_FOUND.jpg"]["label_outcomes"]
        check(outcomes_found.get("PA", {}).get("outcome") == "found_manual",
              f"screened elsewhere BUT found locally -> found_manual still recorded "
              f"(got {outcomes_found.get('PA')})")

        outcomes_unscreened = A.session["segmentations"]["UNSCREENED.jpg"]["label_outcomes"]
        check(outcomes_unscreened.get("PA", {}).get("outcome") == "not_found",
              f"not screened anywhere -> ordinary not_found, unaffected by the exclusion "
              f"mechanism (got {outcomes_unscreened.get('PA')})")

        prov_path = os.path.join(d, "label_provenance.csv")
        check(os.path.exists(prov_path), "label_provenance.csv written")
        with open(prov_path, newline="") as f:
            rows = list(csv.DictReader(f))
        screened_empty_pa = [r for r in rows
                              if r["basename"] == "SCREENED_EMPTY" and r["label"] == "PA"]
        check(screened_empty_pa == [],
              f"no label_provenance.csv row for the excluded (basename, label) pair "
              f"(found {screened_empty_pa})")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── 5. do_export(): prior-only masks never exported or registered ──────

def test_prior_masks_never_exported_or_registered():
    A.app.config["TESTING"] = True
    d = tempfile.mkdtemp(prefix="prior_export_")
    reg_dir = tempfile.mkdtemp(prefix="prior_export_reg_")
    try:
        export_dir = os.path.join(d, "run_20260101_priortest", "step4test_combinedAnnotate")
        os.makedirs(export_dir, exist_ok=True)
        fn = "TCRMP20201022_clip_SCP_T301.jpeg"
        img_path = os.path.join(d, fn)
        Image.new("RGB", (64, 48), (5, 6, 7)).save(img_path, "JPEG")

        os.environ["TCRMP_MASK_REGISTRY_DIR"] = reg_dir
        os.environ["TCRMP_PROJECT_ID"] = "20260101_priortest"

        seg = {
            "image_path": fn, "image_path_abs": img_path,
            "image_width": 64, "image_height": 48,
            "masks": [{
                "id": 0, "status": "accepted", "species": "PA", "category": "Coral",
                "source_type": "manual_click", "source_x": 10, "source_y": 10,
                "review": False, "reviewer": "LO",
                "rle": {"counts": "abc", "size": [48, 64]},
                "polygon_norm": [[0.1, 0.1, 0.3, 0.1, 0.3, 0.3]],
            }],
            "reference_points": [], "reviewed": False, "exported": False,
        }
        A.session["export_dir"] = export_dir
        A.session["input_dir"] = export_dir
        A.session["segmentations"] = {fn: seg}
        A.session["class_map"] = {}
        A.session["review_batch_size"] = 100
        A.session["shuffle"] = False
        A.session["contacts"] = []
        A.session["reviewer"] = "LO"
        A.session["sam_engine"] = None
        # A prior-only species (no local mask at all): if this ever leaked
        # into export, OFRA would show up in class_map/the manifest/the
        # registry, none of which it is entitled to.
        A.session["prior_screening"] = {
            os.path.splitext(fn)[0]: {
                "screened": {},
                "masks": [{"label": "OFRA", "name": "Orbicella franksi",
                           "polygon_norm": [[0.5, 0.5, 0.9, 0.5, 0.9, 0.9]],
                           "project_id": "otherproj", "project_name": "otherrun",
                           "reviewer": "AT", "at": "2026-09-20T00:00:00-04:00",
                           "outcome": "found_manual"}],
            }
        }
        A._build_review_list()

        with A.app.test_client() as c:
            r = c.post("/api/export_batch")
            check(r.status_code == 200,
                  f"export_batch -> 200 (got {r.status_code}, body={r.data!r})")

        check("OFRA" not in A.session["class_map"],
              f"the prior-only species never becomes a YOLO class (class_map={A.session['class_map']})")
        check("PA" in A.session["class_map"],
              f"the real local species IS a YOLO class (class_map={A.session['class_map']})")

        manifest_path = os.path.join(export_dir, "export_manifest.csv")
        check(os.path.exists(manifest_path), "export_manifest.csv written")
        with open(manifest_path) as f:
            rows = list(csv.DictReader(f))
        check(len(rows) == 1, f"exactly 1 manifest row (got {len(rows)})")
        if rows:
            check("OFRA" not in rows[0],
                  f"OFRA (prior-only species) never becomes a manifest column (got {list(rows[0])})")
            check(rows[0].get("PA") == "1", f"PA (the real local mask) counted (got {rows[0].get('PA')})")

        prov_path = os.path.join(export_dir, "label_provenance.csv")
        check(os.path.exists(prov_path), "label_provenance.csv written at export")
        with open(prov_path, newline="") as f:
            prov_rows = list(csv.DictReader(f))
        check(all(r.get("label") != "OFRA" for r in prov_rows),
              f"OFRA never gets a label_provenance.csv row (rows={prov_rows})")

        from _reefreview.mask_registry import MaskRegistry
        reg = MaskRegistry(root=reg_dir)
        reg_rows = reg.rows()
        check(any(r.get("species") == "PA" for r in reg_rows),
              f"the real local PA mask reached the cross-project registry (rows={reg_rows})")
        check(all(r.get("species") != "OFRA" for r in reg_rows),
              f"the prior-only OFRA preview never reaches the registry (rows={reg_rows})")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(reg_dir, ignore_errors=True)
        os.environ.pop("TCRMP_MASK_REGISTRY_DIR", None)
        os.environ.pop("TCRMP_PROJECT_ID", None)


# ── 6. Never touches the dirty counter ──────────────────────────────────

def test_prior_screening_never_marks_dirty():
    d = tempfile.mkdtemp(prefix="prior_dirty_")
    try:
        _write_json(os.path.join(d, "prior_screening.json"),
                    {"version": 1, "frames": {"F": {"screened": {}, "masks": []}}})
        before = A._segmentations_dirty_counter
        A.load_prior_screening(d)
        after = A._segmentations_dirty_counter
        check(before == after,
              f"load_prior_screening() never calls _mark_dirty (counter {before} -> {after})")

        A.session["prior_screening"] = {"F": {"screened": {}, "masks": []}}
        before2 = A._segmentations_dirty_counter
        A._prior_screening_for("F.jpg")
        after2 = A._segmentations_dirty_counter
        check(before2 == after2,
              "_prior_screening_for (a pure read) never marks dirty")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── 0. configure() and resume() actually call the loader ────────────────

def test_configure_and_resume_load_prior_screening():
    A.app.config["TESTING"] = True
    root = tempfile.mkdtemp(prefix="prior_wiring_")
    try:
        run1 = os.path.join(root, "run_1")
        exp1 = os.path.join(run1, "step4test_combinedAnnotate")
        routed = os.path.join(exp1, "routed_input")
        os.makedirs(os.path.join(routed, "ids"), exist_ok=True)
        os.makedirs(os.path.join(routed, "raw"), exist_ok=True)
        Image.new("RGB", (320, 240), (10, 20, 30)).save(
            os.path.join(routed, "raw", "F1.jpg"), "JPEG")
        with open(os.path.join(routed, "ids", "sam_click_prompts.json"), "w") as f:
            json.dump({"F1.jpg": {"raw_image": "raw/F1.jpg",
                                   "points": [{"x": 10, "y": 10, "label": "1",
                                               "species": "PA", "point_type": 1}]}}, f)
        os.makedirs(exp1, exist_ok=True)
        _write_json(os.path.join(exp1, "prior_screening.json"), {
            "version": 1, "generated_at": "2026-09-27T00:00:00-04:00",
            "project_id": "run_1",
            "frames": {"F1": {
                "screened": {"PA": {"outcome": "found_ai", "project_id": "otherA",
                                     "project_name": "other run", "reviewer": "AT",
                                     "at": "2026-09-25T00:00:00-04:00"}},
                "masks": [],
            }},
        })
        with A.app.test_client() as c:
            r = c.post("/api/configure", json={"input_dir": routed, "export_dir": exp1})
            check(r.status_code == 200, f"configure -> 200 (got {r.status_code}, body={r.data!r})")
        check("F1" in A.session.get("prior_screening", {}),
              f"configure() loaded prior_screening.json into the session "
              f"(got keys {list(A.session.get('prior_screening', {}))})")
        if "F1" in A.session.get("prior_screening", {}):
            check(A.session["prior_screening"]["F1"]["screened"]["PA"]["outcome"] == "found_ai",
                  "configure()-loaded prior_screening carries the screened outcome")

        run2 = os.path.join(root, "run_2")
        exp2 = os.path.join(run2, "step4test_combinedAnnotate")
        seg_dir2 = os.path.join(exp2, "segmentations")
        os.makedirs(seg_dir2, exist_ok=True)
        _write_json(os.path.join(seg_dir2, "segmentations.json"), {
            "F2.jpg": {"image_path": "F2.jpg", "image_path_abs": "", "image_width": 10,
                       "image_height": 10, "masks": [], "reference_points": [],
                       "reviewed": False, "exported": False}})
        _write_json(os.path.join(exp2, "prior_screening.json"), {
            "version": 1,
            "frames": {"F2": {"screened": {}, "masks": [
                {"label": "PA", "name": "Porites astreoides",
                 "polygon_norm": [[0.1, 0.1, 0.2, 0.1, 0.2, 0.2]]}]}}})
        with A.app.test_client() as c:
            r = c.post("/api/resume", json={"export_dir": exp2})
            check(r.status_code == 200, f"resume -> 200 (got {r.status_code}, body={r.data!r})")
        check("F2" in A.session.get("prior_screening", {}),
              f"resume() loaded prior_screening.json into the session "
              f"(got keys {list(A.session.get('prior_screening', {}))})")
        if "F2" in A.session.get("prior_screening", {}):
            check(len(A.session["prior_screening"]["F2"]["masks"]) == 1,
                  "resume()-loaded prior_screening carries the masks list")
    finally:
        shutil.rmtree(root, ignore_errors=True)


_TESTS = [
    ("loader: valid file, no log line", test_loader_valid_file_no_log),
    ("loader: missing file", test_loader_missing_file),
    ("loader: malformed JSON", test_loader_malformed_json),
    ("loader: wrong version number", test_loader_wrong_version_number),
    ("loader: version wrong type", test_loader_version_wrong_type),
    ("loader: top level not a dict", test_loader_toplevel_not_a_dict),
    ("loader: 'frames' not a dict", test_loader_frames_not_a_dict),
    ("loader: per-frame wrong types", test_loader_per_frame_wrong_types),
    ("loader: hostile strings and keys", test_loader_hostile_strings_and_keys),
    ("configure()/resume() load prior_screening", test_configure_and_resume_load_prior_screening),
    ("get_image_data: scaling + prior_screened + drop malformed", test_prior_masks_scaling_and_prior_screened),
    ("get_image_data: no prior_screening loaded", test_get_image_data_with_no_prior_screening_loaded),
    ("compute_label_outcomes: screened_elsewhere", test_compute_label_outcomes_screened_elsewhere),
    ("_stamp_provenance: excludes screened_elsewhere", test_stamp_provenance_excludes_screened_elsewhere),
    ("do_export: prior masks never exported or registered", test_prior_masks_never_exported_or_registered),
    ("dirty counter untouched", test_prior_screening_never_marks_dirty),
]

for _label, _fn in _TESTS:
    print(f"\n[{_label}]")
    _fn()

print("\nPASS" if _fail == 0 else f"\n{_fail} FAILED")
sys.exit(1 if _fail else 0)
