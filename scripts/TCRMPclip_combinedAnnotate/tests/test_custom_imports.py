# CA/tests/test_custom_imports.py
"""Custom image imports (TODO T20260815-203910, second half).

Covers:
  - custom_imports.ingest_files: small import delivered at scale 1.0; oversize
    import downscaled into the lores mirror (long edge 1920, same rule as
    pipeline frames) with the correct self-reported scale; EXIF orientation
    baked into the delivered pixels; unsupported extension + missing path
    refused with reasons; delivered-name dedupe; manifest persistence.
  - custom_imports.seg_dict_for: shape-compatible with the annotator's
    MANUAL_ANNOTATE seg_dicts, custom_import flag + provenance fields set.
  - custom_imports.record_in_project: access_log + custom_imports appended to
    the run's project.json (AST timestamps); standalone (no project.json)
    returns False without raising.
  - POST /api/import_images end to end on the live Flask app: multipart
    upload and JSON-paths modes, review queue membership, custom_import flag
    in /api/review_batch + /api/image/<fn>, /images/<fn> serving, flush into
    segmentations.json, project.json recording, routed_input/ untouched, and
    no export artifacts created outside the normal export path.
  - Reconfigure resilience: a fresh /api/configure keeps the import; with
    segmentations.json wiped, the manifest backfills it (_merge_custom_imports).
  - Structural: template ships the tooltipped Import control + Custom badge
    and the handleImportFiles client function.

No pytest, GPU-free. Run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_custom_imports.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
_TPL = os.path.join(_SRC, "templates", "index.html")
sys.path.insert(0, _SRC)

import app as A
import custom_imports as ci

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _jpg(path, w, h, color=(40, 80, 120), exif_orientation=None):
    im = Image.new("RGB", (w, h), color)
    kw = {}
    if exif_orientation is not None:
        ex = Image.Exif()
        ex[274] = exif_orientation
        kw["exif"] = ex
    im.save(path, "JPEG", quality=90, **kw)
    return path


_root = tempfile.mkdtemp(prefix="custom_imports_")
try:
    # ── A. Pure module: ingest_files / seg_dict_for / record_in_project ──────
    print("[A] custom_imports module")
    export_dir = os.path.join(_root, "run_x", "step4test_combinedAnnotate")
    os.makedirs(export_dir, exist_ok=True)

    small = _jpg(os.path.join(_root, "small pic.jpg"), 640, 480)
    big = _jpg(os.path.join(_root, "big4k.jpg"), 4000, 2250)

    entries, skipped = ci.ingest_files(
        export_dir,
        [("small pic.jpg", small), ("big4k.jpg", big),
         ("notes.txt", small), ("ghost.jpg", os.path.join(_root, "nope.jpg"))],
        reviewer="LO", log=lambda m: None)
    check(len(entries) == 2, f"2 of 4 items ingested (got {len(entries)})")
    check(len(skipped) == 2 and {s[0] for s in skipped} == {"notes.txt", "ghost.jpg"},
          f"txt + missing path refused with reasons (got {skipped!r})")

    e_small = next(e for e in entries if "small" in e["filename"])
    e_big = next(e for e in entries if "big4k" in e["filename"])

    check(e_small["scale"] == 1.0 and e_small["width"] == 640 and e_small["height"] == 480,
          "small import delivered at scale 1.0 with dims kept")
    check(e_small["filename"].startswith("custom_") and not e_small["filename"].endswith("_lores.jpg"),
          f"small delivered name custom_-prefixed, no _lores suffix ({e_small['filename']})")
    check(e_big["filename"].endswith("_lores.jpg"),
          f"oversize delivered name carries _lores suffix ({e_big['filename']})")
    check(e_big["width"] == 1920 and e_big["height"] == 1080,
          f"4000x2250 downscaled to 1920x1080 (got {e_big['width']}x{e_big['height']})")
    check(abs(e_big["scale"] - 1920.0 / 4000.0) < 1e-9,
          f"scale == 1920/4000 == 0.48 (got {e_big['scale']})")
    dp = os.path.join(export_dir, e_big["delivered_file"])
    with Image.open(dp) as im:
        check(im.size == (1920, 1080), f"delivered file on disk is 1920x1080 (got {im.size})")
    # Coordinate scaling rule: a point at (x, y) on the delivered image maps
    # back to the original at (x/scale, y/scale). Spot-check the far corner.
    check(abs(1920 / e_big["scale"] - 4000) < 1e-6 and abs(1080 / e_big["scale"] - 2250) < 1e-6,
          "delivered corner maps back to the original corner via 1/scale")
    check(os.path.isfile(os.path.join(export_dir, e_small["original_file"])),
          "original preserved verbatim under custom_imports/originals/")

    # EXIF orientation 6 (rotate 90): delivered pixels must be transposed.
    rot = _jpg(os.path.join(_root, "rot.jpg"), 200, 100, exif_orientation=6)
    ent_rot, _ = ci.ingest_files(export_dir, [("rot.jpg", rot)], log=lambda m: None)
    check(ent_rot and ent_rot[0]["width"] == 100 and ent_rot[0]["height"] == 200,
          f"EXIF orientation baked in (200x100 or=6 -> 100x200, got "
          f"{ent_rot[0]['width']}x{ent_rot[0]['height']})" if ent_rot else "EXIF ingest produced no entry")

    # Dedupe: same source name again -> a _2 delivered name, no overwrite.
    again, _ = ci.ingest_files(export_dir, [("small pic.jpg", small)], log=lambda m: None)
    check(again and again[0]["filename"] != e_small["filename"]
          and again[0]["filename"].startswith("custom_"),
          f"re-import of the same name deduped ({again[0]['filename'] if again else 'none'})")

    manifest = ci.load_manifest(export_dir)
    check(len(manifest) == 4, f"manifest holds all 4 ingested entries (got {len(manifest)})")
    check(all("AST" in e.get("imported_at", "") for e in manifest),
          "manifest timestamps are AST-stamped")

    seg = ci.seg_dict_for(export_dir, e_big)
    check(seg["custom_import"] is True and seg["masks"] == [] and seg["reference_points"] == [],
          "seg_dict_for: custom flag set, maskless, no reference points")
    check(seg["image_path_abs"] == os.path.join(export_dir, e_big["delivered_file"])
          and os.path.isfile(seg["image_path_abs"]),
          "seg_dict_for: image_path_abs resolves to the delivered file")
    check(seg["image_width"] == 1920 and seg["image_height"] == 1080
          and seg["import_scale"] == e_big["scale"],
          "seg_dict_for: delivered dims + import scale recorded")

    # record_in_project: run root project.json gets access_log + custom_imports.
    pj_path = os.path.join(_root, "run_x", "project.json")
    with open(pj_path, "w") as f:
        json.dump({"id": "run_x", "name": "x", "steps": {}, "access_log": []}, f)
    ok = ci.record_in_project(export_dir, [e_big, e_small], initials="LO")
    check(ok is True, "record_in_project True when project.json exists")
    with open(pj_path) as f:
        pj = json.load(f)
    al = pj.get("access_log", [])
    check(len(al) == 1 and al[0]["action"] == "import_images" and al[0]["initials"] == "LO",
          f"access_log entry appended, action=import_images, LO (got {al!r})")
    check(al and al[0]["ts"].endswith("-04:00"), f"access_log ts carries the AST offset ({al[0]['ts'] if al else '?'})")
    check(len(pj.get("custom_imports", [])) == 2
          and pj["custom_imports"][0]["filename"] == e_big["filename"],
          "custom_imports list appended to project.json")

    no_pj_dir = os.path.join(_root, "loose_export")
    os.makedirs(no_pj_dir, exist_ok=True)
    check(ci.record_in_project(no_pj_dir, [e_big]) is False,
          "record_in_project False (no raise) without a project.json")

    # ── B. Flask app end to end ──────────────────────────────────────────────
    print("[B] POST /api/import_images on the live app")
    A.app.config["TESTING"] = True

    # No active session -> 400.
    A.session["export_dir"] = ""
    with A.app.test_client() as c:
        r = c.post("/api/import_images", json={"paths": [small]})
        check(r.status_code == 400, f"import before configure -> 400 (got {r.status_code})")

    # Minimal run: routed_input with ONE prompt frame + raw image.
    run2 = os.path.join(_root, "run_26-01-01_LO_seed")
    exp2 = os.path.join(run2, "step4test_combinedAnnotate")
    routed = os.path.join(exp2, "routed_input")
    os.makedirs(os.path.join(routed, "ids"), exist_ok=True)
    os.makedirs(os.path.join(routed, "raw"), exist_ok=True)
    _jpg(os.path.join(routed, "raw", "FRAME_R.jpg"), 320, 240)
    with open(os.path.join(routed, "ids", "sam_click_prompts.json"), "w") as f:
        json.dump({"FRAME_R.jpg": {"raw_image": "raw/FRAME_R.jpg",
                                   "points": [{"x": 10, "y": 12, "label": "A",
                                               "species": "PA", "point_type": 1}]}}, f)
    with open(os.path.join(run2, "project.json"), "w") as f:
        json.dump({"id": "run_26-01-01_LO_seed", "name": "seed", "steps": {},
                   "access_log": []}, f)

    with A.app.test_client() as c:
        r = c.post("/api/configure", json={"input_dir": routed, "export_dir": exp2})
        check(r.status_code == 200, f"configure on the seed run -> 200 (got {r.status_code})")

        # Multipart upload of an oversize image.
        buf = io.BytesIO()
        Image.new("RGB", (3840, 2160), (5, 90, 60)).save(buf, "JPEG")
        buf.seek(0)
        r = c.post("/api/import_images", data={
            "images": (buf, "field photo.jpg"),
        }, content_type="multipart/form-data")
        check(r.status_code == 200, f"multipart import -> 200 (got {r.status_code})")
        body = json.loads(r.data)
        check(body.get("ok") is True and len(body.get("imported", [])) == 1,
              f"one file imported (got {body!r})")
        custom_fn = body["imported"][0]
        check(custom_fn.startswith("custom_") and custom_fn.endswith("_lores.jpg"),
              f"oversize upload delivered as custom_*_lores.jpg ({custom_fn})")
        check(body.get("project_recorded") is True, "import recorded in project.json")

        # Review queue lists routed frame + import, custom flag only on the import.
        r = c.get("/api/review_batch")
        imgs = {i["filename"]: i for i in json.loads(r.data)["images"]}
        check("FRAME_R.jpg" in imgs and custom_fn in imgs,
              f"review batch lists routed + imported frames (got {sorted(imgs)})")
        check(imgs.get(custom_fn, {}).get("custom_import") is True
              and imgs.get("FRAME_R.jpg", {}).get("custom_import") is False,
              "custom_import flag True only on the import")

        r = c.get(f"/api/image/{custom_fn}")
        d = json.loads(r.data)
        check(d.get("custom_import") is True and d["image_width"] == 1920
              and d["image_height"] == 1080,
              f"/api/image reports custom flag + delivered lores dims (got "
              f"{d.get('image_width')}x{d.get('image_height')})")

        r = c.get(f"/images/{custom_fn}")
        check(r.status_code == 200 and len(r.data) > 1000,
              f"/images/<fn> serves the delivered file (got {r.status_code})")

        # JSON-paths mode.
        r = c.post("/api/import_images", json={"paths": [small]})
        body2 = json.loads(r.data)
        check(r.status_code == 200 and body2.get("ok") is True,
              f"JSON-paths import -> ok (got {r.status_code} {body2!r})")

    # Flushed to disk; project.json carries both imports; routing untouched.
    with open(os.path.join(exp2, "segmentations", "segmentations.json")) as f:
        store = json.load(f)
    check(custom_fn in store and store[custom_fn].get("custom_import") is True,
          "import flushed into segmentations.json with the custom flag")
    with open(os.path.join(run2, "project.json")) as f:
        pj2 = json.load(f)
    check(len(pj2.get("access_log", [])) == 2 and len(pj2.get("custom_imports", [])) == 2,
          f"project.json: 2 access_log entries + 2 custom_imports rows "
          f"(got {len(pj2.get('access_log', []))}/{len(pj2.get('custom_imports', []))})")
    check(sorted(os.listdir(os.path.join(routed, "raw"))) == ["FRAME_R.jpg"],
          "routed_input/raw untouched by imports")
    check(not os.path.isdir(os.path.join(exp2, "all_images"))
          and not os.path.isfile(os.path.join(exp2, "export_manifest.csv")),
          "no export artifacts created outside the normal export path")

    # ── C. Reconfigure resilience + manifest backfill ────────────────────────
    print("[C] reconfigure + backfill")
    with A.app.test_client() as c:
        r = c.post("/api/configure", json={"input_dir": routed, "export_dir": exp2})
        check(r.status_code == 200, "re-configure -> 200")
        r = c.get("/api/review_batch")
        imgs = {i["filename"] for i in json.loads(r.data)["images"]}
        check(custom_fn in imgs, "import survives a fresh configure (store reload)")

    os.remove(os.path.join(exp2, "segmentations", "segmentations.json"))
    with A.app.test_client() as c:
        r = c.post("/api/configure", json={"input_dir": routed, "export_dir": exp2})
        r = c.get("/api/review_batch")
        imgs = {i["filename"] for i in json.loads(r.data)["images"]}
        check(custom_fn in imgs,
              "manifest backfills the import after a segmentations store wipe")
        r = c.get("/api/status")
        st = json.loads(r.data)
        check(st.get("stats", {}).get("custom_imports", 0) == 2,
              f"/api/status counts 2 custom imports (got {st.get('stats', {}).get('custom_imports')})")

    # ── D. Structural: template ships the control + badge, tooltipped ────────
    print("[D] template structure")
    tpl = open(_TPL).read()
    check('id="btn-import-images"' in tpl, "template has #btn-import-images")
    btn_at = tpl.index('id="btn-import-images"')
    check("title=" in tpl[btn_at:btn_at + 700], "Import button carries a title= tooltip")
    check('id="import-file-input"' in tpl and 'accept=".jpg,.jpeg,.png"' in tpl,
          "hidden file input present, jpg/png only")
    check('id="custom-import-badge"' in tpl, "template has the Custom badge span")
    badge_at = tpl.index('id="custom-import-badge"')
    check("title=" in tpl[badge_at:badge_at + 700], "Custom badge carries a title= tooltip")
    check("function handleImportFiles" in tpl and "/api/import_images" in tpl,
          "template defines handleImportFiles() posting to /api/import_images")

finally:
    shutil.rmtree(_root, ignore_errors=True)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
