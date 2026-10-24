#!/usr/bin/env python
"""Overlay labels, predictions.json geometry and manifest.json bookkeeping in
src/run_inference.py.

No-pytest house harness. No GPU and no model weights: every model result is a
fake object with the attribute shape ultralytics returns (boxes.cls, .conf,
.xyxy, .xyxyn; masks.xy, .xyn; orig_shape), and a fake model whose predict()
hands those results back. Every fixture lives in a temp folder removed in the
finally block.

Covers:
  test_detector_overlay_draws_a_label_on_every_box   a box-only result gets one
                        chip per instance, each carrying the class name and the
                        confidence to two decimals, and the box fill honours the
                        opacity value
  test_label_chip_stays_inside_the_image            instances at the top edge,
                        at the right edge, a very long name, and two chips that
                        would overlap
  test_mask_overlay_still_fills_polygon_and_labels_it  the segmentation path
                        keeps its fill, its chip, and the optional box outline
  test_predictions_include_detections_without_masks a detector's detections
                        are written with box corners
  test_polygon_predictions_are_unchanged            the polygon entry shape the
                        review-round seeder reads is pinned exactly
  test_manifest_records_skipped_and_thresholds      end-to-end with a fake model
  test_save_crops_flag_is_accepted_and_writes_no_folder
  test_codes_csv_expands_a_bare_code
  test_names_full_for_run_reads_the_dataset_beside_the_run   the split dataset
                        beside a training run names its class tokens
  test_dataset_names_match_the_evaluation_report_reader      the overlay and the
                        evaluation report read one data.yaml the same way
  test_the_dataset_walk_matches_the_model_lab_reader         this script and
                        _modellab/labruns.py find the same dataset folder on a
                        lab layout and on a training-project layout
  test_dataset_names_beat_the_codes_csv_on_the_chip_and_in_the_manifest
  adversarial: zero detections, a None class tensor, a class id absent from the
        names map, an empty names map, an image that cannot be opened, and a
        filename holding quotes and control characters

Run with:
  env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_inference_manifest.py
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

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import run_inference as RI  # noqa: E402

_fail = 0
_pass = 0

BASE_GREY = 128
IMAGE_SIDE = 256
FOOTER = RI.FOOTER_HEIGHT
NAMES = {
    0: "Acropora-cervicornis",
    1: "Diploria-labyrinthiformis",
    2: "Porites-astreoides",
}
TIMESTAMP = "2026-09-04 15:00 AST"
# One fish class token and its full name, as the staged fish dataset writes them:
# a token that is in no species code table and is not written as a binomial.
FISH_TOKEN = "EPST_Dark"
FISH_FULL_NAME = "Epinephelus striatus, Nassau grouper, dark phase"
FISH_TOKEN_2 = "KYSE_Normal"
FISH_FULL_NAME_2 = "Kyphosus sectatrix, Bermuda chub"


def check(cond, msg):
    """Record one assertion as PASS or FAIL."""
    global _fail, _pass
    if cond:
        _pass += 1
        print(f"PASS: {msg}")
    else:
        _fail += 1
        print(f"FAIL: {msg}")


# ---------------------------------------------------------------- fakes


class FakeTensor:
    """Mimics the slice of the torch tensor API run_inference touches."""

    def __init__(self, values):
        """Wrap values as a numpy array."""
        self._a = np.asarray(values)

    def cpu(self):
        """Return self, as a cpu tensor would."""
        return self

    def numpy(self):
        """The wrapped numpy array."""
        return self._a

    def __len__(self):
        """Number of elements."""
        return len(self._a)

    def __getitem__(self, i):
        """Element i."""
        return self._a[i]

    def __iter__(self):
        """Iterate the elements."""
        return iter(self._a)


class FakeBoxes:
    """ultralytics Boxes: xyxy, xyxyn, cls, conf, len."""

    def __init__(self, xyxy, cls, conf, shape):
        """Build xyxy, normalized xyxyn, and optional cls and conf tensors for an (h, w) image."""
        h, w = shape
        xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.xyxy = FakeTensor(xyxy)
        self.xyxyn = FakeTensor(xyxy / np.array([w, h, w, h], dtype=float))
        self.cls = None if cls is None else FakeTensor(cls)
        self.conf = None if conf is None else FakeTensor(conf)

    def __len__(self):
        """Number of boxes."""
        return len(self.xyxy)


class FakeMasks:
    """ultralytics Masks: xy (pixel polygons) and xyn (normalized)."""

    def __init__(self, polys_xy, shape):
        """Build pixel and normalized polygon lists for an (h, w) image."""
        h, w = shape
        self.xy = [np.asarray(p, dtype=float) for p in polys_xy]
        self.xyn = [p / np.array([w, h], dtype=float) for p in self.xy]


class FakeResult:
    """ultralytics Results: orig_shape (h, w), boxes, masks."""

    def __init__(self, shape, boxes=None, masks=None):
        """Hold the (h, w) shape, boxes and masks."""
        self.orig_shape = shape
        self.boxes = boxes
        self.masks = masks


class FakeModel:
    """YOLO stand-in: predict(path, **kw) returns the canned result for path."""

    def __init__(self, results_by_path, names):
        """Hold the canned results by path and the class-name map."""
        self.names = names
        self._results = results_by_path
        self.calls = []

    def predict(self, src, **kwargs):
        """Record the call and return the canned result for src."""
        self.calls.append((src, kwargs))
        return [self._results[src]]


def box_result(xyxy, cls, conf, shape=(IMAGE_SIDE, IMAGE_SIDE)):
    """A detector result: boxes only, masks None."""
    return FakeResult(shape, boxes=FakeBoxes(xyxy, cls, conf, shape), masks=None)


def mask_result(polys_xy, cls, conf, shape=(IMAGE_SIDE, IMAGE_SIDE)):
    """A segmentation result: boxes derived from each polygon's extent."""
    xyxy = []
    for poly in polys_xy:
        arr = np.asarray(poly, dtype=float)
        xyxy.append([arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()])
    return FakeResult(shape, boxes=FakeBoxes(xyxy, cls, conf, shape),
                      masks=FakeMasks(polys_xy, shape))


def write_grey_image(path, side=IMAGE_SIDE):
    """A flat mid-grey JPEG so any drawing is measurable against the base."""
    Image.new("RGB", (side, side), (BASE_GREY, BASE_GREY, BASE_GREY)).save(path, quality=95)
    return path


def region_mean(img, rect):
    """Mean brightness of an RGB region given as (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    arr = np.asarray(img.crop((x0, y0, x1, y1)).convert("L"), dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


def inside(rect, width, height):
    """True when the chip rectangle sits inside the image above the footer."""
    x0, y0, x1, y1 = rect
    return 0 <= x0 and 0 <= y0 and x1 <= width and y1 <= height - FOOTER


def rects_overlap(a, b):
    """True when two (x0, y0, x1, y1) rectangles share any area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def parse(argv):
    """Parse argv through run_inference's own arg_parse()."""
    old = sys.argv
    try:
        sys.argv = ["run_inference.py"] + argv
        return RI.arg_parse()
    finally:
        sys.argv = old


def base_argv(source_dir, out_dir, extra=()):
    """The argv the orchestrator sends for a directory source, plus any extra flags."""
    return ["--run_dir", "/unused/run", "--out_dir", out_dir, "--source_type", "dir",
            "--source_dir", source_dir, "--imgsz", "256", "--conf", "0.25", "--iou", "0.7",
            "--save_overlays", "--save_predictions", "--run_name", "t3",
            "--project_name", "proj"] + list(extra)


# ---------------------------------------------------------------- overlay


def test_detector_overlay_draws_a_label_on_every_box(root):
    """A box-only result gets one chip per box with the binomial and confidence, and the fill honours the opacity."""
    img = write_grey_image(os.path.join(root, "det.jpg"))
    boxes = [[20, 60, 120, 160], [140, 40, 240, 140], [30, 180, 90, 225]]
    res = box_result(boxes, [0, 1, 2], [0.913, 0.5, 0.257])
    out, info = RI.render_overlay(img, res, NAMES, fill_alpha=0.45, draw_boxes=False,
                                  run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(len(info["chips"]) == 3, "detector overlay: one label chip per box")
    check(info["labels"] == ["Acropora cervicornis 0.91", "Diploria labyrinthiformis 0.50",
                             "Porites astreoides 0.26"],
          f"detector overlay: labels carry the binomial with spaces and two-decimal confidence ({info['labels']})")
    for label, rect in zip(info["labels"], info["chips"]):
        check(region_mean(out, rect) < 90,
              f"detector overlay: chip for '{label}' has a dark backing (mean {region_mean(out, rect):.0f})")
    # the box interior is tinted at the fill opacity and the edge is outlined
    x0, y0, x1, y1 = boxes[0]
    centre = out.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
    check(centre != (BASE_GREY, BASE_GREY, BASE_GREY),
          f"detector overlay: box interior is tinted at fill opacity 0.45 (pixel {centre})")
    edge = out.getpixel((x0, (y0 + y1) // 2))
    check(edge != (BASE_GREY, BASE_GREY, BASE_GREY), f"detector overlay: box edge is outlined (pixel {edge})")
    check(info["instances_box_only"] == 3 and info["instances_with_mask"] == 0,
          "detector overlay: render info counts three box-only instances")
    # opacity zero leaves the interior untouched
    out0, _ = RI.render_overlay(img, res, NAMES, fill_alpha=0.0, draw_boxes=False,
                                run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(out0.getpixel(((x0 + x1) // 2, (y0 + y1) // 2)) == (BASE_GREY, BASE_GREY, BASE_GREY),
          "detector overlay: fill opacity 0 leaves the box interior untouched")


def test_label_chip_stays_inside_the_image(root):
    """Chips at the top edge, at the right edge, with an over-long name, and at a shared corner stay inside and apart."""
    img = write_grey_image(os.path.join(root, "edges.jpg"))
    W = H = IMAGE_SIDE
    long_name = "Pseudodiploria-clivosa-with-an-absurdly-long-synonym-" * 4
    names = {0: "Acropora-cervicornis", 1: "Diploria-labyrinthiformis", 2: long_name}
    boxes = [[10, 0, 90, 60],          # touches the top edge: chip cannot go above it
             [W - 30, 100, W, 150],    # touches the right edge: chip would run off
             [40, 120, 200, 200],      # very long name: must shrink, not overflow
             [10, 0, 100, 70]]         # same top-left as the first: chips would overlap
    res = box_result(boxes, [0, 1, 2, 0], [0.8, 0.7, 0.6, 0.5])
    out, info = RI.render_overlay(img, res, names, fill_alpha=0.3, draw_boxes=False,
                                  run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(len(info["chips"]) == 4, "edges: every instance still gets a chip")
    for label, rect in zip(info["labels"], info["chips"]):
        check(inside(rect, W, H), f"edges: chip for '{label[:30]}' sits inside the image {rect}")
    top_chip = info["chips"][0]
    check(top_chip[1] >= 0, "edges: chip at the top edge is moved inside the image")
    right_chip = info["chips"][1]
    check(right_chip[2] <= W, "edges: chip at the right edge is moved left inside the image")
    long_chip = info["chips"][2]
    check(long_chip[2] - long_chip[0] <= W, "edges: an over-long name shrinks to the image width")
    check(info["font_sizes"][2] >= RI.LABEL_FONT_SIZE_MIN,
          f"edges: the shrunk font never drops below {RI.LABEL_FONT_SIZE_MIN} px")
    check(info["labels"][2].endswith(RI.ELLIPSIS) and len(info["labels"][2]) < len(long_name),
          "edges: a name that cannot fit at the smallest font is elided, not overflowed")
    check(not rects_overlap(info["chips"][0], info["chips"][3]),
          f"edges: two chips anchored at the same corner do not overlap {info['chips'][0]} {info['chips'][3]}")
    check(info["chips_overlapping"] == 0, "edges: render info reports no overlapping chips")


def test_mask_overlay_still_fills_polygon_and_labels_it(root):
    """The segmentation path keeps its fill, its chip, and the optional box outline."""
    img = write_grey_image(os.path.join(root, "mask.jpg"))
    tri = [[60, 60], [200, 60], [130, 200]]
    res = mask_result([tri], [1], [0.77])
    out, info = RI.render_overlay(img, res, NAMES, fill_alpha=0.45, draw_boxes=False,
                                  run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info["labels"] == ["Diploria labyrinthiformis 0.77"], "mask overlay: polygon instance is labelled")
    check(out.getpixel((130, 100)) != (BASE_GREY, BASE_GREY, BASE_GREY), "mask overlay: polygon interior is filled")
    # without draw_boxes the box edge outside the triangle stays grey; with it, it is outlined
    corner = (61, 190)
    check(out.getpixel(corner) == (BASE_GREY, BASE_GREY, BASE_GREY),
          "mask overlay: no box outline when draw_boxes is off")
    out_b, info_b = RI.render_overlay(img, res, NAMES, fill_alpha=0.45, draw_boxes=True,
                                      run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(out_b.getpixel(corner) != (BASE_GREY, BASE_GREY, BASE_GREY),
          "mask overlay: draw_boxes adds the box outline around the polygon")
    check(info_b["instances_with_mask"] == 1 and info_b["draw_boxes_had_effect"] is True,
          "mask overlay: render info says draw_boxes had an effect")
    check(info["draw_boxes_had_effect"] is False, "mask overlay: render info says draw_boxes off had no effect")


# ---------------------------------------------------------------- predictions


def test_predictions_include_detections_without_masks():
    """A detector's detections are written with normalized box corners."""
    res = box_result([[0, 0, 128, 128], [64, 64, 256, 192]], [0, 2], [0.9, 0.31])
    stats = RI.new_detection_stats()
    items = RI.build_prediction_items([("a.jpg", "/abs/a.jpg", res)], NAMES, stats=stats)
    dets = items[0]["detections"]
    check(len(dets) == 2, "box predictions: both maskless detections are written")
    check(all("box_xyxyn" in d and "polygon_xyn" not in d for d in dets),
          "box predictions: entries carry box_xyxyn and no polygon field")
    check(dets[0]["box_xyxyn"] == [0.0, 0.0, 0.5, 0.5], f"box predictions: normalized corners ({dets[0]['box_xyxyn']})")
    check(dets[1] == {"class_id": 2, "class": "Porites-astreoides", "confidence": 0.31,
                      "box_xyxyn": [0.25, 0.25, 1.0, 0.75]},
          f"box predictions: entry carries class id, full class name, confidence and corners ({dets[1]})")
    check(RI.skipped_total(stats) == 0, "box predictions: nothing was skipped")


def test_polygon_predictions_are_unchanged():
    """The polygon entry shape the review-round seeder reads is pinned; a degenerate polygon falls back to its box."""
    poly = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    res = FakeResult((1080, 1920))
    res.boxes = FakeBoxes([[192, 108, 960, 540]], [0], [0.84], (1080, 1920))
    res.masks = FakeMasks([[[192, 108], [960, 108], [960, 540]]], (1080, 1920))
    items = RI.build_prediction_items([("a.jpg", "/abs/a.jpg", res)], {0: "AA"})
    check(set(items[0].keys()) == {"filename", "raw", "width", "height", "detections"},
          "polygon predictions: item keys are pinned")
    check(items[0]["width"] == 1920 and items[0]["height"] == 1080, "polygon predictions: (h, w) maps to width, height")
    d = items[0]["detections"][0]
    check(d == {"class_id": 0, "class": "AA", "confidence": 0.84, "polygon_xyn": [0.1, 0.1, 0.5, 0.1, 0.5, 0.5]},
          f"polygon predictions: entry shape the seeder reads is unchanged ({d})")
    check(list(d.keys()) == ["class_id", "class", "confidence", "polygon_xyn"],
          "polygon predictions: key order is unchanged")
    # a degenerate polygon (two points) falls back to the box and is counted, not dropped
    res2 = FakeResult((100, 200))
    res2.boxes = FakeBoxes([[0, 0, 100, 50]], [0], [0.6], (100, 200))
    res2.masks = FakeMasks([[[0, 0], [100, 0]]], (100, 200))
    stats = RI.new_detection_stats()
    items2 = RI.build_prediction_items([("b.jpg", "/abs/b.jpg", res2)], {0: "AA"}, stats=stats)
    d2 = items2[0]["detections"]
    check(len(d2) == 1 and "box_xyxyn" in d2[0] and "polygon_xyn" not in d2[0],
          "polygon predictions: a two-point polygon is written as its box instead")
    check(stats[RI.STAT_DEGENERATE_POLYGON] == 1 and RI.skipped_total(stats) == 0,
          "polygon predictions: the degenerate polygon is counted as downgraded, not skipped")


# ---------------------------------------------------------------- end to end


def _run_fake(root, results, names, extra_argv=(), code_lookup=None):
    """Write one grey image per result, run the engine with a fake model, return (manifest, out_dir)."""
    src = os.path.join(root, "src")
    out = os.path.join(root, "out")
    os.makedirs(src, exist_ok=True)
    by_path = {}
    for filename, res in results.items():
        path = os.path.join(src, filename)
        if res is not None:
            write_grey_image(path)
        else:
            with open(path, "wb") as f:
                f.write(b"not an image at all")
            res = box_result([[10, 10, 50, 50]], [0], [0.9])
        by_path[path] = res
    args = parse(base_argv(src, out, extra_argv))
    model = FakeModel(by_path, names)
    manifest = RI.run(args, model, code_lookup=code_lookup)
    return manifest, out


def test_manifest_records_skipped_and_thresholds(root):
    """End to end with a fake model: thresholds, skipped counts, render block, errors and predictions.json."""
    sub = os.path.join(root, "e2e")
    results = {
        "one.jpg": box_result([[10, 10, 100, 100]], [0], [0.9]),
        "two.jpg": mask_result([[[10, 10], [100, 10], [50, 90]]], [1], [0.6]),
    }
    # third image: a detection with no geometry at all must be skipped and counted
    empty_geom = FakeResult((IMAGE_SIDE, IMAGE_SIDE))
    empty_geom.boxes = FakeBoxes(np.zeros((0, 4)), [2], [0.4], (IMAGE_SIDE, IMAGE_SIDE))
    results["three.jpg"] = empty_geom
    manifest, out = _run_fake(sub, results, NAMES)
    check(manifest["conf_threshold"] == 0.25 and manifest["iou_threshold"] == 0.7 and manifest["imgsz"] == 256,
          "manifest: records the confidence threshold, the overlap threshold and the image size")
    check(manifest["detections_skipped"] == 1 and manifest["detections_skipped_reasons"] == {RI.STAT_NO_GEOMETRY: 1},
          f"manifest: one detection skipped for missing geometry ({manifest['detections_skipped_reasons']})")
    check(manifest["detections_box_only"] == 1 and manifest["detections_with_polygon"] == 1,
          "manifest: counts box-only and polygon detections")
    dets = {it["filename"]: it["detections"] for it in manifest["items"]}
    check(dets["one.jpg"] == [{"class": "Acropora-cervicornis", "class_id": 0, "confidence": 0.9}],
          f"manifest: per-detection rows carry class name, class id and confidence ({dets['one.jpg']})")
    render = manifest["render"]
    check(render["fill_alpha"] == 0.45 and render["draw_boxes"] is False
          and render["instances_box_only"] == 1 and render["instances_with_mask"] == 1
          and render["draw_boxes_had_effect"] is False,
          f"manifest: render block says what the opacity and box flags did ({render})")
    check(manifest["generated_at"].endswith("-04:00"), "manifest: generated_at carries the AST offset")
    check(manifest["errors"] == [], "manifest: no errors on a clean run")
    with open(os.path.join(out, "predictions.json")) as f:
        preds = json.load(f)
    check(preds["schema_version"] == 2 and preds["detections_skipped"] == 1,
          "predictions.json: schema 2 and the skipped count")
    pdets = {it["filename"]: it["detections"] for it in preds["items"]}
    check("box_xyxyn" in pdets["one.jpg"][0] and "polygon_xyn" in pdets["two.jpg"][0] and pdets["three.jpg"] == [],
          "predictions.json: box entry, polygon entry, and an empty list for the skipped one")
    for name in ("one.jpg", "two.jpg", "three.jpg"):
        check(os.path.isfile(os.path.join(out, "overlays", name)), f"manifest: overlay written for {name}")
    check(manifest["with_any"] == 3 and manifest["total_predictions"] == 3,
          "manifest: every reported instance counts, including the one with no geometry")


def test_save_crops_flag_is_accepted_and_writes_no_folder(root):
    """The compatibility flag parses, creates no folder, and is listed under ignored_flags."""
    sub = os.path.join(root, "crops")
    manifest, out = _run_fake(sub, {"a.jpg": box_result([[10, 10, 60, 60]], [0], [0.9])}, NAMES,
                              extra_argv=["--save_crops"])
    check(not os.path.exists(os.path.join(out, "crops")), "save_crops: no empty crops folder is created")
    check(manifest["ignored_flags"] == ["save_crops"], "save_crops: manifest records the flag as ignored")
    manifest2, _ = _run_fake(os.path.join(root, "nocrops"), {"a.jpg": box_result([[10, 10, 60, 60]], [0], [0.9])}, NAMES)
    check(manifest2["ignored_flags"] == [], "save_crops: nothing is ignored when the flag is absent")


def test_codes_csv_expands_a_bare_code(root):
    """A species code CSV turns a bare code into its full name on the chip and in the manifest."""
    csv_path = os.path.join(root, "codes.csv")
    with open(csv_path, "w") as f:
        f.write("code,category,name\nAA,Coral,Agaricia agaricites\n")
    lookup = RI.load_code_lookup(csv_path)
    check(lookup == {"AA": "Agaricia agaricites"}, "codes csv: loads code to full name")
    check(RI.display_name({0: "AA"}, 0, lookup) == "Agaricia agaricites (AA)", "codes csv: a bare code shows its full name")
    check(RI.display_name({0: "AA"}, 0, None) == "AA", "codes csv: without a lookup the class name is shown verbatim")
    check(RI.display_name({0: "Acropora-cervicornis"}, 0, lookup) == "Acropora cervicornis",
          "binomial rule: a dashed name that is not a code is shown with spaces")
    check(RI.display_name({0: "Porites_astreoides"}, 0, None) == "Porites astreoides",
          "binomial rule: an underscored name is shown with spaces")
    check(RI.class_label({0: "Acropora-cervicornis"}, 0) == "Acropora-cervicornis",
          "binomial rule: the raw label for the manifest and predictions keeps its dash")
    check(RI.display_name({0: "sand_patch"}, 0, None) == "sand_patch",
          "binomial rule: a token that is not a binomial (no capitalised genus) stays verbatim")
    missing = os.path.join(root, "missing.csv")
    try:
        RI.load_code_lookup(missing)
        check(False, "codes csv: a missing file raises")
    except FileNotFoundError as e:
        check(missing in str(e), "codes csv: the error names the missing file")
    # the manifest carries the resolved names beside the raw codes
    manifest, _ = _run_fake(os.path.join(root, "codes_e2e"), {"a.jpg": box_result([[10, 10, 60, 60]], [0], [0.9])},
                            {0: "AA"}, code_lookup=lookup)
    det = manifest["items"][0]["detections"][0]
    check(det["class"] == "AA" and det.get("class_full_name") == "Agaricia agaricites",
          f"codes csv: manifest detection keeps the code and adds the full name ({det})")


# ---------------------------------------------------------------- adversarial


def test_adversarial_zero_detections(root):
    """An empty result renders, writes an empty list, and skips nothing."""
    img = write_grey_image(os.path.join(root, "zero.jpg"))
    res = FakeResult((IMAGE_SIDE, IMAGE_SIDE), boxes=FakeBoxes(np.zeros((0, 4)), [], [], (IMAGE_SIDE, IMAGE_SIDE)))
    out, info = RI.render_overlay(img, res, NAMES, fill_alpha=0.45, draw_boxes=True,
                                  run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info["chips"] == [] and out.size == (IMAGE_SIDE, IMAGE_SIDE), "zero detections: overlay renders with no chips")
    items = RI.build_prediction_items([("zero.jpg", img, res)], NAMES)
    check(items[0]["detections"] == [], "zero detections: predictions item keeps an empty list")
    manifest, _ = _run_fake(os.path.join(root, "zero_e2e"), {"zero.jpg": res}, NAMES)
    check(manifest["with_any"] == 0 and manifest["total_predictions"] == 0 and manifest["detections_skipped"] == 0,
          "zero detections: manifest totals are zero and nothing is skipped")


def test_adversarial_none_class_tensor(root):
    """A None class tensor is labelled honestly; a None confidence is drawn but not written."""
    img = write_grey_image(os.path.join(root, "nocls.jpg"))
    res = box_result([[10, 10, 90, 90]], None, [0.55])
    out, info = RI.render_overlay(img, res, NAMES, fill_alpha=0.45, draw_boxes=False,
                                  run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info["labels"] == [f"{RI.UNKNOWN_CLASS_LABEL} 0.55"], f"None class: overlay labels the box honestly ({info['labels']})")
    stats = RI.new_detection_stats()
    items = RI.build_prediction_items([("nocls.jpg", img, res)], NAMES, stats=stats)
    d = items[0]["detections"]
    check(len(d) == 1 and d[0]["class_id"] is None and d[0]["class"] == RI.UNKNOWN_CLASS_LABEL,
          f"None class: prediction is written with a null class id ({d})")
    # no confidence tensor: drawn with a placeholder, not written to the seeder's file
    res2 = box_result([[10, 10, 90, 90]], [0], None)
    _, info2 = RI.render_overlay(img, res2, NAMES, fill_alpha=0.45, draw_boxes=False,
                                 run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info2["labels"] == [f"Acropora cervicornis {RI.NO_CONFIDENCE_LABEL}"],
          f"None confidence: overlay says the confidence is missing ({info2['labels']})")
    stats2 = RI.new_detection_stats()
    items2 = RI.build_prediction_items([("nocls.jpg", img, res2)], NAMES, stats=stats2)
    check(items2[0]["detections"] == [] and stats2[RI.STAT_NO_CONFIDENCE] == 1,
          "None confidence: skipped from predictions and counted")


def test_adversarial_class_id_absent_and_empty_names(root):
    """An id missing from the map, an empty map, and a list map all still label the box."""
    img = write_grey_image(os.path.join(root, "absent.jpg"))
    res = box_result([[10, 10, 90, 90]], [7], [0.6])
    _, info = RI.render_overlay(img, res, {0: "AA"}, fill_alpha=0.45, draw_boxes=False,
                                run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info["labels"] == ["unnamed class 7 0.60"], f"absent class id: label says the id is unnamed ({info['labels']})")
    items = RI.build_prediction_items([("absent.jpg", img, res)], {0: "AA"})
    check(items[0]["detections"][0]["class"] == "unnamed class 7" and items[0]["detections"][0]["class_id"] == 7,
          "absent class id: prediction keeps the numeric id")
    _, info_empty = RI.render_overlay(img, res, {}, fill_alpha=0.45, draw_boxes=False,
                                      run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(info_empty["labels"] == ["unnamed class 7 0.60"], "empty names map: rendering still labels the box")
    manifest, _ = _run_fake(os.path.join(root, "empty_names"), {"absent.jpg": res}, {})
    check(manifest["class_names"] == {} and manifest["items"][0]["detections"][0]["class"] == "unnamed class 7",
          "empty names map: end-to-end run completes and records the unnamed class")
    check(RI.display_name(["listed"], 0, None) == "listed", "names given as a list resolve by position")


def test_adversarial_unopenable_image(root):
    """An unreadable image raises with its path and is recorded under errors, not dropped."""
    bad = os.path.join(root, "bad.jpg")
    with open(bad, "wb") as f:
        f.write(b"definitely not a jpeg")
    res = box_result([[10, 10, 90, 90]], [0], [0.6])
    try:
        RI.render_overlay(bad, res, NAMES, fill_alpha=0.45, draw_boxes=False,
                          run_name="r", project_name="p", timestamp=TIMESTAMP)
        check(False, "unopenable image: render_overlay raises")
    except OSError as e:
        check(bad in str(e), f"unopenable image: the error names the file ({e})")
    manifest, out = _run_fake(os.path.join(root, "bad_e2e"),
                              {"good.jpg": box_result([[10, 10, 60, 60]], [0], [0.9]), "bad.jpg": None}, NAMES)
    check(len(manifest["items"]) == 1 and manifest["items"][0]["filename"] == "good.jpg",
          "unopenable image: the good image is still in the manifest")
    check(len(manifest["errors"]) == 1 and manifest["errors"][0]["filename"] == "bad.jpg"
          and manifest["errors"][0]["error"],
          f"unopenable image: the failure is recorded in the manifest, not dropped ({manifest['errors']})")
    check(os.path.isfile(os.path.join(out, "manifest.json")), "unopenable image: manifest.json is still written")


def test_adversarial_hostile_filename(root):
    """A filename with quotes and control characters round-trips through both JSON files and the overlay."""
    hostile = 'quote"d \x01ctrl\tname.jpg'
    res = box_result([[10, 10, 90, 90]], [0], [0.6])
    manifest, out = _run_fake(os.path.join(root, "hostile"), {hostile: res}, NAMES)
    check(manifest["items"][0]["filename"] == hostile, "hostile filename: manifest round-trips the exact name")
    with open(os.path.join(out, "manifest.json")) as f:
        reread = json.load(f)
    check(reread["items"][0]["filename"] == hostile, "hostile filename: manifest.json is valid JSON holding the name")
    with open(os.path.join(out, "predictions.json")) as f:
        preds = json.load(f)
    check(preds["items"][0]["filename"] == hostile, "hostile filename: predictions.json holds the name")
    check(os.path.isfile(os.path.join(out, "overlays", hostile)), "hostile filename: overlay written under the same name")
    check(manifest["errors"] == [], "hostile filename: no error recorded")


def _is_whole_part_suffix(text, parts):
    """True when text (after the cut marker) is the join of a trailing run of whole parts."""
    body = text[len(RI.FOOTER_TRUNCATION_PREFIX):] if text.startswith(RI.FOOTER_TRUNCATION_PREFIX) else text
    return any(body == RI.FOOTER_SEPARATOR.join(parts[k:]) for k in range(len(parts)))


def test_footer_text_fits_the_image_width():
    """The footer drops whole parts from the front, keeps the filename whole, and only cuts a lone part."""
    width = IMAGE_SIDE
    limit = width - 2 * RI.FOOTER_TEXT_X
    parts = ["VEGA", "t3_proof", "2026-09-04 19:30 AST", "Acropora-prolifera_348.jpg"]
    text, font = RI.fit_footer(parts, width)
    check(RI._text_width(font, text) <= limit,
          f"footer: a long line fits the image width ({RI._text_width(font, text)} <= {limit})")
    check(text == RI.FOOTER_TRUNCATION_PREFIX + "Acropora-prolifera_348.jpg",
          f"footer: whole parts drop from the front, the filename stays whole, the cut is marked ({text!r})")
    check(_is_whole_part_suffix(text, parts) and font.size >= RI.FOOTER_FONT_SIZE_MIN,
          f"footer: no part is cut mid-token at 256 px ({text!r}, {font.size} px)")
    wide, wide_font = RI.fit_footer(parts, 640)
    check(wide == RI.FOOTER_SEPARATOR.join(parts) and wide_font.size == RI.FOOTER_FONT_SIZE,
          f"footer: at 640 px the whole line fits at full size ({wide!r}, {wide_font.size} px)")
    long_parts = ["VEGA", "coral_intel_det_260902_20260904", "2026-09-04 19:06 AST", "Montastraea-cavernosa_1192.jpg"]
    long_text, long_font = RI.fit_footer(long_parts, width)
    check(long_text == RI.FOOTER_TRUNCATION_PREFIX + "Montastraea-cavernosa_1192.jpg"
          and RI._text_width(long_font, long_text) <= limit,
          f"footer: a 30-character filename survives whole at 256 px ({long_text!r}, {long_font.size} px)")
    short, short_font = RI.fit_footer(["p", "r", "a.jpg"], width)
    check(short == "p · r · a.jpg" and short_font.size == RI.FOOTER_FONT_SIZE,
          f"footer: a short line is untouched at full size ({short!r}, {short_font.size})")
    hostile, _ = RI.fit_footer(['bad\x01"name\t.jpg'], width)
    check("\x01" not in hostile and "\t" not in hostile and '"' in hostile,
          f"footer: control characters are replaced, quotes kept ({hostile!r})")
    huge, huge_font = RI.fit_footer(["x" * 400], width)
    check(RI._text_width(huge_font, huge) <= limit and huge.startswith(RI.FOOTER_TRUNCATION_PREFIX),
          "footer: a single 400-character part is the only case cut mid-token, and the cut is marked")


def test_help_text_promises_nothing_dead():
    """The source promises no crops folder and carries no banned dash or word."""
    src_path = os.path.join(_SRC, "run_inference.py")
    with open(src_path) as f:
        text = f.read()
    check("crops/" not in text, "source: the docstring no longer promises a crops folder")
    check("os.makedirs(crop_dir" not in text and "crop_dir" not in text, "source: no crops folder is created")
    for banned in ("\u2014", "\u2013"):
        check(banned not in text, f"source: no {banned!r} dash in the file")
    check(re.search(r"\bland(s|ed|ing)?\b", text, re.I) is None, "source: the banned word is absent")


def test_selected_frames_csv_must_be_readable(root):
    """A supplied selected-frames CSV must exist and be readable; an empty path means none."""
    check(RI._load_selected_basenames("") == set(), "selected frames: no path supplied gives an empty set")
    good = os.path.join(root, "selected_frames.csv")
    with open(good, "w") as f:
        f.write("basename,site\nTCRMP20250101_video_ABC_T1_frame_000001,ABC\n")
    check(RI._load_selected_basenames(good) == {"TCRMP20250101_video_ABC_T1_frame_000001"},
          "selected frames: a readable csv yields its basenames")
    missing = os.path.join(root, "gone.csv")
    try:
        RI._load_selected_basenames(missing)
        check(False, "selected frames: a supplied path that does not exist raises")
    except FileNotFoundError as e:
        check(missing in str(e), f"selected frames: the error names the missing file ({e})")
    if os.geteuid() != 0:
        locked = os.path.join(root, "locked.csv")
        with open(locked, "w") as f:
            f.write("basename\nx\n")
        os.chmod(locked, 0)
        try:
            RI._load_selected_basenames(locked)
            check(False, "selected frames: an unreadable csv raises instead of returning an empty set")
        except OSError as e:
            check(locked in str(e), f"selected frames: the error names the unreadable file ({e})")
        finally:
            os.chmod(locked, 0o644)


def test_chip_nudges_upward_at_the_footer_line(root):
    """Chips for boxes whose tops sit at the footer line move upward instead of overlapping."""
    img = write_grey_image(os.path.join(root, "footer_line.jpg"))
    H = IMAGE_SIDE
    top = H - FOOTER - 5
    boxes = [[10, top, 80, H], [12, top, 90, H], [14, top, 100, H]]
    res = box_result(boxes, [0, 0, 0], [0.9, 0.8, 0.7])
    _, info = RI.render_overlay(img, res, NAMES, fill_alpha=0.3, draw_boxes=False,
                                run_name="r", project_name="p", timestamp=TIMESTAMP)
    chips = info["chips"]
    check(len(chips) == 3, "footer line: every box at the footer line gets a chip")
    pairs = [(a, b) for i, a in enumerate(chips) for b in chips[i + 1:]]
    check(all(not rects_overlap(a, b) for a, b in pairs), f"footer line: no two chips overlap {chips}")
    check(all(inside(c, IMAGE_SIDE, H) for c in chips), f"footer line: every chip stays above the footer {chips}")
    check(min(c[1] for c in chips) < chips[0][1], "footer line: later chips moved upward, not into the footer")
    check(info["chips_overlapping"] == 0, "footer line: render info reports no overlapping chips")
    exact = [[10, H - FOOTER, 80, H], [12, H - FOOTER, 90, H]]
    _, info2 = RI.render_overlay(img, box_result(exact, [0, 0], [0.9, 0.8]), NAMES, fill_alpha=0.3,
                                 draw_boxes=False, run_name="r", project_name="p", timestamp=TIMESTAMP)
    check(not rects_overlap(info2["chips"][0], info2["chips"][1]) and info2["chips_overlapping"] == 0,
          f"footer line: chips for boxes whose top is exactly the footer line do not overlap {info2['chips']}")


def test_malformed_names_and_masks_are_counted(root):
    """Non-integer class-map keys are listed and an unreadable masks object is counted as a box fallback."""
    names, dropped = RI.normalise_class_names({0: "AA", "x": "bad key", "1": "BB", None: "none"})
    check(names == {0: "AA", 1: "BB"} and sorted(map(str, dropped)) == ["None", "x"],
          f"class map: non-integer keys are reported, not dropped silently ({dropped})")
    manifest, _ = _run_fake(os.path.join(root, "bad_names"), {"a.jpg": box_result([[10, 10, 60, 60]], [0], [0.9])},
                            {0: "AA", "x": "bad key"})
    check(manifest["class_names_dropped"] == ["x"], f"class map: the manifest lists the dropped keys ({manifest['class_names_dropped']})")
    res = FakeResult((100, 200))
    res.boxes = FakeBoxes([[0, 0, 100, 50]], [0], [0.6], (100, 200))
    res.masks = FakeMasks([], (100, 200))
    res.masks.xy = [[1, 2, 3]]
    res.masks.xyn = [[0.1, 0.2, 0.3]]
    stats = RI.new_detection_stats()
    items = RI.build_prediction_items([("m.jpg", "/abs/m.jpg", res)], {0: "AA"}, stats=stats)
    d = items[0]["detections"]
    check(len(d) == 1 and "box_xyxyn" in d[0], "malformed masks: the instance is still written as its box")
    check(stats[RI.STAT_DEGENERATE_POLYGON] == 1, "malformed masks: the fallback is counted, not silent")
    res.masks.xy = object()
    res.masks.xyn = object()
    stats2 = RI.new_detection_stats()
    items2 = RI.build_prediction_items([("m.jpg", "/abs/m.jpg", res)], {0: "AA"}, stats=stats2)
    check(len(items2[0]["detections"]) == 1 and stats2[RI.STAT_DEGENERATE_POLYGON] == 1,
          "malformed masks: a non-iterable masks object is counted the same way")


def _lab_run_dir(root, data_yaml_text):
    """A lab layout (<lab>/runs/<run>/ beside <lab>/dataset/) and the run folder inside it.

    Parameters:
        root: the folder the lab is created in.
        data_yaml_text: the dataset's data.yaml body, or None for a lab with no dataset.
    Returns:
        The run folder path.
    """
    lab = os.path.join(root, "lab")
    run_dir = os.path.join(lab, "runs", "run_a")
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    if data_yaml_text is not None:
        os.makedirs(os.path.join(lab, "dataset"), exist_ok=True)
        with open(os.path.join(lab, "dataset", "data.yaml"), "w", encoding="utf-8") as fh:
            fh.write(data_yaml_text)
    return run_dir


def test_names_full_for_run_reads_the_dataset_beside_the_run(root):
    """The split dataset beside a run names its classes; a run without one, or without names_full, names none."""
    sub = os.path.join(root, "names_full")
    named = _lab_run_dir(os.path.join(sub, "named"),
                         f"path: .\nnames:\n  0: {FISH_TOKEN}\n  1: {FISH_TOKEN_2}\n"
                         f"names_full:\n  0: {FISH_FULL_NAME}\n  1: {FISH_FULL_NAME_2}\n")
    check(RI.names_full_for_run(named) == {FISH_TOKEN: FISH_FULL_NAME, FISH_TOKEN_2: FISH_FULL_NAME_2},
          f"dataset names: the run's dataset names both classes ({RI.names_full_for_run(named)})")
    plain = _lab_run_dir(os.path.join(sub, "plain"), f"path: .\nnames:\n  0: {FISH_TOKEN}\n")
    check(RI.names_full_for_run(plain) == {},
          "dataset names: a dataset without names_full names nothing")
    bare = _lab_run_dir(os.path.join(sub, "bare"), None)
    check(RI.names_full_for_run(bare) == {},
          "dataset names: a run with no dataset beside it names nothing")
    check(RI.names_full_for_run("") == {} and RI.names_full_for_run(None) == {},
          "dataset names: a blank run folder names nothing")
    broken = _lab_run_dir(os.path.join(sub, "broken"), "- this is a list, not a mapping\n")
    check(RI.names_full_for_run(broken) == {},
          "dataset names: a damaged data.yaml names nothing instead of stopping the inference run")
    listed = _lab_run_dir(os.path.join(sub, "listed"),
                          f"path: .\nnames:\n  - {FISH_TOKEN}\nnames_full:\n  - {FISH_FULL_NAME}\n")
    check(RI.names_full_for_run(listed) == {FISH_TOKEN: FISH_FULL_NAME},
          "dataset names: a class map written as a list is read in id order")


def test_dataset_names_match_the_evaluation_report_reader(root):
    """run_inference and evaluate_run read the same data.yaml into the same names, so the overlay and the report agree."""
    import evaluate_run as ER
    body = (f"path: .\nnames:\n  0: {FISH_TOKEN}\n  1: {FISH_TOKEN_2}\n  2: Acropora-cervicornis\n"
            f"names_full:\n  0: {FISH_FULL_NAME}\n  1: '  '\n  3: Ignored id\n")
    run_dir = _lab_run_dir(os.path.join(root, "twin"), body)
    data_yaml = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "dataset", "data.yaml")
    check(RI.names_full_for_run(run_dir) == ER.names_full_from_data_yaml(data_yaml),
          f"the two readers agree: {RI.names_full_for_run(run_dir)} vs {ER.names_full_from_data_yaml(data_yaml)}")
    check(RI.names_full_for_run(run_dir) == {FISH_TOKEN: FISH_FULL_NAME},
          "a blank name and a name for an id the class map does not carry are both left out")


def test_the_dataset_walk_matches_the_model_lab_reader(root):
    """run_inference._dataset_dir_for_run and _modellab/labruns.dataset_for_run find the same dataset on a lab and on a project layout."""
    import importlib.util
    labruns_path = os.path.join(_HERE, "..", "..", "_modellab", "labruns.py")
    spec = importlib.util.spec_from_file_location("_t6_labruns", os.path.abspath(labruns_path))
    labruns = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labruns)
    sub = os.path.join(root, "walk")
    body = f"path: .\nnames:\n  0: {FISH_TOKEN}\n"
    cases = []
    # A lab: <lab>/runs/<run> with <lab>/dataset beside it.
    cases.append(("lab", _lab_run_dir(os.path.join(sub, "lab_case"), body)))
    # A training project: <project>/step6_trainModel/runs/<run> with the dataset beside it.
    step6 = os.path.join(sub, "project", "step6_trainModel")
    os.makedirs(os.path.join(step6, "runs", "run_a", "weights"), exist_ok=True)
    os.makedirs(os.path.join(step6, "dataset"), exist_ok=True)
    with open(os.path.join(step6, "dataset", "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(body)
    cases.append(("project", os.path.join(step6, "runs", "run_a")))
    # A run with no dataset beside it, a trailing separator, and nothing at all.
    cases.append(("no dataset", _lab_run_dir(os.path.join(sub, "bare_case"), None)))
    cases.append(("trailing separator", _lab_run_dir(os.path.join(sub, "slash_case"), body) + os.sep))
    cases.append(("blank", ""))
    for label, run_dir in cases:
        mine = RI._dataset_dir_for_run(run_dir)
        theirs = labruns.dataset_for_run(run_dir)
        check(mine == theirs, f"{label}: the two walks disagree ({mine!r} vs {theirs!r})")
    check(RI._dataset_dir_for_run(cases[0][1]).endswith(os.path.join("lab", "dataset")),
          f"the lab case really found a dataset: {RI._dataset_dir_for_run(cases[0][1])}")
    check(RI._dataset_dir_for_run(cases[2][1]) == "",
          "a run with no dataset beside it finds none, in both readers")


def test_dataset_names_beat_the_codes_csv_on_the_chip_and_in_the_manifest(root):
    """The run's dataset overrides the species code CSV, and the overlay chip and the manifest carry the winning name."""
    sub = os.path.join(root, "merge")
    os.makedirs(sub, exist_ok=True)
    codes_csv = os.path.join(sub, "codes.csv")
    with open(codes_csv, "w", encoding="utf-8") as fh:
        fh.write(f"code,category,name\n{FISH_TOKEN},Fish,An older name for the code\nAA,Coral,Agaricia agaricites\n")
    run_dir = _lab_run_dir(sub, f"path: .\nnames:\n  0: {FISH_TOKEN}\nnames_full:\n  0: {FISH_FULL_NAME}\n")
    merged = RI.build_code_lookup(codes_csv, run_dir)
    check(merged[FISH_TOKEN] == FISH_FULL_NAME,
          f"merge: the run's dataset wins over the codes csv ({merged[FISH_TOKEN]})")
    check(merged["AA"] == "Agaricia agaricites",
          "merge: a code only the csv knows is kept")
    check(RI.display_name({0: FISH_TOKEN}, 0, merged) == f"{FISH_FULL_NAME} ({FISH_TOKEN})",
          f"merge: the overlay chip reads the full name with the code beside it ({RI.display_name({0: FISH_TOKEN}, 0, merged)})")
    csv_only = RI.build_code_lookup(codes_csv, _lab_run_dir(os.path.join(sub, "no_dataset"), None))
    check(csv_only[FISH_TOKEN] == "An older name for the code",
          "merge: without a dataset the codes csv still names the class")
    dataset_only = RI.build_code_lookup("", run_dir)
    check(dataset_only == {FISH_TOKEN: FISH_FULL_NAME},
          f"merge: the dataset alone names the class ({dataset_only})")
    check(RI.build_code_lookup("", _lab_run_dir(os.path.join(sub, "neither"), None)) is None,
          "merge: with neither source the lookup is None, so every label is shown as it is")
    manifest, _ = _run_fake(os.path.join(sub, "e2e"),
                            {"a.jpg": box_result([[10, 10, 60, 60]], [0], [0.9])},
                            {0: FISH_TOKEN}, code_lookup=merged)
    det = manifest["items"][0]["detections"][0]
    check(det["class"] == FISH_TOKEN and det.get("class_full_name") == FISH_FULL_NAME,
          f"merge: the manifest keeps the token and adds the full name ({det})")
    check(manifest["class_full_names"] == {FISH_TOKEN: FISH_FULL_NAME},
          f"merge: the manifest's top-level names map carries the fish name ({manifest['class_full_names']})")


def main():
    """Run every case in one temp folder, remove it, and exit nonzero on any failure."""
    root = tempfile.mkdtemp(prefix="t3_inference_manifest_")
    try:
        test_detector_overlay_draws_a_label_on_every_box(root)
        test_label_chip_stays_inside_the_image(root)
        test_mask_overlay_still_fills_polygon_and_labels_it(root)
        test_predictions_include_detections_without_masks()
        test_polygon_predictions_are_unchanged()
        test_manifest_records_skipped_and_thresholds(root)
        test_save_crops_flag_is_accepted_and_writes_no_folder(root)
        test_codes_csv_expands_a_bare_code(root)
        test_adversarial_zero_detections(root)
        test_adversarial_none_class_tensor(root)
        test_adversarial_class_id_absent_and_empty_names(root)
        test_adversarial_unopenable_image(root)
        test_adversarial_hostile_filename(root)
        test_footer_text_fits_the_image_width()
        test_help_text_promises_nothing_dead()
        test_selected_frames_csv_must_be_readable(root)
        test_chip_nudges_upward_at_the_footer_line(root)
        test_malformed_names_and_masks_are_counted(root)
        test_names_full_for_run_reads_the_dataset_beside_the_run(root)
        test_dataset_names_match_the_evaluation_report_reader(root)
        test_the_dataset_walk_matches_the_model_lab_reader(root)
        test_dataset_names_beat_the_codes_csv_on_the_chip_and_in_the_manifest(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{_pass} passed, {_fail} failed")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
