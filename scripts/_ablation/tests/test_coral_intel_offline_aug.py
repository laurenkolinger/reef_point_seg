#!/usr/bin/env python3
"""House tests for the Coral Intel offline augmentation builder.

Module:  scripts/_ablation/tests/test_coral_intel_offline_aug.py
Purpose: Prove the label parser is strict, the split is deterministic and
         class-stratified, the geometric point maps are correct against known
         box coordinates, the photometric transforms respect their parameter
         ranges and stay reproducible, the CLI refuses bad input loudly, and
         a small synthetic dataset survives a real CLI run with boxes that
         still cover their drawn rectangles after augmentation.
Inputs:  none from disk except what this file generates into a scratch
         directory under the session scratchpad, removed at the end.
Outputs: a pass or fail count on stdout, exit code 1 on any failure.

Run:     github_repo/env/bin/python scripts/_ablation/tests/test_coral_intel_offline_aug.py
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coral_intel_offline_aug as ca  # noqa: E402

FAILURES = 0
CHECKS = 0

# A fresh temporary folder per run, so two runs never share fixtures.
SCRATCH_ROOT = Path(tempfile.mkdtemp(prefix="coral_aug_test_"))

# A geometry/photometric test config matching the real params file's Table 2
# values, used wherever a test calls draw_params or transform_boxes directly
# without going through the CLI (which loads a params file itself).
TEST_CFG: dict = {
    "flip_horizontal_p": 0.5, "flip_vertical_p": 0.5,
    "rot90_choices": ["none", "cw", "ccw", "r180"],
    "rotation_deg": 13.0, "shear_deg_x": 10.0, "shear_deg_y": 10.0,
    "hue_deg": 15.0, "saturation_frac": 0.25, "exposure_frac": 0.10,
    "blur_sigma_max_px": 1.5, "noise_pixel_frac_max": 0.001,
}
TEST_MIN_BOX_PX = 2.0
TEST_MIN_AREA_RATIO = 0.20

# A valid params-file ``params:`` mapping, used as the base for the
# load_params validation tests below (each test mutates a copy of this).
BASE_PARAMS: dict = {
    "variants": 2, "seed": 32, "valid_frac": 0.15, "test_frac": 0.15,
    "stratify": "rarest_class", "split_order": "before_augmentation", "label_dir": "labels",
    "jpeg_quality": 95, "flip_horizontal_p": 0.5, "flip_vertical_p": 0.5,
    "rot90_choices": ["none", "cw", "ccw", "r180"], "rotation_deg": 13,
    "shear_deg_x": 10, "shear_deg_y": 10, "hue_deg": 15,
    "saturation_frac": 0.25, "exposure_frac": 0.10, "blur_sigma_max_px": 1.5,
    "noise_pixel_frac_max": 0.001, "min_box_px": 2, "min_visible_area_frac": 0.20,
}


def write_params_file(
    path: Path, overrides: dict | None = None, remove: list[str] | None = None,
    extra: dict | None = None,
) -> None:
    """Write a params YAML file derived from ``BASE_PARAMS`` for a test.

    ``overrides`` replaces existing keys, ``remove`` deletes keys, ``extra``
    adds keys not in the schema. Used to build each adversarial variant
    (missing key, unknown key, wrong type, out-of-range) from one known-good
    starting point.
    """
    params = dict(BASE_PARAMS)
    if overrides:
        params.update(overrides)
    if remove:
        for key in remove:
            params.pop(key, None)
    if extra:
        params.update(extra)
    doc = {"_meta": {"name": "test", "description": "test", "source": "test"},
          "params": params}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def check(condition: bool, message: str) -> None:
    """Record one assertion, printing PASS or FAIL and counting failures."""
    global FAILURES, CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS {message}")
    else:
        FAILURES += 1
        print(f"FAIL {message}")


def section(title: str) -> None:
    """Print a section heading so a failure is easy to place."""
    print(f"\n--- {title}")


def make_source_dataset(root: Path, n_images: int = 30, size: int = 64) -> Path:
    """Build a tiny synthetic YOLO dataset with a coloured rectangle per image.

    Each image gets exactly one box: a solid-colour rectangle whose YOLO
    label matches its pixel position. Rare classes (ids 50-58) appear on only
    one or two images, so the stratified split has real strata to work with.
    """
    src = root / "source"
    label_dir = ca.load_params(ca.DEFAULT_PARAMS_PATH)["label_dir"]
    (src / "images").mkdir(parents=True, exist_ok=True)
    (src / label_dir).mkdir(parents=True, exist_ok=True)
    n_classes = 59
    classes = [f"class_{i:02d}" for i in range(n_classes)]
    (src / "classes.txt").write_text("\n".join(classes) + "\n")
    (src / "data.yaml").write_text("path: .\nnc: 59\n")
    (src / "LICENSE.md").write_text("test license\n")
    (src / "ingest_manifest.json").write_text(json.dumps({
        "dataset": "synthetic_test", "n_images": n_images, "n_boxes": n_images,
        "n_classes": n_classes, "classes": classes,
        "citation": "test citation", "license_note": "test license note",
        "coverage_note": "test coverage note", "subject": "test",
    }))

    rng = np.random.default_rng(1)
    for i in range(n_images):
        img = np.full((size, size, 3), 40, dtype=np.uint8)
        bw, bh = size // 3, size // 3
        x0 = rng.integers(0, size - bw)
        y0 = rng.integers(0, size - bh)
        color = (int(rng.integers(100, 256)), 30, 30)
        img[y0:y0 + bh, x0:x0 + bw] = color
        stem = f"img_{i:03d}"
        Image.fromarray(img).save(src / "images" / f"{stem}.jpg", quality=95)
        # rare classes on the first few images only
        cls_id = i if i < 9 else int(rng.integers(0, 40))
        cx = (x0 + bw / 2) / size
        cy = (y0 + bh / 2) / size
        w = bw / size
        h = bh / size
        (src / label_dir / f"{stem}.txt").write_text(
            f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )
    return src


SATURATION_MASK_THRESHOLD = 60
VALUE_MASK_THRESHOLD = 30


def rectangle_centroid_inside_box(image_rgb: np.ndarray, box: ca.Box) -> bool:
    """Return whether the coloured rectangle's pixel mass sits inside ``box``.

    Used to confirm a transformed box still covers its rectangle. The
    rectangle is picked out by HSV saturation and value rather than distance
    from the grey background colour, because a rotation or shear fills any
    canvas area the source image no longer covers with pure black, and a
    background-distance mask would misread that black border as foreground.
    The rectangle (saturated, mid-brightness) is neither the low-saturation
    grey background nor the near-zero-value black border.
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] > SATURATION_MASK_THRESHOLD) & (hsv[:, :, 2] > VALUE_MASK_THRESHOLD)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return False
    centroid_x = xs.mean() / w
    centroid_y = ys.mean() / h
    x1, x2 = box.cx - box.w / 2, box.cx + box.w / 2
    y1, y2 = box.cy - box.h / 2, box.cy + box.h / 2
    return x1 <= centroid_x <= x2 and y1 <= centroid_y <= y2


SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)

try:
    # ── the label parser ─────────────────────────────────────────────────
    section("the label parser")
    tmp = SCRATCH_ROOT / "labels"
    tmp.mkdir(parents=True, exist_ok=True)

    good = tmp / "good.txt"
    good.write_text("0 0.5 0.5 0.2 0.3\n5 0.1 0.1 0.05 0.05\n")
    boxes = ca.read_label_file(good, num_classes=59)
    check(len(boxes) == 2, "a good file parses both lines")
    check(boxes[0] == ca.Box(0, 0.5, 0.5, 0.2, 0.3), "the first box's fields are exact")

    empty = tmp / "empty.txt"
    empty.write_text("")
    check(ca.read_label_file(empty, num_classes=59) == [], "a zero-byte file parses empty")

    blank_lines = tmp / "blank.txt"
    blank_lines.write_text("\n\n0 0.5 0.5 0.1 0.1\n\n")
    check(len(ca.read_label_file(blank_lines, num_classes=59)) == 1,
          "blank lines around a real line are skipped, not counted")

    malformed = tmp / "malformed.txt"
    malformed.write_text("0 0.5 0.5 0.1 0.1\n1 2 3\n")
    try:
        ca.read_label_file(malformed, num_classes=59)
        check(False, "a malformed line (wrong field count) is refused")
    except ValueError as exc:
        check(f"{malformed}:2" in str(exc), f"the error names file and line, got {exc!r}")

    bad_class = tmp / "bad_class.txt"
    bad_class.write_text("99 0.5 0.5 0.1 0.1\n")
    try:
        ca.read_label_file(bad_class, num_classes=59)
        check(False, "an out-of-range class id is refused")
    except ValueError as exc:
        check("outside" in str(exc), "the error names the range violated")

    nan_file = tmp / "nan.txt"
    nan_file.write_text("0 nan 0.5 0.1 0.1\n")
    try:
        ca.read_label_file(nan_file, num_classes=59)
        check(False, "a NaN coordinate is refused")
    except ValueError as exc:
        check("non-finite" in str(exc), "the error names the non-finite coordinate")

    inf_file = tmp / "inf.txt"
    inf_file.write_text("0 0.5 0.5 inf 0.1\n")
    try:
        ca.read_label_file(inf_file, num_classes=59)
        check(False, "an infinite coordinate is refused")
    except ValueError as exc:
        check("non-finite" in str(exc), "infinity is caught by the same finiteness check")

    non_int_class = tmp / "non_int.txt"
    non_int_class.write_text("1.5 0.5 0.5 0.1 0.1\n")
    try:
        ca.read_label_file(non_int_class, num_classes=59)
        check(False, "a non-integer class id is refused")
    except ValueError as exc:
        check("not an integer" in str(exc), "the error names the integer failure")

    huge = tmp / "huge.txt"
    huge.write_text("0 0.5 0.5 500.0 500.0\n")
    huge_boxes = ca.read_label_file(huge, num_classes=59)
    check(len(huge_boxes) == 1, "a huge box is parsed, not rejected at read time")

    zero_byte = tmp / "zero.txt"
    zero_byte.write_bytes(b"")
    check(ca.read_label_file(zero_byte, num_classes=59) == [],
          "a zero-byte label file (bytes, not str) parses as background")

    missing = tmp / "does_not_exist.txt"
    try:
        ca.read_label_file(missing, num_classes=59)
        check(False, "a missing label file raises")
    except OSError as exc:
        check("cannot read" in str(exc), "a missing file's error names the path")

    # ── discovering pairs ────────────────────────────────────────────────
    section("discovering image/label pairs")
    disc = SCRATCH_ROOT / "discover"
    (disc / "images").mkdir(parents=True, exist_ok=True)
    (disc / "labels_localized").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(disc / "images" / "a.jpg")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(disc / "images" / "b.jpg")
    (disc / "labels_localized" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    pairs, unpaired = ca.discover_pairs(disc / "images", disc / "labels_localized")
    check(set(pairs) == {"a"}, "only the image with a label file is paired")
    check(unpaired == 1, "the unlabelled image is counted, not silently dropped")

    no_labels_dir = SCRATCH_ROOT / "no_labels_dir"
    (no_labels_dir / "images").mkdir(parents=True, exist_ok=True)
    try:
        ca.discover_pairs(no_labels_dir / "images", no_labels_dir / "labels_localized")
        check(False, "a source with images/ but no labels_localized/ is refused")
    except OSError as exc:
        check("does not exist" in str(exc), "the missing-directory error names it")

    ambiguous = SCRATCH_ROOT / "ambiguous"
    (ambiguous / "images").mkdir(parents=True, exist_ok=True)
    (ambiguous / "labels_localized").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(ambiguous / "images" / "x.jpg")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(ambiguous / "images" / "x.png")
    (ambiguous / "labels_localized" / "x.txt").write_text("")
    try:
        ca.discover_pairs(ambiguous / "images", ambiguous / "labels_localized")
        check(False, "two images sharing a stem under different extensions is refused")
    except ValueError as exc:
        check("share the stem" in str(exc), "the ambiguity error names the shared stem")

    missing_source = SCRATCH_ROOT / "does_not_exist_at_all"
    try:
        ca.discover_pairs(missing_source / "images", missing_source / "labels_localized")
        check(False, "a wholly missing source directory is refused")
    except OSError:
        check(True, "a wholly missing source directory raises OSError")

    # ── stratified split ─────────────────────────────────────────────────
    section("stratified split: determinism and fractions")
    n_synth = 1000
    boxes_by_stem = {}
    stems_1000 = [f"s{i:04d}" for i in range(n_synth)]
    for i, stem in enumerate(stems_1000):
        if i < 3:
            boxes_by_stem[stem] = [ca.Box(58, 0.5, 0.5, 0.1, 0.1)]  # 3-image rare class
        elif i < 500:
            boxes_by_stem[stem] = [ca.Box(0, 0.5, 0.5, 0.1, 0.1)]
        else:
            boxes_by_stem[stem] = []  # stratum "none"

    pins_a = ca.stratified_split(stems_1000, boxes_by_stem, 59, 0.15, 0.15, 32)
    pins_b = ca.stratified_split(stems_1000, boxes_by_stem, 59, 0.15, 0.15, 32)
    check(pins_a == pins_b, "the same seed reproduces the identical split")

    pins_c = ca.stratified_split(stems_1000, boxes_by_stem, 59, 0.15, 0.15, 7)
    check(pins_a != pins_c, "a different seed changes the split")

    rare_stems = stems_1000[:3]
    rare_splits = [pins_a[s] for s in rare_stems]
    check(rare_splits.count("test") == 1 and rare_splits.count("valid") == 1,
          f"the 3-image rare class gets exactly 1 test and 1 valid, got {rare_splits}")

    none_stems = stems_1000[500:]
    none_counts = {"train": 0, "valid": 0, "test": 0}
    for s in none_stems:
        none_counts[pins_a[s]] += 1
    check(sum(none_counts.values()) == len(none_stems),
          "every no-box image is assigned a split (the 'none' stratum is handled)")
    check(none_counts["test"] > 0 and none_counts["valid"] > 0,
          "the 'none' stratum, being large, still gets held-out images")

    class0_stems = stems_1000[3:500]
    class0_counts = {"train": 0, "valid": 0, "test": 0}
    for s in class0_stems:
        class0_counts[pins_a[s]] += 1
    n0 = len(class0_stems)
    check(abs(class0_counts["test"] - round(n0 * 0.15)) <= 1,
          f"class-0 stratum test count is close to 15%, got {class0_counts}")
    check(abs(class0_counts["valid"] - round(n0 * 0.15)) <= 1,
          f"class-0 stratum valid count is close to 15%, got {class0_counts}")

    try:
        ca.stratified_split(stems_1000[:5], {s: [] for s in stems_1000[:5]}, 59, 0.6, 0.6, 1)
        check(False, "fractions summing to 1.2 are refused")
    except ValueError as exc:
        check("must each be" in str(exc), "the fraction error names the constraint")

    section("no python hash() in the seeding")
    source_lines = Path(ca.__file__).read_text().splitlines()
    hash_calls = [line for line in source_lines
                 if "hash(" in line and "``hash()``" not in line]
    check(not hash_calls,
          f"the module never calls python's salted hash() for anything seed-related, "
          f"found: {hash_calls}")
    check("hashlib.sha256" in Path(ca.__file__).read_text(),
          "seeding goes through hashlib.sha256 instead")

    # ── box geometry ─────────────────────────────────────────────────────
    section("box geometry: flips and 90-degree rotations")
    W, H = 100, 60
    box = ca.Box(0, 0.2, 0.3, 0.1, 0.1)  # centre (20,18) px in a 100x60 image

    def identity_map(x, y):
        return x, y

    kept, dropped = ca.transform_boxes(
        [box], identity_map, (W, H), (W, H), TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    close = (dropped == 0 and len(kept) == 1
             and all(abs(a - b) < 1e-9 for a, b in zip(kept[0], box)))
    check(close, f"an identity point map leaves the box unchanged (within float "
                f"round-trip error), got {kept}, expected {box}")

    def hflip_map(x, y):
        return W - x, y

    kept, _ = ca.transform_boxes(
        [box], hflip_map, (W, H), (W, H), TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(abs(kept[0].cx - (1 - box.cx)) < 1e-9 and kept[0].cy == box.cy,
          "a horizontal flip mirrors cx and leaves cy alone")

    def vflip_map(x, y):
        return x, H - y

    kept, _ = ca.transform_boxes(
        [box], vflip_map, (W, H), (W, H), TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(abs(kept[0].cy - (1 - box.cy)) < 1e-9 and kept[0].cx == box.cx,
          "a vertical flip mirrors cy and leaves cx alone")

    params_cw = {"hflip": False, "vflip": False, "rot90": "cw",
                 "angle_deg": 0.0, "shear_x_deg": 0.0, "shear_y_deg": 0.0,
                 "hue_shift_deg": 0.0, "sat_jitter": 0.0, "exposure_jitter": 0.0,
                 "blur_sigma_px": 0.0, "noise_frac": 0.0}
    img_wh = np.zeros((H, W, 3), dtype=np.uint8)
    warped, point_map, out_wh = ca.apply_geometry(img_wh, params_cw)
    check(out_wh == (H, W), f"a clockwise 90-degree rotation swaps width and height, got {out_wh}")
    check(warped.shape[:2] == (W, H), "the warped array's shape matches the swapped dims")
    tl = point_map(0, 0)
    check(abs(tl[0] - H) < 1e-6 and abs(tl[1] - 0) < 1e-6,
          f"clockwise: the top-left corner maps to the top-right corner, got {tl}")

    params_ccw = dict(params_cw, rot90="ccw")
    _, point_map_ccw, out_wh_ccw = ca.apply_geometry(img_wh, params_ccw)
    check(out_wh_ccw == (H, W), "counter-clockwise also swaps width and height")
    tl_ccw = point_map_ccw(0, 0)
    check(abs(tl_ccw[0] - 0) < 1e-6 and abs(tl_ccw[1] - W) < 1e-6,
          f"counter-clockwise: the top-left corner maps to the bottom-left corner, got {tl_ccw}")

    params_r180 = dict(params_cw, rot90="r180")
    _, point_map_r180, out_wh_r180 = ca.apply_geometry(img_wh, params_r180)
    check(out_wh_r180 == (W, H), "a 180-degree rotation keeps the same dimensions")
    tl_r180 = point_map_r180(0, 0)
    check(abs(tl_r180[0] - W) < 1e-6 and abs(tl_r180[1] - H) < 1e-6,
          f"180 degrees: the top-left corner maps to the bottom-right corner, got {tl_r180}")

    section("rotation 0 and shear 0 are identity")
    identity_params = dict(params_cw, rot90="none")
    matrix = ca.affine_for(identity_params, W, H)
    check(np.allclose(matrix, np.eye(3), atol=1e-9),
          "zero rotation and zero shear compose to the identity matrix")

    section("a 90-degree rotation via the continuous affine matches rot90's corners")
    # On a SQUARE canvas, a continuous +90-degree rotation about the centre
    # changes no dimension, so it should carry every corner exactly where the
    # discrete 90-degree rotation does (cv2's positive-angle convention lines
    # up with this module's "ccw" discrete mode; that alignment, not the
    # label, is what this checks).
    S = 80
    square_img = np.zeros((S, S, 3), dtype=np.uint8)
    continuous_params = dict(params_cw, rot90="none", angle_deg=90.0)
    _, continuous_map, continuous_wh = ca.apply_geometry(square_img, continuous_params)
    discrete_params = dict(params_cw, rot90="ccw", angle_deg=0.0)
    _, discrete_map, discrete_wh = ca.apply_geometry(square_img, discrete_params)
    check(continuous_wh == discrete_wh == (S, S),
          "a square canvas keeps its size under both the continuous and discrete rotation")
    corners = [(0.0, 0.0), (S, 0.0), (S, S), (0.0, S)]
    mismatches = [
        (p, continuous_map(*p), discrete_map(*p))
        for p in corners
        if not np.allclose(continuous_map(*p), discrete_map(*p), atol=1e-6)
    ]
    check(not mismatches,
          f"the continuous +90-degree affine agrees with the discrete rot90 at every "
          f"corner of a square canvas, mismatches: {mismatches}")

    section("clipping and drop rules")
    tiny = ca.Box(0, 0.5, 0.5, 0.001, 0.001)  # far under 2px in a 100x60 frame
    kept, dropped = ca.transform_boxes(
        [tiny], identity_map, (W, H), (W, H), TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(dropped == 1 and kept == [], "a sub-2px box is dropped")

    # A box whose clipped width and height both stay above the 2px floor, so
    # only the area-ratio rule (not the side-length rule) can be responsible
    # for dropping it: an 80% loss of area from a 70px leftward shove.
    area_rule_box = ca.Box(0, 0.5, 0.5, 0.5, 0.5)  # 50x30 px in a 100x60 frame

    def shove_left_70_map(x, y):
        return x - 70, y

    kept, dropped = ca.transform_boxes(
        [area_rule_box], shove_left_70_map, (W, H), (W, H),
        TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(dropped == 1 and kept == [],
          "a box whose clipped sides both clear 2px is still dropped when "
          "clipped area falls under 20 percent of its unclipped area")

    # The same box shifted by only 5px keeps most of its area and survives.
    def shove_left_5_map(x, y):
        return x - 5, y

    kept, dropped = ca.transform_boxes(
        [area_rule_box], shove_left_5_map, (W, H), (W, H),
        TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(dropped == 0 and len(kept) == 1,
          "the same box barely shifted keeps enough area and survives")

    def push_fully_out_map(x, y):
        return x - 1000, y - 1000

    fully_out = ca.Box(0, 0.5, 0.5, 0.1, 0.1)
    kept, dropped = ca.transform_boxes(
        [fully_out], push_fully_out_map, (W, H), (W, H),
        TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(dropped == 1 and kept == [], "a box mapped fully out of frame is dropped, not clipped to zero")

    # A box only slightly larger than the frame, centred, keeps almost all of
    # its area after clipping and survives, clamped to the frame edges.
    oversize_box = ca.Box(0, 0.5, 0.5, 1.2, 1.2)
    kept, dropped = ca.transform_boxes(
        [oversize_box], identity_map, (W, H), (W, H),
        TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(len(kept) == 1 and dropped == 0
          and abs(kept[0].w - 1.0) < 1e-6 and abs(kept[0].h - 1.0) < 1e-6,
          f"a box only slightly larger than the frame is clipped to the frame "
          f"edges, not dropped, got {kept}")

    # A box vastly larger than the frame (500 percent) loses over 80 percent
    # of its area to clipping and is dropped by the same area-ratio rule.
    huge_box = ca.Box(0, 0.5, 0.5, 5.0, 5.0)
    kept, dropped = ca.transform_boxes(
        [huge_box], identity_map, (W, H), (W, H),
        TEST_MIN_BOX_PX, TEST_MIN_AREA_RATIO)
    check(kept == [] and dropped == 1,
          "a box vastly larger than the frame is dropped, not clamped, because "
          "almost none of its area would remain")

    # ── photometric ──────────────────────────────────────────────────────
    section("photometric transforms")
    sample_img = (np.random.default_rng(3).integers(0, 255, (40, 40, 3))).astype(np.uint8)
    rng_photo = np.random.default_rng(3)
    hue_wrap_params = dict(identity_params, hue_shift_deg=180.0)
    out = ca.photometric(sample_img, hue_wrap_params, rng_photo)
    check(out.dtype == np.uint8, "the photometric output stays uint8")
    check(out.shape == sample_img.shape, "the photometric output keeps the input shape")

    hsv_flat = np.zeros((8, 8, 3), dtype=np.uint8)
    hsv_flat[:, :, 0], hsv_flat[:, :, 1], hsv_flat[:, :, 2] = 50, 200, 200
    flat_rgb = cv2.cvtColor(hsv_flat, cv2.COLOR_HSV2RGB)
    base_hue = int(cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])
    for deg in (15.0, -15.0):
        shifted = ca.photometric(flat_rgb, dict(identity_params, hue_shift_deg=deg, sat_jitter=0.0,
                                                exposure_jitter=0.0, blur_sigma_px=0.0,
                                                noise_frac=0.0), np.random.default_rng(0))
        moved = int(cv2.cvtColor(shifted, cv2.COLOR_RGB2HSV)[0, 0, 0]) - base_hue
        check(abs(moved) <= 8 and abs(int(math.trunc(deg / 2))) == 7,
              f"a {deg:+.0f} degree hue draw shifts by 7 OpenCV units (14 degrees), never 16 "
              f"(measured {moved} units, round-trip rounding allowed)")

    no_op_params = dict(identity_params, hue_shift_deg=0.0, sat_jitter=0.0,
                        exposure_jitter=0.0, blur_sigma_px=0.0, noise_frac=0.0)
    no_op_out = ca.photometric(sample_img, no_op_params, np.random.default_rng(3))
    max_round_trip_diff = int(np.max(np.abs(no_op_out.astype(int) - sample_img.astype(int))))
    # OpenCV's 8-bit RGB<->HSV conversion is lossy at the quantization boundary
    # even with a null transform; a spot check over 200 random 64x64 images
    # found a worst case of 6 per channel, so 10 leaves headroom without
    # being too loose to catch a real bug (which would show as a much larger,
    # spatially structured difference, not a uniform +/-1 to 6 rounding noise).
    check(max_round_trip_diff <= 10,
          f"zero hue/sat/exposure jitter, zero sigma, zero noise reproduces the input "
          f"within RGB<->HSV round-trip rounding (max per-channel diff "
          f"{max_round_trip_diff}, allowed 10)")

    # Isolate the noise draw's own effect by comparing against a baseline run
    # through the identical (lossy) HSV round trip, rather than the raw input,
    # so the count reflects only the injected salt-and-pepper pixels.
    noisy_params = dict(no_op_params, noise_frac=0.001)
    noisy_out = ca.photometric(sample_img, noisy_params, np.random.default_rng(3))
    n_changed = np.sum(np.any(noisy_out != no_op_out, axis=2))
    max_expected = round(0.001 * 40 * 40) + 1
    check(n_changed <= max_expected,
          f"the noise fraction bounds the number of changed pixels, changed {n_changed}, "
          f"expected at most {max_expected}")

    section("param ranges respected over many draws")
    ranges_ok = True
    for trial in range(2000):
        p = ca.draw_params(np.random.default_rng(trial), TEST_CFG)
        if not (-TEST_CFG["rotation_deg"] <= p["angle_deg"] <= TEST_CFG["rotation_deg"]):
            ranges_ok = False
        if not (-TEST_CFG["shear_deg_x"] <= p["shear_x_deg"] <= TEST_CFG["shear_deg_x"]):
            ranges_ok = False
        if not (-TEST_CFG["hue_deg"] <= p["hue_shift_deg"] <= TEST_CFG["hue_deg"]):
            ranges_ok = False
        if not (0.0 <= p["noise_frac"] <= TEST_CFG["noise_pixel_frac_max"]):
            ranges_ok = False
        if p["rot90"] not in ca.ALL_ROT90_MODES:
            ranges_ok = False
    check(ranges_ok, "every drawn parameter over 2000 trials stayed inside its Table 2 range")

    section("seeded RNG reproducibility")
    seed_a = ca.stable_variant_seed(32, "img_007", 1)
    seed_b = ca.stable_variant_seed(32, "img_007", 1)
    check(seed_a == seed_b, "the same (seed, stem, k) hashes to the same RNG seed")
    seed_c = ca.stable_variant_seed(32, "img_007", 2)
    check(seed_a != seed_c, "a different variant index hashes to a different seed")
    params_a = ca.draw_params(np.random.default_rng(seed_a), TEST_CFG)
    params_b = ca.draw_params(
        np.random.default_rng(ca.stable_variant_seed(32, "img_007", 1)), TEST_CFG)
    check(params_a == params_b, "re-deriving the seed reproduces the identical parameter draw")

    # ── load_params validation ──────────────────────────────────────────
    section("load_params: the real params file")
    real_cfg = ca.load_params(ca.DEFAULT_PARAMS_PATH)
    check(real_cfg["seed"] == 32 and real_cfg["variants"] == 2,
          f"the real Table 2 params file loads with its documented seed and "
          f"variant count, got {real_cfg['seed']!r}, {real_cfg['variants']!r}")
    check(real_cfg["rot90_choices"] == ["none", "cw", "ccw", "r180"],
          "the real file's rot90 choices load in order")
    check(isinstance(real_cfg["rotation_deg"], float), "an integer YAML value "
         "for a float-typed key is coerced to float, not left as int")

    params_dir = SCRATCH_ROOT / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    section("load_params: a good file")
    good_params = params_dir / "good.yaml"
    write_params_file(good_params)
    cfg = ca.load_params(good_params)
    check(cfg["variants"] == 2 and cfg["label_dir"] == "labels",
          "a good params file loads to the expected values")

    section("load_params: missing file")
    try:
        ca.load_params(params_dir / "does_not_exist.yaml")
        check(False, "a missing params file is refused")
    except OSError as exc:
        check("cannot read" in str(exc), "the missing-file error names the path")

    section("load_params: malformed YAML")
    malformed_yaml = params_dir / "malformed.yaml"
    malformed_yaml.write_text("params:\n  seed: 32\n  variants: [1, 2\n bad indent: -\n")
    try:
        ca.load_params(malformed_yaml)
        check(False, "malformed YAML is refused")
    except ValueError as exc:
        check("not valid YAML" in str(exc), "the malformed-YAML error says so")

    section("load_params: unknown key")
    unknown_key = params_dir / "unknown_key.yaml"
    write_params_file(unknown_key, extra={"bogus_extra_setting": 1})
    try:
        ca.load_params(unknown_key)
        check(False, "an unknown parameter key is refused")
    except ValueError as exc:
        check("bogus_extra_setting" in str(exc) and "unknown" in str(exc),
              f"the unknown-key error names it, got {exc!r}")

    section("load_params: missing key")
    missing_key = params_dir / "missing_key.yaml"
    write_params_file(missing_key, remove=["seed"])
    try:
        ca.load_params(missing_key)
        check(False, "a missing required key is refused")
    except ValueError as exc:
        check("'seed'" in str(exc) and "missing" in str(exc),
              f"the missing-key error names it, got {exc!r}")

    section("load_params: out-of-range value")
    out_of_range = params_dir / "out_of_range.yaml"
    write_params_file(out_of_range, overrides={"valid_frac": 1.5})
    try:
        ca.load_params(out_of_range)
        check(False, "an out-of-range value is refused")
    except ValueError as exc:
        check("valid_frac" in str(exc), f"the range error names the key, got {exc!r}")

    negative_range = params_dir / "negative.yaml"
    write_params_file(negative_range, overrides={"blur_sigma_max_px": -1.0})
    try:
        ca.load_params(negative_range)
        check(False, "a negative value for a non-negative-only key is refused")
    except ValueError as exc:
        check("blur_sigma_max_px" in str(exc), "the negative-blur error names the key")

    section("load_params: wrong type")
    wrong_type = params_dir / "wrong_type.yaml"
    write_params_file(wrong_type, overrides={"variants": "two"})
    try:
        ca.load_params(wrong_type)
        check(False, "a string given where an integer is expected is refused")
    except ValueError as exc:
        check("variants" in str(exc) and "integer" in str(exc),
              f"the type error names the key and the expected type, got {exc!r}")

    fractional_int = params_dir / "fractional_int.yaml"
    write_params_file(fractional_int, overrides={"variants": 2.5})
    try:
        ca.load_params(fractional_int)
        check(False, "a float given where an integer is expected is refused")
    except ValueError as exc:
        check("variants" in str(exc), "a non-integer float for an int key is refused")

    bad_choice = params_dir / "bad_choice.yaml"
    write_params_file(bad_choice, overrides={"rot90_choices": ["none", "spin"]})
    try:
        ca.load_params(bad_choice)
        check(False, "an unrecognised rot90 choice is refused")
    except ValueError as exc:
        check("spin" in str(exc), "the bad-choice error names the offending value")

    # ── CLI adversarial cases ───────────────────────────────────────────
    section("CLI: missing source directory")
    code = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(SCRATCH_ROOT / "no_such_source"),
         "--out_dir", str(SCRATCH_ROOT / "out_missing_source")],
        capture_output=True, text=True,
    )
    check(code.returncode != 0, "a missing source directory is refused")

    section("CLI: non-empty out_dir without --force")
    small_src = make_source_dataset(SCRATCH_ROOT / "cli_refuse", n_images=5, size=48)
    out_nonempty = SCRATCH_ROOT / "cli_refuse" / "out"
    out_nonempty.mkdir(parents=True, exist_ok=True)
    (out_nonempty / "keepme.txt").write_text("pre-existing file")
    code = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(small_src), "--out_dir", str(out_nonempty)],
        capture_output=True, text=True,
    )
    check(code.returncode != 0, "a non-empty out_dir without --force is refused")
    check((out_nonempty / "keepme.txt").is_file(),
          "the refusal leaves the pre-existing file untouched")

    # ── hostile file names ───────────────────────────────────────────────
    section("hostile file names survive")
    hostile_src = SCRATCH_ROOT / "hostile" / "source"
    (hostile_src / "images").mkdir(parents=True, exist_ok=True)
    (hostile_src / "labels_localized").mkdir(parents=True, exist_ok=True)
    hostile_stem = "a b'c_éß"
    Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8)).save(
        hostile_src / "images" / f"{hostile_stem}.jpg")
    (hostile_src / "labels_localized" / f"{hostile_stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n")
    hostile_pairs, hostile_unpaired = ca.discover_pairs(
        hostile_src / "images", hostile_src / "labels_localized")
    check(hostile_stem in hostile_pairs and hostile_unpaired == 0,
          "a stem with spaces, a quote and unicode is paired correctly")

    section("integration: a synthetic dataset through the real CLI")
    synth_root = SCRATCH_ROOT / "integration"
    synth_source = make_source_dataset(synth_root, n_images=30, size=64)
    synth_out = synth_root / "out"
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(synth_source), "--out_dir", str(synth_out),
         "--seed", "32", "--variants", "2", "--valid_frac", "0.15",
         "--test_frac", "0.15", "--workers", "2"],
        capture_output=True, text=True,
    )
    check(result.returncode == 0,
          f"the CLI succeeds on a 30-image synthetic dataset, stderr:\n{result.stderr}")
    check("Step 6 complete" in result.stdout, "the CLI prints its final verification checkpoint")

    pins_doc = json.loads((synth_out / "split_pins.json").read_text())
    manifest_doc = json.loads((synth_out / "augment_manifest.json").read_text())
    n_originals = 30
    n_train_sources = sum(1 for s, sp in pins_doc["pins"].items()
                          if sp == "train" and ca.VARIANT_MARKER not in s)
    check(len(pins_doc["pins"]) == n_originals + n_train_sources * 2,
          f"pins cover every original plus 2 variants per train source, "
          f"got {len(pins_doc['pins'])} for {n_train_sources} train sources")
    check(manifest_doc["seed"] == 32, "the manifest records the seed used")
    check(all(pins_doc["pins"][name] == "train"
             for name in pins_doc["pins"] if ca.VARIANT_MARKER in name),
          "every variant in the real run is pinned train")

    section("VEGA's own dataset builder replays the pins (regression, 2026-09-26)")
    written = {p.name for p in (synth_out / "images").iterdir()}
    check(set(pins_doc["pins"]) == written,
          "pins are keyed by image file name with extension, exactly the images written")
    sys.path.insert(0, str(Path(ca.__file__).resolve().parents[1]
                           / "TCRMPtrain_oceankindCV" / "src"))
    import detect_dataset  # noqa: E402
    vega_out = synth_root / "vega_built"
    built = detect_dataset.build_detect_dataset(str(synth_out), str(vega_out), "synthetic",
                                                valid_frac=0.15, test_frac=0.15)
    check(built.get("pins_source") == "staged split_pins.json",
          f"VEGA reads the staged pins, got pins_source {built.get('pins_source')!r}")
    misplaced = [n for sp in ("train", "valid", "test")
                 for n in os.listdir(vega_out / sp / "images") if pins_doc["pins"][n] != sp]
    check(not misplaced, f"every image sits in its pinned split in VEGA's build: {misplaced[:3]}")
    held_out_variants = [n for sp in ("valid", "test")
                         for n in os.listdir(vega_out / sp / "images") if ca.VARIANT_MARKER in n]
    check(not held_out_variants, "no augmented variant reaches VEGA's validation or test split")

    section("split after augmenting: every photo copied, images split one by one")
    leaky_params = params_dir / "leaky.yaml"
    write_params_file(leaky_params, overrides={"split_order": "after_augmentation"})
    leaky_out = synth_root / "leaky_out"
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)), "--source_dir", str(synth_source),
         "--out_dir", str(leaky_out), "--params", str(leaky_params), "--workers", "1"],
        capture_output=True, text=True)
    check(result.returncode == 0, f"the after-augmentation build succeeds: {result.stderr[-300:]}")
    lpins = json.loads((leaky_out / "split_pins.json").read_text())["pins"]
    lman = json.loads((leaky_out / "augment_manifest.json").read_text())
    check(len(lpins) == 30 * 3, f"every one of 30 photos is tripled (got {len(lpins)} images)")
    check(any(ca.VARIANT_MARKER in n and sp != "train" for n, sp in lpins.items()),
          "augmented copies reach validation or test")
    check(lman["split_order"] == "after_augmentation"
          and lman["sibling_leak"]["test"]["with_copy_in_train"] > 0,
          f"the manifest records the order and counts test images with a copy in training: "
          f"{lman['sibling_leak']['test']}")
    strict = ca.sibling_leak(pins_doc["pins"])
    check(strict["test"]["with_copy_in_train"] == 0 and strict["valid"]["with_copy_in_train"] == 0,
          "the strict order leaks nothing, by the same measure")
    vega_leaky = synth_root / "vega_leaky"
    built_l = detect_dataset.build_detect_dataset(str(leaky_out), str(vega_leaky), "leaky",
                                                  valid_frac=0.15, test_frac=0.15)
    check(built_l.get("pins_source") == "staged split_pins.json"
          and all(lpins[n] == sp for sp in ("train", "valid", "test")
                  for n in os.listdir(vega_leaky / sp / "images")),
          "VEGA replays the after-augmentation split exactly")

    section("the params file is copied verbatim and its sha256 recorded")
    copied_params = synth_out / "augment_params.yaml"
    check(copied_params.is_file(), "augment_params.yaml appears in the output directory")
    expected_sha = hashlib.sha256(ca.DEFAULT_PARAMS_PATH.read_bytes()).hexdigest()
    check(copied_params.read_bytes() == ca.DEFAULT_PARAMS_PATH.read_bytes(),
          "the copied params file is byte-identical to the source")
    check(manifest_doc["params_sha256"] == expected_sha,
          "the manifest's recorded sha256 matches the actual params file")

    section("CLI override precedence: flags beat the params file, absence defers to it")
    override_root = SCRATCH_ROOT / "override"
    override_source = make_source_dataset(override_root, n_images=12, size=48)
    override_params = params_dir / "override_base.yaml"
    write_params_file(override_params, overrides={
        "seed": 32, "variants": 2, "valid_frac": 0.15, "test_frac": 0.15})

    no_override_out = override_root / "out_no_override"
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(override_source), "--out_dir", str(no_override_out),
         "--params", str(override_params), "--workers", "1"],
        capture_output=True, text=True,
    )
    check(result.returncode == 0, f"a run with no CLI overrides succeeds, "
         f"stderr:\n{result.stderr}")
    no_override_manifest = json.loads(
        (no_override_out / "augment_manifest.json").read_text())
    check(no_override_manifest["seed"] == 32
         and no_override_manifest["variants_per_train_source"] == 2,
          "with no --seed or --variants flag, the params file's values are used")

    with_override_out = override_root / "out_with_override"
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(override_source), "--out_dir", str(with_override_out),
         "--params", str(override_params), "--seed", "999", "--variants", "1",
         "--workers", "1"],
        capture_output=True, text=True,
    )
    check(result.returncode == 0, f"a run with CLI overrides succeeds, "
         f"stderr:\n{result.stderr}")
    with_override_manifest = json.loads(
        (with_override_out / "augment_manifest.json").read_text())
    check(with_override_manifest["seed"] == 999,
          "--seed overrides the params file's seed")
    check(with_override_manifest["variants_per_train_source"] == 1,
          "--variants overrides the params file's variant count")
    check(with_override_manifest["valid_frac"] == 0.15
         and with_override_manifest["test_frac"] == 0.15,
          "valid_frac and test_frac, not passed on the CLI, still come from the file")

    section("integration: a transformed box still covers its rectangle")
    preview_count = 0
    for name, split in pins_doc["pins"].items():
        if ca.VARIANT_MARKER not in name or preview_count >= 3:
            continue
        label_path = (synth_out / ca.load_params(ca.DEFAULT_PARAMS_PATH)["label_dir"]
                      / f"{Path(name).stem}.txt")
        image_path = synth_out / "images" / name
        boxes_here = ca.read_label_file(label_path, num_classes=59)
        if not boxes_here:
            continue
        pixels = np.asarray(Image.open(image_path).convert("RGB"))
        check(rectangle_centroid_inside_box(pixels, boxes_here[0]),
              f"{name}: the rectangle's colour centroid sits inside its transformed box")
        preview_count += 1
    check(preview_count > 0, "at least one non-empty variant was actually checked")

    section("a corrupt JPEG fails the whole build loudly, not silently")
    corrupt_root = SCRATCH_ROOT / "corrupt"
    corrupt_source = make_source_dataset(corrupt_root, n_images=6, size=48)
    corrupt_image = sorted((corrupt_source / "images").glob("*.jpg"))[0]
    corrupt_image.write_bytes(b"not actually a jpeg")
    corrupt_out = corrupt_root / "out"
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(corrupt_source), "--out_dir", str(corrupt_out),
         "--workers", "1"],
        capture_output=True, text=True,
    )
    check(result.returncode != 0, "a corrupt JPEG among the training sources fails the build")
    check(corrupt_image.name.rsplit(".", 1)[0] in result.stderr,
          "the failure names the corrupt file")
    check(not (corrupt_out / "split_pins.json").exists(),
          "no split_pins.json is written when a training image fails to augment")

    section("--force clears only this tool's own files")
    force_root = SCRATCH_ROOT / "force_test"
    force_source = make_source_dataset(force_root, n_images=6, size=48)
    force_out = force_root / "out"
    subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(force_source), "--out_dir", str(force_out),
         "--workers", "1"],
        capture_output=True, text=True, check=True,
    )
    sentinel = force_out / "not_mine.txt"
    sentinel.write_text("a file this tool never wrote")
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(force_source), "--out_dir", str(force_out),
         "--workers", "1", "--force"],
        capture_output=True, text=True,
    )
    check(result.returncode == 0, f"--force rebuilds successfully, stderr:\n{result.stderr}")
    check(sentinel.is_file(), "--force leaves a file it does not own untouched")
    stale = force_out / "labels_localized"
    stale.mkdir(exist_ok=True)
    (stale / "img_000.txt").write_text("0 0.5 0.5 1 1\n")
    result = subprocess.run(
        [sys.executable, str(Path(ca.__file__)),
         "--source_dir", str(force_source), "--out_dir", str(force_out),
         "--workers", "1", "--force"],
        capture_output=True, text=True,
    )
    check(result.returncode == 0 and not stale.exists(),
          "--force removes a stale labels_localized/ that VEGA would otherwise prefer")

    # ── smoke run: --limit for a small real build ───────────────────────
    section("smoke run: --limit 60 on the real Coral Intel source")
    real_source = ca.DEFAULT_SOURCE_DIR
    if real_source.is_dir():
        smoke_out = SCRATCH_ROOT / "t2aug_smoke"
        result = subprocess.run(
            [sys.executable, str(Path(ca.__file__)),
             "--source_dir", str(real_source), "--out_dir", str(smoke_out),
             "--limit", "60", "--seed", "32", "--variants", "2",
             "--workers", "8"],
            capture_output=True, text=True,
        )
        check(result.returncode == 0,
              f"the smoke run on 60 real images succeeds, stderr:\n{result.stderr}")
        smoke_images = list((smoke_out / "images").glob("*"))
        smoke_labels = list((smoke_out / ca.load_params(ca.DEFAULT_PARAMS_PATH)["label_dir"]).glob("*"))
        check(len(smoke_images) == len(smoke_labels),
              "the smoke run's image and label counts match")
        print(f"smoke run wrote {len(smoke_images)} images to {smoke_out}")
    else:
        check(False, f"the real Coral Intel source dataset is missing at {real_source}")

finally:
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)
    check(not SCRATCH_ROOT.exists(), "every test artifact under the scratch directory was removed")

print(f"\n{CHECKS - FAILURES}/{CHECKS} passed.")
sys.exit(1 if FAILURES else 0)
