#!/usr/bin/env python3
"""Offline augmentation for the Coral Intel bounding-box replication.

Module:  scripts/_ablation/coral_intel_offline_aug.py
Purpose: Build a new staged bounding-box dataset that replicates Table 2 of
         Cordero et al. 2026 (Coral Intel: a YOLOv8 deep learning framework for
         monitoring Caribbean corals, Intelligent Marine Technology and
         Systems 4:18) as an offline, Roboflow-style augmentation pass. VEGA
         (the reef_point_seg training UI) has no offline augmentation step of
         its own, so this tool runs once, ahead of any VEGA training launch,
         and writes a dataset VEGA can load exactly like any other staged
         bounding-box dataset.

         Every training image keeps its original plus two to three augmented
         siblings (2 by default, ``--variants``), each drawing every Table 2
         transform independently at random: horizontal and vertical flip, a
         90 degree rotation, a small continuous rotation and shear, a hue,
         saturation and exposure jitter, a light Gaussian blur, and sparse
         salt-and-pepper noise. Validation and test images are never
         augmented, and the split is fixed before augmentation runs, so no
         augmented sibling of a held-out image ever enters training.

Inputs:  a staged YOLO box dataset directory holding images/ (JPEG), a
         label directory named by the params file's label_dir (labels/ by
         default, which keeps the ingest's full-frame boxes as the paper did)
         with one YOLO text file per image (class id plus normalized centre
         x, y, width, height; some files may be empty),
         classes.txt (one class name per line), data.yaml, ingest_manifest.json
         and LICENSE.md.
Outputs: a new dataset directory holding images/ (hardlinked originals plus
         augmented JPEGs), the same label directory (copied originals plus augmented
         YOLO text files), classes.txt, data.yaml and LICENSE.md copied from
         the source, split_pins.json (which split every image belongs to),
         augment_manifest.json (the augmentation design and its counts as
         executed) and ingest_manifest.json (a derived copy of the source
         manifest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps

# Images are never deleted: --force moves the previous images/ folder to
# <out>/_superseded/<stamp>/ (scripts/_superseded.py, one folder up).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _superseded as _sup  # noqa: E402

AST = timezone(timedelta(hours=-4))

REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_DIR / "supporting_data" / "bbox_datasets" / "coral_intel"
DEFAULT_OUT_DIR = REPO_DIR / "supporting_data" / "bbox_datasets" / "coral_intel_t2aug"
DEFAULT_PARAMS_PATH = (
    REPO_DIR / "config" / "augment_presets" / "coral_intel_table2.yaml"
)

DEFAULT_WORKERS = 16

# Image discovery.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Every 90-degree-stepped rotation this module knows how to apply. The params
# file's own ``rot90_choices`` must be a subset of this set.
ALL_ROT90_MODES = ("none", "cw", "ccw", "r180")

# Blur sigmas this small are visually a no-op; skip the convolution rather
# than run it on a kernel that would round to 1x1 anyway. Not a Table 2
# quantity, so it is not a params-file setting.
BLUR_SIGMA_SKIP_THRESHOLD_PX = 0.05

VARIANT_MARKER = "__t2aug"
# When the train, validation and test split is drawn. Before augmentation (the
# strict order) every copy of a photograph stays in its photograph's set and
# only training photographs are augmented. After augmentation (the order the
# Coral Intel counts would also allow) every photograph is augmented and the
# images are split one by one, so copies of one photograph can sit in training
# and in test at once; used only to measure what that does to the scores.
SPLIT_BEFORE = "before_augmentation"
SPLIT_AFTER = "after_augmentation"
VARIANT_IMAGE_EXT = ".jpg"

# The metadata entries this tool owns. ``--force`` clears exactly these and
# nothing else in the output directory.
MANAGED_OUTPUT_ENTRIES = (
    "images", "classes.txt", "data.yaml", "LICENSE.md", "split_pins.json",
    "augment_manifest.json", "ingest_manifest.json", "augment_params.yaml",
)
# Every label folder name VEGA reads from a staged dataset, in its order of
# preference (detect_dataset.BBOX_LABEL_DIRS). ``--force`` clears all of them,
# because VEGA reads labels_localized/ whenever it exists: a rebuild that
# switched to labels/ and left an old labels_localized/ behind would train on
# the old labels without a word.
VEGA_LABEL_DIRS = ("labels_localized", "labels")

# The augmentation parameter schema. Every key under a params file's
# ``params:`` mapping must appear here exactly once; ``load_params`` rejects
# an unknown key, a missing key, a wrong type, or a value outside its range.
# ``max_exclusive`` marks a bound that must not be reached (a fraction that
# must leave room for something else); a plain ``max`` may be reached.
PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "variants": {"type": int, "min": 1},
    "seed": {"type": int},
    "valid_frac": {"type": float, "min": 0.0, "max": 1.0, "max_exclusive": True},
    "test_frac": {"type": float, "min": 0.0, "max": 1.0, "max_exclusive": True},
    "stratify": {"type": str, "choices": {"rarest_class"}},
    "split_order": {"type": str, "choices": {SPLIT_BEFORE, SPLIT_AFTER}},
    "label_dir": {"type": str, "nonempty": True},
    "jpeg_quality": {"type": int, "min": 1, "max": 100},
    "flip_horizontal_p": {"type": float, "min": 0.0, "max": 1.0},
    "flip_vertical_p": {"type": float, "min": 0.0, "max": 1.0},
    "rot90_choices": {"type": list, "item_choices": set(ALL_ROT90_MODES)},
    "rotation_deg": {"type": float, "min": 0.0, "max": 180.0},
    "shear_deg_x": {"type": float, "min": 0.0, "max": 89.0},
    "shear_deg_y": {"type": float, "min": 0.0, "max": 89.0},
    "hue_deg": {"type": float, "min": 0.0, "max": 180.0},
    "saturation_frac": {"type": float, "min": 0.0, "max": 1.0},
    "exposure_frac": {"type": float, "min": 0.0, "max": 1.0},
    "blur_sigma_max_px": {"type": float, "min": 0.0},
    "noise_pixel_frac_max": {"type": float, "min": 0.0, "max": 1.0},
    "min_box_px": {"type": float, "min": 0.0},
    "min_visible_area_frac": {"type": float, "min": 0.0, "max": 1.0},
}


class Box(NamedTuple):
    """One YOLO-format bounding box: class id and normalized centre and size."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float


# ── time ─────────────────────────────────────────────────────────────────────

def ast_now() -> datetime:
    """Return the current time in Atlantic Standard Time."""
    return datetime.now(AST)


def stamp(when: datetime | None = None) -> str:
    """Return a human timestamp, ``YYYY-MM-DD HH:MM AST``."""
    return (when or ast_now()).strftime("%Y-%m-%d %H:%M AST")


# ── reading the source dataset ──────────────────────────────────────────────

def read_classes(path: Path) -> list[str]:
    """Read one class name per line from ``path``.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is empty.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise OSError(f"cannot read the class list {path}: {exc}") from exc
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        raise ValueError(f"the class list {path} has no names")
    return names


def _validate_param(key: str, value: Any, rule: Mapping[str, Any], path: Path) -> Any:
    """Check one parameter's type, choices and numeric range.

    Returns the value, coerced to ``float`` when the schema expects a number
    and a plain int was given.

    Raises:
        ValueError: the value's type, choice set or numeric range is wrong;
            the message names ``key`` and ``path``.
    """
    expected_type = rule["type"]
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{path}: parameter {key!r} must be an integer, got {value!r}")
    elif expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{path}: parameter {key!r} must be a number, got {value!r}")
        value = float(value)
    elif expected_type is str:
        if not isinstance(value, str):
            raise ValueError(
                f"{path}: parameter {key!r} must be a string, got {value!r}")
        if rule.get("nonempty") and not value.strip():
            raise ValueError(f"{path}: parameter {key!r} must not be empty")
    elif expected_type is list:
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"{path}: parameter {key!r} must be a non-empty list, got {value!r}")
        choices = rule.get("item_choices")
        if choices:
            bad = [v for v in value if v not in choices]
            if bad:
                raise ValueError(
                    f"{path}: parameter {key!r} has unrecognised value(s) {bad}, "
                    f"expected values from {sorted(choices)}")
        if len(set(value)) != len(value):
            raise ValueError(
                f"{path}: parameter {key!r} has duplicate entries: {value}")

    if "choices" in rule and value not in rule["choices"]:
        raise ValueError(
            f"{path}: parameter {key!r} must be one of {sorted(rule['choices'])}, "
            f"got {value!r}")

    if expected_type in (int, float):
        lo, hi = rule.get("min"), rule.get("max")
        max_exclusive = rule.get("max_exclusive", False)
        if lo is not None and value < lo:
            raise ValueError(f"{path}: parameter {key!r} must be >= {lo}, got {value}")
        if hi is not None:
            if max_exclusive and value >= hi:
                raise ValueError(f"{path}: parameter {key!r} must be < {hi}, got {value}")
            if not max_exclusive and value > hi:
                raise ValueError(f"{path}: parameter {key!r} must be <= {hi}, got {value}")
    return value


def load_params(path: Path) -> dict[str, Any]:
    """Read and strictly validate an augmentation params file.

    The file must be a YAML mapping with a ``params:`` section holding
    exactly the keys in ``PARAM_SCHEMA``, each of the right type and within
    its allowed range. A ``_meta:`` section is expected but not itself
    validated; it is descriptive only.

    Raises:
        OSError: the file cannot be read.
        ValueError: the file is not valid YAML, is not shaped as expected,
            is missing a required parameter, carries an unknown parameter, or
            a parameter's type, choice or range is wrong. Every message names
            the offending key and ``path``.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise OSError(f"cannot read the augmentation params file {path}: {exc}") from exc
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(doc, dict) or "params" not in doc:
        raise ValueError(f"{path} must be a mapping with a top-level 'params' key")
    params = doc["params"]
    if not isinstance(params, dict):
        raise ValueError(f"'params' in {path} must be a mapping of setting names to values")

    missing = sorted(set(PARAM_SCHEMA) - set(params))
    if missing:
        raise ValueError(f"{path} is missing required parameter(s): {missing}")
    unknown = sorted(set(params) - set(PARAM_SCHEMA))
    if unknown:
        raise ValueError(f"{path} has unknown parameter(s): {unknown}")

    return {key: _validate_param(key, params[key], rule, path)
            for key, rule in PARAM_SCHEMA.items()}


def list_image_files(images_dir: Path) -> list[Path]:
    """Return every image file directly under ``images_dir``, sorted by name.

    Only regular files whose extension (case-insensitive) is jpg, jpeg or png
    are returned; directories and any other file type are skipped.
    """
    if not images_dir.is_dir():
        raise OSError(f"the images directory {images_dir} does not exist")
    found = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(found, key=lambda p: p.name)


def discover_pairs(images_dir: Path, labels_dir: Path) -> tuple[dict[str, Path], int]:
    """Pair every source image with its label file by stem.

    Parameters:
        images_dir: directory of source images.
        labels_dir: directory of ``<stem>.txt`` YOLO label files.
    Returns:
        a tuple of (stem -> image path, for every image with a label file),
        and the count of images that have no matching label file.
    Raises:
        OSError: ``labels_dir`` does not exist.
        ValueError: two images share a stem under different extensions, which
            would make the pairing ambiguous.
    """
    if not labels_dir.is_dir():
        raise OSError(f"the labels directory {labels_dir} does not exist")
    pairs: dict[str, Path] = {}
    unpaired = 0
    for image_path in list_image_files(images_dir):
        stem = image_path.stem
        if stem in pairs:
            raise ValueError(
                f"two source images share the stem {stem!r} under different "
                f"extensions ({pairs[stem].name} and {image_path.name}); "
                "the label pairing would be ambiguous"
            )
        label_path = labels_dir / f"{stem}.txt"
        if label_path.is_file():
            pairs[stem] = image_path
        else:
            unpaired += 1
    return pairs, unpaired


def read_label_file(path: Path, num_classes: int) -> list[Box]:
    """Parse one YOLO label file strictly.

    Each non-blank line must be ``cls cx cy w h``: an integer class id and
    four finite floats. A zero-byte or all-blank file parses to an empty list
    (a background image with no boxes), which is valid.

    Raises:
        OSError: the file cannot be read.
        ValueError: a line has the wrong field count, a non-integer class id,
            a class id outside ``[0, num_classes)``, or a non-finite or
            unparsable coordinate. The message names the file and the
            1-based line number.
    """
    try:
        text = path.read_text()
    except OSError as exc:
        raise OSError(f"cannot read label file {path}: {exc}") from exc
    boxes: list[Box] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_no}: expected 5 fields 'cls cx cy w h', got "
                f"{len(parts)} in {raw!r}"
            )
        try:
            cls_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: class id {parts[0]!r} is not an integer"
            ) from exc
        if not 0 <= cls_id < num_classes:
            raise ValueError(
                f"{path}:{line_no}: class id {cls_id} is outside "
                f"[0, {num_classes})"
            )
        try:
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_no}: cannot parse coordinates in {raw!r}"
            ) from exc
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ValueError(
                f"{path}:{line_no}: a non-finite coordinate in {raw!r}"
            )
        boxes.append(Box(cls_id, cx, cy, w, h))
    return boxes


def load_all_labels(stems: Sequence[str], labels_dir: Path,
                     num_classes: int) -> dict[str, list[Box]]:
    """Parse the label file for every stem in ``stems``.

    Raises whatever ``read_label_file`` raises, with the failing file already
    named in the message, so the caller does not need to add context.
    """
    return {stem: read_label_file(labels_dir / f"{stem}.txt", num_classes)
            for stem in stems}


# ── stratified split ────────────────────────────────────────────────────────

def rarity_counts(boxes_by_stem: Mapping[str, list[Box]],
                   num_classes: int) -> dict[int, int]:
    """Count, for each class id, how many images contain at least one box of it."""
    counts = {c: 0 for c in range(num_classes)}
    for boxes in boxes_by_stem.values():
        for cls_id in {b.cls for b in boxes}:
            counts[cls_id] += 1
    return counts


def stratum_for(boxes: Sequence[Box], rarity: Mapping[int, int]) -> str:
    """Return an image's split stratum: its rarest class, or ``"none"``.

    Rarity is the number of images (in the working set) that contain the
    class; ties are broken by the lower class id, so the assignment is
    deterministic given the same rarity table.
    """
    present = {b.cls for b in boxes}
    if not present:
        return "none"
    best = min(present, key=lambda c: (rarity[c], c))
    return f"class_{best:03d}"


def stratified_split(stems: Sequence[str], boxes_by_stem: Mapping[str, list[Box]],
                      num_classes: int, valid_frac: float,
                      test_frac: float, seed: int) -> dict[str, str]:
    """Split source images into train, valid and test, stratified by rarity.

    Within each stratum (images sorted by stem name), the order is shuffled
    with ``random.Random(f"{seed}:{stratum}")`` so the split is deterministic
    given the seed and reproducible across runs. ``n_test`` and ``n_valid``
    are ``round(n * frac)``, each bumped to at least 1 when the stratum holds
    3 or more images, so no stratum with enough images is left with no
    held-out example.

    Raises:
        ValueError: the fractions are negative or would leave no room for a
            training split.
    """
    if valid_frac < 0 or test_frac < 0 or valid_frac + test_frac >= 1.0:
        raise ValueError(
            f"valid_frac {valid_frac} and test_frac {test_frac} must each be "
            ">= 0 and sum to under 1.0, or no image would remain for training"
        )
    rarity = rarity_counts(boxes_by_stem, num_classes)
    groups: dict[str, list[str]] = {}
    for stem in stems:
        groups.setdefault(stratum_for(boxes_by_stem[stem], rarity), []).append(stem)

    pins: dict[str, str] = {}
    for group_key in sorted(groups):
        members = sorted(groups[group_key])
        rng = random.Random(f"{seed}:{group_key}")
        rng.shuffle(members)
        n = len(members)
        n_test = round(n * test_frac)
        n_valid = round(n * valid_frac)
        if n >= 3:
            n_test = n_test or 1
            n_valid = n_valid or 1
        n_test = min(n_test, n)
        n_valid = min(n_valid, n - n_test)
        for i, stem in enumerate(members):
            if i < n_test:
                pins[stem] = "test"
            elif i < n_test + n_valid:
                pins[stem] = "valid"
            else:
                pins[stem] = "train"
    return pins


# ── geometry ─────────────────────────────────────────────────────────────────

def draw_params(rng: np.random.Generator, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Draw one variant's Table 2 parameters, every transform independent.

    Every range comes from ``cfg`` (a validated params dict from
    ``load_params``), never from a module constant, so a params file fully
    controls the augmentation. Uses a ``numpy.random.Generator`` (never
    Python's ``hash()``) so a caller who seeds it deterministically gets a
    reproducible draw.
    """
    rot90_choices = cfg["rot90_choices"]
    return {
        "hflip": bool(rng.random() < cfg["flip_horizontal_p"]),
        "vflip": bool(rng.random() < cfg["flip_vertical_p"]),
        "rot90": rot90_choices[int(rng.integers(0, len(rot90_choices)))],
        "angle_deg": float(rng.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])),
        "shear_x_deg": float(rng.uniform(-cfg["shear_deg_x"], cfg["shear_deg_x"])),
        "shear_y_deg": float(rng.uniform(-cfg["shear_deg_y"], cfg["shear_deg_y"])),
        "hue_shift_deg": float(rng.uniform(-cfg["hue_deg"], cfg["hue_deg"])),
        "sat_jitter": float(rng.uniform(-cfg["saturation_frac"], cfg["saturation_frac"])),
        "exposure_jitter": float(rng.uniform(-cfg["exposure_frac"], cfg["exposure_frac"])),
        "blur_sigma_px": float(rng.uniform(0.0, cfg["blur_sigma_max_px"])),
        "noise_frac": float(rng.uniform(0.0, cfg["noise_pixel_frac_max"])),
    }


def stable_variant_seed(seed: int, stem: str, k: int) -> int:
    """Hash ``(seed, stem, k)`` to a stable 64-bit integer RNG seed.

    Uses ``hashlib.sha256`` rather than Python's ``hash()``, which is salted
    per-process and would make two runs of this tool draw different
    parameters for the same variant.
    """
    digest = hashlib.sha256(f"{seed}:{stem}:{k}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rot90_output_size(w: int, h: int, mode: str) -> tuple[int, int]:
    """Return the (width, height) after a 90-degree-stepped rotation."""
    if mode in ("cw", "ccw"):
        return h, w
    return w, h


def affine_for(params: Mapping[str, Any], w: int, h: int) -> np.ndarray:
    """Build the 3x3 homogeneous matrix for rotation then shear about the centre.

    ``w`` and ``h`` are the dimensions of the image the matrix will act on
    (after any flip and 90-degree rotation already applied), so the centre of
    rotation is that image's own centre. The matrix is a forward point map:
    for a point ``p`` in the input, ``matrix @ [x, y, 1]`` is its position in
    the output, which is also exactly what ``cv2.warpAffine`` expects as
    ``M`` under its default (non-inverse) convention.
    """
    cx, cy = w / 2.0, h / 2.0
    rotation_2x3 = cv2.getRotationMatrix2D((cx, cy), params["angle_deg"], 1.0)
    rotation = np.vstack([rotation_2x3, [0.0, 0.0, 1.0]])
    shx = math.tan(math.radians(params["shear_x_deg"]))
    shy = math.tan(math.radians(params["shear_y_deg"]))
    shear_linear = np.array([[1.0, shx, 0.0], [shy, 1.0, 0.0], [0.0, 0.0, 1.0]])
    to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    from_origin = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    shear = from_origin @ shear_linear @ to_origin
    return shear @ rotation


def apply_geometry(
    img: np.ndarray, params: Mapping[str, Any],
) -> tuple[np.ndarray, Callable[[float, float], tuple[float, float]], tuple[int, int]]:
    """Apply flip, 90-degree rotation, then continuous rotation and shear.

    Returns the transformed image, a ``point_map(x, y) -> (x, y)`` function
    that carries any point from the input image's pixel space to the output
    image's pixel space through the same sequence of transforms, and the
    output image's (width, height).
    """
    h0, w0 = img.shape[:2]

    flip_code: int | None
    if params["hflip"] and params["vflip"]:
        flip_code = -1
    elif params["hflip"]:
        flip_code = 1
    elif params["vflip"]:
        flip_code = 0
    else:
        flip_code = None
    flipped = cv2.flip(img, flip_code) if flip_code is not None else img

    def flip_point(x: float, y: float) -> tuple[float, float]:
        nx = (w0 - x) if params["hflip"] else x
        ny = (h0 - y) if params["vflip"] else y
        return nx, ny

    rot_flags = {
        "none": None, "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "r180": cv2.ROTATE_180,
    }
    mode = params["rot90"]
    flag = rot_flags[mode]
    rotated = cv2.rotate(flipped, flag) if flag is not None else flipped
    w1, h1 = w0, h0
    w2, h2 = rot90_output_size(w1, h1, mode)

    def rot90_point(x: float, y: float) -> tuple[float, float]:
        if mode == "none":
            return x, y
        if mode == "cw":
            return h1 - y, x
        if mode == "ccw":
            return y, w1 - x
        return w1 - x, h1 - y  # "r180"

    matrix = affine_for(params, w2, h2)
    warped = cv2.warpAffine(
        rotated, matrix[:2, :], (w2, h2),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    def affine_point(x: float, y: float) -> tuple[float, float]:
        vec = matrix @ np.array([x, y, 1.0])
        return float(vec[0]), float(vec[1])

    def point_map(x: float, y: float) -> tuple[float, float]:
        x, y = flip_point(x, y)
        x, y = rot90_point(x, y)
        return affine_point(x, y)

    return warped, point_map, (w2, h2)


def transform_boxes(
    boxes: Sequence[Box], point_map: Callable[[float, float], tuple[float, float]],
    in_wh: tuple[int, int], out_wh: tuple[int, int],
    min_box_px: float, min_area_ratio: float,
) -> tuple[list[Box], int]:
    """Carry every box through a geometric transform by its four corners.

    Each corner is mapped through ``point_map``; the axis-aligned box
    enclosing the four mapped corners is clipped to the output frame. A box
    is dropped (not written) when the clipped width or height is under
    ``min_box_px`` pixels, or when the clipped area is under
    ``min_area_ratio`` of the unclipped enclosing area (most of it fell
    outside the frame).

    Returns:
        the surviving boxes (normalized to ``out_wh``), and the count dropped.
    """
    in_w, in_h = in_wh
    out_w, out_h = out_wh
    kept: list[Box] = []
    dropped = 0
    for box in boxes:
        x1 = (box.cx - box.w / 2.0) * in_w
        x2 = (box.cx + box.w / 2.0) * in_w
        y1 = (box.cy - box.h / 2.0) * in_h
        y2 = (box.cy + box.h / 2.0) * in_h
        mapped = [point_map(x, y) for x, y in
                  ((x1, y1), (x2, y1), (x2, y2), (x1, y2))]
        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        tx1, tx2 = min(xs), max(xs)
        ty1, ty2 = min(ys), max(ys)
        unclipped_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)

        cx1, cx2 = max(0.0, tx1), min(float(out_w), tx2)
        cy1, cy2 = max(0.0, ty1), min(float(out_h), ty2)
        clipped_w = max(0.0, cx2 - cx1)
        clipped_h = max(0.0, cy2 - cy1)
        clipped_area = clipped_w * clipped_h

        if (clipped_w < min_box_px or clipped_h < min_box_px
                or unclipped_area <= 0.0
                or clipped_area < min_area_ratio * unclipped_area):
            dropped += 1
            continue

        kept.append(Box(
            box.cls,
            (cx1 + cx2) / 2.0 / out_w,
            (cy1 + cy2) / 2.0 / out_h,
            clipped_w / out_w,
            clipped_h / out_h,
        ))
    return kept, dropped


# ── photometric transforms ──────────────────────────────────────────────────

def photometric(img: np.ndarray, params: Mapping[str, Any],
                 rng: np.random.Generator) -> np.ndarray:
    """Apply hue, saturation, exposure, blur and salt-and-pepper noise.

    ``img`` is RGB, uint8. Hue is shifted in OpenCV's 0-179 hue units (a
    degree shift is halved, per the OpenCV convention) and wrapped; the noise
    draw uses ``rng`` so it stays reproducible under the same seed as the
    geometric parameters. Returns a same-shape, same-dtype RGB image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    # Truncated toward zero so the shift never exceeds the drawn angle: a
    # rounded 15 degrees would become 8 units, which is 16 degrees.
    hue_shift = int(math.trunc(params["hue_shift_deg"] / 2.0))
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * (1.0 + params["sat_jitter"]), 0, 255)
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2].astype(np.float32) * (1.0 + params["exposure_jitter"]), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    sigma = params["blur_sigma_px"]
    if sigma >= BLUR_SIGMA_SKIP_THRESHOLD_PX:
        ksize = int(round(6.0 * sigma + 1.0))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(1, ksize)
        out = cv2.GaussianBlur(out, (ksize, ksize), sigma)

    frac = params["noise_frac"]
    if frac > 0.0:
        height, width = out.shape[:2]
        total_pixels = height * width
        n_affected = int(round(frac * total_pixels))
        if n_affected > 0:
            flat_idx = rng.choice(total_pixels, size=n_affected, replace=False)
            ys, xs = np.unravel_index(flat_idx, (height, width))
            half = n_affected // 2
            out[ys[:half], xs[:half]] = 255
            out[ys[half:], xs[half:]] = 0

    return out


# ── image I/O ────────────────────────────────────────────────────────────────

def load_image_rgb(path: Path) -> np.ndarray:
    """Read an image as EXIF-oriented RGB pixels.

    Raises:
        OSError: the file is missing, unreadable, or an unrecognised or
            truncated image; the message names ``path``.
    """
    try:
        with Image.open(path) as handle:
            oriented = ImageOps.exif_transpose(handle)
            array = np.asarray(oriented.convert("RGB"))
    except (OSError, ValueError) as exc:
        raise OSError(f"cannot read source image {path}: {exc}") from exc
    return array


def save_image_jpg(image_rgb: np.ndarray, path: Path, quality: int) -> None:
    """Write an RGB array as a JPEG at ``quality``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(image_rgb).save(path, format="JPEG", quality=quality)
    except OSError as exc:
        raise OSError(f"cannot write image {path}: {exc}") from exc


def write_label_file(path: Path, boxes: Sequence[Box]) -> None:
    """Write YOLO-format boxes to ``path``; an empty sequence writes an empty file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.cls} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    body = "\n".join(lines) + ("\n" if lines else "")
    try:
        path.write_text(body)
    except OSError as exc:
        raise OSError(f"cannot write label file {path}: {exc}") from exc


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst``, falling back to a byte copy across filesystems."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ── one training image's augmented variants ─────────────────────────────────

def build_variants_for_stem(
    stem: str, image_path: Path, boxes: Sequence[Box], variants: int, seed: int,
    cfg: Mapping[str, Any], out_images_dir: Path, out_labels_dir: Path,
) -> list[dict[str, Any]]:
    """Generate every augmented variant of one training source image.

    The source image is loaded once; each of the ``variants`` outputs draws
    its own parameters from an RNG seeded by ``stable_variant_seed``, so the
    variants of one image differ from each other and the whole run is
    reproducible given ``seed``. ``variants`` and ``seed`` are passed
    separately from ``cfg`` because the CLI may override the params file's
    own values for either.

    Returns:
        one record per variant: name, source stem, drawn parameters, boxes
        kept, boxes dropped, and whether the result is an empty (background)
        image.
    Raises:
        OSError: the source image cannot be read or a variant cannot be
            written; the message names the file.
    """
    img = load_image_rgb(image_path)
    in_h, in_w = img.shape[:2]
    records: list[dict[str, Any]] = []
    for k in range(1, variants + 1):
        rng = np.random.default_rng(stable_variant_seed(seed, stem, k))
        params = draw_params(rng, cfg)
        geom_img, point_map, out_wh = apply_geometry(img, params)
        kept_boxes, dropped = transform_boxes(
            boxes, point_map, (in_w, in_h), out_wh,
            cfg["min_box_px"], cfg["min_visible_area_frac"])
        final_img = photometric(geom_img, params, rng)
        variant_name = f"{stem}{VARIANT_MARKER}{k}"
        save_image_jpg(final_img, out_images_dir / f"{variant_name}{VARIANT_IMAGE_EXT}",
                       cfg["jpeg_quality"])
        write_label_file(out_labels_dir / f"{variant_name}.txt", kept_boxes)
        records.append({
            "name": variant_name, "source": stem, "params": params,
            "boxes_kept": len(kept_boxes), "boxes_dropped": dropped,
            "empty": len(kept_boxes) == 0,
        })
    return records


# ── output-directory management ─────────────────────────────────────────────

def clear_managed_outputs(out_dir: Path, label_dir_name: str) -> None:
    """Clear exactly the entries this tool writes under ``out_dir``.

    Nothing outside ``MANAGED_OUTPUT_ENTRIES``, ``VEGA_LABEL_DIRS`` and
    ``label_dir_name``, and nothing outside ``out_dir``, is touched. An entry
    that holds real image files (the previous ``images/`` folder) is moved to
    ``<out_dir>/_superseded/<stamp>/<name>``, never deleted; label folders,
    metadata files and folders of symlinks hold no image bytes and are removed.
    """
    stamp = _sup.stamp_now()
    for name in dict.fromkeys((*MANAGED_OUTPUT_ENTRIES, *VEGA_LABEL_DIRS, label_dir_name)):
        target = out_dir / name
        if target.is_symlink() or target.exists():
            _sup.supersede_if_images(str(target), base_dir=str(out_dir), stamp=stamp)


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    """Write ``doc`` to ``path`` as indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    except OSError as exc:
        raise OSError(f"cannot write {path}: {exc}") from exc


def pins_by_filename(pins: Mapping[str, str], stem_to_image: Mapping[str, Path],
                     records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Key the split by image FILE NAME, extension included, for VEGA.

    VEGA's detect_dataset.read_source_pins matches pins against image
    basenames with their extension ("Acropora-palmata_748.jpg"). Keyed by
    bare stem, not one pin matches, and VEGA silently draws a fresh random
    split over originals and variants alike, which puts augmented copies of
    test photographs into training.

    Parameters:
        pins: {source stem: split} from stratified_split.
        stem_to_image: {source stem: original image path}.
        records: one per written variant, each with its "name" (a stem).
    Returns:
        {image file name: split}, every variant pinned train.
    """
    by_name = {stem_to_image[stem].name: split for stem, split in pins.items()}
    for record in records:
        by_name[f"{record['name']}{VARIANT_IMAGE_EXT}"] = "train"
    return by_name


def source_of(image_name: str) -> str:
    """The source photograph stem an image file name came from."""
    return Path(image_name).stem.split(VARIANT_MARKER)[0]


def sibling_leak(file_pins: Mapping[str, str]) -> dict[str, Any]:
    """Count held-out images whose source photograph also has an image in training.

    Returns, for valid and test: how many images the set holds, how many have
    a copy of the same photograph in training, and that share. Zero under the
    strict order; the measure of the overlap under the other.
    """
    trained = {source_of(n) for n, split in file_pins.items() if split == "train"}
    report: dict[str, Any] = {}
    for split in ("valid", "test"):
        names = [n for n, sp in file_pins.items() if sp == split]
        shared = sum(1 for n in names if source_of(n) in trained)
        report[split] = {"images": len(names), "with_copy_in_train": shared,
                         "share": round(shared / len(names), 4) if names else 0.0}
    return report


def pins_digest(pins: Mapping[str, str]) -> str:
    """Return a sha256 hex digest over the sorted (name, split) pairs of ``pins``."""
    canonical = json.dumps(sorted(pins.items()), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── verification ─────────────────────────────────────────────────────────────

def verify_dataset(out_dir: Path, pins: Mapping[str, str], variants: int,
                    train_source_count: int, label_dir_name: str,
                    split_after: bool = False) -> list[str]:
    """Check the finished dataset's own internal consistency.

    ``train_source_count`` is the number of photographs that were augmented:
    the training photographs, or every photograph when ``split_after``.
    With ``split_after`` a copy may sit in any set, so that check is skipped.

    Returns a list of problem descriptions; an empty list means every check
    passed.
    """
    problems: list[str] = []
    images_dir = out_dir / "images"
    labels_dir = out_dir / label_dir_name
    image_files = {p.name for p in images_dir.iterdir() if p.is_file()}
    image_stems = {Path(name).stem for name in image_files}
    label_stems = {p.stem for p in labels_dir.iterdir() if p.is_file()}

    if len(image_stems) != len(label_stems) or len(image_stems) != len(pins):
        problems.append(
            f"images/ has {len(image_stems)} files, {label_dir_name}/ has "
            f"{len(label_stems)}, split_pins.json covers {len(pins)}; these must match"
        )
    if image_stems != label_stems:
        problems.append(
            f"{len(image_stems - label_stems)} image(s) lack a label file, "
            f"{len(label_stems - image_stems)} label(s) lack an image"
        )
    if set(pins) != image_files:
        problems.append("split_pins.json does not key exactly the image file names "
                        "written (VEGA matches pins by file name with extension)")

    variant_names = {name for name in image_files if VARIANT_MARKER in name}
    non_train = sorted(n for n in variant_names if pins.get(n) != "train")
    if non_train and not split_after:
        problems.append(
            f"{len(non_train)} variant(s) are not pinned train: {non_train[:5]}"
        )

    expected_variants = train_source_count * variants
    if len(variant_names) != expected_variants:
        problems.append(
            f"expected {expected_variants} variant images "
            f"({train_source_count} augmented sources x {variants} variants), "
            f"found {len(variant_names)}"
        )

    source_stems = image_stems - variant_names
    split_counts = Counter(pins.get(s) for s in source_stems)
    if sum(split_counts.values()) != len(source_stems):
        problems.append("a source image's split assignment is missing or duplicated")

    return problems


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    ``--seed``, ``--variants``, ``--valid_frac`` and ``--test_frac`` default
    to ``None``: when omitted, the params file's own value is used; when
    given, the flag overrides the file for that one run.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source_dir", default=str(DEFAULT_SOURCE_DIR),
                        help="the staged YOLO box dataset to augment")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR),
                        help="where the augmented dataset is written")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH),
                        help="the augmentation params YAML file (Table 2 settings)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the params file's split and augmentation seed")
    parser.add_argument("--variants", type=int, default=None,
                        help="override the params file's augmented siblings per training image")
    parser.add_argument("--valid_frac", type=float, default=None,
                        help="override the params file's validation fraction per stratum")
    parser.add_argument("--test_frac", type=float, default=None,
                        help="override the params file's test fraction per stratum")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="worker processes for augmenting training images")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N source images (sorted by name); for a smoke run")
    parser.add_argument("--force", action="store_true",
                        help="clear this tool's own files from a non-empty out_dir before writing")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: build the split, hardlink originals, augment train, verify."""
    args = build_arg_parser().parse_args(argv)
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    params_path = Path(args.params)

    try:
        cfg = load_params(params_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else cfg["seed"]
    variants = args.variants if args.variants is not None else cfg["variants"]
    valid_frac = args.valid_frac if args.valid_frac is not None else cfg["valid_frac"]
    test_frac = args.test_frac if args.test_frac is not None else cfg["test_frac"]
    label_dir_name = cfg["label_dir"]

    for required in ("images", label_dir_name, "classes.txt", "data.yaml",
                     "LICENSE.md", "ingest_manifest.json"):
        if not (source_dir / required).exists():
            print(f"ERROR: the source dataset {source_dir} is missing {required}",
                 file=sys.stderr)
            return 2

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            print(f"ERROR: {out_dir} is not empty; pass --force to clear this "
                 "tool's own files there and rebuild", file=sys.stderr)
            return 2
        clear_managed_outputs(out_dir, label_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = read_classes(source_dir / "classes.txt")
    num_classes = len(classes)

    try:
        stem_to_image, unpaired_count = discover_pairs(
            source_dir / "images", source_dir / label_dir_name)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stems = sorted(stem_to_image)
    if args.limit is not None:
        stems = stems[:args.limit]
    if not stems:
        print("ERROR: no paired images to build a dataset from", file=sys.stderr)
        return 2

    try:
        boxes_by_stem = load_all_labels(stems, source_dir / label_dir_name, num_classes)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    total_source_boxes = sum(len(v) for v in boxes_by_stem.values())
    print(f"Step 1 complete: {len(stems)} paired images, {unpaired_count} image(s) "
         f"with no label file, {total_source_boxes} source boxes parsed")

    split_after = cfg["split_order"] == SPLIT_AFTER
    if split_after:
        pins = {}
        print("Step 2 complete: split deferred until every photograph is augmented "
              "(split_order after_augmentation)")
    else:
        try:
            pins = stratified_split(stems, boxes_by_stem, num_classes,
                                    valid_frac, test_frac, seed)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"Step 2 complete: split {dict(Counter(pins.values()))}")

    out_images_dir = out_dir / "images"
    out_labels_dir = out_dir / label_dir_name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        link_or_copy(stem_to_image[stem], out_images_dir / stem_to_image[stem].name)
        shutil.copy2(source_dir / label_dir_name / f"{stem}.txt",
                    out_labels_dir / f"{stem}.txt")
    print(f"Step 3 complete: {len(stems)} original images and label files staged")

    train_stems = list(stems) if split_after else [s for s in stems if pins[s] == "train"]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if args.workers > 1 and train_stems:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(build_variants_for_stem, stem, stem_to_image[stem],
                           boxes_by_stem[stem], variants, seed, cfg,
                           out_images_dir, out_labels_dir): stem
                for stem in train_stems
            }
            for future in as_completed(futures):
                stem = futures[future]
                try:
                    records.extend(future.result())
                except Exception as exc:  # noqa: BLE001, a worker's error must not be silent
                    errors.append(f"{stem}: {exc}")
    else:
        for stem in train_stems:
            try:
                records.extend(build_variants_for_stem(
                    stem, stem_to_image[stem], boxes_by_stem[stem],
                    variants, seed, cfg, out_images_dir, out_labels_dir))
            except Exception as exc:  # noqa: BLE001, see above
                errors.append(f"{stem}: {exc}")

    if errors:
        for line in errors:
            print(f"ERROR: {line}", file=sys.stderr)
        print(f"ERROR: {len(errors)} source image(s) failed augmentation; "
             f"{out_dir} holds a partial dataset. Fix the input and rerun with "
             "--force.", file=sys.stderr)
        return 2

    # Workers finish in any order; sort so the manifest is identical per build.
    records.sort(key=lambda r: r["name"])
    boxes_dropped_total = sum(r["boxes_dropped"] for r in records)
    empty_variants = sum(1 for r in records if r["empty"])
    print(f"Step 4 complete: {len(records)} augmented variants written for "
         f"{len(train_stems)} training sources, {boxes_dropped_total} boxes "
         f"dropped by clipping, {empty_variants} variants ended empty")

    shutil.copy2(source_dir / "classes.txt", out_dir / "classes.txt")
    copied_classes = read_classes(out_dir / "classes.txt")
    if copied_classes != classes:
        print("ERROR: the copied classes.txt does not match the source",
             file=sys.stderr)
        return 2
    shutil.copy2(source_dir / "data.yaml", out_dir / "data.yaml")
    shutil.copy2(source_dir / "LICENSE.md", out_dir / "LICENSE.md")
    shutil.copy2(params_path, out_dir / "augment_params.yaml")
    params_sha256 = hashlib.sha256(params_path.read_bytes()).hexdigest()

    if split_after:
        try:
            all_pins, pins = split_images_after(stems, stem_to_image, boxes_by_stem, records,
                                                out_labels_dir, num_classes, valid_frac,
                                                test_frac, seed)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        all_pins = pins_by_filename(pins, stem_to_image, records)
    leak = sibling_leak(all_pins)
    print(f"Step 4b complete: split {dict(Counter(all_pins.values()))} images; test images "
          f"with a copy of the same photograph in training: "
          f"{leak['test']['with_copy_in_train']} of {leak['test']['images']}")
    write_json(out_dir / "split_pins.json", {
        "pins": all_pins, "seed": seed,
        "method": "stratified_by_rarest_class",
        "split_order": cfg["split_order"],
    })

    write_json(out_dir / "augment_manifest.json", build_augment_manifest(
        source_dir=source_dir, classes=classes, stems=stems, pins=pins,
        boxes_by_stem=boxes_by_stem, records=records, cfg=cfg, seed=seed,
        variants=variants, valid_frac=valid_frac, test_frac=test_frac,
        params_path=params_path, params_sha256=params_sha256, file_pins=all_pins,
    ) | {"split_order": cfg["split_order"], "sibling_leak": leak})

    source_manifest = json.loads((source_dir / "ingest_manifest.json").read_text())
    derived_manifest = dict(source_manifest)
    derived_manifest["dataset"] = out_dir.name
    derived_manifest["n_images"] = len(all_pins)
    derived_manifest["n_boxes"] = (
        sum(len(v) for v in boxes_by_stem.values())
        + sum(r["boxes_kept"] for r in records)
    )
    derived_manifest["ingested_at"] = stamp()
    derived_manifest["derived_from"] = str(source_dir)
    derived_manifest["augmentation"] = {
        "method": "offline Table 2 replication (Cordero et al. 2026)",
        "variants_per_train_source": variants,
        "seed": seed,
        "params_file": str(params_path),
        "params_sha256": params_sha256,
    }
    write_json(out_dir / "ingest_manifest.json", derived_manifest)
    print("Step 5 complete: metadata written (classes.txt, data.yaml, LICENSE.md, "
         "augment_params.yaml, split_pins.json, augment_manifest.json, "
         "ingest_manifest.json)")

    problems = verify_dataset(out_dir, all_pins, variants, len(train_stems), label_dir_name,
                              split_after=split_after)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 2
    print(f"Step 6 complete: verification passed, {len(all_pins)} images total "
         f"({len(stems)} originals, {len(records)} augmented variants)")
    return 0


def split_images_after(stems: Sequence[str], stem_to_image: Mapping[str, Path],
                       boxes_by_stem: Mapping[str, list[Box]],
                       records: Sequence[Mapping[str, Any]], labels_dir: Path,
                       num_classes: int, valid_frac: float, test_frac: float,
                       seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Split every image, originals and copies alike, one image at a time.

    Each image is stratified on its own rarest species with the same rule and
    seed as the strict split, so the two datasets differ only in the order of
    splitting and augmenting.

    Returns (file_pins, source_pins): {image file name: split} for VEGA, and
    {source stem: split of its original image} for the manifest's counts.
    Raises ValueError or OSError when a copy's label file cannot be read.
    """
    image_boxes: dict[str, list[Box]] = dict(boxes_by_stem)
    for record in records:
        image_boxes[record["name"]] = read_label_file(
            labels_dir / f"{record['name']}.txt", num_classes)
    image_pins = stratified_split(sorted(image_boxes), image_boxes, num_classes,
                                  valid_frac, test_frac, seed)
    file_pins = {stem_to_image[s].name: image_pins[s] for s in stems}
    for record in records:
        file_pins[f"{record['name']}{VARIANT_IMAGE_EXT}"] = image_pins[record["name"]]
    return file_pins, {s: image_pins[s] for s in stems}


def build_augment_manifest(
    source_dir: Path, classes: Sequence[str], stems: Sequence[str],
    pins: Mapping[str, str], boxes_by_stem: Mapping[str, list[Box]],
    records: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], seed: int,
    variants: int, valid_frac: float, test_frac: float,
    params_path: Path, params_sha256: str, file_pins: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble ``augment_manifest.json``'s content.

    ``file_pins`` is the split exactly as written to split_pins.json (keyed by
    image file name), so the recorded image counts and pins digest describe
    the file VEGA reads.

    Per-class counts are image-presence counts (how many source images in
    that split contain the class), matching the rarity measure the split
    itself is stratified on, not raw box counts. The parameter ranges
    recorded are whatever ``cfg`` (the loaded params file) actually held, not
    a hardcoded copy of Table 2.
    """
    all_pins = dict(file_pins)

    per_split_source: dict[str, int] = dict(Counter(pins.values()))
    per_split_images: dict[str, int] = dict(Counter(all_pins.values()))
    per_split_boxes: dict[str, int] = {split: 0 for split in per_split_source}
    for stem in stems:
        per_split_boxes[pins[stem]] += len(boxes_by_stem[stem])
    for record in records:
        per_split_boxes["train"] += record["boxes_kept"]

    per_class_per_split: dict[str, dict[str, int]] = {
        split: {name: 0 for name in classes} for split in per_split_source
    }
    for stem in stems:
        split = pins[stem]
        for cls_id in {b.cls for b in boxes_by_stem[stem]}:
            per_class_per_split[split][classes[cls_id]] += 1

    return {
        "source_dataset": str(source_dir),
        "created_at": ast_now().isoformat(),
        "seed": seed,
        "variants_per_train_source": variants,
        "valid_frac": valid_frac,
        "test_frac": test_frac,
        "params_file": str(params_path),
        "params_sha256": params_sha256,
        "counts": {
            "source_images_per_split": per_split_source,
            "output_images_per_split": per_split_images,
            "boxes_per_split": per_split_boxes,
            "source_images_per_class_per_split": per_class_per_split,
        },
        "table2_parameter_ranges": {
            "rotation_angle_deg": [-cfg["rotation_deg"], cfg["rotation_deg"]],
            "shear_angle_deg_x": [-cfg["shear_deg_x"], cfg["shear_deg_x"]],
            "shear_angle_deg_y": [-cfg["shear_deg_y"], cfg["shear_deg_y"]],
            "hue_shift_deg": [-cfg["hue_deg"], cfg["hue_deg"]],
            "saturation_jitter": [-cfg["saturation_frac"], cfg["saturation_frac"]],
            "exposure_jitter": [-cfg["exposure_frac"], cfg["exposure_frac"]],
            "blur_sigma_px": [0.0, cfg["blur_sigma_max_px"]],
            "noise_fraction": [0.0, cfg["noise_pixel_frac_max"]],
            "rot90_modes": list(cfg["rot90_choices"]),
            "flip_horizontal_p": cfg["flip_horizontal_p"],
            "flip_vertical_p": cfg["flip_vertical_p"],
        },
        "boxes_dropped_by_clipping": sum(r["boxes_dropped"] for r in records),
        "variants_ended_empty": sum(1 for r in records if r["empty"]),
        "pins_sha256": pins_digest(all_pins),
        "variants": list(records),
    }


if __name__ == "__main__":
    sys.exit(main())
