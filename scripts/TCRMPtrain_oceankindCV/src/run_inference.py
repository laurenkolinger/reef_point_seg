#!/usr/bin/env python3
"""Module: run_inference.py (TCRMPtrain_oceankindCV)

Purpose: run a trained Ultralytics model over a set of images for the Model
Inference step of the orchestrator and write, per image, an overlay in which
every instance is outlined and labelled with its class name and confidence,
plus a machine-readable predictions.json and a manifest.json the gallery
renders.

Inputs:
    --run_dir            training run holding weights/best.pt (or last.pt)
    --source_type        dir: every image under --source_dir
                         unused: TCRMP frames under --clip_dir not listed in
                                 --selected_frames (the step 3 csv), sampled
                         full: every raw frame under --clip_dir
    --codes_csv          optional species code CSV (code,category,name) so a
                         class named by a bare code shows its full name. The
                         split dataset beside --run_dir is read as well, and
                         its names_full table wins over this file, because a
                         staged dataset names its own classes.

Outputs (inside --out_dir):
    overlays/            one overlay per image, written when --save_overlays
    predictions.json     per-detection geometry, written when --save_predictions:
                         polygon_xyn when the model returns masks, box_xyxyn
                         otherwise; read by the review-round seeder
    manifest.json        gallery items, per-image detections with class names,
                         thresholds, skipped counts, render settings, errors and
                         the blind-spot signal
"""

import argparse
import colorsys
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
PTS_VARIANT_MARKER = '_pts.'
PTS_SUFFIX = '_pts'

AST = timezone(timedelta(hours=-4))
FOOTER_STAMP_FORMAT = '%Y-%m-%d %H:%M AST'

OVERLAY_DIRNAME = 'overlays'
MANIFEST_FILENAME = 'manifest.json'
PREDICTIONS_FILENAME = 'predictions.json'
PREDICTIONS_SCHEMA_VERSION = 2
JPEG_QUALITY = 85
PROGRESS_EVERY = 10
RANDOM_SAMPLE_SEED = 0

# Label chips: black backing, per-class colour text, DejaVu Sans Bold.
LABEL_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
LABEL_FONT_SIZE_MAX = 18
LABEL_FONT_SIZE_MIN = 11
LABEL_FONT_DIVISOR = 18          # base size is the shorter image side divided by this
LABEL_CHIP_PAD = 3
LABEL_CHIP_BACKING = (0, 0, 0, 220)
CHIP_NUDGE_ATTEMPTS = 8          # positions tried below a chip, then above it, to clear an earlier chip
ELLIPSIS = '...'
UNKNOWN_CLASS_LABEL = 'unknown class'
# The same binomial rule evaluate_run.py uses: Genus then one or two lowercase
# epithets joined by a dash, an underscore or a space, shown with spaces.
BINOMIAL_TOKEN_RE = re.compile(r'[A-Z][a-z]+(?:[ _-][a-z]+){1,2}')
BINOMIAL_SEPARATOR_RE = re.compile(r'[_-]')
NO_CONFIDENCE_LABEL = 'n/a'
UNKNOWN_CLASS_COLOUR = (160, 160, 160)
HUE_STEP_DEGREES = 137           # spreads consecutive class ids around the colour wheel
COLOUR_SATURATION = 0.78
COLOUR_VALUE = 0.95
BOX_OUTLINE_WIDTH = 2
OPAQUE = 255

# Footer strip: pink text on a translucent black bar.
FOOTER_HEIGHT = 22
FOOTER_FONT_SIZE = 12
FOOTER_FONT_SIZE_MIN = 9         # the footer shrinks to this before it cuts the line
FOOTER_TEXT_COLOUR = (255, 61, 139, 255)
FOOTER_BACKING = (0, 0, 0, 180)
FOOTER_SEPARATOR = ' · '
FOOTER_TEXT_X = 8
FOOTER_TEXT_Y = 3
FOOTER_TRUNCATION_PREFIX = '… '

# Blind-spot signal thresholds.
BLIND_SPOT_LOW_CONF = 0.35
BLIND_SPOT_AVG_CONF = 0.4
BLIND_SPOT_LOW_FRACTION = 0.5
BLIND_SPOT_NO_DETECTION_FRACTION = 0.2

# Detection bookkeeping (counter keys shared by predictions.json and the manifest).
STAT_NO_GEOMETRY = 'no_geometry'
STAT_NO_CONFIDENCE = 'no_confidence'
STAT_DEGENERATE_POLYGON = 'degenerate_polygon_written_as_box'
STAT_POLYGON_WRITTEN = 'polygon_written'
STAT_BOX_WRITTEN = 'box_written'
SKIP_REASONS = (STAT_NO_GEOMETRY, STAT_NO_CONFIDENCE)
MIN_POLYGON_POINTS = 3
COORD_DECIMALS = 6
CONFIDENCE_DECIMALS = 4
RENDER_TOTAL_KEYS = ('instances_with_mask', 'instances_box_only',
                     'instances_without_geometry', 'chips_overlapping')
IGNORED_FLAG_SAVE_CROPS = 'save_crops'


def ast_now():
    """Return the current time in Atlantic Standard Time as ISO-8601 with offset, seconds precision."""
    return datetime.now(AST).isoformat(timespec='seconds')


def ast_stamp():
    """Return the current Atlantic Standard Time formatted for the overlay footer."""
    return datetime.now(AST).strftime(FOOTER_STAMP_FORMAT)


def arg_parse():
    """Parse the command line. Returns an argparse.Namespace."""
    p = argparse.ArgumentParser(description='TCRMP Model Inference (internal step 8)')
    p.add_argument('--run_dir', required=True, help='training run holding weights/best.pt')
    p.add_argument('--out_dir', required=True, help='destination for overlays, predictions and manifest')
    p.add_argument('--source_type', required=True, choices=['dir', 'unused', 'full'])
    p.add_argument('--source_dir', default='', help='required when source_type=dir')
    p.add_argument('--clip_dir', default='', help='TCRMP_clip root (unused/full)')
    p.add_argument('--selected_frames', default='',
                   help='step 3 selected_frames.csv (for unused-frame exclusion)')
    p.add_argument('--sample_count', type=int, default=100)
    p.add_argument('--sample_mode', default='random', choices=['random', 'systematic'],
                   help='random draws with a fixed seed, so the same pool gives the same sample')
    p.add_argument('--include_pts_variant', action='store_true',
                   help='include the _pts.jpg overlay variants alongside raw frames')
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--save_overlays', action='store_true')
    p.add_argument('--save_crops', action='store_true',
                   help='Accepted for compatibility with the Model Inference panel; no crop images '
                        'are written and the manifest lists the flag under ignored_flags.')
    p.add_argument('--save_predictions', action='store_true',
                   help='Also write predictions.json with one entry per detection: a normalized '
                        'polygon when the model returns masks, normalized box corners otherwise. '
                        'The Refine loop seeder reads this file.')
    p.add_argument('--device', default='0')
    p.add_argument('--codes_csv', default='',
                   help='Optional species code CSV (code,category,name). When given, a class named '
                        'by a bare code is shown with its full name on the overlay and in the manifest.')

    # Rendering controls, all user-settable from the Model Inference panel.
    p.add_argument('--mask_alpha', type=float, default=0.45,
                   help='fill opacity for every instance, mask polygon or box '
                        '(0.0 = outline only, 1.0 = solid)')
    p.add_argument('--draw_boxes', action='store_true',
                   help='also outline the box around each mask polygon; a box-only instance is '
                        'always outlined, so this adds nothing for a detector')
    p.add_argument('--run_name', default='',
                   help='human-friendly name stamped in the overlay footer')
    p.add_argument('--project_name', default='',
                   help='project name stamped in the overlay footer')
    return p.parse_args()


# ---------------------------------------------------------------- sources


def _find_best_weights(run_dir):
    """Return the path to best.pt, else last.pt, under run_dir/weights. Raises FileNotFoundError."""
    for cand in ('best.pt', 'last.pt'):
        p = os.path.join(run_dir, 'weights', cand)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No weights in {os.path.join(run_dir, "weights")}: expected best.pt or last.pt')


def _load_selected_basenames(path):
    """Return the set of basenames already in step 3's selected_frames.csv.

    An empty path means no file was supplied and yields an empty set. A path
    that was supplied must exist and be readable: a silent empty set would make
    every frame count as unused, so a missing path raises FileNotFoundError and
    a path that is not a regular readable file raises OSError, both naming the path.
    """
    used = set()
    if not path:
        return used
    if not os.path.exists(path):
        raise FileNotFoundError(f'selected frames csv not found: {path!r}')
    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        raise OSError(f'selected frames csv is not a readable file: {path!r}')
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                bn = (row.get('basename') or row.get('filename') or '').strip()
                if bn:
                    used.add(bn)
    except (OSError, csv.Error) as e:
        raise OSError(f'cannot read selected frames csv {path}: {e}') from e
    return used


def _walk_images(root, include_pts):
    """Yield absolute paths to images under root. Skips _pts.jpg variants unless asked."""
    if not root or not os.path.isdir(root):
        return
    for dp, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            if not include_pts and PTS_VARIANT_MARKER in f.lower():
                continue
            yield os.path.join(dp, f)


def _frame_basename(path):
    """Basename without extension and without a trailing _pts, to match the step 3 csv."""
    b = os.path.splitext(os.path.basename(path))[0]
    return b[:-len(PTS_SUFFIX)] if b.lower().endswith(PTS_SUFFIX) else b


def collect_source(args):
    """Return the sorted list of absolute image paths for args.source_type.

    dir: every image under source_dir. full: every raw frame under clip_dir.
    unused: frames under clip_dir absent from selected_frames, sampled to
    sample_count either at random (fixed seed) or every Nth. Raises ValueError
    when the named folder does not exist.
    """
    if args.source_type == 'dir':
        if not args.source_dir or not os.path.isdir(args.source_dir):
            raise ValueError(f'source_dir not found: {args.source_dir}')
        return sorted(_walk_images(args.source_dir, include_pts=args.include_pts_variant))

    if args.source_type not in ('unused', 'full'):
        return []
    if not args.clip_dir or not os.path.isdir(args.clip_dir):
        raise ValueError(f'clip_dir not found: {args.clip_dir}')
    all_imgs = sorted(_walk_images(args.clip_dir, include_pts=args.include_pts_variant))
    if args.source_type == 'full':
        return all_imgs

    used_basenames = _load_selected_basenames(args.selected_frames)
    candidates = [p for p in all_imgs if _frame_basename(p) not in used_basenames]
    if not candidates:
        return []
    n = min(args.sample_count, len(candidates))
    if args.sample_mode == 'random':
        random.seed(RANDOM_SAMPLE_SEED)
        return sorted(random.sample(candidates, n))
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


# ---------------------------------------------------------------- class names


def normalise_class_names(names):
    """Return the model's class map as ({int id: str name}, dropped_keys).

    Accepts the dict ultralytics returns, a dict with string-digit keys, or a
    list indexed by class id. Keys that are not integers cannot address a
    class id and are returned in dropped_keys so the caller can report them.
    Anything else yields an empty map.
    """
    if isinstance(names, dict):
        out, dropped = {}, []
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                dropped.append(k)
        return out, dropped
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}, []
    return {}, []


def _lookup_name(class_names, cls_id):
    """Return the raw name for cls_id from a dict or list, or None when absent or blank."""
    name = None
    if isinstance(class_names, dict):
        name = class_names.get(cls_id, class_names.get(str(cls_id)))
    elif isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
        name = class_names[cls_id]
    if name is None or not str(name).strip():
        return None
    return str(name)


def class_label(class_names, cls_id):
    """Return the class label the model was trained with, never a bare number.

    A None id reads UNKNOWN_CLASS_LABEL; an id absent from the map (or mapped
    to a blank) reads 'unnamed class N'.
    """
    if cls_id is None:
        return UNKNOWN_CLASS_LABEL
    name = _lookup_name(class_names, cls_id)
    return name if name is not None else f'unnamed class {cls_id}'


def pretty_label(label, code_lookup):
    """Human form of a raw class label: 'Full name (CODE)' for a known code; a token
    written as a binomial (Acropora-cervicornis, Porites_astreoides) with its separators
    shown as spaces, the rule the evaluation report uses; anything else verbatim.
    The raw label is what the manifest and predictions.json keep."""
    if code_lookup and label in code_lookup:
        return f'{code_lookup[label]} ({label})'
    if BINOMIAL_TOKEN_RE.fullmatch(label):
        return BINOMIAL_SEPARATOR_RE.sub(' ', label)
    return label


def display_name(class_names, cls_id, code_lookup):
    """Return the label to show a person for a class id (see pretty_label)."""
    return pretty_label(class_label(class_names, cls_id), code_lookup)


def load_code_lookup(path):
    """Read a species code CSV (code,category,name) into {code: full name}.

    Raises FileNotFoundError when the file is absent and OSError when it cannot be read.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'codes csv not found: {path}')
    lookup = {}
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                code = (row.get('code') or row.get('species_code') or '').strip()
                name = (row.get('name') or row.get('species_name') or '').strip()
                if code and name:
                    lookup[code] = name
    except (OSError, csv.Error) as e:
        raise OSError(f'cannot read codes csv {path}: {e}') from e
    return lookup


def _dataset_dir_for_run(run_dir):
    """The split dataset a training run was built on: dataset/ beside its runs/ parent.

    The same two-step walk as _modellab/labruns.py::dataset_for_run, kept here
    so this script runs on its own, with nothing but its own folder on the
    import path, the way the orchestrator spawns it.

    Parameters:
        run_dir: the training run folder, or '' / None when there is none.
    Returns:
        The dataset folder holding data.yaml, or '' when there is none.
    """
    if not run_dir:
        return ''
    parent = os.path.dirname(os.path.dirname(os.path.abspath(run_dir)))
    dataset = os.path.join(parent, 'dataset')
    if os.path.isdir(dataset) and os.path.isfile(os.path.join(dataset, 'data.yaml')):
        return dataset
    return ''


def _class_map_of(value, path, key):
    """One data.yaml class map as {class id: value}, from a mapping or a list.

    Parameters:
        value: the raw value read from the document, or None when absent.
        path: the file it came from, named in every error.
        key: the key it was read from, named in every error.
    Returns:
        dict of integer class id to the raw value; empty when value is None.
    Raises:
        ValueError when value is neither a mapping nor a list, or when a
        mapping key is not an integer.
    """
    if value is None:
        return {}
    if isinstance(value, list):
        return dict(enumerate(value))
    if not isinstance(value, dict):
        raise ValueError(f'{path} key {key} is a {type(value).__name__}, '
                         'not a mapping of class id to value or a list in class-id order')
    out = {}
    for raw_id, item in value.items():
        try:
            cid = int(str(raw_id).strip())
        except ValueError:
            raise ValueError(f'{path} key {key} has class id {raw_id!r}, which is not an integer') from None
        out[cid] = item
    return out


def names_full_for_run(run_dir):
    """{class token: full name} from the split dataset beside a training run.

    A dataset staged with a names table writes names_full (class id to full
    species name) beside names (class id to class token) in its data.yaml;
    the two are joined on the class id. The evaluation report reads the same
    two keys the same way (evaluate_run.names_full_from_data_yaml), and the
    two readers are pinned against each other by the inference test suite.

    A damaged or unreadable data.yaml is reported on the console and names
    nothing, because a name for the chip is optional context and must never
    stop an inference run.

    Parameters:
        run_dir: the training run folder, or '' / None when there is none.
    Returns:
        {token: full name}; empty when there is no dataset, no names_full, or
        nothing readable.
    """
    dataset = _dataset_dir_for_run(run_dir)
    if not dataset:
        return {}
    path = os.path.join(dataset, 'data.yaml')
    try:
        with open(path, encoding='utf-8') as fh:
            doc = yaml.safe_load(fh)
        if doc is None:
            doc = {}
        if not isinstance(doc, dict):
            raise ValueError(f'{path} does not hold a mapping, so it names no classes')
        names = _class_map_of(doc.get('names'), path, 'names')
        full = _class_map_of(doc.get('names_full'), path, 'names_full')
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f'[infer] WARNING: the dataset at {path} could not be read for class names, '
              f'so class tokens are shown as they are: {exc}')
        return {}
    joined = {}
    for cid, value in full.items():
        token = str(names.get(cid, '')).strip()
        name = str(value).strip()
        if token and name:
            joined[token] = name
    return joined


def build_code_lookup(codes_csv, run_dir):
    """{class token: full name} for the overlay chips and the manifest, from both sources.

    The optional species code CSV (--codes_csv) is read first and the split
    dataset beside the run is merged over it, because a staged dataset names
    its own classes and a token such as EPST_Dark is in no species code table.

    Parameters:
        codes_csv: path to the code CSV, or '' when none was asked for.
        run_dir: the training run folder, whose sibling dataset may name the classes.
    Returns:
        {token: full name}, or None when neither source names a class, which
        is what the renderers read as "show every label as it is".
    Raises:
        FileNotFoundError when codes_csv is given and is not a file.
        OSError when codes_csv is given and cannot be read.
    """
    lookup = {}
    if codes_csv:
        lookup.update(load_code_lookup(codes_csv))
        print(f'[infer] species codes: {len(lookup)} names from {codes_csv}')
    dataset_names = names_full_for_run(run_dir)
    if dataset_names:
        lookup.update(dataset_names)
        print(f"[infer] species names: {len(dataset_names)} from the run's dataset")
    return lookup or None


# ---------------------------------------------------------------- result flattening


def _as_numpy(x):
    """Return x as a numpy array (torch tensors moved to the cpu), or None when x is None."""
    if x is None:
        return None
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return np.asarray(x)


def _polygon_list(seq):
    """Return a list of (N, 2) float arrays from masks.xy or masks.xyn, or None when absent."""
    if seq is None:
        return None
    try:
        return [np.asarray(_as_numpy(p), dtype=float).reshape(-1, 2) for p in seq]
    except (TypeError, ValueError):
        return None


def _length(x):
    """len(x) for anything sized, else 0."""
    try:
        return len(x)
    except TypeError:
        return 0


def _at(seq, i):
    """seq[i] when seq holds an index i, else None."""
    if seq is None or i >= _length(seq):
        return None
    return seq[i]


def instance_rows(result):
    """Flatten one ultralytics result into plain per-instance dicts.

    Parameters:
        result: object with .boxes (cls, conf, xyxy, xyxyn) and optional .masks (xy, xyn);
            arrays may be torch tensors, numpy arrays or lists, and any may be missing.
    Returns:
        list of dicts with keys cls (int or None), conf (float or None), xyxy and xyxyn
        (4-float lists or None), polygon_xy and polygon_xyn ((N, 2) arrays or None), and
        polygon_malformed (True when a masks object was present but unreadable).
        The row count is the longest array present, so a malformed result with
        mismatched arrays yields rows with None in the missing slots.
    """
    boxes = getattr(result, 'boxes', None)
    masks = getattr(result, 'masks', None)
    cls = _as_numpy(getattr(boxes, 'cls', None))
    conf = _as_numpy(getattr(boxes, 'conf', None))
    xyxy = _as_numpy(getattr(boxes, 'xyxy', None))
    xyxyn = _as_numpy(getattr(boxes, 'xyxyn', None))
    raw_xy = getattr(masks, 'xy', None)
    raw_xyn = getattr(masks, 'xyn', None)
    poly_xy = _polygon_list(raw_xy)
    poly_xyn = _polygon_list(raw_xyn)
    # masks were present but could not be read as polygons: every row records it
    polygons_malformed = (raw_xy is not None and poly_xy is None) or (raw_xyn is not None and poly_xyn is None)
    n = max(_length(a) for a in (cls, conf, xyxy, xyxyn, poly_xy, poly_xyn))
    rows = []
    for i in range(n):
        c = _at(cls, i)
        cf = _at(conf, i)
        box = _at(xyxy, i)
        boxn = _at(xyxyn, i)
        rows.append({
            'cls': None if c is None else int(float(c)),
            'conf': None if cf is None else float(cf),
            'xyxy': None if box is None else [float(v) for v in box],
            'xyxyn': None if boxn is None else [float(v) for v in boxn],
            'polygon_xy': _at(poly_xy, i),
            'polygon_xyn': _at(poly_xyn, i),
            'polygon_malformed': polygons_malformed,
        })
    return rows


# ---------------------------------------------------------------- predictions.json


def new_detection_stats():
    """Return a fresh counter for the detection bookkeeping keys."""
    return Counter()


def skipped_total(stats):
    """Number of detections not written to predictions.json, for any reason."""
    return sum(stats[r] for r in SKIP_REASONS)


def _prediction_entry(row, class_names, stats):
    """Build one predictions.json detection from an instance row, or None when skipped.

    A polygon with at least MIN_POLYGON_POINTS points is written as polygon_xyn,
    exactly the shape the review-round seeder reads. Otherwise the normalized box
    corners are written as box_xyxyn, and when a polygon existed but was too short
    or unreadable that fallback is counted under STAT_DEGENERATE_POLYGON. A row
    with no confidence or no geometry is counted in stats and not written.
    """
    if row['conf'] is None:
        stats[STAT_NO_CONFIDENCE] += 1
        return None
    entry = {
        'class_id': row['cls'],
        'class': class_label(class_names, row['cls']),
        'confidence': round(row['conf'], CONFIDENCE_DECIMALS),
    }
    poly = row['polygon_xyn']
    if poly is not None and len(poly) >= MIN_POLYGON_POINTS:
        entry['polygon_xyn'] = [round(float(v), COORD_DECIMALS) for pt in poly for v in pt]
        stats[STAT_POLYGON_WRITTEN] += 1
        return entry
    if row['xyxyn'] is not None:
        if poly is not None or row.get('polygon_malformed'):
            stats[STAT_DEGENERATE_POLYGON] += 1
        entry['box_xyxyn'] = [round(float(v), COORD_DECIMALS) for v in row['xyxyn']]
        stats[STAT_BOX_WRITTEN] += 1
        return entry
    stats[STAT_NO_GEOMETRY] += 1
    return None


def build_prediction_items(results_by_file, class_names, stats=None):
    """Turn (filename, raw_abs, ultralytics_result) triples into predictions.json items.

    Parameters:
        results_by_file: iterable of (filename, absolute source path, result).
        class_names: the model's class map.
        stats: optional Counter from new_detection_stats(); receives the skip and
            geometry counts so the caller can report them.
    Returns:
        list of {filename, raw, width, height, detections}; zero-detection frames
        keep an empty detections list.
    """
    if stats is None:
        stats = new_detection_stats()
    items = []
    for filename, raw_abs, r in results_by_file:
        h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
        dets = []
        for row in instance_rows(r):
            entry = _prediction_entry(row, class_names, stats)
            if entry is not None:
                dets.append(entry)
        items.append({'filename': filename, 'raw': raw_abs,
                      'width': w, 'height': h, 'detections': dets})
    return items


# ---------------------------------------------------------------- overlay rendering


def class_colour(cls_id):
    """Stable per-class RGB colour: the same hue for a class id on every image; grey for None."""
    if cls_id is None:
        return UNKNOWN_CLASS_COLOUR
    h = ((int(cls_id) * HUE_STEP_DEGREES) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, COLOUR_SATURATION, COLOUR_VALUE)
    return int(r * OPAQUE), int(g * OPAQUE), int(b * OPAQUE)


@lru_cache(maxsize=None)
def _load_font(size):
    """Load the label font at a pixel size, falling back to Pillow's default face when the file is absent."""
    try:
        return ImageFont.truetype(LABEL_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _text_box(font, text):
    """Return (left, top, right, bottom) of text drawn at the origin with font."""
    return font.getbbox(text)


def _base_font_size(width, height):
    """Label size scaled to the image: shorter side over LABEL_FONT_DIVISOR, clamped to the min and max."""
    return max(LABEL_FONT_SIZE_MIN, min(LABEL_FONT_SIZE_MAX, round(min(width, height) / LABEL_FONT_DIVISOR)))


def _fit_label(text, max_width, base_size):
    """Return (font, shown_text, size) so the label fits in max_width pixels.

    Shrinks from base_size down to LABEL_FONT_SIZE_MIN; if the text still does
    not fit at the smallest size, it is elided with ELLIPSIS instead of overflowing.
    """
    for size in range(base_size, LABEL_FONT_SIZE_MIN - 1, -1):
        font = _load_font(size)
        if _text_width(font, text) <= max_width:
            return font, text, size
    font = _load_font(LABEL_FONT_SIZE_MIN)
    shown = text
    while len(shown) > 1 and _text_width(font, shown + ELLIPSIS) > max_width:
        shown = shown[:-1]
    return font, shown + ELLIPSIS, LABEL_FONT_SIZE_MIN


def _rects_overlap(a, b):
    """True when two (x0, y0, x1, y1) rectangles share any area."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _collides(rect, occupied):
    """True when rect overlaps any rectangle already placed."""
    return any(_rects_overlap(rect, o) for o in occupied)


def _nudge_positions(y, chip_h, max_y):
    """Alternative top edges for a chip, one chip height apart: downward from y to max_y
    (at most CHIP_NUDGE_ATTEMPTS positions), then upward from y to 0 until the list holds
    twice CHIP_NUDGE_ATTEMPTS in all, so the upward run fills whatever the downward run left."""
    positions = []
    cy = y
    while cy < max_y and len(positions) < CHIP_NUDGE_ATTEMPTS:
        cy = min(cy + chip_h, max_y)
        positions.append(cy)
    cy = y
    while cy > 0 and len(positions) < 2 * CHIP_NUDGE_ATTEMPTS:
        cy = max(cy - chip_h, 0)
        positions.append(cy)
    return positions


def _place_chip(anchor, chip_w, chip_h, width, height, occupied):
    """Choose where a chip of chip_w by chip_h goes for an instance whose top-left is anchor.

    The chip sits just above the anchor, or just inside it when the anchor is at
    the top edge; it is kept inside the image and above the footer. When it
    overlaps an earlier chip it moves down a chip height at a time, and when the
    footer line stops it moving down it moves up instead.
    Returns ((x0, y0, x1, y1), overlapped) where overlapped is True only when no
    clear position was found, in which case the chip stays at its first position.
    """
    ax, ay = anchor
    max_x = max(0, width - chip_w)
    max_y = max(0, height - FOOTER_HEIGHT - chip_h)
    x = min(max(int(ax), 0), max_x)
    y = int(ay) - chip_h
    if y < 0:
        y = int(ay)
    y = min(max(y, 0), max_y)
    first = (x, y, x + chip_w, y + chip_h)
    if not _collides(first, occupied):
        return first, False
    for cy in _nudge_positions(y, chip_h, max_y):
        rect = (x, cy, x + chip_w, cy + chip_h)
        if not _collides(rect, occupied):
            return rect, False
    return first, True


def _draw_label_chip(draw, text, font, anchor, size, occupied, colour):
    """Draw one label chip (black backing, coloured text) and return (rect, overlapped)."""
    left, top, right, bottom = _text_box(font, text)
    chip_w = right - left + 2 * LABEL_CHIP_PAD
    chip_h = bottom - top + 2 * LABEL_CHIP_PAD
    rect, overlapped = _place_chip(anchor, chip_w, chip_h, size[0], size[1], occupied)
    draw.rectangle(rect, fill=LABEL_CHIP_BACKING)
    draw.text((rect[0] + LABEL_CHIP_PAD - left, rect[1] + LABEL_CHIP_PAD - top),
              text, fill=(*colour, OPAQUE), font=font)
    return rect, overlapped


def _draw_instance_geometry(draw, row, colour, alpha_255, draw_boxes):
    """Draw one instance's fill and outline on the RGBA overlay.

    A mask polygon is filled at alpha_255 and outlined; its box is outlined too
    when draw_boxes is set. A box-only instance is filled at alpha_255 and always
    outlined, since the box is its only geometry.
    Returns (anchor_xy, kind) where kind is 'mask', 'box', or None when the row
    has no geometry to draw.
    """
    poly = row['polygon_xy']
    box = row['xyxy']
    if poly is not None and len(poly) >= MIN_POLYGON_POINTS:
        pts = [(float(x), float(y)) for x, y in poly]
        draw.polygon(pts, fill=(*colour, alpha_255), outline=(*colour, OPAQUE))
        if draw_boxes and box is not None:
            draw.rectangle(box, outline=(*colour, OPAQUE), width=BOX_OUTLINE_WIDTH)
        return (min(p[0] for p in pts), min(p[1] for p in pts)), 'mask'
    if box is not None:
        fill = (*colour, alpha_255) if alpha_255 else None
        draw.rectangle(box, fill=fill, outline=(*colour, OPAQUE), width=BOX_OUTLINE_WIDTH)
        return (box[0], box[1]), 'box'
    return None, None


def _confidence_text(conf):
    """Confidence to two decimals, or the placeholder when the model gave none."""
    return NO_CONFIDENCE_LABEL if conf is None else f'{conf:.2f}'


def _text_width(font, text):
    """Rendered width of text in pixels."""
    left, _, right, _ = _text_box(font, text)
    return right - left


def _printable(text):
    """The text with every unprintable character replaced by '?'."""
    return ''.join(ch if ch.isprintable() else '?' for ch in text)


def fit_footer(parts, width):
    """Join the footer parts and fit them to the image width. Returns (text, font).

    parts run from least to most important (project, run name, timestamp,
    filename). The whole line is tried from FOOTER_FONT_SIZE down to
    FOOTER_FONT_SIZE_MIN; when it still does not fit, whole parts are dropped
    from the front, one at a time, and the sizes tried again, so a part is never
    cut mid-token. Only when the last part alone is too wide are its leading
    characters cut. The ellipsis prefix marks any drop or cut.
    """
    kept = [_printable(p) for p in parts if p]
    max_width = width - 2 * FOOTER_TEXT_X
    for dropped in range(len(kept)):
        prefix = FOOTER_TRUNCATION_PREFIX if dropped else ''
        text = prefix + FOOTER_SEPARATOR.join(kept[dropped:])
        for size in range(FOOTER_FONT_SIZE, FOOTER_FONT_SIZE_MIN - 1, -1):
            font = _load_font(size)
            if _text_width(font, text) <= max_width:
                return text, font
    font = _load_font(FOOTER_FONT_SIZE_MIN)
    text = kept[-1] if kept else ''
    while len(text) > 1 and _text_width(font, FOOTER_TRUNCATION_PREFIX + text) > max_width:
        text = text[1:]
    return FOOTER_TRUNCATION_PREFIX + text, font


def _paste_footer(image_rgba, text, font):
    """Paste the pink-on-black footer strip along the bottom edge of an RGBA image."""
    width, height = image_rgba.size
    footer = Image.new('RGBA', (width, FOOTER_HEIGHT), FOOTER_BACKING)
    ImageDraw.Draw(footer).text((FOOTER_TEXT_X, FOOTER_TEXT_Y), text, fill=FOOTER_TEXT_COLOUR, font=font)
    image_rgba.paste(footer, (0, height - FOOTER_HEIGHT), footer)


def _label_instances(draw, rows, anchors, class_names, code_lookup, size, info):
    """Draw one label chip per instance that has geometry, after all fills, so no fill covers a chip.

    Appends each chip's rect, shown text and font size to info and counts the
    chips that could not be moved clear of an earlier one.
    """
    width, height = size
    base_size = _base_font_size(width, height)
    occupied = []
    for row, (anchor, colour) in zip(rows, anchors):
        if anchor is None:
            continue
        text = f'{display_name(class_names, row["cls"], code_lookup)} {_confidence_text(row["conf"])}'
        font, shown, font_size = _fit_label(text, width - 2 * LABEL_CHIP_PAD, base_size)
        rect, overlapped = _draw_label_chip(draw, shown, font, anchor, size, occupied, colour)
        occupied.append(rect)
        info['chips'].append(rect)
        info['labels'].append(shown)
        info['font_sizes'].append(font_size)
        info['chips_overlapping'] += int(overlapped)


def render_overlay(img_path, result, class_names, *, fill_alpha, draw_boxes,
                   run_name, project_name, timestamp, code_lookup=None):
    """Render one image's overlay: every instance filled, outlined and labelled.

    Parameters:
        img_path: source image; raises OSError naming the path when it cannot be opened.
        result: one ultralytics result (boxes, optional masks).
        class_names: the model's class map.
        fill_alpha: fill opacity 0.0 to 1.0 applied to mask polygons and to boxes alike.
        draw_boxes: outline the box around each mask polygon as well.
        run_name, project_name, timestamp: footer text.
        code_lookup: optional {code: full name} so bare codes show their full name.
    Returns:
        (PIL RGB image, info) where info holds chips (one rect per labelled
        instance), labels (the text drawn), font_sizes, instances_with_mask,
        instances_box_only, instances_without_geometry, chips_overlapping and
        draw_boxes_had_effect.
    """
    try:
        base = Image.open(img_path).convert('RGB')
    except (OSError, ValueError) as e:
        raise OSError(f'cannot open image for overlay: {img_path}: {e}') from e
    width, height = base.size
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha_255 = int(max(0.0, min(1.0, fill_alpha)) * OPAQUE)
    info = {key: 0 for key in RENDER_TOTAL_KEYS}
    info.update({'chips': [], 'labels': [], 'font_sizes': []})

    rows = instance_rows(result)
    anchors = []
    for row in rows:
        colour = class_colour(row['cls'])
        anchor, kind = _draw_instance_geometry(draw, row, colour, alpha_255, draw_boxes)
        anchors.append((anchor, colour))
        if kind == 'mask':
            info['instances_with_mask'] += 1
        elif kind == 'box':
            info['instances_box_only'] += 1
        else:
            info['instances_without_geometry'] += 1
    _label_instances(draw, rows, anchors, class_names, code_lookup, (width, height), info)
    info['draw_boxes_had_effect'] = bool(draw_boxes and info['instances_with_mask'])

    out = Image.alpha_composite(base.convert('RGBA'), overlay)
    text, font = fit_footer([project_name, run_name, timestamp, os.path.basename(img_path)], width)
    _paste_footer(out, text, font)
    return out.convert('RGB'), info


# ---------------------------------------------------------------- inference loop


def _manifest_detections(rows, class_names, code_lookup):
    """Per-detection manifest rows: class label, class id, confidence, and the full name when a code resolves."""
    dets = []
    for row in rows:
        label = class_label(class_names, row['cls'])
        det = {'class': label, 'class_id': row['cls'], 'confidence': row['conf']}
        if code_lookup and label in code_lookup:
            det['class_full_name'] = code_lookup[label]
        dets.append(det)
    return dets


def _empty_summary():
    """The accumulator infer_images fills: manifest items, prediction items, stats and totals."""
    return {
        'items': [], 'pred_items': [], 'errors': [],
        'confidences_by_class': {}, 'total_predictions': 0, 'with_any': 0,
        'render_totals': {key: 0 for key in RENDER_TOTAL_KEYS}, 'wall_seconds': 0.0,
    }


def _infer_one(model, src, args, class_names, code_lookup, summary, stats):
    """Predict one image, render its overlay when asked, and append its rows to summary."""
    results = model.predict(src, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            device=args.device, save=False, verbose=False)
    if not results:
        return
    r = results[0]
    rows = instance_rows(r)
    filename = os.path.basename(src)

    rel_overlay = ''
    if args.save_overlays:
        img, info = render_overlay(src, r, class_names, fill_alpha=args.mask_alpha,
                                   draw_boxes=args.draw_boxes, run_name=args.run_name,
                                   project_name=args.project_name, timestamp=ast_stamp(),
                                   code_lookup=code_lookup)
        rel_overlay = os.path.join(OVERLAY_DIRNAME, filename)
        img.save(os.path.join(args.out_dir, rel_overlay), quality=JPEG_QUALITY)
        for key in RENDER_TOTAL_KEYS:
            summary['render_totals'][key] += info[key]

    det_rows = _manifest_detections(rows, class_names, code_lookup)
    confs = [d['confidence'] for d in det_rows if d['confidence'] is not None]
    for det in det_rows:
        if det['confidence'] is not None:
            summary['confidences_by_class'].setdefault(det['class'], []).append(det['confidence'])
    summary['pred_items'].extend(build_prediction_items([(filename, src, r)], class_names, stats))
    summary['total_predictions'] += len(rows)
    summary['with_any'] += int(bool(rows))
    summary['items'].append({
        'filename': filename,
        'raw': os.path.relpath(src, args.out_dir) if src.startswith(args.out_dir) else src,
        'overlay': rel_overlay,
        'n_detections': len(rows),
        'max_conf': max(confs) if confs else 0.0,
        'detections': det_rows,
    })


def infer_images(model, images, args, class_names, code_lookup, stats):
    """Run the model over every image and return the filled summary.

    A failure on one image is printed and recorded under summary['errors'] with
    the filename and the message; the run continues with the next image.
    """
    summary = _empty_summary()
    t0 = time.time()
    for idx, src in enumerate(images):
        try:
            _infer_one(model, src, args, class_names, code_lookup, summary, stats)
        except Exception as e:  # one bad image must not end the run; it is recorded, not hidden
            print(f'[infer] ERROR on {src}: {e}')
            summary['errors'].append({'filename': os.path.basename(src), 'path': src, 'error': str(e)})
        if (idx + 1) % PROGRESS_EVERY == 0 or idx == len(images) - 1:
            print(f'[infer] {idx + 1}/{len(images)} images, {summary["with_any"]} with a detection, '
                  f'{summary["total_predictions"] / (idx + 1):.1f} detections per image')
    summary['wall_seconds'] = time.time() - t0
    return summary


# ---------------------------------------------------------------- outputs


def blind_spot_summary(confidences_by_class, n_items, with_any, class_names, code_lookup):
    """One HTML sentence per weak signal: low average confidence, noisy classes, many empty images."""
    bits = []
    for label, confs in confidences_by_class.items():
        if not confs:
            continue
        shown = pretty_label(label, code_lookup)
        avg = sum(confs) / len(confs)
        low = sum(1 for c in confs if c < BLIND_SPOT_LOW_CONF) / len(confs)
        if avg < BLIND_SPOT_AVG_CONF:
            bits.append(f'<code>{shown}</code>: average confidence {avg:.2f}, low; the model is unsure')
        elif low > BLIND_SPOT_LOW_FRACTION:
            bits.append(f'<code>{shown}</code>: {low * 100:.0f}% of predictions below confidence '
                        f'{BLIND_SPOT_LOW_CONF}, noisy')
    images_no_det = n_items - with_any
    if images_no_det and n_items and images_no_det / n_items > BLIND_SPOT_NO_DETECTION_FRACTION:
        bits.append(f'{images_no_det}/{n_items} images got no detection '
                    f'({images_no_det / n_items * 100:.0f}%): possible blind spot or out-of-distribution data.')
    return FOOTER_SEPARATOR.join(bits)


def build_manifest(args, class_names, summary, stats, code_lookup, predictions_written, class_names_dropped=()):
    """Assemble manifest.json: items, totals, thresholds, skipped counts, render settings, errors.

    class_names_dropped lists the model class-map keys that were not integers
    and so could not name a class id."""
    totals = summary['render_totals']
    confidences = summary['confidences_by_class']
    return {
        'items': summary['items'],
        'total_predictions': summary['total_predictions'],
        'with_any': summary['with_any'],
        'class_names': class_names,
        'class_names_dropped': [str(k) for k in class_names_dropped],
        'class_full_names': ({name: code_lookup[name] for name in class_names.values() if name in code_lookup}
                             if code_lookup else {}),
        'confidences_by_class': {
            k: {'count': len(v), 'avg': sum(v) / len(v), 'min': min(v), 'max': max(v)}
            for k, v in confidences.items()
        },
        'conf_threshold': args.conf,
        'iou_threshold': args.iou,
        'imgsz': args.imgsz,
        'device': args.device,
        'detections_skipped': skipped_total(stats),
        'detections_skipped_reasons': {r: stats[r] for r in SKIP_REASONS if stats[r]},
        'detections_with_polygon': stats[STAT_POLYGON_WRITTEN],
        'detections_box_only': stats[STAT_BOX_WRITTEN],
        'detections_degenerate_polygon_written_as_box': stats[STAT_DEGENERATE_POLYGON],
        'render': {
            'fill_alpha': args.mask_alpha,
            'draw_boxes': args.draw_boxes,
            'overlays_saved': args.save_overlays,
            'instances_with_mask': totals['instances_with_mask'],
            'instances_box_only': totals['instances_box_only'],
            'instances_without_geometry': totals['instances_without_geometry'],
            'chips_overlapping': totals['chips_overlapping'],
            'draw_boxes_had_effect': bool(args.draw_boxes and totals['instances_with_mask']),
            'note': 'Every instance is filled at fill_alpha and outlined. draw_boxes adds a box '
                    'outline only around mask polygons; a box-only instance is always outlined.',
        },
        'ignored_flags': [IGNORED_FLAG_SAVE_CROPS] if args.save_crops else [],
        'errors': summary['errors'],
        'wall_seconds': summary['wall_seconds'],
        'blind_spots': blind_spot_summary(confidences, len(summary['items']), summary['with_any'],
                                          class_names, code_lookup),
        'args': vars(args),
        'predictions': PREDICTIONS_FILENAME if predictions_written else '',
        'generated_at': ast_now(),
    }


def write_predictions(args, class_names, pred_items, stats):
    """Write predictions.json (schema 2: polygon_xyn or box_xyxyn per detection) and print the counts."""
    predictions = {
        'schema_version': PREDICTIONS_SCHEMA_VERSION,
        'generated_at': ast_now(),
        'run_dir': args.run_dir,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'class_names': {str(k): v for k, v in class_names.items()},
        'detections_with_polygon': stats[STAT_POLYGON_WRITTEN],
        'detections_box_only': stats[STAT_BOX_WRITTEN],
        'detections_skipped': skipped_total(stats),
        'detections_skipped_reasons': {r: stats[r] for r in SKIP_REASONS if stats[r]},
        'items': pred_items,
    }
    path = os.path.join(args.out_dir, PREDICTIONS_FILENAME)
    with open(path, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f'[infer] wrote {path}: {len(pred_items)} items, '
          f'{stats[STAT_POLYGON_WRITTEN]} polygon detections, {stats[STAT_BOX_WRITTEN]} box detections, '
          f'{skipped_total(stats)} skipped')


def load_model(weights):
    """Load the Ultralytics model from a weights file (imported here so tests need no GPU stack)."""
    from ultralytics import YOLO
    return YOLO(weights)


def run(args, model, code_lookup=None):
    """Run inference end to end with an already loaded model and return the manifest dict.

    Creates out_dir (and overlays/ when asked), resolves the image source,
    infers every image, writes predictions.json when asked, and writes manifest.json.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_overlays:
        os.makedirs(os.path.join(args.out_dir, OVERLAY_DIRNAME), exist_ok=True)
    if args.save_crops:
        print('[infer] --save_crops is accepted for compatibility with the Model Inference panel and '
              'does nothing: no crop images are written. The manifest lists it under ignored_flags.')
    class_names, dropped_keys = normalise_class_names(getattr(model, 'names', None))
    if dropped_keys:
        print(f'[infer] WARNING: {len(dropped_keys)} class map key(s) are not integers and name no class: '
              f'{dropped_keys}. They are listed in the manifest under class_names_dropped.')
    print(f'[infer] source_type={args.source_type} imgsz={args.imgsz} conf={args.conf} iou={args.iou} '
          f'classes={len(class_names)}')

    images = collect_source(args)
    print(f'[infer] resolved {len(images)} images')
    if not images:
        print('[infer] nothing to infer; writing an empty manifest')

    stats = new_detection_stats()
    summary = infer_images(model, images, args, class_names, code_lookup, stats)
    if args.save_predictions:
        write_predictions(args, class_names, summary['pred_items'], stats)
    manifest = build_manifest(args, class_names, summary, stats, code_lookup, args.save_predictions,
                              class_names_dropped=dropped_keys)
    manifest_path = os.path.join(args.out_dir, MANIFEST_FILENAME)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f'[infer] done. {len(summary["items"])} images, {summary["total_predictions"]} detections, '
          f'{summary["with_any"]} images with at least one detection, {len(summary["errors"])} errors, '
          f'{skipped_total(stats)} detections skipped. Manifest: {manifest_path}')
    return manifest


def main():
    """Parse, validate, load the model, run, and write the outputs."""
    args = arg_parse()
    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f'run_dir not found: {args.run_dir}')
    weights = _find_best_weights(args.run_dir)
    print(f'[infer] weights={weights}')
    code_lookup = build_code_lookup(args.codes_csv, args.run_dir)
    run(args, load_model(weights), code_lookup=code_lookup)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
