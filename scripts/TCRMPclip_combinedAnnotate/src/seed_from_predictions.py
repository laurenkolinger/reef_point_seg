"""seed_from_predictions.py: seed model predictions.json into a step4test
combined-annotator export as PENDING, editable masks for human review.

Part of the active-learning loop: Step 8 inference writes predictions.json
(filename, raw abs path, per-detection normalized polygon + confidence).
This script turns those detections into ordinary seg_dict/mask entries in
segmentations/segmentations.json (source_type='model', status='pending'),
symlinks the raw frame into routed_input/raw/, and writes a round manifest
so a later "reset this round" tool (Task 5) can find exactly what it added.

The frozen holdout is the hard rule here: a frame the training split assigned
to valid or test is never seeded, so a review round can never put a held-out
frame in front of a reviewer and from there back into training. The manifest
that says which frames those are is required (--split_manifest), and a
manifest that cannot answer the question is refused rather than worked around.

Three refusals, each exiting EXIT_REFUSED with one sentence on stderr, which is
what the orchestrator's Refine route shows the operator:
  - a split manifest that is missing, unreadable, not a JSON object, or holds
    nothing out (_load_split_manifest, validate_holdout_guard);
  - a predictions file that is missing, unreadable, or not the shape the
    inference step writes (_load_predictions);
  - a predictions file whose every detection is a bounding box with no outline
    (refuse_box_only_predictions), because a detection model's output cannot
    become a reviewable mask and the round would otherwise report success with
    nothing on any frame.

No prompts; fully flag-driven. Requires cv2/numpy/PIL (present in REPO/env).
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from datetime import datetime

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mask_ops
import provenance


# ── Frozen-holdout constants ────────────────────────────────────────────
# Both spellings of the per-image pin table a split_manifest.json can carry.
# The segmentation splitter (scripts/TCRMPtrain_oceankindCV/src/pinned_split.py)
# writes "pinned"; the detection splitter
# (scripts/TCRMPtrain_oceankindCV/src/detect_dataset.py, build_detect_dataset)
# writes "pins". Reading only "pinned" made this guard a no-op on every
# detection project: the table read as empty, no frame answered as held out,
# and valid and test frames were free to be seeded, reviewed and exported back
# into the training set (found and fixed 2026-09-16). Step 7's evaluator reads
# the two spellings the same way (evaluate_run.py, MANIFEST_PIN_KEYS), so this
# is the platform's reading of a pin table, not a local workaround.
HOLDOUT_TABLE_KEYS = ('pinned', 'pins')

# The three splits a pin table may assign, and the two of them that are held
# out of training and must never be seeded for review.
SPLIT_NAMES = ('train', 'valid', 'test')
HOLDOUT_SPLITS = ('valid', 'test')

# The holdout policy whose held-out frames are derived from the TCRMP transect
# number in the frame name instead of being named one by one in the pin table.
TRANSECT_HOLDOUT_MODE = 'transect'
# The keys under holdout_transects. pinned_split.py writes "val" and "test"
# there, which are not the pin table's split names ("valid" and "test").
TRANSECT_HOLDOUT_KEYS = ('val', 'test')

# Exit status for every refusal. The orchestrator's Refine route
# (scripts/pipeline_orchestrator/app.py, step4loop_run) reports a non-zero exit
# as "Seeder exited <code>: <stderr>", so the stderr message is what the
# operator reads; every refusal below names the file at fault and the way out.
EXIT_REFUSED = 2

# A polygon needs three points, so six normalized coordinates, to enclose any
# area. Step 8 writes nothing shorter (run_inference.py MIN_POLYGON_POINTS);
# anything shorter reaching here came from a hand-edited or third-party
# predictions file.
MIN_POLYGON_COORDS = 6
# Normalized box corners, x1 y1 x2 y2, the shape a detection model's output
# carries instead of an outline (run_inference.py writes it as box_xyxyn).
BOX_COORD_COUNT = 4


# ── Transect helper (DUPLICATED, kept byte-identical with the sibling copy) ──
# keep in sync with TR/src/pinned_split.py:parse_transect (byte-identical, like the provenance.py copies)
def parse_transect(basename):
    """Transect number from a TCRMP frame basename, or None.
    Real names end in .jpeg/.jpg; strip the extension, then the trailing
    _T<digits>. Only a 3-digit T-number yields a transect (its first digit);
    1-/2-/4-digit T-numbers (and unparseable names) yield None -> train, never holdout."""
    stem = os.path.splitext(basename)[0]
    m = re.search(r'_T(\d+)$', stem)
    if m and len(m.group(1)) == 3:
        return int(m.group(1)[0])
    return None


def _load_codes_csv(path):
    """basename-agnostic species code -> {name, category} lookup."""
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get('code') or row.get('species_code') or '').strip()
            if not code:
                continue
            lookup[code] = {
                'name': (row.get('name') or row.get('species_name') or '').strip(),
                'category': (row.get('category') or '').strip(),
            }
    return lookup


def _load_lores_manifest(path):
    """basename (no ext) -> row dict, from a make_lores_variants.py manifest."""
    lookup = {}
    if not path or not os.path.exists(path):
        return lookup
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = (row.get('basename') or '').strip()
            if b:
                lookup[b] = row
    return lookup


def _load_split_manifest(path: str) -> dict:
    """Read the frozen-split manifest and prove it can hold frames out.

    Parameters:
        path: the --split_manifest value, the split_manifest.json written
            beside the step-6 dataset by either splitter.

    Returns:
        The parsed manifest document, already checked by
        validate_holdout_guard.

    Exits EXIT_REFUSED with a message on stderr when the file is absent,
    unreadable, not JSON, not a JSON object, or carries no holdout record this
    seeder can use. Seeding on a manifest that answers no question would put
    held-out frames back into training, so every one of those is a refusal and
    never a warning.
    """
    if not path or not os.path.exists(path):
        print(f"ERROR: --split_manifest is required and was not found: {path}. Run the training "
              "step so the split is written beside the dataset, then start the review round "
              "again.", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except ValueError as exc:
        # ValueError, not json.JSONDecodeError: a manifest holding bytes that
        # are not UTF-8 text (a truncated copy, an image written over it) fails
        # in the codec, inside f.read(), before the JSON parser ever runs, and
        # UnicodeDecodeError is not a JSONDecodeError. Both are ValueError
        # subclasses, so one clause turns either one into the message below
        # instead of a traceback the operator cannot act on.
        print(f"ERROR: split manifest {path} is not valid JSON ({exc}). Re-run the training "
              "step so the split is rewritten beside the dataset, or repair the file, then "
              "start the review round again.", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    except OSError as exc:
        print(f"ERROR: split manifest {path} could not be read: {exc}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    try:
        validate_holdout_guard(data, path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    return data


def _load_predictions(path: str) -> dict:
    """Read the inference step's predictions.json and prove it can be seeded.

    Parameters:
        path: the --predictions value, the predictions.json the inference step
            writes when it is run with predictions saved.

    Returns:
        The parsed document: a JSON object whose "items", when it carries any,
        is a list of per-frame objects, which is the only shape the seeding
        loop reads.

    Exits EXIT_REFUSED with a message on stderr when the file is absent,
    unreadable, not JSON text, not a JSON object, or carries items that are not
    a list of objects. The same reasoning as the split manifest: the Refine
    route shows this message to the operator, and a sentence naming the file
    and the step that rewrites it beats a stack trace.
    """
    if not path or not os.path.exists(path):
        print(f"ERROR: --predictions is required and was not found: {path}. Run the inference "
              "step with predictions saved, then start the review round again.", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except ValueError as exc:
        # ValueError covers both a JSON parse failure and a file that is not
        # UTF-8 text at all, exactly as in _load_split_manifest.
        print(f"ERROR: predictions file {path} is not valid JSON ({exc}). Run the inference step "
              "again so it is rewritten, then start the review round again.", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    except OSError as exc:
        print(f"ERROR: predictions file {path} could not be read: {exc}", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    if not isinstance(data, dict):
        print(f"ERROR: predictions file {path} must hold a JSON object of the inference run, got "
              f"{type(data).__name__}. Run the inference step again so it is rewritten.",
              file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    items = data.get('items')
    if items is None:
        return data
    if not isinstance(items, list):
        print(f"ERROR: predictions file {path}: 'items' must be a list of per-frame records, got "
              f"{type(items).__name__}. Run the inference step again so it is rewritten.",
              file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    bad = [i for i, item in enumerate(items) if not isinstance(item, dict)]
    if bad:
        print(f"ERROR: predictions file {path}: {len(bad)} entry(ies) under 'items' are not "
              f"per-frame objects, starting at position {bad[0]}. Run the inference step again so "
              "it is rewritten.", file=sys.stderr)
        sys.exit(EXIT_REFUSED)
    return data


def validate_holdout_guard(split_manifest: object, manifest_path: str) -> None:
    """Refuse a split manifest that cannot keep a single frame out of a round.

    A usable manifest carries at least one of two mechanisms: a pin table
    (under either spelling in HOLDOUT_TABLE_KEYS) naming at least one image, or
    the transect policy with at least one held-out transect number. A manifest
    with neither says nothing about any frame, so every frame would read as
    training material and the guard would pass valid and test frames straight
    through. Step 7's evaluator draws the line in the same place
    (evaluate_run.py, _split_pinned).

    Parameters:
        split_manifest: the parsed split_manifest.json document.
        manifest_path: where it was read from, named in every refusal so the
            operator knows which file to fix.

    Returns:
        None when the manifest can hold frames out.

    Raises:
        ValueError when the document is not a JSON object, when a pin table is
        not an object of basename to split, when a pin names a split outside
        SPLIT_NAMES, or when the manifest holds nothing out at all.
    """
    if not isinstance(split_manifest, dict):
        raise ValueError(
            f"split manifest {manifest_path} must hold a JSON object of the split record, got "
            f"{type(split_manifest).__name__}. Re-run the training step so the split is "
            "rewritten beside the dataset.")

    n_pins = 0
    for key in HOLDOUT_TABLE_KEYS:
        table = split_manifest.get(key)
        if table is None:
            continue
        if not isinstance(table, dict):
            raise ValueError(
                f"split manifest {manifest_path}: '{key}' must be an object of image basename "
                f"to split, got {type(table).__name__}. A pin table that cannot be read is a "
                "holdout that cannot be honored, so this seeder refuses rather than guess.")
        bad = [(name, split) for name, split in table.items() if split not in SPLIT_NAMES]
        if bad:
            raise ValueError(
                f"split manifest {manifest_path}: '{key}' gives {len(bad)} image(s) a split "
                f"outside {SPLIT_NAMES}, starting with {bad[:3]!r}. A frame whose split cannot "
                "be read may be a held-out frame, so this seeder refuses rather than seed it.")
        n_pins += len(table)

    if n_pins:
        return
    if _transect_holdout_numbers(split_manifest):
        return
    raise ValueError(
        f"split manifest {manifest_path} holds nothing out: it names no image under "
        f"{HOLDOUT_TABLE_KEYS} and it does not use the transect holdout, so this seeder cannot "
        "tell a training frame from a validation or test frame. Seeding from it would put "
        "held-out frames in front of a reviewer and from there back into training. Re-run the "
        "training step so the split is written beside the dataset, then point --split_manifest "
        "at that dataset's split_manifest.json.")


def _transect_holdout_numbers(split_manifest: object) -> set[int]:
    """The transect numbers a transect-policy manifest holds out.

    Parameters:
        split_manifest: the parsed split_manifest.json document.

    Returns:
        A set of transect numbers from holdout_transects (both the "val" and
        the "test" list), empty when the manifest does not use the transect
        policy or names no transect. A number written as a digit string ("5")
        or as a whole-number float (5.0, the shape a JSON writer gives an
        integer it rounded) is read as that number, because a wider holdout
        excludes more and this guard errs toward excluding. A value naming no
        whole transect (5.5, or text) is left out: parse_transect only ever
        returns a whole number, so no frame could match it either way.
    """
    if not isinstance(split_manifest, dict):
        return set()
    if split_manifest.get('holdout_mode') != TRANSECT_HOLDOUT_MODE:
        return set()
    table = split_manifest.get('holdout_transects')
    if not isinstance(table, dict):
        return set()
    numbers = set()
    for key in TRANSECT_HOLDOUT_KEYS:
        values = table.get(key)
        if not isinstance(values, (list, tuple)):
            continue
        for value in values:
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                numbers.add(value)
            elif isinstance(value, float) and value.is_integer():
                numbers.add(int(value))
            elif isinstance(value, str) and value.strip().isdigit():
                numbers.add(int(value.strip()))
    return numbers


def _is_holdout(basename: str, split_manifest: dict) -> bool:
    """Report whether a frame belongs to the frozen holdout.

    Parameters:
        basename: the frame filename as predictions.json spells it.
        split_manifest: the parsed split_manifest.json document, already
            through validate_holdout_guard.

    Returns:
        True when either pin table (HOLDOUT_TABLE_KEYS, read as one table
        because the two splitters spell the same thing differently) assigns the
        frame to valid or test, or when the manifest uses the transect policy
        and the frame's transect is held out. False otherwise, which means the
        frame is training material and may be seeded for review.

    Example:
        _is_holdout('TCRMP20240101_clip_BPT_T601.jpeg', {'holdout_mode':
        'transect', 'holdout_transects': {'val': [5], 'test': [6]}}) is True.
    """
    for key in HOLDOUT_TABLE_KEYS:
        table = split_manifest.get(key)
        if isinstance(table, dict) and table.get(basename) in HOLDOUT_SPLITS:
            return True
    transect = parse_transect(basename)
    return transect is not None and transect in _transect_holdout_numbers(split_manifest)


def _usable_polygon(det: dict) -> list[float]:
    """The detection's normalized outline as a flat coordinate list, or [].

    Parameters:
        det: one predictions.json detection.

    Returns:
        The polygon_xyn values as floats when the detection carries an outline
        this seeder can turn into a mask: a list of numbers, an even count, at
        least MIN_POLYGON_COORDS of them. An empty list otherwise, which
        includes a box-only detection (no polygon_xyn at all), a truncated
        outline, and an outline whose values are not numbers. The caller counts
        every empty answer rather than dropping it silently.
    """
    polygon = det.get('polygon_xyn')
    if not isinstance(polygon, (list, tuple)):
        return []
    if len(polygon) < MIN_POLYGON_COORDS or len(polygon) % 2:
        return []
    values = []
    for value in polygon:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []
        values.append(float(value))
    return values


def _has_box(det: dict) -> bool:
    """Report whether a detection carries normalized box corners.

    Parameters:
        det: one predictions.json detection.

    Returns:
        True when box_xyxyn holds BOX_COORD_COUNT numbers, which is what step 8
        writes for a detection model's output (a box with no outline). The
        caller uses it to tell a box-only detection, which a future
        rectangle-to-mask step could seed, from a detection carrying no
        geometry at all, which nothing can seed.
    """
    box = det.get('box_xyxyn')
    if not isinstance(box, (list, tuple)) or len(box) != BOX_COORD_COUNT:
        return False
    return all(not isinstance(v, bool) and isinstance(v, (int, float)) for v in box)


def _iter_detections(predictions: dict) -> Iterator[dict]:
    """Every detection in a predictions.json document, in file order.

    Parameters:
        predictions: the parsed predictions.json document.

    Yields:
        One detection object at a time. _load_predictions has already refused a
        document that is not an object and items that are not a list of
        objects; the checks repeated here cover a detections value that is not
        a list and an entry inside it that is not an object, so this pass can
        never raise before the refusal it exists to make.
    """
    items = predictions.get('items') if isinstance(predictions, dict) else None
    if not isinstance(items, list):
        return
    for item in items:
        detections = item.get('detections') if isinstance(item, dict) else None
        if not isinstance(detections, list):
            continue
        for det in detections:
            if isinstance(det, dict):
                yield det


def refuse_box_only_predictions(predictions: dict, predictions_path: str) -> None:
    """Refuse a predictions file that carries boxes and not one outline.

    A Refine round exists to put editable masks in front of a reviewer, and a
    mask is built from an outline. A detection (bounding-box) model writes
    box_xyxyn and never polygon_xyn (run_inference.py, _prediction_entry), so a
    round seeded from one opens the review tool with no mask on any frame.
    Until 2026-09-16 that round still exited 0, and the orchestrator's Refine
    route shows the seeder's stderr only when the exit is non-zero, so the
    operator read a green "Seeded 12 frame(s), 0 mask(s) for review." and was
    told nothing. Refusing here is what puts the reason in front of the person
    who started the round.

    Parameters:
        predictions: the parsed predictions.json document.
        predictions_path: where it was read from, named in the refusal so the
            operator knows which inference run produced it.

    Returns:
        None when one detection anywhere carries an outline this seeder can
        turn into a mask, and also when the file carries no box-only detection
        at all: an empty round, or one whose detections are all below the
        confidence floor, is an ordinary summary and not a refusal. A mixed
        round (some outlines, some boxes) seeds its outlines and reports the
        boxes in skipped_box_only.

    Exits EXIT_REFUSED with a message on stderr when every detection in the
    file carries a bounding box and none carries an outline.
    """
    n_box_only = 0
    for det in _iter_detections(predictions):
        if _usable_polygon(det):
            return
        if _has_box(det):
            n_box_only += 1
    if not n_box_only:
        return
    print(f"ERROR: {predictions_path} carries {n_box_only} bounding-box detection(s) and no "
          "outline, so this round would open the review tool with no mask on any frame. A "
          "detection (bounding-box) model cannot seed a review round; run the inference step "
          "again with a segmentation model, then start the round.", file=sys.stderr)
    sys.exit(EXIT_REFUSED)


def _polygon_xyn_to_px(polygon_xyn, w, h):
    flat = []
    for i in range(0, len(polygon_xyn), 2):
        flat.append(polygon_xyn[i] * w)
        flat.append(polygon_xyn[i + 1] * h)
    return flat


def _centroid(polygon_px):
    xs = polygon_px[0::2]
    ys = polygon_px[1::2]
    if not xs:
        return 0, 0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Seed step8 predictions.json into a step4test export as pending review masks.")
    p.add_argument('--predictions', required=True, help="Path to predictions.json (Step 8 output).")
    p.add_argument('--export_dir', required=True, help="step4test export dir (contains/receives segmentations/).")
    p.add_argument('--codes_csv', required=True, help="Master codes CSV (code,category,name) for species -> name/category.")
    p.add_argument('--split_manifest', required=True,
                   help="Required: the split_manifest.json written beside the step-6 dataset by "
                        "either splitter, pinned_split.py (pin table 'pinned', transect holdout) "
                        "or detect_dataset.py (pin table 'pins'). Frames pinned or derived to "
                        "valid or test are never seeded, and a manifest that holds nothing out is "
                        "refused rather than worked around.")
    p.add_argument('--conf_min', type=float, default=0.25, help="Minimum detection confidence to seed (default 0.25).")
    p.add_argument('--max_frames', type=int, default=0, help="Cap on frames seeded this round (0 = no cap).")
    p.add_argument('--skip_empty', action='store_true', help="Skip zero-detection frames instead of seeding masks:[].")
    p.add_argument('--lores_manifest', default='', help="Optional make_lores_variants.py manifest CSV; prefer lores file when original width > 1920.")
    p.add_argument('--round_note', default='', help="Free-text note stored in the round manifest.")
    p.add_argument('--dry_run', action='store_true', help="Print the summary only; write nothing.")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    split_manifest = _load_split_manifest(args.split_manifest)
    codes_lookup = _load_codes_csv(args.codes_csv)
    lores_lookup = _load_lores_manifest(args.lores_manifest) if args.lores_manifest else {}

    predictions = _load_predictions(args.predictions)
    # Before any folder is made or any frame is symlinked: a detection model's
    # predictions cannot become masks, and a round seeded from them would show
    # the operator a successful round with nothing to review.
    refuse_box_only_predictions(predictions, os.path.abspath(args.predictions))

    export_dir = args.export_dir
    seg_dir = os.path.join(export_dir, 'segmentations')
    raw_dir = os.path.join(export_dir, 'routed_input', 'raw')
    rounds_dir = os.path.join(export_dir, 'loop_rounds')
    seg_path = os.path.join(seg_dir, 'segmentations.json')

    if not args.dry_run:
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(rounds_dir, exist_ok=True)

    seg_dict = {}
    if os.path.exists(seg_path):
        with open(seg_path, 'r') as f:
            seg_dict = json.load(f)

    seeded_frames = 0
    seeded_masks = 0
    skipped_existing = []
    skipped_below_conf = 0
    skipped_missing_raw = 0
    skipped_holdout = 0
    skipped_review_class = 0
    skipped_box_only = 0
    skipped_no_geometry = 0
    round_seeded = {}

    items = predictions.get('items', [])
    for item in items:
        if args.max_frames and seeded_frames >= args.max_frames:
            break

        filename = item.get('filename')
        if not filename:
            continue

        if filename in seg_dict:
            skipped_existing.append(filename)
            continue

        if _is_holdout(filename, split_manifest):
            skipped_holdout += 1
            continue

        raw_abs = item.get('raw')
        if not raw_abs or not os.path.exists(raw_abs):
            skipped_missing_raw += 1
            continue

        width = item.get('width')
        height = item.get('height')

        # Lores preference: only when the original is wider than 1920 and the
        # manifest maps this basename (basename column has no extension).
        seed_src = raw_abs
        stem = os.path.splitext(filename)[0]
        lores_row = lores_lookup.get(stem)
        if lores_row and width and width > 1920:
            lores_path = lores_row.get('lores_path')
            if lores_path and os.path.exists(lores_path):
                seed_src = lores_path
                with Image.open(lores_path) as im:
                    width, height = im.size

        detections = item.get('detections', []) or []

        masks = []
        next_mask_id = 0
        for det in detections:
            confidence = det.get('confidence', 0.0)
            if confidence < args.conf_min:
                skipped_below_conf += 1
                continue

            # A detection model returns boxes, not outlines, and step 8 writes
            # those as box_xyxyn with no polygon_xyn. This seeder builds a mask
            # from an outline, so such a detection cannot be seeded; until
            # 2026-09-16 it was dropped with no counter, which read to a
            # reviewer as "Seeded N frames, 0 masks" with nothing wrong. A file
            # that is all boxes was already refused above, so what reaches here
            # is a mixed round: its outlines seed and both kinds of drop are
            # counted in the summary and warned about on stderr.
            polygon_xyn = _usable_polygon(det)
            if not polygon_xyn:
                if _has_box(det):
                    skipped_box_only += 1
                else:
                    skipped_no_geometry += 1
                continue

            polygon_px = _polygon_xyn_to_px(polygon_xyn, width, height)
            binary_mask = mask_ops.polygon_to_mask(
                [polygon_px[i:i + 2] for i in range(0, len(polygon_px), 2)],
                height, width)
            cx, cy = _centroid(polygon_px)

            code = det.get('class', '')
            # Expert-review lock (2026-09-06): mask_ops.build_mask_dict already
            # refuses to write a REVIEW label onto a new mask, so a detection
            # carrying that class would seed an unlabelled mask with no word
            # said. No trained model can carry it today, because the training
            # export keeps flagged masks out of the training set, but the
            # predictions file is a path this tool is handed, so the drop is
            # counted and reported rather than left silent.
            if mask_ops.review_tag_requested({'species': code}):
                skipped_review_class += 1
                continue
            info = codes_lookup.get(code, {})
            point_info = {
                'label': 'M',
                'species': code,
                'name': info.get('name', code),
                'category': info.get('category', ''),
                'x': cx,
                'y': cy,
            }

            mask_dict = mask_ops.build_mask_dict(
                mask_id=next_mask_id,
                binary_mask=binary_mask,
                score=confidence,
                point_info=point_info,
                source_type='model',
            )
            if mask_dict is None:
                continue
            mask_dict['status'] = 'pending'
            masks.append(mask_dict)
            next_mask_id += 1

        if args.skip_empty and not masks:
            continue

        # Symlink raw frame into routed_input/raw/; skip on name collision
        # with a different target (never overwrite).
        link_path = os.path.join(raw_dir, filename)
        if not args.dry_run:
            if os.path.islink(link_path) or os.path.exists(link_path):
                existing_target = os.path.realpath(link_path) if os.path.islink(link_path) else None
                if existing_target != os.path.realpath(seed_src):
                    skipped_existing.append(filename)
                    continue
            else:
                os.symlink(seed_src, link_path)

        seg_dict[filename] = {
            'image_path': f'raw/{filename}',
            'image_path_abs': seed_src,
            'image_width': width,
            'image_height': height,
            'masks': masks,
            'reference_points': [],
            'processed_at': provenance.ast_now(),
            'reviewed': False,
            'exported': False,
            'label_outcomes': {},
        }

        round_seeded[filename] = [m['id'] for m in masks]
        seeded_frames += 1
        seeded_masks += len(masks)

    round_manifest = {
        'schema_version': 1,
        'at': provenance.ast_now(),
        'predictions': os.path.abspath(args.predictions),
        'params': {
            'conf_min': args.conf_min,
            'max_frames': args.max_frames,
            'skip_empty': args.skip_empty,
        },
        'seeded': round_seeded,
        'skipped_existing': skipped_existing,
        'note': args.round_note,
    }

    round_manifest_rel = None
    if not args.dry_run and seeded_frames > 0:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=seg_dir, suffix='.tmp')
        os.close(tmp_fd)
        with open(tmp_path, 'w') as f:
            json.dump(seg_dict, f, indent=2)
        os.replace(tmp_path, seg_path)

        # Build the round filename timestamp from ast_now's own clock so it
        # always matches the AST timestamp recorded 'at' in the manifest.
        ast_dt = datetime.fromisoformat(round_manifest['at'])
        stamp = ast_dt.strftime('%Y%m%d_%H%M%S')
        round_manifest_name = f'round_{stamp}.json'
        round_manifest_path = os.path.join(rounds_dir, round_manifest_name)
        tmp_fd2, tmp_path2 = tempfile.mkstemp(dir=rounds_dir, suffix='.tmp')
        os.close(tmp_fd2)
        with open(tmp_path2, 'w') as f:
            json.dump(round_manifest, f, indent=2)
        os.replace(tmp_path2, round_manifest_path)
        round_manifest_rel = os.path.join('loop_rounds', round_manifest_name)
    elif seeded_frames > 0:
        # dry_run: still report where the manifest WOULD go.
        ast_dt = datetime.fromisoformat(round_manifest['at'])
        stamp = ast_dt.strftime('%Y%m%d_%H%M%S')
        round_manifest_rel = os.path.join('loop_rounds', f'round_{stamp}.json')

    summary = {
        'seeded_frames': seeded_frames,
        'seeded_masks': seeded_masks,
        'skipped_existing': len(skipped_existing),
        'skipped_below_conf': skipped_below_conf,
        'skipped_missing_raw': skipped_missing_raw,
        'skipped_holdout': skipped_holdout,
        'skipped_review_class': skipped_review_class,
        'skipped_box_only': skipped_box_only,
        'skipped_no_geometry': skipped_no_geometry,
        'round_manifest': round_manifest_rel,
    }
    predictions_abs = os.path.abspath(args.predictions)
    if skipped_box_only:
        print(f"WARNING: {skipped_box_only} detection(s) in {predictions_abs} carried a bounding "
              "box and no outline, so no mask was seeded for them. A detection model's "
              "predictions cannot be reviewed as masks; train or pick a segmentation model for "
              "this review round.", file=sys.stderr)
    if skipped_no_geometry:
        print(f"WARNING: {skipped_no_geometry} detection(s) in {predictions_abs} carried neither "
              "an outline nor a bounding box and were dropped. Check that inference run's "
              "predictions.json before trusting this round's counts.", file=sys.stderr)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
