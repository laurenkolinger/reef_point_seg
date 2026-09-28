#!/usr/bin/env python3
"""Pinned (frozen-holdout) train/valid/test split for the TCRMP YOLO-seg export.

Unlike bal_train_test_split.py (random stratified split, reshuffled every
run), this tool freezes the validation/test holdout so metrics stay
comparable across retraining rounds and video-adjacent frames never leak
into the holdout.

Default holdout mode is BY TRANSECT: transect 5 -> valid, transect 6 -> test,
transects 1-4 -> train, at every site. The assignment is recorded in
dataset/split_manifest.json and honored on every subsequent run: existing
pins never move, only new frames get assigned.

Every manifest also records the split itself (schema version 2, 2026-09-04):
the seed string the run carried (--rand), the validation and test fractions
it carried (--valid, --test), the holdout mode, per-split counts, and a split
fingerprint computed by detect_dataset.split_fingerprint over the final
basename-to-split table, under the same scheme the detection splitter
writes. detect_dataset.read_split_record reads either splitter's manifest,
so two training runs are compared only when their fingerprints match. The
seed and fractions are what the run was given: on a transect holdout they
reach no image, and on a rebuild they reach only images the manifest did not
already pin. The fingerprint, not those numbers, says what is on disk.

See scripts/TCRMPtrain_oceankindCV/tests/test_pinned_split.py and the task
brief for the full rule set (manifest replay, legacy adoption, graceful
degrade + auto-upgrade, pinned-random, min_samples, per-class train
coverage, empty-train guard).

Run: env/bin/python scripts/TCRMPtrain_oceankindCV/src/pinned_split.py --src <export> --out <step6_dir>/dataset
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mode as _mode

import yaml

# detect_dataset.py sits beside this file and owns the fingerprint scheme both
# splitters share. run_step6.sh runs this file as a script, which puts this
# folder first on sys.path; a caller that imports pinned_split from elsewhere
# gets the same folder added here, so the sibling resolves the same way in
# both cases. The import is side-effect free and pulls in no third-party
# package (detect_dataset imports PyYAML lazily).
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from detect_dataset import SPLIT_FINGERPRINT_SCHEME, split_fingerprint  # noqa: E402

# Atlantic Standard Time, fixed UTC-4, no daylight saving.
AST = timezone(timedelta(hours=-4))


def ast_now():
    """AST ISO-8601 timestamp, second precision, e.g. 2026-06-25T08:13:42-04:00."""
    return datetime.now(AST).isoformat(timespec='seconds')


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


# Version of the split_manifest.json this tool writes. Version 2 (2026-09-04)
# added seed, valid_frac, test_frac, split_fingerprint,
# split_fingerprint_scheme and counts. A version 1 manifest is still read and
# its pins still replayed; detect_dataset.read_split_record reports it as not
# recorded.
SCHEMA_VERSION = 2

# The three split names, in the order the dataset folders are written.
SPLIT_NAMES = ('train', 'valid', 'test')

# The manifest's file name inside the dataset folder.
MANIFEST_NAME = 'split_manifest.json'

# The holdout policies this tool writes into holdout_mode. A manifest naming
# any other policy was not written by this tool and is refused rather than
# replayed (detect_dataset.py writes "lab-random" for its own datasets, which
# this tool never reads). A manifest with no holdout_mode replays as transect.
HOLDOUT_MODES = ('transect', 'transect-degraded', 'pinned-random', 'adopted-random')


def arg_parse(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` or sys.argv) into a Namespace."""
    parser = argparse.ArgumentParser(
        description='Create a pinned (frozen-holdout) train/valid/test split for yolov5+.')
    parser.add_argument('--src', dest='src_dir', required=True, type=str,
                         help='Source export directory (has all_images/ all_labels/ data.yaml)')
    parser.add_argument('--out', dest='out_dir', required=True, type=str,
                         help='Output directory for the dataset split')
    parser.add_argument('--valid', dest='valid', default=0.2, type=float,
                         help='Fraction for validation when degrading/pinned-random, 0-1')
    parser.add_argument('--test', dest='test', default=0.1, type=float,
                         help='Fraction for test when degrading/pinned-random, 0-1')
    parser.add_argument('--min_samples', dest='min_samples', default=10, type=int,
                         help='Minimum number of TRAIN-split labeled images per class')
    parser.add_argument('--include-classes', dest='include_classes', default=None, type=str,
                         help='Comma-separated list of class IDs to keep (default: all)')
    parser.add_argument('--holdout-mode', dest='holdout_mode', default='transect',
                         choices=['transect', 'pinned-random'],
                         help='Fresh-split policy (ignored once a manifest exists)')
    parser.add_argument('--val-transects', dest='val_transects', default='5', type=str,
                         help='Comma-separated transect numbers assigned to validation')
    parser.add_argument('--test-transects', dest='test_transects', default='6', type=str,
                         help='Comma-separated transect numbers assigned to test')
    parser.add_argument('--manifest', dest='manifest_path', default=None, type=str,
                         help='Path to split_manifest.json (default: <out>/split_manifest.json)')
    parser.add_argument('--rand', dest='random_state', default=1, type=int,
                         help='Seed for pinned-random / degrade fallback slices')
    parser.add_argument('--yes', '-y', dest='assume_yes', action='store_true',
                         help='Present for CLI parity with bal_train_test_split.py; this tool never prompts.')
    return parser.parse_args(argv)


def _parse_int_csv(text: str, flag: str) -> set[int]:
    """Set of integers from the comma-separated ``text`` given to ``flag``.
    Empty items are skipped, so '5,' is {5}. Raises ValueError naming the
    flag and the offending item when an item is not an integer."""
    result = set()
    for item in text.split(','):
        item = item.strip()
        if item == '':
            continue
        try:
            result.add(int(item))
        except ValueError:
            raise ValueError(f"{flag} must be comma-separated integers, got {text!r} "
                             f"(item {item!r} is not an integer)") from None
    return result


# keep in sync with detect_dataset._validate_fractions: both splitters must
# accept exactly the same fractions, and that helper is private to its module.
def validate_fractions(valid_frac: float, test_frac: float) -> None:
    """Refuse a held-out fraction that is not a real number in [0, 1), and a
    pair that leaves no image for training. Raises ValueError naming the
    flag. A bool is refused too: True would silently read as 1.0."""
    for flag, value in (('--valid', valid_frac), ('--test', test_frac)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
            raise ValueError(f"{flag} must be a number between 0 and 1 (got {value!r})")
        if not 0 <= value < 1:
            raise ValueError(f"{flag} must be at least 0 and below 1 (got {value!r})")
    if valid_frac + test_frac >= 1:
        raise ValueError(f"--valid + --test must leave images for training "
                         f"(got {valid_frac} + {test_frac}, which is 1 or more)")


def validated_inputs(args: argparse.Namespace) -> tuple[set[int], set[int], set[int] | None]:
    """Parse and validate the transect lists, the class filter and the
    fractions from parsed arguments. Returns (val_transects, test_transects,
    include_set), include_set None when every class is kept. Raises
    ValueError naming the flag for a non-integer transect or class id, a
    transect named in both lists, an empty --include-classes, or a fraction
    outside [0, 1)."""
    val_transects = _parse_int_csv(args.val_transects, '--val-transects')
    test_transects = _parse_int_csv(args.test_transects, '--test-transects')
    overlap = val_transects & test_transects
    if overlap:
        raise ValueError(f"--val-transects and --test-transects both name transect(s) {sorted(overlap)}; "
                         "a transect is held out for validation or for test, not both")
    include_set = None
    if args.include_classes:
        include_set = _parse_int_csv(args.include_classes, '--include-classes')
        if not include_set:
            raise ValueError("--include-classes was empty after parsing")
    validate_fractions(args.valid, args.test)
    return val_transects, test_transects, include_set


def load_original_names(src_dir: str) -> dict | None:
    """Class names from <src_dir>/data.yaml, or None when absent or unreadable."""
    yaml_path = os.path.join(src_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if data and 'names' in data:
                    return data['names']
        except Exception as e:
            print(f"Warning: Could not load class names from {yaml_path}: {e}")
    return None


def clip_prefix(basename):
    """Group key for leakage-aware random sampling: strip the trailing
    _T<digits> so frames from the same clip stay together."""
    stem = os.path.splitext(basename)[0]
    m = re.search(r'_T\d+$', stem)
    return stem[:m.start()] if m else stem


def _report_skipped_label_line(label_path: str, lineno: int, first_field: str) -> None:
    """Say which label line was skipped and why: its first field is not a
    class id. Printed, not raised, so one stray line never stops a split."""
    print(f"WARNING: {label_path} line {lineno}: first field {first_field!r} is not a class id; "
          "line skipped")


def label_line_classes(label_path: str) -> list[int]:
    """Class ids, one per line, in a YOLO-seg label file; [] when the file
    is missing or empty. A line whose first field is not an integer is
    skipped and reported with the file and line number."""
    classes = []
    if not os.path.exists(label_path):
        return classes
    with open(label_path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            first = stripped.split()[0]
            try:
                classes.append(int(first))
            except ValueError:
                _report_skipped_label_line(label_path, lineno, first)
    return classes


def build_image_index(src_dir):
    """Return sorted list of image basenames present in all_images/, and the
    label dir path (labels may or may not exist per image)."""
    image_dir = os.path.join(src_dir, 'all_images')
    label_dir = os.path.join(src_dir, 'all_labels')
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Source directory {image_dir} does not exist")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Source directory {label_dir} does not exist")
    images = sorted(os.listdir(image_dir))
    return images, image_dir, label_dir


def label_path_for(image_name: str, label_dir: str) -> str:
    """Path of the YOLO label file for ``image_name`` inside ``label_dir``."""
    stem = os.path.splitext(image_name)[0]
    return os.path.join(label_dir, stem + '.txt')


def load_manifest(manifest_path: str | None) -> dict | None:
    """The manifest at ``manifest_path`` as a dict, or None when there is no
    such file. A file that exists but cannot be read as JSON, is not a JSON
    object, carries a holdout_mode outside HOLDOUT_MODES, or carries a
    'pinned' table that is not {basename: split} with every split one of
    SPLIT_NAMES raises ValueError naming the file: ignoring it would
    re-split those images, which the pins exist to prevent. A manifest with
    no holdout_mode is returned as is and replays as transect."""
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    advice = "fix or remove it before rebuilding, because ignoring it would re-split every image"
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError(f"split manifest {manifest_path} cannot be read as JSON ({e}); {advice}") from e
    if not isinstance(manifest, dict):
        raise ValueError(f"split manifest {manifest_path} must hold a JSON object, "
                         f"got {type(manifest).__name__}; {advice}")
    holdout_mode = manifest.get('holdout_mode')
    if holdout_mode is not None and holdout_mode not in HOLDOUT_MODES:
        raise ValueError(f"split manifest {manifest_path}: holdout_mode must be one of {HOLDOUT_MODES}, "
                         f"got {holdout_mode!r}; {advice}")
    pinned = manifest.get('pinned')
    if pinned is None:
        return manifest
    if not isinstance(pinned, dict):
        raise ValueError(f"split manifest {manifest_path}: 'pinned' must be an object of basename "
                         f"to split, got {type(pinned).__name__}; {advice}")
    bad = [(name, split) for name, split in pinned.items() if split not in SPLIT_NAMES]
    if bad:
        raise ValueError(f"split manifest {manifest_path}: {len(bad)} pin(s) carry a split that is "
                         f"not one of {SPLIT_NAMES}, for example {bad[:3]!r}; {advice}")
    return manifest


def existing_split_basenames(out_dir):
    """For legacy adoption: read basenames already sitting in valid/ and
    test/ image dirs of a previously-built dataset (no manifest present)."""
    result = {}
    for split in ('valid', 'test'):
        img_dir = os.path.join(out_dir, split, 'images')
        if os.path.isdir(img_dir):
            result[split] = sorted(os.listdir(img_dir))
        else:
            result[split] = None
    return result


def assign_fresh_transect(images, val_transects, test_transects):
    """Rule 3: assign every image by parse_transect. Returns dict basename->split."""
    pins = {}
    for img in images:
        t = parse_transect(img)
        if t in test_transects:
            pins[img] = 'test'
        elif t in val_transects:
            pins[img] = 'valid'
        else:
            pins[img] = 'train'
    return pins


def _grouped_random_slice(candidates, frac, rng):
    """Pick a rng-seeded slice of `candidates` (basenames) sized ~frac,
    grouped by clip_prefix so whole clips move together (limits frame
    leakage between adjacent video frames). Groups are shuffled and added
    smallest-first so a handful of oversized groups cannot blow the slice
    far past the target. Falls back to an ungrouped random sample if
    grouping is too coarse to approximate the target at all (e.g. one or
    two giant groups covering everything) so the slice is never empty or a
    near-total sweep just because groups are coarse."""
    groups = {}
    for c in candidates:
        groups.setdefault(clip_prefix(c), []).append(c)
    target_n = int(round(len(candidates) * frac))
    if target_n <= 0:
        # A tiny pool can round a requested nonzero fraction down to 0
        # (e.g. 5 candidates * 0.1 -> round(0.5) == 0). A holdout fraction
        # was actually requested and candidates exist, so floor to 1 rather
        # than silently returning an empty holdout.
        if frac > 0 and len(candidates) > 0:
            target_n = 1
        else:
            return set()
    if len(groups) <= 1:
        return set(rng.sample(candidates, min(target_n, max(1, len(candidates) - 1))))

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    group_keys.sort(key=lambda gk: len(groups[gk]))
    picked = []
    for gk in group_keys:
        if len(picked) >= target_n:
            break
        picked.extend(groups[gk])
    # Grouping overshot badly (e.g. the only available group is far larger
    # than the target): fall back to an ungrouped sample sized to target_n.
    if len(picked) > 2 * target_n and len(picked) >= len(candidates) // 3:
        picked = rng.sample(candidates, target_n)
    return set(picked)


def assign_pinned_random(images, val_transects, test_transects, valid_frac, test_frac, seed):
    """Rule 4: test by transect (leak-free), valid = random slice of the rest."""
    pins = {}
    test_set = set()
    remaining = []
    for img in images:
        t = parse_transect(img)
        if t in test_transects:
            pins[img] = 'test'
            test_set.add(img)
        else:
            remaining.append(img)
    rng = random.Random(seed)
    valid_set = _grouped_random_slice(remaining, valid_frac, rng)
    for img in remaining:
        pins[img] = 'valid' if img in valid_set else 'train'
    return pins


def assign_degrade_random(images, valid_frac, test_frac, seed):
    """Rule 3b: no transect info usable -> fully random holdout (grouped)."""
    rng = random.Random(seed)
    test_set = _grouped_random_slice(images, test_frac, rng)
    remaining = [img for img in images if img not in test_set]
    rng2 = random.Random(seed)
    valid_set = _grouped_random_slice(remaining, valid_frac, rng2)
    pins = {}
    for img in images:
        if img in test_set:
            pins[img] = 'test'
        elif img in valid_set:
            pins[img] = 'valid'
        else:
            pins[img] = 'train'
    return pins


def build_manifest(holdout_mode: str, val_transects: set[int], test_transects: set[int],
                   pins: dict[str, str], *, seed: str, valid_frac: float, test_frac: float,
                   degrade_warning: str | None = None, created_at: str | None = None) -> dict:
    """The split_manifest.json document for a finished assignment.

    Parameters: holdout_mode, the policy that produced the assignment
    (transect, transect-degraded, pinned-random or adopted-random);
    val_transects and test_transects, the transect numbers the run treats as
    validation and test; pins, {basename: split} exactly as written to disk;
    seed, the --rand value as a non-empty string; valid_frac and test_frac,
    the --valid and --test values, recorded as given (they reach only images
    assigned by a random draw in this run); degrade_warning, the text of a
    degraded holdout, or None; created_at, the timestamp of the first
    manifest, kept across rebuilds, or None for a fresh one.

    Returns the document: every version 1 key unchanged, plus the version 2
    record (seed, valid_frac, test_frac, split_fingerprint,
    split_fingerprint_scheme, counts). Raises TypeError for a seed that is
    not a string, ValueError for an empty seed, a bad fraction or a pin
    outside SPLIT_NAMES.
    """
    if not isinstance(seed, str):
        raise TypeError(f"seed must be recorded as a string, got {type(seed).__name__}")
    if not seed.strip():
        raise ValueError("seed must be a non-empty string")
    validate_fractions(valid_frac, test_frac)
    fingerprint = split_fingerprint(pins)
    now = ast_now()
    m = {
        'schema_version': SCHEMA_VERSION,
        'created_at': created_at or now,
        'updated_at': now,
        'holdout_mode': holdout_mode,
        'transect_rule': 'stem _T + 3 digits, first digit = transect',
        'holdout_transects': {'val': sorted(val_transects), 'test': sorted(test_transects)},
        'seed': seed,
        'valid_frac': float(valid_frac),
        'test_frac': float(test_frac),
        'split_fingerprint': fingerprint,
        'split_fingerprint_scheme': SPLIT_FINGERPRINT_SCHEME,
        'counts': {s: sum(1 for v in pins.values() if v == s) for s in SPLIT_NAMES},
        'pinned': dict(sorted(pins.items())),
    }
    if degrade_warning:
        m['degrade_warning'] = degrade_warning
    return m


def replay_manifest(manifest, images, val_transects, test_transects):
    """Rule 1: rebuild pins honoring every existing manifest entry; images
    not yet in the manifest get assigned per the manifest's own holdout_mode.
    Returns (pins, warned_missing) where warned_missing lists manifest
    basenames absent from src (skipped, never reassigned)."""
    prior_pins = manifest.get('pinned', {})
    holdout_mode = manifest.get('holdout_mode', 'transect')
    image_set = set(images)
    pins = {}
    warned_missing = []
    for basename, split in prior_pins.items():
        if basename in image_set:
            pins[basename] = split
        else:
            warned_missing.append(basename)
    new_images = [img for img in images if img not in pins]
    for img in new_images:
        if holdout_mode == 'transect':
            t = parse_transect(img)
            if t in test_transects:
                pins[img] = 'test'
            elif t in val_transects:
                pins[img] = 'valid'
            else:
                pins[img] = 'train'
        else:
            # adopted-random / pinned-random / transect-degraded: new frames -> train
            pins[img] = 'train'
    return pins, warned_missing


def can_auto_upgrade(manifest, images, val_transects, test_transects):
    """Rule 1 auto-upgrade carve-out: a transect-degraded manifest upgrades
    to a clean transect split once the source has labeled frames on BOTH the
    previously-empty val AND test holdout transect(s) (either one still
    empty means the upgrade would silently ship a split missing a holdout,
    so both must be present)."""
    if manifest.get('holdout_mode') != 'transect-degraded':
        return False
    has_val = any(parse_transect(img) in val_transects for img in images)
    has_test = any(parse_transect(img) in test_transects for img in images)
    return has_val and has_test


def apply_min_samples_and_coverage(pins: dict[str, str], label_dir: str, min_samples: int,
                                   include_set: set[int] | None,
                                   original_names: dict | None) -> tuple[set[int], list[str]]:
    """Compute, for the TRAIN split only (min_samples semantics match
    bal_train_test_split: counts are over images whose label MODE is that
    class), which class ids to drop. Each label file is read once, so a
    skipped line is reported once here. Returns (dropped_ids, warnings)."""
    warnings = []
    train_images = [img for img, split in pins.items() if split == 'train']

    def image_mode_class(img: str) -> int | None:
        """The most common class id in the image's label file after the
        include filter, or None when it has no kept class."""
        label_path = label_path_for(img, label_dir)
        classes = label_line_classes(label_path)
        if include_set is not None:
            classes = [c for c in classes if c in include_set]
        if not classes:
            return None
        return _mode(classes)

    mode_by_image = {img: image_mode_class(img) for img in pins}
    train_counts = {}
    for img in train_images:
        c = mode_by_image[img]
        if c is not None:
            train_counts[c] = train_counts.get(c, 0) + 1

    all_ids = set(original_names.keys()) if original_names else set(train_counts.keys())
    if include_set is not None:
        all_ids = {i for i in all_ids if i in include_set}

    dropped = set()
    # --min_samples: too few labeled TRAIN images for the class.
    for cid in all_ids:
        count = train_counts.get(cid, 0)
        if count < min_samples:
            dropped.add(cid)
            warnings.append(
                f"class {cid} has only {count} train images (< min_samples={min_samples}); dropping")

    # Rule 5b: per-class train-coverage: a class with zero TRAIN instances
    # (all its frames sit in valid/test) is dropped even if it wasn't
    # already caught above, and we name the offending transect(s).
    all_split_counts = {}
    for img, split in pins.items():
        c = mode_by_image[img]
        if c is not None:
            all_split_counts.setdefault(c, {'train': 0, 'valid': 0, 'test': 0})
            all_split_counts[c][split] += 1

    for cid, counts in all_split_counts.items():
        if include_set is not None and cid not in include_set:
            continue
        if counts.get('train', 0) == 0 and (counts.get('valid', 0) or counts.get('test', 0)):
            if cid not in dropped:
                dropped.add(cid)
            offending = set()
            for img, split in pins.items():
                if split == 'train':
                    continue
                c = mode_by_image[img]
                if c == cid:
                    t = parse_transect(img)
                    if t is not None:
                        offending.add(t)
            t_str = ','.join(str(t) for t in sorted(offending)) if offending else 'unknown'
            warnings.append(
                f"class {cid} has zero train instances (all frames on transect(s) {t_str}); dropping from training")

    return dropped, warnings


def create_data_yaml(write_dir, path_for_yaml, original_names, dropped_ids, include_set):
    """Write data.yaml/test.yaml into `write_dir` (the tmp build dir), but
    stamp `path:` as `path_for_yaml` (the FINAL out_dir the files get
    renamed into) so the yaml is correct once the atomic rename completes."""
    if original_names:
        names = dict(original_names)
    else:
        names = {}
    filtered = {}
    for cid, nm in names.items():
        if cid in dropped_ids:
            filtered[cid] = None
        elif include_set is not None and cid not in include_set:
            filtered[cid] = None
        else:
            filtered[cid] = nm
    data_yaml = {
        'path': os.path.abspath(path_for_yaml),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': filtered,
    }
    with open(os.path.join(write_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    # Vestigial test.yaml, kept for backward compatibility only. Never read
    # by evaluate_run.py (which resolves data.yaml + --split test).
    test_yaml = dict(data_yaml)
    test_yaml['val'] = 'test/images'
    with open(os.path.join(write_dir, 'test.yaml'), 'w') as f:
        yaml.dump(test_yaml, f, sort_keys=False)
    return data_yaml


def copy_label_filtered(src_path: str, dst_path: str, include_set: set[int] | None,
                        dropped_ids: set[int]) -> None:
    """Copy the label file at ``src_path`` to ``dst_path`` keeping only lines
    whose class is in ``include_set`` (None keeps all) and not in
    ``dropped_ids``; a missing source writes an empty label file. A line
    whose first field is not a class id is skipped and reported."""
    kept = []
    if os.path.exists(src_path):
        with open(src_path, 'r') as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                first = stripped.split()[0]
                try:
                    cid = int(first)
                except ValueError:
                    _report_skipped_label_line(src_path, lineno, first)
                    continue
                if include_set is not None and cid not in include_set:
                    continue
                if cid in dropped_ids:
                    continue
                kept.append(line if line.endswith('\n') else line + '\n')
    with open(dst_path, 'w') as f:
        f.writelines(kept)


def write_dataset(tmp_dir: str, pins: dict[str, str], image_dir: str, label_dir: str,
                  include_set: set[int] | None, dropped_ids: set[int]) -> None:
    """Place every pinned image and its filtered label file under
    <tmp_dir>/<split>/images and labels; an image missing from ``image_dir``
    is skipped."""
    for split in SPLIT_NAMES:
        os.makedirs(os.path.join(tmp_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(tmp_dir, split, 'labels'), exist_ok=True)
    for img, split in sorted(pins.items()):
        src_img = os.path.join(image_dir, img)
        if not os.path.exists(src_img):
            continue
        dst_img = os.path.join(tmp_dir, split, 'images', img)
        shutil.copy(src_img, dst_img)
        stem = os.path.splitext(img)[0]
        src_lbl = os.path.join(label_dir, stem + '.txt')
        dst_lbl = os.path.join(tmp_dir, split, 'labels', stem + '.txt')
        copy_label_filtered(src_lbl, dst_lbl, include_set, dropped_ids)


def main(argv: list[str] | None = None) -> int:
    """Entry point: validate the inputs, assign every image a split honoring
    any existing manifest, write the dataset and its manifest atomically.
    Returns 0. Exits 2 on a refused input, an unreadable manifest or an
    empty train split; raises OSError when the destination cannot be
    written."""
    args = arg_parse(argv)
    src_dir = args.src_dir
    out_dir = args.out_dir
    manifest_path = args.manifest_path or os.path.join(out_dir, MANIFEST_NAME)
    # The seed is recorded as a string, the form both splitters' manifests share.
    seed = str(args.random_state)

    try:
        val_transects, test_transects, include_set = validated_inputs(args)
        manifest = load_manifest(manifest_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    images, image_dir, label_dir = build_image_index(src_dir)
    original_names = load_original_names(src_dir)

    degrade_warning = None
    created_at = None

    if manifest is not None:
        # Rule 1: manifest exists.
        if can_auto_upgrade(manifest, images, val_transects, test_transects):
            print(f"[pinned_split] auto-upgrading transect-degraded manifest to clean transect split "
                  f"(val transect(s) {sorted(val_transects)} and test transect(s) {sorted(test_transects)} now labeled)")
            pins = assign_fresh_transect(images, val_transects, test_transects)
            holdout_mode = 'transect'
            created_at = manifest.get('created_at')
            # Safety net: re-check both holdouts after the fresh assignment.
            # can_auto_upgrade already requires labeled frames on both, but
            # verify against the actual pins (not just presence in `images`)
            # so a split with an empty val or test is never stamped 'transect'.
            has_test = any(s == 'test' for s in pins.values())
            has_valid = any(s == 'valid' for s in pins.values())
            if not has_test or not has_valid:
                empty_names = []
                if not has_test:
                    empty_names.append('T' + '/T'.join(str(t) for t in sorted(test_transects)))
                if not has_valid:
                    empty_names.append('T' + '/T'.join(str(t) for t in sorted(val_transects)))
                degrade_warning = (
                    f"{' and '.join(empty_names)} unlabeled: using a random holdout for now; "
                    f"will upgrade to the transect split once labeled.")
                print(f"WARNING: {degrade_warning}")
                pins = assign_degrade_random(images, args.valid, args.test, args.random_state)
                holdout_mode = 'transect-degraded'
        else:
            pins, warned_missing = replay_manifest(manifest, images, val_transects, test_transects)
            holdout_mode = manifest.get('holdout_mode', 'transect')
            created_at = manifest.get('created_at')
            for w in warned_missing:
                print(f"WARNING: pinned image '{w}' missing from --src; skipped (not reassigned)")
            degrade_warning = manifest.get('degrade_warning')
    else:
        legacy = existing_split_basenames(out_dir)
        if legacy['valid'] is not None or legacy['test'] is not None:
            # Rule 2: legacy adoption.
            print("[pinned_split] no manifest found but dataset/valid or dataset/test exists; "
                  "adopting the existing split byte-for-byte (holdout_mode=adopted-random)")
            pins = {}
            adopted_valid = set(legacy['valid'] or [])
            adopted_test = set(legacy['test'] or [])
            for img in images:
                if img in adopted_test:
                    pins[img] = 'test'
                elif img in adopted_valid:
                    pins[img] = 'valid'
                else:
                    pins[img] = 'train'
            holdout_mode = 'adopted-random'
        else:
            # Rule 3 / 4: fresh split.
            if args.holdout_mode == 'pinned-random':
                pins = assign_pinned_random(images, val_transects, test_transects,
                                             args.valid, args.test, args.random_state)
                holdout_mode = 'pinned-random'
            else:
                pins = assign_fresh_transect(images, val_transects, test_transects)
                holdout_mode = 'transect'
                has_test = any(s == 'test' for s in pins.values())
                has_valid = any(s == 'valid' for s in pins.values())
                if not has_test or not has_valid:
                    # Rule 3b: graceful degrade.
                    empty_names = []
                    if not has_test:
                        empty_names.append('T' + '/T'.join(str(t) for t in sorted(test_transects)))
                    if not has_valid:
                        empty_names.append('T' + '/T'.join(str(t) for t in sorted(val_transects)))
                    degrade_warning = (
                        f"{' and '.join(empty_names)} unlabeled: using a random holdout for now; "
                        f"will upgrade to the transect split once labeled.")
                    print(f"WARNING: {degrade_warning}")
                    pins = assign_degrade_random(images, args.valid, args.test, args.random_state)
                    holdout_mode = 'transect-degraded'

    # Rule 6: empty-split guard (hard-fail ONLY on empty TRAIN).
    n_train = sum(1 for s in pins.values() if s == 'train')
    n_valid = sum(1 for s in pins.values() if s == 'valid')
    n_test = sum(1 for s in pins.values() if s == 'test')
    if n_train == 0:
        print(f"ERROR: pinned split produced an empty train set (train=0 valid={n_valid} test={n_test}); "
              f"no labeled data to train on.")
        sys.exit(2)

    dropped_ids, cov_warnings = apply_min_samples_and_coverage(
        pins, label_dir, args.min_samples, include_set, original_names)
    for w in cov_warnings:
        print(f"WARNING: {w}")

    # Never write a partial dataset: build in a tmp dir, then rename into place.
    try:
        os.makedirs(out_dir, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(prefix='.pinned_split_tmp_', dir=out_dir)
    except OSError as e:
        raise OSError(f"cannot create the dataset folder {out_dir} or a build folder inside it "
                      f"({e}); check that the folder exists and is writable") from e
    try:
        write_dataset(tmp_dir, pins, image_dir, label_dir, include_set, dropped_ids)
        data_yaml = create_data_yaml(tmp_dir, out_dir, original_names, dropped_ids, include_set)

        manifest_out = build_manifest(holdout_mode, val_transects, test_transects, pins,
                                       seed=seed, valid_frac=args.valid, test_frac=args.test,
                                       degrade_warning=degrade_warning, created_at=created_at)
        with open(os.path.join(tmp_dir, MANIFEST_NAME), 'w', encoding='utf-8') as f:
            json.dump(manifest_out, f, indent=2, sort_keys=False)
            f.write('\n')

        for split in SPLIT_NAMES:
            final = os.path.join(out_dir, split)
            if os.path.exists(final):
                shutil.rmtree(final)
            shutil.move(os.path.join(tmp_dir, split), final)
        for fname in ('data.yaml', 'test.yaml', MANIFEST_NAME):
            shutil.move(os.path.join(tmp_dir, fname), os.path.join(out_dir, fname))
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[pinned_split] holdout_mode={holdout_mode} train={n_train} valid={n_valid} test={n_test} "
          f"seed={seed} split_fingerprint={manifest_out['split_fingerprint']}")
    if dropped_ids:
        print(f"[pinned_split] classes dropped from training: {sorted(dropped_ids)}")
    print(f"[pinned_split] wrote {out_dir}/data.yaml, test.yaml, split_manifest.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
