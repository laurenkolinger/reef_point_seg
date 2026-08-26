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

See scripts/TCRMPtrain_oceankindCV/tests/test_pinned_split.py and the task
brief for the full rule set (manifest replay, legacy adoption, graceful
degrade + auto-upgrade, pinned-random, min_samples, per-class train
coverage, empty-train guard).

Run: env/bin/python scripts/TCRMPtrain_oceankindCV/src/pinned_split.py --src <export> --out <step6_dir>/dataset
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from statistics import mode as _mode

import yaml

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


SCHEMA_VERSION = 1


def arg_parse(argv=None):
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


def _parse_int_csv(s):
    return {int(x) for x in s.split(',') if x.strip() != ''}


def load_original_names(src_dir):
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
    _T<digits> so frames from the same clip land together."""
    stem = os.path.splitext(basename)[0]
    m = re.search(r'_T\d+$', stem)
    return stem[:m.start()] if m else stem


def label_line_classes(label_path):
    """Class IDs present in a YOLO-seg label file (empty set if empty/missing)."""
    classes = []
    if not os.path.exists(label_path):
        return classes
    with open(label_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                classes.append(int(stripped.split()[0]))
            except ValueError:
                continue
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


def label_path_for(image_name, label_dir):
    stem = os.path.splitext(image_name)[0]
    return os.path.join(label_dir, stem + '.txt')


def load_manifest(manifest_path):
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return None


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


def build_manifest(holdout_mode, val_transects, test_transects, pins, degrade_warning=None, created_at=None):
    now = ast_now()
    m = {
        'schema_version': SCHEMA_VERSION,
        'created_at': created_at or now,
        'updated_at': now,
        'holdout_mode': holdout_mode,
        'transect_rule': 'stem _T + 3 digits, first digit = transect',
        'holdout_transects': {'val': sorted(val_transects), 'test': sorted(test_transects)},
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


def apply_min_samples_and_coverage(pins, label_dir, min_samples, include_set, original_names):
    """Compute, for the TRAIN split only (min_samples semantics match
    bal_train_test_split: counts are over images whose label MODE is that
    class), which class ids to drop. Returns (dropped_ids, warnings)."""
    warnings = []
    train_images = [img for img, split in pins.items() if split == 'train']

    def image_mode_class(img):
        label_path = label_path_for(img, label_dir)
        classes = label_line_classes(label_path)
        if include_set is not None:
            classes = [c for c in classes if c in include_set]
        if not classes:
            return None
        return _mode(classes)

    train_counts = {}
    for img in train_images:
        c = image_mode_class(img)
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
    # (all its frames landed in valid/test) is dropped even if it wasn't
    # already caught above, and we name the offending transect(s).
    all_split_counts = {}
    for img, split in pins.items():
        c = image_mode_class(img)
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
                c = image_mode_class(img)
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


def copy_label_filtered(src_path, dst_path, include_set, dropped_ids):
    kept = []
    if os.path.exists(src_path):
        with open(src_path, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    cid = int(stripped.split()[0])
                except ValueError:
                    continue
                if include_set is not None and cid not in include_set:
                    continue
                if cid in dropped_ids:
                    continue
                kept.append(line if line.endswith('\n') else line + '\n')
    with open(dst_path, 'w') as f:
        f.writelines(kept)


def write_dataset(tmp_dir, pins, image_dir, label_dir, include_set, dropped_ids):
    for split in ('train', 'valid', 'test'):
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


def main(argv=None):
    args = arg_parse(argv)
    src_dir = args.src_dir
    out_dir = args.out_dir
    manifest_path = args.manifest_path or os.path.join(out_dir, 'split_manifest.json')

    val_transects = _parse_int_csv(args.val_transects)
    test_transects = _parse_int_csv(args.test_transects)

    include_set = None
    if args.include_classes:
        include_set = _parse_int_csv(args.include_classes)
        if not include_set:
            print("ERROR: --include-classes was empty after parsing")
            sys.exit(2)

    images, image_dir, label_dir = build_image_index(src_dir)
    original_names = load_original_names(src_dir)

    manifest = load_manifest(manifest_path)
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
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix='.pinned_split_tmp_', dir=out_dir)
    try:
        write_dataset(tmp_dir, pins, image_dir, label_dir, include_set, dropped_ids)
        data_yaml = create_data_yaml(tmp_dir, out_dir, original_names, dropped_ids, include_set)

        manifest_out = build_manifest(holdout_mode, val_transects, test_transects, pins,
                                       degrade_warning=degrade_warning, created_at=created_at)
        with open(os.path.join(tmp_dir, 'split_manifest.json'), 'w') as f:
            json.dump(manifest_out, f, indent=2, sort_keys=False)
            f.write('\n')

        for split in ('train', 'valid', 'test'):
            final = os.path.join(out_dir, split)
            if os.path.exists(final):
                shutil.rmtree(final)
            shutil.move(os.path.join(tmp_dir, split), final)
        for fname in ('data.yaml', 'test.yaml', 'split_manifest.json'):
            shutil.move(os.path.join(tmp_dir, fname), os.path.join(out_dir, fname))
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[pinned_split] holdout_mode={holdout_mode} train={n_train} valid={n_valid} test={n_test}")
    if dropped_ids:
        print(f"[pinned_split] classes dropped from training: {sorted(dropped_ids)}")
    print(f"[pinned_split] wrote {out_dir}/data.yaml, test.yaml, split_manifest.json")
    return 0


if __name__ == '__main__':
    sys.exit(main())
