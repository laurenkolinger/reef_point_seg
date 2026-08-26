"""
YOLO Segmentation format export.

Produces:
  all_images/   — symlinks (or copies) of raw images
  all_labels/   — one .txt per image with normalized polygon coords
  data.yaml     — YOLO dataset config
  class_map.json — persistent species -> class_id mapping
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


def load_class_map(export_dir):
    """Load persistent class_map.json, or return empty dict."""
    path = os.path.join(export_dir, 'class_map.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_class_map(export_dir, class_map):
    """Save class_map.json."""
    path = os.path.join(export_dir, 'class_map.json')
    with open(path, 'w') as f:
        json.dump(class_map, f, indent=2)


def get_or_create_class_id(class_map, species_code):
    """Get class_id for a species, creating a new one if needed."""
    if species_code in class_map:
        return class_map[species_code]
    new_id = len(class_map)
    class_map[species_code] = new_id
    return new_id


def mask_to_yolo_line(polygon_norm, class_id):
    """Convert one normalized polygon to a YOLO segmentation label line.

    Args:
        polygon_norm: flat list [x1, y1, x2, y2, ...] normalized 0-1
        class_id: integer class ID

    Returns:
        string line for .txt label file
    """
    coords = ' '.join(f'{v:.6f}' for v in polygon_norm)
    return f'{class_id} {coords}'


def _project_id_from_dir(export_dir):
    """Nearest run_* path segment of export_dir (with the run_ prefix
    stripped so it matches project.json's unprefixed 'id' field, which is
    what the label-coverage matrix keys registry rows by), else its parent
    folder name. Mirrors app.py's _derive_project_id (kept local here since
    export_yolo must not import the Flask app module)."""
    parts = os.path.abspath(export_dir).split(os.sep)
    for p in reversed(parts):
        if p.startswith('run_'):
            return p[len('run_'):]
    return os.path.basename(os.path.dirname(os.path.abspath(export_dir))) or 'project'


def _upsert_registry_for_frame(filename, seg, export_dir):
    """Upsert every mask of this frame (accepted AND rejected) into the
    permanent cross-project mask registry, keyed by canonical uid. Never
    raises: a registry hiccup must never break the export.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        scripts_dir = os.path.dirname(os.path.dirname(here))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from _reefreview.mask_registry import MaskRegistry, build_registry_record

        registry_root = os.environ.get('TCRMP_MASK_REGISTRY_DIR') or MaskRegistry().root
        reg = MaskRegistry(root=registry_root)
        # The orchestrator stamps TCRMP_PROJECT_ID (project.json's unprefixed
        # 'id') into os.environ for the annotator apps; the label-coverage
        # matrix (_matrix/builder.py) groups/looks up registry rows by that
        # SAME unprefixed id. Prefer it so registry writes and matrix reads
        # agree on the key; fall back to the (also unprefixed) dir-derived id
        # for a standalone launch with no orchestrator env.
        project_id = os.environ.get('TCRMP_PROJECT_ID') or _project_id_from_dir(export_dir)
        project_name = project_id

        for mask in seg.get('masks', []):
            try:
                rec = build_registry_record(filename, mask, project_id, project_name)
                if rec is None:
                    continue
                reg.upsert(rec)
            except Exception as e:
                log.warning("Registry upsert failed for a mask in %s (non-fatal): %s",
                            filename, e)
    except Exception as e:
        log.warning("Registry upsert failed for %s (non-fatal): %s", filename, e)


def update_data_yaml(export_dir, class_map):
    """Write/overwrite data.yaml with current class mapping."""
    # Sort by class_id for deterministic output
    sorted_names = sorted(class_map.items(), key=lambda x: x[1])

    lines = [
        f'path: {os.path.abspath(export_dir)}',
        'train: all_images',
        'val: all_images',
        f'nc: {len(sorted_names)}',
        'names:',
    ]
    for species, cid in sorted_names:
        lines.append(f'  {cid}: {species}')

    path = os.path.join(export_dir, 'data.yaml')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log.info("Updated data.yaml with %d classes", len(sorted_names))


def export_batch(segmentations, export_dir, class_map, symlink=True):
    """Export a batch of segmented images to YOLO format.

    Args:
        segmentations: dict of {filename: segmentation_dict} where each has
                       'image_path' (absolute), 'masks' list, 'image_width', 'image_height'
        export_dir: root output directory
        class_map: species -> class_id dict (will be extended)
        symlink: if True, symlink images instead of copying

    Returns:
        dict with export stats
    """
    images_dir = os.path.join(export_dir, 'all_images')
    labels_dir = os.path.join(export_dir, 'all_labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    exported_images = 0
    exported_masks = 0
    blocked_unlabeled = 0
    new_classes = 0

    for filename, seg in segmentations.items():
        # Feed the cross-project canonical mask registry: every mask of this
        # frame (accepted AND rejected) gets a census row, keyed on its
        # content-stable uid. Runs before any skip guard below so a frame
        # whose masks are all rejected (or otherwise excluded from export)
        # still lands in the registry. Never fatal to the export.
        _upsert_registry_for_frame(filename, seg, export_dir)

        image_path = seg.get('image_path_abs')
        if not image_path or not os.path.exists(image_path):
            log.warning("Skipping %s: image not found at %s", filename, image_path)
            continue

        # Filter to accepted masks only. REVIEW-flagged masks are NEVER trained —
        # they go to the expert-review export instead, so exclude them here even
        # if a user accepted one.
        accepted = [m for m in seg.get('masks', [])
                    if m.get('status') == 'accepted'
                    and not m.get('review') and m.get('species') != 'REVIEW']

        # Backstop: an accepted mask with no species assigned must never reach
        # the training set (it would silently become a dropped/uncounted
        # label). Hard-skip it here and count it so nothing vanishes
        # unreported.
        unlabeled = [m for m in accepted if (m.get('species', '') or '') == '']
        if unlabeled:
            blocked_unlabeled += len(unlabeled)
            accepted = [m for m in accepted if (m.get('species', '') or '') != '']

        if not accepted:
            continue

        # Image file (symlink or copy)
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        img_dest = os.path.join(images_dir, filename)
        if not os.path.exists(img_dest):
            if symlink:
                os.symlink(os.path.abspath(image_path), img_dest)
            else:
                import shutil
                shutil.copy2(image_path, img_dest)

        # Label file
        label_lines = []
        for mask in accepted:
            species = mask.get('species', 'UNK')
            prev_count = len(class_map)
            cid = get_or_create_class_id(class_map, species)
            if len(class_map) > prev_count:
                new_classes += 1

            # Use the first (largest) polygon for YOLO label
            polygons_norm = mask.get('polygon_norm', [])
            for poly in polygons_norm:
                if len(poly) >= 6:  # at least 3 points
                    label_lines.append(mask_to_yolo_line(poly, cid))
                    exported_masks += 1

        label_path = os.path.join(labels_dir, stem + '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines) + '\n' if label_lines else '')

        exported_images += 1

    # Save updated class map and data.yaml
    save_class_map(export_dir, class_map)
    update_data_yaml(export_dir, class_map)

    # Log export
    log_path = os.path.join(export_dir, 'export_log.txt')
    with open(log_path, 'a') as f:
        f.write(f'{datetime.now().isoformat()} — exported {exported_images} images, '
                f'{exported_masks} masks, {blocked_unlabeled} blocked-unlabeled, '
                f'{new_classes} new classes\n')

    stats = {
        'exported_images': exported_images,
        'exported_masks': exported_masks,
        'blocked_unlabeled': blocked_unlabeled,
        'new_classes': new_classes,
        'total_classes': len(class_map),
    }
    log.info("Export complete: %s", stats)
    return stats
