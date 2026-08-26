"""
Custom image imports for the combined annotator (Step 4, 4.test).

Lets an operator bring ANY image into the ACTIVE run's annotation queue,
alongside the pipeline-routed frames, without touching routing output
(routed_input/ can be invalidated and rebuilt by the orchestrator, so
imports must never live there). Everything an import owns lives under the
export dir (the run's step4test_combinedAnnotate/ folder):

  custom_imports/
    originals/            verbatim copies of the files the user supplied
    raw/                  the DELIVERED annotation images (see lores rule)
    imports_manifest.json one entry per import (dims, scale, provenance)

Lores rule (same contract as pipeline frames): SAM3 resizes its input to
1008px, so a 4K frame annotated at full size produces junk masks. The
pipeline delivers a <=1920px lores twin of every oversized frame and scales
prompt coordinates by (delivered long edge / original long edge). Imports
follow the identical rule: any import whose long edge exceeds LORES_LONG_EDGE
is downscaled into custom_imports/raw/ as <stem>_lores.jpg (LANCZOS, quality
92, mirroring make_lores_variants.py) and its manifest entry records the
scale so any coordinate measured on the delivered image maps back to the
original as (x / scale, y / scale). Imports carry no incoming points, so
nothing else needs scaling: all annotation happens directly on the delivered
image. An already-small import is re-encoded at scale 1.0.

Every delivered image is EXIF-transposed before saving: the browser canvas
honors EXIF orientation but cv2.imread does not, and the annotator uses both
on the same file, so orientation is baked into the pixels to keep them in
agreement (phone photos are the common case here; TCRMP clip frames carry no
EXIF rotation).

Delivered filenames are prefixed "custom_" and deduped, so an import can
never collide with (or shadow) a routed pipeline frame in the segmentations
store, the review queue, or the YOLO export's all_images/.

The mask registry and label matrix are only ever touched through the normal
export path: an imported frame's seg_dict flows through the same
export_batch/registry-upsert hooks as a routed frame, and nothing here
writes to either.

Stdlib + PIL only. No pytest; tests/test_custom_imports.py drives this.
"""

import json
import os
import shutil
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageOps

LORES_LONG_EDGE = 1920           # same long edge as make_lores_variants.py
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png'}
SUBDIR = 'custom_imports'

# AST is UTC-4, no DST. All timestamps in this module are AST (house rule).
_AST = timezone(timedelta(hours=-4), name='AST')


def _now_ast():
    return datetime.now(_AST)


def imports_dir(export_dir):
    return os.path.join(export_dir, SUBDIR)


def manifest_path(export_dir):
    return os.path.join(imports_dir(export_dir), 'imports_manifest.json')


def load_manifest(export_dir):
    """All import entries for this run, oldest first. [] when none/corrupt."""
    path = manifest_path(export_dir)
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        entries = data.get('imports', [])
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def _save_manifest(export_dir, entries):
    path = manifest_path(export_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({
            'schema': 1,
            'updated_at': _now_ast().strftime('%Y-%m-%d %H:%M:%S AST'),
            'imports': entries,
        }, f, indent=1)
    os.replace(tmp, path)


def _sanitize_stem(name):
    """Filesystem-safe stem from a user-supplied filename (basename only)."""
    stem, _ = os.path.splitext(os.path.basename(name or ''))
    clean = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in stem)
    return clean.strip('._') or 'image'


def _unique(name, taken):
    """Dedupe a delivered filename against every name already in use."""
    if name not in taken:
        return name
    stem, ext = os.path.splitext(name)
    n = 2
    while f"{stem}_{n}{ext}" in taken:
        n += 1
    return f"{stem}_{n}{ext}"


def ingest_files(export_dir, items, taken_names=None, long_edge=LORES_LONG_EDGE,
                 reviewer='', log=print):
    """Copy user images into the run and apply the lores rule.

    items: list of (original_filename, source) where source is either a
    filesystem path (str) or raw bytes (an upload). Returns
    (entries, skipped): manifest entries for every ingested file, and
    [(name, reason)] for every refused one. Never raises for a bad file;
    it lands in skipped instead.
    """
    entries = []
    skipped = []
    existing = load_manifest(export_dir)
    taken = set(taken_names or [])
    taken.update(e.get('filename', '') for e in existing)

    orig_dir = os.path.join(imports_dir(export_dir), 'originals')
    raw_dir = os.path.join(imports_dir(export_dir), 'raw')

    for orig_name, source in items:
        ext = os.path.splitext(str(orig_name))[1].lower()
        if ext not in ALLOWED_EXTS:
            skipped.append((str(orig_name), f'unsupported extension {ext or "(none)"}'))
            continue
        if isinstance(source, str) and not os.path.isfile(source):
            skipped.append((str(orig_name), 'no such file on this machine'))
            continue

        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)

        stem = _sanitize_stem(orig_name)
        original_name = _unique(stem + ext, {os.path.basename(e.get('original_file', ''))
                                             for e in existing} |
                                            {os.path.basename(e.get('original_file', ''))
                                             for e in entries})
        original_path = os.path.join(orig_dir, original_name)
        try:
            if isinstance(source, (bytes, bytearray)):
                with open(original_path, 'wb') as f:
                    f.write(source)
            else:
                shutil.copy2(source, original_path)
        except Exception as e:
            skipped.append((str(orig_name), f'copy failed: {e}'))
            continue

        try:
            with Image.open(original_path) as im:
                im = ImageOps.exif_transpose(im)
                ow, oh = im.size
                if max(ow, oh) > long_edge:
                    scale = long_edge / float(max(ow, oh))
                    dw, dh = round(ow * scale), round(oh * scale)
                    delivered_name = _unique(f'custom_{stem}_lores.jpg', taken)
                    delivered_path = os.path.join(raw_dir, delivered_name)
                    im.convert('RGB').resize((dw, dh), Image.LANCZOS).save(
                        delivered_path, 'JPEG', quality=92)
                    # Self-correcting scale from the ACTUAL delivered dims,
                    # same policy as placePoints' lores_delivery().
                    scale = max(dw, dh) / float(max(ow, oh))
                else:
                    dw, dh, scale = ow, oh, 1.0
                    delivered_name = _unique(f'custom_{stem}.jpg', taken)
                    delivered_path = os.path.join(raw_dir, delivered_name)
                    im.convert('RGB').save(delivered_path, 'JPEG', quality=95)
        except Exception as e:
            skipped.append((str(orig_name), f'not a readable image: {e}'))
            try:
                os.remove(original_path)
            except OSError:
                pass
            continue

        taken.add(delivered_name)
        entries.append({
            'filename': delivered_name,
            'original_name': str(orig_name),
            'original_file': os.path.join(SUBDIR, 'originals', original_name),
            'delivered_file': os.path.join(SUBDIR, 'raw', delivered_name),
            'orig_width': ow, 'orig_height': oh,
            'width': dw, 'height': dh,
            'scale': scale,
            'imported_at': _now_ast().strftime('%Y-%m-%d %H:%M:%S AST'),
            'reviewer': reviewer or '',
        })
        log(f"[import] {orig_name} -> {delivered_name} "
            f"({ow}x{oh} -> {dw}x{dh}, scale {scale:.4f})")

    if entries:
        _save_manifest(export_dir, existing + entries)
    return entries, skipped


def seg_dict_for(export_dir, entry):
    """Build the annotator seg_dict for one manifest entry.

    Shape-identical to the MANUAL_ANNOTATE seg_dicts configure() builds for
    routed frames (masks empty, no reference points), plus the custom-import
    provenance fields the UI badge and manifest reporting read."""
    return {
        'image_path': entry['delivered_file'],
        'image_path_abs': os.path.join(export_dir, entry['delivered_file']),
        'image_width': entry.get('width', 0),
        'image_height': entry.get('height', 0),
        'masks': [],
        'reference_points': [],
        'processed_at': datetime.now().isoformat(),
        'reviewed': False,
        'exported': False,
        'custom_import': True,
        'import_scale': entry.get('scale', 1.0),
        'import_original': entry.get('original_file', ''),
    }


def find_project_json(export_dir):
    """Walk up from the export dir to the run root's project.json ('' if none).

    The export dir is normally <run_*>/step4test_combinedAnnotate, so one hop
    up finds it; a couple more levels are tried so a nested layout still
    resolves. Standalone (non-run) sessions have no project.json: fine."""
    d = os.path.abspath(export_dir)
    for _ in range(4):
        cand = os.path.join(d, 'project.json')
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return ''


def record_in_project(export_dir, entries, initials='LO'):
    """Record an import batch in the run's project.json (access_log + a
    cumulative top-level custom_imports list). Atomic read-modify-write with
    os.replace, matching project_manager.save_project. Returns True when a
    project.json was found and updated; False (never raises) otherwise, so a
    standalone session or a transient write error can never block an import."""
    if not entries:
        return False
    pj = find_project_json(export_dir)
    if not pj:
        return False
    try:
        with open(pj) as f:
            state = json.load(f)
        if not isinstance(state, dict):
            return False
        names = [e['filename'] for e in entries]
        state.setdefault('access_log', []).append({
            'ts': _now_ast().isoformat(timespec='seconds'),
            'initials': initials or 'LO',
            'action': 'import_images',
            'purpose': (f"imported {len(entries)} custom image(s) into the "
                        f"combined annotator: {', '.join(names)}"),
        })
        rec = state.setdefault('custom_imports', [])
        for e in entries:
            rec.append({
                'filename': e['filename'],
                'original_name': e.get('original_name', ''),
                'delivered_file': e.get('delivered_file', ''),
                'scale': e.get('scale', 1.0),
                'imported_at': e.get('imported_at', ''),
                'by': initials or 'LO',
            })
        tmp = pj + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, pj)
        return True
    except Exception:
        return False
