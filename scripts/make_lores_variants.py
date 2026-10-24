#!/usr/bin/env python
"""Generate ~1920px 'lores' copies of oversized (>1920 long edge) TCRMP clip
frames into a parallel mirror tree, with a manifest. One-time maintenance tool;
re-run when new 4K data lands. Idempotent.

  env/bin/python scripts/make_lores_variants.py            # default clip tree
  env/bin/python scripts/make_lores_variants.py --dry-run
  env/bin/python scripts/make_lores_variants.py --force
"""
import argparse
import csv
import os
import tempfile
from PIL import Image

import _superseded as _sup

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(_REPO, "supporting_data", "TCRMP_clip")
DEFAULT_LORES = os.path.join(_REPO, "supporting_data", "TCRMP_clip_lores")
LONG_EDGE = 1920
JPEG_QUALITY = 92
TEMP_PREFIX = ".lores_"   # hidden, so the clip index and the walks never list it
MANIFEST_FIELDS = ["basename", "source_path", "lores_path", "orig_width",
                   "orig_height", "lores_width", "lores_height", "scale"]


def _canon_rank(path):
    """Lower = more canonical. Prefer shallower paths and penalize known
    non-canonical variants (re-export subdirs / edited copies), mirroring the
    intent of chooseImages' build_clip_index tie-breaker."""
    p = path.lower()
    penalty = 0
    if os.sep + "jpeg" + os.sep in p:
        penalty += 100
    if "_edit" in os.path.basename(p):
        penalty += 100
    return (penalty, path.count(os.sep), len(path))


def _all_files(clip_root):
    """Every non-hidden file path under clip_root, via the shared persisted
    index (scripts/clip_index.py) when available, else a direct walk."""
    try:
        import clip_index
        return clip_index.list_files(clip_root)
    except Exception:
        paths = []
        for root, dirs, files in os.walk(clip_root):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d != _sup.SUPERSEDED_DIR_NAME]
            for fn in files:
                if not fn.startswith("."):
                    paths.append(os.path.join(root, fn))
        return paths


def _iter_images(clip_root):
    for path in _all_files(clip_root):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() not in (".jpg", ".jpeg"):
            continue
        # Skip derived preview overlays (the "_pts" annotated variants):
        # they are not source frames, so a lores copy is wasted disk and a
        # dead manifest row (chooseImages only ever looks up a frame's raw
        # basename, never "<basename>_pts").
        if stem.lower().endswith("_pts"):
            continue
        yield path


def _write_lores(src, lores_path, size, lores_root, stamp, log=print):
    """Write the low-res copy of src to lores_path without losing an old copy.

    The new JPEG is written to a hidden temp file in the same folder first, so
    a failed write never truncates the copy already there. An existing copy is
    then moved to <lores_root>/_superseded/<stamp>/<relative path> (images are
    never deleted) and the temp file is renamed into place.

    Parameters:
        src: the full-size source frame.
        lores_path: where the low-res copy belongs, inside lores_root.
        size: (width, height) of the copy.
        lores_root: the mirror tree the holding folder hangs from.
        stamp: YYYYMMDD-HHMMSS shared by every move in one run.
        log: a callable taking one line.
    Raises:
        OSError: the temp file could not be written or moved; any old copy
            is still at lores_path or in the holding folder.
    """
    folder = os.path.dirname(lores_path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=".jpg", dir=folder)
    os.close(fd)
    try:
        with Image.open(src) as im:
            im = im.convert("RGB").resize(size, Image.LANCZOS)
            im.save(tmp_path, "JPEG", quality=JPEG_QUALITY)
        if os.path.lexists(lores_path):
            _sup.supersede(lores_path, base_dir=lores_root, stamp=stamp, log=log)
        os.replace(tmp_path, lores_path)
    except BaseException:
        # tmp_path is the half-written file this call made, never a copy anyone relies on
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def generate(clip_root, lores_root, long_edge=LONG_EDGE, force=False,
             dry_run=False, log=print):
    stats = {"scanned": 0, "generated": 0, "skipped_small": 0, "up_to_date": 0,
              "manifest_rows": 0}
    manifest_rows = []
    stamp = _sup.stamp_now()
    for src in _iter_images(clip_root):
        stats["scanned"] += 1
        try:
            with Image.open(src) as im:
                ow, oh = im.size
        except Exception as e:
            log(f"WARNING: cannot read {src}: {e}")
            continue
        if max(ow, oh) <= long_edge:
            stats["skipped_small"] += 1
            continue
        rel = os.path.relpath(src, clip_root)
        stem, _ = os.path.splitext(rel)
        lores_rel = stem + "_lores.jpg"
        lores_path = os.path.join(lores_root, lores_rel)
        scale = long_edge / max(ow, oh)
        lw, lh = round(ow * scale), round(oh * scale)
        basename = os.path.basename(stem)
        manifest_rows.append({
            "basename": basename, "source_path": src, "lores_path": lores_path,
            "orig_width": ow, "orig_height": oh, "lores_width": lw,
            "lores_height": lh, "scale": scale,
        })
        if (not force and os.path.isfile(lores_path)
                and os.path.getmtime(lores_path) >= os.path.getmtime(src)):
            stats["up_to_date"] += 1
            continue
        if dry_run:
            log(f"[dry-run] would write {lores_path} ({lw}x{lh})")
            stats["generated"] += 1
            continue
        _write_lores(src, lores_path, (lw, lh), lores_root, stamp, log)
        stats["generated"] += 1
        log(f"wrote {lores_path} ({lw}x{lh})")

    # Dedup manifest rows by basename: the tree can contain duplicate frames
    # (e.g. a JPEG/ re-export or an _edit variant alongside the canonical
    # original), each producing its own row for the same basename. Keep only
    # the most-canonical source per basename so load_lores_manifest's
    # last-write-wins can't land on a non-canonical crop and misplace points.
    best = {}
    for r in manifest_rows:
        b = r["basename"]
        rank = _canon_rank(r["source_path"])
        if b not in best or rank < _canon_rank(best[b]["source_path"]):
            best[b] = r
    stats["manifest_rows"] = len(best)

    if not dry_run:
        os.makedirs(lores_root, exist_ok=True)
        with open(os.path.join(lores_root, "lores_manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for r in sorted(best.values(), key=lambda x: x["basename"]):
                w.writerow(r)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_CLIP)
    ap.add_argument("--lores-root", default=DEFAULT_LORES)
    ap.add_argument("--long-edge", type=int, default=LONG_EDGE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    s = generate(args.root, args.lores_root, args.long_edge, args.force, args.dry_run)
    print(f"scanned={s['scanned']} generated={s['generated']} "
          f"skipped_small={s['skipped_small']} up_to_date={s['up_to_date']} "
          f"manifest_rows={s['manifest_rows']}")


if __name__ == "__main__":
    main()
