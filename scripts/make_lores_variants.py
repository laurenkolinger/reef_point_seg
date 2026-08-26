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
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(_REPO, "supporting_data", "TCRMP_clip")
DEFAULT_LORES = os.path.join(_REPO, "supporting_data", "TCRMP_clip_lores")
LONG_EDGE = 1920
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


def _iter_images(clip_root):
    for root, dirs, files in os.walk(clip_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in (".jpg", ".jpeg"):
                continue
            # Skip derived preview overlays (the "_pts" annotated variants):
            # they are not source frames, so a lores copy is wasted disk and a
            # dead manifest row (chooseImages only ever looks up a frame's raw
            # basename, never "<basename>_pts").
            if stem.lower().endswith("_pts"):
                continue
            yield os.path.join(root, fn)


def generate(clip_root, lores_root, long_edge=LONG_EDGE, force=False,
             dry_run=False, log=print):
    stats = {"scanned": 0, "generated": 0, "skipped_small": 0, "up_to_date": 0,
              "manifest_rows": 0}
    manifest_rows = []
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
        os.makedirs(os.path.dirname(lores_path), exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB").resize((lw, lh), Image.LANCZOS)
            im.save(lores_path, "JPEG", quality=92)
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
