"""
_reefreview — shared helpers for the reef_point_seg expert-review round-trip.

This package is imported by BOTH the step-5 segmentation app
(TCRMPclip_segmentImages) and the standalone "Add Expert IDs" importer
(TCRMPclip_addExpertIDs). Each app does:

    sys.path.insert(0, <scripts_dir>)   # the parent of this package
    from _reefreview import library, review_repo, render_review, ...

It deliberately has NO heavy dependencies (no torch / SAM3). Geometry is
numpy/OpenCV + PIL only, so the importer stays lightweight.

Modules:
  filename_parse  TCRMP clip filename -> {year, date, site, transect, frame}
  uid             deterministic, dedupe-stable mask UID
  mask_geom       COCO-RLE decode + overlap math (mirrors mask_ops.rle_decode)
  codes           master_codes.csv loader + group hierarchy + confidence defs
  library         permanent, cross-project expert-ID library (manifest+polygons)
  render_review   per-mask closeup (outline + bbox) + alpha mask + full frame
  review_repo     the /mnt/tear/REVIEW_reefpointseg GitHub-Pages repo (push)
"""

__all__ = [
    "filename_parse", "uid", "mask_geom", "codes",
    "library", "render_review", "review_repo",
]
