"""
Minimal mask geometry — COCO-style RLE decode + overlap math.

A standalone copy of the column-major (order='F') RLE codec used in
TCRMPclip_segmentImages/src/mask_ops.py:68-98, so the importer can reason about
mask overlap WITHOUT importing the heavy segmentation app (no torch / SAM3).

Do NOT hand these RLEs to pycocotools — they are the project's custom
column-major encoding.
"""

import numpy as np


def rle_decode(rle, shape=None):
    """Decode a custom column-major COCO RLE dict to a boolean mask.

    rle: {'counts': [...], 'size': [H, W]}. Odd-indexed runs are foreground.
    shape: optional (H, W) override.
    """
    if shape is None:
        h, w = rle['size']
    else:
        h, w = shape
    counts = rle['counts']
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:
            flat[pos:pos + c] = True
        pos += c
    return flat.reshape((h, w), order='F')


def rle_encode(binary_mask):
    """Encode a boolean mask to the same column-major RLE form (for the library)."""
    h, w = binary_mask.shape
    flat = binary_mask.flatten(order='F').astype(np.uint8)
    diff = np.diff(np.concatenate([[0], flat, [0]]))
    transitions = np.where(diff != 0)[0]
    boundaries = np.concatenate([[0], transitions, [len(flat)]])
    counts = np.diff(boundaries).tolist()
    return {'counts': counts, 'size': [int(h), int(w)]}


def area(binary_mask):
    return int(np.count_nonzero(binary_mask))


def intersection(a, b):
    return int(np.count_nonzero(a & b))


def iou(a, b):
    inter = intersection(a, b)
    union = int(np.count_nonzero(a | b))
    return inter / union if union else 0.0


def intersection_over_area(new_mask, other_mask):
    """Fraction of `new_mask` covered by `other_mask` (the >50% relabel test)."""
    a = area(new_mask)
    if a == 0:
        return 0.0
    return intersection(new_mask, other_mask) / a


def decode_pair(rle_a, rle_b):
    """Decode two RLEs onto a common (H, W) grid. Returns (a, b) or (None, None)
    if their declared sizes differ (different source images -> not comparable)."""
    sa = tuple(rle_a.get('size', []))
    sb = tuple(rle_b.get('size', []))
    if not sa or sa != sb:
        return None, None
    a = rle_decode(rle_a, sa)
    b = rle_decode(rle_b, sb)
    return a, b
