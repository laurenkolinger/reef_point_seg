"""
Mask operations — overlap resolution, merging, RLE encoding, polygon conversion.

All operations are CPU-only (numpy/OpenCV). No GPU needed.
"""

import numpy as np
import cv2


# ── Mask cleanup (single connected component, no shards) ────────────

def _fill_holes(mask, min_hole=500):
    """Fill small holes in a binary mask."""
    inv = 1 - mask
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] < min_hole:
            mask[labels2 == i] = 1
    return mask

def clean_mask(binary_mask, min_fragment=500, fill_holes=True, bridge_gaps=True):
    """Clean a binary mask into a single unified shape:
    - Morphological closing to bridge nearby disconnected regions
    - Keep only the largest connected component
    - Fill small interior holes

    Args:
        binary_mask: (H, W) bool or uint8 array
        min_fragment: remove components smaller than this (pixels)
        fill_holes: if True, fill holes smaller than min_fragment
        bridge_gaps: if True, use morphological closing to connect nearby parts

    Returns:
        cleaned uint8 mask (0/1)
    """
    mask = binary_mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask

    # Bridge small gaps between disconnected parts via morphological closing
    # (dilate then erode — connects components separated by small gaps)
    if bridge_gaps:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find connected components, keep only the largest
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        if fill_holes:
            mask = _fill_holes(mask, min_fragment)
        return mask

    # Component 0 is background; find largest foreground component
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    largest_idx = np.argmax(areas) + 1    # +1 because we skipped bg
    mask = (labels == largest_idx).astype(np.uint8)

    # Fill small holes
    if fill_holes:
        mask = _fill_holes(mask, min_fragment)

    return mask


# ── RLE encoding/decoding (COCO-compatible) ─────────────────────────

def rle_encode(binary_mask):
    """Encode a boolean mask as COCO-style RLE.
    Counts alternate: [bg_run, fg_run, bg_run, fg_run, ...].
    First run is always background (may be 0 if mask starts with foreground).
    Returns dict with 'counts' (list of ints) and 'size' [H, W]."""
    h, w = binary_mask.shape
    flat = binary_mask.flatten(order='F').astype(np.uint8)
    # Find transition points
    diff = np.diff(np.concatenate([[0], flat, [0]]))
    transitions = np.where(diff != 0)[0]
    # Build counts including initial run from position 0
    boundaries = np.concatenate([[0], transitions, [len(flat)]])
    counts = np.diff(boundaries).tolist()
    return {'counts': counts, 'size': [h, w]}


def rle_decode(rle, shape=None):
    """Decode COCO-style RLE to boolean mask.
    shape: (H, W) — if None, uses rle['size']."""
    if shape is None:
        h, w = rle['size']
    else:
        h, w = shape
    counts = rle['counts']
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:  # odd runs are foreground
            flat[pos:pos + c] = True
        pos += c
    return flat.reshape((h, w), order='F')


# ── Polygon conversion ──────────────────────────────────────────────

def mask_to_polygon(binary_mask, simplify_epsilon=0.001):
    """Convert binary mask to polygon contours.

    Args:
        binary_mask: (H, W) boolean numpy array
        simplify_epsilon: approxPolyDP epsilon as fraction of perimeter

    Returns:
        polygon_px: list of polygons, each [x1,y1, x2,y2, ...] in pixel coords
        polygon_norm: same but normalized to 0-1 by image dims
    """
    h, w = binary_mask.shape
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygon_px = []
    polygon_norm = []

    for contour in contours:
        if len(contour) < 3:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue

        # Flatten to [x1, y1, x2, y2, ...]
        pts = approx.squeeze()
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        flat_px = pts.flatten().tolist()
        flat_norm = []
        for i in range(0, len(flat_px), 2):
            flat_norm.append(round(flat_px[i] / w, 6))
            flat_norm.append(round(flat_px[i + 1] / h, 6))

        polygon_px.append(flat_px)
        polygon_norm.append(flat_norm)

    return polygon_px, polygon_norm


def polygon_to_mask(polygon_px, h, w):
    """Convert a pixel-coordinate polygon back to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(polygon_px, dtype=np.float32).reshape(-1, 2)
    pts = pts.astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


# ── Bounding box from mask ──────────────────────────────────────────

def mask_bbox(binary_mask):
    """Return [x_min, y_min, x_max, y_max] of a binary mask, or None if empty."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def mask_centroid(binary_mask):
    """Return (cx, cy) centroid of a binary mask, or None if empty."""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))


# ── Overlap resolution ──────────────────────────────────────────────

def resolve_overlaps(masks, strategy="larger_wins", min_area=200, thin_ratio=0.05):
    """Resolve overlapping masks by priority-ordered subtraction.

    Args:
        masks: list of dicts, each must have 'mask' (bool np array), 'score', 'area'
        strategy: "larger_wins" | "higher_score" | "first_wins"
        min_area: discard masks with fewer pixels than this
        thin_ratio: discard if remaining_area / bbox_area < this

    Returns:
        list of mask dicts (modified in place), with some removed
    """
    if not masks:
        return masks

    # Sort by priority
    if strategy == "larger_wins":
        masks.sort(key=lambda m: m.get('area', 0), reverse=True)
    elif strategy == "higher_score":
        masks.sort(key=lambda m: m.get('score', 0), reverse=True)
    # "first_wins" keeps original order

    h, w = masks[0]['mask'].shape
    occupied = np.zeros((h, w), dtype=bool)
    kept = []

    for m in masks:
        binary = m['mask']
        overlap = binary & occupied
        if overlap.any():
            binary = binary & ~occupied
            # Clean to single connected component after subtraction
            binary = clean_mask(binary, min_fragment=min_area).astype(bool)
            area = int(binary.sum())
            if area < min_area:
                continue
            # Sliver check
            bbox = mask_bbox(binary)
            if bbox is not None:
                bx0, by0, bx1, by1 = bbox
                bbox_area = max((bx1 - bx0) * (by1 - by0), 1)
                if area / bbox_area < thin_ratio:
                    continue
            m['mask'] = binary
            m['area'] = area
            m['bbox'] = bbox

        occupied |= m['mask']
        kept.append(m)

    return kept


# ── Merge close masks ───────────────────────────────────────────────

def merge_close_masks(masks, distance_px=30, same_species_only=True):
    """Merge masks whose centroids are within distance_px.

    Args:
        masks: list of mask dicts with 'mask', 'species', 'label', etc.
        distance_px: maximum centroid distance for merging
        same_species_only: if True, only merge masks with same species code

    Returns:
        list of mask dicts (merged masks replace originals)
    """
    if len(masks) < 2:
        return masks

    # Compute centroids
    centroids = []
    for m in masks:
        c = mask_centroid(m['mask'])
        centroids.append(c)

    # Find merge pairs (greedy)
    merged_into = {}  # index -> merged_into_index
    for i in range(len(masks)):
        if i in merged_into:
            continue
        ci = centroids[i]
        if ci is None:
            continue
        for j in range(i + 1, len(masks)):
            if j in merged_into:
                continue
            cj = centroids[j]
            if cj is None:
                continue
            if same_species_only and masks[i].get('species') != masks[j].get('species'):
                continue
            dist = np.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
            if dist <= distance_px:
                merged_into[j] = i

    # Apply merges
    result = []
    for i in range(len(masks)):
        if i in merged_into:
            continue
        m = masks[i]
        # Collect all masks merged into this one
        for j, target in merged_into.items():
            if target == i:
                m['mask'] = m['mask'] | masks[j]['mask']
                # Keep the label of the primary mask
        m['area'] = int(m['mask'].sum())
        m['bbox'] = mask_bbox(m['mask'])
        result.append(m)

    return result


# ── Manual merge (two specific masks) ───────────────────────────────

def check_no_overlap(mask_a, mask_b):
    """Return True if masks do not overlap."""
    return not (mask_a & mask_b).any()


def union_masks(mask_a, mask_b):
    """Union two binary masks."""
    return mask_a | mask_b


# ── Build full mask dict from SAM3 result ───────────────────────────

def build_mask_dict(mask_id, binary_mask, score, point_info, source_type="auto",
                    simplify_epsilon=0.001):
    """Build a complete mask dict from a SAM3 result.

    Args:
        mask_id: integer ID
        binary_mask: (H, W) boolean numpy array
        score: float confidence score
        point_info: dict with 'label', 'species', 'category', 'x', 'y', 'name'
        source_type: 'auto' | 'exemplar' | 'manual_box'
        simplify_epsilon: polygon simplification factor

    Returns:
        mask dict ready for storage, or None if mask is too small
    """
    # Clean: keep largest connected component, fill holes
    binary_mask = clean_mask(binary_mask, min_fragment=500).astype(bool)

    area = int(binary_mask.sum())
    bbox = mask_bbox(binary_mask)
    if bbox is None or area == 0:
        return None

    rle = rle_encode(binary_mask)
    polygon_px, polygon_norm = mask_to_polygon(binary_mask, simplify_epsilon)

    if not polygon_px:
        return None

    return {
        'id': mask_id,
        'label': point_info.get('label', '?'),
        'species': point_info.get('species', 'UNK'),
        'name': point_info.get('name', ''),
        'category': point_info.get('category', ''),
        'source_x': point_info.get('x', 0),
        'source_y': point_info.get('y', 0),
        'polygon_px': polygon_px,
        'polygon_norm': polygon_norm,
        'rle': rle,
        'bbox': bbox,
        'area': area,
        'score': round(score, 4),
        'status': 'pending',
        'refinement_clicks': [],
        'source_type': source_type,
    }


def update_mask_geometry(mask_dict, binary_mask, simplify_epsilon=0.001):
    """Recompute polygon/RLE/bbox/area after mask modification."""
    # Clean: keep largest connected component, fill holes
    binary_mask = clean_mask(binary_mask, min_fragment=500).astype(bool)

    area = int(binary_mask.sum())
    bbox = mask_bbox(binary_mask)
    if bbox is None or area == 0:
        return None

    rle = rle_encode(binary_mask)
    polygon_px, polygon_norm = mask_to_polygon(binary_mask, simplify_epsilon)
    if not polygon_px:
        return None

    mask_dict['rle'] = rle
    mask_dict['polygon_px'] = polygon_px
    mask_dict['polygon_norm'] = polygon_norm
    mask_dict['bbox'] = bbox
    mask_dict['area'] = area
    return mask_dict
