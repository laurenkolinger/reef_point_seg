"""
Mask operations — overlap resolution, merging, RLE encoding, polygon conversion.

All operations are CPU-only (numpy/OpenCV). No GPU needed.
"""

import threading
import unicodedata

import numpy as np
import cv2


# ── Expert-review flagging policy ──────────────────────────────
#
# Lauren Olinger's decision of 2026-09-06: VICARIUS accepts no new expert-review
# tags on any mask, in any annotator. Setting REVIEW_TAGGING_ENABLED back to
# True is the whole reversal; no other line in this file, and no line in the
# apps that import it, has to change with it.
#
# Masks flagged before that decision keep their tag. Nothing in this block
# rewrites, strips or hides a tag already stored: the lock refuses new ones.
#
# The refusal here is deliberately BROADER than the platform's read predicate,
# and the gap widened on 2026-09-06 when the category joined the spellings it
# catches. The refusal recognises FOUR: the 'review' key, 'species',
# 'species_code' and 'category', each tested on its own and each normalised for
# case, surrounding whitespace, invisible characters and letters borrowed from
# another script. The read predicate (_is_review in each app.py) recognises
# TWO: a truthy mask['review'] or a species of exactly 'REVIEW'. That asymmetry is deliberate and permanent. The refusal decides what
# may be WRITTEN, so it is as wide as the ways a tag can be spelled; the read
# predicate decides what the masks already stored on disk MEAN, and it drives
# the training export, the coverage matrix, the expert-review export and the
# provenance ledger, so widening it would change the meaning of those masks.

REVIEW_TAGGING_ENABLED = False

REVIEW_LOCKED_MSG = (
    "Flagging for expert review is temporarily unavailable. "
    "VICARIUS is accepting no new review tags."
)

REVIEW_TAG = "REVIEW"

# Payload keys that carry the tag as a species string. 'species_code' is the
# spelling the Place Points tool writes into ids/sam_click_prompts.json, and it
# is tested on its own so a payload holding an empty 'species' beside a
# 'species_code' of REVIEW is still caught.
REVIEW_SPECIES_KEYS = ('species', 'species_code')

# Payload key that carries the tag as a category. 'Review' is not one of the
# real categories (DEFAULT_CATEGORIES in each app's config.py holds only
# 'Target species only'), and across the three segmentation files under
# inprocess/ it pairs one for one with the flag: 130 masks carry both, none
# carries either on its own. It is the tag's third spelling, so it is tested as
# a key in its own right rather than as a decoration on a species.
REVIEW_CATEGORY_KEY = 'category'

# Every payload key whose STRING value can spell the tag out. One greppable
# policy point: review_tag_requested and strip_review_tag both walk this tuple,
# so a fourth string spelling is one line here and nothing else.
REVIEW_TAG_STRING_KEYS = REVIEW_SPECIES_KEYS + (REVIEW_CATEGORY_KEY,)

# Payload key that carries the tag as a boolean.
REVIEW_FLAG_KEY = 'review'

# The display name the annotators write beside the tag. No predicate reads a
# mask's name and no registry column stores it, so it is display only, but a
# mask whose tag the lock has dropped must not go on announcing itself as
# flagged. The exemplar tray writes the tag itself into the name instead, so a
# name carries the tag in one of two spellings.
REVIEW_DISPLAY_NAME = 'Flagged for expert review'

# Both of those spellings as normalise_review_value returns them. Both are plain
# ASCII, so upper() is that normalisation.
REVIEW_NAME_TAGS = (REVIEW_TAG, REVIEW_DISPLAY_NAME.upper())

# Unicode general categories removed from a value before it is compared with the
# tag. 'Cf' is the format class, which holds the zero-width space, the
# zero-width joiner and non-joiner, the word joiner, the byte-order mark and the
# soft hyphen; 'Cc' is the control class. Python's str.strip() removes none of
# them, so 'REVIEW' followed by a zero-width space used to read as a species of
# its own while rendering exactly like the tag. A real species code or category
# name carries neither class of character, so removing them can only change a
# value that was already carrying something invisible.
INVISIBLE_CHARACTER_CATEGORIES = ('Cc', 'Cf')

# How many review tags the lock has dropped since this process started, with the
# lock that guards the count: an annotator serves requests on several threads,
# so the counter is incremented under it rather than with a bare +=.
_review_tags_dropped = 0
_review_tags_dropped_lock = threading.Lock()


def _spells_the_tag_in_another_script(text):
    """Report whether a cleaned value is the tag with letters swapped out.

    NFKC folds the width and compatibility variants of a Latin letter onto the
    letter itself, so a fullwidth W is already handled by the time this runs. It
    does not fold a letter from another script: Cyrillic Er, the Cyrillic R, is
    a separate character that reads as the tag on screen, in the mask registry
    and in the coverage matrix. Rather than keep a table of look-alikes, which
    would have to grow with every script, this compares position by position: a
    value is the tag in disguise when it is exactly as long as the tag, every
    ASCII character sits where the tag has that same character, and every
    remaining character is a letter.

    A value made only of ASCII passes this test only when it already IS the tag,
    so no ordinary code can be caught by it, and a longer or shorter value
    (REVIEWED, PREREVIEW, REVIEW_PA, RE VIEW) never matches at all.

    Args:
        text: one value already upper-cased and stripped of its invisible
            characters.

    Returns:
        bool: True when the value spells the tag with one or more of its letters
        replaced by a letter from another script.

    Example:
        _spells_the_tag_in_another_script('\u0420EVIEW') returns True, and
        _spells_the_tag_in_another_script('REVIEWED') returns False.
    """
    if len(text) != len(REVIEW_TAG):
        return False
    for char, tag_char in zip(text, REVIEW_TAG):
        if char.isascii():
            if char != tag_char:
                return False
        elif not char.isalpha():
            return False
    return True


def normalise_review_value(value):
    """Normalise one payload value for comparison against the review tag.

    The value is NFKC-normalised (which folds a fullwidth or otherwise
    compatibility-spelled Latin letter onto the plain letter), stripped of its
    invisible characters (see INVISIBLE_CHARACTER_CATEGORIES), then trimmed and
    upper-cased. A value that comes out of that as the tag written in another
    script is returned as the tag itself, so one comparison settles every
    spelling. Anything else is returned as its own cleaned text, so a species
    that merely contains the word is unaffected.

    Args:
        value: any value read out of a request body, a prompt file or a mask
            dict. None, numbers and other non-strings are all accepted.

    Returns:
        str: the cleaned upper-case text, REVIEW when that text is the tag or a
        near-spelling of it, or '' when the value is None.

    Example:
        normalise_review_value('  review ') returns 'REVIEW', and so does
        normalise_review_value('REVIEW\u200b'); normalise_review_value(
        'REVIEWED') returns 'REVIEWED'.
    """
    if value is None:
        return ''
    folded = unicodedata.normalize('NFKC', str(value))
    visible = ''.join(
        char for char in folded
        if unicodedata.category(char) not in INVISIBLE_CHARACTER_CATEGORIES
    )
    text = visible.strip().upper()
    if _spells_the_tag_in_another_script(text):
        return REVIEW_TAG
    return text


def review_tag_requested(payload):
    """Report whether a payload asks for an expert-review tag.

    Catches every spelling an annotator can produce: a truthy 'review' key (the
    routes coerce that key with bool(), so the strings '0', 'no' and 'false' are
    all truthy and all count as a request), and a 'species', 'species_code' or
    'category' that normalises to REVIEW in any case or with surrounding
    whitespace. The three string keys are tested independently, never as a
    fallback chain, so a payload carrying an empty 'species' beside a
    'species_code' of REVIEW is still caught, and so is one whose only review
    spelling is the category.

    Args:
        payload: a request body, a point info dict, a mask dict, or anything
            else. A value with no dict-like get() answers False rather than
            raising, so a caller can pass an unparsed request body straight in.

    Returns:
        bool: True when the payload asks for a review tag.

    Example:
        review_tag_requested({'species': 'PA', 'category': 'Review'}) returns
        True, and so does review_tag_requested({'species': '',
        'species_code': ' review '}).
    """
    getter = getattr(payload, 'get', None)
    if not callable(getter):
        return False
    if bool(getter(REVIEW_FLAG_KEY)):
        return True
    for key in REVIEW_TAG_STRING_KEYS:
        if normalise_review_value(getter(key)) == REVIEW_TAG:
            return True
    return False


def strip_review_tag(payload):
    """Return a copy of a payload with any expert-review tag removed.

    Use it where a caller wants to carry on without the tag rather than refuse
    the whole request: the 'review' key is set to False, and a 'species',
    'species_code' or 'category' that normalises to REVIEW is blanked. Each key
    is judged on its own value, so a payload carrying a real species beside a
    Review category keeps the species and loses only the category. Every other
    key is copied across unchanged, and the argument itself is never modified.

    Args:
        payload: a dict-like request body, point info or mask dict.

    Returns:
        dict: a new dict carrying no review tag. review_tag_requested() of the
        result is always False.

    Raises:
        TypeError: when payload cannot be copied into a dict, because a caller
            that cannot name the keys it is stripping has passed the wrong
            thing. Guard with review_tag_requested() first, which tolerates any
            value at all.

    Example:
        strip_review_tag({'species': 'PA', 'category': 'Review'}) returns
        {'species': 'PA', 'category': '', 'review': False}.
    """
    try:
        stripped = dict(payload)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "strip_review_tag needs a dict-like payload (a request body, a "
            "point info or a mask dict), got "
            f"{type(payload).__name__}"
        ) from exc
    stripped[REVIEW_FLAG_KEY] = False
    for key in REVIEW_TAG_STRING_KEYS:
        if normalise_review_value(stripped.get(key)) == REVIEW_TAG:
            stripped[key] = ''
    return stripped


def review_tags_dropped():
    """Report how many review tags the lock has dropped in this process.

    build_mask_dict cannot refuse a review tag by raising: the batch
    segmentation pass calls it in a loop over every prompt in a frame, and an
    exception there would abandon the whole frame. It counts the drop instead,
    and a caller reads the count to report what it dropped, the way
    merge_overlapping_same_id returns its merged_count.

    Returns:
        int: the running total since the process started, or since the last
        reset_review_tags_dropped() call.

    Example:
        before = review_tags_dropped()
        ... build one frame's masks ...
        dropped = review_tags_dropped() - before
    """
    with _review_tags_dropped_lock:
        return _review_tags_dropped


def reset_review_tags_dropped():
    """Set the dropped-review-tag counter back to zero.

    Returns:
        int: the count that was cleared, so one call both reports a batch total
        and starts the next batch from zero.
    """
    global _review_tags_dropped
    with _review_tags_dropped_lock:
        cleared = _review_tags_dropped
        _review_tags_dropped = 0
    return cleared


def _record_dropped_review_tag():
    """Add one to the dropped-review-tag counter, safely across threads."""
    global _review_tags_dropped
    with _review_tags_dropped_lock:
        _review_tags_dropped += 1


def _blank_if_review(value):
    """Return '' when a value spells out the review tag, else the value itself.

    Args:
        value: a species or category value out of a point info dict.

    Returns:
        The value unchanged, or '' when it normalises to REVIEW.
    """
    if normalise_review_value(value) == REVIEW_TAG:
        return ''
    return value


def _blank_if_review_name(value):
    """Return '' when a display name announces the review tag, else the name.

    A species and a category carry the tag as the single word REVIEW, which
    _blank_if_review settles. A name carries it as the sentence the annotators
    write beside the tag ('Flagged for expert review') or as the tag itself, so
    it needs both spellings. Every other name is returned unchanged, so a mask
    that keeps a real species keeps that species' real name with it.

    Args:
        value: the name out of a point info dict.

    Returns:
        The name unchanged, or '' when it announces the review tag.

    Example:
        _blank_if_review_name('Flagged for expert review') returns '', and
        _blank_if_review_name('Porites astreoides') returns it unchanged.
    """
    if normalise_review_value(value) in REVIEW_NAME_TAGS:
        return ''
    return value


def _review_promotion_blocked(survivor_value, absorbed_value):
    """Report whether a merge must leave one survivor field alone.

    A merge promotes the absorbed mask's species, name and category onto the
    survivor when the absorbed mask carries the accepted expert ID. Under the
    lock that promotion may not hand the survivor a REVIEW value it did not
    already carry, because that would tag a mask that was never flagged.

    Args:
        survivor_value: the survivor's current value for the field.
        absorbed_value: the absorbed mask's value for the same field.

    Returns:
        bool: True when the field must keep the survivor's own value.

    Example:
        _review_promotion_blocked('PA', 'REVIEW') returns True while the lock
        is on; _review_promotion_blocked('REVIEW', 'REVIEW') returns False,
        because the survivor was flagged already.
    """
    if REVIEW_TAGGING_ENABLED:
        return False
    if normalise_review_value(absorbed_value) != REVIEW_TAG:
        return False
    return normalise_review_value(survivor_value) != REVIEW_TAG


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


# Placeholder species codes that are NOT a real organism ID. Two masks both
# carrying one of these are DIFFERENT organisms awaiting different expert IDs —
# unioning them would silently destroy one mask's distinct review state, so they
# are excluded from same-id grouping entirely.
PLACEHOLDER_SPECIES = frozenset({'REVIEW', '', '?'})


def _merge_review_state(survivor, absorbed):
    """Carry an absorbed mask's expert-review state onto the survivor when two
    real same-species masks are unioned.

    - review_uid: keep the survivor's if set, else adopt the absorbed one's.
    - reviews[]: concatenate, de-duplicated by (reviewer, code, confidence, at).
    - expert_id: prefer an existing expert_id over none (an accepted ID must not
      be silently downgraded to pending).

    Under the expert-review lock (REVIEW_TAGGING_ENABLED False, Lauren Olinger,
    2026-09-06) a merge preserves review state but never adds any: a survivor
    that was not flagged gains neither the review boolean nor a REVIEW species,
    while review_uid, reviews[] and expert_id still travel as before.

    Returns True on success, or False to signal the caller MUST REFUSE to merge
    this pair (two DIFFERENT accepted expert_ids conflict — merging would force
    a single organism to claim two different expert codes).
    """
    s_eid = survivor.get('expert_id')
    a_eid = absorbed.get('expert_id')

    def _accepted_code(eid):
        if isinstance(eid, dict) and eid.get('mode') == 'EXPERT':
            return eid.get('code')
        return None

    s_code = _accepted_code(s_eid)
    a_code = _accepted_code(a_eid)
    if s_code is not None and a_code is not None and s_code != a_code:
        # Two conflicting expert-accepted IDs — refuse.
        return False

    # review_uid: survivor wins if present, else inherit the absorbed one's.
    if not survivor.get('review_uid') and absorbed.get('review_uid'):
        survivor['review_uid'] = absorbed['review_uid']

    # reviews[]: concatenate + de-dup (keep order, survivor first).
    s_reviews = survivor.get('reviews') or []
    a_reviews = absorbed.get('reviews') or []
    if a_reviews:
        seen = set()
        combined_reviews = []
        for r in list(s_reviews) + list(a_reviews):
            key = (
                (r or {}).get('reviewer'), (r or {}).get('code'),
                (r or {}).get('confidence'), (r or {}).get('at'),
            )
            if key in seen:
                continue
            seen.add(key)
            combined_reviews.append(r)
        survivor['reviews'] = combined_reviews

    # expert_id: prefer an accepted ID over none (never downgrade).
    if s_code is None and a_code is not None:
        survivor['expert_id'] = a_eid
        # Promote the accepted species/name/category alongside the expert_id so
        # the survivor stays coloured expert-green (mirrors reload semantics).
        for f in ('species', 'name', 'category'):
            if f in absorbed and absorbed.get(f) is not None:
                if _review_promotion_blocked(survivor.get(f), absorbed.get(f)):
                    continue
                survivor[f] = absorbed[f]

    # Carry the pending-review flag if either side was flagged. Under the lock
    # the survivor keeps a flag it already had and gains none: writing one here
    # would tag a mask that was never flagged, which is the addition Lauren
    # blocked on 2026-09-06.
    if REVIEW_TAGGING_ENABLED and absorbed.get('review') and not survivor.get('review'):
        survivor['review'] = True

    return True


def merge_overlapping_same_id(masks, image_height, image_width,
                              simplify_epsilon=0.001, min_area=1):
    """Union OVERLAPPING masks that share the same REAL species/label into one.

    On-demand counterpart to the segmentation-time `larger_wins` overlap
    resolution (which DROPS the smaller of two overlapping same-species masks
    instead of unioning them). Here we keep every pixel: any two non-rejected
    masks with the same REAL `species` that physically overlap are unioned, the
    survivor's geometry recomputed, and the absorbed mask removed.

    Placeholder species (REVIEW / '' / '?') are NEVER grouped together: two
    REVIEW masks are distinct organisms awaiting different expert IDs, so
    unioning them would silently drop one mask's review_uid / reviews[] /
    expert_id. They pass through untouched.

    When real same-species masks ARE unioned, the survivor PRESERVES the review
    state of every absorbed mask: review_uid is carried forward, reviews[] are
    merged, and an accepted expert_id is never downgraded (prefer expert over
    pending). If two members carry DIFFERENT accepted expert_ids, that pair is
    REFUSED (left unmerged) and reported via the returned `refused` list.

    Operates only on the binary footprints decoded from each mask's RLE, so it
    needs no SAM3 engine. Rejected masks are left untouched (they are "deleted"
    in the operator's mental model). The lowest-id mask in each overlap group
    becomes the survivor so its identity/colour stays stable.

    Args:
        masks: list of mask dicts (each with 'id', 'species', 'rle', 'status').
        image_height, image_width: frame dimensions for RLE decode.
        simplify_epsilon: polygon simplification for the recomputed survivor.
        min_area: drop a unioned survivor below this many pixels (defensive).

    Returns:
        (new_masks, merged_count, refused) where new_masks is the rebuilt mask
        list, merged_count is how many masks were absorbed (removed), and
        refused is a list of {'survivor_id', 'absorbed_id', 'reason'} describing
        any pair left unmerged due to conflicting accepted expert IDs.
    """
    h, w = int(image_height), int(image_width)
    if h <= 0 or w <= 0:
        return masks, 0, []

    # Decode every live mask once. Rejected masks pass through untouched.
    # Masks carrying a placeholder species also pass through (see above).
    live = []   # (mask_dict, binary)
    passthrough = []
    for m in masks:
        if m.get('status') == 'rejected':
            passthrough.append(m)
            continue
        if (m.get('species') or '') in PLACEHOLDER_SPECIES:
            passthrough.append(m)
            continue
        rle = m.get('rle')
        if not rle:
            passthrough.append(m)
            continue
        try:
            binary = np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
        except Exception:
            passthrough.append(m)
            continue
        live.append([m, binary])

    # Group by REAL species, union any pair that overlaps (transitive via a
    # simple union-find over indices within each species bucket).
    by_species = {}
    for i, (m, _b) in enumerate(live):
        by_species.setdefault(m.get('species', ''), []).append(i)

    parent = list(range(len(live)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _species, idxs in by_species.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if (live[ia][1] & live[ib][1]).any():
                    _union(ia, ib)

    # Collapse each group: lowest-id member is the survivor.
    groups = {}
    for i in range(len(live)):
        groups.setdefault(_find(i), []).append(i)

    new_masks = []
    merged_count = 0
    refused = []
    for root, members in groups.items():
        if len(members) == 1:
            new_masks.append(live[members[0]][0])
            continue
        # Pick the survivor = lowest mask id (stable identity/colour).
        members.sort(key=lambda i: live[i][0].get('id', 0))
        survivor, surv_bin = live[members[0]]
        combined = surv_bin.copy()
        absorbed_members = []
        for i in members[1:]:
            absorbed = live[i][0]
            # Refuse to merge a pair with conflicting accepted expert IDs —
            # carry the review state first so any conflict is detected before we
            # touch geometry.
            if not _merge_review_state(survivor, absorbed):
                refused.append({
                    'survivor_id': survivor.get('id'),
                    'absorbed_id': absorbed.get('id'),
                    'reason': 'conflicting accepted expert IDs',
                })
                # Leave this member as its own mask (un-absorbed).
                new_masks.append(absorbed)
                continue
            combined |= live[i][1]
            absorbed_members.append(i)
            merged_count += 1
        if not absorbed_members:
            # Every other member was refused — survivor stands alone unchanged.
            new_masks.append(survivor)
            continue
        if int(combined.sum()) < min_area:
            # Degenerate — keep originals rather than lose them.
            new_masks.append(survivor)
            for i in absorbed_members:
                new_masks.append(live[i][0])
            merged_count -= len(absorbed_members)
            continue
        update_mask_geometry(survivor, combined, simplify_epsilon)
        new_masks.append(survivor)

    new_masks.extend(passthrough)
    # Preserve a stable order: by id ascending (matches list-pane sort intent).
    new_masks.sort(key=lambda m: (m.get('id', 0)))
    return new_masks, merged_count, refused


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

    Raises:
        TypeError: when point_info has no dict-like get(), which is a caller
            fault rather than a data condition.

    Expert-review lock (Lauren Olinger, 2026-09-06): while
    REVIEW_TAGGING_ENABLED is False, no mask is born carrying a review tag. A
    point_info asking for one produces a mask with an empty species and no
    review boolean (this function never writes that key), and the drop is
    counted for review_tags_dropped() rather than raised, because the batch
    segmentation pass calls this in a loop over one frame's prompts and an
    exception would abandon the whole frame.

    Example:
        build_mask_dict(0, binary, 0.9, {'species': 'REVIEW'}) returns a mask
        whose species is '' while the lock is on.
    """
    if not callable(getattr(point_info, 'get', None)):
        raise TypeError(
            "build_mask_dict needs a dict-like point_info carrying the label, "
            "species and click position, got "
            f"{type(point_info).__name__}"
        )
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

    species = point_info.get('species', 'UNK')
    category = point_info.get('category', '')
    name = point_info.get('name', '')
    if not REVIEW_TAGGING_ENABLED and review_tag_requested(point_info):
        # Born clean: one condition governs all three display fields. A species
        # or a category that spells out the review tag is blanked, so is a name
        # that announces it, and the drop is counted. An empty species routes
        # the mask into the annotator's existing empty-species accept guard, so
        # it reads as unlabelled rather than as a mask awaiting an expert, and a
        # blanked name stops it reading 'Flagged for expert review' when it is
        # no longer flagged. A real species keeps its own name.
        species = _blank_if_review(species)
        category = _blank_if_review(category)
        name = _blank_if_review_name(name)
        _record_dropped_review_tag()

    return {
        'id': mask_id,
        'label': point_info.get('label', '?'),
        'species': species,
        'name': name,
        'category': category,
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
