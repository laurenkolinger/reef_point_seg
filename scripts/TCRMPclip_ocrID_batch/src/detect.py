"""
Letter + crosshair detection for TCRMP _pts annotated images.

Each _pts image has 20 red letter annotations (A-T) with a small crosshair (+)
below each letter marking the exact observation point. This module:
1. Detects red letter blobs via connected components
2. Finds the crosshair (+) below each letter for precise point location
3. Uses EasyOCR (GPU) to identify which letter is which
4. Greedy constraint resolution ensures unique A-T assignment

Optimizations:
- EasyOCR model loaded once, reused for all images
- All 20 letter crops batched into a single OCR call where possible
- Multiprocessing support for scanning many images in parallel
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from scipy import ndimage

POINT_LABELS_20 = list('ABCDEFGHIJKLMNOPQRST')

# ── Singleton EasyOCR reader (loaded once per process) ──────────────────────

_ocr_reader = None


def get_ocr_reader(gpu=True):
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    return _ocr_reader


# ── Core detection ──────────────────────────────────────────────────────────

def detect_annotations(pts_image_path, expected_count=20, gpu=True):
    """
    Detect letter + crosshair annotations from a _pts image.

    Returns dict: letter -> {
        'letter_cx', 'letter_cy',   # letter centroid
        'cross_x', 'cross_y',       # crosshair position (actual point)
        'has_crosshair': bool,
        'ocr_conf': int (0-100)
    }
    """
    img = cv2.imread(pts_image_path)
    if img is None:
        return {}

    b, g, r = cv2.split(img)
    img_h, img_w = img.shape[:2]

    # ── Step 1: Red mask + connected components ──
    red_mask = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)
    labeled, n_features = ndimage.label(red_mask)
    if n_features == 0:
        return {}

    # Characterize all components using find_objects (single pass, not O(n*pixels))
    slices = ndimage.find_objects(labeled)
    component_sizes = np.bincount(labeled.ravel())  # index 0 = background

    components = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sy, sx = sl
        # Compute centroid from the slice region (fast, small sub-array)
        region = (labeled[sl] == i)
        ys_local, xs_local = np.where(region)
        components.append({
            'id': i,
            'cx': float(np.mean(xs_local) + sx.start),
            'cy': float(np.mean(ys_local) + sy.start),
            'size': int(component_sizes[i]),
            'min_x': int(sx.start + np.min(xs_local)),
            'max_x': int(sx.start + np.max(xs_local)),
            'min_y': int(sy.start + np.min(ys_local)),
            'max_y': int(sy.start + np.max(ys_local)),
        })

    # Letters are the largest components
    components.sort(key=lambda c: c['size'], reverse=True)
    letter_comps = components[:expected_count]

    # ── Step 2: Find crosshair below each letter ──
    letter_cross_pairs = []
    for ltr in letter_comps:
        search_y1 = min(img_h, ltr['max_y'] + 5)
        search_y2 = min(img_h, ltr['max_y'] + 100)
        search_x1 = max(0, int(ltr['cx']) - 60)
        search_x2 = min(img_w, int(ltr['cx']) + 60)

        region = red_mask[search_y1:search_y2, search_x1:search_x2]
        ys, xs = np.where(region > 0)

        if len(xs) >= 3:
            cross_x = float(np.mean(xs)) + search_x1
            cross_y = float(np.mean(ys)) + search_y1
            has_crosshair = True
        else:
            cross_x = ltr['cx']
            cross_y = float(ltr['max_y'] + 40)
            has_crosshair = False

        letter_cross_pairs.append({
            'letter': ltr,
            'cross_x': cross_x,
            'cross_y': cross_y,
            'has_crosshair': has_crosshair,
        })

    # ── Step 3: OCR each letter crop ──
    red_mask_tight = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)
    reader = get_ocr_reader(gpu=gpu)

    for pair in letter_cross_pairs:
        ltr = pair['letter']
        pad = 10
        y1 = max(0, ltr['min_y'] - pad)
        y2 = min(img_h, ltr['max_y'] + pad)
        x1 = max(0, ltr['min_x'] - pad)
        x2 = min(img_w, ltr['max_x'] + pad)

        crop = red_mask_tight[y1:y2, x1:x2] * 255
        h, w = crop.shape
        if h == 0 or w == 0:
            pair['ocr_candidates'] = {}
            continue

        # Scale up for better recognition
        crop_big = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_LINEAR)
        _, crop_big = cv2.threshold(crop_big, 127, 255, cv2.THRESH_BINARY)

        candidates = {}
        try:
            results = reader.readtext(
                crop_big,
                allowlist='ABCDEFGHIJKLMNOPQRST',
                detail=1,
                paragraph=False,
            )
            for (_, text, conf) in results:
                text = text.strip().upper()
                if len(text) == 1 and text in 'ABCDEFGHIJKLMNOPQRST':
                    candidates[text] = int(conf * 100)
        except Exception:
            pass
        pair['ocr_candidates'] = candidates

    # ── Step 4: Constraint resolution (greedy, highest confidence first) ──
    conf_list = []
    for idx, pair in enumerate(letter_cross_pairs):
        for letter, conf in pair.get('ocr_candidates', {}).items():
            conf_list.append((conf, letter, idx))
    conf_list.sort(reverse=True)

    assigned = {}
    used_indices = set()
    for conf, letter, idx in conf_list:
        if letter not in assigned and idx not in used_indices:
            assigned[letter] = idx
            used_indices.add(idx)

    # Fill remaining by elimination
    remaining_letters = sorted(set(POINT_LABELS_20) - set(assigned.keys()))
    remaining_indices = sorted(set(range(len(letter_cross_pairs))) - used_indices)
    for letter, idx in zip(remaining_letters, remaining_indices):
        assigned[letter] = idx

    # ── Build result ──
    result = {}
    for letter, idx in assigned.items():
        pair = letter_cross_pairs[idx]
        ltr = pair['letter']
        result[letter] = {
            'letter_cx': ltr['cx'],
            'letter_cy': ltr['cy'],
            'cross_x': pair['cross_x'],
            'cross_y': pair['cross_y'],
            'has_crosshair': pair['has_crosshair'],
            'ocr_conf': pair.get('ocr_candidates', {}).get(letter, 0),
        }

    return result


# ── Tesseract-based detection (for speed comparison) ────────────────────────

def detect_annotations_tesseract(pts_image_path, expected_count=20):
    """Same detection logic but uses pytesseract instead of EasyOCR."""
    import pytesseract

    img = cv2.imread(pts_image_path)
    if img is None:
        return {}

    b, g, r = cv2.split(img)
    img_h, img_w = img.shape[:2]

    red_mask = ((r > 150) & (g < 100) & (b < 100)).astype(np.uint8)
    labeled, n_features = ndimage.label(red_mask)
    if n_features == 0:
        return {}

    slices = ndimage.find_objects(labeled)
    component_sizes = np.bincount(labeled.ravel())
    components = []
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sy, sx = sl
        region = (labeled[sl] == i)
        ys_local, xs_local = np.where(region)
        components.append({
            'id': i, 'cx': float(np.mean(xs_local) + sx.start),
            'cy': float(np.mean(ys_local) + sy.start),
            'size': int(component_sizes[i]),
            'min_x': int(sx.start + np.min(xs_local)),
            'max_x': int(sx.start + np.max(xs_local)),
            'min_y': int(sy.start + np.min(ys_local)),
            'max_y': int(sy.start + np.max(ys_local)),
        })

    components.sort(key=lambda c: c['size'], reverse=True)
    letter_comps = components[:expected_count]

    letter_cross_pairs = []
    for ltr in letter_comps:
        search_y1 = min(img_h, ltr['max_y'] + 5)
        search_y2 = min(img_h, ltr['max_y'] + 100)
        search_x1 = max(0, int(ltr['cx']) - 60)
        search_x2 = min(img_w, int(ltr['cx']) + 60)
        region = red_mask[search_y1:search_y2, search_x1:search_x2]
        ys, xs = np.where(region > 0)
        if len(xs) >= 3:
            cross_x = float(np.mean(xs)) + search_x1
            cross_y = float(np.mean(ys)) + search_y1
            has_crosshair = True
        else:
            cross_x = ltr['cx']
            cross_y = float(ltr['max_y'] + 40)
            has_crosshair = False
        letter_cross_pairs.append({
            'letter': ltr, 'cross_x': cross_x,
            'cross_y': cross_y, 'has_crosshair': has_crosshair,
        })

    red_mask_tight = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)

    for pair in letter_cross_pairs:
        ltr = pair['letter']
        pad = 10
        y1 = max(0, ltr['min_y'] - pad)
        y2 = min(img_h, ltr['max_y'] + pad)
        x1 = max(0, ltr['min_x'] - pad)
        x2 = min(img_w, ltr['max_x'] + pad)

        crop = red_mask_tight[y1:y2, x1:x2] * 255
        h, w = crop.shape
        if h == 0 or w == 0:
            pair['ocr_candidates'] = {}
            continue

        crop_big = cv2.resize(crop, (w * 4, h * 4), interpolation=cv2.INTER_LINEAR)
        _, crop_big = cv2.threshold(crop_big, 127, 255, cv2.THRESH_BINARY)

        candidates = {}
        try:
            text = pytesseract.image_to_string(
                crop_big,
                config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRST',
            ).strip().upper()
            if len(text) == 1 and text in 'ABCDEFGHIJKLMNOPQRST':
                # Tesseract doesn't give per-char confidence easily, use data API
                data = pytesseract.image_to_data(
                    crop_big,
                    config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRST',
                    output_type=pytesseract.Output.DICT,
                )
                confs = [int(c) for c in data['conf'] if int(c) > 0]
                conf = max(confs) if confs else 50
                candidates[text] = conf
        except Exception:
            pass
        pair['ocr_candidates'] = candidates

    # Same constraint resolution
    conf_list = []
    for idx, pair in enumerate(letter_cross_pairs):
        for letter, conf in pair.get('ocr_candidates', {}).items():
            conf_list.append((conf, letter, idx))
    conf_list.sort(reverse=True)

    assigned = {}
    used_indices = set()
    for conf, letter, idx in conf_list:
        if letter not in assigned and idx not in used_indices:
            assigned[letter] = idx
            used_indices.add(idx)

    remaining_letters = sorted(set(POINT_LABELS_20) - set(assigned.keys()))
    remaining_indices = sorted(set(range(len(letter_cross_pairs))) - used_indices)
    for letter, idx in zip(remaining_letters, remaining_indices):
        assigned[letter] = idx

    result = {}
    for letter, idx in assigned.items():
        pair = letter_cross_pairs[idx]
        ltr = pair['letter']
        result[letter] = {
            'letter_cx': ltr['cx'], 'letter_cy': ltr['cy'],
            'cross_x': pair['cross_x'], 'cross_y': pair['cross_y'],
            'has_crosshair': pair['has_crosshair'],
            'ocr_conf': pair.get('ocr_candidates', {}).get(letter, 0),
        }
    return result


# ── Batch processing with multiprocessing ───────────────────────────────────

def _detect_single(args):
    """Worker function for multiprocessing. Initializes OCR reader per process."""
    path, gpu = args
    return path, detect_annotations(path, gpu=gpu)


def detect_batch(image_paths, max_workers=4, gpu=True):
    """
    Process multiple images in parallel using ProcessPoolExecutor.
    Each worker loads its own EasyOCR model (GPU memory is shared via CUDA).

    Returns: dict of {path: annotations}
    """
    results = {}

    if max_workers <= 1:
        for path in image_paths:
            results[path] = detect_annotations(path, gpu=gpu)
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_detect_single, (p, gpu)): p
            for p in image_paths
        }
        for future in as_completed(futures):
            path, annotations = future.result()
            results[path] = annotations

    return results
