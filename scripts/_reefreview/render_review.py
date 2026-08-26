"""
Per-mask review imagery for the expert viewer.

Produces, for one flagged mask:
  crop.jpg  cropped closeup (bbox + padding) with the mask OUTLINE and a BBOX
            frame baked in (always-on decorations the spec asks for)
  mask.png  the mask footprint within the crop as a white-on-transparent alpha
            cutout — the viewer tints + composites this over crop.jpg at the
            opacity-slider value, giving a true adjustable-opacity overlay
  full.jpg  a downscaled full frame with the mask outlined + source dot, for the
            optional "see the whole frame" toggle

Geometry comes straight off the stored mask dict (bbox, polygon_px, rle); no
SAM3 call is needed. Outline color matches the cyan REVIEW accent used elsewhere.
"""

import os

import numpy as np
from PIL import Image, ImageDraw

from . import mask_geom

REVIEW_RGB = (0, 200, 220)      # cyan accent for REVIEW masks
BBOX_RGB = (255, 210, 0)        # amber bbox frame


def _polys_px(mask):
    """Return list of [(x,y), ...] polygons in pixel coords from polygon_px."""
    out = []
    for flat in mask.get('polygon_px', []) or []:
        if len(flat) >= 6:
            out.append([(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)])
    return out


def render_closeup(image_path, mask, out_crop, out_mask,
                   pad_px=40, max_edge=720):
    """Write crop.jpg (outline + bbox baked) and mask.png (alpha cutout).

    Returns {crop_w, crop_h, scale, crop_origin:[x0,y0]} or None on failure.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    bbox = mask.get('bbox')
    if not bbox:
        return None

    img = Image.open(image_path).convert('RGB')
    W, H = img.size

    x0, y0, x1, y1 = bbox
    cx0 = max(0, int(x0) - pad_px)
    cy0 = max(0, int(y0) - pad_px)
    cx1 = min(W, int(x1) + pad_px)
    cy1 = min(H, int(y1) + pad_px)
    if cx1 <= cx0 or cy1 <= cy0:
        return None

    crop = img.crop((cx0, cy0, cx1, cy1))
    cw, ch = crop.size
    long_edge = max(cw, ch)
    scale = (max_edge / long_edge) if long_edge > max_edge else 1.0
    sw, sh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    if scale != 1.0:
        crop = crop.resize((sw, sh), Image.LANCZOS)

    draw = ImageDraw.Draw(crop)

    # Mask outline (shifted to crop origin, scaled).
    for poly in _polys_px(mask):
        pts = [((px - cx0) * scale, (py - cy0) * scale) for px, py in poly]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=REVIEW_RGB, width=max(2, int(3 * scale)))

    # Bbox frame (the original bbox inset within the padded crop).
    bx0 = (x0 - cx0) * scale
    by0 = (y0 - cy0) * scale
    bx1 = (x1 - cx0) * scale
    by1 = (y1 - cy0) * scale
    draw.rectangle([bx0, by0, bx1, by1], outline=BBOX_RGB, width=max(1, int(2 * scale)))

    os.makedirs(os.path.dirname(out_crop), exist_ok=True)
    crop.save(out_crop, 'JPEG', quality=85)

    # Mask alpha cutout, same pixel grid as the (scaled) crop.
    rle = mask.get('rle')
    if rle:
        binary = mask_geom.rle_decode(rle, shape=tuple(rle['size']))
        sub = binary[cy0:cy1, cx0:cx1]
        alpha = (sub.astype(np.uint8)) * 255
        a_img = Image.fromarray(alpha, mode='L')
        if scale != 1.0:
            a_img = a_img.resize((sw, sh), Image.NEAREST)
        rgba = Image.new('RGBA', a_img.size, (255, 255, 255, 0))
        white = Image.new('RGBA', a_img.size, (255, 255, 255, 255))
        rgba = Image.composite(white, rgba, a_img)
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)
        rgba.save(out_mask, 'PNG')

    return {'crop_w': sw, 'crop_h': sh, 'scale': scale, 'crop_origin': [cx0, cy0]}


def render_full(image_path, mask, out_full, max_edge=1400):
    """Write full.jpg — the whole frame downscaled, this mask outlined + a dot
    at the source click. Returns {w, h} or None."""
    if not image_path or not os.path.exists(image_path):
        return None
    img = Image.open(image_path).convert('RGB')
    W, H = img.size
    long_edge = max(W, H)
    scale = (max_edge / long_edge) if long_edge > max_edge else 1.0
    sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    if scale != 1.0:
        img = img.resize((sw, sh), Image.LANCZOS)

    draw = ImageDraw.Draw(img)
    for poly in _polys_px(mask):
        pts = [(px * scale, py * scale) for px, py in poly]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=REVIEW_RGB, width=max(2, int(3 * scale)))

    sx, sy = mask.get('source_x'), mask.get('source_y')
    if sx is not None and sy is not None:
        px, py = sx * scale, sy * scale
        r = max(4, int(6 * scale))
        draw.ellipse([px - r, py - r, px + r, py + r],
                     outline=(255, 255, 255), width=2, fill=REVIEW_RGB)

    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    img.save(out_full, 'JPEG', quality=82)
    return {'w': sw, 'h': sh}
