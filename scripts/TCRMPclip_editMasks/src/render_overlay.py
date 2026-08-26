"""
Generate downscaled overlay images showing segmentation masks with labels.

Produces a JPEG with:
  - 10% opacity filled mask polygons (category-colored)
  - Solid outline around each mask
  - Label text (letter + species code) at mask centroid
  - Small dot at the source (x,y) click coordinate

Called after initial SAM3 segmentation and again after review/export.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Category color mapping (RGB tuples)
CATEGORY_COLORS = {
    'Coral':         (220, 50, 50),
    'Sponge':        (160, 80, 200),
    'Macroalgae':    (50, 160, 50),
    'Gorgonian':     (230, 150, 30),
    'Dca':           (60, 140, 200),
    'Turf':          (140, 160, 40),
    'Non-living':    (130, 130, 130),
    'Cyanobacteria': (40, 180, 170),
    'Calcareous':    (180, 160, 100),
}

# Slight hue variation per species
def _species_color(category, species):
    base = CATEGORY_COLORS.get(category, (130, 130, 130))
    if not species:
        return base
    offset = sum(ord(c) for c in species) % 40 - 20
    return tuple(max(0, min(255, c + offset)) for c in base)


def _try_font(size):
    """Try to load a monospace font, fall back to default."""
    for name in ['/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
                 '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf']:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def render_overlay(image_path, seg_dict, output_path, max_long_edge=1600):
    """Render a downscaled overlay image showing masks + labels.

    Args:
        image_path: path to the raw source image
        seg_dict: segmentation dict with 'masks', 'image_width', 'image_height'
        output_path: where to save the JPEG
        max_long_edge: downscale so longest edge <= this (pixels)
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Downscale
    long_edge = max(orig_w, orig_h)
    if long_edge > max_long_edge:
        sc = max_long_edge / long_edge
        new_w = int(orig_w * sc)
        new_h = int(orig_h * sc)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        sc = 1.0
        new_w, new_h = orig_w, orig_h

    # Create overlay layer for semi-transparent fills
    overlay = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    # Draw on the main image for outlines and text
    draw = ImageDraw.Draw(img)

    font_label = _try_font(max(11, int(14 * sc)))
    font_small = _try_font(max(9, int(11 * sc)))

    masks = seg_dict.get('masks', [])

    for m in masks:
        status = m.get('status', 'pending')
        if status == 'rejected':
            continue

        color = _species_color(m.get('category', ''), m.get('species', ''))
        fill_alpha = 25  # ~10% of 255

        # Status-specific outline
        if status == 'accepted':
            outline_color = color
            outline_width = max(1, int(2 * sc))
        else:
            # pending: dashed feel via thinner line
            outline_color = tuple(min(255, c + 40) for c in color)
            outline_width = max(1, int(1.5 * sc))

        polygons_px = m.get('polygon_px', [])

        for poly in polygons_px:
            if len(poly) < 6:
                continue
            # Scale polygon to output size
            scaled = []
            for i in range(0, len(poly), 2):
                scaled.append((poly[i] * sc, poly[i + 1] * sc))

            # Fill on overlay (10% opacity)
            ov_draw.polygon(scaled, fill=(*color, fill_alpha))

            # Outline on main image
            draw.polygon(scaled, outline=outline_color, width=outline_width)

        # Label at centroid of first polygon
        if polygons_px and len(polygons_px[0]) >= 4:
            poly = polygons_px[0]
            cx = sum(poly[i] for i in range(0, len(poly), 2)) / (len(poly) // 2) * sc
            cy = sum(poly[i] for i in range(1, len(poly), 2)) / (len(poly) // 2) * sc

            label_text = f"{m.get('label', '?')} {m.get('species', '')}"

            # Text with background
            bbox = font_label.getbbox(label_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = cx - tw / 2
            ty = cy - th / 2

            # Background rectangle
            pad = 2
            draw.rectangle(
                [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
                fill=(0, 0, 0, 180),
            )
            draw.text((tx, ty), label_text, fill=(255, 255, 255), font=font_label)

        # Source click point dot
        sx = m.get('source_x')
        sy = m.get('source_y')
        if sx is not None and sy is not None:
            px = sx * sc
            py = sy * sc
            r = max(3, int(4 * sc))
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(255, 255, 255), width=1)

    # Composite the overlay onto the main image
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, "JPEG", quality=88)


def render_segmentation_overlays(seg_dict, filename, export_dir, stage="auto"):
    """Render overlay for a single image and save to the flat overlay dir.

    Saves to: export_dir/segmentations/overlays_{stage}/{stem}_seg.jpg
    """
    image_path = seg_dict.get('image_path_abs')
    if not image_path or not os.path.exists(image_path):
        return None

    stem = os.path.splitext(filename)[0]
    overlay_dir = os.path.join(export_dir, 'segmentations', f'overlays_{stage}')
    output_path = os.path.join(overlay_dir, f'{stem}_seg.jpg')

    render_overlay(image_path, seg_dict, output_path)
    return output_path
