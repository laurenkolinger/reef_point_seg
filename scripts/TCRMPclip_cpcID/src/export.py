"""
Export functions - write point data in SAM-click-prompt compatible format.

Output structure:
  raw/                        - clean source images
  ids/
    sam_click_prompts.json    - SAM3 click prompt format
    point_coords.csv          - flat CSV of all points
  test_pts/                   - overlay images with crosshair + labels
  log.txt                     - processing log
"""

import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def export_sam_prompts(data, output_path):
    """Write sam_click_prompts.json."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def export_point_coords(rows, output_path):
    """Write point_coords.csv."""
    fieldnames = [
        "raw_image", "label", "species_code", "species_name",
        "category", "x", "y", "source",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_raw_image(src_path, dest_dir):
    """Copy an image to the raw/ directory. Returns the destination filename."""
    dest = dest_dir / src_path.name
    if not dest.exists():
        shutil.copy2(str(src_path), str(dest))
    return src_path.name


def generate_test_image(image_path, points, output_path):
    """Draw crosshairs and letter labels on an image for visual verification."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    font, font_small = _load_fonts(48, 28)
    cross_size = 20

    for pt in points:
        x, y = pt["x"], pt["y"]
        label = pt["label"]
        species = pt.get("species", "")

        # Crosshair
        draw.line([(x - cross_size, y), (x + cross_size, y)], fill="red", width=3)
        draw.line([(x, y - cross_size), (x, y + cross_size)], fill="red", width=3)

        # Letter label above
        text_x = x - 12
        text_y = y - cross_size - 54
        bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(
            [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
            fill=(0, 0, 0, 180),
        )
        draw.text((text_x, text_y), label, fill="red", font=font)

        # Species code below
        if species:
            sp_y = y + cross_size + 4
            bbox2 = draw.textbbox((text_x, sp_y), species, font=font_small)
            draw.rectangle(
                [bbox2[0] - 1, bbox2[1] - 1, bbox2[2] + 1, bbox2[3] + 1],
                fill=(0, 0, 0, 180),
            )
            draw.text((text_x, sp_y), species, fill="yellow", font=font_small)

    img.save(output_path, quality=92)
    return True


def _load_fonts(size_large, size_small):
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                return (
                    ImageFont.truetype(font_path, size_large),
                    ImageFont.truetype(font_path, size_small),
                )
            except Exception:
                pass
    default = ImageFont.load_default()
    return default, default
