"""
Core processing logic - ties parser, species lookup, and export together.
"""

from pathlib import Path
from PIL import Image

from parser import parse_cpc, parse_filename, find_image_for_cpc
from export import copy_raw_image, generate_test_image


def process_directory(cpc_dir_info, species_lookup, output_dir, log_fn,
                      test_image_limit=0, test_images_generated=0):
    """Process all CPC files in a single directory.

    Args:
        cpc_dir_info: dict from find_cpc_directories()
        species_lookup: SpeciesLookup instance
        output_dir: Path to output root
        log_fn: callable(msg) for logging
        test_image_limit: max test images to generate total
        test_images_generated: how many already generated

    Returns:
        (sam_prompts, csv_rows, stats)
    """
    cpc_dir = cpc_dir_info["cpc_dir"]
    image_dir = cpc_dir_info["image_dir"]
    dir_date = cpc_dir_info["dir_date"]
    dir_site = cpc_dir_info["dir_site"]

    raw_dir = output_dir / "raw"
    test_dir = output_dir / "test_pts"

    sam_prompts = {}
    csv_rows = []
    stats = {
        "images": 0,
        "points": 0,
        "species_matched": 0,
        "species_missed": 0,
        "fuzzy_dates": 0,
        "no_image": 0,
        "test_images": 0,
    }

    for cpc_path in cpc_dir_info["cpc_files"]:
        try:
            cpc_data = parse_cpc(str(cpc_path))
        except Exception as e:
            log_fn(f"  WARNING: Failed to parse {cpc_path.name}: {e}")
            continue

        # Find matching image
        image_path = find_image_for_cpc(cpc_path, image_dir)
        if not image_path:
            if image_dir != cpc_dir:
                image_path = find_image_for_cpc(cpc_path, cpc_dir)
            if not image_path:
                stats["no_image"] += 1
                log_fn(f"  WARNING: No image for {cpc_path.name}")
                continue

        # Get actual image dimensions for coordinate scaling
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        scale_x = img_w / cpc_data["canvas_w"]
        scale_y = img_h / cpc_data["canvas_h"]

        # Parse filename with fallbacks from directory name
        file_info = parse_filename(
            cpc_path.stem,
            fallback_date=dir_date,
            fallback_site=dir_site,
        )

        # Copy raw image
        raw_filename = copy_raw_image(image_path, raw_dir)

        # Build points
        points = []
        for coord, label_info in zip(cpc_data["coords"], cpc_data["labels"]):
            px = round(coord[0] * scale_x, 1)
            py = round(coord[1] * scale_y, 1)
            label = label_info["label"]

            species_code = ""
            species_name = ""
            category = ""

            if file_info and species_lookup:
                sp, matched_date, exact = species_lookup.lookup(
                    file_info["date"],
                    file_info["site"],
                    file_info["transect"],
                    file_info["frame"],
                    label,
                )
                if sp:
                    species_code = sp["species_code"]
                    species_name = sp["species_name"]
                    category = sp["category"]
                    stats["species_matched"] += 1
                    if not exact:
                        stats["fuzzy_dates"] += 1
                else:
                    stats["species_missed"] += 1

            # Fall back to CPC-embedded species code
            if not species_code and label_info["species_code_cpc"]:
                species_code = label_info["species_code_cpc"]

            point = {
                "label": label,
                "species": species_code,
                "name": species_name,
                "category": category,
                "x": px,
                "y": py,
                "point_type": 1,
            }
            points.append(point)

            csv_rows.append({
                "raw_image": raw_filename,
                "label": label,
                "species_code": species_code,
                "species_name": species_name,
                "category": category,
                "x": px,
                "y": py,
                "source": "cpc",
            })

        sam_prompts[raw_filename] = {
            "raw_image": f"raw/{raw_filename}",
            "points": points,
        }

        stats["images"] += 1
        stats["points"] += len(points)

        # Test overlay
        if (test_images_generated + stats["test_images"]) < test_image_limit:
            test_out = test_dir / (cpc_path.stem + "_test.jpg")
            try:
                generate_test_image(str(image_path), points, str(test_out))
                stats["test_images"] += 1
            except Exception as e:
                log_fn(f"  WARNING: test image failed for {cpc_path.name}: {e}")

    return sam_prompts, csv_rows, stats
