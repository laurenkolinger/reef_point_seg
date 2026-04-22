#!/usr/bin/env python3
"""
Generate test overlay images from existing CPC Reader output.

Reads sam_click_prompts.json and raw/ images, draws crosshair + label overlays
into test_pts/. Can be run after a batch to generate all overlays in parallel.

Usage:
    python generate_test_pts.py <output_root> [--workers N]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from export import generate_test_image


def generate_for_directory(output_dir):
    """Generate test_pts for all images in a single output directory.

    Returns (generated, skipped, errors) counts.
    """
    output_dir = Path(output_dir)
    sam_path = output_dir / "ids" / "sam_click_prompts.json"
    raw_dir = output_dir / "raw"
    test_dir = output_dir / "test_pts"

    if not sam_path.exists():
        return 0, 0, 0

    os.makedirs(test_dir, exist_ok=True)

    with open(sam_path) as f:
        sam_data = json.load(f)

    generated = 0
    skipped = 0
    errors = 0

    for image_name, entry in sam_data.items():
        stem = Path(image_name).stem
        test_out = test_dir / f"{stem}_test.jpg"

        if test_out.exists():
            skipped += 1
            continue

        raw_path = raw_dir / image_name
        if not raw_path.exists():
            errors += 1
            continue

        try:
            generate_test_image(str(raw_path), entry["points"], str(test_out))
            generated += 1
        except Exception as e:
            print(f"  ERROR: {image_name}: {e}", file=sys.stderr)
            errors += 1

    return generated, skipped, errors


def _process_one_dir(dir_path):
    """Worker function for parallel processing."""
    try:
        gen, skip, err = generate_for_directory(dir_path)
        return str(dir_path), gen, skip, err
    except Exception:
        return str(dir_path), 0, 0, -1


def main():
    ap = argparse.ArgumentParser(
        description="Generate test overlay images from existing output"
    )
    ap.add_argument("output_root", help="Output directory (single or batch root)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = ap.parse_args()

    root = Path(args.output_root).resolve()

    if (root / "ids" / "sam_click_prompts.json").exists():
        dirs = [root]
    else:
        dirs = sorted(
            d.parent.parent
            for d in root.rglob("ids/sam_click_prompts.json")
        )

    if not dirs:
        print(f"No sam_click_prompts.json found under {root}")
        sys.exit(1)

    print(f"Generating test_pts for {len(dirs)} directories...")

    total_gen = 0
    total_skip = 0
    total_err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one_dir, d): d for d in dirs}
        done = 0
        for future in as_completed(futures):
            dir_path, gen, skip, err = future.result()
            total_gen += gen
            total_skip += skip
            total_err += max(err, 0)
            done += 1
            name = Path(dir_path).name
            if gen > 0:
                print(f"  [{done}/{len(dirs)}] {name}: {gen} generated"
                      + (f", {skip} existed" if skip else "")
                      + (f", {err} errors" if err > 0 else ""))
            elif done % 20 == 0:
                print(f"  [{done}/{len(dirs)}] progress...")

    print(f"\nDone: {total_gen} generated, {total_skip} already existed, {total_err} errors")


if __name__ == "__main__":
    main()
