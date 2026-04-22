#!/usr/bin/env python3
"""
Speed comparison: EasyOCR (GPU) vs Tesseract (CPU) on _pts images.

Runs detection on 3 sample images with each engine and reports:
- Per-image time
- Total time
- Letters correctly identified (with confidence)
- Speedup factor

Usage:
    python speed_test.py [--images-dir PATH]
"""
import os
import sys
import time
import glob
import argparse
import json

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect import detect_annotations, detect_annotations_tesseract, get_ocr_reader


def find_test_images(images_dir, count=3):
    """Find _pts images for testing."""
    patterns = [
        os.path.join(images_dir, '**', '*_pts.jpg'),
        os.path.join(images_dir, '**', '*_pts.jpeg'),
        os.path.join(images_dir, '**', '*_pts.png'),
    ]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))
    all_files.sort()
    return all_files[:count]


def format_results(annotations):
    """Summarize detection results."""
    if not annotations:
        return "  (no detections)"
    lines = []
    identified = sum(1 for v in annotations.values() if v['ocr_conf'] > 0)
    crosshairs = sum(1 for v in annotations.values() if v['has_crosshair'])
    lines.append(f"  Detected: {len(annotations)} letters, {identified} OCR-identified, {crosshairs} crosshairs")
    for letter in sorted(annotations.keys()):
        ann = annotations[letter]
        conf_str = f"{ann['ocr_conf']}%" if ann['ocr_conf'] > 0 else "unidentified"
        cross_str = f"({ann['cross_x']:.0f}, {ann['cross_y']:.0f})" if ann['has_crosshair'] else "(no crosshair)"
        lines.append(f"    {letter}: conf={conf_str}  point={cross_str}")
    return "\n".join(lines)


def run_speed_test(images_dir):
    print("=" * 70)
    print("TCRMPclip_letterID - OCR Speed Test")
    print("=" * 70)

    # Find test images
    test_images = find_test_images(images_dir)
    if not test_images:
        print(f"ERROR: No _pts images found in {images_dir}")
        sys.exit(1)

    print(f"\nTest images ({len(test_images)}):")
    for img in test_images:
        print(f"  {os.path.basename(img)}")

    # ── GPU info ──
    try:
        import torch
        print(f"\nPyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  WARNING: CUDA not available, EasyOCR will fall back to CPU")
    except ImportError:
        print("\n  WARNING: PyTorch not installed")

    # ══════════════════════════════════════════════════════════════════
    # EasyOCR (GPU)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("EasyOCR (GPU-accelerated)")
    print("─" * 70)

    # Warm up: load model (not counted in per-image timing)
    print("Loading EasyOCR model (one-time)...", end=" ", flush=True)
    t0 = time.perf_counter()
    reader = get_ocr_reader(gpu=True)
    model_load_time = time.perf_counter() - t0
    print(f"{model_load_time:.2f}s")

    easyocr_times = []
    easyocr_results = []
    for img_path in test_images:
        name = os.path.basename(img_path)
        t0 = time.perf_counter()
        annotations = detect_annotations(img_path, gpu=True)
        elapsed = time.perf_counter() - t0
        easyocr_times.append(elapsed)
        easyocr_results.append(annotations)
        print(f"\n[{name}] {elapsed:.3f}s")
        print(format_results(annotations))

    easyocr_total = sum(easyocr_times)
    easyocr_avg = easyocr_total / len(easyocr_times)
    print(f"\nEasyOCR summary: total={easyocr_total:.3f}s  avg={easyocr_avg:.3f}s/image")

    # ══════════════════════════════════════════════════════════════════
    # Tesseract (CPU)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("Tesseract (CPU-only)")
    print("─" * 70)

    tesseract_available = True
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("Tesseract binary found")
    except Exception as e:
        print(f"Tesseract not available: {e}")
        print("Install with: sudo apt-get install tesseract-ocr")
        tesseract_available = False

    tesseract_times = []
    tesseract_results = []
    if tesseract_available:
        for img_path in test_images:
            name = os.path.basename(img_path)
            t0 = time.perf_counter()
            annotations = detect_annotations_tesseract(img_path)
            elapsed = time.perf_counter() - t0
            tesseract_times.append(elapsed)
            tesseract_results.append(annotations)
            print(f"\n[{name}] {elapsed:.3f}s")
            print(format_results(annotations))

        tess_total = sum(tesseract_times)
        tess_avg = tess_total / len(tesseract_times)
        print(f"\nTesseract summary: total={tess_total:.3f}s  avg={tess_avg:.3f}s/image")

    # ══════════════════════════════════════════════════════════════════
    # Comparison
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Engine':<20} {'Total (3 imgs)':<18} {'Avg/image':<15} {'Model load':<12}")
    print(f"{'─'*20} {'─'*18} {'─'*15} {'─'*12}")
    print(f"{'EasyOCR (GPU)':<20} {easyocr_total:<18.3f} {easyocr_avg:<15.3f} {model_load_time:<12.2f}")
    if tesseract_available:
        print(f"{'Tesseract (CPU)':<20} {tess_total:<18.3f} {tess_avg:<15.3f} {'N/A':<12}")
        speedup = tess_avg / easyocr_avg if easyocr_avg > 0 else float('inf')
        print(f"\nEasyOCR is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than Tesseract per image")
    else:
        print(f"{'Tesseract (CPU)':<20} {'(not installed)':<18}")

    # Accuracy comparison
    if tesseract_available:
        print(f"\n{'Engine':<20} {'Letters IDed':<15} {'Avg confidence':<18}")
        print(f"{'─'*20} {'─'*15} {'─'*18}")
        for label, results in [("EasyOCR", easyocr_results), ("Tesseract", tesseract_results)]:
            total_id = 0
            total_conf = 0
            total_pts = 0
            for ann in results:
                for v in ann.values():
                    total_pts += 1
                    if v['ocr_conf'] > 0:
                        total_id += 1
                        total_conf += v['ocr_conf']
            avg_conf = total_conf / total_id if total_id > 0 else 0
            print(f"{label:<20} {total_id}/{total_pts:<13} {avg_conf:<18.1f}")

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default=os.path.join(
        os.path.dirname(__file__), '..', '..', 'input', 'TCRMP_clip'))
    args = parser.parse_args()
    run_speed_test(args.images_dir)
