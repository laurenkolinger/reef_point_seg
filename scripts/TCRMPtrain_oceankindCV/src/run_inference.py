#!/usr/bin/env python3
"""Run a trained Ultralytics model against one of four image sources and
emit overlays + a manifest.json that the orchestrator renders as a gallery.

Source modes:
    dir     — arbitrary directory of images
    unused  — TCRMP frames under --clip_dir that are NOT listed in
              --selected_frames (the step-3 csv). Random / systematic sample.
    full    — every raw frame under --clip_dir. Usually very large.

Outputs (inside --out_dir):
    overlays/            per-image overlay renderings (.jpg)
    crops/               per-detection crops (only if --save_crops)
    manifest.json        gallery + per-image summary + blind-spot signal
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta


IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

AST = timezone(timedelta(hours=-4))


def ast_now():
    """AST (UTC-4, fixed) timestamp, seconds precision, ISO-8601."""
    return datetime.now(AST).isoformat(timespec='seconds')


def arg_parse():
    p = argparse.ArgumentParser(description='TCRMP step 8 model inference')
    p.add_argument('--run_dir', required=True, help='dir with weights/best.pt')
    p.add_argument('--out_dir', required=True, help='destination for overlays + manifest')
    p.add_argument('--source_type', required=True, choices=['dir', 'unused', 'full'])
    p.add_argument('--source_dir', default='', help='required when source_type=dir')
    p.add_argument('--clip_dir', default='', help='TCRMP_clip root (unused/full)')
    p.add_argument('--selected_frames', default='',
                   help='step 3 selected_frames.csv (for unused-frame exclusion)')
    p.add_argument('--sample_count', type=int, default=100)
    p.add_argument('--sample_mode', default='random', choices=['random', 'systematic'])
    p.add_argument('--include_pts_variant', action='store_true',
                   help='include the _pts.jpg overlay variants alongside raw frames')
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--save_overlays', action='store_true')
    p.add_argument('--save_crops', action='store_true')
    p.add_argument('--save_predictions', action='store_true',
                   help='Also write predictions.json with per-detection normalized mask '
                        'polygons and confidences (machine-readable, consumed by the Refine loop)')
    p.add_argument('--device', default='0')

    # Rendering controls — all user-settable from step 8 panel.
    p.add_argument('--mask_alpha', type=float, default=0.45,
                   help='fill opacity for mask overlays (0.0 = invisible, 1.0 = solid)')
    p.add_argument('--draw_boxes', action='store_true',
                   help='draw bounding boxes in addition to masks (default: masks only)')
    p.add_argument('--run_name', default='',
                   help='human-friendly name stamped in the overlay footer')
    p.add_argument('--project_name', default='',
                   help='project name stamped in the overlay footer')
    return p.parse_args()


def _find_best_weights(run_dir):
    for cand in ('best.pt', 'last.pt'):
        p = os.path.join(run_dir, 'weights', cand)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'No weights in {run_dir}/weights/')


def _load_selected_basenames(path):
    """Return the set of basenames already in step 3's selected_frames.csv."""
    used = set()
    if not path or not os.path.isfile(path):
        return used
    try:
        with open(path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                bn = (row.get('basename') or row.get('filename') or '').strip()
                if bn:
                    used.add(bn)
    except Exception as e:
        print(f'[infer] could not read {path}: {e}')
    return used


def _walk_images(root, include_pts):
    """Yield absolute paths to images under root. Skips _pts.jpg unless asked."""
    if not root or not os.path.isdir(root):
        return
    for dp, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            if not include_pts and '_pts.' in f.lower():
                continue
            yield os.path.join(dp, f)


def collect_source(args):
    """Return a list of absolute image paths per source_type."""
    if args.source_type == 'dir':
        if not args.source_dir or not os.path.isdir(args.source_dir):
            raise ValueError(f'source_dir not found: {args.source_dir}')
        imgs = list(_walk_images(args.source_dir, include_pts=args.include_pts_variant))
        imgs.sort()
        return imgs

    if args.source_type in ('unused', 'full'):
        if not args.clip_dir or not os.path.isdir(args.clip_dir):
            raise ValueError(f'clip_dir not found: {args.clip_dir}')
        all_imgs = list(_walk_images(args.clip_dir, include_pts=args.include_pts_variant))
        all_imgs.sort()
        if args.source_type == 'full':
            return all_imgs

        # unused: subtract anything whose basename (no extension) appears in selected_frames
        used_basenames = _load_selected_basenames(args.selected_frames)
        def _bn(p):
            b = os.path.splitext(os.path.basename(p))[0]
            # strip _pts if present so we match the CSV's basename
            return b[:-4] if b.lower().endswith('_pts') else b
        candidates = [p for p in all_imgs if _bn(p) not in used_basenames]
        if not candidates:
            return []
        n = min(args.sample_count, len(candidates))
        if args.sample_mode == 'random':
            random.seed(0)
            return sorted(random.sample(candidates, n))
        # systematic: every ceil(N/n)-th
        step = max(1, len(candidates) // n)
        return candidates[::step][:n]

    return []


def build_prediction_items(results_by_file, class_names):
    """(filename, raw_abs, ultralytics_result) -> predictions.json items.
    Keeps zero-detection frames (empty detections list); skips detections
    whose result has no mask geometry."""
    items = []
    for filename, raw_abs, r in results_by_file:
        h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
        dets = []
        xyn = getattr(getattr(r, "masks", None), "xyn", None)
        if xyn is not None:
            for j, poly in enumerate(xyn):
                flat = [round(float(v), 6) for pt in poly for v in pt]
                if len(flat) < 6:
                    continue
                cid = int(float(r.boxes.cls[j]))
                dets.append({
                    "class_id": cid,
                    "class": str(class_names.get(cid, cid)),
                    "confidence": round(float(r.boxes.conf[j]), 4),
                    "polygon_xyn": flat,
                })
        items.append({"filename": filename, "raw": raw_abs,
                      "width": w, "height": h, "detections": dets})
    return items


def main():
    args = arg_parse()
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, 'overlays')
    crop_dir = os.path.join(args.out_dir, 'crops')
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)
    if args.save_crops:
        os.makedirs(crop_dir, exist_ok=True)

    weights = _find_best_weights(args.run_dir)
    print(f'[infer] weights={weights}')
    print(f'[infer] source_type={args.source_type} imgsz={args.imgsz} conf={args.conf} iou={args.iou}')

    images = collect_source(args)
    print(f'[infer] resolved {len(images)} images')
    if not images:
        print('[infer] nothing to do, exiting')
        with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
            json.dump({'items': [], 'total_predictions': 0, 'with_any': 0,
                       'generated_at': datetime.now().isoformat()}, f, indent=2)
        return

    from ultralytics import YOLO
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    import colorsys

    model = YOLO(weights)
    names = getattr(model, 'names', None) or {}

    # Stable per-class color palette (same hue assignment for a given class id
    # across every image — makes the gallery easy to scan).
    def _class_color(cls_id):
        h = ((int(cls_id) * 137) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.78, 0.95)
        return int(r * 255), int(g * 255), int(b * 255)

    def _render_overlay(img_path, result, *, alpha, draw_boxes,
                        run_name, project_name, timestamp):
        """Build a custom overlay:
          - filled mask polygons with per-class color at `alpha`
          - mask outlines in the same color, solid
          - optional bounding boxes (default off — user complained 'can't see')
          - small pink footer strip with project / run / timestamp / filename
        """
        base = Image.open(img_path).convert('RGB')
        W, H = base.size
        overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        draw_o = ImageDraw.Draw(overlay)

        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)
        n = int(len(boxes)) if boxes is not None else 0

        # Try to get polygons in image coords — masks.xy is a list of np arrays.
        polys = None
        if masks is not None:
            try:
                polys = list(masks.xy)
            except Exception:
                polys = None

        # Font for labels + footer
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
            small = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
        except Exception:
            font = ImageFont.load_default()
            small = ImageFont.load_default()

        fill_a = int(max(0.0, min(1.0, alpha)) * 255)

        if n:
            cls_arr = boxes.cls.cpu().numpy().astype(int) if boxes is not None else [0] * n
            conf_arr = boxes.conf.cpu().numpy() if boxes is not None else [0.0] * n
            xyxy_arr = boxes.xyxy.cpu().numpy() if boxes is not None else None

            for i in range(n):
                color = _class_color(cls_arr[i])
                # 1) Filled mask polygon
                if polys and i < len(polys) and len(polys[i]) >= 3:
                    pts = [tuple(p) for p in polys[i]]
                    draw_o.polygon(pts, fill=(*color, fill_a), outline=(*color, 255))
                    # Class label near first point
                    nm = names.get(int(cls_arr[i]), f'cls{cls_arr[i]}')
                    lbl = f'{nm} {conf_arr[i]:.2f}'
                    tx, ty = int(pts[0][0]), max(0, int(pts[0][1]) - 22)
                    # Label chip
                    bbox = draw_o.textbbox((tx, ty), lbl, font=font)
                    draw_o.rectangle(bbox, fill=(0, 0, 0, 220))
                    draw_o.text((tx, ty), lbl, fill=(*color, 255), font=font)
                elif xyxy_arr is not None:
                    # No polygon -> fall back to box outline so we don't lose the instance
                    x1, y1, x2, y2 = xyxy_arr[i]
                    draw_o.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)

                # 2) Optional bounding boxes on top of masks
                if draw_boxes and xyxy_arr is not None:
                    x1, y1, x2, y2 = xyxy_arr[i]
                    draw_o.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=2)

        # Composite the RGBA overlay onto the base
        out = Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')

        # Footer strip with metadata — pink on semi-transparent black bar
        footer_h = 22
        footer = Image.new('RGBA', (W, footer_h), (0, 0, 0, 180))
        draw_f = ImageDraw.Draw(footer)
        parts = []
        if project_name:
            parts.append(project_name)
        if run_name:
            parts.append(run_name)
        if timestamp:
            parts.append(timestamp)
        parts.append(os.path.basename(img_path))
        footer_text = ' · '.join(parts)
        # Truncate if too wide
        max_chars = max(20, int(W / 7))
        if len(footer_text) > max_chars:
            footer_text = '… ' + footer_text[-(max_chars - 2):]
        draw_f.text((8, 3), footer_text, fill=(255, 61, 139, 255), font=small)

        out_rgba = out.convert('RGBA')
        # Paste footer at the bottom
        footer_pos = (0, H - footer_h)
        out_rgba.paste(footer, footer_pos, footer)
        return out_rgba.convert('RGB')

    items = []
    total_preds = 0
    with_any = 0
    confidences_by_class = {}
    unknown_class_total = 0
    results_by_file = []
    t0 = time.time()

    for idx, src in enumerate(images):
        try:
            results = model.predict(src, conf=args.conf, iou=args.iou,
                                    imgsz=args.imgsz, device=args.device,
                                    save=False, verbose=False)
            if not results:
                continue
            r = results[0]
            n_det = int(len(r.boxes)) if r.boxes is not None else 0
            total_preds += n_det
            if n_det > 0:
                with_any += 1

            if args.save_predictions:
                results_by_file.append((os.path.basename(src), src, r))

            rel_overlay = ''
            if args.save_overlays:
                img = _render_overlay(
                    src, r,
                    alpha=args.mask_alpha,
                    draw_boxes=args.draw_boxes,
                    run_name=args.run_name,
                    project_name=args.project_name,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M'),
                )
                rel_overlay = os.path.join('overlays', os.path.basename(src))
                dst = os.path.join(args.out_dir, rel_overlay)
                img.save(dst, quality=85)

            # Per-detection class + confidence stats
            max_conf = 0.0
            det_rows = []
            if r.boxes is not None and len(r.boxes):
                cls_arr = r.boxes.cls.cpu().numpy().astype(int)
                conf_arr = r.boxes.conf.cpu().numpy()
                for c, cf in zip(cls_arr, conf_arr):
                    nm = names.get(int(c), f'cls{c}')
                    confidences_by_class.setdefault(nm, []).append(float(cf))
                    det_rows.append({'class': nm, 'confidence': float(cf)})
                    if cf > max_conf:
                        max_conf = float(cf)

            items.append({
                'filename': os.path.basename(src),
                'raw': os.path.relpath(src, args.out_dir) if src.startswith(args.out_dir) else src,
                'overlay': rel_overlay or '',
                'n_detections': n_det,
                'max_conf': max_conf,
                'detections': det_rows,
            })

            if (idx + 1) % 10 == 0 or idx == len(images) - 1:
                print(f'[infer] {idx + 1}/{len(images)} · {with_any} with det · '
                      f'avg {total_preds / max(1, idx + 1):.1f} det/img')
        except Exception as e:
            print(f'[infer] ERROR on {src}: {e}')

    wall = time.time() - t0

    # Blind-spot signal: low-confidence classes + classes with unusually few hits.
    blind_bits = []
    for nm, confs in confidences_by_class.items():
        if not confs:
            continue
        avg = sum(confs) / len(confs)
        low = sum(1 for c in confs if c < 0.35) / len(confs)
        if avg < 0.4:
            blind_bits.append(f'<code>{nm}</code>: avg confidence {avg:.2f} (low — model is unsure)')
        elif low > 0.5:
            blind_bits.append(f'<code>{nm}</code>: {low*100:.0f}% of predictions below conf 0.35 — noisy')
    images_no_det = len(items) - with_any
    if images_no_det and len(items):
        frac = images_no_det / len(items)
        if frac > 0.2:
            blind_bits.append(
                f'{images_no_det}/{len(items)} images got NO detection '
                f'({frac*100:.0f}%) — possible blind spot or out-of-distribution data.')

    manifest = {
        'items': items,
        'total_predictions': total_preds,
        'with_any': with_any,
        'class_names': names,
        'confidences_by_class': {
            k: {'count': len(v), 'avg': sum(v) / len(v), 'min': min(v), 'max': max(v)}
            for k, v in confidences_by_class.items()
        },
        'wall_seconds': wall,
        'blind_spots': ' · '.join(blind_bits) if blind_bits else '',
        'args': vars(args),
        'generated_at': datetime.now().isoformat(),
    }

    if args.save_predictions:
        pred_items = build_prediction_items(results_by_file, names)
        skipped_maskless = sum(
            1 for (_, _, r) in results_by_file
            if getattr(getattr(r, 'masks', None), 'xyn', None) is None
            and (r.boxes is not None and len(r.boxes) > 0)
        )
        predictions = {
            'schema_version': 1,
            'generated_at': ast_now(),
            'run_dir': args.run_dir,
            'imgsz': args.imgsz,
            'conf': args.conf,
            'iou': args.iou,
            'class_names': {str(k): v for k, v in names.items()},
            'items': pred_items,
        }
        with open(os.path.join(args.out_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        manifest['predictions'] = 'predictions.json'
        print(f'[infer] wrote predictions.json ({len(pred_items)} items, '
              f'{skipped_maskless} detections skipped for missing mask geometry)')

    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f'[infer] done. {len(items)} images, {total_preds} predictions, {with_any} with at least 1 detection.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
