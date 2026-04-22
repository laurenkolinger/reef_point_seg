#!/usr/bin/env python3
"""Evaluate a trained Ultralytics segmentation run and produce a
dense, cheatsheet-style evaluation PDF (plus JSON + Markdown).

Invoked by the orchestrator's step 7. Consumes:
    --run_dir    path to a directory containing weights/best.pt (typically
                 {step6_dir}/runs/<run_name>/)
    --dataset_dir   the split dataset dir from step 6 ({step6_dir}/dataset)
    --out_dir    where to write evaluation outputs (typically {step7_dir})
    --split      test | val | train
    --imgsz / --batch / --conf / --iou / --preview_count
    --pdf_export_dir  optional extra dir to copy the PDF to

Emits (inside --out_dir):
    metrics.json          machine-readable numbers
    report.md             markdown version of the report
    report.pdf            a multi-page PDF cheatsheet (fixed per-page layout)
    previews/             rendered overlay images used in the PDF
    eval_log.txt          stdout tee of ultralytics val

PDF page layout is STABLE across runs — page N always contains the same
section, so multiple PDFs can be tabbed through side-by-side and compared
page-for-page.

    Page 1   Cover sheet + top-line cheatsheet + TOC
    Page 2   Training history (hyperparams + loss curves)
    Page 3   Box metrics cheatsheet
    Page 4   Mask metrics cheatsheet
    Page 5   Per-class breakdown (+ horizontal bar chart)
    Page 6   Confusion matrix (normalized + absolute)
    Page 7+  Prediction samples (2 per page, GT vs pred side-by-side)
    Last     Glossary cheatsheet
"""

import argparse
import csv as _csv
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def arg_parse():
    p = argparse.ArgumentParser(description='TCRMP step 7 evaluate + report')
    p.add_argument('--run_dir', required=True, help='dir with weights/best.pt')
    p.add_argument('--dataset_dir', required=True, help='step 6 dataset dir (has data.yaml / test.yaml)')
    p.add_argument('--out_dir', required=True, help='destination for metrics/report/pdf/previews')
    p.add_argument('--split', default='test', choices=['test', 'val', 'train'])
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=-1)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.6)
    p.add_argument('--preview_count', type=int, default=8)
    p.add_argument('--pdf_export_dir', default='')
    p.add_argument('--device', default='0')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────
# Ultralytics val
# ─────────────────────────────────────────────────────────────────────────

def _find_best_weights(run_dir):
    cand = [os.path.join(run_dir, 'weights', 'best.pt'),
            os.path.join(run_dir, 'weights', 'last.pt')]
    for c in cand:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f'No weights/best.pt or weights/last.pt in {run_dir}')


def _resolve_data_yaml(dataset_dir, split):
    cand = [os.path.join(dataset_dir, 'data.yaml')]
    for c in cand:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f'No data.yaml in {dataset_dir}')


def run_val(args):
    from ultralytics import YOLO
    weights = _find_best_weights(args.run_dir)
    data_yaml = _resolve_data_yaml(args.dataset_dir, args.split)
    model = YOLO(weights)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    val_dir = os.path.join(out_dir, 'val_run')
    os.makedirs(val_dir, exist_ok=True)

    batch = args.batch if args.batch and args.batch > 0 else 8

    t0 = time.time()
    results = model.val(
        data=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=out_dir,
        name='val_run',
        exist_ok=True,
        plots=True,
        save_json=True,
        save_txt=False,
        save_conf=True,
    )
    wall = time.time() - t0

    return {
        'weights': weights,
        'data_yaml': data_yaml,
        'wall_seconds': wall,
        'results': results,
        'val_dir': val_dir,
        'model': model,
    }


# ─────────────────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_metrics(val_out):
    """Turn ultralytics val output into a plain dict we can render / serialize."""
    r = val_out['results']
    names = None
    try:
        names = r.names
    except Exception:
        pass

    def _rd(box_obj):
        out = {}
        for k in ('p', 'r', 'map50', 'map75', 'map', 'f1'):
            try:
                v = getattr(box_obj, 'mean_' + k, None)
                if v is None:
                    v = getattr(box_obj, k, None)
                    if hasattr(v, 'mean'):
                        v = float(v.mean())
                out[k] = float(v) if v is not None else None
            except Exception:
                out[k] = None
        return out

    box = getattr(r, 'box', None)
    seg = getattr(r, 'seg', None)

    per_class = {}
    per_class_instances = {}
    try:
        if box is not None and names:
            for i, nm in names.items():
                try:
                    p, rr, ap50, ap = box.class_result(i)
                    per_class.setdefault(nm, {}).update({
                        'box_precision': float(p),
                        'box_recall': float(rr),
                        'box_map50': float(ap50),
                        'box_map50_95': float(ap),
                    })
                except Exception:
                    pass
        if seg is not None and names:
            for i, nm in names.items():
                try:
                    p, rr, ap50, ap = seg.class_result(i)
                    per_class.setdefault(nm, {}).update({
                        'mask_precision': float(p),
                        'mask_recall': float(rr),
                        'mask_map50': float(ap50),
                        'mask_map50_95': float(ap),
                    })
                except Exception:
                    pass
    except Exception as e:
        print(f'[eval] per-class extraction partial: {e}')

    # Try to pull per-class instance counts from the confusion matrix if present
    try:
        cm = getattr(r, 'confusion_matrix', None)
        if cm is not None and names:
            mat = getattr(cm, 'matrix', None)
            if mat is not None:
                import numpy as _np
                arr = _np.asarray(mat)
                # Ultralytics confusion matrix: columns = predicted, rows = true;
                # with a trailing "background" row/col.
                n = len(names)
                if arr.shape[0] >= n:
                    for i, nm in names.items():
                        try:
                            per_class_instances[nm] = int(arr[:n, i].sum())
                        except Exception:
                            pass
    except Exception:
        pass

    metrics = {
        'wall_seconds': val_out['wall_seconds'],
        'weights': val_out['weights'],
        'data_yaml': val_out['data_yaml'],
        'class_names': names,
        'overall': {
            'box_map50': _rd(box).get('map50') if box else None,
            'box_map50_95': _rd(box).get('map') if box else None,
            'box_precision': _rd(box).get('p') if box else None,
            'box_recall': _rd(box).get('r') if box else None,
            'mask_map50': _rd(seg).get('map50') if seg else None,
            'mask_map50_95': _rd(seg).get('map') if seg else None,
            'mask_precision': _rd(seg).get('p') if seg else None,
            'mask_recall': _rd(seg).get('r') if seg else None,
        },
        'per_class': per_class,
        'per_class_instances': per_class_instances,
    }
    try:
        metrics['fitness'] = float(r.fitness)
    except Exception:
        pass
    try:
        metrics['speed_ms'] = {k: float(v) for k, v in r.speed.items()}
    except Exception:
        pass
    return metrics


# ─────────────────────────────────────────────────────────────────────────
# Dataset / training-config context
# ─────────────────────────────────────────────────────────────────────────

def gather_context(run_dir, dataset_dir):
    ctx = {'run_dir': run_dir, 'dataset_dir': dataset_dir}

    args_yaml = os.path.join(run_dir, 'args.yaml')
    if os.path.isfile(args_yaml):
        try:
            import yaml
            with open(args_yaml) as f:
                ctx['train_args'] = yaml.safe_load(f)
        except Exception:
            pass

    res_csv = os.path.join(run_dir, 'results.csv')
    if os.path.isfile(res_csv):
        try:
            with open(res_csv) as f:
                rows = list(_csv.reader(f))
            ctx['epochs_completed'] = max(0, len(rows) - 1)
            ctx['results_csv'] = res_csv
        except Exception:
            pass

    for split in ('train', 'valid', 'test'):
        p = os.path.join(dataset_dir, split, 'images')
        if os.path.isdir(p):
            try:
                ctx.setdefault('split_counts', {})[split] = sum(
                    1 for n in os.listdir(p)
                    if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
                )
            except Exception:
                pass

    # best epoch from results.csv (highest fitness proxy: highest mAP50-95(M))
    ctx['best_epoch'] = _find_best_epoch(os.path.join(run_dir, 'results.csv'))
    return ctx


def _find_best_epoch(results_csv):
    """Scan results.csv for the epoch with highest mAP50-95(M) (or (B) fallback)."""
    if not os.path.isfile(results_csv):
        return None
    try:
        with open(results_csv) as f:
            reader = _csv.reader(f)
            hdr = next(reader)
            hdr = [h.strip() for h in hdr]
            col = None
            for cand in ('metrics/mAP50-95(M)', 'metrics/mAP50-95(B)', 'metrics/mAP50(M)', 'metrics/mAP50(B)'):
                if cand in hdr:
                    col = hdr.index(cand)
                    break
            if col is None:
                return None
            best_val = -1.0
            best_ep = None
            ep_col = hdr.index('epoch') if 'epoch' in hdr else 0
            for row in reader:
                try:
                    v = float(row[col])
                    ep = int(float(row[ep_col]))
                    if v > best_val:
                        best_val = v
                        best_ep = ep
                except Exception:
                    continue
            return best_ep
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────
# Preview renderings — GT vs predicted, side-by-side
# ─────────────────────────────────────────────────────────────────────────

def _parse_yolo_seg_labels(label_path, img_w, img_h):
    """Parse a YOLOv8 segmentation .txt -> list of (class_id, polygon_np_pixel_coords)."""
    import numpy as np
    out = []
    if not os.path.isfile(label_path):
        return out
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cls = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                # may be polygon (pairs) OR box-style; for seg we expect polygons
                if len(coords) % 2 != 0:
                    continue
                poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
                poly[:, 0] *= img_w
                poly[:, 1] *= img_h
                out.append((cls, poly))
    except Exception:
        return out
    return out


def _render_side_by_side_preview(image_path, label_path, model, conf, iou, imgsz, out_path, names):
    """Composite a single PNG: left = image with GT polygon overlay; right = image
    with predicted mask overlay. Returns dict with metadata for caption."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
    except Exception as e:
        print(f'[eval] PIL import failed: {e}')
        return None

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f'[eval] failed to open {image_path}: {e}')
        return None
    w, h = img.size

    # GT side
    gt_img = img.copy()
    gt_draw = ImageDraw.Draw(gt_img, 'RGBA')
    gts = _parse_yolo_seg_labels(label_path, w, h)
    for cls, poly in gts:
        pts = [tuple(p) for p in poly.tolist()]
        if len(pts) >= 3:
            gt_draw.polygon(pts, fill=(226, 0, 116, 90), outline=(226, 0, 116, 255))

    # Pred side
    pr_img = img.copy()
    n_pred = 0
    try:
        results = model.predict(image_path, conf=conf, iou=iou, imgsz=imgsz,
                                save=False, verbose=False)
        if results:
            r = results[0]
            arr = r.plot()  # BGR numpy with model overlays
            pr_img = Image.fromarray(arr[..., ::-1])
            n_pred = int(len(r.masks)) if r.masks is not None else int(len(r.boxes)) if r.boxes is not None else 0
    except Exception as e:
        print(f'[eval] predict failed for {image_path}: {e}')

    # Resize to same height, place side by side
    target_h = 720
    scale_g = target_h / h
    gt_r = gt_img.resize((int(w * scale_g), target_h))
    pr_r = pr_img.resize((int(pr_img.size[0] * (target_h / pr_img.size[1])), target_h))

    label_band = 36
    composite_w = gt_r.size[0] + pr_r.size[0] + 10
    composite_h = target_h + label_band
    composite = Image.new('RGB', (composite_w, composite_h), 'white')
    composite.paste(gt_r, (0, label_band))
    composite.paste(pr_r, (gt_r.size[0] + 10, label_band))

    # Draw captions
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, gt_r.size[0], label_band], fill=(226, 0, 116))
    draw.text((8, 6),
              f'Ground truth: {len(gts)} human-drawn instance(s)',
              fill='white', font=font)
    draw.rectangle([gt_r.size[0] + 10, 0, composite_w, label_band], fill=(40, 40, 40))
    draw.text((gt_r.size[0] + 18, 6),
              f'Model prediction at confidence >= {conf:.2f}: {n_pred} detection(s)',
              fill='white', font=font)

    composite.save(out_path, quality=88)
    return {
        'source': image_path,
        'overlay': out_path,
        'filename': os.path.basename(image_path),
        'n_gt': len(gts),
        'n_pred': n_pred,
    }


def render_previews(model, dataset_dir, split, out_dir, count, conf, iou, imgsz):
    preview_dir = os.path.join(out_dir, 'previews')
    os.makedirs(preview_dir, exist_ok=True)

    split_dir = 'test' if split == 'test' else ('valid' if split == 'val' else 'train')
    img_dir = os.path.join(dataset_dir, split_dir, 'images')
    lbl_dir = os.path.join(dataset_dir, split_dir, 'labels')
    if not os.path.isdir(img_dir):
        return []

    candidates = sorted([n for n in os.listdir(img_dir)
                         if n.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not candidates:
        return []

    random.seed(0)
    sample = candidates if len(candidates) <= count else random.sample(candidates, count)
    sample.sort()

    names = None
    try:
        names = model.names
    except Exception:
        pass

    saved = []
    for fn in sample:
        src = os.path.join(img_dir, fn)
        stem = os.path.splitext(fn)[0]
        lbl = os.path.join(lbl_dir, stem + '.txt')
        dst = os.path.join(preview_dir, stem + '_sbs.jpg')
        try:
            info = _render_side_by_side_preview(src, lbl, model, conf, iou, imgsz, dst, names)
            if info:
                saved.append(info)
        except Exception as e:
            print(f'[eval] preview failed for {fn}: {e}')
    return saved


# ─────────────────────────────────────────────────────────────────────────
# Interpretation bands — concrete "good range" for inline annotations
# ─────────────────────────────────────────────────────────────────────────

def _band_map50_95(v):
    if v is None: return ('—', '#888')
    if v < 0.2:  return ('not learning', '#b00020')
    if v < 0.4:  return ('weak', '#c06000')
    if v < 0.6:  return ('good', '#2a6a2a')
    if v < 0.8:  return ('very good', '#1f7a1f')
    return ('excellent', '#0d5a0d')


def _band_map50(v):
    if v is None: return ('—', '#888')
    if v < 0.35: return ('not learning', '#b00020')
    if v < 0.55: return ('weak', '#c06000')
    if v < 0.75: return ('good', '#2a6a2a')
    if v < 0.95: return ('very good', '#1f7a1f')
    return ('excellent', '#0d5a0d')


def _band_pr(v):
    if v is None: return ('—', '#888')
    if v < 0.5:  return ('poor', '#b00020')
    if v < 0.7:  return ('OK', '#c06000')
    if v < 0.85: return ('good', '#2a6a2a')
    return ('excellent', '#0d5a0d')


GOOD_RANGES = {
    'mAP50-95': '<0.2 poor | 0.2-0.4 weak | 0.4-0.6 good | 0.6-0.8 very good | >0.8 excellent',
    'mAP50':    '<0.35 poor | 0.35-0.55 weak | 0.55-0.75 good | 0.75-0.95 very good | >0.95 excellent',
    'P/R':      '<0.5 poor | 0.5-0.7 OK | 0.7-0.85 good | >0.85 excellent',
    'F1':       '<0.5 poor | 0.5-0.7 OK | 0.7-0.85 good | >0.85 excellent',
}


# ─────────────────────────────────────────────────────────────────────────
# Plain-language prose constants
#
# These constants hold every user-visible string that the PDF renders.
# Rules for any edits here:
#   - Every string must be a complete sentence with a subject and verb.
#   - Abbreviations are spelled out on first use within each constant
#     (the PDF pages are read independently, so the reader may only see
#     one of these strings on a given page).
#   - No shorthand: write "is greater than" rather than ">>", write
#     "meaning" rather than "=>", and avoid telegraphic fragments.
# ─────────────────────────────────────────────────────────────────────────

METRIC_EXPLANATIONS = {
    'precision': (
        "Precision (P) is the fraction of the model's predictions that turned "
        "out to be real corals. A precision of 0.80 means four out of every "
        "five predicted corals are genuine and the fifth is a false alarm."
    ),
    'recall': (
        "Recall (R) is the fraction of the real corals in the image that the "
        "model actually found. A recall of 0.80 means the model located four "
        "out of every five corals a human labeler drew."
    ),
    'f1': (
        "The F1 score is the balanced average of precision and recall, so a "
        "model only gets a high F1 if it is both accurate in what it predicts "
        "and thorough in finding everything that is there."
    ),
    'map50': (
        "Mean average precision at an intersection-over-union (IoU) threshold "
        "of 0.5 (mAP50) is the lenient detection score: a prediction counts as "
        "correct if it overlaps the human-drawn region by at least 50 percent."
    ),
    'map50_95': (
        "Mean average precision averaged across IoU thresholds from 0.5 to "
        "0.95 (mAP50-95) is the strict detection score: a prediction only earns "
        "full credit if it overlaps the human-drawn region almost perfectly."
    ),
}


FIGURE_CAPTIONS = {
    'loss_history': (
        "Training and validation loss across every epoch. Both curves should "
        "fall together and then flatten; if the validation curve turns upward "
        "while the training curve keeps dropping, the model is memorizing the "
        "training images rather than learning general features."
    ),
    'pr_curve_box': (
        "Precision-recall curve for bounding boxes. The area under this curve "
        "at IoU 0.5 equals the box mAP50 score reported above. A curve that "
        "stays high across most of the horizontal axis means the model remains "
        "accurate even as you ask it to find more corals."
    ),
    'pr_curve_mask': (
        "Precision-recall curve for segmentation masks. The area under this "
        "curve at IoU 0.5 equals the mask mAP50 score. A curve that hugs the "
        "top-right corner means the predicted coral outlines match the human "
        "labels closely across a wide range of confidence cutoffs."
    ),
    'f1_curve_box': (
        "F1 score for bounding boxes as a function of the confidence "
        "threshold. The peak of this curve tells you the confidence cutoff at "
        "which the model is best balanced between false alarms and missed "
        "corals, which is the cutoff you should use when deploying the model."
    ),
    'f1_curve_mask': (
        "F1 score for segmentation masks as a function of the confidence "
        "threshold. The peak tells you the cutoff at which mask-based "
        "precision and recall are best balanced; use this cutoff in "
        "production if you rely on the segmented outlines rather than the "
        "bounding rectangles."
    ),
    'per_class_bar': (
        "Horizontal bar chart of the strict segmentation score (mAP50-95) "
        "for each coral class, sorted from best to worst. Short bars at the "
        "bottom highlight which species the model struggles with most and "
        "which labels or training examples may need attention."
    ),
    'confusion_norm': (
        "Normalized confusion matrix: each row is one true (human-labeled) "
        "class and sums to 1.0, so each cell shows the fraction of that true "
        "class that the model predicted as the column class. The diagonal is "
        "correct predictions; bright off-diagonal cells are species the model "
        "confuses with one another."
    ),
    'confusion_abs': (
        "Absolute confusion matrix with raw counts of predictions. Rows are "
        "the human-labeled (ground-truth) class and columns are the predicted "
        "class. Small diagonal numbers indicate classes with few test "
        "examples, whose scores carry high uncertainty."
    ),
    'sample_pair': (
        "Left panel shows the image with the human-drawn ground-truth (GT) "
        "outlines in pink. Right panel shows the same image with the model's "
        "predicted outlines drawn by the trained network. Compare the shape "
        "and placement of the pink regions against the prediction to spot "
        "missed corals, false alarms, and sloppy boundaries."
    ),
}


SECTION_INTROS = {
    'cover': (
        "This report evaluates one trained coral segmentation model on a held-"
        "out split of labeled images. Every page below covers one topic in the "
        "same order across runs, so two reports can be compared page-by-page. "
        "Abbreviations are defined the first time they appear on each page."
    ),
    'training_history': (
        "This page shows the hyperparameters used for training and the loss "
        "curves recorded at every epoch. The loss is a number that the "
        "optimizer tries to push down; smaller is better. Four losses are "
        "tracked: box (how well bounding rectangles fit), seg (how well "
        "masks fit), cls (how well the species label is predicted), and dfl "
        "(a helper loss that sharpens box edges)."
    ),
    'box_metrics': (
        "This page summarizes the model's performance when a prediction is "
        "judged by the overlap of its bounding rectangle with the human-drawn "
        "(ground-truth, or GT) rectangle. This is the lenient view: the model "
        "only has to place a box in roughly the right spot, not trace the "
        "coral's outline pixel-by-pixel."
    ),
    'mask_metrics': (
        "This page summarizes the model's performance when a prediction is "
        "judged by the overlap of its segmentation mask with the human-drawn "
        "(ground-truth, or GT) mask. This is the strict view: the model must "
        "trace the coral's outline closely, not just place a rectangle near it."
    ),
    'per_class': (
        "This page breaks the overall scores down by coral species so you can "
        "see which classes the model handles well and which it struggles with. "
        "All numbers below use the mask metrics (outline-based), and the "
        "'Inst' column counts how many instances of that class are present in "
        "the evaluation split."
    ),
    'confusion': (
        "A confusion matrix cross-tabulates every prediction against the "
        "human-drawn (ground-truth, GT) label. Rows are the true class. "
        "Columns are the class the model predicted. Cells on the diagonal "
        "are correct; cells off the diagonal are mistakes, and the exact "
        "off-diagonal cell tells you which species the model confused with "
        "which."
    ),
    'samples': (
        "Each row below shows one evaluation image twice. On the left, the "
        "human-drawn ground-truth (GT) outlines are overlaid in pink. On the "
        "right, the same image is overlaid with the model's predicted "
        "outlines. Compare the two panels to see where the model agrees with "
        "the human labels and where it does not."
    ),
    'glossary': (
        "Every term and abbreviation used in this report is defined below. "
        "The 'Good' column gives a rough band for what a healthy value looks "
        "like on this kind of data; the 'Red flag' column names the symptom "
        "you see when that metric is too low."
    ),
}


AXIS_EXPLANATIONS = {
    'pr_curve_box': {
        'what': (
            "Precision-recall curve for bounding-box predictions."
        ),
        'x_axis': (
            "Recall (R) on the horizontal axis: the fraction of all real "
            "corals the model managed to find."
        ),
        'y_axis': (
            "Precision (P) on the vertical axis: the fraction of the model's "
            "predictions that were correct."
        ),
        'read': (
            "The top-right corner is ideal because it means both precision "
            "and recall are high. The curve always slopes downward: asking "
            "the model to find more corals forces it to accept some false "
            "alarms, so precision falls as recall rises."
        ),
        'connects': (
            "The area under this curve at intersection over union (IoU) 0.5 "
            "is the same number reported above as mAP50 for boxes. At "
            "stricter IoU thresholds the curve drops if the model's boxes "
            "overlap the true region only loosely."
        ),
    },
    'pr_curve_mask': {
        'what': (
            "Precision-recall curve for segmentation masks (pixel-level "
            "outlines)."
        ),
        'x_axis': (
            "Recall (R) on the horizontal axis: the fraction of real corals "
            "whose outlines the model successfully traced."
        ),
        'y_axis': (
            "Precision (P) on the vertical axis: the fraction of predicted "
            "outlines that correctly match a real coral."
        ),
        'read': (
            "A curve that stays close to the top-right corner means the "
            "model draws accurate outlines for a large share of corals. "
            "Curves that drop off steeply on the right mean the model gives "
            "up accuracy quickly when asked to cover more of the image."
        ),
        'connects': (
            "The area under this curve at intersection over union (IoU) 0.5 "
            "is the mask mAP50 reported above. When this curve sits above "
            "the box curve on the same page, the predicted outlines are "
            "tighter than the predicted rectangles."
        ),
    },
    'f1_curve_box': {
        'what': (
            "Box F1 score (the balanced average of precision and recall) as "
            "a function of the confidence threshold applied at deployment."
        ),
        'x_axis': (
            "Confidence threshold on the horizontal axis: predictions with a "
            "score below this value are discarded."
        ),
        'y_axis': (
            "F1 score on the vertical axis: higher means the model is better "
            "balanced between finding real corals and avoiding false alarms."
        ),
        'read': (
            "The peak of the curve identifies the confidence threshold at "
            "which the model performs best overall. Use that threshold when "
            "running the model on new images."
        ),
        'connects': (
            "The height of the peak relates directly to the mean precision "
            "and recall on the box metrics table above; a sharp peak means "
            "small changes in the threshold matter a lot."
        ),
    },
    'f1_curve_mask': {
        'what': (
            "Mask F1 score as a function of the confidence threshold "
            "applied at deployment."
        ),
        'x_axis': (
            "Confidence threshold on the horizontal axis: predicted masks "
            "with a score below this value are discarded."
        ),
        'y_axis': (
            "F1 score on the vertical axis: higher means the predicted "
            "outlines balance correctness and completeness well."
        ),
        'read': (
            "Find the peak and read off the matching confidence value. That "
            "is the threshold at which mask-based precision and recall are "
            "best balanced."
        ),
        'connects': (
            "The peak F1 usually sits slightly below the box F1 peak "
            "because outlines are held to a stricter overlap standard than "
            "rectangles."
        ),
    },
    'loss_history': {
        'what': (
            "Four training-loss panels plotted over all training epochs: "
            "box, segmentation, classification, and distribution focal loss "
            "(DFL, a helper loss that sharpens box edges)."
        ),
        'x_axis': (
            "Epoch number on the horizontal axis: one epoch is one full "
            "pass through the training images."
        ),
        'y_axis': (
            "Loss value on the vertical axis: an internal error score the "
            "optimizer tries to minimize. Lower is better."
        ),
        'read': (
            "Both the training curve (solid pink) and the validation curve "
            "(dashed black) should fall and then plateau together. If the "
            "validation curve starts climbing while the training curve "
            "keeps falling, the model has begun memorizing the training "
            "set rather than learning general coral features."
        ),
        'connects': (
            "A smooth plateau at the end of training is what lets the "
            "checkpoint-selector (fitness) lock in a stable best-epoch "
            "weight file, which is what the rest of this report evaluates."
        ),
    },
    'per_class_bar': {
        'what': (
            "Strict segmentation score (mAP50-95 on masks) for each coral "
            "class, sorted from strongest to weakest."
        ),
        'x_axis': (
            "mAP50-95 value on the horizontal axis: 0 is no overlap with "
            "the human labels, 1 is a perfect pixel-level match."
        ),
        'y_axis': (
            "One row per coral class on the vertical axis; longer bars are "
            "better."
        ),
        'read': (
            "The shortest bars are the classes the model struggles with "
            "most. Compare them against the 'Inst' (instance count) column "
            "in the table above: classes with very few examples in the "
            "evaluation split often look weak simply because there is "
            "little to measure."
        ),
        'connects': (
            "A weak class here usually shows up on the confusion-matrix "
            "page (p.6) as a row that spills off the diagonal into one or "
            "more similar-looking species."
        ),
    },
    'confusion_norm': {
        'what': (
            "Row-normalized confusion matrix: each row is one true "
            "(human-labeled) class, and the cells in that row sum to 1.0."
        ),
        'x_axis': (
            "Predicted class on the horizontal axis: what the model "
            "decided the coral was."
        ),
        'y_axis': (
            "True class on the vertical axis: what the human labeler "
            "actually drew."
        ),
        'read': (
            "Each cell shows the fraction of that true class that the "
            "model predicted as the column class. The diagonal is "
            "correct predictions; the brighter a cell off the diagonal, "
            "the more often those two classes get confused. Because each "
            "row sums to 1.0, rare classes and common classes are "
            "directly comparable."
        ),
        'connects': (
            "Whichever row has the dimmest diagonal is the class with the "
            "lowest recall in the per-class table; the bright cell in "
            "that row names the species it gets mistaken for."
        ),
    },
    'confusion_abs': {
        'what': (
            "Absolute confusion matrix: raw counts rather than fractions."
        ),
        'x_axis': (
            "Predicted class on the horizontal axis."
        ),
        'y_axis': (
            "True (human-labeled, ground-truth) class on the vertical axis."
        ),
        'read': (
            "Large numbers on the diagonal mean the model got that class "
            "right many times; small diagonals mean few test examples "
            "exist for that class, so its scores are noisy and should not "
            "be over-interpreted."
        ),
        'connects': (
            "The row sums here equal the instance counts reported in the "
            "per-class table. Compare these totals against training "
            "instance counts to see whether underperforming classes were "
            "simply underrepresented during training."
        ),
    },
}


# AXIS_EXPLANATIONS_FULL — the versions rendered directly beneath each plot.
# Every string below is a complete sentence (no em-dash fragments) so that a
# reader who looks only at the plot and its caption understands the axes.
AXIS_EXPLANATIONS_FULL = {
    'pr_curve_box': {
        'what': (
            "This is the precision-recall curve computed on the model's "
            "bounding-box predictions against the human-drawn ground-truth "
            "boxes on the evaluation split."
        ),
        'x_axis': (
            "The horizontal axis is recall, the fraction of all true "
            "corals in the dataset that the model successfully found. A "
            "recall of 1.0 means every coral was detected; a recall of "
            "0.0 means none were."
        ),
        'y_axis': (
            "The vertical axis is precision, the fraction of the model's "
            "bounding-box predictions that were actually correct. A "
            "precision of 1.0 means the model never raised a false "
            "alarm; 0.0 means every prediction was wrong."
        ),
        'read': (
            "The top-right corner (precision 1.0, recall 1.0) is ideal. "
            "The curve almost always slopes downward because asking the "
            "model to find more corals forces it to accept predictions "
            "it is less sure about. The area under this curve at an "
            "intersection-over-union (IoU) of 0.5 is the box mAP50 value "
            "reported in the summary table above."
        ),
    },
    'pr_curve_mask': {
        'what': (
            "This is the precision-recall curve computed on the model's "
            "pixel-level segmentation masks (outlines) against the "
            "human-drawn ground-truth outlines on the evaluation split."
        ),
        'x_axis': (
            "The horizontal axis is recall, the fraction of all real "
            "coral outlines in the dataset that the model successfully "
            "traced. A recall of 1.0 means every coral outline was "
            "recovered; 0.0 means none were."
        ),
        'y_axis': (
            "The vertical axis is precision, the fraction of the model's "
            "predicted outlines that correctly match a real coral. A "
            "precision of 1.0 means no false outlines; 0.0 means every "
            "predicted outline was wrong."
        ),
        'read': (
            "A curve that stays close to the top-right corner means the "
            "model draws accurate outlines for a large share of corals. "
            "The area under this curve at intersection-over-union (IoU) "
            "0.5 is the mask mAP50 value reported above. When this "
            "curve sits above the box curve on the same run the "
            "predicted outlines are tighter than the predicted "
            "rectangles."
        ),
    },
    'f1_curve_box': {
        'what': (
            "This is the F1 score for bounding-box predictions plotted "
            "as a function of the confidence threshold applied at "
            "deployment. F1 is the balanced average of precision and "
            "recall, computed as 2 x P x R / (P + R)."
        ),
        'x_axis': (
            "The horizontal axis is the model's prediction confidence, "
            "a number between 0 and 1 that expresses how certain the "
            "model is about each prediction. Predictions whose "
            "confidence falls below the chosen threshold are discarded."
        ),
        'y_axis': (
            "The vertical axis is the F1 score (higher is better). "
            "A high F1 means the model is well balanced between "
            "finding real corals and avoiding false alarms."
        ),
        'read': (
            "The peak of the curve identifies the single confidence "
            "threshold at which the model performs best overall. Use "
            "that threshold when running the model on new images. A "
            "sharp, narrow peak means the model is sensitive to the "
            "threshold; a broad plateau means it is forgiving."
        ),
    },
    'f1_curve_mask': {
        'what': (
            "This is the F1 score for the predicted segmentation masks "
            "(pixel-level outlines) plotted as a function of the "
            "confidence threshold applied at deployment."
        ),
        'x_axis': (
            "The horizontal axis is the model's prediction confidence "
            "for each mask, a number between 0 and 1. Masks below the "
            "chosen threshold are discarded before scoring."
        ),
        'y_axis': (
            "The vertical axis is the F1 score (the balanced average "
            "of precision and recall) for masks; higher is better."
        ),
        'read': (
            "Find the peak and read off the matching confidence value "
            "on the horizontal axis. That is the threshold at which "
            "mask-based precision and recall are best balanced. The "
            "peak usually sits slightly below the box F1 peak because "
            "mask metrics are held to a stricter overlap standard than "
            "box metrics."
        ),
    },
    'loss_history': {
        'what': (
            "These are the four training losses recorded at every "
            "training epoch: the bounding-box loss, the segmentation "
            "(mask) loss, the classification loss, and the distribution "
            "focal loss (DFL), a helper loss that sharpens box edges."
        ),
        'x_axis': (
            "The horizontal axis is the epoch number: one epoch is one "
            "full pass through the training images. Early epochs are "
            "on the left; late epochs on the right."
        ),
        'y_axis': (
            "The vertical axis is the loss value, an internal error "
            "score that the optimizer attempts to minimize. Smaller is "
            "better; the absolute value is not directly interpretable "
            "but its trend is."
        ),
        'read': (
            "Both the training-loss curve (solid pink) and the "
            "validation-loss curve (dashed black) should fall together "
            "and then flatten. If the validation curve turns upward "
            "while the training curve keeps dropping, the model has "
            "begun memorizing the training images rather than learning "
            "general coral features (overfitting)."
        ),
    },
    'per_class_bar': {
        'what': (
            "This is the strict mask score (mAP50-95) for every coral "
            "class on the evaluation split, sorted from strongest class "
            "at the top to weakest class at the bottom."
        ),
        'x_axis': (
            "The horizontal axis is mAP50-95, the mean average "
            "precision averaged over ten intersection-over-union (IoU) "
            "thresholds from 0.5 to 0.95. A value of 0 means no "
            "overlap with the human labels; 1 means pixel-perfect."
        ),
        'y_axis': (
            "The vertical axis is one row per coral class; longer "
            "bars are better. Classes are sorted so the strongest "
            "species appear at the top of the chart."
        ),
        'read': (
            "The shortest bars are the classes the model struggles "
            "with most. Cross-reference with the instance-count "
            "column in the table above: classes with very few "
            "evaluation examples look weak simply because there is "
            "little to measure, so a short bar on a rare class is "
            "not necessarily a model problem."
        ),
    },
    'confusion_norm': {
        'what': (
            "This is the row-normalized confusion matrix. Each row is "
            "one true (human-labeled) class and the cells in that row "
            "sum to 1.0, which makes rare classes and common classes "
            "directly comparable."
        ),
        'x_axis': (
            "The horizontal axis is the predicted class — what the "
            "model decided each detection was. Each column names one "
            "possible prediction label."
        ),
        'y_axis': (
            "The vertical axis is the true class — what the human "
            "labeler actually drew. Each row names one possible "
            "ground-truth label."
        ),
        'read': (
            "Each cell shows the fraction of that true class that "
            "the model predicted as the column class. The diagonal "
            "is correct predictions; the brighter a cell is off the "
            "diagonal, the more often the two classes get confused. "
            "A dim diagonal cell corresponds to a low-recall class "
            "in the per-class table on the previous page."
        ),
    },
    'confusion_abs': {
        'what': (
            "This is the absolute confusion matrix. Cells contain raw "
            "counts of predictions rather than fractions, so row sums "
            "equal the total number of evaluation instances for each "
            "true class."
        ),
        'x_axis': (
            "The horizontal axis is the predicted class — what the "
            "model decided each detection was."
        ),
        'y_axis': (
            "The vertical axis is the true (human-labeled, "
            "ground-truth) class — what the human labeler drew."
        ),
        'read': (
            "Large numbers on the diagonal mean the model got that "
            "class right many times. Small diagonals mean few "
            "evaluation examples exist for that class, so its "
            "per-class scores are noisy and should not be "
            "over-interpreted. Compare the row sums here against "
            "training-set instance counts to see whether "
            "underperforming classes were simply underrepresented "
            "during training."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────
# Glossary — compact cheatsheet style
# ─────────────────────────────────────────────────────────────────────────

GLOSSARY_TABLE = [
    # (term, definition, good range, red flag) — every cell is a complete sentence
    # or a complete noun phrase. No shorthand symbols; abbreviations are spelled
    # out before their abbreviation is introduced.
    ('Ground truth (GT)',
     "The human-drawn labels the model is being compared against; the "
     "'correct answer' the model is graded on.",
     "Good ground-truth labels are consistent across images and drawn "
     "tightly around each coral.",
     "Noisy or inconsistent ground-truth labels cause every metric below "
     "to look worse than the model actually is."),
    ('Intersection over union (IoU)',
     "The ratio of the area where a prediction and the ground-truth region "
     "overlap to the area they jointly cover; values run from 0 (no "
     "overlap) to 1 (pixel-perfect match).",
     "An IoU of 0.5 or higher counts as a loose match; 0.75 or higher "
     "counts as a strict match.",
     "If the typical IoU is below 0.3, the model is putting predictions "
     "in the wrong part of the image."),
    ('Precision (P)',
     "The fraction of the model's predictions that are correct. If the "
     "model makes 100 predictions and 80 match a real coral, precision "
     "is 0.80.",
     "A precision above 0.85 is excellent; 0.70 to 0.85 is good.",
     "A precision below 0.5 means the model raises too many false alarms "
     "to be trusted without filtering."),
    ('Recall (R)',
     "The fraction of the real corals (ground-truth instances) that the "
     "model found. If 100 corals exist and the model locates 80, recall "
     "is 0.80.",
     "A recall above 0.85 is excellent; 0.70 to 0.85 is good.",
     "A recall below 0.5 means the model is missing more real corals "
     "than it finds."),
    ('mAP50',
     "The area under the precision-recall curve when a prediction is "
     "considered correct at an IoU of 0.5 or higher, averaged across all "
     "coral classes. It is the lenient detection score.",
     "An mAP50 between 0.55 and 0.75 is good; above 0.75 is very good.",
     "An mAP50 below 0.35 means the model is not reliably detecting "
     "corals at all."),
    ('mAP50-95',
     "The mean average precision computed at ten IoU thresholds from 0.5 "
     "to 0.95 in steps of 0.05, then averaged. It is the strict score "
     "because it rewards tight, accurate localization.",
     "An mAP50-95 between 0.4 and 0.6 is good; above 0.6 is very good.",
     "An mAP50-95 below 0.2 means the model's boundaries are very loose "
     "even when it does find a coral."),
    ('Box vs. mask metrics',
     "Box metrics judge the model by the overlap of predicted and true "
     "bounding rectangles. Mask metrics judge it by the overlap of "
     "predicted and true pixel outlines; mask metrics are strictly "
     "harder.",
     "Mask scores are expected to be slightly lower than box scores on "
     "the same run.",
     "If mask scores are much lower than box scores, the model finds "
     "corals but draws their outlines poorly."),
    ('Confidence threshold',
     "The minimum score a prediction needs to be kept. Predictions below "
     "this score are discarded before any metrics are computed.",
     "A default of 0.25 works for most runs; the F1-vs-confidence curve "
     "on pages 3 and 4 shows the optimal value.",
     "Too high a threshold silently drops real detections; too low lets "
     "false alarms through."),
    ('Non-maximum suppression (NMS) IoU',
     "A cleanup rule: when two predictions of the same class overlap by "
     "more than this IoU, only the higher-confidence one is kept.",
     "Values between 0.6 and 0.7 are standard.",
     "A value set too low leaves duplicate detections of the same coral "
     "on the output."),
    ('Fitness',
     "Ultralytics's internal checkpoint-selector score — a weighted "
     "combination of mAP values used to pick which epoch to save as "
     "'best.pt'.",
     "Useful for comparing two runs of the same architecture on the "
     "same data.",
     "Not directly comparable across different model architectures."),
    ('Confidence',
     "The model's self-reported certainty for each prediction, expressed "
     "as a number between 0 (no confidence) and 1 (maximum confidence). "
     "Predictions whose confidence score falls below the chosen "
     "threshold are discarded before the rest of the metrics are "
     "computed.",
     "Most runs use a confidence threshold between 0.20 and 0.40; the "
     "F1-vs-confidence curves on pages 3 and 4 show the best value for "
     "this run.",
     "If every prediction is close to 1.0 the model is overconfident; "
     "if every prediction sits near 0.5 the model has not learned to "
     "distinguish confident calls from guesses."),
    ('F1 score',
     "The harmonic mean (balanced average) of precision and recall. It "
     "is computed as 2 x P x R / (P + R) and rewards models only when "
     "both precision and recall are high together — a model with "
     "precision 0.9 but recall 0.1 earns an F1 near 0.18, not 0.5.",
     "An F1 above 0.75 is good; above 0.85 is excellent.",
     "An F1 below 0.5 means the model cannot be relied on without "
     "substantial manual correction."),
]


# ─────────────────────────────────────────────────────────────────────────
# Number/duration formatters + value verdict helpers
# ─────────────────────────────────────────────────────────────────────────

def fmt_num(x, nd=3):
    if x is None:
        return 'N/A'
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return str(x)


def fmt_dur(secs):
    if secs is None:
        return 'N/A'
    s = int(secs)
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m {s}s'
    h, m = divmod(m, 60)
    return f'{h}h {m}m {s}s'


# ─────────────────────────────────────────────────────────────────────────
# Helpers locating ultralytics artefacts
# ─────────────────────────────────────────────────────────────────────────

def _find_val_plots(out_dir):
    val_run = os.path.join(out_dir, 'val_run')
    wanted = [
        'confusion_matrix.png', 'confusion_matrix_normalized.png',
        'BoxPR_curve.png', 'MaskPR_curve.png',
        'BoxF1_curve.png', 'MaskF1_curve.png',
        'BoxP_curve.png',  'MaskP_curve.png',
        'BoxR_curve.png',  'MaskR_curve.png',
        'val_batch0_labels.jpg', 'val_batch0_pred.jpg',
        'labels.jpg', 'labels_correlogram.jpg',
    ]
    out = {}
    if os.path.isdir(val_run):
        for n in wanted:
            p = os.path.join(val_run, n)
            if os.path.isfile(p):
                out[n] = p
    return out


def _find_train_plots(run_dir):
    """Same search but inside the training run dir (ultralytics saves curves here
    after training, and they persist on rerun)."""
    wanted = ['BoxPR_curve.png', 'MaskPR_curve.png',
              'BoxF1_curve.png', 'MaskF1_curve.png',
              'confusion_matrix.png', 'confusion_matrix_normalized.png']
    out = {}
    if os.path.isdir(run_dir):
        for n in wanted:
            p = os.path.join(run_dir, n)
            if os.path.isfile(p):
                out[n] = p
    return out


# ─────────────────────────────────────────────────────────────────────────
# New plotting helpers (matplotlib figures saved to PNG)
# ─────────────────────────────────────────────────────────────────────────

def _plot_per_class_bar(per_class_data, out_path, metric_key='mask_map50_95'):
    """Horizontal bar chart of per-class metric, sorted descending. Pink palette."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    items = []
    for cls, d in per_class_data.items():
        v = d.get(metric_key)
        if v is None:
            continue
        items.append((cls, float(v)))
    if not items:
        return None
    items.sort(key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    values = [i[1] for i in items]

    h = min(3.2, max(1.2, 0.35 * len(items) + 0.8))
    fig, ax = plt.subplots(figsize=(6.5, h), dpi=130)
    y_pos = range(len(items))
    ax.barh(list(y_pos), values, color='#E20074', alpha=0.85, edgecolor='#8a003f')
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('mAP50-95 (mask)', fontsize=8)
    ax.set_xlim(0, max(1.0, max(values) * 1.15))
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', alpha=0.3, linewidth=0.4)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', fontsize=7, va='center', color='#222')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    plt.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return out_path


def _plot_training_history(results_csv_path, out_path):
    """2x2 grid: box/seg/cls/dfl losses, train vs val on each panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.isfile(results_csv_path):
        return None

    with open(results_csv_path) as f:
        rows = list(_csv.reader(f))
    if len(rows) < 2:
        return None
    hdr = [h.strip() for h in rows[0]]
    data = rows[1:]

    def col(name):
        if name not in hdr:
            return None
        idx = hdr.index(name)
        out = []
        for r in data:
            if idx >= len(r):
                out.append(np.nan); continue
            try:
                out.append(float(r[idx]))
            except Exception:
                out.append(np.nan)
        return np.array(out)

    epochs = col('epoch')
    if epochs is None:
        epochs = np.arange(len(data))

    panels = [
        ('box',  'train/box_loss',  'val/box_loss',  'Box loss'),
        ('seg',  'train/seg_loss',  'val/seg_loss',  'Seg loss'),
        ('cls',  'train/cls_loss',  'val/cls_loss',  'Cls loss'),
        ('dfl',  'train/dfl_loss',  'val/dfl_loss',  'DFL loss'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8), dpi=130)
    for ax, (_k, tc, vc, title) in zip(axes.ravel(), panels):
        t = col(tc)
        v = col(vc)
        if t is not None:
            ax.plot(epochs, t, color='#E20074', lw=1.2, label='train')
        if v is not None:
            ax.plot(epochs, v, color='#333', lw=1.0, label='val', linestyle='--')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.4)
        ax.legend(fontsize=7, loc='upper right')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    return out_path


def _interpret_convergence(results_csv_path):
    """Plain-language commentary on how training converged, written as
    complete sentences so the paragraph can be pasted into the PDF as-is.

    Always spells out the terms 'training loss' and 'validation loss' (the
    internal error scores the optimizer drives down), so the reader does
    not need to know what the loss curves mean coming in.
    """
    if not os.path.isfile(results_csv_path):
        return ''
    try:
        with open(results_csv_path) as f:
            rows = list(_csv.reader(f))
        if len(rows) < 6:
            n = max(0, len(rows) - 1)
            return (
                f"Training ran for only {n} epochs, which is too few to draw "
                "firm conclusions about convergence or overfitting."
            )
        hdr = [h.strip() for h in rows[0]]
        data = rows[1:]
        import numpy as np
        def col(name):
            if name not in hdr: return None
            idx = hdr.index(name)
            return np.array([float(r[idx]) if idx < len(r) and r[idx].strip() else np.nan
                             for r in data])
        vbox = col('val/box_loss')
        tbox = col('train/box_loss')
        n = len(data)
        if vbox is None or tbox is None:
            return (
                f"Training ran for {n} epochs, but the loss columns were not "
                "found in results.csv, so the convergence behavior cannot be "
                "described here."
            )
        final = np.nanmin(vbox[-10:]) if n >= 10 else np.nanmin(vbox)
        converge_ep = n
        for i, v in enumerate(vbox):
            if not np.isnan(v) and v <= final * 1.02:
                converge_ep = i + 1
                break

        tail = max(5, n // 5)
        overfit = False
        if n >= tail * 2:
            tr_slope = np.nanmean(tbox[-tail:]) - np.nanmean(tbox[-tail*2:-tail])
            va_slope = np.nanmean(vbox[-tail:]) - np.nanmean(vbox[-tail*2:-tail])
            overfit = (tr_slope < -0.002 and va_slope > 0.002)

        sent1 = (
            f"The training loss and validation loss both fell through "
            f"epoch {n} and reached their plateau near epoch "
            f"{converge_ep}, which is where the model stopped learning new "
            "generalizable features."
        )
        if overfit:
            sent2 = (
                f"Over the final {tail} epochs, the training loss kept "
                "dropping while the validation loss rose slightly, which is "
                "the classic symptom of overfitting: the model is starting "
                "to memorize the training images rather than learn patterns "
                "that transfer to new data."
            )
            sent3 = (
                "Because the checkpoint saved as 'best.pt' is picked from "
                "the best validation epoch, the saved weights predate this "
                "overfit period and should still generalize, but future "
                "training runs would benefit from stopping earlier."
            )
        else:
            sent2 = (
                f"Across the final {tail} epochs there was no sign of "
                "overfitting: the validation loss did not rise while the "
                "training loss continued to fall, which means the model was "
                "still learning generalizable patterns rather than "
                "memorizing the training set."
            )
            sent3 = (
                "The 'best.pt' checkpoint chosen by the trainer corresponds "
                "to an epoch where validation performance had plateaued, so "
                "it is a reasonable representative of the run's quality."
            )
        return ' '.join([sent1, sent2, sent3])
    except Exception:
        return ''


# ─────────────────────────────────────────────────────────────────────────
# Markdown report (kept lean — PDF is the primary artefact)
# ─────────────────────────────────────────────────────────────────────────

def build_report_md(metrics, ctx, previews, args):
    L = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_name = os.path.basename(ctx['run_dir'].rstrip('/'))
    o = metrics.get('overall') or {}
    ta = ctx.get('train_args', {}) or {}
    counts = ctx.get('split_counts', {}) or {}

    L.append(f'# Evaluation report for run `{run_name}`')
    L.append('')
    L.append(
        f"This report was generated on {now}. It evaluates the model on "
        f"the '{args.split}' split, and the full evaluation took "
        f"{fmt_dur(metrics['wall_seconds'])} of wall-clock time."
    )
    L.append('')
    L.append('## Top-line metrics')
    L.append('')
    L.append(
        "Precision (P) is the fraction of the model's predictions that "
        "are correct. Recall (R) is the fraction of the real corals the "
        "model finds. Mean average precision at intersection-over-union "
        "0.5 (mAP50) is the lenient detection score; averaged from 0.5 "
        "to 0.95 (mAP50-95) is the strict score. Box columns score the "
        "model's predicted rectangles; mask columns score its pixel "
        "outlines."
    )
    L.append('')
    L.append('| Metric | Box | Mask | Good range |')
    L.append('|---|---|---|---|')
    L.append(f'| mAP50 | {fmt_num(o.get("box_map50"))} | {fmt_num(o.get("mask_map50"))} | {GOOD_RANGES["mAP50"]} |')
    L.append(f'| mAP50-95 | {fmt_num(o.get("box_map50_95"))} | {fmt_num(o.get("mask_map50_95"))} | {GOOD_RANGES["mAP50-95"]} |')
    L.append(f'| Precision | {fmt_num(o.get("box_precision"))} | {fmt_num(o.get("mask_precision"))} | {GOOD_RANGES["P/R"]} |')
    L.append(f'| Recall | {fmt_num(o.get("box_recall"))} | {fmt_num(o.get("mask_recall"))} | {GOOD_RANGES["P/R"]} |')
    L.append('')
    L.append('## Training context')
    L.append('')
    L.append(f'- model={ta.get("model", "?")} · imgsz={ta.get("imgsz", "?")} · batch={ta.get("batch", "?")}')
    L.append(f'- epochs_requested={ta.get("epochs", "?")} · epochs_completed={ctx.get("epochs_completed", "?")} · best_epoch={ctx.get("best_epoch", "?")}')
    L.append(f'- splits: train={counts.get("train", "?")} valid={counts.get("valid", "?")} test={counts.get("test", "?")}')
    L.append(f'- eval_args: conf={args.conf} iou={args.iou} imgsz={args.imgsz}')
    L.append('')

    pc = metrics.get('per_class') or {}
    if pc:
        L.append('## Per-class')
        L.append('')
        L.append('| Class | Inst | P | R | mAP50 | mAP50-95 |')
        L.append('|---|---|---|---|---|---|')
        inst = metrics.get('per_class_instances') or {}
        for cls in sorted(pc.keys()):
            d = pc[cls]
            L.append(f'| `{cls}` | {inst.get(cls, "?")} | {fmt_num(d.get("mask_precision"))} | '
                     f'{fmt_num(d.get("mask_recall"))} | {fmt_num(d.get("mask_map50"))} | '
                     f'{fmt_num(d.get("mask_map50_95"))} |')
        L.append('')

    if previews:
        L.append(f'## Prediction samples ({len(previews)} images)')
        L.append('')
        L.append(
            "Each image is shown with the human-drawn ground-truth (GT) "
            "outlines on the left and the model's predicted outlines on "
            "the right."
        )
        L.append('')
        for p in previews:
            rel = os.path.relpath(p['overlay'], args.out_dir)
            L.append(
                f"- **{p['filename']}** — the human labeler drew "
                f"{p.get('n_gt', '?')} ground-truth instance(s); the model "
                f"produced {p.get('n_pred', '?')} detection(s)."
            )
            L.append(f'  ![{p["filename"]}]({rel})')
        L.append('')

    L.append('## Glossary')
    L.append('')
    L.append(
        "Every term and abbreviation used above is defined below as a "
        "complete sentence. The 'Good' column gives a healthy band for "
        "each metric on this kind of data; the 'Red flag' column names "
        "the symptom you see when it is too low."
    )
    L.append('')
    L.append('| Term | Definition | Good | Red flag |')
    L.append('|---|---|---|---|')
    for row in GLOSSARY_TABLE:
        # Escape pipe chars that might appear in sentences
        cells = [str(c).replace('|', '\\|') for c in row]
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')
    L.append(f'_Weights_: `{metrics["weights"]}`  ·  _Data_: `{metrics["data_yaml"]}`')
    return '\n'.join(L)


# ─────────────────────────────────────────────────────────────────────────
# Metric-pair interpretation prose (used by PDF builder)
# ─────────────────────────────────────────────────────────────────────────

def _interpret_metric_pair(which, p, r, m50, m5095, f1):
    """Plain-language prose commentary on one row of metrics (box or mask).

    Returns 2-3 complete sentences with every abbreviation spelled out
    on first use, so the output can be dropped straight into the PDF.
    """
    label = 'bounding-box' if which == 'box' else 'segmentation-mask'
    if None in (p, r, m50, m5095):
        return (
            f"The {label} metrics are incomplete for this run, so a full "
            "interpretation cannot be written; consult results.csv and "
            "the validation log for details."
        )

    if abs(p - r) > 0.15:
        if p > r:
            sent_pr = (
                f"Precision (P) is {p:.2f} and recall (R) is {r:.2f} on "
                f"the {label} metric, so the model is noticeably more "
                "careful than it is thorough: when it does make a "
                "prediction the prediction is usually correct, but a "
                "meaningful fraction of real corals are being missed. "
                "Lowering the confidence threshold would raise recall "
                "at the cost of some precision."
            )
        else:
            sent_pr = (
                f"Precision (P) is {p:.2f} and recall (R) is {r:.2f} on "
                f"the {label} metric, so the model is more thorough "
                "than it is careful: it finds most real corals but also "
                "raises a meaningful number of false alarms. Raising "
                "the confidence threshold would reduce false alarms at "
                "some cost to recall."
            )
    else:
        sent_pr = (
            f"Precision (P) is {p:.2f} and recall (R) is {r:.2f} on "
            f"the {label} metric, which means the model is about as "
            "likely to miss a real coral as it is to falsely flag "
            "something that is not one — a balanced operating point."
        )

    gap = m50 - m5095
    if gap > 0.20:
        sent_gap = (
            f"The gap between the lenient score (mAP50, measured at "
            f"intersection-over-union, IoU, of 0.5) and the strict "
            f"score (mAP50-95, averaged from IoU 0.5 to 0.95) is "
            f"{gap:.2f}, which is larger than usual and means the "
            "model is finding the right general region but its "
            "predictions do not line up tightly with the human-drawn "
            "boundaries."
        )
    elif gap > 0.10:
        sent_gap = (
            f"The gap between the lenient score (mAP50, measured at "
            f"intersection-over-union, IoU, of 0.5) and the strict "
            f"score (mAP50-95, averaged from IoU 0.5 to 0.95) is "
            f"{gap:.2f}, which is typical for this kind of model: "
            "predictions overlap the human-drawn region well but not "
            "pixel-perfectly."
        )
    else:
        sent_gap = (
            f"The gap between the lenient score (mAP50, measured at "
            f"intersection-over-union, IoU, of 0.5) and the strict "
            f"score (mAP50-95, averaged from IoU 0.5 to 0.95) is "
            f"only {gap:.2f}, which indicates that the model's "
            "predictions line up almost pixel-for-pixel with the "
            "human-drawn boundaries."
        )

    return ' '.join([sent_pr, sent_gap])


# ─────────────────────────────────────────────────────────────────────────
# PDF — WeasyPrint-based (HTML + CSS) cheatsheet with fixed per-page layout
# ─────────────────────────────────────────────────────────────────────────

def _html_escape(s):
    """Minimal HTML-escape for text content."""
    if s is None:
        return ''
    return (
        str(s)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def _axis_block_html(key):
    """Render the 4-line axis explanation block for a plot."""
    e = AXIS_EXPLANATIONS_FULL.get(key)
    if not e:
        return ''
    read_full = (e.get('read', '') + ' ' + e.get('connects', '')).strip()
    items = [
        ('What:', e.get('what', '')),
        ('Horizontal axis:', e.get('x_axis', '')),
        ('Vertical axis:', e.get('y_axis', '')),
        ('How to read:', read_full),
    ]
    parts = ['<div class="axis-block">']
    for lbl, body in items:
        parts.append(
            f'<div class="axis-line"><span class="axis-label">{_html_escape(lbl)}</span>'
            f'<span class="axis-body">{_html_escape(body)}</span></div>'
        )
    parts.append('</div>')
    return ''.join(parts)


def _img_src(path):
    """Build a file:// URI for an absolute image path. Returns empty on None."""
    if not path:
        return ''
    try:
        ap = os.path.abspath(path)
        return 'file://' + ap
    except Exception:
        return ''


def _build_css(brand='#E20074'):
    """Return the inline CSS string. All sizes in pt so they hit paper faithfully."""
    return r"""
    @page {
        size: letter;
        margin: 0.3in;
        @bottom-center {
            content: "GT = human-drawn ground-truth label   |   pred = model prediction   |   P = precision (fraction of predictions correct)   |   R = recall (fraction of real corals found)   |   IoU = intersection over union   |   mAP50 = mean average precision at IoU >= 0.5   |   mAP50-95 = mAP averaged over IoU 0.5 to 0.95";
            font-family: Helvetica, Arial, "DejaVu Sans", sans-serif;
            font-style: italic;
            font-size: 6pt;
            color: #666;
        }
        @bottom-right {
            content: "p." counter(page) " / " counter(pages);
            font-family: Helvetica, Arial, "DejaVu Sans", sans-serif;
            font-size: 6pt;
            color: #888;
        }
    }

    html, body {
        font-family: Helvetica, Arial, "DejaVu Sans", sans-serif;
        font-size: 8pt;
        line-height: 1.15;
        color: #1a1a1a;
        margin: 0;
        padding: 0;
    }

    p { margin: 0 0 3pt 0; }
    h1, h2, h3 { margin: 0; padding: 0; font-weight: bold; }

    h1.page-title {
        font-size: 14pt;
        color: """ + brand + r""";
        border-bottom: 1.2pt solid """ + brand + r""";
        padding-bottom: 2pt;
        margin-bottom: 4pt;
    }
    h2.section {
        font-size: 11pt;
        color: #1a1a1a;
        border-bottom: 0.5pt solid #bbb;
        padding-bottom: 1pt;
        margin-top: 6pt;
        margin-bottom: 3pt;
    }
    h3.subsection {
        font-size: 9pt;
        color: #333;
        margin-top: 4pt;
        margin-bottom: 2pt;
    }

    .mono { font-family: "DejaVu Sans Mono", Menlo, Consolas, monospace; }
    .dim { color: #666; }
    .brand { color: """ + brand + r"""; }
    .subtitle { font-size: 8pt; color: #444; margin-bottom: 4pt; line-height: 1.2; }

    section.page {
        page-break-after: always;
        page-break-inside: avoid;
    }
    section.page.last {
        page-break-after: auto;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8pt;
        margin-bottom: 4pt;
    }
    table.tight td, table.tight th {
        padding: 2px 4px;
        border: 0.5pt solid #ccc;
        vertical-align: top;
        text-align: left;
        word-wrap: break-word;
    }
    table.tight th {
        background: """ + brand + r""";
        color: white;
        font-weight: bold;
        font-size: 8pt;
    }
    table.tight tr:nth-child(even) td { background: #f6f6f6; }
    table.tight td.num { text-align: right; font-variant-numeric: tabular-nums; }
    table.tight td.path {
        font-family: "DejaVu Sans Mono", monospace;
        font-size: 7pt;
        word-break: break-all;
        white-space: normal;
    }
    table.kv td { padding: 1.5px 4px; border: 0.4pt solid #ddd; font-size: 7.5pt; }
    table.kv td.key {
        font-family: "DejaVu Sans Mono", monospace;
        color: #555;
        white-space: nowrap;
        width: 35%;
    }
    table.kv td.val {
        font-family: "DejaVu Sans Mono", monospace;
        word-break: break-all;
        white-space: normal;
    }
    table.kv tr:nth-child(even) td { background: #f6f6f6; }

    /* Two-column layouts */
    .row { display: table; width: 100%; table-layout: fixed; }
    .col { display: table-cell; vertical-align: top; padding: 0 2pt; }
    .col.left  { width: 50%; padding-left: 0; padding-right: 4pt; }
    .col.right { width: 50%; padding-right: 0; padding-left: 4pt; }
    .col-38 { width: 38%; }
    .col-60 { width: 60%; padding-left: 4pt; }
    .col-55 { width: 55%; padding-right: 4pt; }
    .col-43 { width: 43%; padding-left: 4pt; }

    /* 4-quadrant top-line big numbers */
    .quads {
        display: table;
        width: 100%;
        table-layout: fixed;
        margin: 4pt 0 4pt 0;
    }
    .quad {
        display: table-cell;
        border: 0.8pt solid #ccc;
        border-radius: 3pt;
        padding: 5pt 3pt;
        text-align: center;
        vertical-align: middle;
        width: 25%;
    }
    .quad + .quad { border-left: 0.8pt solid #ccc; }
    .quad .label { font-size: 8pt; color: #555; display: block; margin-bottom: 2pt; }
    .quad .value {
        font-size: 18pt;
        font-weight: bold;
        color: """ + brand + r""";
        display: block;
        line-height: 1.0;
        margin: 1pt 0;
    }
    .quad .verdict {
        font-size: 7.5pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.3pt;
        display: block;
        margin-top: 2pt;
    }

    /* Figures */
    img.fig { display: block; max-width: 100%; height: auto; margin: 0 auto; }
    .fig-wrap { text-align: center; margin: 2pt 0; }
    .fig-caption {
        font-size: 6.5pt;
        color: #666;
        font-style: italic;
        text-align: center;
        margin-top: 1pt;
        line-height: 1.2;
    }

    /* Axis explanation block beneath each plot */
    .axis-block {
        font-size: 7pt;
        line-height: 1.25;
        margin: 3pt 0 4pt 0;
        padding: 3pt 4pt;
        background: #fafafa;
        border-left: 1.5pt solid """ + brand + r""";
    }
    .axis-line { margin-bottom: 1.5pt; }
    .axis-label {
        font-weight: bold;
        font-style: normal;
        color: #222;
        display: inline;
        margin-right: 3pt;
    }
    .axis-body {
        font-style: italic;
        color: #222;
    }

    /* Interpretation paragraph */
    .interp {
        font-size: 8pt;
        margin-top: 4pt;
        padding: 3pt 5pt;
        background: #fff8fb;
        border-left: 1.5pt solid """ + brand + r""";
    }
    .interp .lead { font-weight: bold; color: """ + brand + r"""; }

    /* Sample tiles — 2x2 grid */
    .sample-grid {
        display: table;
        width: 100%;
        table-layout: fixed;
        border-spacing: 4pt 4pt;
    }
    .sample-row { display: table-row; }
    .sample-tile {
        display: table-cell;
        width: 50%;
        vertical-align: top;
        border: 0.5pt solid #ccc;
        padding: 2pt;
    }
    .sample-tile img {
        display: block;
        max-width: 100%;
        height: auto;
        max-height: 2.2in;
        margin: 0 auto;
    }
    .sample-caption {
        font-size: 6.5pt;
        color: #333;
        text-align: left;
        margin-top: 1pt;
        line-height: 1.2;
    }
    .sample-caption .fn {
        font-family: "DejaVu Sans Mono", monospace;
        color: #000;
        font-weight: bold;
    }

    /* Glossary 2-column */
    .glossary {
        column-count: 2;
        column-gap: 0.2in;
        column-fill: balance;
        font-size: 7pt;
        line-height: 1.25;
    }
    .glossary .entry {
        break-inside: avoid;
        margin-bottom: 4pt;
        padding-bottom: 3pt;
        border-bottom: 0.3pt dotted #bbb;
    }
    .glossary .term {
        font-weight: bold;
        color: """ + brand + r""";
        font-size: 7.5pt;
        display: block;
        margin-bottom: 1pt;
    }
    .glossary .defn { margin-bottom: 1pt; }
    .glossary .good { color: #1a6a1a; font-style: italic; }
    .glossary .red { color: #a00020; font-style: italic; }

    /* TOC */
    ul.toc { list-style: none; padding: 0; margin: 0; columns: 2; column-gap: 0.3in; font-size: 8pt; }
    ul.toc li { margin: 1pt 0; padding: 0; break-inside: avoid; }
    ul.toc .pg { color: #666; margin-left: 4pt; font-variant-numeric: tabular-nums; }

    /* Verdict color classes (for quad + table cells) */
    .v-excellent { color: #0d5a0d; }
    .v-very-good { color: #1f7a1f; }
    .v-good      { color: #2a6a2a; }
    .v-ok        { color: #c06000; }
    .v-weak      { color: #c06000; }
    .v-poor      { color: #b00020; }
    .v-not-learning { color: #b00020; }
    .v-na        { color: #888; }

    .callout {
        padding: 4pt 6pt;
        background: #fff8fb;
        border: 0.5pt solid #f0c0d8;
        border-radius: 2pt;
        margin: 3pt 0;
        font-size: 7.5pt;
    }
    .callout .headline { font-weight: bold; color: """ + brand + r"""; }

    /* Per-class notes list */
    .reading-notes { font-size: 7pt; color: #444; line-height: 1.3; }
    .reading-notes li { margin-bottom: 1.5pt; }
    """


_VERDICT_CLASS = {
    'excellent': 'v-excellent',
    'very good': 'v-very-good',
    'good': 'v-good',
    'OK': 'v-ok',
    'weak': 'v-weak',
    'poor': 'v-poor',
    'not learning': 'v-not-learning',
    '—': 'v-na',
}


def _verdict_html(val, band_fn):
    """Return (formatted_value, verdict_word, css_class)."""
    verdict, _color = band_fn(val)
    css = _VERDICT_CLASS.get(verdict, 'v-na')
    return fmt_num(val), verdict, css


def _render_cover(metrics, ctx, args, toc_entries, now, run_name):
    o = metrics.get('overall') or {}
    ta = ctx.get('train_args', {}) or {}
    counts = ctx.get('split_counts', {}) or {}

    # 4 quadrants
    quads = []
    for lbl, val, band in (
        ('Box mAP50',     o.get('box_map50'),     _band_map50),
        ('Box mAP50-95',  o.get('box_map50_95'),  _band_map50_95),
        ('Mask mAP50',    o.get('mask_map50'),    _band_map50),
        ('Mask mAP50-95', o.get('mask_map50_95'), _band_map50_95),
    ):
        fv, verdict, css = _verdict_html(val, band)
        quads.append(
            f'<div class="quad"><span class="label">{_html_escape(lbl)}</span>'
            f'<span class="value">{_html_escape(fv)}</span>'
            f'<span class="verdict {css}">{_html_escape(verdict)}</span></div>'
        )
    quad_html = '<div class="quads">' + ''.join(quads) + '</div>'

    # Identity table (2-col key-value, split into two kv tables side by side)
    ident_items = [
        ('run', run_name),
        ('dataset', os.path.basename(ctx['dataset_dir'].rstrip('/'))),
        ('base model', str(ta.get('model', '?'))),
        ('train imgsz', str(ta.get('imgsz', '?'))),
        ('epochs req/done', f'{ta.get("epochs", "?")} / {ctx.get("epochs_completed", "?")}'),
        ('best.pt epoch', str(ctx.get('best_epoch', '?'))),
        ('eval wall', fmt_dur(metrics.get('wall_seconds'))),
        ('split sizes',
         f'train={counts.get("train", "?")} valid={counts.get("valid", "?")} test={counts.get("test", "?")}'),
        ('weights', metrics.get('weights', '')),
        ('dataset yaml', metrics.get('data_yaml', '')),
        ('eval args', f'split={args.split} conf={args.conf} iou={args.iou} imgsz={args.imgsz}'),
        ('generated', now),
    ]
    half = (len(ident_items) + 1) // 2
    lcol = ident_items[:half]
    rcol = ident_items[half:]

    def _kv_tbl(rows):
        body = ''.join(
            f'<tr><td class="key">{_html_escape(k)}</td>'
            f'<td class="val">{_html_escape(v)}</td></tr>'
            for k, v in rows
        )
        return f'<table class="kv">{body}</table>'

    ident_html = (
        '<div class="row">'
        f'<div class="col left">{_kv_tbl(lcol)}</div>'
        f'<div class="col right">{_kv_tbl(rcol)}</div>'
        '</div>'
    )

    # Secondary numbers table
    sec_rows = [
        ('Precision', fmt_num(o.get('box_precision')), fmt_num(o.get('mask_precision')), GOOD_RANGES['P/R']),
        ('Recall',    fmt_num(o.get('box_recall')),    fmt_num(o.get('mask_recall')),    GOOD_RANGES['P/R']),
    ]
    speed = metrics.get('speed_ms') or {}
    total_ms = sum(speed.values()) if speed else None
    fitness_note = f'Infer {fmt_num(total_ms, 1)} ms/img' if total_ms else ''
    sec_rows.append(('Fitness', fmt_num(metrics.get('fitness'), 4), '', fitness_note))
    sec_body = ''.join(
        '<tr>'
        f'<td>{_html_escape(r[0])}</td>'
        f'<td class="num">{_html_escape(r[1])}</td>'
        f'<td class="num">{_html_escape(r[2])}</td>'
        f'<td>{_html_escape(r[3])}</td>'
        '</tr>' for r in sec_rows
    )
    sec_html = (
        '<table class="tight">'
        '<tr><th style="width:18%">Metric</th><th style="width:16%">Box</th>'
        '<th style="width:16%">Mask</th><th>Good range</th></tr>'
        f'{sec_body}</table>'
    )

    # TOC
    toc_items = ''.join(
        f'<li>{_html_escape(name)}<span class="pg">p.{pg}</span></li>'
        for name, pg in toc_entries
    )
    toc_html = f'<ul class="toc">{toc_items}</ul>'

    intro = (
        f"This report evaluates run '{_html_escape(run_name)}' on the "
        f"'{_html_escape(args.split)}' split. Every page that follows covers one topic "
        "in the same order across runs so two reports can be compared "
        "page-by-page; abbreviations are defined the first time they appear "
        "on each page."
    )

    return (
        '<section class="page">'
        '<h1 class="page-title">Evaluation report</h1>'
        f'<div class="subtitle">{intro}</div>'
        '<h2 class="section">Top-line metrics</h2>'
        f'{quad_html}'
        '<h2 class="section">Secondary numbers</h2>'
        f'{sec_html}'
        '<h2 class="section">Run identity</h2>'
        f'{ident_html}'
        '<h2 class="section">Contents (stable page numbering)</h2>'
        f'{toc_html}'
        '</section>'
    )


def _render_training(metrics, ctx, args, figures):
    ta = ctx.get('train_args', {}) or {}
    keys = [
        'model', 'epochs', 'batch', 'imgsz', 'optimizer',
        'lr0', 'lrf', 'momentum', 'weight_decay', 'patience',
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate',
        'scale', 'shear', 'perspective', 'flipud', 'fliplr',
        'mosaic', 'mixup', 'cutmix', 'copy_paste',
    ]
    # 2-column hyperparam list via CSS columns
    kv_rows = ''.join(
        f'<tr><td class="key">{_html_escape(k)}</td>'
        f'<td class="val">{_html_escape(ta.get(k, "—"))}</td></tr>'
        for k in keys
    )
    cfg_tbl = f'<table class="kv">{kv_rows}</table>'

    hist_img = figures.get('training_history')
    if hist_img:
        hist_html = (
            f'<div class="fig-wrap"><img class="fig" src="{_img_src(hist_img)}" alt="Loss curves"/></div>'
        )
    else:
        hist_html = '<div class="dim">[loss curves unavailable — results.csv missing or unparseable]</div>'

    axis_html = _axis_block_html('loss_history')
    interp = _interpret_convergence(os.path.join(ctx['run_dir'], 'results.csv'))
    interp_html = (
        f'<div class="interp"><span class="lead">Convergence. </span>{_html_escape(interp)}</div>'
        if interp else ''
    )

    return (
        '<section class="page">'
        '<h1 class="page-title">Page 2 — Training history</h1>'
        '<div class="subtitle">'
        "Loss is the internal error score the optimizer tries to push down. "
        "Four losses are tracked: box (how well bounding rectangles fit), "
        "seg (how well masks fit), cls (how well the species label is "
        "predicted), and dfl (a helper loss that sharpens box edges). "
        "Hyperparameters shown at left were loaded from the run's args.yaml."
        '</div>'
        '<div class="row">'
        '<div class="col col-38">'
        '<h3 class="subsection">Training config</h3>'
        f'{cfg_tbl}'
        '</div>'
        '<div class="col col-60">'
        '<h3 class="subsection">Loss curves (train vs validation)</h3>'
        f'{hist_html}'
        f'{axis_html}'
        '</div>'
        '</div>'
        f'{interp_html}'
        '</section>'
    )


def _render_metric_page(which, page_num, metrics, ctx, args, figures):
    o = metrics.get('overall') or {}
    if which == 'box':
        p = o.get('box_precision'); r = o.get('box_recall')
        m50 = o.get('box_map50');   m5095 = o.get('box_map50_95')
        title = f'Page {page_num} — Box metrics'
        sub = (
            "Box metrics compare the model's predicted bounding rectangle "
            "against the human-drawn (ground-truth, GT) rectangle using "
            "intersection over union (IoU). This is the lenient view — the "
            "model only has to place a box roughly over each coral."
        )
        pr_fig = figures.get('box_pr')
        f1_fig = figures.get('box_f1')
        pr_key = 'pr_curve_box'
        f1_key = 'f1_curve_box'
    else:
        p = o.get('mask_precision'); r = o.get('mask_recall')
        m50 = o.get('mask_map50');   m5095 = o.get('mask_map50_95')
        title = f'Page {page_num} — Mask metrics'
        sub = (
            "Mask metrics compare the model's predicted pixel outline "
            "against the human-drawn (ground-truth, GT) outline using "
            "intersection over union (IoU). This is the strict view — the "
            "model has to trace each coral, not just rectangle it."
        )
        pr_fig = figures.get('mask_pr')
        f1_fig = figures.get('mask_f1')
        pr_key = 'pr_curve_mask'
        f1_key = 'f1_curve_mask'

    f1 = None
    if p is not None and r is not None and (p + r) > 0:
        f1 = 2 * p * r / (p + r)

    def _row(label, val, band_fn, good_key):
        fv, verdict, css = _verdict_html(val, band_fn)
        return (
            '<tr>'
            f'<td>{_html_escape(label)}</td>'
            f'<td class="num"><b>{_html_escape(fv)}</b> '
            f'<span class="{css}">({_html_escape(verdict)})</span></td>'
            f'<td>{_html_escape(GOOD_RANGES[good_key])}</td>'
            '</tr>'
        )
    summary_body = (
        _row('Precision (P)', p,   _band_pr,       'P/R')
        + _row('Recall (R)',    r,   _band_pr,       'P/R')
        + _row('F1',            f1,  _band_pr,       'F1')
        + _row('mAP50',         m50, _band_map50,    'mAP50')
        + _row('mAP50-95',      m5095, _band_map50_95, 'mAP50-95')
    )
    summary_tbl = (
        '<table class="tight">'
        '<tr><th style="width:18%">Metric</th><th style="width:28%">Value</th>'
        '<th>Good range</th></tr>'
        f'{summary_body}</table>'
    )

    def _fig_block(path, alt):
        if path and os.path.isfile(path):
            return f'<div class="fig-wrap"><img class="fig" src="{_img_src(path)}" alt="{_html_escape(alt)}"/></div>'
        return f'<div class="dim">[{_html_escape(alt)} unavailable]</div>'

    pr_html = _fig_block(pr_fig, f'{which} PR curve')
    f1_html = _fig_block(f1_fig, f'{which} F1 curve')
    pr_axis = _axis_block_html(pr_key)
    f1_axis = _axis_block_html(f1_key)

    interp_lines = [_interpret_metric_pair(which, p, r, m50, m5095, f1)]
    if which == 'mask':
        bm50 = o.get('box_map50'); bm5095 = o.get('box_map50_95')
        if bm50 is not None and m50 is not None:
            d50 = bm50 - m50
            d5095 = (bm5095 - m5095) if (bm5095 is not None and m5095 is not None) else None
            parts = [
                "Comparing box scores against mask scores on the same run, "
                f"the mask mAP50 trails the box mAP50 by {d50:+.3f}"
            ]
            if d5095 is not None:
                parts.append(f" and the mask mAP50-95 trails the box mAP50-95 by {d5095:+.3f}")
            parts.append('. ')
            if d5095 is not None and d5095 > 0.10:
                parts.append(
                    "This is a large gap, which means the model is finding "
                    "corals successfully but drawing loose outlines around "
                    "them; the segmentation head has room to improve even "
                    "though detection is working."
                )
            elif d5095 is not None and d5095 < 0.03:
                parts.append(
                    "This is a very small gap (and mask scores may even "
                    "slightly exceed box scores on the strict metric), "
                    "which means the predicted outlines are at least as "
                    "tight as the predicted rectangles — a strong sign "
                    "that the segmentation head is working well."
                )
            else:
                parts.append(
                    "This is a typical gap for this kind of model: mask "
                    "metrics are always held to a stricter standard than "
                    "box metrics, so a small deficit is expected and "
                    "acceptable."
                )
            interp_lines.append(''.join(parts))
    interp_text = ' '.join(interp_lines)

    return (
        '<section class="page">'
        f'<h1 class="page-title">{_html_escape(title)}</h1>'
        f'<div class="subtitle">{_html_escape(sub)}</div>'
        '<h2 class="section">Summary</h2>'
        f'{summary_tbl}'
        '<h2 class="section">Curves</h2>'
        '<div class="row">'
        '<div class="col left">'
        '<h3 class="subsection">Precision-recall curve</h3>'
        f'{pr_html}'
        f'{pr_axis}'
        '</div>'
        '<div class="col right">'
        '<h3 class="subsection">F1 vs confidence curve</h3>'
        f'{f1_html}'
        f'{f1_axis}'
        '</div>'
        '</div>'
        f'<div class="interp"><span class="lead">Interpretation. </span>{_html_escape(interp_text)}</div>'
        '</section>'
    )


def _render_perclass(metrics, ctx, args, figures):
    pc = metrics.get('per_class') or {}
    inst = metrics.get('per_class_instances') or {}
    if pc:
        ordered = sorted(
            pc.items(),
            key=lambda kv: (kv[1].get('mask_map50_95') if kv[1].get('mask_map50_95') is not None else -1),
            reverse=True,
        )
    else:
        ordered = []

    rows_html = []
    for cls, d in ordered:
        rows_html.append(
            '<tr>'
            f'<td class="mono">{_html_escape(cls)}</td>'
            f'<td class="num">{_html_escape(inst.get(cls, "?"))}</td>'
            f'<td class="num">{_html_escape(fmt_num(d.get("mask_precision")))}</td>'
            f'<td class="num">{_html_escape(fmt_num(d.get("mask_recall")))}</td>'
            f'<td class="num">{_html_escape(fmt_num(d.get("mask_map50")))}</td>'
            f'<td class="num">{_html_escape(fmt_num(d.get("mask_map50_95")))}</td>'
            '</tr>'
        )
    if not rows_html:
        rows_html.append(
            '<tr><td colspan="6" class="dim">[no per-class data available]</td></tr>'
        )
    pc_tbl = (
        '<table class="tight">'
        '<tr><th>Class</th><th>Inst</th><th>P</th><th>R</th>'
        '<th>mAP50</th><th>mAP50-95</th></tr>'
        + ''.join(rows_html)
        + '</table>'
    )

    bar_img = figures.get('per_class_bar')
    bar_html = (
        f'<div class="fig-wrap"><img class="fig" src="{_img_src(bar_img)}" alt="Per-class mAP50-95 bar chart"/></div>'
        if bar_img and os.path.isfile(bar_img)
        else '<div class="dim">[per-class bar chart unavailable]</div>'
    )

    axis_html = _axis_block_html('per_class_bar')

    callout_html = ''
    ordered_mm = [(k, v.get('mask_map50_95')) for k, v in pc.items()
                  if v.get('mask_map50_95') is not None]
    if ordered_mm:
        ordered_mm.sort(key=lambda kv: kv[1], reverse=True)
        top = ordered_mm[0]
        bot = ordered_mm[-1]
        headline = (
            f"The strongest class on this run is '{top[0]}', with a strict "
            f"mask score (mAP50-95) of {top[1]:.3f}; the weakest is "
            f"'{bot[0]}', with a strict mask score of {bot[1]:.3f}."
        )
        hint = (
            "Before concluding the model is failing on a weak class, check "
            "three things in order. First, count how many training examples "
            "of that class exist — few examples produce noisy scores. "
            "Second, look at the confusion-matrix page (page 6) to see "
            "which other species the model is confusing it with. Third, "
            "spot-check the human-drawn ground-truth (GT) outlines for "
            "that class in the sample pages to confirm the labels "
            "themselves are clean."
        )
        callout_html = (
            '<div class="callout">'
            f'<div class="headline">{_html_escape(headline)}</div>'
            f'<div style="margin-top:2pt;">{_html_escape(hint)}</div>'
            '</div>'
        )

    return (
        '<section class="page">'
        '<h1 class="page-title">Page 5 — Per-class breakdown</h1>'
        '<div class="subtitle">'
        "Precision (P), recall (R), and mean average precision (mAP) scores "
        "broken out by coral species, computed on the segmentation masks "
        "(the pixel outlines). The 'Inst' column counts how many instances "
        "of that class are present in the evaluation split."
        '</div>'
        '<div class="row">'
        '<div class="col col-55">'
        '<h3 class="subsection">Per-class table</h3>'
        f'{pc_tbl}'
        '</div>'
        '<div class="col col-43">'
        '<h3 class="subsection">mAP50-95 per class (sorted)</h3>'
        f'{bar_html}'
        '</div>'
        '</div>'
        f'{axis_html}'
        f'{callout_html}'
        '</section>'
    )


def _render_confusion(metrics, ctx, args, figures):
    norm_fig = figures.get('confusion_norm')
    abs_fig = figures.get('confusion_abs')

    def _fig(path, alt):
        if path and os.path.isfile(path):
            return f'<div class="fig-wrap"><img class="fig" src="{_img_src(path)}" alt="{_html_escape(alt)}"/></div>'
        return f'<div class="dim">[{_html_escape(alt)} unavailable]</div>'

    norm_html = _fig(norm_fig, 'confusion_matrix_normalized.png')
    abs_html = _fig(abs_fig, 'confusion_matrix.png')
    norm_axis = _axis_block_html('confusion_norm')
    abs_axis = _axis_block_html('confusion_abs')

    reading = [
        "A cell in the 'background' row under a coral class is a false "
        "positive: the model drew a coral where the human labeler marked "
        "empty background.",
        "A cell in a coral row under the 'background' column is a missed "
        "detection: a real coral the human labeled that the model failed "
        "to find.",
        "A cell off the diagonal between two coral classes is a species "
        "confusion: the model found the coral but assigned it the wrong "
        "species.",
    ]
    reading_html = (
        '<ul class="reading-notes">'
        + ''.join(f'<li>{_html_escape(t)}</li>' for t in reading)
        + '</ul>'
    )

    return (
        '<section class="page">'
        '<h1 class="page-title">Page 6 — Confusion matrix</h1>'
        '<div class="subtitle">'
        "Rows are the true (human-labeled, ground-truth or GT) class. "
        "Columns are the class the model predicted. Cells on the diagonal "
        "are correct predictions; every cell off the diagonal is a mistake."
        '</div>'
        '<div class="row">'
        '<div class="col left">'
        '<h3 class="subsection">Normalized (rows sum to 1.0)</h3>'
        f'{norm_html}'
        f'{norm_axis}'
        '</div>'
        '<div class="col right">'
        '<h3 class="subsection">Absolute (raw prediction counts)</h3>'
        f'{abs_html}'
        f'{abs_axis}'
        '</div>'
        '</div>'
        '<h3 class="subsection">How to read off-diagonal cells</h3>'
        f'{reading_html}'
        '</section>'
    )


def _render_samples(previews, args, start_page):
    """Render sample images 4-per-page in a 2x2 grid."""
    if not previews:
        return [], start_page
    pages_html = []
    pg = start_page
    for chunk_idx in range(0, len(previews), 4):
        chunk = previews[chunk_idx:chunk_idx + 4]
        tiles = []
        for p in chunk:
            fn = p.get('filename', '')
            n_gt = p.get('n_gt', '?')
            n_pred = p.get('n_pred', '?')
            overlay = p.get('overlay')
            img_tag = (
                f'<img src="{_img_src(overlay)}" alt="{_html_escape(fn)}"/>'
                if overlay and os.path.isfile(overlay)
                else '<div class="dim">[image missing]</div>'
            )
            tiles.append(
                '<div class="sample-tile">'
                f'{img_tag}'
                f'<div class="sample-caption">'
                f'<span class="fn">{_html_escape(fn)}</span><br/>'
                f'Ground truth: {_html_escape(n_gt)} human-drawn instance(s). '
                f'Prediction at conf &ge; {args.conf:.2f}: {_html_escape(n_pred)} detection(s).'
                '</div>'
                '</div>'
            )
        # pad to 4 so grid stays 2x2
        while len(tiles) < 4:
            tiles.append('<div class="sample-tile" style="border:none;"></div>')
        grid = (
            '<div class="sample-grid">'
            f'<div class="sample-row">{tiles[0]}{tiles[1]}</div>'
            f'<div class="sample-row">{tiles[2]}{tiles[3]}</div>'
            '</div>'
        )
        last_idx = chunk_idx + len(chunk)
        title = (
            f'Page {pg} — Prediction samples '
            f'({chunk_idx + 1}-{last_idx} of {len(previews)})'
        )
        sub = (
            "In each tile the left panel shows the image overlaid with "
            "the human-drawn ground-truth (GT) outlines, and the right "
            "panel shows the same image overlaid with the model's "
            f"predictions at a confidence threshold of {args.conf:.2f} "
            "or higher."
        )
        pages_html.append(
            '<section class="page">'
            f'<h1 class="page-title">{_html_escape(title)}</h1>'
            f'<div class="subtitle">{_html_escape(sub)}</div>'
            f'{grid}'
            '</section>'
        )
        pg += 1
    return pages_html, pg


def _render_glossary(metrics, ctx, args, page_num, now):
    entries = []
    for term, definition, good, red_flag in GLOSSARY_TABLE:
        entries.append(
            '<div class="entry">'
            f'<span class="term">{_html_escape(term)}</span>'
            f'<span class="defn">{_html_escape(definition)}</span><br/>'
            f'<span class="good">Good: {_html_escape(good)}</span><br/>'
            f'<span class="red">Red flag: {_html_escape(red_flag)}</span>'
            '</div>'
        )
    glossary_html = '<div class="glossary">' + ''.join(entries) + '</div>'

    prov_rows = [
        ('weights', metrics.get('weights', '')),
        ('data yaml', metrics.get('data_yaml', '')),
        ('split', args.split),
        ('imgsz / conf / iou', f'{args.imgsz} / {args.conf} / {args.iou}'),
        ('generated', now),
        ('eval wall', fmt_dur(metrics.get('wall_seconds'))),
    ]
    prov_body = ''.join(
        f'<tr><td class="key">{_html_escape(k)}</td>'
        f'<td class="val">{_html_escape(v)}</td></tr>'
        for k, v in prov_rows
    )
    prov_tbl = f'<table class="kv">{prov_body}</table>'

    return (
        '<section class="page last">'
        f'<h1 class="page-title">Page {page_num} — Glossary</h1>'
        '<div class="subtitle">'
        "Every term and abbreviation used in this report is defined below "
        "as a complete sentence. Each entry includes a healthy value band "
        "and the symptom to watch for when that value drops too low."
        '</div>'
        f'{glossary_html}'
        '<h2 class="section">Provenance</h2>'
        f'{prov_tbl}'
        '</section>'
    )


def _collect_figures(ctx, args, out_dir):
    """Generate/resolve every PNG the HTML will reference. Returns dict of
    logical_name -> absolute PNG path (or None if not available)."""
    figures = {}
    # Training history PNG we generate via matplotlib helper (kept).
    csv_path = os.path.join(ctx['run_dir'], 'results.csv')
    hist_path = os.path.join(out_dir, '_pdfplots_training.png')
    try:
        figures['training_history'] = _plot_training_history(csv_path, hist_path)
    except Exception as e:
        print(f'[eval] training history plot failed: {e}')
        figures['training_history'] = None

    # Per-class bar chart PNG (kept helper).
    pc = {}  # filled by build_pdf before calling this
    # Resolve ultralytics plots (val dir overrides train dir).
    vp = _find_val_plots(out_dir)
    tp = _find_train_plots(ctx['run_dir'])
    merged = {**tp, **vp}
    figures['box_pr'] = merged.get('BoxPR_curve.png')
    figures['mask_pr'] = merged.get('MaskPR_curve.png')
    figures['box_f1'] = merged.get('BoxF1_curve.png')
    figures['mask_f1'] = merged.get('MaskF1_curve.png')
    figures['confusion_norm'] = merged.get('confusion_matrix_normalized.png')
    figures['confusion_abs'] = merged.get('confusion_matrix.png')
    return figures


def build_pdf(report_md, metrics, ctx, previews, args, pdf_path):
    """Render the evaluation report as a dense WeasyPrint PDF.

    Page layout (STABLE across runs):
        1: Cover + 4-quadrant top-line + TOC
        2: Training history (config + loss curves + convergence prose)
        3: Box metrics (summary + PR + F1 + interpretation)
        4: Mask metrics (same layout + box-vs-mask delta)
        5: Per-class (table + bar chart + strongest/weakest callout)
        6: Confusion matrix (normalized + absolute)
        7..N: Prediction samples, 4 per page in 2x2 grid
        last: Glossary (2-column CSS + provenance)
    """
    from weasyprint import HTML, CSS

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    run_name = os.path.basename(ctx['run_dir'].rstrip('/'))
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Generate / collect figures.
    figures = _collect_figures(ctx, args, out_dir)
    pc_data = metrics.get('per_class') or {}
    if pc_data:
        bar_path = os.path.join(out_dir, '_pdfplots_perclass.png')
        try:
            figures['per_class_bar'] = _plot_per_class_bar(
                pc_data, bar_path, metric_key='mask_map50_95')
        except Exception as e:
            print(f'[eval] per-class bar plot failed: {e}')
            figures['per_class_bar'] = None
    else:
        figures['per_class_bar'] = None

    # TOC assembly. Samples are grouped 4-per-page (was 2-per-page before).
    n_sample_pages = (len(previews) + 3) // 4 if previews else 0
    toc = [
        ('Cover + top-line', 1),
        ('Training history', 2),
        ('Box metrics', 3),
        ('Mask metrics', 4),
        ('Per-class breakdown', 5),
        ('Confusion matrix', 6),
    ]
    cur = 7
    if n_sample_pages > 0:
        toc.append((f'Prediction samples ({len(previews)} imgs)', cur))
        cur += n_sample_pages
    glossary_page = cur
    toc.append(('Glossary cheatsheet', glossary_page))

    # Compose HTML.
    parts = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>Evaluation report — {_html_escape(run_name)}</title>'
        '</head><body>',
    ]
    parts.append(_render_cover(metrics, ctx, args, toc, now, run_name))
    parts.append(_render_training(metrics, ctx, args, figures))
    parts.append(_render_metric_page('box', 3, metrics, ctx, args, figures))
    parts.append(_render_metric_page('mask', 4, metrics, ctx, args, figures))
    parts.append(_render_perclass(metrics, ctx, args, figures))
    parts.append(_render_confusion(metrics, ctx, args, figures))
    sample_pages, next_pg = _render_samples(previews, args, start_page=7)
    parts.extend(sample_pages)
    parts.append(_render_glossary(metrics, ctx, args, glossary_page, now))
    parts.append('</body></html>')

    html_str = ''.join(parts)
    css_str = _build_css()

    # Write the HTML alongside the PDF for debugging / inspection.
    try:
        with open(os.path.join(out_dir, 'report.html'), 'w') as f:
            f.write(html_str)
    except Exception:
        pass

    HTML(string=html_str, base_url=os.path.abspath(out_dir)).write_pdf(
        pdf_path,
        stylesheets=[CSS(string=css_str)],
    )
    print(f'[eval] PDF written via WeasyPrint: {pdf_path}')


# ─────────────────────────────────────────────────────────────────────────
# Legacy helpers removed: matplotlib-based _table(), _embed_image(),
# _axis_block(), _page_title(), _footer, _wrap_cell(), _ellipsize() etc.
# were replaced by pure HTML+CSS rendering above.
# ─────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    args = arg_parse()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[eval] run_dir={args.run_dir}')
    print(f'[eval] dataset_dir={args.dataset_dir}')
    print(f'[eval] split={args.split} imgsz={args.imgsz} conf={args.conf} iou={args.iou}')

    val_out = run_val(args)
    metrics = extract_metrics(val_out)
    ctx = gather_context(args.run_dir, args.dataset_dir)

    previews = render_previews(
        val_out['model'], args.dataset_dir, args.split,
        args.out_dir, args.preview_count, args.conf, args.iou, args.imgsz,
    )

    report_md = build_report_md(metrics, ctx, previews, args)
    md_path = os.path.join(args.out_dir, 'report.md')
    json_path = os.path.join(args.out_dir, 'metrics.json')
    pdf_path = os.path.join(args.out_dir, 'report.pdf')
    with open(md_path, 'w') as f:
        f.write(report_md)
    with open(json_path, 'w') as f:
        json.dump({'metrics': metrics, 'context': ctx, 'previews': previews,
                   'args': vars(args), 'generated_at': datetime.now().isoformat()},
                  f, indent=2, default=str)

    print('[eval] building PDF...')
    try:
        build_pdf(report_md, metrics, ctx, previews, args, pdf_path)
    except Exception as e:
        traceback.print_exc()
        print(f'[eval] PDF build failed: {e}')
        pdf_path = None

    if pdf_path and args.pdf_export_dir:
        try:
            os.makedirs(args.pdf_export_dir, exist_ok=True)
            dest = os.path.join(args.pdf_export_dir,
                                f'{os.path.basename(ctx["run_dir"])}_report.pdf')
            shutil.copy2(pdf_path, dest)
            print(f'[eval] PDF also exported to {dest}')
        except Exception as e:
            print(f'[eval] external PDF copy failed: {e}')

    print('[eval] done.')
    print(f'[eval] md={md_path}')
    print(f'[eval] json={json_path}')
    print(f'[eval] pdf={pdf_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
