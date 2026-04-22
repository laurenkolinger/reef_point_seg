"""
SAM3 engine wrapper — handles model loading and all segmentation operations
(click, box, exemplar, refinement).

Uses HuggingFace Transformers API (facebook/sam3):
  - Sam3TrackerModel + Sam3TrackerProcessor for point/box prompts
  - Sam3Model + Sam3Processor for exemplar (visual) scanning

Reference: /mnt/rip/vicarius_drive/hopper/ai_sam3_friday13march26/scripts/
"""

import logging
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)


class SAM3Engine:
    """SAM3 wrapper using HuggingFace Transformers (facebook/sam3)."""

    def __init__(self, config):
        self.device_tracker = config.SAM3_DEVICE_TRACKER   # for point/box
        self.device_exemplar = config.SAM3_DEVICE_EXEMPLAR  # for exemplar scan
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.mask_size = config.SAM3_MASK_SIZE  # tight/medium/generous

        # Tracker model (point/box prompts)
        log.info("Loading SAM3 tracker model on %s...", self.device_tracker)
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self._tracker_proc = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
        self._tracker = Sam3TrackerModel.from_pretrained(
            "facebook/sam3", torch_dtype=torch.bfloat16
        ).to(self.device_tracker)
        self._tracker.eval()
        log.info("Tracker ready. GPU mem: %.2f GB",
                 torch.cuda.memory_allocated(int(self.device_tracker[-1])) / 1e9)

        # Exemplar model (visual scan) — loaded lazily on first use
        self._text_model = None
        self._text_proc = None

        # Current image cache (for the tracker — no persistent embedding,
        # but we cache the PIL image to avoid re-reading from disk)
        self._current_image_path = None
        self._current_image = None
        self._image_size = None  # (width, height)

    # ── Image loading ───────────────────────────────────────────────

    def set_image(self, image_path):
        """Load and cache image. Returns (width, height).
        The tracker model processes the full image on each call (no persistent
        embedding like SAM2), but caching avoids repeated disk reads."""
        if self._current_image_path == image_path and self._current_image is not None:
            return self._image_size

        self._current_image = Image.open(image_path).convert("RGB")
        self._image_size = (self._current_image.width, self._current_image.height)
        self._current_image_path = image_path

        log.info("Loaded image %s (%dx%d)", image_path, *self._image_size)
        return self._image_size

    # ── Single point segmentation ───────────────────────────────────

    def segment_point(self, x, y, label=1):
        """Segment using a single click. label=1=foreground, label=0=background.
        Returns dict with 'mask' (bool np array), 'score', 'bbox' or None."""
        return self._point_prompt([(x, y)], [label])

    # ── Refinement with accumulated clicks ──────────────────────────

    def refine_mask(self, clicks, labels):
        """Refine with multiple positive/negative clicks.
        clicks: list of (x, y) tuples
        labels: list of ints (1=positive, 0=negative)
        Returns result dict or None."""
        return self._point_prompt(clicks, labels)

    # ── Box prompt (redraw from scratch) ────────────────────────────

    def segment_box(self, x_min, y_min, x_max, y_max):
        """Segment using a bounding box prompt. Returns result dict or None."""
        assert self._current_image is not None, "Call set_image() first"

        inputs = self._tracker_proc(
            images=self._current_image,
            input_boxes=[[[x_min, y_min, x_max, y_max]]],
            return_tensors="pt",
        ).to(self.device_tracker)

        with torch.no_grad():
            out = self._tracker(**inputs)

        masks = self._tracker_proc.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        binary = self._extract_mask(masks)
        if binary is None or binary.sum() == 0:
            return None

        return {
            'mask': binary.astype(bool),
            'score': 0.9,  # tracker doesn't return per-mask scores for box
            'bbox': None,
        }

    # ── Exemplar scan (visual exemplar from bbox) ───────────────────

    def exemplar_scan(self, bbox):
        """Use a bounding box as visual exemplar to find similar objects.
        Uses Sam3Model (text/exemplar model) with box as positive exemplar.
        Returns list of result dicts."""
        assert self._current_image is not None, "Call set_image() first"

        self._ensure_exemplar_model()

        x_min, y_min, x_max, y_max = bbox
        try:
            inputs = self._text_proc(
                images=self._current_image,
                input_boxes=[[[x_min, y_min, x_max, y_max]]],
                input_boxes_labels=[[1]],
                return_tensors="pt",
            ).to(self.device_exemplar)

            with torch.no_grad():
                out = self._text_model(**inputs)

            results = self._text_proc.post_process_instance_segmentation(
                out,
                threshold=self.confidence_threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            output = []
            scores = results.get("scores", [])
            for i, m in enumerate(results.get("masks", [])):
                binary = (m.cpu().float().numpy() > 0.5).astype(np.uint8)
                if binary.sum() < 200:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                output.append({
                    'mask': binary.astype(bool),
                    'score': score,
                    'bbox': None,
                })
            return output

        except Exception as e:
            log.warning("Exemplar scan failed: %s", e)
            return []

    # ── Cleanup ─────────────────────────────────────────────────────

    def release(self):
        """Free all GPU memory."""
        self._tracker = None
        self._tracker_proc = None
        self._text_model = None
        self._text_proc = None
        self._current_image = None
        self._current_image_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("SAM3 engine released.")

    # ── Internal ────────────────────────────────────────────────────

    def _point_prompt(self, points, labels):
        """Run point prompt via tracker model. Returns result dict or None."""
        assert self._current_image is not None, "Call set_image() first"

        # Format: 4 levels — [image_level][object_level][point_level][coords]
        # For single image, single object, N points:
        #   input_points=[[[ [x1,y1], [x2,y2], ... ]]]
        #   input_labels=[[[ l1, l2, ... ]]]
        pts_nested = [[[list(p) for p in points]]]
        lbl_nested = [[labels]]

        inputs = self._tracker_proc(
            images=self._current_image,
            input_points=pts_nested,
            input_labels=lbl_nested,
            return_tensors="pt",
        ).to(self.device_tracker)

        with torch.no_grad():
            out = self._tracker(**inputs)

        masks = self._tracker_proc.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"]
        )[0]

        binary = self._extract_mask(masks)
        if binary is None or binary.sum() == 0:
            return None

        # Tracker returns IoU scores per mask variant
        iou_scores = out.iou_scores
        score = 0.9  # default
        if iou_scores is not None:
            score = float(iou_scores.cpu().flatten()[-1])

        return {
            'mask': binary.astype(bool),
            'score': score,
            'bbox': None,
        }

    def _extract_mask(self, masks):
        """Extract the best binary mask from tracker output.
        Tracker returns 3 mask variants: tight(0), medium(1), generous(2)."""
        size_idx = {"tight": 0, "medium": 1, "generous": 2}
        idx = size_idx.get(self.mask_size, 2)

        if masks.ndim == 4 and masks.shape[1] >= 3:
            mn = masks[0, idx].float().numpy()
        elif masks.ndim == 4:
            mn = masks[0, -1].float().numpy()
        elif masks.ndim == 3:
            mn = masks[0].float().numpy()
        else:
            mn = masks.float().numpy()

        binary = (mn > 0.5).astype(np.uint8)
        return binary

    def _ensure_exemplar_model(self):
        """Lazily load the exemplar/text model on first use."""
        if self._text_model is not None:
            return
        log.info("Loading SAM3 exemplar model on %s...", self.device_exemplar)
        from transformers import Sam3Model, Sam3Processor
        self._text_proc = Sam3Processor.from_pretrained("facebook/sam3")
        self._text_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device_exemplar)
        self._text_model.eval()
        log.info("Exemplar model ready.")
