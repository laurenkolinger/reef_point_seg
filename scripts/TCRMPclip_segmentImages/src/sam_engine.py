"""
SAM3 engine wrapper — handles model loading and all segmentation operations
(click, box, exemplar, refinement).

Uses HuggingFace Transformers API (facebook/sam3):
  - Sam3TrackerModel + Sam3TrackerProcessor for point/box prompts
  - Sam3Model + Sam3Processor for exemplar (visual) scanning

Checkpoint note: facebook/sam3 ships as a `sam3_video` checkpoint. The
single-image point/box tracker weights are nested under `tracker_model.*`,
so a plain Sam3TrackerModel.from_pretrained() only partially loads and leaves
the prompt encoder + mask decoder at random init (degenerate full-image/empty
masks, IoU pinned ~0.5). _load_image_tracker() backfills those nested weights.
The exemplar/detector model (Sam3Model) loads cleanly and needs no backfill.
"""

import hashlib
import logging
import os

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

# How the most recent _load_image_tracker() call obtained full tracker weights:
#   "flat"           checkpoint layout was already flat, nothing to backfill
#   "cache"          backfill weights read from the module-local cache file
#   "video_backfill" full Sam3VideoModel loaded on CPU to lift nested weights
# Tests and boot logs read this to prove the cache path is actually taken.
LAST_TRACKER_LOAD_PATH = None

# Bump when the cache payload layout changes; stale files are simply ignored
# (the fingerprint no longer matches) and rewritten on the next backfill.
_TRACKER_CACHE_VERSION = 1


def _hf_load(cls, *args, **kwargs):
    """Offline-first from_pretrained. The lab network's DNS drops out
    intermittently, and transformers raises instead of falling back to the
    local cache when its online probe fails mid-load — which killed the
    annotator at boot (2026-08-13) even with all of facebook/sam3 cached.
    Try the cache first; touch the network only when something is genuinely
    missing (fresh machine, first download). The cache-only flag is merged,
    not passed positionally, so a caller passing local_files_only itself can
    never raise TypeError and silently defeat the offline-first attempt."""
    try:
        return cls.from_pretrained(*args, **{**kwargs, "local_files_only": True})
    except Exception as exc:
        log.warning("Cache-only load failed for %s (%s); retrying online.",
                    cls.__name__, exc)
        return cls.from_pretrained(*args, **kwargs)


def _tracker_cache_dir():
    """Module-local cache dir for the merged tracker weights. Lives under
    supporting_data/ (already gitignored, like the model_weights it sits
    beside). Overridable for tests via TCRMP_SAM3_CACHE_DIR."""
    override = os.environ.get("TCRMP_SAM3_CACHE_DIR")
    if override:
        return override
    src = os.path.dirname(os.path.abspath(__file__))          # .../<app>/src
    repo = os.path.dirname(os.path.dirname(os.path.dirname(src)))
    return os.path.join(repo, "supporting_data", "sam3_cache")


def _fingerprint_from_stats(snapshot_id, weight_stats, dtype):
    """Pure cache-key derivation: sha256 over the resolved checkpoint snapshot
    id (the HF commit hash directory name), each weight file's (name, size,
    mtime_ns), the requested dtype, and the cache schema version. Any checkpoint
    update, re-download, or dtype change yields a new key, so a stale cache can
    never be loaded silently."""
    h = hashlib.sha256()
    h.update(f"v{_TRACKER_CACHE_VERSION}|{snapshot_id}|{dtype}".encode())
    for name, size, mtime_ns in sorted(weight_stats):
        h.update(f"|{name}:{size}:{mtime_ns}".encode())
    return h.hexdigest()[:16]


def _checkpoint_fingerprint(dtype):
    """Fingerprint the locally-cached facebook/sam3 checkpoint. Returns a hex
    key, or None when the snapshot cannot be resolved offline (then the cache
    is simply skipped; the video backfill path still works)."""
    try:
        from huggingface_hub import snapshot_download
        snap = snapshot_download("facebook/sam3", local_files_only=True)
        stats = []
        for fn in sorted(os.listdir(snap)):
            if fn.endswith((".safetensors", ".bin")) or fn == "config.json":
                st = os.stat(os.path.join(snap, fn))
                stats.append((fn, st.st_size, st.st_mtime_ns))
        if not stats:
            return None
        return _fingerprint_from_stats(os.path.basename(snap), stats, dtype)
    except Exception as exc:
        log.warning("SAM3 checkpoint fingerprint unavailable (%s); "
                    "tracker weight cache disabled for this boot.", exc)
        return None


def _load_cached_backfill(cache_path, missing):
    """Load the cached backfill state_dict if it covers every missing key.
    Returns the state_dict or None (corrupt/stale cache is deleted)."""
    try:
        sd = torch.load(cache_path, map_location="cpu", weights_only=True)
        if missing <= set(sd.keys()):
            return sd
        log.warning("SAM3 tracker cache %s does not cover %d missing keys; "
                    "rebuilding.", cache_path, len(missing - set(sd.keys())))
    except Exception as exc:
        log.warning("SAM3 tracker cache %s unreadable (%s); rebuilding.",
                    cache_path, exc)
    try:
        os.remove(cache_path)
    except OSError:
        pass
    return None


def _write_cached_backfill(cache_path, sd, missing):
    """Persist only the backfilled (originally-missing) keys — ~9 MB instead of
    the full multi-GB tracker state. Atomic write so a killed boot can never
    leave a truncated cache behind."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {k: v for k, v in sd.items() if k in missing}
        tmp = cache_path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, cache_path)
        log.info("SAM3 tracker backfill cached to %s (%d keys).",
                 cache_path, len(payload))
    except Exception as exc:
        log.warning("Could not write SAM3 tracker cache %s: %s", cache_path, exc)


def _load_image_tracker(dtype):
    """Build the single-image SAM3 point/box tracker with fully-loaded weights.

    facebook/sam3 is a `sam3_video` checkpoint whose tracker weights live under
    `tracker_model.*`. Sam3TrackerModel.from_pretrained() therefore loads the
    vision encoder but leaves the prompt encoder + mask decoder at random init.
    We build the standalone tracker (correct single-image forward + processor
    interface), then backfill the nested weights from the video checkpoint and
    fail loudly if any originally-missing weight is left unrecovered.

    The backfill result is cached (Task 1.2, 2026-08-26): the first boot
    persists the recovered keys to a module-local file keyed by the checkpoint
    fingerprint; later boots load that file directly and skip the CPU-side
    Sam3VideoModel load entirely (the slow part on a cold page cache: a second
    multi-GB safetensors read plus full video-model construction).
    """
    global LAST_TRACKER_LOAD_PATH
    from transformers import Sam3TrackerModel, Sam3VideoModel

    model, info = _hf_load(
        Sam3TrackerModel, "facebook/sam3", torch_dtype=dtype,
        output_loading_info=True,
    )
    missing = set(info.get("missing_keys", []))
    if not missing:
        LAST_TRACKER_LOAD_PATH = "flat"
        return model  # checkpoint layout already flat; nothing to backfill

    key = _checkpoint_fingerprint(dtype)
    cache_path = (os.path.join(_tracker_cache_dir(), f"tracker_backfill_{key}.pt")
                  if key else None)

    if cache_path and os.path.isfile(cache_path):
        sd = _load_cached_backfill(cache_path, missing)
        if sd is not None:
            model.load_state_dict(sd, strict=False)
            LAST_TRACKER_LOAD_PATH = "cache"
            log.info("SAM3 tracker weights backfilled from cache %s "
                     "(%d keys; video-model load skipped).",
                     cache_path, len(missing))
            return model

    # Load the video checkpoint on CPU just to lift the tracker submodule.
    video = _hf_load(Sam3VideoModel, "facebook/sam3", torch_dtype=dtype)
    prefix = "tracker_model."
    sd = {
        k[len(prefix):]: v
        for k, v in video.state_dict().items()
        if k.startswith(prefix)
    }
    del video
    model.load_state_dict(sd, strict=False)

    unrecovered = missing - set(sd.keys())
    if unrecovered:
        raise RuntimeError(
            "SAM3 tracker weight load incomplete: %d weights left at random "
            "init (e.g. %s). The facebook/sam3 checkpoint layout may have "
            "changed." % (len(unrecovered), sorted(unrecovered)[:3])
        )
    LAST_TRACKER_LOAD_PATH = "video_backfill"
    log.info("SAM3 tracker weights backfilled from sam3_video checkpoint "
             "(%d nested keys recovered).", len(missing))
    if cache_path:
        _write_cached_backfill(cache_path, sd, missing)
    return model


class SAM3Engine:
    """SAM3 wrapper using HuggingFace Transformers (facebook/sam3)."""

    def __init__(self, config):
        self.device_tracker = config.SAM3_DEVICE_TRACKER   # for point/box
        self.device_exemplar = config.SAM3_DEVICE_EXEMPLAR  # for exemplar scan
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.mask_size = config.SAM3_MASK_SIZE  # tight/medium/generous

        # Tracker model (point/box prompts)
        log.info("Loading SAM3 tracker model on %s...", self.device_tracker)
        from transformers import Sam3TrackerProcessor

        self._tracker_proc = _hf_load(Sam3TrackerProcessor, "facebook/sam3")
        self._tracker = _load_image_tracker(torch.bfloat16).to(self.device_tracker)
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
        self._text_proc = _hf_load(Sam3Processor, "facebook/sam3")
        self._text_model = _hf_load(Sam3Model, "facebook/sam3").to(self.device_exemplar)
        self._text_model.eval()
        log.info("Exemplar model ready.")
