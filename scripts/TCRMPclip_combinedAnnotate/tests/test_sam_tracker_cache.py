"""Tests for the cached SAM3 tracker backfill (Task 1.2, 2026-08-26).

The merged tracker weights (the keys Sam3TrackerModel.from_pretrained leaves
at random init) are persisted to a module-local cache keyed by the checkpoint
fingerprint. First boot: video-model backfill + cache write. Later boots: cache
read, NO Sam3VideoModel construction. These tests drive _load_image_tracker
with fake loaders (no real multi-GB model load), so they run in the normal
suite. Set TCRMP_SAM3_REAL_PARITY=1 to also run the real-checkpoint mask
parity check (loads SAM3 twice; used for the recorded before/after numbers).

Run: env/bin/python .../tests/test_sam_tracker_cache.py
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

CACHE_DIR = tempfile.mkdtemp(prefix="sam3_cache_test_")
os.environ["TCRMP_SAM3_CACHE_DIR"] = CACHE_DIR

import torch
import sam_engine as S

_fail = 0
def check(c, m):
    global _fail
    if not c: _fail += 1; print(f"  FAIL {m}")
    else: print(f"  PASS {m}")


# ── Pure fingerprint derivation ─────────────────────────────────────
stats = [("model.safetensors", 3300, 111), ("config.json", 10, 222)]
k1 = S._fingerprint_from_stats("abc123", stats, torch.bfloat16)
k2 = S._fingerprint_from_stats("abc123", list(reversed(stats)), torch.bfloat16)
check(k1 == k2, "fingerprint is order-independent and deterministic")
check(k1 != S._fingerprint_from_stats("abc124", stats, torch.bfloat16),
      "fingerprint changes with snapshot id")
check(k1 != S._fingerprint_from_stats("abc123",
      [("model.safetensors", 3300, 999), ("config.json", 10, 222)],
      torch.bfloat16),
      "fingerprint changes with weight-file mtime")
check(k1 != S._fingerprint_from_stats("abc123", stats, torch.float32),
      "fingerprint changes with dtype")

# ── Fake loaders: cache write on first load, cache hit on second ────
MISSING = ["prompt_encoder.w", "mask_decoder.w"]
TRACKER_SD = {"prompt_encoder.w": torch.tensor([1.0, 2.0]),
              "mask_decoder.w": torch.tensor([3.0]),
              "vision_encoder.w": torch.tensor([4.0])}

class FakeTracker:
    def __init__(self):
        self.loaded = {}
    def load_state_dict(self, sd, strict=True):
        self.loaded.update(sd)

class FakeVideo:
    constructed = 0
    def __init__(self):
        FakeVideo.constructed += 1
    def state_dict(self):
        return {"tracker_model." + k: v for k, v in TRACKER_SD.items()}

def fake_hf_load(cls, *args, **kwargs):
    name = getattr(cls, "__name__", "")
    if name == "Sam3TrackerModel":
        return FakeTracker(), {"missing_keys": list(MISSING)}
    if name == "Sam3VideoModel":
        return FakeVideo()
    raise AssertionError(f"unexpected load of {name}")

real_hf_load = S._hf_load
real_fingerprint = S._checkpoint_fingerprint
S._hf_load = fake_hf_load
S._checkpoint_fingerprint = lambda dtype: "deadbeefcafe0000"

try:
    FakeVideo.constructed = 0
    m1 = S._load_image_tracker(torch.bfloat16)
    check(S.LAST_TRACKER_LOAD_PATH == "video_backfill",
          f"first load path is video_backfill (got {S.LAST_TRACKER_LOAD_PATH})")
    check(FakeVideo.constructed == 1, "first load constructs the video model once")
    cache_path = os.path.join(CACHE_DIR, "tracker_backfill_deadbeefcafe0000.pt")
    check(os.path.isfile(cache_path), "cache file written on first load")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    check(set(payload.keys()) == set(MISSING),
          f"cache holds exactly the missing keys (got {sorted(payload)})")

    m2 = S._load_image_tracker(torch.bfloat16)
    check(S.LAST_TRACKER_LOAD_PATH == "cache",
          f"second load path is cache (got {S.LAST_TRACKER_LOAD_PATH})")
    check(FakeVideo.constructed == 1,
          "second load does NOT construct the video model")
    parity = all(torch.equal(m2.loaded[k], TRACKER_SD[k]) for k in MISSING)
    check(parity, "cached weights identical to video-backfill weights")

    # Corrupt cache falls back to backfill and rewrites.
    with open(cache_path, "wb") as f:
        f.write(b"not a torch file")
    m3 = S._load_image_tracker(torch.bfloat16)
    check(S.LAST_TRACKER_LOAD_PATH == "video_backfill",
          "corrupt cache falls back to video backfill")
    check(os.path.isfile(cache_path)
          and set(torch.load(cache_path, map_location="cpu",
                             weights_only=True).keys()) == set(MISSING),
          "corrupt cache rewritten with a good payload")

    # Cache that does not cover the missing set is treated as stale.
    torch.save({"prompt_encoder.w": torch.tensor([9.0])}, cache_path)
    m4 = S._load_image_tracker(torch.bfloat16)
    check(S.LAST_TRACKER_LOAD_PATH == "video_backfill",
          "under-covering cache treated as stale (video backfill)")

    # No fingerprint (offline snapshot unresolvable): backfill works, no cache IO.
    S._checkpoint_fingerprint = lambda dtype: None
    shutil.rmtree(CACHE_DIR); os.makedirs(CACHE_DIR)
    m5 = S._load_image_tracker(torch.bfloat16)
    check(S.LAST_TRACKER_LOAD_PATH == "video_backfill" and not os.listdir(CACHE_DIR),
          "missing fingerprint disables cache but load still succeeds")
finally:
    S._hf_load = real_hf_load
    S._checkpoint_fingerprint = real_fingerprint

# ── Real fingerprint on this box resolves (offline) ─────────────────
fp = S._checkpoint_fingerprint(torch.bfloat16)
check(fp is None or (isinstance(fp, str) and len(fp) == 16),
      f"real fingerprint is a 16-hex key or None (got {fp!r})")

# ── Optional: real-checkpoint mask parity (TCRMP_SAM3_REAL_PARITY=1) ─
if os.environ.get("TCRMP_SAM3_REAL_PARITY") == "1":
    import numpy as np
    from PIL import Image

    class Cfg:
        SAM3_DEVICE_TRACKER = "cuda:0" if torch.cuda.is_available() else "cpu"
        SAM3_DEVICE_EXEMPLAR = SAM3_DEVICE_TRACKER
        CONFIDENCE_THRESHOLD = 0.5
        SAM3_MASK_SIZE = "generous"

    img_path = os.path.join(CACHE_DIR, "fixture.jpg")
    rng = np.random.default_rng(7)
    arr = (rng.random((256, 256, 3)) * 60).astype("uint8")
    arr[64:192, 64:192] = [200, 60, 60]  # bright square to segment
    Image.fromarray(arr).save(img_path, "JPEG", quality=95)

    e1 = S.SAM3Engine(Cfg())  # empty cache dir: video_backfill + cache write
    p1 = S.LAST_TRACKER_LOAD_PATH
    e1.set_image(img_path)
    r1 = e1.segment_point(128, 128)
    e1.release()

    e2 = S.SAM3Engine(Cfg())  # cache hit
    p2 = S.LAST_TRACKER_LOAD_PATH
    e2.set_image(img_path)
    r2 = e2.segment_point(128, 128)
    e2.release()

    check(p1 == "video_backfill" and p2 == "cache",
          f"real load paths fresh={p1} cached={p2}")
    check(r1 is not None and r2 is not None, "both engines return a mask")
    if r1 is not None and r2 is not None:
        check(np.array_equal(r1["mask"], r2["mask"]),
              "cached-engine mask identical to fresh-engine mask")
        check(abs(r1["score"] - r2["score"]) < 1e-6,
              f"scores match ({r1['score']:.4f} vs {r2['score']:.4f})")
else:
    print("  SKIP real-checkpoint parity (set TCRMP_SAM3_REAL_PARITY=1)")

shutil.rmtree(CACHE_DIR, ignore_errors=True)
check(not os.path.isdir(CACHE_DIR), "test cache dir removed")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
