"""Manual GPU benchmark for the cached SAM3 prompt path (2026-08-26).

NOT registered in scripts/run_all_tests.sh and deliberately not named
test_*: it loads the real facebook/sam3 checkpoint and needs a free GPU, so
it is run by hand whenever sam_engine's caching behaviour changes. It reports,
for a single frame:

  first prompt    vision encoder + mask decoder, paid once per frame
  cached prompt   mask decoder only, what every later click on that frame costs
  uncached mean   what every click cost before the embedding cache existed
  parity          the cached-embedding mask equals the uncached mask
  exemplar        a cold scan versus a second scan on cached vision features

Device: TCRMP_BENCH_DEVICE, default cuda:0. cuda:1 usually hosts the live
annotator, so leave this on cuda:0 or another idle card.
Image:  TCRMP_BENCH_IMAGE, else the first raw jpeg found under any
        inprocess/run_*/step4test_combinedAnnotate/routed_input/raw/.
        The image is only ever read.

Exits 1 if the cached and uncached masks disagree, 0 otherwise.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/bench_sam3_prompts.py
"""
import glob
import os
import sys
import time

DEVICE = os.environ.get("TCRMP_BENCH_DEVICE", "cuda:0")
# config reads these at import time, so they must be set before it is imported.
os.environ["TCRMP_SAM3_DEVICE_TRACKER"] = DEVICE
os.environ["TCRMP_SAM3_DEVICE_EXEMPLAR"] = DEVICE
os.environ["TCRMP_EXEMPLAR_WARMUP"] = "0"   # this script loads it explicitly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

import numpy as np                      # noqa: E402
import config as cfg                    # noqa: E402
import sam_engine as S                  # noqa: E402


def pick_image():
    """First readable raw frame from any in-process run, or TCRMP_BENCH_IMAGE."""
    override = os.environ.get("TCRMP_BENCH_IMAGE")
    if override:
        return override
    runs = os.path.join(MODULE_ROOT, "inprocess", "run_*")
    for pattern in (
        os.path.join(runs, "step4test_combinedAnnotate", "routed_input", "raw", "*.jp*g"),
        os.path.join(runs, "**", "*.jp*g"),
    ):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            return hits[0]
    return None


def main():
    img = pick_image()
    if img is None:
        print("no benchmark image found; set TCRMP_BENCH_IMAGE")
        return 1
    print(f"device {DEVICE}")
    print(f"image  {img}")

    eng = S.SAM3Engine(cfg)
    w, h = eng.set_image(img)
    pts = [(w * f, h * f) for f in (0.3, 0.5, 0.7)]

    t0 = time.perf_counter()
    eng.segment_point(*pts[0])
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for p in pts[1:]:
        eng.segment_point(*p)
    t_cached = (time.perf_counter() - t0) / len(pts[1:])

    # Uncached reference: drop the cache before each prompt, which is what the
    # engine did on every click before the embedding cache.
    uncached = []
    for p in pts:
        eng._image_embeddings = None
        t0 = time.perf_counter()
        eng.segment_point(*p)
        uncached.append(time.perf_counter() - t0)
    t_uncached = float(np.mean(uncached))

    print(f"first prompt (encoder+decoder) {t_first:.3f}s; "
          f"cached prompt {t_cached:.3f}s; uncached mean {t_uncached:.3f}s")
    if t_cached > 0:
        print(f"cached speedup vs uncached: {t_uncached / t_cached:.1f}x")

    a = eng.segment_point(*pts[1])["mask"]
    eng._image_embeddings = None
    b = eng.segment_point(*pts[1])["mask"]
    parity = bool(np.array_equal(a, b))
    print("cached==uncached mask:", parity)

    # Exemplar: model load, cold scan, then a scan on the cached vision features.
    bbox = [w * 0.40, h * 0.40, w * 0.55, h * 0.55]
    t0 = time.perf_counter()
    eng._ensure_exemplar_model()
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    cold = eng.exemplar_scan(bbox)
    t_ex_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    warm = eng.exemplar_scan(bbox)
    t_ex_cached = time.perf_counter() - t0

    ex_parity = (len(cold) == len(warm)
                 and all(np.array_equal(x["mask"], y["mask"])
                         for x, y in zip(cold, warm)))
    print(f"exemplar model load {t_load:.3f}s; cold scan {t_ex_cold:.3f}s "
          f"({len(cold)} candidates); cached scan {t_ex_cached:.3f}s "
          f"({len(warm)} candidates)")
    print("exemplar cold==cached masks:", ex_parity)

    eng.release()
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
