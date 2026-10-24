"""Tracker image embeddings and exemplar vision features are computed once per
frame and reused; clear_image and release drop them; warm-up loads the exemplar
model once and survives a loader that raises; set_mask_size validates.
Fake models record kwargs; no GPU, no checkpoint.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_sam_embedding_cache.py
"""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import atexit, logging, os, shutil, sys, tempfile, threading, types
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
os.environ["TCRMP_EXEMPLAR_WARMUP"] = "0"
import torch
from PIL import Image
import sam_engine as S

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

class FakeOut:
    def __init__(self):
        self.pred_masks = torch.zeros((1, 1, 3, 8, 8)); self.pred_masks[0, 0, :, 2:6, 2:6] = 1.0
        self.iou_scores = torch.tensor([[[0.5, 0.6, 0.7]]])

class FakeTracker:
    def __init__(self): self.calls = []; self.embed_calls = 0
    def get_image_embeddings(self, pixel_values):
        self.embed_calls += 1; return ["emb"]
    def __call__(self, **kw):
        self.calls.append(kw); return FakeOut()
    def eval(self): return self

class FakeBatch(dict):
    def to(self, dev): return self

class FakeProc:
    def __init__(self): self.calls = []
    def __call__(self, images=None, original_sizes=None, **kw):
        self.calls.append({"images": images, "original_sizes": original_sizes, **kw})
        d = FakeBatch(kw)
        if images is not None: d["pixel_values"] = torch.zeros((1, 3, 8, 8))
        d["original_sizes"] = torch.tensor(original_sizes if original_sizes else [[8, 8]])
        return d
    def post_process_masks(self, masks, original_sizes):
        return [masks[0]]

class FakeVisionOut: pass
class FakeExemplar:
    def __init__(self): self.calls = []; self.feat_calls = 0
    def get_vision_features(self, pixel_values):
        self.feat_calls += 1; return FakeVisionOut()
    def __call__(self, **kw):
        self.calls.append(kw); return types.SimpleNamespace()
    def eval(self): return self
class FakeExProc(FakeProc):
    def post_process_instance_segmentation(self, out, threshold, mask_threshold, target_sizes):
        return [{"masks": [], "scores": []}]

eng = S.SAM3Engine.__new__(S.SAM3Engine)
eng.__dict__.update(dict(device_tracker="cpu", device_exemplar="cpu", mask_size="generous",
    confidence_threshold=0.5, _tracker=FakeTracker(), _tracker_proc=FakeProc(),
    _text_model=None, _text_proc=None, _current_image=None, _current_image_path=None,
    _image_size=None, _image_embeddings=None, _exemplar_features=None,
    _prompt_lock=threading.Lock(), _exemplar_lock=threading.Lock()))

tmp = tempfile.mkdtemp()
atexit.register(shutil.rmtree, tmp, ignore_errors=True)
p1 = os.path.join(tmp, "a.jpg"); p2 = os.path.join(tmp, "b.jpg")
Image.new("RGB", (8, 8)).save(p1); Image.new("RGB", (8, 8)).save(p2)

eng.set_image(p1)
r1 = eng.segment_point(3, 3)
r2 = eng.refine_mask([(3, 3), (4, 4)], [1, 0])
check(eng._tracker.embed_calls == 1, "embeddings computed once for two prompts on the same frame")
check(all("image_embeddings" in c and "pixel_values" not in c for c in eng._tracker.calls),
      "forward receives image_embeddings, never pixel_values")
check(all(c["images"] is None and c["original_sizes"] == [[8, 8]] for c in eng._tracker_proc.calls[1:]),
      "prompt processor calls pass original_sizes only ([[h, w]])")
r3 = eng.segment_box(1, 1, 6, 6)
check(eng._tracker.embed_calls == 1 and "input_boxes" in eng._tracker.calls[-1], "box prompt reuses embeddings")
r4 = eng.refine_mask([(3, 3)], [1], box=[1, 1, 6, 6])
last = eng._tracker.calls[-1]
check("input_boxes" in last and "input_points" in last, "refine with box passes both box and points")
eng.set_image(p2)
check(eng._image_embeddings is None and eng._exemplar_features is None, "new frame invalidates both caches")
eng.segment_point(2, 2)
check(eng._tracker.embed_calls == 2, "new frame recomputes embeddings once")

# mask size
eng.set_mask_size("tight"); check(eng.mask_size == "tight", "set_mask_size accepts tight")
try:
    eng.set_mask_size("huge"); check(False, "set_mask_size rejects unknown")
except ValueError:
    check(True, "set_mask_size rejects unknown")
eng.set_mask_size("generous")

# exemplar feature cache
eng._text_model = FakeExemplar(); eng._text_proc = FakeExProc()
eng.exemplar_scan([0, 0, 4, 4]); eng.exemplar_scan([1, 1, 5, 5])
check(eng._text_model.feat_calls == 1, "exemplar vision features computed once per frame")
check(all("vision_embeds" in c and "pixel_values" not in c for c in eng._text_model.calls),
      "exemplar forward receives vision_embeds")

# clear_image: the public way to drop a frame (app.py must not null the
# attributes by hand, that would strand the embeddings on the GPU)
eng.clear_image()
check(eng._current_image is None and eng._current_image_path is None
      and eng._image_size is None, "clear_image nulls the current frame")
check(eng._image_embeddings is None and eng._exemplar_features is None,
      "clear_image nulls both caches")
eng.set_image(p2)
check(eng._image_size == (8, 8), "set_image reloads after clear_image")

# warm-up: loader called once, guarded by the lock, disabled by env
calls = {"n": 0}
def fake_loader():
    calls["n"] += 1; eng._text_model = FakeExemplar(); eng._text_proc = FakeExProc()
eng._text_model = None; eng._text_proc = None
eng._load_exemplar_model = fake_loader
os.environ["TCRMP_EXEMPLAR_WARMUP"] = "0"
check(eng.warm_exemplar_async() is None, "warm-up disabled by TCRMP_EXEMPLAR_WARMUP=0")
os.environ["TCRMP_EXEMPLAR_WARMUP"] = "1"
t = eng.warm_exemplar_async(); t.join(5)
eng._ensure_exemplar_model()
check(calls["n"] == 1, "warm-up plus a later ensure load the exemplar model exactly once")

# a warm-up whose loader raises logs a warning instead of killing the thread
class _Capture(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self); self.messages = []
    def emit(self, record): self.messages.append(record.getMessage())

def boom_loader():
    raise RuntimeError("simulated exemplar load failure")

cap = _Capture()
S.log.addHandler(cap); S.log.propagate = False   # keep the expected warning off stderr
eng._text_model = None; eng._text_proc = None
eng._load_exemplar_model = boom_loader
t = eng.warm_exemplar_async(); t.join(5)
S.log.removeHandler(cap); S.log.propagate = True
check(not t.is_alive(), "warm-up thread exits cleanly when the loader raises")
check(eng._text_model is None, "a failed warm-up leaves the exemplar model unloaded")
check(any("warm-up failed" in m for m in cap.messages),
      "a failed warm-up logs a warning instead of raising")
eng._load_exemplar_model = fake_loader
eng._ensure_exemplar_model()
check(eng._text_model is not None, "the lazy path still loads after a failed warm-up")

# release drops the frame caches and the exemplar model
eng.release()
check(eng._image_embeddings is None and eng._exemplar_features is None,
      "release nulls both caches")
check(eng._text_model is None and eng._text_proc is None
      and eng._current_image is None, "release nulls the exemplar model and frame")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
