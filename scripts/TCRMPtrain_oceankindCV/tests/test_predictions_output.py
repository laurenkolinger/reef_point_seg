"""Unit tests for run_inference.build_prediction_items.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_predictions_output.py"""
import json, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import run_inference as RI

_fail = 0
def check(cond, msg):
    global _fail
    if not cond:
        _fail += 1
        print("FAIL:", msg)

class _Stub: pass

def fake_result(polys_xyn, clss, confs, shape=(1080, 1920)):
    r = _Stub()
    r.orig_shape = shape
    r.boxes = _Stub(); r.boxes.cls = clss; r.boxes.conf = confs
    if polys_xyn is None:
        r.masks = None
    else:
        r.masks = _Stub(); r.masks.xyn = polys_xyn
    return r

import numpy as np
res = fake_result([np.array([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]])], [0.0], [0.84])
items = RI.build_prediction_items([("a.jpg", "/abs/a.jpg", res)], {0: "SS"})
check(len(items) == 1, "one item")
it = items[0]
check(it["width"] == 1920 and it["height"] == 1080, "orig_shape (h,w) mapped to width/height")
d = it["detections"][0]
check(d["class"] == "SS" and d["class_id"] == 0, "class mapping")
check(abs(d["confidence"] - 0.84) < 1e-6, "confidence")
check(d["polygon_xyn"] == [0.1, 0.1, 0.5, 0.1, 0.5, 0.5], "flat xyn polygon")
res_nomask = fake_result(None, [0.0], [0.9])
items2 = RI.build_prediction_items([("b.jpg", "/abs/b.jpg", res_nomask)], {0: "SS"})
check(items2[0]["detections"] == [], "maskless detections skipped, frame kept")
print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
