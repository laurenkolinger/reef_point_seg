"""EasyOCR must be constructed offline-first (download_enabled=False), with a
download-enabled fallback only when the offline constructor fails (fresh
machine or corrupt model file). Regression for the 2026-08-13 network sweep:
the default Reader() urlretrieve-downloads models at first use and crashes an
OCR run when lab DNS is down.
Run: env/bin/python .../tests/test_ocr_offline_load.py"""
import os, sys, hashlib, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

_fail = 0
def check(c, m):
    global _fail
    if not c: _fail += 1; print(f"  FAIL {m}")
    else: print(f"  PASS {m}")

calls = []
class _FakeReader:
    def __init__(self, langs, gpu=True, verbose=False, download_enabled=True):
        calls.append({"langs": langs, "gpu": gpu, "download_enabled": download_enabled})
        if len(calls) == 1 and not download_enabled:
            raise FileNotFoundError("simulated missing model")

fake = types.ModuleType("easyocr"); fake.Reader = _FakeReader
sys.modules["easyocr"] = fake

import detect as D
D._ocr_reader = None
try:
    r = D.get_ocr_reader(gpu=False)
except Exception as e:
    r = None
    print(f"  (get_ocr_reader raised: {e})")
check(r is not None, "reader constructed via fallback")
check(bool(calls) and calls[0]["download_enabled"] is False,
      "first attempt is offline (download_enabled=False)")
check(len(calls) == 2 and calls[1]["download_enabled"] is True,
      "fallback retries with downloads enabled")
check(all(c["gpu"] is False for c in calls), "gpu flag preserved on both attempts")

calls.clear(); D._ocr_reader = None
class _HealthyReader:
    def __init__(self, langs, gpu=True, verbose=False, download_enabled=True):
        calls.append({"download_enabled": download_enabled})
fake.Reader = _HealthyReader
r2 = D.get_ocr_reader()
check(len(calls) == 1 and calls[0]["download_enabled"] is False,
      "healthy cache constructs offline only (no fallback attempt)")
r3 = D.get_ocr_reader()
check(len(calls) == 1, "singleton: second call reuses the reader")

# The two detect.py copies must not drift apart.
def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
a = md5(os.path.join(HERE, "..", "src", "detect.py"))
b = md5(os.path.join(HERE, "..", "..", "TCRMPclip_ocrID_batch", "src", "detect.py"))
check(a == b, "placePoints and ocrID_batch detect.py copies identical")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
