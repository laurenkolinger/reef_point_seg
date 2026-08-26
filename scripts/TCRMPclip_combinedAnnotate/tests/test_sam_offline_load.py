"""Offline-robustness tests for sam_engine HF loading.

The lab network's DNS drops out intermittently. transformers raises instead
of falling back to the local cache when its online probe fails mid-load,
which killed the annotator at boot (2026-08-13) even with all of
facebook/sam3 cached. These tests simulate a dead network by pointing
HF_ENDPOINT at an unresolvable host BEFORE importing transformers, then
assert the engine's offline-first loader still resolves from cache.

Run: env/bin/python .../tests/test_sam_offline_load.py
"""
import os, sys, hashlib

# Must be set before transformers/huggingface_hub import (read at import time).
os.environ["HF_ENDPOINT"] = "https://nonexistent-hf-endpoint.invalid"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import sam_engine as S

_fail = 0
def check(c, m):
    global _fail
    if not c: _fail += 1; print(f"  FAIL {m}")
    else: print(f"  PASS {m}")

# The offline-first helper exists.
check(hasattr(S, "_hf_load"), "sam_engine exposes _hf_load offline-first helper")

# Processor loads from local cache even though the HF endpoint is unreachable.
# (This exact load is what crashed the annotator at boot on 2026-08-13.)
from transformers import Sam3TrackerProcessor
try:
    proc = S._hf_load(Sam3TrackerProcessor, "facebook/sam3")
    check(proc is not None, "Sam3TrackerProcessor loads with dead HF endpoint")
except Exception as e:
    check(False, f"Sam3TrackerProcessor load raised with dead endpoint: {e}")

# All engine from_pretrained call sites go through _hf_load (none left direct).
src_path = os.path.join(HERE, "..", "src", "sam_engine.py")
with open(src_path) as f:
    body = f.read()
direct = [
    ln for ln in body.splitlines()
    if ".from_pretrained(" in ln
    and "def _hf_load" not in ln
    and "cls.from_pretrained" not in ln
    and not ln.lstrip().startswith("#")
    and "from_pretrained()" not in ln  # prose in docstrings/comments
]
check(not direct, f"no direct from_pretrained call sites remain: {direct[:3]}")

# The three app copies of sam_engine.py must not drift apart.
apps_root = os.path.abspath(os.path.join(HERE, "..", ".."))
copies = [
    os.path.join(apps_root, d, "src", "sam_engine.py")
    for d in ("TCRMPclip_combinedAnnotate", "TCRMPclip_editMasks",
              "TCRMPclip_segmentImages")
]
sums = set()
for p in copies:
    with open(p, "rb") as f:
        sums.add(hashlib.md5(f.read()).hexdigest())
check(len(sums) == 1, "combinedAnnotate/editMasks/segmentImages copies identical")

# _hf_load must tolerate a caller passing local_files_only itself (latent
# TypeError found by the 2026-08-13 adversarial review): first attempt forces
# cache-only, fallback preserves caller kwargs verbatim.
class _FakeLoader:
    calls = []
    @classmethod
    def from_pretrained(cls, *a, **kw):
        cls.calls.append(kw)
        if len(cls.calls) == 1:
            raise RuntimeError("simulated cache miss")
        return "loaded"

_FakeLoader.calls = []
try:
    out = S._hf_load(_FakeLoader, "repo/x", local_files_only=False)
except Exception as e:
    out = f"raised: {e}"
check(out == "loaded", f"_hf_load returns fallback result (got {out!r})")
check(bool(_FakeLoader.calls) and _FakeLoader.calls[0].get("local_files_only") is True,
      "first attempt forces local_files_only=True even when caller passed False")
check(len(_FakeLoader.calls) == 2 and _FakeLoader.calls[1].get("local_files_only") is False,
      "fallback preserves caller kwargs verbatim")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
