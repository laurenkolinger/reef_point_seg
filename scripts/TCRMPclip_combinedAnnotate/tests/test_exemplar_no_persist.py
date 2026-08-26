"""/exemplar returns candidates but does NOT persist them to seg['masks'].

No pytest dependency — run with the unified env python:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_exemplar_no_persist.py

The candidate-finding pipeline is exercised through the pure helper
`_exemplar_candidates(seg, mode, threshold, mask_id, engine)` extracted from the
`/exemplar` route. A tiny test-only `_StubExemplarEngine` returns N synthetic
candidate masks so this runs headlessly (the real SAM3 engine is never loaded).

Contract under test:
  * `_exemplar_candidates` RETURNS a list of candidate mask dicts.
  * It NEVER mutates `seg['masks']` (neither mode 'this' nor mode 'all').
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
sys.path.insert(0, os.path.abspath(_SRC))

# Pin config the same way the sibling app test does, before `import app`, so the
# backend reads a deterministic target-species set at import time.
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA')
os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')

import app as A

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


# One accepted PA mask with a real bbox so mode='all' has an exemplar source to
# scan; its rle covers a small rectangle so the exclusion mask is well-formed.
import numpy as np
from mask_ops import rle_encode

_seed = np.zeros((48, 64), dtype=bool)
_seed[4:12, 4:12] = True
seg = {
    "image_width": 64,
    "image_height": 48,
    "masks": [{
        "id": 0,
        "status": "accepted",
        "species": "PA",
        "category": "Coral",
        "bbox": [4, 4, 12, 12],
        "rle": rle_encode(_seed),
    }],
}

# _exemplar_candidates(seg, mode, threshold, mask_id, engine) is the pure helper
# extracted in Step 3: it returns a list of candidate dicts and must NOT mutate seg.
before = len(seg["masks"])
cands = A._exemplar_candidates(seg, mode="all", threshold=0.5, mask_id=None,
                               engine=A._StubExemplarEngine(n=2))
check(len(seg["masks"]) == before, "seg['masks'] not mutated by candidate scan (mode=all)")
check(len(cands) == 2, f"returns 2 candidates, got {len(cands)}")
check(isinstance(cands, list) and all(isinstance(c, dict) for c in cands),
      "returns a list of mask dicts")

# mode='this' must also leave seg['masks'] untouched.
before_this = len(seg["masks"])
cands_this = A._exemplar_candidates(seg, mode="this", threshold=0.5, mask_id=0,
                                    engine=A._StubExemplarEngine(n=2))
check(len(seg["masks"]) == before_this, "seg['masks'] not mutated by candidate scan (mode=this)")
check(len(cands_this) == 2, f"mode=this returns 2 candidates, got {len(cands_this)}")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
