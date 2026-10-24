"""Shared annotator files must stay byte-identical across the app copies.

combinedAnnotate (CA) is the upstream copy. editMasks (EM) and segmentImages
(SI) are forks that carry their own app.py/config.py/templates but share the
mask, SAM, provenance and export helpers verbatim. Drift between the copies is
how a fix shipped in one annotator silently misses the others, so this test
hashes each shared file and fails on any mismatch.

Deliberate exception: segmentImages' mask_ops.py is NOT compared. Its
clean_mask keeps the fixed 15px MORPH_CLOSE kernel, while CA/EM scale the
kernel with image size (~15px at 1920, ~30px at 4K) so 4K colonies bridge
their fragments. That 11-line difference is intentional, so segmentImages
syncs sam_engine.py only.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_shared_file_parity.py
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

import hashlib
import os
import sys

S = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAIRS = [
    ("TCRMPclip_combinedAnnotate", "TCRMPclip_editMasks",
     ("mask_ops.py", "sam_engine.py", "provenance.py", "export_yolo.py")),
    # mask_ops.py deliberately omitted here: see the 4K kernel note above.
    ("TCRMPclip_combinedAnnotate", "TCRMPclip_segmentImages",
     ("sam_engine.py",)),
]

_fail = 0


def h(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


for a, b, files in PAIRS:
    for f in files:
        pa = os.path.join(S, a, "src", f)
        pb = os.path.join(S, b, "src", f)
        same = os.path.isfile(pa) and os.path.isfile(pb) and h(pa) == h(pb)
        print(("  PASS " if same else "  FAIL ") + f"{a}/{f} == {b}/{f}")
        _fail += 0 if same else 1

# The segmentImages exception is only legitimate while it stays the documented
# kernel difference. Guard the shape of that difference so an unrelated fix
# cannot hide behind it.
_si_mask_ops = os.path.join(S, "TCRMPclip_segmentImages", "src", "mask_ops.py")
_ca_mask_ops = os.path.join(S, "TCRMPclip_combinedAnnotate", "src", "mask_ops.py")
if os.path.isfile(_si_mask_ops) and os.path.isfile(_ca_mask_ops):
    with open(_si_mask_ops) as fh:
        si_body = fh.read()
    with open(_ca_mask_ops) as fh:
        ca_body = fh.read()
    si_fixed = "cv2.MORPH_ELLIPSE, (15, 15)" in si_body
    ca_scaled = "0.008 * max(h, w)" in ca_body
    print(("  PASS " if si_fixed else "  FAIL ")
          + "segmentImages mask_ops keeps the fixed 15px kernel (documented fork)")
    _fail += 0 if si_fixed else 1
    print(("  PASS " if ca_scaled else "  FAIL ")
          + "combinedAnnotate mask_ops keeps the size-scaled kernel")
    _fail += 0 if ca_scaled else 1
else:
    print("  FAIL segmentImages/combinedAnnotate mask_ops.py missing")
    _fail += 1

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
