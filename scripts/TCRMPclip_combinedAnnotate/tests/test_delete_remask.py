"""A rejected mask must NOT clip a new overlapping mask in its freed region.

Server-side clip-logic repro for the delete-then-remask bug (Task 17). After a
mask is deleted, placing a new overlapping mask in the freed region must not be
clipped to a sliver. The /add route clips a new mask against _blocking_masks(),
which must exclude (a) masks already status=='rejected' server-side, and (b) any
id the client reports as just-deleted via rejected_ids (the race case where the
reject PUT has not landed yet). It must still BLOCK live (pending/accepted)
neighbors so the no-overlap rule holds for real masks.

Run with env/bin/python. Uses _blocking_masks + the clip math directly.
"""
import os, sys
import numpy as np
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import app as A
from mask_ops import rle_encode

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

# A 10x10 frame; mask A occupies the left half.
H = W = 10
a = np.zeros((H, W), dtype=bool); a[:, :5] = True

# (1) A rejected mask must not be in the blocking set (excluded by status).
masks = [{"id": 0, "status": "rejected", "rle": rle_encode(a)}]
block = A._blocking_masks(masks, rejected_ids=[])
check(block == [], "rejected mask excluded from blocking by status")

# (2) A client-reported just-deleted id must also be excluded even if status
#     hasn't flipped server-side yet (the reload-race case: the reject PUT has
#     not landed, so the in-memory/disk copy still says 'pending').
masks2 = [{"id": 7, "status": "pending", "rle": rle_encode(a)}]
check(A._blocking_masks(masks2, rejected_ids=[7]) == [],
      "client rejected_ids excluded even when status still pending")

# (3) The no-overlap rule must still hold for LIVE masks: a pending neighbor NOT
#     in rejected_ids must remain in the blocking set.
check([m["id"] for m in A._blocking_masks(masks2, rejected_ids=[])] == [7],
      "live (pending) neighbor still blocks when not deleted")

# (4) End-to-end clip math: simulate /add's clip. A NEW mask covering the whole
#     frame, clipped only against the blocking set. With mask A deleted (rejected
#     OR reported in rejected_ids), the new mask keeps its full footprint in the
#     freed left half — it is NOT trimmed to a sliver.
def clip(new_binary, blocking):
    from mask_ops import rle_decode
    h, w = new_binary.shape
    forbidden = np.zeros((h, w), dtype=bool)
    for m in blocking:
        rle = m.get("rle")
        if not rle:
            continue
        forbidden |= np.asarray(rle_decode(rle, shape=(h, w)), dtype=bool)
    return new_binary & ~forbidden

new_full = np.ones((H, W), dtype=bool)

# Deleted-by-status: freed region fully available, new mask keeps all 100 px.
clipped = clip(new_full, A._blocking_masks(masks, rejected_ids=[]))
check(int(clipped.sum()) == H * W, "deleted-by-status: new mask keeps full region (not a sliver)")

# Deleted-by-rejected_ids (race): same — freed region fully available.
clipped2 = clip(new_full, A._blocking_masks(masks2, rejected_ids=[7]))
check(int(clipped2.sum()) == H * W, "deleted-via-rejected_ids: new mask keeps full region (not a sliver)")

# Live neighbor present (control): the new mask IS trimmed off the left half,
# proving the clip still protects real masks (no overlap with live masks).
clipped3 = clip(new_full, A._blocking_masks(masks2, rejected_ids=[]))
check(int(clipped3.sum()) == H * (W - 5),
      "live neighbor present: new mask correctly trimmed (no-overlap preserved)")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
