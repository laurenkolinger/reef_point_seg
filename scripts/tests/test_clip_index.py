"""Persisted clip index (Task 1.4, 2026-08-26): hit path, staleness detection
via the directory-mtime sentinel, and listing parity between the cached path
and a fresh walk for all three consumers (chooseImages build_clip_index,
placePoints _walk_index jpg+cpc, make_lores_variants _iter_images).

Run: env/bin/python scripts/tests/test_clip_index.py
"""
import os
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

INDEX_DIR = tempfile.mkdtemp(prefix="clip_index_store_")
os.environ["TCRMP_CLIP_INDEX_DIR"] = INDEX_DIR

import clip_index as CI

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1


def touch(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


FIX = tempfile.mkdtemp(prefix="clip_index_fixture_")

# Fixture dirs must vanish even if a check raises midway.
import atexit
atexit.register(lambda: (shutil.rmtree(INDEX_DIR, ignore_errors=True),
                         shutil.rmtree(FIX, ignore_errors=True)))

site = os.path.join(FIX, "TCRMP2018_clip", "Annual", "SiteA")
touch(os.path.join(site, "frame_a.jpg"))
touch(os.path.join(site, "frame_b.jpeg"))
touch(os.path.join(site, "frame_b.cpc"))
touch(os.path.join(site, "frame_c_pts.jpg"))
touch(os.path.join(site, ".hidden.jpg"))                    # dot-file: skipped
touch(os.path.join(site, "JPEG", "frame_a.jpg"))            # noncanonical dup
touch(os.path.join(FIX, "TCRMP2018_clip", "Annual", "SiteA_edit", "frame_a.jpg"))
touch(os.path.join(FIX, ".git", "junk.jpg"))                # hidden dir: skipped

# ── Walk then cache hit ─────────────────────────────────────────────
l1 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "walk", "first list is a walk")
check(len(os.listdir(INDEX_DIR)) == 1, "index file persisted")
l2 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "cache", "second list served from cache")
check(l1 == l2, "cached listing identical to walked listing")
check(all(os.path.isabs(p) for p in l1), "paths are absolute")
names = {os.path.relpath(p, FIX) for p in l1}
check(os.path.join("TCRMP2018_clip", "Annual", "SiteA", "frame_a.jpg") in names,
      "listing contains canonical frame")
check(not any(".hidden" in n or n.startswith(".git") for n in names),
      "dot-files and hidden dirs excluded")
check(len(l1) == 6, f"6 files listed (got {len(l1)}: {sorted(names)})")

# ── Staleness: new file in an existing dir ──────────────────────────
# Directory mtimes tick at kernel-clock granularity (~4ms); a mutation in the
# same tick as the scan is invisible. Irrelevant at production timescales,
# but the fixture mutates microseconds after building, so let a tick pass.
time.sleep(0.05)
touch(os.path.join(site, "frame_new.jpg"))
l3 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "walk", "new file in existing dir detected as stale")
check(any(p.endswith("frame_new.jpg") for p in l3), "rebuilt listing has the new file")
check(CI.list_files(FIX) == l3 and CI.LAST_INDEX_SOURCE == "cache",
      "index current again after rebuild")

# ── Staleness: brand-new subdirectory with a file ───────────────────
time.sleep(0.05)
touch(os.path.join(FIX, "TCRMP2019_clip", "Annual", "SiteB", "frame_d.jpg"))
l4 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "walk", "new nested dir detected as stale")
check(any(p.endswith("frame_d.jpg") for p in l4), "rebuilt listing has the nested file")

# ── Staleness: file removal ─────────────────────────────────────────
time.sleep(0.05)
os.remove(os.path.join(site, "frame_new.jpg"))
l5 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "walk", "file removal detected as stale")
check(not any(p.endswith("frame_new.jpg") for p in l5), "removed file gone from listing")

# ── Corrupt index file: silent rebuild ──────────────────────────────
idx_file = os.path.join(INDEX_DIR, os.listdir(INDEX_DIR)[0])
with open(idx_file, "w") as f:
    f.write("{not json")
l6 = CI.list_files(FIX)
check(CI.LAST_INDEX_SOURCE == "walk" and l6 == l5, "corrupt index rebuilt via walk")

# ── Missing clip_dir ────────────────────────────────────────────────
check(CI.list_files(os.path.join(FIX, "nope")) == [], "missing clip_dir lists []")
check(CI.list_files("") == [], "empty clip_dir lists []")

# ── Consumer parity: chooseImages ───────────────────────────────────
sys.path.insert(0, os.path.join(SCRIPTS, "TCRMPcvr_chooseImages", "src"))
import select_images as SI

cached = SI.build_clip_index(FIX)
check(SI._persisted_file_list(FIX) is not None, "chooseImages sees the index")
real_pfl = SI._persisted_file_list
SI._persisted_file_list = lambda d: None   # force the fallback walk
walked = SI.build_clip_index(FIX)
SI._persisted_file_list = real_pfl
check(cached == walked and len(cached) > 0,
      f"chooseImages stem index parity cache vs walk ({len(cached)} stems)")
check(cached["frame_a"].endswith(os.path.join("SiteA", "frame_a.jpg")),
      "canonical ranking preserved (flat original beats JPEG/ and _edit dups)")

# ── Consumer parity: placePoints (jpg + cpc) ────────────────────────
# Both apps ship a src/config.py; drop chooseImages' copy from the module
# cache and path so placePoints' app.py binds to its own config.
sys.modules.pop("config", None)
sys.path.remove(os.path.join(SCRIPTS, "TCRMPcvr_chooseImages", "src"))
sys.path.insert(0, os.path.join(SCRIPTS, "TCRMPclip_placePoints", "src"))
import app as PP

pp_cached_img = PP.build_clip_index(FIX)
pp_cached_cpc = PP.build_cpc_index(FIX)
real_pp = PP._persisted_file_list
PP._persisted_file_list = lambda d: None
pp_walk_img = PP.build_clip_index(FIX)
pp_walk_cpc = PP.build_cpc_index(FIX)
PP._persisted_file_list = real_pp
check(pp_cached_img == pp_walk_img and len(pp_cached_img) > 0,
      f"placePoints image index parity ({len(pp_cached_img)} stems)")
check(pp_cached_cpc == pp_walk_cpc and list(pp_cached_cpc) == ["frame_b"],
      "placePoints cpc index parity")

# ── Consumer parity: make_lores_variants ────────────────────────────
import make_lores_variants as ML

ml_cached = sorted(ML._iter_images(FIX))
real_af = ML._all_files
ML._all_files = lambda root: [p for p in CI.list_files(root, rebuild=True)]
ml_walked = sorted(ML._iter_images(FIX))
ML._all_files = real_af
check(ml_cached == ml_walked and ml_cached, f"make_lores listing parity ({len(ml_cached)} images)")
check(not any(p.endswith("_pts.jpg") for p in ml_cached), "_pts overlays still skipped")

# ── Cleanup (house rule: test artifacts removed and verified) ───────
shutil.rmtree(FIX, ignore_errors=True)
shutil.rmtree(INDEX_DIR, ignore_errors=True)
check(not os.path.isdir(FIX) and not os.path.isdir(INDEX_DIR),
      "fixture tree and index store removed")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
