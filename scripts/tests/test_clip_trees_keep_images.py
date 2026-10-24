"""Clip-tree maintenance never deletes or truncates an image (Lauren, 2026-10-15).

Covers make_lores_variants.generate (a refreshed low-res copy is staged, the old
copy moves to TCRMP_clip_lores/_superseded/<stamp>/<relative path>), the
_superseded skip in clip_index and in the walk fallbacks of make_lores_variants,
select_images and placePoints, and sync_from_source.sh (a changed image is
backed up into <DST tree>/_superseded/<stamp>/ and the folder is never copied
back). Every original byte must survive with the same SHA-256.

Run: env/bin/python scripts/tests/test_clip_trees_keep_images.py
"""
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)
import make_lores_variants as M
import clip_index as CI

SUPERSEDED = "_superseded"
SYNC_SCRIPT = os.path.join(SCRIPTS, "sync_from_source.sh")
SYNC_TIMEOUT_S = 60

_fail = 0


def check(c, m):
    """Print PASS or FAIL for m and count failures."""
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def sha(path):
    """SHA-256 hex digest of the file at path."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def tree_hashes(root):
    """Every file under root as {sha256: [paths]}, for byte-survival checks."""
    out = {}
    for dp, _d, fs in os.walk(root):
        for fn in fs:
            p = os.path.join(dp, fn)
            out.setdefault(sha(p), []).append(p)
    return out


def all_files(root):
    """Sorted list of every file path under root."""
    return sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(root) for f in fs)


def in_superseded(path):
    """True when a path component is exactly _superseded."""
    return SUPERSEDED in path.split(os.sep)


print("make_lores_variants: refresh keeps the old low-res copy")
with tempfile.TemporaryDirectory(prefix="ml_") as d:
    clip = os.path.join(d, "TCRMP_clip", "TCRMP2018_clip", "Annual", "T105")
    os.makedirs(clip)
    src = os.path.join(clip, "TCRMP20181207_clip_MRS_T105.jpg")
    Image.new("RGB", (3840, 2160), (10, 20, 30)).save(src)
    lores_root = os.path.join(d, "TCRMP_clip_lores")
    M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
    lp = os.path.join(lores_root, "TCRMP2018_clip", "Annual", "T105",
                      "TCRMP20181207_clip_MRS_T105_lores.jpg")
    old_hash = sha(lp)
    # source changes (new content, newer mtime) -> lores must be refreshed
    Image.new("RGB", (3840, 2160), (200, 100, 50)).save(src)
    future = time.time() + 5
    os.utime(src, (future, future))
    M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
    check(sha(lp) != old_hash, "the low-res copy was refreshed with new pixels")
    held = [p for p in all_files(lores_root) if in_superseded(p)]
    check(len(held) == 1, "exactly one old copy sits under _superseded")
    check(bool(held) and sha(held[0]) == old_hash, "the superseded copy has the old bytes (same hash)")
    check(bool(held) and held[0].endswith(os.path.join(
        "TCRMP2018_clip", "Annual", "T105", "TCRMP20181207_clip_MRS_T105_lores.jpg")),
        "the superseded copy keeps its relative path")
    check(bool(held) and os.path.relpath(held[0], lores_root).startswith(SUPERSEDED + os.sep),
          "the holding folder is TCRMP_clip_lores/_superseded/<stamp>/")
    check(not any(".tmp" in f or f.startswith(".lores") for f in os.listdir(os.path.dirname(lp))),
          "no temp file left beside the refreshed copy")
    # --force on an unchanged source also keeps the copy being replaced
    prev = sha(lp)
    M.generate(os.path.join(d, "TCRMP_clip"), lores_root, force=True, log=lambda *_: None)
    check(prev in tree_hashes(lores_root), "force run: the replaced copy's bytes still exist")
    # the _superseded folder is never read as a source or mirrored
    sup_clip = os.path.join(d, "TCRMP_clip", SUPERSEDED, "20261015-000000")
    os.makedirs(sup_clip)
    Image.new("RGB", (3840, 2160), (1, 2, 3)).save(os.path.join(sup_clip, "TCRMP20000101_clip_ZZZ_T1.jpg"))
    st = M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
    check(st["scanned"] == 1, "a _superseded folder inside the clip tree is not scanned")
    # a failed write leaves the existing copy untouched
    real_save = Image.Image.save
    before = sha(lp)
    future2 = time.time() + 10
    os.utime(src, (future2, future2))

    def boom(self, *a, **k):
        raise OSError("disk full")
    Image.Image.save = boom
    try:
        try:
            M.generate(os.path.join(d, "TCRMP_clip"), lores_root, log=lambda *_: None)
            raised = False
        except OSError:
            raised = True
    finally:
        Image.Image.save = real_save
    check(raised, "a failed write raises instead of passing silently")
    check(os.path.isfile(lp) and sha(lp) == before, "a failed write leaves the existing copy whole")

print("make_lores_variants fallback walk skips _superseded")
with tempfile.TemporaryDirectory(prefix="mlw_") as d:
    os.makedirs(os.path.join(d, "a", SUPERSEDED, "s"))
    for p in ("a/x.jpg", "a/%s/s/y.jpg" % SUPERSEDED):
        Image.new("RGB", (8, 8)).save(os.path.join(d, p))
    real = CI.list_files
    CI.list_files = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("force fallback"))
    try:
        got = M._all_files(d)
    finally:
        CI.list_files = real
    check([os.path.basename(p) for p in got] == ["x.jpg"], "direct walk returns x.jpg only")

print("clip_index._scan skips _superseded")
with tempfile.TemporaryDirectory(prefix="ci_") as d:
    os.makedirs(os.path.join(d, "a", SUPERSEDED, "s"))
    Image.new("RGB", (8, 8)).save(os.path.join(d, "a", "x.jpg"))
    Image.new("RGB", (8, 8)).save(os.path.join(d, "a", SUPERSEDED, "s", "y.jpg"))
    files, dirs = CI._scan(d)
    check(files == [os.path.join("a", "x.jpg")], "scan lists x.jpg only")
    check(not any(in_superseded(k) for k in dirs), "no _superseded directory recorded")
    check([os.path.basename(p) for p in CI.list_files(d, rebuild=True)] == ["x.jpg"],
          "list_files omits the held copy")

print("select_images and placePoints walk fallbacks skip _superseded")
for modname, subdir, fn, args in (
        ("select_images", "TCRMPcvr_chooseImages", "build_clip_index", ()),
        ("app", "TCRMPclip_placePoints", "_walk_index", ((".jpg", ".jpeg"),))):
    srcdir = os.path.join(SCRIPTS, subdir, "src")
    sys.path.insert(0, srcdir)
    try:
        for stale in (modname, "config"):  # each tool ships its own config module
            sys.modules.pop(stale, None)
        mod = importlib.import_module(modname)
    except Exception as e:  # import needs the module's own deps
        check(False, f"{modname} imports ({e})")
        continue
    finally:
        sys.path.remove(srcdir)
    with tempfile.TemporaryDirectory(prefix="walk_") as d:
        os.makedirs(os.path.join(d, "a", SUPERSEDED, "s"))
        Image.new("RGB", (8, 8)).save(os.path.join(d, "a", "TCRMP1_clip_A_T1.jpg"))
        Image.new("RGB", (8, 8)).save(os.path.join(d, "a", SUPERSEDED, "s", "TCRMP2_clip_B_T1.jpg"))
        mod._persisted_file_list = lambda _d: None
        idx = getattr(mod, fn)(d, *args)
        check(sorted(idx) == ["TCRMP1_clip_A_T1"], f"{modname}.{fn} indexes the live frame only")

print("sync_from_source.sh keeps a changed image")


def sync_fixture(base):
    """Build a stub source tree and a destination tree under base; returns (src, dst)."""
    src = os.path.join(base, "src")
    dst = os.path.join(base, "dst")
    os.makedirs(os.path.join(src, "input", "TCRMP_clip", "A"))
    os.makedirs(os.path.join(dst, "supporting_data", "TCRMP_clip", "A"))
    os.makedirs(os.path.join(dst, "scripts"))
    shutil.copy(SYNC_SCRIPT, os.path.join(dst, "scripts", "sync_from_source.sh"))
    return src, dst


def run_sync(dst, src, *extra):
    """Run the copied sync script against the stub trees, data target only."""
    env = dict(os.environ, TCRMP_SYNC_SRC=src)
    return subprocess.run(["bash", os.path.join(dst, "scripts", "sync_from_source.sh"), *extra, "data"],
                          env=env, capture_output=True, text=True, timeout=SYNC_TIMEOUT_S)


with tempfile.TemporaryDirectory(prefix="sync_") as base:
    src, dst = sync_fixture(base)
    sp = os.path.join(src, "input", "TCRMP_clip", "A", "f.jpg")
    dp = os.path.join(dst, "supporting_data", "TCRMP_clip", "A", "f.jpg")
    Image.new("RGB", (16, 16), (1, 1, 1)).save(dp)
    old = sha(dp)
    Image.new("RGB", (16, 16), (250, 5, 5)).save(sp)
    new = sha(sp)
    r = run_sync(dst, src, "--apply")
    check(r.returncode == 0, f"sync exits 0 ({r.stderr.strip()[:120]})")
    check(sha(dp) == new, "the live copy now has the new bytes")
    held = [p for p in all_files(os.path.join(dst, "supporting_data", "TCRMP_clip")) if in_superseded(p)]
    check(len(held) == 1 and sha(held[0]) == old, "the replaced image was moved aside with the same hash")
    check(bool(held) and held[0].endswith(os.path.join("A", "f.jpg")), "relative path kept under _superseded/<stamp>/")
    # a second sync with the stub holding no _superseded and an old _superseded in dst: not clobbered
    os.makedirs(os.path.join(src, "input", "TCRMP_clip", SUPERSEDED))
    r = run_sync(dst, src, "--apply")
    check(r.returncode == 0, "second sync exits 0")
    check(not os.path.exists(os.path.join(dst, "supporting_data", "TCRMP_clip", SUPERSEDED, "x")),
          "nothing invented")
    held2 = [p for p in all_files(os.path.join(dst, "supporting_data", "TCRMP_clip")) if in_superseded(p)]
    check(old in {sha(p) for p in held2}, "the held image survives a second sync")
    # source _superseded folder is not copied in
    Image.new("RGB", (8, 8), (9, 9, 9)).save(os.path.join(src, "input", "TCRMP_clip", SUPERSEDED, "z.jpg"))
    run_sync(dst, src, "--apply")
    check(not any(p.endswith("z.jpg") for p in all_files(dst)), "a _superseded folder in the source is not copied")
    # dry run changes nothing
    Image.new("RGB", (16, 16), (7, 7, 7)).save(sp)
    before = sha(dp)
    run_sync(dst, src)
    check(sha(dp) == before, "dry run leaves the live copy alone")

print("sync_from_source.sh projects target keeps a changed image")
with tempfile.TemporaryDirectory(prefix="syncp_") as base:
    src, dst = sync_fixture(base)
    os.makedirs(os.path.join(src, "projects", "p1"))
    os.makedirs(os.path.join(dst, "projects", "p1"))
    sp = os.path.join(src, "projects", "p1", "i.png")
    dp = os.path.join(dst, "projects", "p1", "i.png")
    Image.new("RGB", (16, 16), (3, 3, 3)).save(dp)
    old = sha(dp)
    os.utime(dp, (1000000000, 1000000000))  # rsync skips equal size and mtime
    Image.new("RGB", (16, 16), (90, 90, 90)).save(sp)
    env = dict(os.environ, TCRMP_SYNC_SRC=src)
    # the project-path rewriter needs the module env; stub it out of the copy
    os.makedirs(os.path.join(dst, "env", "bin"))
    with open(os.path.join(dst, "env", "bin", "python"), "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(os.path.join(dst, "env", "bin", "python"), 0o755)
    r = subprocess.run(["bash", os.path.join(dst, "scripts", "sync_from_source.sh"), "--apply", "projects"],
                       env=env, capture_output=True, text=True, timeout=SYNC_TIMEOUT_S)
    check(r.returncode == 0, f"projects sync exits 0 ({r.stderr.strip()[:120]})")
    held = [p for p in all_files(os.path.join(dst, "projects")) if in_superseded(p)]
    check(len(held) == 1 and sha(held[0]) == old, "the replaced project image sits in projects/_superseded/<stamp>/")

sys.exit(1 if _fail else 0)
