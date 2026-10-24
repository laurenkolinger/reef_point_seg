"""Dataset rebuilds and cleanups never delete an image (Lauren, 2026-10-15).

Covers the Step 6 paths that used to remove a previous split or dataset:
detect_dataset._swap_into_place, pinned_split.main, bal_train_test_split.py,
restructure_boundingbox_directories.clean_directory and handle_mismatches,
and validate_dataset.handle_mismatches. Fixtures hold real PNG, JPEG and WEBP
bytes; every original must survive with the same SHA-256, in place or under a
_superseded/<stamp>/ folder. A tree holding only symlinks may still be removed.

Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_images_never_deleted.py
"""
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import builtins
import hashlib
import os
import subprocess
import sys
import tempfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
TOOLS = os.path.join(HERE, "..", "oceankind_CV", "tools")
sys.path.insert(0, SRC)
sys.path.insert(0, TOOLS)

import detect_dataset as DD
import pinned_split as PS

SUPERSEDED = "_superseded"
IMAGE_FORMATS = (("a.png", "PNG"), ("b.jpg", "JPEG"), ("c.webp", "WEBP"))
CLI_TIMEOUT_S = 120

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


def hashes_under(root):
    """Set of SHA-256 digests of every regular file under root."""
    return {sha(os.path.join(dp, f)) for dp, _d, fs in os.walk(root) for f in fs
            if not os.path.islink(os.path.join(dp, f))}


def write_images(folder, seed=0):
    """Write one real PNG, JPEG and WEBP into folder, varied by seed; return {path: sha}."""
    os.makedirs(folder, exist_ok=True)
    out = {}
    for i, (name, fmt) in enumerate(IMAGE_FORMATS):
        p = os.path.join(folder, name)
        Image.new("RGB", (24 + i, 24), (seed * 40 % 256, 10 * i, 99)).save(p, fmt)
        out[p] = sha(p)
    return out


def held_under(root):
    """Digests of files inside any _superseded folder under root."""
    out = set()
    for dp, _d, fs in os.walk(root):
        if SUPERSEDED in dp.split(os.sep):
            out |= {sha(os.path.join(dp, f)) for f in fs}
    return out


print("detect_dataset._swap_into_place")
with tempfile.TemporaryDirectory(prefix="swap_") as d:
    final = os.path.join(d, "dataset")
    orig = write_images(os.path.join(final, "train", "images"))
    new = os.path.join(d, "build")
    write_images(os.path.join(new, "train", "images"), seed=5)
    DD._swap_into_place(new, final)
    check(hashes_under(final) .isdisjoint(set(orig.values())), "the new dataset is in place")
    check(set(orig.values()) <= held_under(d), "all three old images sit under <parent>/_superseded/<stamp>/ with the same hash")
    check(any(p.startswith(os.path.join(d, SUPERSEDED)) for p in
              (os.path.join(dp, f) for dp, _x, fs in os.walk(d) for f in fs)),
          "the holding folder hangs from the dataset's parent")
    check(not [n for n in os.listdir(d) if n.startswith(".dataset_old_")], "no .dataset_old_ leftover")

with tempfile.TemporaryDirectory(prefix="swap_fail_") as d:
    final = os.path.join(d, "dataset")
    orig = write_images(os.path.join(final, "train", "images"))
    new = os.path.join(d, "build")
    write_images(os.path.join(new, "train", "images"), seed=5)
    real_rename = os.rename

    def failing_rename(a, b, *rest, **kw):
        if a == new:
            raise OSError(28, "No space left on device")
        return real_rename(a, b, *rest, **kw)
    os.rename = failing_rename
    try:
        try:
            DD._swap_into_place(new, final)
            raised = False
        except OSError:
            raised = True
    finally:
        os.rename = real_rename
    check(raised, "a failed swap raises")
    check(set(orig.values()) <= hashes_under(final), "a failed swap leaves the old dataset whole at its path")

with tempfile.TemporaryDirectory(prefix="swap_links_") as d:
    final = os.path.join(d, "dataset")
    real = write_images(os.path.join(d, "elsewhere"))
    os.makedirs(os.path.join(final, "train", "images"))
    for p in real:
        os.symlink(p, os.path.join(final, "train", "images", os.path.basename(p)))
    new = os.path.join(d, "build")
    write_images(os.path.join(new, "train", "images"), seed=5)
    DD._swap_into_place(new, final)
    check(not os.path.exists(os.path.join(d, SUPERSEDED)), "a symlink-only dataset is replaced without a holding folder")
    check(set(real.values()) <= hashes_under(os.path.join(d, "elsewhere")), "the link targets are untouched")

print("holds_real_images and set_aside_images")
with tempfile.TemporaryDirectory(prefix="hri_") as d:
    only_webp = os.path.join(d, "w")
    write_images(only_webp)
    for n in ("a.png", "b.jpg"):
        os.remove(os.path.join(only_webp, n))
    check(DD.holds_real_images(only_webp), "a folder holding only a WEBP counts as holding images")
    txt = os.path.join(d, "t")
    os.makedirs(txt)
    open(os.path.join(txt, "x.txt"), "w").close()
    check(not DD.holds_real_images(txt), "a folder of text files holds no images")
    check(not DD.holds_real_images(os.path.join(d, "missing")), "a missing path holds none")
    check(DD.set_aside_images(os.path.join(d, "missing"), d) is None, "a missing path is a no-op")

print("pinned_split.main rebuild")
with tempfile.TemporaryDirectory(prefix="ps_") as d:
    src = os.path.join(d, "export")
    out = os.path.join(d, "dataset")
    img_dir = os.path.join(src, "all_images")
    lbl_dir = os.path.join(src, "all_labels")
    os.makedirs(img_dir)
    os.makedirs(lbl_dir)
    for site in ("AAA", "BBB"):
        for t in range(1, 7):
            for i in range(3):
                name = f"TCRMP20140101_clip_{site}_T{t}{i:02d}"
                Image.new("RGB", (16, 16), (t * 30, i * 50, 7)).save(os.path.join(img_dir, name + ".jpeg"), "JPEG")
                with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                    f.write("0 0.5 0.5 0.1 0.1 0.5 0.4 0.4 0.5\n")
    import yaml
    with open(os.path.join(src, "data.yaml"), "w") as f:
        yaml.dump({"path": src, "train": "all_images", "val": "all_images", "names": {0: "SS"}}, f)
    PS.main(["--src", src, "--out", out, "--min_samples", "1"])
    first = hashes_under(out)
    first_imgs = {sha(os.path.join(dp, f)) for dp, _x, fs in os.walk(out) for f in fs if f.endswith(".jpeg")}
    PS.main(["--src", src, "--out", out, "--min_samples", "1"])
    check(first_imgs <= held_under(d), "rebuild: every image of the previous split sits under _superseded/<stamp>/")
    check(first_imgs <= hashes_under(out) | held_under(d), "rebuild: nothing lost")
    check(os.path.isfile(os.path.join(out, "data.yaml")), "rebuild: the new dataset is written")
    check(hashes_under(img_dir) >= first_imgs, "rebuild: the export images are untouched")

print("bal_train_test_split.py --yes")
with tempfile.TemporaryDirectory(prefix="bal_") as d:
    base = os.path.join(d, "ds")
    imgs = os.path.join(base, "all_images")
    lbls = os.path.join(base, "all_labels")
    os.makedirs(imgs)
    os.makedirs(lbls)
    for i in range(24):
        Image.new("RGB", (12, 12), (i * 10, 5, 5)).save(os.path.join(imgs, f"f{i:02d}.png"))
        with open(os.path.join(lbls, f"f{i:02d}.txt"), "w") as f:
            f.write(f"{i % 2} 0.5 0.5 0.2 0.2\n")
    old = write_images(os.path.join(base, "train", "images"), seed=9)
    old_hashes = set(old.values())
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "bal_train_test_split.py"), "--src", base,
                        "--yes", "--min_samples", "1"], capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    check("Traceback" not in r.stderr, f"split script ran without a traceback ({r.stderr.strip()[-160:]})")
    check(old_hashes <= held_under(base), "old train images sit under <dataset>/_superseded/<stamp>/ with the same hash")
    check(os.path.isdir(os.path.join(base, "train", "images")), "the new train folder exists")

print("validate_dataset.handle_mismatches")
import validate_dataset as VD
with tempfile.TemporaryDirectory(prefix="vd_") as d:
    ds = os.path.join(d, "ds")
    imgs = os.path.join(ds, "all_images")
    lbls = os.path.join(ds, "all_labels")
    os.makedirs(lbls)
    kept = write_images(imgs)
    unlabeled = {os.path.splitext(os.path.basename(p))[0] for p in kept}
    real_input = builtins.input
    builtins.input = lambda *_a: "y"
    try:
        VD.handle_mismatches(imgs, lbls, unlabeled, set())
    finally:
        builtins.input = real_input
    check(not [f for f in os.listdir(imgs)], "the unlabeled images left all_images")
    check(set(kept.values()) <= held_under(ds), "all three sit under <dataset>/_superseded/<stamp>/ with the same hash")

print("restructure_boundingbox_directories")
import restructure_boundingbox_directories as RB
with tempfile.TemporaryDirectory(prefix="rb_") as d:
    dest = os.path.join(d, "dest")
    old = write_images(os.path.join(dest, "all_images"))
    RB.clean_directory(dest)
    check(os.path.isdir(dest) and not os.listdir(dest), "clean_directory leaves an empty folder")
    check(set(old.values()) <= held_under(d), "the previous images sit under <parent>/_superseded/<stamp>/")
    ds = os.path.join(d, "ds2")
    imgs = os.path.join(ds, "all_images")
    lbls = os.path.join(ds, "all_labels")
    os.makedirs(lbls)
    kept = write_images(imgs)
    real_input = builtins.input
    builtins.input = lambda *_a: "y"
    try:
        RB.handle_mismatches(imgs, lbls, {os.path.splitext(os.path.basename(p))[0] for p in kept}, set())
    finally:
        builtins.input = real_input
    check(set(kept.values()) <= held_under(ds), "handle_mismatches: images set aside with the same hash")

sys.exit(1 if _fail else 0)
