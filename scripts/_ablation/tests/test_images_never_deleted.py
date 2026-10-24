"""--force on the Coral Intel offline augmentation never deletes an image
(Lauren, 2026-10-15).

clear_managed_outputs used to remove the previous images/ folder. It must move
a folder holding real image files to <out>/_superseded/<stamp>/ with every
byte intact, still clear the metadata files it owns, remove a folder of
symlinks only, and never touch an entry it does not own.

Run: env/bin/python scripts/_ablation/tests/test_images_never_deleted.py
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
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coral_intel_offline_aug as ca  # noqa: E402

SUPERSEDED = "_superseded"
FORMATS = (("a.png", "PNG"), ("b.jpg", "JPEG"))
_fail = 0


def check(c, m):
    """Print PASS or FAIL for m and count failures."""
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def sha(p):
    """SHA-256 hex digest of the file at p."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_images(folder):
    """Write a real PNG and JPEG into folder; return the set of their digests."""
    os.makedirs(folder, exist_ok=True)
    out = set()
    for i, (name, fmt) in enumerate(FORMATS):
        p = os.path.join(folder, name)
        Image.new("RGB", (20 + i, 20), (i * 90, 30, 30)).save(p, fmt)
        out.add(sha(p))
    return out


def held(root):
    """Digests of every file under a _superseded folder below root."""
    return {sha(os.path.join(dp, f)) for dp, _d, fs in os.walk(root)
            if SUPERSEDED in dp.split(os.sep) for f in fs}


with tempfile.TemporaryDirectory(prefix="ca_clear_") as d:
    out = Path(d) / "out"
    digests = make_images(out / "images")
    (out / "labels").mkdir()
    (out / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (out / "data.yaml").write_text("names: {0: x}\n")
    (out / "mine_not.txt").write_text("keep me\n")
    ca.clear_managed_outputs(out, "labels")
    check(digests <= held(d), "the previous images sit under <out>/_superseded/<stamp>/ with the same hash")
    check(not (out / "images").exists(), "images/ is cleared from its old place")
    check(not (out / "data.yaml").exists() and not (out / "labels").exists(), "metadata files and label folders are still cleared")
    check((out / "mine_not.txt").read_text() == "keep me\n", "an entry the tool does not own is untouched")

with tempfile.TemporaryDirectory(prefix="ca_clear_links_") as d:
    out = Path(d) / "out"
    real = Path(d) / "elsewhere"
    digests = make_images(real)
    (out / "images").mkdir(parents=True)
    for n, _f in FORMATS:
        os.symlink(real / n, out / "images" / n)
    ca.clear_managed_outputs(out, "labels")
    check(not (out / "images").exists() and not (out / SUPERSEDED).exists(),
          "an images/ folder of symlinks is removed without a holding folder")
    check(digests <= {sha(real / n) for n, _f in FORMATS}, "the link targets are untouched")

with tempfile.TemporaryDirectory(prefix="ca_clear_second_") as d:
    out = Path(d) / "out"
    first = make_images(out / "images")
    ca.clear_managed_outputs(out, "labels")
    second = make_images(out / "images")
    ca.clear_managed_outputs(out, "labels")
    check(first <= held(d) and second <= held(d), "two forced rebuilds keep both generations of images")

sys.exit(1 if _fail else 0)
