"""A custom import that is not a readable image is set aside, never deleted
(Lauren, 2026-10-15).

ingest_files copies the upload to custom_imports/originals/ first. When the
file cannot be opened as an image, that copy moves to
custom_imports/_superseded/<stamp>/originals/<name> with its bytes unchanged,
and the refusal is still reported in skipped.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_custom_imports_keep_images.py
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
import io
import os
import sys
import tempfile

from PIL import Image

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, _SRC)
import custom_imports as ci

SUPERSEDED = "_superseded"
_fail = 0


def check(c, m):
    """Print PASS or FAIL for m and count failures."""
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def held_files(export_dir):
    """Paths of every file under custom_imports/_superseded."""
    root = os.path.join(export_dir, ci.SUBDIR, SUPERSEDED)
    return [os.path.join(dp, f) for dp, _d, fs in os.walk(root) for f in fs]


def jpeg_bytes():
    """Bytes of a small valid JPEG."""
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (9, 99, 199)).save(buf, "JPEG")
    return buf.getvalue()


with tempfile.TemporaryDirectory(prefix="ci_keep_") as export:
    garbage = b"this is not an image at all" * 50
    good = jpeg_bytes()
    truncated = good[: len(good) // 3]
    items = [("junk.jpg", garbage), ("cut.jpg", truncated), ("fine.jpg", good)]
    entries, skipped = ci.ingest_files(export, items, log=lambda *_: None)
    check(len(entries) == 1 and entries[0]["original_name"] == "fine.jpg", "the readable image is imported")
    check({n for n, _r in skipped} == {"junk.jpg", "cut.jpg"}, "both unreadable uploads are reported as skipped")
    check(all("not a readable image" in r for _n, r in skipped), "the reason still says not a readable image")
    held = {hashlib.sha256(open(p, "rb").read()).hexdigest(): p for p in held_files(export)}
    check(hashlib.sha256(garbage).hexdigest() in held, "the non-image upload sits in custom_imports/_superseded/<stamp>/ with its bytes")
    check(hashlib.sha256(truncated).hexdigest() in held, "the truncated JPEG sits there too, same hash")
    check(all(os.sep + "originals" + os.sep in p for p in held.values()), "the relative path originals/<name> is kept")
    left = os.listdir(os.path.join(export, ci.SUBDIR, "originals"))
    check(left == [os.listdir(os.path.join(export, ci.SUBDIR, "originals"))[0]] and len(left) == 1,
          "only the imported image's original remains in originals/")
    # the same junk name again does not overwrite the held copy
    ci.ingest_files(export, [("junk.jpg", garbage + b"2")], log=lambda *_: None)
    check(len(held_files(export)) == 3, "a second unreadable upload with the same name adds a third held file")

sys.exit(1 if _fail else 0)
