"""Module: _superseded
Purpose: Move an image, or a folder holding images, into a dated holding
    folder instead of deleting it. Images are never deleted (Lauren,
    2026-10-15): every VEGA path that used to remove or replace one calls
    supersede() so the bytes stay on disk.
Inputs: a file or folder path, and the base folder the holding folder hangs from.
Outputs: <base>/_superseded/<YYYYMMDD-HHMMSS>/<path relative to base>, the
    original bytes unchanged. No code in VEGA prunes a _superseded folder.
    Readers that walk a whole project skip these folders (is_superseded_path).
"""

import errno
import os
import shutil
from datetime import datetime, timedelta, timezone

SUPERSEDED_DIR_NAME = "_superseded"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".jfif", ".png", ".tif", ".tiff", ".webp",
                    ".bmp", ".gif", ".heic", ".heif")
# The only file types a tree may hold and still be removed outright.
NON_IMAGE_EXTENSIONS = (".json", ".csv", ".txt", ".yaml", ".yml", ".log", ".md", ".lock")
# Model files a step reset has always removed; allowed only where a caller asks.
MODEL_ARTIFACT_EXTENSIONS = (".pt", ".pth", ".onnx", ".engine", ".npy", ".npz")
# Files that are a person's annotation work even though they hold no image.
ANNOTATION_RECORD_NAMES = ("segmentations.json", "label_provenance.csv", "export_manifest.csv")
ANNOTATION_RECORD_DIRS = ("all_labels",)
STAMP_FORMAT = "%Y%m%d-%H%M%S"
_AST = timezone(timedelta(hours=-4))
_MAX_SUFFIX_TRIES = 100000


def stamp_now():
    """The current Atlantic Standard Time as YYYYMMDD-HHMMSS."""
    return datetime.now(_AST).strftime(STAMP_FORMAT)


def is_superseded_path(path, root=None):
    """True when path sits inside a _superseded holding folder.

    Parameters:
        path: any file or folder path.
        root: the project (or other) folder the path is read from. When given,
            only the part of path below root is examined, so a project that
            itself lives under a _superseded folder still reads as live.
    Returns:
        bool
    """
    text = os.path.abspath(path)
    if root is not None:
        rel = os.path.relpath(text, os.path.abspath(root))
        if rel.startswith(os.pardir):
            return False
        text = rel
    return SUPERSEDED_DIR_NAME in text.split(os.sep)


def is_image_file(path):
    """True for a regular (non-symlink) file with an image extension."""
    return (path.lower().endswith(IMAGE_EXTENSIONS)
            and os.path.isfile(path) and not os.path.islink(path))


def _regular_files(path):
    """Yield every regular (non-symlink) file under path, or path itself."""
    if os.path.islink(path):
        return
    if os.path.isfile(path):
        yield path
        return
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            full = os.path.join(dirpath, name)
            if not os.path.islink(full):
                yield full


def holds_image_files(path):
    """True when path is an image file or a folder with one anywhere inside.

    Symlinks are not counted: a link holds no image bytes.
    """
    return any(f.lower().endswith(IMAGE_EXTENSIONS) for f in _regular_files(path))


def is_removable_tree(path, extra_extensions=()):
    """True when every regular file under path is a known non-image type.

    The safe default is to keep: a file of any type not on the list
    (NON_IMAGE_EXTENSIONS plus extra_extensions) makes the tree non-removable,
    so an image format nobody listed is moved, not lost. Symlinks and empty
    folders do not count against removal.

    Parameters:
        path: file or folder.
        extra_extensions: more extensions the caller accepts as removable.
    """
    allowed = tuple(NON_IMAGE_EXTENSIONS) + tuple(extra_extensions)
    return all(f.lower().endswith(allowed) for f in _regular_files(path))


def holds_annotation_work(path):
    """True when path holds an outline record or label file a person produced.

    segmentations.json, label_provenance.csv, export_manifest.csv or an
    all_labels folder anywhere under path. These carry no image but are
    annotation work and are never deleted.
    """
    if os.path.islink(path):
        return False
    if os.path.isfile(path):
        return os.path.basename(path) in ANNOTATION_RECORD_NAMES
    if os.path.basename(os.path.normpath(path)) in ANNOTATION_RECORD_DIRS:
        return True
    for dirpath, dirs, files in os.walk(path):
        if any(d in ANNOTATION_RECORD_DIRS for d in dirs):
            return True
        if any(f in ANNOTATION_RECORD_NAMES for f in files):
            return True
    return False


def _suffixed(dest, n, is_dir):
    """dest with -n added before the extension (X-1.jpeg); folders get X-1."""
    if n == 0:
        return dest
    if is_dir:
        return f"{dest}-{n}"
    stem, ext = os.path.splitext(dest)
    return f"{stem}-{n}{ext}"


def _reserve_and_move(src, dest, is_dir):
    """Move src to the first free name at or after dest, never overwriting.

    Each candidate is claimed with an operation that fails when the name
    exists: os.link (then the source name is unlinked) for a file, os.symlink
    for a link, os.mkdir for a folder (the folder is then renamed onto the
    claimed empty one). Two movers racing for one name cannot both win it.
    Falls back to an exclusive placeholder plus copy where hard links are not
    possible (another filesystem).
    """
    for n in range(_MAX_SUFFIX_TRIES):
        cand = _suffixed(dest, n, is_dir)
        try:
            if is_dir:
                os.mkdir(cand)
                try:
                    os.rename(src, cand)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        os.rmdir(cand)
                        raise
                    shutil.copytree(src, cand, symlinks=True, dirs_exist_ok=True)
                    shutil.rmtree(src)
            elif os.path.islink(src):
                os.symlink(os.readlink(src), cand)
                os.unlink(src)
            else:
                try:
                    os.link(src, cand)
                except FileExistsError:
                    raise
                except OSError:
                    fd = os.open(cand, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    shutil.copy2(src, cand)
                os.unlink(src)
            return cand
        except FileExistsError:
            continue
    raise OSError(f"no free name for {dest} after {_MAX_SUFFIX_TRIES} tries")


def supersede(path, base_dir=None, stamp=None, log=print, holding_root=None, rel_name=None):
    """Move path into <base_dir>/_superseded/<stamp>/<path relative to base_dir>.

    The holding folder is created if missing and an existing one is reused, so
    several moves in one operation share a stamp. A destination that already
    exists is never overwritten: the new one gets -1, -2 ... before its
    extension, claimed exclusively so concurrent movers cannot collide.

    Parameters:
        path: the file or folder to set aside; a missing path returns None.
        base_dir: the folder the holding folder hangs from; default the
            parent of path. path must sit inside it.
        stamp: YYYYMMDD-HHMMSS; default the current time.
        log: a callable taking one line.
        holding_root: put the _superseded folder here instead of under base_dir
            (for content whose own tree must stay clean, such as a git working
            tree). The result is <holding_root>/_superseded/<stamp>/<rel>.
        rel_name: the relative path to use under the stamp folder instead of
            the path relative to base_dir.
    Returns:
        The new location, or None when path did not exist.
    Raises:
        ValueError: path is not inside base_dir.
        OSError: the move failed; the source is left where it was.
    """
    if not os.path.lexists(path):
        return None
    base = os.path.abspath(base_dir or os.path.dirname(os.path.abspath(path)))
    rel = os.path.relpath(os.path.abspath(path), base)
    if rel.startswith(os.pardir):
        raise ValueError(f"{path} is not inside {base}")
    if rel_name:
        rel = rel_name
    holding = os.path.join(holding_root or base, SUPERSEDED_DIR_NAME, stamp or stamp_now())
    is_dir = os.path.isdir(path) and not os.path.islink(path)
    dest = os.path.join(holding, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    final = _reserve_and_move(path, dest, is_dir)
    log(f"[superseded] moved {path} to {final}")
    return final


def supersede_if_images(path, base_dir=None, stamp=None, log=print, extra_removable=()):
    """Set path aside unless it is plainly safe to remove, then remove it.

    A tree is removed only when every regular file in it is a known non-image
    type (is_removable_tree). Anything else, including an image format nobody
    listed, is moved whole. Returns the new location, or None when path was
    absent or removed.
    """
    if not os.path.lexists(path):
        return None
    if not os.path.islink(path) and not is_removable_tree(path, extra_removable):
        return supersede(path, base_dir=base_dir, stamp=stamp, log=log)
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    return None
