"""Persisted clip-tree file index (Task 1.4, 2026-08-26).

The TCRMP clip tree holds ~167k files in ~800 directories, and three code
paths used to re-walk all of it on every use: Step-3 image selection
(chooseImages build_clip_index), the Step-4 routing configure (placePoints
build_clip_index + build_cpc_index, two walks), and the lores generator
(make_lores_variants). This module gives them one shared, persisted index:

  - One full walk builds a JSON index under supporting_data/clip_index/
    (gitignored), one file per indexed tree, keyed by a hash of its abspath.
  - The index records every non-hidden file relpath PLUS every directory's
    mtime_ns. Staleness check = re-stat the ~800 recorded dirs (no readdir):
    any file add/remove/rename touches its parent directory's mtime, and a
    brand-new directory touches its (recorded) parent, so either shows up as
    a changed or missing dir stat. In-place file content edits do not change
    paths and are irrelevant to a path index.
  - Stale, missing, or unreadable index -> full walk + atomic rewrite.
  - Granularity caveat: directory mtimes advance at kernel-clock granularity
    (a few ms), so a file added in the same tick as the scan is undetectable
    until the next mutation. Irrelevant at production timescales (data lands
    minutes to months apart from index builds).

Consumers call list_files(clip_dir) and derive their own stem maps, so their
filtering/ranking behavior is unchanged; each keeps its original walk as a
last-resort fallback if this module errors.
"""
import hashlib
import json
import os
from datetime import datetime

SCHEMA_VERSION = 1

# How the most recent list_files() call was served: "cache" or "walk".
# Tests and logs read this to prove the hit path is actually taken.
LAST_INDEX_SOURCE = None


def _index_dir():
    """Where index files live. supporting_data/ is already gitignored.
    Overridable for tests via TCRMP_CLIP_INDEX_DIR."""
    override = os.environ.get("TCRMP_CLIP_INDEX_DIR")
    if override:
        return override
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "supporting_data", "clip_index")


def _index_path(clip_dir):
    key = hashlib.sha1(os.path.abspath(clip_dir).encode()).hexdigest()[:12]
    return os.path.join(_index_dir(), f"clip_index_{key}.json")


def _scan(clip_dir):
    """One full walk. Returns (files, dirs): files is a sorted list of
    relpaths of every non-hidden file; dirs maps every visited directory
    relpath ('' for the root) to its mtime_ns. Hidden dirs (.AppleDouble,
    .git) and dot-files are skipped, mirroring the historical walkers."""
    files = []
    dirs = {}
    for root, dnames, fnames in os.walk(clip_dir):
        dnames[:] = [d for d in dnames if not d.startswith(".")]
        rel = os.path.relpath(root, clip_dir)
        rel = "" if rel == "." else rel
        try:
            dirs[rel] = os.stat(root).st_mtime_ns
        except OSError:
            continue
        for fn in fnames:
            if fn.startswith("."):
                continue
            files.append(os.path.join(rel, fn) if rel else fn)
    files.sort()
    return files, dirs


def _is_current(clip_dir, recorded_dirs):
    """True when every recorded directory still exists with the same mtime_ns.
    ~800 stat calls, no directory reads."""
    root_abs = os.path.abspath(clip_dir)
    for rel, mtime_ns in recorded_dirs.items():
        p = os.path.join(root_abs, rel) if rel else root_abs
        try:
            if os.stat(p).st_mtime_ns != mtime_ns:
                return False
        except OSError:
            return False
    return True


def _load(clip_dir):
    """Read + validate the persisted index. Returns the file list or None."""
    path = _index_path(clip_dir)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    if data.get("clip_dir") != os.path.abspath(clip_dir):
        return None
    dirs = data.get("dirs")
    files = data.get("files")
    if not isinstance(dirs, dict) or not isinstance(files, list):
        return None
    if not _is_current(clip_dir, dirs):
        return None
    return files


def _write(clip_dir, files, dirs):
    """Atomic index write; failure to persist never fails the caller."""
    path = _index_path(clip_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "built": datetime.now().strftime("%Y-%m-%d %H:%M AST"),
            "clip_dir": os.path.abspath(clip_dir),
            "file_count": len(files),
            "dirs": dirs,
            "files": files,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except OSError:
        pass


def list_files(clip_dir, rebuild=False):
    """Absolute paths of every non-hidden file under clip_dir.

    Paths are joined on clip_dir exactly as the caller passed it (absolute in,
    absolute out; relative in, relative out) so listings are byte-identical to
    what the historical walkers produced. Serves from the persisted index when
    its directory-mtime sentinel is intact; otherwise walks the tree and
    rewrites the index. Returns [] for a missing/empty clip_dir (matching the
    historical walkers' behavior). rebuild=True forces the walk."""
    global LAST_INDEX_SOURCE
    if not clip_dir or not os.path.isdir(clip_dir):
        LAST_INDEX_SOURCE = "walk"
        return []
    if not rebuild:
        files = _load(clip_dir)
        if files is not None:
            LAST_INDEX_SOURCE = "cache"
            return [os.path.join(clip_dir, rel) for rel in files]
    files, dirs = _scan(clip_dir)
    _write(clip_dir, files, dirs)
    LAST_INDEX_SOURCE = "walk"
    return [os.path.join(clip_dir, rel) for rel in files]
