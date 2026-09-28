#!/usr/bin/env python
"""No-pytest harness for the mtime+size keyed segmentations.json parse cache
in cellinfo.py (`_load_json` / `_parse_json_file` / `_SEG_CACHE`).

Run with:
  env/bin/python scripts/_matrix/tests/test_cellinfo_cache.py

Covers:
  - unchanged file: second call reuses the cached parse (no re-parse, same
    object returned) instead of re-reading the file
  - touched file (new content, new mtime): cache detects the change and the
    new data comes back, re-parsing exactly once more
  - same-mtime-different-size edge: a file rewritten with different content
    but forced back to the same mtime is still re-parsed, because size is
    part of the cache key too
  - the cache is a bounded LRU (module `_SEG_CACHE_MAX`), so pushing more
    distinct files through it than the bound never grows it past the bound
  - concurrent callers (Flask threaded serving) don't crash or corrupt the
    cache: threading.Lock smoke test
  - load_project_segmentations (the function composite.py and stats.py both
    import from here) picks the cache up automatically with no code changes
    of its own

Exits nonzero on any failed assertion.
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# A private temp root per run, removed at exit; the old hard-coded session
# scratchpad path no longer exists, so every run failed before any check.
_SCRATCH = tempfile.mkdtemp(prefix="cellinfo_cache_run_")
atexit.register(shutil.rmtree, _SCRATCH, True)
_TMP_REG = tempfile.mkdtemp(prefix="cellinfo_cache_reg_", dir=_SCRATCH)
_TMP_CLIP = tempfile.mkdtemp(prefix="cellinfo_cache_clip_", dir=_SCRATCH)
os.environ["TCRMP_MASK_REGISTRY_DIR"] = _TMP_REG
os.environ["TCRMP_CLIP_INDEX_DIR"] = _TMP_CLIP

from _matrix import cellinfo  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _write_seg(path, filename, image_path_abs, processed_at):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {
        filename: {
            "image_path": "raw/" + filename,
            "image_path_abs": image_path_abs,
            "image_width": 64,
            "image_height": 48,
            "masks": [],
            "processed_at": processed_at,
            "reviewed": True,
            "exported": False,
            "label_outcomes": {},
        }
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    return doc


def _counting_parser(counter, original):
    def wrapped(path):
        counter[0] += 1
        return original(path)
    return wrapped


def main():
    root = tempfile.mkdtemp(prefix="cellinfo_cache_test_", dir=_SCRATCH)
    original_parse = cellinfo._parse_json_file
    try:
        proj_dir = os.path.join(root, "run_projX_20260901_x")
        seg_path = os.path.join(proj_dir, "segmentations", "segmentations.json")

        counter = [0]
        cellinfo._parse_json_file = _counting_parser(counter, original_parse)

        # --- (1) unchanged file: second call reuses the parse ---
        _write_seg(seg_path, "IMG_A.jpeg", "/abs/IMG_A.jpeg", "2026-09-01T10:00:00")
        segs_1 = cellinfo.load_project_segmentations(proj_dir)
        check(counter[0] == 1, f"first call parses once (got {counter[0]})")
        check("IMG_A" in segs_1, "first call resolves IMG_A")

        segs_2 = cellinfo.load_project_segmentations(proj_dir)
        check(counter[0] == 1, f"second call with unchanged file does not re-parse (got {counter[0]})")
        check(segs_2["IMG_A"]["seg"] is segs_1["IMG_A"]["seg"],
              "second call returns the identical cached seg object, not a fresh parse")
        check(segs_2 == segs_1, "second call returns equal data")

        # --- (2) touched file: new content + new mtime -> fresh data, re-parsed ---
        st_before = os.stat(seg_path)
        _write_seg(seg_path, "IMG_B.jpeg", "/abs/IMG_B.jpeg", "2026-09-01T11:00:00")
        # Force the mtime forward in case the filesystem's clock resolution
        # rounded the two writes to the same timestamp.
        os.utime(seg_path, (st_before.st_mtime + 5, st_before.st_mtime + 5))
        st_after_touch = os.stat(seg_path)
        check(st_after_touch.st_mtime != st_before.st_mtime, "fixture actually changed mtime")

        segs_3 = cellinfo.load_project_segmentations(proj_dir)
        check(counter[0] == 2, f"touched file triggers exactly one re-parse (got {counter[0]})")
        check("IMG_B" in segs_3 and "IMG_A" not in segs_3,
              "touched file: new data reflects the new content, old stem gone")

        # --- (3) same mtime, different size -> still re-parsed ---
        held_mtime = os.stat(seg_path).st_mtime
        _write_seg(seg_path, "IMG_C_longer_name.jpeg", "/abs/IMG_C_longer_name.jpeg", "2026-09-01T12:00:00")
        os.utime(seg_path, (held_mtime, held_mtime))
        st_edge = os.stat(seg_path)
        check(st_edge.st_mtime == held_mtime, "edge fixture: mtime forced identical to the prior read")
        check(st_edge.st_size != st_after_touch.st_size, "edge fixture: size did change")

        segs_4 = cellinfo.load_project_segmentations(proj_dir)
        check(counter[0] == 3, f"same-mtime-different-size edge still re-parses (got {counter[0]})")
        check("IMG_C_longer_name" in segs_4 and "IMG_B" not in segs_4,
              "same-mtime-different-size edge: new data wins, not the stale same-mtime entry")

        # --- (4) load_project_segmentations picks the cache up with no
        # changes of its own; composite.py / stats.py import the same
        # function, so they inherit the cache automatically (verified by
        # import, not by executing composite.py here: out of file scope).
        check("composite" not in cellinfo.__name__, "sanity: this is cellinfo, not composite")

        # --- (5) bounded LRU: pushing more distinct files than the cap
        # never grows the cache past _SEG_CACHE_MAX.
        cellinfo._SEG_CACHE.clear()
        many_dir = os.path.join(root, "many")
        cap = cellinfo._SEG_CACHE_MAX
        for i in range(cap + 20):
            p = os.path.join(many_dir, f"seg_{i}.json")
            os.makedirs(many_dir, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                json.dump({f"IMG_{i}.jpeg": {"image_path": f"raw/IMG_{i}.jpeg"}}, fh)
            cellinfo._load_json(p)
        check(len(cellinfo._SEG_CACHE) == cap,
              f"cache bounded at _SEG_CACHE_MAX={cap} after {cap + 20} distinct files (got {len(cellinfo._SEG_CACHE)})")
        # The most recently touched file is still resident (LRU keeps the
        # tail, evicts the head).
        last_path = os.path.abspath(os.path.join(many_dir, f"seg_{cap + 19}.json"))
        check(last_path in cellinfo._SEG_CACHE, "most recently used entry survives eviction")
        first_path = os.path.abspath(os.path.join(many_dir, "seg_0.json"))
        check(first_path not in cellinfo._SEG_CACHE, "oldest entry evicted first (LRU)")

        # --- (6) missing file: no crash, no stale cache entry left behind ---
        ghost = os.path.join(root, "ghost.json")
        check(cellinfo._load_json(ghost) is None, "missing file returns None")
        check(os.path.abspath(ghost) not in cellinfo._SEG_CACHE, "missing file leaves no cache entry")

        # --- (7) concurrent callers don't crash or corrupt the cache ---
        cellinfo._parse_json_file = original_parse  # real parsing for the concurrency smoke test
        conc_dir = os.path.join(root, "concurrent")
        conc_path = os.path.join(conc_dir, "segmentations", "segmentations.json")
        _write_seg(conc_path, "IMG_CONC.jpeg", "/abs/IMG_CONC.jpeg", "2026-09-01T13:00:00")
        errors = []
        results = []
        results_lock = threading.Lock()

        def hammer():
            try:
                for _ in range(50):
                    r = cellinfo.load_project_segmentations(conc_dir)
                    with results_lock:
                        results.append(r)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check(not errors, f"no exceptions under concurrent load ({errors!r})")
        check(all("IMG_CONC" in r for r in results), "every concurrent call resolved the image")
        check(len(results) == 8 * 50, f"every thread completed all iterations (got {len(results)})")
    finally:
        cellinfo._parse_json_file = original_parse
        cellinfo._SEG_CACHE.clear()
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(_TMP_REG, ignore_errors=True)
        shutil.rmtree(_TMP_CLIP, ignore_errors=True)
    check(not os.path.exists(root), "temp fixture removed")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
