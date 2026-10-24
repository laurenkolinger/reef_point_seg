"""scripts/_superseded.py: the holding-folder helper every never-delete path uses.

Covers is_superseded_path, the shared image list (.webp .bmp .gif .jfif .heic
included), the tree rule (a tree is removable only when every regular file in
it is a known non-image type), a .webp-only folder being moved and not removed,
exclusive destination reservation under 32 concurrent moves of one relative
path (suffix before the extension: X-1.jpeg), and a holding root outside the
base folder.

Run: env/bin/python scripts/tests/test_superseded_helper.py   (CUDA_VISIBLE_DEVICES="")
"""
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import _superseded as sup  # noqa: E402

_R = []
_TMP = []


def check(c, m):
    if not c:
        raise AssertionError(m)


def run(fn):
    try:
        fn()
        _R.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _tmp():
    d = tempfile.mkdtemp(prefix="sup_")
    _TMP.append(d)
    return d


def _w(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def test_is_superseded_path():
    check(sup.is_superseded_path("/a/_superseded/20261015-101010/b.json"), "component match")
    check(not sup.is_superseded_path("/a/b/c.json"), "live path")
    check(not sup.is_superseded_path("/a/_superseded_not/b"), "only the exact name")
    check(sup.is_superseded_path("/p/_superseded/x/y.json", root="/p"), "relative to root")
    check(not sup.is_superseded_path("/q/_superseded/p/live/y.json", root="/q/_superseded/p"),
          "a root that itself sits under _superseded does not condemn a live path inside it")


def test_image_list_includes_the_extra_types():
    d = _tmp()
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp", ".gif", ".jfif", ".heic"):
        p = _w(os.path.join(d, "a" + ext.upper()))
        check(sup.is_image_file(p), f"{ext} is an image")


def test_webp_only_folder_is_moved_not_removed():
    d = _tmp()
    h = hashlib.sha256(b"webp-bytes").hexdigest()
    _w(os.path.join(d, "f", "x.webp"), b"webp-bytes")
    dest = sup.supersede_if_images(os.path.join(d, "f"), base_dir=d, log=lambda m: None)
    check(dest and not os.path.exists(os.path.join(d, "f")), "folder left its place")
    kept = [os.path.join(dp, n) for dp, _x, fs in os.walk(dest) for n in fs]
    check(len(kept) == 1 and hashlib.sha256(open(kept[0], "rb").read()).hexdigest() == h, "bytes kept")


def test_tree_rule_removes_only_known_non_image_trees():
    d = _tmp()
    _w(os.path.join(d, "ok", "a.json")); _w(os.path.join(d, "ok", "sub", "b.CSV"))
    _w(os.path.join(d, "ok", "c.lock"))
    check(sup.is_removable_tree(os.path.join(d, "ok")), "json/csv/lock only is removable")
    _w(os.path.join(d, "odd", "a.json")); _w(os.path.join(d, "odd", "b.xyz"))
    check(not sup.is_removable_tree(os.path.join(d, "odd")), "an unknown type blocks removal")
    check(sup.supersede_if_images(os.path.join(d, "odd"), base_dir=d, log=lambda m: None),
          "the odd folder was moved")
    check(sup.supersede_if_images(os.path.join(d, "ok"), base_dir=d, log=lambda m: None) is None
          and not os.path.exists(os.path.join(d, "ok")), "the plain folder was removed")
    os.makedirs(os.path.join(d, "empty"))
    check(sup.is_removable_tree(os.path.join(d, "empty")), "an empty folder is removable")


def test_symlinks_do_not_make_a_tree_unremovable():
    d = _tmp()
    _w(os.path.join(d, "t", "real.jpg"))
    os.makedirs(os.path.join(d, "l"))
    os.symlink(os.path.join(d, "t", "real.jpg"), os.path.join(d, "l", "a.jpg"))
    check(sup.is_removable_tree(os.path.join(d, "l")), "a link holds no bytes")


def test_suffix_goes_before_the_extension():
    d = _tmp()
    stamp = "20261015-101010"
    outs = []
    for i in range(3):
        _w(os.path.join(d, "r", "X.jpeg"), f"v{i}".encode())
        outs.append(sup.supersede(os.path.join(d, "r", "X.jpeg"), base_dir=d, stamp=stamp, log=lambda m: None))
    names = sorted(os.path.basename(o) for o in outs)
    check(names == ["X-1.jpeg", "X-2.jpeg", "X.jpeg"], f"names: {names}")


def test_concurrent_moves_of_one_relative_path_never_overwrite():
    d = _tmp()
    stamp = "20261015-101010"
    n = 32
    srcs = []
    for i in range(n):
        # distinct sources whose relative path to base maps to the SAME destination
        srcs.append(_w(os.path.join(d, f"src{i}", "r", "X.jpeg"), f"v{i}".encode()))
    results, errs = [], []
    barrier = threading.Barrier(n)

    def go(i):
        try:
            barrier.wait()
            results.append(sup.supersede(srcs[i], base_dir=os.path.join(d, f"src{i}"),
                                         stamp=stamp, holding_root=os.path.join(d, "hold"),
                                         log=lambda m: None))
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check(not errs, f"errors: {errs}")
    check(len(set(results)) == n, f"{len(set(results))} distinct destinations of {n}")
    got = sorted(open(r, "rb").read() for r in results)
    check(got == sorted(f"v{i}".encode() for i in range(n)), "every version's bytes survive")


def test_holding_root_outside_the_base():
    d = _tmp()
    hold = _tmp()
    _w(os.path.join(d, "items", "U1", "crop.jpg"), b"c")
    dest = sup.supersede(os.path.join(d, "items", "U1"), base_dir=d, holding_root=hold,
                         rel_name="review_items/U1", stamp="20261015-101010", log=lambda m: None)
    check(dest.startswith(hold + os.sep) and dest.endswith(os.path.join("20261015-101010", "review_items", "U1")),
          f"destination: {dest}")
    check(os.path.isfile(os.path.join(dest, "crop.jpg")), "contents kept")
    check(not os.path.exists(os.path.join(d, "_superseded")), "nothing created under the base")


def test_a_missing_path_and_an_outside_path():
    d = _tmp()
    check(sup.supersede(os.path.join(d, "nope"), base_dir=d, log=lambda m: None) is None, "missing -> None")
    other = _w(os.path.join(_tmp(), "z.jpg"))
    try:
        sup.supersede(other, base_dir=d, log=lambda m: None)
    except ValueError:
        return
    raise AssertionError("a path outside base_dir must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            run(fn)
    for t in _TMP:
        shutil.rmtree(t, ignore_errors=True)
    left = [t for t in _TMP if os.path.exists(t)]
    print("fixtures removed" if not left else f"fixtures NOT removed: {left}")
    ok = sum(1 for _, p, _ in _R if p)
    for n, p, det in _R:
        if not p:
            print(f"\n--- {n}\n{det}")
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) and not left else 1)
