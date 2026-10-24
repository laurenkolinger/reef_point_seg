"""A retracted review item is set aside outside the public site's git tree.

The review site folder is a git working tree and commit_push runs git add -A.
Images are never deleted, so remove_item / remove_project move the item folder
to <expert library>/_superseded/<stamp>/review_items/<uid>, never inside the
site repo, where the next commit would re-publish the retracted crop. The
site's .gitignore also lists _superseded/, and ensure_repo re-asserts every
line on each call, not only when the file or the lock line is missing.

Uses a temporary git repository; the expert library root is a temp folder via
TCRMP_EXPERT_LIBRARY_DIR (never the live inprocess/_expert_id_library).

Run: env/bin/python scripts/_reefreview/tests/test_review_removal_not_published.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _SCRIPTS)
sys.dont_write_bytecode = True

_LIB = tempfile.mkdtemp(prefix="sup_lib_")
os.environ["TCRMP_EXPERT_LIBRARY_DIR"] = _LIB
for k, v in (("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@example.org"),
             ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@example.org")):
    os.environ[k] = v

from PIL import Image  # noqa: E402
from _reefreview.review_repo import ReviewRepo  # noqa: E402

_R = []
_TMP = [_LIB]


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


def _jpeg(path, shade):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8), (shade, 5, 5)).save(path, "JPEG")
    return path


def _repo():
    root = tempfile.mkdtemp(prefix="sup_rr_")
    _TMP.append(root)
    repo = ReviewRepo(os.path.join(root, "REVIEW"), remote_url="", log_fn=lambda m: None)
    repo.add_item("U1", {"project_id": "P1"}, crop_src=_jpeg(os.path.join(root, "s", "c1.jpg"), 10))
    repo.add_item("U2", {"project_id": "P2"}, crop_src=_jpeg(os.path.join(root, "s", "c2.jpg"), 90))
    return repo


def _tree(repo):
    out = subprocess.run(["git", "-C", repo.root, "ls-tree", "-r", "--name-only", "HEAD"],
                         capture_output=True, text=True).stdout
    return out.split()


def test_removed_item_never_appears_in_the_committed_tree():
    repo = _repo()
    repo.commit_push("add", push=False)
    check(any(p.startswith("items/U1/") for p in _tree(repo)), "the item was published before removal")
    repo.remove_item("U1")
    repo.remove_project("P2")
    repo.commit_push("retract", push=False)
    tree = _tree(repo)
    check(not any("U1" in p or "U2" in p for p in tree), f"retracted crops still in HEAD: {tree}")
    check(not any("_superseded" in p for p in tree), f"holding folder published: {tree}")
    check(not os.path.exists(os.path.join(repo.root, "_superseded")), "nothing created inside the site repo")


def test_the_crops_are_kept_in_the_library_holding_folder():
    repo = _repo()
    repo.remove_item("U1")
    held = [os.path.join(dp, n) for dp, _d, fs in os.walk(os.path.join(_LIB, "_superseded")) for n in fs]
    check(any(p.endswith(os.path.join("review_items", "U1", "crop.jpg")) for p in held),
          f"crop kept outside the site repo: {held}")


def test_gitignore_has_every_line_on_every_ensure():
    repo = _repo()
    gi = os.path.join(repo.root, ".gitignore")
    check("_superseded/" in open(gi).read(), "fresh .gitignore lists _superseded/")
    with open(gi, "w") as fh:
        fh.write(".reefreview.lock\n*.tmp\n")      # an older file: lock line present, new line missing
    repo.ensure_repo()
    text = open(gi).read()
    check("_superseded/" in text and ".reefreview.lock" in text and "*.tmp" in text,
          f"missing line healed without losing the others: {text!r}")


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
