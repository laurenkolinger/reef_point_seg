#!/usr/bin/env python
"""No-pytest harness for the composite-to-project pipeline (Task H).

Covers:
  * build_composite(project_ids=...) filtering: only the named projects supply
    candidate masks, a preference pointing at a filtered-out project is
    ignored, an unknown id raises ValueError before anything is written, and
    composite.json records project_filter + source_projects (always).
  * create_training_project(): run-dir naming, the relative step4test symlink
    into the composite, project.json shape (project_type, composite block,
    skipped steps, current_step 6), the run README, the linked_projects
    append, and the collision suffix.
  * builder.scan_projects / stats.compute_stats skip a composite_training run
    dir entirely, so a training-set project never feeds the matrix back into
    itself even when a ledger is planted inside it.

Run with:
  env/bin/python scripts/_matrix/tests/test_composite_projects.py
"""

# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _matrix.builder import build_matrix, scan_projects  # noqa: E402
from _matrix.cellinfo import cell_info  # noqa: E402
from _matrix.composite import (  # noqa: E402
    build_composite,
    composite_root,
    create_training_project,
)  # noqa: E402
from _matrix.preferences import Preferences  # noqa: E402
from _matrix.stats import compute_stats  # noqa: E402
from _matrix.tests._fixture_cell import (  # noqa: E402
    T101, T102, T2015, build_root, make_image, mask, seg_entry, write,
)
from _reefreview.mask_registry import MaskRegistry  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def _meta(out_dir):
    with open(os.path.join(out_dir, "composite.json"), encoding="utf-8") as fh:
        return json.load(fh)


def _labels(out_dir, stem):
    p = os.path.join(out_dir, "all_labels", stem + ".txt")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# 1. project_ids filtering
# ---------------------------------------------------------------------------

def test_project_filter(root):
    print("\nbuild_composite(project_ids=...):")
    unfiltered = build_composite(root, name="all_projects", by="LO")
    meta_all = _meta(unfiltered["out_dir"])
    check(meta_all.get("project_filter") is None,
          f"an unfiltered build records project_filter null (got {meta_all.get('project_filter')!r})")
    src_all = {s["project_id"]: s["n_masks"] for s in (meta_all.get("source_projects") or [])}
    check(src_all, f"source_projects is recorded even without a filter (got {src_all})")
    check("projA" in src_all and "projB" in src_all,
          f"both fixture projects contribute unfiltered (got {sorted(src_all)})")
    check(sum(src_all.values()) == meta_all["n_masks"],
          f"source_projects mask counts sum to n_masks ({sum(src_all.values())} vs {meta_all['n_masks']})")

    filtered = build_composite(root, name="only_a", by="LO", project_ids=["projA"])
    meta_a = _meta(filtered["out_dir"])
    check(meta_a.get("project_filter") == ["projA"],
          f"project_filter records the requested ids (got {meta_a.get('project_filter')!r})")
    src_a = {s["project_id"] for s in (meta_a.get("source_projects") or [])}
    check(src_a == {"projA"}, f"only the filtered project exports masks (got {sorted(src_a)})")
    check(meta_a["projects"] == ["projA"],
          f"the projects listing agrees with the filter (got {meta_a['projects']})")
    check(meta_a["n_masks"] < meta_all["n_masks"],
          f"the filtered build exports fewer masks ({meta_a['n_masks']} vs {meta_all['n_masks']})")

    # projB holds the only image projA never saw; it must be absent entirely.
    check(_labels(filtered["out_dir"], "TCRMP20151001_clip_BIT_T201") is None,
          "an image only the filtered-out project holds is not exported")

    unknown_raised = ""
    try:
        build_composite(root, name="bad_filter", by="LO", project_ids=["projA", "nope"])
    except ValueError as e:
        unknown_raised = str(e)
    check("nope" in unknown_raised,
          f"an unknown project id raises ValueError naming it (got {unknown_raised!r})")
    check(not os.path.exists(os.path.join(composite_root(root), "bad_filter")),
          "the rejected build wrote nothing")


def test_preference_outside_filter_is_ignored(root):
    print("\na preference pointing outside the filter:")
    # T101/PA: projA is manual, projB is expert. Prefer projB, then filter to
    # projA only: the preference cannot be honored, so projA's mask exports.
    Preferences(root).set(_stem(T101), "PA", "projB", [], "LO", "test")
    res = build_composite(root, name="pref_outside", by="LO",
                          project_ids=["projA"], labels=["PA"])
    meta = _meta(res["out_dir"])
    src = {s["project_id"] for s in (meta.get("source_projects") or [])}
    check(src == {"projA"},
          f"the filtered-out preferred project contributes nothing (got {sorted(src)})")
    with open(os.path.join(res["out_dir"], "composite_manifest.csv"), encoding="utf-8") as fh:
        rows = fh.read()
    check("projB" not in rows, "no manifest row names the filtered-out project")
    Preferences(root).clear(_stem(T101), "PA")


# ---------------------------------------------------------------------------
# 2. create_training_project
# ---------------------------------------------------------------------------

def test_create_training_project(root):
    print("\ncreate_training_project:")
    built = build_composite(root, name="trainset_one", by="LO")
    out_dir = built["out_dir"]

    res = create_training_project(root, "trainset_one", "lo")
    run_dir = res["run_dir"]
    base = os.path.basename(run_dir)
    check(os.path.isdir(run_dir), f"the run dir exists ({run_dir})")
    check(base.startswith("run_") and "_LO_train_trainset_one" in base,
          f"run dir is run_YY-MM-DD_INITIALS_train_<set> (got {base})")
    check(os.path.dirname(run_dir) == os.path.abspath(root),
          "the run dir is minted directly under the inprocess root")

    link = os.path.join(run_dir, "step4test_combinedAnnotate")
    check(os.path.islink(link), "step4test_combinedAnnotate is a symlink")
    check(os.readlink(link) == os.path.join("..", "_composite", "trainset_one"),
          f"the symlink is relative to the composite (got {os.readlink(link)!r})")
    check(os.path.realpath(link) == os.path.realpath(out_dir),
          "the symlink resolves to the composite export")
    check(os.path.isfile(os.path.join(link, "data.yaml")),
          "data.yaml is readable through the symlink (Train reads this layout)")
    check(os.path.isdir(os.path.join(link, "all_images")),
          "all_images/ is readable through the symlink (no image is duplicated)")

    with open(os.path.join(run_dir, "project.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    check(meta.get("project_type") == "composite_training",
          f"project.json marks the project type (got {meta.get('project_type')!r})")
    comp = meta.get("composite") or {}
    check(comp.get("name") == "trainset_one", f"composite.name (got {comp.get('name')!r})")
    check(comp.get("path") == os.path.join("inprocess", "_composite", "trainset_one"),
          f"composite.path is the relative export path (got {comp.get('path')!r})")
    check(isinstance(comp.get("source_projects"), list) and comp["source_projects"],
          f"composite.source_projects copied from composite.json (got {comp.get('source_projects')!r})")
    check(all(s.get("project_id") for s in comp["source_projects"]),
          f"every source carries its project id (got {comp['source_projects']})")
    check(any(s.get("name") == "Project A" for s in comp["source_projects"]),
          f"a live source project carries its display name (got {comp['source_projects']})")
    check(len(str(comp.get("created_at") or "")) == 19,
          f"composite.created_at is an AST stamp (got {comp.get('created_at')!r})")
    check(meta.get("initials") == "LO", f"initials uppercased (got {meta.get('initials')!r})")
    check(meta.get("name") == "TRAIN_trainset_one", f"display name (got {meta.get('name')!r})")
    check(meta.get("current_step") == 6, f"opens on Train (got {meta.get('current_step')!r})")
    check(res["project_id"] == meta.get("id"), "the returned project_id matches project.json id")
    check(res["display_name"] == meta.get("name"), "the returned display_name matches project.json")
    for s in ("1", "2", "3", "4"):
        st = (meta.get("steps") or {}).get(s) or {}
        check(st.get("status") == "skipped", f"step {s} is skipped (got {st.get('status')!r})")
        check("trainset_one" in str(st.get("note") or ""),
              f"step {s} note names the training set (got {st.get('note')!r})")
    check("step4test" not in (meta.get("steps") or {}),
          "no step4test entry is invented (regular projects carry none)")

    readme = os.path.join(run_dir, "README.md")
    check(os.path.isfile(readme), "the run dir carries a README.md")
    text = open(readme, encoding="utf-8").read()
    check("trainset_one" in text, "the README names the training set")
    check("projA" in text and "projB" in text, "the README names the source projects")

    linked = _meta(out_dir).get("linked_projects") or []
    check(linked == [base], f"composite.json linked_projects holds the run dir (got {linked})")

    # Collision: a second project for the same set on the same day.
    res2 = create_training_project(root, "trainset_one", "LO")
    base2 = os.path.basename(res2["run_dir"])
    check(base2 == base + "_2", f"a same-day repeat suffixes _2 (got {base2})")
    linked2 = _meta(out_dir).get("linked_projects") or []
    check(linked2 == [base, base2], f"both runs are linked (got {linked2})")

    missing = ""
    try:
        create_training_project(root, "no_such_set", "LO")
    except FileNotFoundError as e:
        missing = str(e)
    check("no_such_set" in missing, f"an unknown composite raises FileNotFoundError (got {missing!r})")

    bad = ""
    try:
        create_training_project(root, "trainset_one", "12345")
    except ValueError as e:
        bad = str(e)
    check(bad, f"bad initials raise ValueError (got {bad!r})")

    escaped = ""
    try:
        create_training_project(root, os.path.join("..", os.path.basename(base)), "LO")
    except FileNotFoundError as e:
        escaped = str(e)
    check(escaped, "a name that walks out of the composite root is refused")

    # A set with no images trains on nothing: refuse at mint time with a
    # message that says what to do, and leave no half-made run dir behind.
    empty_dir = os.path.join(composite_root(root), "empty_set")
    os.makedirs(os.path.join(empty_dir, "all_images"))
    os.makedirs(os.path.join(empty_dir, "all_labels"))
    write(os.path.join(empty_dir, "data.yaml"), "path: x\ntrain: all_images\nnc: 0\nnames:\n")
    before = set(os.listdir(root))
    empty_err = ""
    try:
        create_training_project(root, "empty_set", "LO")
    except ValueError as e:
        empty_err = str(e)
    check("no images" in empty_err,
          f"an image-free training set is refused with a clean message (got {empty_err!r})")
    check(set(os.listdir(root)) == before, "the refused mint left no run dir behind")
    return base, base2


def test_concurrent_mint_is_safe(root):
    """Twelve simultaneous mints: every one gets its own run dir, every one is
    recorded in linked_projects, and no reader can see a truncated
    composite.json. Without the lock plus the mkdir claim this loses entries
    and raises FileExistsError."""
    print("\nconcurrent create_training_project:")
    import threading
    build_composite(root, name="race_set", by="LO")
    out_dir = os.path.join(composite_root(root), "race_set")

    results, errors = [], []
    reader_bad = []
    stop = threading.Event()

    def _reader():
        # Hammer the file a real matrix listing would read while the minters
        # write it; any truncated read shows up as a JSON error here.
        while not stop.is_set():
            try:
                with open(os.path.join(out_dir, "composite.json"), encoding="utf-8") as fh:
                    json.load(fh)
            except FileNotFoundError:
                pass
            except ValueError as e:
                reader_bad.append(str(e))

    def _mint():
        try:
            results.append(create_training_project(root, "race_set", "LO"))
        except Exception as e:  # noqa: BLE001 - the point is that none escape
            errors.append(f"{type(e).__name__}: {e}")

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    threads = [threading.Thread(target=_mint) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    reader.join(timeout=5)

    check(not errors, f"no mint raised (got {errors})")
    check(len(results) == 12, f"every mint returned a project (got {len(results)})")
    dirs = [r["run_dir_name"] for r in results]
    check(len(set(dirs)) == 12, f"every mint got its own run dir (got {sorted(dirs)})")
    check(all(os.path.isdir(d) for d in (r["run_dir"] for r in results)),
          "every returned run dir exists on disk")
    linked = _meta(out_dir).get("linked_projects") or []
    check(sorted(linked) == sorted(dirs),
          f"linked_projects holds every run, none lost ({len(linked)} of {len(dirs)})")
    check(not reader_bad, f"a concurrent reader never saw a truncated composite.json "
                          f"(got {reader_bad[:2]})")
    return dirs


# ---------------------------------------------------------------------------
# 3. the cell payload names the training sets a cell exports into
# ---------------------------------------------------------------------------

def test_cell_lists_composites(root):
    print("\n/api/cell composites:")
    # Built above: all_projects (everything), only_a (projA only),
    # pref_outside (projA, labels=["PA"] only), trainset_one (everything).
    pa = cell_info(root, T101, "PA")
    check(isinstance(pa.get("composites"), list), "the cell payload carries a composites list")
    check("all_projects" in pa["composites"] and "only_a" in pa["composites"],
          f"a cell lists every training set holding it (got {pa['composites']})")
    check(pa["composites"] == sorted(pa["composites"]), "the names are sorted")

    ofav = cell_info(root, T101, "OFAV")
    check("pref_outside" not in ofav["composites"],
          f"a label the set never exported is not listed (got {ofav['composites']})")
    check("all_projects" in ofav["composites"],
          f"the unfiltered set still lists this label (got {ofav['composites']})")

    bit = cell_info(root, T2015, "PA")
    check("only_a" not in bit["composites"],
          f"an image the filtered set never held is not listed (got {bit['composites']})")

    pending = cell_info(root, T102, "DSTO")
    check(pending["composites"] == [],
          f"a cell no set exports gets an empty list (got {pending['composites']})")


# ---------------------------------------------------------------------------
# 4. builder / stats never scan a training-set project
# ---------------------------------------------------------------------------

def test_scan_skips_training_projects(root, run_basenames):
    print("\nbuilder + stats skip composite_training run dirs:")
    run_dir = os.path.join(root, run_basenames[0])
    # The ids scan_projects reports are project.json "id" values, NOT run dir
    # basenames (the id drops the run_ prefix), so every assertion below
    # compares against the real ids. Comparing basenames against ids can never
    # fail and would pass with the skip removed.
    train_ids, train_run_dirs = [], []
    for base in run_basenames:
        with open(os.path.join(root, base, "project.json"), encoding="utf-8") as fh:
            train_ids.append(json.load(fh)["id"])
        train_run_dirs.append(base)
    check(all(i and not i.startswith("run_") for i in train_ids),
          f"the fixture ids are real project.json ids (got {train_ids})")

    # Plant, inside the training project, one of each thing a scan reads:
    #   label_provenance.csv   the builder's ledger chain
    #   segmentations.json     the builder's legacy chain AND every STATS
    #                          record derived from a live project
    #   a registry row         keyed to this project's id, which STATS counts
    #                          only while the id belongs to a live project
    # With the skip in place none of it may reach the matrix or STATS.
    write(os.path.join(run_dir, "step4_routeChosenImages", "label_provenance.csv"),
          "basename,label,outcome,source,reviewer,at,project_id\n"
          "IMG_planted,PLANTED,found_manual,step4test,ZZ,2026-09-01T10:00:00,x\n")
    planted_img = "TCRMP20140930_clip_SCP_T999.jpeg"
    raw_dir = os.path.join(run_dir, "step5_segmentImages", "routed_input", "raw")
    img_path = make_image(raw_dir, planted_img)
    write(os.path.join(run_dir, "step5_segmentImages", "segmentations", "segmentations.json"),
          json.dumps({planted_img: seg_entry(img_path, [mask(0, "PLANTED", reviewer="ZZ")])},
                     indent=1))
    reg = MaskRegistry(os.path.join(root, "_mask_registry"))
    reg.upsert({
        "uid": "PLANTED-x0y0", "source_image": planted_img,
        "basename": os.path.splitext(planted_img)[0], "species": "PLANTED",
        "category": "", "source_type": "manual_click", "review": "", "expert_mode": "",
        "status": "accepted", "project_id": train_ids[0], "project_name": "planted",
    })

    ids = {p["project_id"] for p in scan_projects(root)}
    run_dirs = {p["run_dir"] for p in scan_projects(root)}
    for pid in train_ids:
        check(pid not in ids, f"scan_projects skips the training project id {pid}")
    for base in train_run_dirs:
        check(base not in run_dirs, f"scan_projects skips the training run dir {base}")
    check("projA" in ids and "projB" in ids, "the real annotation projects still scan")

    matrix = build_matrix(root)
    check("PLANTED" not in (matrix.get("labels") or []),
          "the planted ledger and segmentations contribute no label to the matrix")
    check("IMG_planted" not in (matrix.get("images") or []),
          "the planted ledger contributes no image to the matrix")
    check(os.path.splitext(planted_img)[0] not in (matrix.get("images") or []),
          "the planted segmentations contribute no image to the matrix")
    mids = {p["project_id"] for p in matrix.get("projects") or []}
    for pid in train_ids:
        check(pid not in mids, f"the matrix project list omits {pid}")

    stats = compute_stats(root)
    check(not any(s["code"] == "PLANTED" for s in stats.get("species") or []),
          "STATS counts no mask from the training project (segmentations or registry)")
    check(train_ids[0] not in (stats.get("legacy_projects") or []),
          "STATS derives no legacy records from the training project")
    # Fail loud, not silent: the registry row keyed to a project STATS does not
    # scan is reported as an unknown-project row rather than dropped, which
    # also proves the row really was planted.
    excluded = stats.get("excluded") or {}
    check(train_ids[0] in (excluded.get("unknown_project_ids") or []),
          f"the planted registry row is reported as excluded (got {excluded})")


def main():
    root = tempfile.mkdtemp(prefix="matrix_comproj_test_")
    try:
        build_root(root)
        test_project_filter(root)
        test_preference_outside_filter_is_ignored(root)
        bases = test_create_training_project(root)
        raced = test_concurrent_mint_is_safe(root)
        test_cell_lists_composites(root)
        test_scan_skips_training_projects(root, list(bases) + raced[:1])
    finally:
        shutil.rmtree(root, ignore_errors=True)
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
