"""The VEGA tutorial projects and the outlines set they train on.

No pytest. Run with the unified env python:
    env/bin/python scripts/_tutorial/tests/test_tutorial_projects.py

Four parts, all of them run by __main__:

  BUILT SET       the training set on disk at
                  inprocess/_composite/tutorial_outlines is read and checked:
                  every file a training set owes is there, every frame has a
                  label and every label a frame, every frame link resolves to a
                  file, split_pins.json pins exactly the frames on disk, those
                  pins are the ones the real split builder makes from the
                  survey's transect rule, both species appear in all three
                  splits, each clears the split builder's own minimum train
                  count, and no two frame names differ only in letter case.

  BUILT PROJECTS  the two projects on disk are read and checked: the right
                  project type, the right recorded training source, opening on
                  Train, the marker file, the outlines project's relative
                  symlink to its set, no imagery or link at all in the boxes
                  project, and the set naming its project back.

  BUILDERS        both tools, over fixtures in a temp folder: the happy path
                  end to end, a build reproducible from its seed, a frame whose
                  link is broken passed over, a parent that pins nothing given
                  pins here, both projects minted, a second run changing
                  nothing, and a rebuild that remakes a project and leaves the
                  fixture alone.

  ADVERSARIAL     every refusal the two tools promise. For the set: a parent
                  that is not a training set, a parent whose data.yaml declares
                  no class, a quota that is zero, fractional or a boolean, a
                  train quota below the split builder's minimum, a bucket with
                  too few frames, a destination that already exists, a
                  destination whose holding folder is not there, a name outside
                  the plain-name rule, a label line that is not a polygon, a
                  label naming a class the parent does not declare, a
                  coordinate outside 0 to 1, a label file with no frame, a
                  hostile label file of nulls, a corrupt parent manifest, and
                  a write that fails partway leaving no half written folder.
                  For the projects: a missing fixture, an empty one, a training
                  set whose records cannot be read, and seven ways a folder
                  that is not a tutorial project of this fixture is refused a
                  removal, each of which then checks the folder is still there.

Nothing here trains, reaches a GPU, opens a port, mints into the real run
folder root, or writes anywhere but its own temp folders, which are removed in
a finally block and whose removal is verified. The built set and the built
projects are read and never written.
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
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUTORIAL = os.path.dirname(_HERE)
_SCRIPTS = os.path.dirname(_TUTORIAL)
_REPO = os.path.dirname(_SCRIPTS)
_MODULE_ROOT = os.path.dirname(_REPO)
for _path in (_SCRIPTS, _TUTORIAL):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import build_tutorial_outlines as bto  # noqa: E402
import build_tutorial_projects as btp  # noqa: E402
from _boxes import ingest_yolo_boxes as ing  # noqa: E402

# What this suite reads and never writes.
INPROCESS = os.path.join(_MODULE_ROOT, "inprocess")
COMPOSITE_ROOT = os.path.join(INPROCESS, bto.COMPOSITE_DIRNAME)
BUILT_SET = os.path.join(COMPOSITE_ROOT, bto.DEFAULT_NAME)
BUILT_BOX_SET = os.path.join(_REPO, "supporting_data", "bbox_datasets",
                             "tutorial_boxes")

# The tool's own defaults, read from the tool rather than retyped, so a change
# to a default that is not carried into a rebuild fails here instead of passing
# quietly.
EXPECT_SPECIES = dict(bto.SPECIES_QUOTAS)
EXPECT_NEGATIVES = dict(bto.NEGATIVE_QUOTAS)
EXPECT_IMAGES = 2 * sum(EXPECT_SPECIES.values()) + sum(EXPECT_NEGATIVES.values())

# Every file a training set owes a reader, plus the one this tool adds.
REQUIRED_FILES = (bto.DATA_YAML_NAME, bto.CLASS_MAP_NAME, bto.MANIFEST_CSV_NAME,
                  bto.COMPOSITE_JSON_NAME, bto.SPLIT_PINS_NAME, bto.README_NAME)

# Fixture sizes for the builder half: enough that a train quota can clear the
# split builder's minimum and every bucket still has frames to spare.
FIXTURE_CLASSES = ("AA", "BB")
FIXTURE_PER_BUCKET = {"train": 14, "valid": 4, "test": 4}
FIXTURE_SPECIES_QUOTAS = {"train": 10, "valid": 2, "test": 2}
FIXTURE_NEGATIVE_QUOTAS = {"train": 10, "valid": 2, "test": 2}
FIXTURE_IMAGES = (2 * sum(FIXTURE_SPECIES_QUOTAS.values())
                  + sum(FIXTURE_NEGATIVE_QUOTAS.values()))

# The transect each split is drawn from, matching the survey's own rule.
TRAIN_TRANSECTS = (1, 2, 3, 4)
VALID_TRANSECT = 5
TEST_TRANSECT = 6

# One polygon, three points, well inside the frame.
FIXTURE_POLYGON = "0.10 0.10 0.30 0.10 0.30 0.30"

# A one pixel JPEG, so a fixture frame is a real image file without Pillow
# being asked to draw anything.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffc20011"
    "08000100010101110000ffc40014000100000000000000000000"
    "00000000000009ffda0008010100000000013fffc40014010100"
    "0000000000000000000000000000ffda0008010100010502ffc4"
    "0014100100000000000000000000000000000000ffda00080101"
    "00010500013fffc40014110100000000000000000000000000000"
    "000ffda0008010100063f02ffc40014100100000000000000000"
    "00000000000000000ffda0008010100013f21ffda0008010100"
    "00000010ffc4001410010000000000000000000000000000000"
    "0ffda0008010100013f10ffd9")

_RESULTS = []
_FIXTURE_ROOTS = []


def check(cond, msg):
    """Raise AssertionError with ``msg`` when ``cond`` is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, record its outcome, print a per-test line."""
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as exc:
        _RESULTS.append((fn.__name__, False, f"{exc}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {exc}")


def refuses(fn, needle, where):
    """Run ``fn`` expecting a refusal whose message carries ``needle``."""
    try:
        fn()
    except (bto.TutorialSetError, ing.IngestError, btp.TutorialProjectError) as exc:
        check(needle in str(exc),
              f"{where}: refusal said {str(exc)!r}, expected it to name {needle!r}")
        return str(exc)
    raise AssertionError(f"{where}: no refusal was raised")


def temp_root(label):
    """A temp folder removed and verified by __main__."""
    root = tempfile.mkdtemp(prefix=f"tutorial-projects-{label}-")
    _FIXTURE_ROOTS.append(root)
    return root


def read_json(path):
    """One JSON document."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Building a parent training set to take a subset of ───────────────────

def frame_name(transect, number):
    """A TCRMP survey frame name whose transect the split builder can parse."""
    return f"TCRMP20200101_clip_ZZA_T{transect}{number:02d}.jpg"


def make_parent(root, *, name="parent", classes=FIXTURE_CLASSES,
                per_bucket=None, broken=0, with_manifest=True):
    """Write a training set in the shape the coverage matrix writes one.

    Every bucket (one per class, plus the frames with no outline) gets
    ``per_bucket`` frames in each split, filed on the transect the split
    builder reads: transects 1 to 4 train, transect 5 validates, transect 6
    tests.

    Parameters:
      root: the folder to write the set inside.
      name: the set's folder name.
      classes: the class tokens it declares.
      per_bucket: {'train': n, 'valid': n, 'test': n} per bucket.
      broken: how many extra frames are written as links to nothing.
      with_manifest: write a composite_manifest.csv beside the frames.
    Returns:
      the set's absolute path.
    """
    per_bucket = per_bucket or FIXTURE_PER_BUCKET
    dest = os.path.join(root, name)
    os.makedirs(os.path.join(dest, bto.IMAGES_DIR))
    os.makedirs(os.path.join(dest, bto.LABELS_DIR))
    counters = {t: 10 for t in list(TRAIN_TRANSECTS) + [VALID_TRANSECT, TEST_TRANSECT]}
    rows = []

    def next_name(split, index):
        transect = (TRAIN_TRANSECTS[index % len(TRAIN_TRANSECTS)] if split == "train"
                    else VALID_TRANSECT if split == "valid" else TEST_TRANSECT)
        counters[transect] += 1
        return frame_name(transect, counters[transect])

    buckets = list(range(len(classes))) + [None]
    for bucket in buckets:
        for split, count in per_bucket.items():
            for index in range(count):
                fname = next_name(split, index)
                write_frame(dest, fname)
                lines = [] if bucket is None else [f"{bucket} {FIXTURE_POLYGON}"]
                write_label(dest, fname, "\n".join(lines))
                if lines:
                    rows.append({"basename": os.path.splitext(fname)[0],
                                 "label": classes[bucket],
                                 "project_id": "fixture_project", "uid": fname,
                                 "status": "accepted", "source_type": "manual_click",
                                 "reviewer": "LO", "chosen_by_preference": "",
                                 "at": "2026-09-17T00:00:00-04:00"})
    for index in range(broken):
        fname = next_name("train", index)
        os.symlink(os.path.join(root, "gone", fname),
                   os.path.join(dest, bto.IMAGES_DIR, fname))
        write_label(dest, fname, f"0 {FIXTURE_POLYGON}")

    write_data_yaml(dest, classes)
    if with_manifest:
        header = ("basename,label,project_id,uid,status,source_type,reviewer,"
                  "chosen_by_preference,at")
        lines = [header] + [",".join(str(row[key]) for key in header.split(","))
                            for row in rows]
        _write(os.path.join(dest, bto.MANIFEST_CSV_NAME), "\n".join(lines) + "\n")
    return dest


def write_frame(dest, fname):
    """One real, tiny frame file inside a set's all_images/."""
    with open(os.path.join(dest, bto.IMAGES_DIR, fname), "wb") as fh:
        fh.write(_TINY_JPEG)


def write_label(dest, fname, text):
    """One label file beside a frame."""
    path = os.path.join(dest, bto.LABELS_DIR,
                        os.path.splitext(fname)[0] + bto.LABEL_EXT)
    _write(path, (text + "\n") if text else "")


def write_data_yaml(dest, classes):
    """A set's data.yaml, in the coverage matrix's own shape."""
    lines = [f"path: {os.path.abspath(dest)}", f"train: {bto.IMAGES_DIR}",
             f"val: {bto.IMAGES_DIR}", f"nc: {len(classes)}", "names:"]
    for index, token in enumerate(classes):
        lines.append(f"  {index}: {token}")
    _write(os.path.join(dest, bto.DATA_YAML_NAME), "\n".join(lines) + "\n")


def _write(path, text):
    """One text file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def build_fixture_set(root, *, parent=None, name=bto.DEFAULT_NAME, dest_root=None,
                      species=None, negatives=None, seed=bto.DEFAULT_SEED):
    """Build a tutorial set from a fixture parent, silently.

    Returns:
      (composite.json document, the set's path).
    """
    parent = parent or make_parent(root)
    dest_root = dest_root or os.path.join(root, "out")
    os.makedirs(dest_root, exist_ok=True)
    dest = os.path.join(dest_root, name)
    meta = bto.build_tutorial_outlines(
        parent, dest, species_quotas=species or FIXTURE_SPECIES_QUOTAS,
        negative_quotas=negatives or FIXTURE_NEGATIVE_QUOTAS, seed=seed,
        log=lambda *a, **k: None)
    return meta, dest


def fixture_inprocess(root):
    """A run folder root holding a tutorial outlines set, ready to mint into.

    Returns:
      (inprocess root, the set's path).
    """
    inprocess = os.path.join(root, "inprocess")
    composite_root = os.path.join(inprocess, bto.COMPOSITE_DIRNAME)
    os.makedirs(composite_root)
    _meta, dest = build_fixture_set(root, dest_root=composite_root)
    return inprocess, dest


def mint_both(inprocess, **kwargs):
    """Mint both tutorial projects into a temp run folder root, silently."""
    return btp.build_tutorial_projects(inprocess, btp.FIXTURES,
                                       log=lambda *a, **k: None, **kwargs)


# ── The built set ────────────────────────────────────────────────────────

def test_the_built_set_is_there_with_every_file():
    check(os.path.isdir(BUILT_SET), f"{BUILT_SET} is not there")
    for folder in (bto.IMAGES_DIR, bto.LABELS_DIR):
        check(os.path.isdir(os.path.join(BUILT_SET, folder)),
              f"{BUILT_SET} has no {folder}/ folder")
    missing = [n for n in REQUIRED_FILES
               if not os.path.isfile(os.path.join(BUILT_SET, n))]
    check(not missing, f"{BUILT_SET} is missing {', '.join(missing)}")


def built_images():
    """The frame names in the built set, sorted."""
    return sorted(n for n in os.listdir(os.path.join(BUILT_SET, bto.IMAGES_DIR))
                  if os.path.splitext(n)[1].lower() in ing.IMAGE_EXTS)


def built_labels():
    """The label file names in the built set, sorted."""
    return sorted(n for n in os.listdir(os.path.join(BUILT_SET, bto.LABELS_DIR))
                  if n.endswith(bto.LABEL_EXT))


def built_pins():
    """{frame: split} from the built set's split_pins.json."""
    return read_json(os.path.join(BUILT_SET, bto.SPLIT_PINS_NAME))["pins"]


def test_frames_and_labels_pair_and_match_the_records():
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    images, labels = built_images(), built_labels()
    check(len(images) == len(labels) == meta["n_images"] == EXPECT_IMAGES,
          f"{len(images)} frame(s), {len(labels)} label file(s), "
          f"{meta['n_images']} in the records, {EXPECT_IMAGES} expected")
    unpaired = sorted(set(os.path.splitext(n)[0] for n in images)
                      ^ set(n[:-len(bto.LABEL_EXT)] for n in labels))
    check(not unpaired, f"{len(unpaired)} frame(s) or label file(s) unpaired: "
                        f"{unpaired[:5]}")


def test_every_frame_link_resolves():
    broken = [n for n in built_images()
              if not os.path.exists(os.path.join(BUILT_SET, bto.IMAGES_DIR, n))]
    check(not broken, f"{len(broken)} frame link(s) resolve to nothing: {broken[:5]}")


def test_no_two_frame_names_differ_only_in_case():
    lowered = [n.lower() for n in built_images()]
    check(len(set(lowered)) == len(lowered),
          "two frame names differ only in letter case, which the sync on this "
          "tree folds into one file")


def test_split_pins_cover_exactly_the_frames_on_disk():
    pins = built_pins()
    check(sorted(pins) == built_images(),
          f"split_pins.json pins {len(pins)} frame(s) and the set holds "
          f"{len(built_images())}")
    bad = sorted(n for n, split in pins.items()
                 if split not in ("train", "valid", "test"))
    check(not bad, f"{len(bad)} pin(s) name a split that is not one of train, "
                   f"valid, test: {bad[:5]}")


def test_the_pins_are_the_transect_rule():
    pinned_split = bto._load_pinned_split()
    pins = built_pins()
    replayed = pinned_split.assign_fresh_transect(sorted(pins), {5}, {6})
    moved = sorted(n for n in pins if replayed[n] != pins[n])
    check(not moved, f"{len(moved)} frame(s) are pinned to a split the real "
                     f"split builder does not give them: {moved[:5]}")


def test_the_split_sizes_are_what_the_quotas_ask_for():
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    pins = built_pins()
    for split in ("train", "valid", "test"):
        want = 2 * EXPECT_SPECIES[split] + EXPECT_NEGATIVES[split]
        have = sum(1 for value in pins.values() if value == split)
        check(have == want == meta["by_split"][split],
              f"{split}: {have} frame(s) pinned, {meta['by_split'][split]} in "
              f"the records, {want} asked for")


def built_counts():
    """(names, counts) read off the built set's labels and pins."""
    names = bto.read_parent_classes(BUILT_SET)
    return names, bto.count_set(BUILT_SET, names, built_pins())


def test_every_species_is_in_every_split():
    names, counts = built_counts()
    thin = []
    for class_id in sorted(names):
        for split in ("train", "valid", "test"):
            have = counts["by_label_split"].get((names[class_id], split), 0)
            if have < EXPECT_SPECIES[split]:
                thin.append(f"{names[class_id]} {split}: {have}")
    check(not thin, "a species is missing from a split, so the evaluation "
                    f"report would have no row to read: {'; '.join(thin)}")


def test_every_species_clears_the_splitters_minimum_train_count():
    names, _counts = built_counts()
    pins = built_pins()
    train_modes = {}
    for name, split in pins.items():
        if split != "train":
            continue
        path = os.path.join(BUILT_SET, bto.LABELS_DIR,
                            os.path.splitext(name)[0] + bto.LABEL_EXT)
        with open(path, encoding="utf-8") as fh:
            ids = bto.parse_polygons(fh.read(), len(names), path)
        bucket = bto.bucket_of(ids)
        if bucket != bto.NEGATIVE_BUCKET:
            train_modes[bucket] = train_modes.get(bucket, 0) + 1
    thin = [f"{names[c]}: {train_modes.get(c, 0)}" for c in sorted(names)
            if train_modes.get(c, 0) < bto.PINNED_SPLIT_MIN_SAMPLES]
    check(not thin, "the split builder would drop a class for too few train "
                    f"frames: {'; '.join(thin)}")


def test_the_tools_min_samples_constant_matches_the_splitter():
    pinned_split = bto._load_pinned_split()
    actual = pinned_split.arg_parse(["--src", ".", "--out", "."]).min_samples
    check(bto.PINNED_SPLIT_MIN_SAMPLES == actual,
          f"the tool holds {bto.PINNED_SPLIT_MIN_SAMPLES} and the split builder's "
          f"--min_samples default is {actual}")


def test_the_class_tables_agree():
    names = bto.read_parent_classes(BUILT_SET)
    class_map = read_json(os.path.join(BUILT_SET, bto.CLASS_MAP_NAME))
    check(class_map == {token: cid for cid, token in names.items()},
          f"class_map.json says {class_map} and data.yaml says {names}")
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    check(meta["labels"] == [names[c] for c in sorted(names)],
          f"composite.json labels {meta['labels']} and data.yaml names "
          f"{[names[c] for c in sorted(names)]}")


def test_the_set_is_a_real_subset_of_its_parent():
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    parent = os.path.join(COMPOSITE_ROOT, meta["tutorial"]["parent"])
    check(os.path.isdir(parent), f"the parent set at {parent} is not there")
    parent_images = set(os.listdir(os.path.join(parent, bto.IMAGES_DIR)))
    stray = sorted(set(built_images()) - parent_images)
    check(not stray, f"{len(stray)} frame(s) are not the parent's: {stray[:5]}")
    check(len(built_images()) < len(parent_images),
          "the set is not smaller than its parent")


def test_the_parent_pins_nothing_and_this_set_does():
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    parent = os.path.join(COMPOSITE_ROOT, meta["tutorial"]["parent"])
    check(not os.path.exists(os.path.join(parent, bto.SPLIT_PINS_NAME)),
          f"{parent} already carries {bto.SPLIT_PINS_NAME}, so this test no "
          "longer proves the pins were written here")
    check(os.path.isfile(os.path.join(BUILT_SET, bto.SPLIT_PINS_NAME)),
          f"{BUILT_SET} carries no {bto.SPLIT_PINS_NAME}")


# ── The built projects ───────────────────────────────────────────────────

def built_project(fixture):
    """The run folder of the built tutorial project for one fixture."""
    found = btp.find_tutorial_projects(INPROCESS, fixture)
    check(found, f"no tutorial project for {fixture.name} under {INPROCESS}")
    return found[0]


def test_both_tutorial_projects_are_there_with_the_right_type_and_source():
    for fixture in btp.FIXTURES:
        run_dir = built_project(fixture)
        meta = btp._read_project_json(run_dir)
        check(meta is not None, f"{run_dir} has no readable project.json")
        want = btp.PROJECT_TYPE_FOR_KIND[fixture.kind]
        check(meta["project_type"] == want,
              f"{run_dir} declares {meta['project_type']!r}, expected {want!r}")
        check(btp._recorded_source_name(meta) == fixture.name,
              f"{run_dir} records source {btp._recorded_source_name(meta)!r}, "
              f"expected {fixture.name!r}")
        check(meta["current_step"] == btp.TRAIN_STEP,
              f"{run_dir} opens on step {meta['current_step']}, expected "
              f"{btp.TRAIN_STEP} (Train)")
        check(meta["name"] == fixture.display_name,
              f"{run_dir} is named {meta['name']!r}, expected "
              f"{fixture.display_name!r}")


def test_the_two_project_names_sort_together():
    names = sorted(os.path.basename(built_project(f)) for f in btp.FIXTURES)
    prefix = os.path.commonprefix(names)
    check(prefix.endswith("train_tutorial_"),
          f"the two run folder names share only {prefix!r}, so they do not sort "
          "next to each other in the launcher's project list")


def test_the_outlines_project_reaches_its_set_through_a_relative_symlink():
    fixture = [f for f in btp.FIXTURES if f.kind == "composite"][0]
    run_dir = built_project(fixture)
    link = os.path.join(run_dir, btp.TRAINING_LINK_DIRNAME)
    check(os.path.islink(link), f"{link} is not a symlink")
    check(not os.path.isabs(os.readlink(link)),
          f"{link} points at the absolute path {os.readlink(link)}; a relative "
          "link keeps the pair together if the run folder root moves")
    check(os.path.realpath(link) == os.path.realpath(BUILT_SET),
          f"{link} points at {os.path.realpath(link)}, expected {BUILT_SET}")


def test_the_boxes_project_holds_no_imagery_and_no_link():
    fixture = [f for f in btp.FIXTURES if f.kind == "bbox"][0]
    run_dir = built_project(fixture)
    links = [n for n in os.listdir(run_dir)
             if os.path.islink(os.path.join(run_dir, n))]
    check(not links, f"{run_dir} holds the link(s) {links}; imagery from outside "
                     "the survey never enters a run folder")
    images = [n for n in os.listdir(run_dir)
              if os.path.splitext(n)[1].lower() in ing.IMAGE_EXTS]
    check(not images, f"{run_dir} holds the picture(s) {images}")
    meta = btp._read_project_json(run_dir)
    check(os.path.realpath(meta["training_source"]["path"])
          == os.path.realpath(BUILT_BOX_SET),
          f"{run_dir} records its dataset at "
          f"{meta['training_source']['path']}, expected {BUILT_BOX_SET}")


def test_the_training_set_names_its_tutorial_project():
    fixture = [f for f in btp.FIXTURES if f.kind == "composite"][0]
    run_name = os.path.basename(built_project(fixture))
    meta = read_json(os.path.join(BUILT_SET, bto.COMPOSITE_JSON_NAME))
    check(run_name in (meta.get("linked_projects") or []),
          f"{BUILT_SET} does not name {run_name} in linked_projects, so the "
          "coverage matrix cannot find the project built from it")


def test_both_built_projects_carry_a_readable_marker():
    for fixture in btp.FIXTURES:
        run_dir = built_project(fixture)
        marker = btp.read_marker(run_dir)
        check(marker is not None, f"{run_dir} has no readable {btp.MARKER_NAME}")
        check(marker["fixture"] == fixture.name and marker["tool"] == btp.MARKER_TOOL,
              f"{run_dir} carries a marker for {marker.get('fixture')!r} written "
              f"by {marker.get('tool')!r}")


# ── The outlines builder ─────────────────────────────────────────────────

def test_the_outlines_builder_end_to_end():
    root = temp_root("happy")
    meta, dest = build_fixture_set(root)
    check(meta["n_images"] == FIXTURE_IMAGES,
          f"built {meta['n_images']} frame(s), expected {FIXTURE_IMAGES}")
    check(meta["n_negatives"] == sum(FIXTURE_NEGATIVE_QUOTAS.values()),
          f"built {meta['n_negatives']} frame(s) with no outline, expected "
          f"{sum(FIXTURE_NEGATIVE_QUOTAS.values())}")
    missing = [n for n in REQUIRED_FILES if not os.path.isfile(os.path.join(dest, n))]
    check(not missing, f"{dest} is missing {', '.join(missing)}")
    pins = read_json(os.path.join(dest, bto.SPLIT_PINS_NAME))["pins"]
    check(len(pins) == FIXTURE_IMAGES, f"{len(pins)} pin(s), expected {FIXTURE_IMAGES}")
    rows = open(os.path.join(dest, bto.MANIFEST_CSV_NAME), encoding="utf-8").read()
    check(rows.startswith("basename,"), "the carried manifest lost its header")


def test_the_outlines_build_is_reproducible_from_the_seed():
    root = temp_root("seed")
    parent = make_parent(root)
    first, first_dest = build_fixture_set(root, parent=parent,
                                          dest_root=os.path.join(root, "a"))
    second, second_dest = build_fixture_set(root, parent=parent,
                                            dest_root=os.path.join(root, "b"))
    a = read_json(os.path.join(first_dest, bto.SPLIT_PINS_NAME))["pins"]
    b = read_json(os.path.join(second_dest, bto.SPLIT_PINS_NAME))["pins"]
    check(a == b, "two builds from the same seed pinned different frames")
    check(first["by_label"] == second["by_label"],
          f"two builds counted differently: {first['by_label']} and "
          f"{second['by_label']}")


def test_a_different_seed_chooses_different_frames():
    root = temp_root("seed2")
    parent = make_parent(root)
    _a, a_dest = build_fixture_set(root, parent=parent,
                                   dest_root=os.path.join(root, "a"))
    _b, b_dest = build_fixture_set(root, parent=parent,
                                   dest_root=os.path.join(root, "b"),
                                   seed="another-seed")
    a = set(read_json(os.path.join(a_dest, bto.SPLIT_PINS_NAME))["pins"])
    b = set(read_json(os.path.join(b_dest, bto.SPLIT_PINS_NAME))["pins"])
    check(a != b, "two different seeds chose exactly the same frames, so the "
                  "seed is not reaching the shuffle")


def test_a_frame_whose_link_is_broken_is_passed_over():
    root = temp_root("broken")
    parent = make_parent(root, broken=3)
    meta, dest = build_fixture_set(root, parent=parent)
    check(meta["tutorial"]["parent_frames_skipped_unresolved_link"] == 3,
          "the build did not record the 3 frames whose link resolves to nothing, "
          f"it recorded {meta['tutorial']['parent_frames_skipped_unresolved_link']}")
    broken = [n for n in os.listdir(os.path.join(dest, bto.IMAGES_DIR))
              if not os.path.exists(os.path.join(dest, bto.IMAGES_DIR, n))]
    check(not broken, f"a frame whose link resolves to nothing was copied: {broken}")


def test_a_parent_with_no_pins_gets_pins_written_here():
    root = temp_root("pins")
    parent = make_parent(root)
    check(not os.path.exists(os.path.join(parent, bto.SPLIT_PINS_NAME)),
          "the fixture parent unexpectedly carries split pins")
    _meta, dest = build_fixture_set(root, parent=parent)
    pins = read_json(os.path.join(dest, bto.SPLIT_PINS_NAME))
    check(pins["schema_version"] == bto.SPLIT_PINS_SCHEMA_VERSION,
          f"split_pins.json says schema {pins['schema_version']}")
    check(bto.DEFAULT_SEED in pins["source"] or "seed" in pins["source"],
          f"split_pins.json does not say how it was pinned: {pins['source']!r}")


def test_a_parent_that_is_not_a_training_set_is_refused():
    root = temp_root("notaset")
    empty = os.path.join(root, "empty")
    os.makedirs(empty)
    refuses(lambda: build_fixture_set(root, parent=empty),
            "is not a training set", "a parent with no all_images/")


def test_a_parent_whose_data_yaml_declares_no_class_is_refused():
    root = temp_root("noclass")
    parent = make_parent(root)
    _write(os.path.join(parent, bto.DATA_YAML_NAME), "path: x\nnames:\n")
    refuses(lambda: build_fixture_set(root, parent=parent),
            "declares no class", "a parent whose data.yaml names nothing")


def test_a_zero_or_fractional_or_boolean_quota_is_refused():
    root = temp_root("quotas")
    parent = make_parent(root)
    for bad in ({"train": 0, "valid": 2, "test": 2},
                {"train": 10.5, "valid": 2, "test": 2},
                {"train": True, "valid": 2, "test": 2},
                {"train": 10, "valid": 2}):
        refuses(lambda bad=bad: build_fixture_set(root, parent=parent, species=bad),
                "--species-quota", f"the species quota {bad}")


def test_a_train_quota_below_the_splitters_minimum_is_refused():
    root = temp_root("minsamples")
    parent = make_parent(root)
    thin = {"train": bto.PINNED_SPLIT_MIN_SAMPLES - 1, "valid": 2, "test": 2}
    refuses(lambda: build_fixture_set(root, parent=parent, species=thin),
            "min_samples", "a train quota below the split builder's minimum")


def test_a_bucket_with_too_few_frames_is_refused():
    root = temp_root("short")
    parent = make_parent(root, per_bucket={"train": 14, "valid": 1, "test": 4})
    message = refuses(lambda: build_fixture_set(root, parent=parent),
                      "cannot fill the tutorial set", "a parent short on valid frames")
    check("valid" in message, f"the refusal did not name the split: {message!r}")


def test_a_destination_that_already_exists_is_refused():
    root = temp_root("exists")
    parent = make_parent(root)
    dest_root = os.path.join(root, "out")
    os.makedirs(os.path.join(dest_root, bto.DEFAULT_NAME))
    refuses(lambda: build_fixture_set(root, parent=parent, dest_root=dest_root),
            "already exists", "a destination that is already there")


def test_a_destination_whose_holder_is_not_there_is_refused():
    root = temp_root("noholder")
    parent = make_parent(root)
    gone = os.path.join(root, "gone", "deeper")
    refuses(lambda: bto.build_tutorial_outlines(
        parent, os.path.join(gone, bto.DEFAULT_NAME),
        species_quotas=FIXTURE_SPECIES_QUOTAS,
        negative_quotas=FIXTURE_NEGATIVE_QUOTAS, log=lambda *a, **k: None),
        "is not there", "a destination whose holding folder is absent")


def test_a_name_outside_the_plain_name_rule_is_refused():
    refuses(lambda: ing.check_dataset_name("not a name"),
            "not a usable dataset folder name", "a name with a space")


def test_a_label_line_that_is_not_a_polygon_is_refused():
    root = temp_root("shortline")
    parent = make_parent(root)
    name = sorted(os.listdir(os.path.join(parent, bto.LABELS_DIR)))[0]
    _write(os.path.join(parent, bto.LABELS_DIR, name), "0 0.1 0.2 0.3\n")
    refuses(lambda: build_fixture_set(root, parent=parent),
            "is not a polygon", "a label line with three coordinates")


def test_a_label_naming_an_undeclared_class_is_refused():
    root = temp_root("badclass")
    parent = make_parent(root)
    name = sorted(os.listdir(os.path.join(parent, bto.LABELS_DIR)))[0]
    _write(os.path.join(parent, bto.LABELS_DIR, name), f"9 {FIXTURE_POLYGON}\n")
    refuses(lambda: build_fixture_set(root, parent=parent),
            "names class 9", "a label naming a class the parent does not declare")


def test_a_coordinate_outside_zero_to_one_is_refused():
    root = temp_root("badcoord")
    parent = make_parent(root)
    name = sorted(os.listdir(os.path.join(parent, bto.LABELS_DIR)))[0]
    _write(os.path.join(parent, bto.LABELS_DIR, name),
           "0 0.1 0.1 1.4 0.1 0.3 0.3\n")
    refuses(lambda: build_fixture_set(root, parent=parent),
            "fraction of the frame", "a coordinate above 1")


def test_a_label_file_with_no_frame_is_refused():
    root = temp_root("orphan")
    parent = make_parent(root)
    _write(os.path.join(parent, bto.LABELS_DIR, "nothing_here.txt"),
           f"0 {FIXTURE_POLYGON}\n")
    refuses(lambda: build_fixture_set(root, parent=parent),
            "with no frame", "a label file whose frame is absent")


def test_a_hostile_label_file_of_nulls_is_refused_not_crashed():
    root = temp_root("nulls")
    parent = make_parent(root)
    name = sorted(os.listdir(os.path.join(parent, bto.LABELS_DIR)))[0]
    with open(os.path.join(parent, bto.LABELS_DIR, name), "wb") as fh:
        fh.write(b"\x00" * 4096)
    refuses(lambda: build_fixture_set(root, parent=parent),
            "line 1", "a label file of null bytes")


def test_a_corrupt_parent_manifest_is_refused_before_anything_is_written():
    root = temp_root("badmanifest")
    parent = make_parent(root)
    with open(os.path.join(parent, bto.MANIFEST_CSV_NAME), "wb") as fh:
        fh.write(b"basename,label\n\xff\xfe not text at all\n")
    dest_root = os.path.join(root, "out")
    refuses(lambda: build_fixture_set(root, parent=parent, dest_root=dest_root),
            "could not be read", "a manifest that is not text")
    check(not os.path.exists(os.path.join(dest_root, bto.DEFAULT_NAME)),
          "a folder was left behind by a build that was refused")


def test_a_build_that_fails_partway_leaves_no_folder_behind():
    root = temp_root("partial")
    parent = make_parent(root)
    dest_root = os.path.join(root, "out")
    os.makedirs(dest_root)
    dest = os.path.join(dest_root, bto.DEFAULT_NAME)
    original = bto.write_data_yaml

    def explode(*_args, **_kwargs):
        raise bto.TutorialSetError("the disk filled up mid write")

    bto.write_data_yaml = explode
    try:
        refuses(lambda: bto.build_tutorial_outlines(
            parent, dest, species_quotas=FIXTURE_SPECIES_QUOTAS,
            negative_quotas=FIXTURE_NEGATIVE_QUOTAS, log=lambda *a, **k: None),
            "disk filled up", "a write that fails partway")
    finally:
        bto.write_data_yaml = original
    check(not os.path.exists(dest),
          f"{dest} was left half written, which would be offered in the Train "
          "step's source picker and would block the retry that fixes it")


def test_the_outlines_command_line_refuses_and_says_so():
    root = temp_root("cli-set")
    code = bto.main(["--parent", os.path.join(root, "gone"), "--dest-root", root,
                     "--name", bto.DEFAULT_NAME, "--quiet"])
    check(code == 1, f"the command line answered {code}, expected 1")


def test_the_outlines_command_line_builds_and_answers_zero():
    root = temp_root("cli-set-ok")
    parent = make_parent(root)
    out = os.path.join(root, "out")
    os.makedirs(out)
    code = bto.main(["--parent", parent, "--dest-root", out, "--name",
                     bto.DEFAULT_NAME, "--species-quota", "10", "2", "2",
                     "--negative-quota", "10", "2", "2", "--quiet"])
    check(code == 0, f"the command line answered {code}, expected 0")
    check(os.path.isfile(os.path.join(out, bto.DEFAULT_NAME, bto.SPLIT_PINS_NAME)),
          "the command line wrote no split pins")


# ── The projects builder ─────────────────────────────────────────────────

def test_both_projects_mint_with_the_right_type_and_source():
    root = temp_root("mint")
    inprocess, dest = fixture_inprocess(root)
    records = mint_both(inprocess)
    check(len(records) == 2, f"{len(records)} project(s) minted, expected 2")
    for record in records:
        fixture = record["fixture"]
        meta = btp._read_project_json(record["run_dir"])
        check(meta["project_type"] == btp.PROJECT_TYPE_FOR_KIND[fixture.kind],
              f"{record['run_dir_name']} declares {meta['project_type']!r}")
        check(btp._recorded_source_name(meta) == fixture.name,
              f"{record['run_dir_name']} records {btp._recorded_source_name(meta)!r}")
        check(record["action"] == "minted",
              f"{record['run_dir_name']} was {record['action']}, expected minted")
    linked = read_json(os.path.join(dest, bto.COMPOSITE_JSON_NAME))["linked_projects"]
    check(len(linked) == 1, f"the set names {linked} as its projects, expected one")
    btp.verify(inprocess, records, lambda *a, **k: None)


def test_a_second_run_changes_nothing():
    root = temp_root("idempotent")
    inprocess, _dest = fixture_inprocess(root)
    first = mint_both(inprocess)
    before = sorted(os.listdir(inprocess))
    stamps = {r["run_dir"]: btp.read_marker(r["run_dir"])["minted_at"] for r in first}
    second = mint_both(inprocess)
    check(sorted(os.listdir(inprocess)) == before,
          "a second run changed what is under the run folder root")
    check([r["action"] for r in second] == ["kept", "kept"],
          f"a second run reported {[r['action'] for r in second]}, expected kept")
    for record in second:
        check(btp.read_marker(record["run_dir"])["minted_at"] == stamps[record["run_dir"]],
              f"{record['run_dir_name']} was rewritten by the second run")


def test_rebuild_remakes_a_project_and_leaves_the_fixture_alone():
    root = temp_root("rebuild")
    inprocess, dest = fixture_inprocess(root)
    mint_both(inprocess)
    frames_before = sorted(os.listdir(os.path.join(dest, bto.IMAGES_DIR)))
    labels_before = sorted(os.listdir(os.path.join(dest, bto.LABELS_DIR)))
    wrecked = os.path.join(inprocess, "run_26-09-17_LO_train_tutorial_outlines",
                           "project.json")
    check(os.path.isfile(wrecked), f"{wrecked} is not there")
    _write(wrecked, "this is not json")
    records = mint_both(inprocess, rebuild=True)
    check([r["action"] for r in records] == ["rebuilt", "rebuilt"],
          f"the rebuild reported {[r['action'] for r in records]}")
    meta = btp._read_project_json(
        os.path.join(inprocess, "run_26-09-17_LO_train_tutorial_outlines"))
    check(meta is not None and meta["project_type"] == btp.COMPOSITE_PROJECT_TYPE,
          "the rebuilt project does not read back as a training project")
    check(sorted(os.listdir(os.path.join(dest, bto.IMAGES_DIR))) == frames_before,
          "the rebuild changed the fixture's frames")
    check(sorted(os.listdir(os.path.join(dest, bto.LABELS_DIR))) == labels_before,
          "the rebuild changed the fixture's label files")


def fake_run_dir(inprocess, name, *, marker=None, project=None, raw_marker=None):
    """A run folder that is not a tutorial project, for the refusal tests."""
    run_dir = os.path.join(inprocess, name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "step1_makeAllPoints"), exist_ok=True)
    if project is not None:
        _write(os.path.join(run_dir, "project.json"), json.dumps(project))
    if raw_marker is not None:
        _write(os.path.join(run_dir, btp.MARKER_NAME), raw_marker)
    elif marker is not None:
        _write(os.path.join(run_dir, btp.MARKER_NAME), json.dumps(marker))
    return run_dir


def good_marker(fixture):
    """The marker a real tutorial project of ``fixture`` carries."""
    return {"schema_version": 1, "tool": btp.MARKER_TOOL, "fixture": fixture.name,
            "fixture_kind": fixture.kind, "by": "LO"}


def good_project(fixture):
    """The project.json a real tutorial project of ``fixture`` carries."""
    block = {"name": fixture.name}
    key = "composite" if fixture.kind == "composite" else "training_source"
    return {"project_type": btp.PROJECT_TYPE_FOR_KIND[fixture.kind], key: block}


def outlines_fixture():
    """The outlines Fixture record."""
    return [f for f in btp.FIXTURES if f.kind == "composite"][0]


def refuse_removal(run_dir, inprocess, needle, where):
    """Expect a removal refusal and check the folder is still on disk."""
    fixture = outlines_fixture()
    refuses(lambda: btp.remove_tutorial_project(inprocess, run_dir, fixture,
                                                lambda *a, **k: None),
            needle, where)
    check(os.path.isdir(run_dir), f"{where}: {run_dir} was removed anyway")


def test_rebuild_refuses_a_project_with_no_marker():
    root = temp_root("nomarker")
    inprocess, _dest = fixture_inprocess(root)
    run_dir = fake_run_dir(inprocess, "run_26-04-18_AT_real_annotation_work",
                           project={"project_type": None, "id": "real"})
    _write(os.path.join(run_dir, "step1_makeAllPoints", "all_points.csv"),
           "irreplaceable\n")
    refuse_removal(run_dir, inprocess, "carries no readable",
                   "a real annotation project")
    check(os.path.isfile(os.path.join(run_dir, "step1_makeAllPoints",
                                      "all_points.csv")),
          "the annotation work inside the refused folder was touched")


def test_rebuild_refuses_a_marker_naming_another_fixture():
    root = temp_root("otherfixture")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    marker = good_marker(fixture)
    marker["fixture"] = "someone_elses_set"
    run_dir = fake_run_dir(inprocess, "run_26-09-17_LO_train_someone_elses_set",
                           marker=marker, project=good_project(fixture))
    refuse_removal(run_dir, inprocess, "someone_elses_set",
                   "a marker naming another fixture")


def test_rebuild_refuses_a_marker_written_by_another_tool():
    root = temp_root("othertool")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    marker = good_marker(fixture)
    marker["tool"] = "somebody/else.py"
    run_dir = fake_run_dir(inprocess, "run_26-09-17_LO_train_tutorial_outlines_2",
                           marker=marker, project=good_project(fixture))
    refuse_removal(run_dir, inprocess, "written by", "a marker from another tool")


def test_rebuild_refuses_a_project_whose_project_json_disagrees():
    root = temp_root("wrongtype")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    run_dir = fake_run_dir(inprocess, "run_26-09-17_LO_train_tutorial_outlines_3",
                           marker=good_marker(fixture),
                           project={"project_type": "annotation",
                                    "composite": {"name": fixture.name}})
    refuse_removal(run_dir, inprocess, "declares project_type",
                   "a folder whose project.json says something else")


def test_rebuild_refuses_a_project_recording_another_source():
    root = temp_root("wrongsource")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    run_dir = fake_run_dir(inprocess, "run_26-09-17_LO_train_tutorial_outlines_4",
                           marker=good_marker(fixture),
                           project={"project_type": btp.COMPOSITE_PROJECT_TYPE,
                                    "composite": {"name": "qs20260902_pa_ss"}})
    refuse_removal(run_dir, inprocess, "records its training source",
                   "a folder recording another training source")


def test_rebuild_refuses_a_corrupt_marker():
    root = temp_root("corruptmarker")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    run_dir = fake_run_dir(inprocess, "run_26-09-17_LO_train_tutorial_outlines_5",
                           raw_marker="{ not json at all",
                           project=good_project(fixture))
    refuse_removal(run_dir, inprocess, "carries no readable",
                   "a marker that is not JSON")


def test_rebuild_refuses_a_folder_outside_the_run_folder_root():
    root = temp_root("outside")
    inprocess, _dest = fixture_inprocess(root)
    fixture = outlines_fixture()
    elsewhere = os.path.join(root, "elsewhere")
    os.makedirs(elsewhere)
    run_dir = fake_run_dir(elsewhere, "run_26-09-17_LO_train_tutorial_outlines",
                           marker=good_marker(fixture),
                           project=good_project(fixture))
    refuse_removal(run_dir, inprocess, "does not sit directly inside",
                   "a folder outside the run folder root")


def test_rebuild_refuses_a_link_pointing_at_a_tutorial_project():
    root = temp_root("linked")
    inprocess, _dest = fixture_inprocess(root)
    records = mint_both(inprocess)
    real = [r for r in records if r["fixture"].kind == "composite"][0]["run_dir"]
    link = os.path.join(inprocess, "run_26-09-17_LO_train_tutorial_outlines_link")
    os.symlink(real, link)
    fixture = outlines_fixture()
    refuses(lambda: btp.remove_tutorial_project(inprocess, link, fixture,
                                                lambda *a, **k: None),
            "is a link", "a link pointing at a tutorial project")
    check(os.path.isdir(real), f"{real} was removed through the link")


def test_a_missing_fixture_is_refused_and_names_its_builder():
    root = temp_root("nofixture")
    inprocess = os.path.join(root, "inprocess")
    os.makedirs(os.path.join(inprocess, bto.COMPOSITE_DIRNAME))
    message = refuses(lambda: mint_both(inprocess), "is not at",
                      "a run folder root with no outlines set")
    check("build_tutorial_outlines.py" in message,
          f"the refusal did not name the builder: {message!r}")


def test_an_empty_fixture_is_refused():
    root = temp_root("emptyfixture")
    inprocess = os.path.join(root, "inprocess")
    empty = os.path.join(inprocess, bto.COMPOSITE_DIRNAME, bto.DEFAULT_NAME)
    os.makedirs(os.path.join(empty, bto.IMAGES_DIR))
    _write(os.path.join(empty, bto.DATA_YAML_NAME), "names:\n  0: AA\n")
    refuses(lambda: btp.check_fixture(outlines_fixture(), inprocess),
            "holds no frames", "a training set with no frames")


def test_an_unreadable_composite_json_still_mints_the_project():
    root = temp_root("badcompositejson")
    inprocess, dest = fixture_inprocess(root)
    _write(os.path.join(dest, bto.COMPOSITE_JSON_NAME), "{ not json")
    records = btp.build_tutorial_projects(inprocess, (outlines_fixture(),),
                                          log=lambda *a, **k: None)
    check(len(records) == 1, "the project was not minted")
    meta = btp._read_project_json(records[0]["run_dir"])
    check(meta["project_type"] == btp.COMPOSITE_PROJECT_TYPE,
          "the minted project is not a training project")


def test_a_corrupt_split_pins_file_does_not_stop_the_counts():
    root = temp_root("badpins")
    inprocess, dest = fixture_inprocess(root)
    _write(os.path.join(dest, bto.SPLIT_PINS_NAME), "{ not json")
    counts = btp.describe_fixture(outlines_fixture(), dest)
    check(counts["n_images"] == FIXTURE_IMAGES,
          f"counted {counts['n_images']} frame(s), expected {FIXTURE_IMAGES}")
    check(counts["by_split"] == {}, "a corrupt pins file produced split counts")


def test_the_projects_command_line_refuses_and_says_so():
    root = temp_root("cli-refuse")
    code = btp.main(["--inprocess", os.path.join(root, "gone"), "--quiet"])
    check(code == 1, f"the command line answered {code}, expected 1")


def test_the_projects_command_line_mints_and_answers_zero():
    root = temp_root("cli-mint")
    inprocess, _dest = fixture_inprocess(root)
    code = btp.main(["--inprocess", inprocess, "--track", "outlines", "--quiet"])
    check(code == 0, f"the command line answered {code}, expected 0")
    found = btp.find_tutorial_projects(inprocess, outlines_fixture())
    check(len(found) == 1, f"the command line left {len(found)} project(s)")
    check(not btp.find_tutorial_projects(
        inprocess, [f for f in btp.FIXTURES if f.kind == "bbox"][0]),
        "--track outlines minted the boxes project too")


def test_an_unknown_track_is_refused():
    refuses(lambda: btp.chosen_fixtures("outlins"), "--track takes one of",
            "a misspelled track")


if __name__ == "__main__":
    try:
        print("BUILT SET:")
        run(test_the_built_set_is_there_with_every_file)
        run(test_frames_and_labels_pair_and_match_the_records)
        run(test_every_frame_link_resolves)
        run(test_no_two_frame_names_differ_only_in_case)
        run(test_split_pins_cover_exactly_the_frames_on_disk)
        run(test_the_pins_are_the_transect_rule)
        run(test_the_split_sizes_are_what_the_quotas_ask_for)
        run(test_every_species_is_in_every_split)
        run(test_every_species_clears_the_splitters_minimum_train_count)
        run(test_the_tools_min_samples_constant_matches_the_splitter)
        run(test_the_class_tables_agree)
        run(test_the_set_is_a_real_subset_of_its_parent)
        run(test_the_parent_pins_nothing_and_this_set_does)

        print("BUILT PROJECTS:")
        run(test_both_tutorial_projects_are_there_with_the_right_type_and_source)
        run(test_the_two_project_names_sort_together)
        run(test_the_outlines_project_reaches_its_set_through_a_relative_symlink)
        run(test_the_boxes_project_holds_no_imagery_and_no_link)
        run(test_the_training_set_names_its_tutorial_project)
        run(test_both_built_projects_carry_a_readable_marker)

        print("OUTLINES BUILDER:")
        run(test_the_outlines_builder_end_to_end)
        run(test_the_outlines_build_is_reproducible_from_the_seed)
        run(test_a_different_seed_chooses_different_frames)
        run(test_a_frame_whose_link_is_broken_is_passed_over)
        run(test_a_parent_with_no_pins_gets_pins_written_here)

        print("OUTLINES BUILDER, ADVERSARIAL:")
        run(test_a_parent_that_is_not_a_training_set_is_refused)
        run(test_a_parent_whose_data_yaml_declares_no_class_is_refused)
        run(test_a_zero_or_fractional_or_boolean_quota_is_refused)
        run(test_a_train_quota_below_the_splitters_minimum_is_refused)
        run(test_a_bucket_with_too_few_frames_is_refused)
        run(test_a_destination_that_already_exists_is_refused)
        run(test_a_destination_whose_holder_is_not_there_is_refused)
        run(test_a_name_outside_the_plain_name_rule_is_refused)
        run(test_a_label_line_that_is_not_a_polygon_is_refused)
        run(test_a_label_naming_an_undeclared_class_is_refused)
        run(test_a_coordinate_outside_zero_to_one_is_refused)
        run(test_a_label_file_with_no_frame_is_refused)
        run(test_a_hostile_label_file_of_nulls_is_refused_not_crashed)
        run(test_a_corrupt_parent_manifest_is_refused_before_anything_is_written)
        run(test_a_build_that_fails_partway_leaves_no_folder_behind)

        print("PROJECTS BUILDER:")
        run(test_both_projects_mint_with_the_right_type_and_source)
        run(test_a_second_run_changes_nothing)
        run(test_rebuild_remakes_a_project_and_leaves_the_fixture_alone)

        print("PROJECTS BUILDER, ADVERSARIAL:")
        run(test_rebuild_refuses_a_project_with_no_marker)
        run(test_rebuild_refuses_a_marker_naming_another_fixture)
        run(test_rebuild_refuses_a_marker_written_by_another_tool)
        run(test_rebuild_refuses_a_project_whose_project_json_disagrees)
        run(test_rebuild_refuses_a_project_recording_another_source)
        run(test_rebuild_refuses_a_corrupt_marker)
        run(test_rebuild_refuses_a_folder_outside_the_run_folder_root)
        run(test_rebuild_refuses_a_link_pointing_at_a_tutorial_project)
        run(test_a_missing_fixture_is_refused_and_names_its_builder)
        run(test_an_empty_fixture_is_refused)
        run(test_an_unreadable_composite_json_still_mints_the_project)
        run(test_a_corrupt_split_pins_file_does_not_stop_the_counts)

        print("COMMAND LINE:")
        run(test_the_outlines_command_line_refuses_and_says_so)
        run(test_the_outlines_command_line_builds_and_answers_zero)
        run(test_the_projects_command_line_refuses_and_says_so)
        run(test_the_projects_command_line_mints_and_answers_zero)
        run(test_an_unknown_track_is_refused)
    finally:
        left = []
        for root in _FIXTURE_ROOTS:
            shutil.rmtree(root, ignore_errors=True)
            if os.path.exists(root):
                left.append(root)
        print(f"\ncleaned up {len(_FIXTURE_ROOTS)} fixture folder(s); "
              f"{len(left)} left on disk" + (f": {left}" if left else ""))

    failed = [r for r in _RESULTS if not r[1]]
    print(f"{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _ok, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
