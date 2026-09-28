#!/usr/bin/env python
"""External run importer: the lab layout, the runs card listing, and every refusal.

No-pytest house harness. Every fixture is a synthetic staged dataset and a set
of fake ultralytics run folders built in a temp dir and removed in a finally
block: no test here reads a real dataset or a real training run, no test starts
a process that trains or evaluates, and nothing is written outside the temp
dirs.

Covers:
  layout      lab.json, dataset/data.yaml, dataset/split_manifest.json, and
              per run weights/best.pt, args.yaml, results.csv, provenance.json
  listing     labruns.list_all_runs sees both imported runs with origin
              lab:<name> and task detect, and split_fingerprint_for_run reads
              the lab's fingerprint through a run folder
  ledger      no rounds.csv row and no champion record is invented
  compare     two imported runs read as comparable with each other, because
              they carry one metric family and one split fingerprint
  pins        the lab's dataset is built on the staged split_pins.json, and its
              fingerprint equals the one a training project gets from the same
              staged folder under different fractions
  refusals    a reserved, slashed or empty lab name, a lab folder that already
              exists, a run folder with no weights/best.pt or no args.yaml, a
              run folder that is not a folder, a run name the ledger could not
              carry or longer than the module's name rule allows, two runs that
              would share one name, a run trained for
              segmentation, an args.yaml that is not text, a dataset that is
              not staged, a fraction out of range, a note that is not text or
              is too long, an empty run list, and a fresh split without the
              override
  rollback    every refusal after the lab folder is claimed removes the lab
              folder, so a half-written lab never survives, and a folder that
              cannot be removed is reported rather than passed over (proved by
              a refusing rmtree, so the branch is exercised on any account)
  reads       weights that cannot be read, and a split_pins.json that is not
              JSON or pins a split outside train, valid and test, each stop the
              import by name
  symlinks    a source run folder that is a symlink to a folder outside the
              module root is copied as a real folder the runs card can list
  paths       a relative projects folder, dataset and run folder are recorded
              absolute, so a provenance record resolves from anywhere
  optional    a run with no results.csv is imported and the absence is recorded
  hashes      a 3 MB best.pt is hashed in chunks and the copy is verified
  concurrent  two threads importing the same lab name: exactly one wins, the
              loser refuses and leaves the winner's lab whole
  cli         the command line writes the lab, and exits 2 with
              "import stopped:" on a refusal
  cleanup     every temp fixture this suite made is gone, and no lab this suite
              named appears in the module's own inprocess/_models folder

Run with:
  env/bin/python scripts/_boxes/tests/test_import_external_runs.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
_REPO = os.path.dirname(_SCRIPTS)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from _boxes import import_external_runs as imp  # noqa: E402
from _modellab import labruns  # noqa: E402

# The importer under test, and the interpreter the module's suites run under.
# The repo's own env is preferred so a CLI test never runs under a foreign
# interpreter; sys.executable covers a checkout whose env is not built yet.
SCRIPT = os.path.join(_SCRIPTS, "_boxes", "import_external_runs.py")
_ENV_PYTHON = os.path.join(_REPO, "env", "bin", "python")
PYTHON = _ENV_PYTHON if os.path.isfile(_ENV_PYTHON) else sys.executable

# Every temp fixture this suite makes carries this prefix, so the closing
# cleanup test can prove each one was removed.
TEMP_PREFIX = "importruns_"

# Lab names this suite writes. The closing cleanup test proves none of them
# appears in the module's own inprocess/_models folder.
LAB_NAMES = ("fish_paper", "lab1", "lab2", "lab3", "sym_lab", "race_lab",
             "cli_lab", "name_lab", "task_lab", "opt_lab", "hash_lab",
             "frac_lab", "arg_lab", "rel_lab", "long_lab", "long_lab2")

# How many images the synthetic staged dataset holds, and how its pins split
# them: four train, two valid, two test.
STAGED_IMAGES = 8
STAGED_COUNTS = {"train": 4, "valid": 2, "test": 2}

# A best.pt big enough to cross the importer's one-megabyte hash chunk several
# times, so the chunked read is exercised without a slow fixture.
HUGE_WEIGHTS_BYTES = 3 * 1024 * 1024

# A sha256 digest in hex.
SHA256_HEX_CHARS = 64

# Length of the AST stamp labruns.ast_now writes: YYYY-MM-DD HH:MM:SS.
AST_STAMP_CHARS = 19

# The longest lab name the module's shared rule allows (labruns.LAB_NAME_RE),
# which an imported run name is held to as well: a run name longer than this
# would be truncated in an evaluation folder name.
LAB_NAME_MAX_CHARS = 60

# What a refused filesystem call says in the rollback and hash-fault test.
DENIED_BY_TEST = "refused by the test"

_pass = 0
_fail = 0

# Absolute paths of the temp directories _mkdtemp handed out this run.
_TEMP_DIRS = []


def check(cond, msg):
    global _pass, _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if cond:
        _pass += 1
    else:
        _fail += 1


def _mkdtemp(tag):
    """A temp directory for one test, recorded so the cleanup test can check it.

    Parameters:
      tag: a short word naming the test, for example "layout".
    Returns:
      the absolute path of the new directory.
    """
    path = tempfile.mkdtemp(prefix=TEMP_PREFIX + tag + "_")
    _TEMP_DIRS.append(path)
    return path


def write(path, text):
    """Write text to path, making the parent folders first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def write_jpeg(path, size=(24, 24), color=(20, 90, 140)):
    """A real (tiny) JPEG, so a staged image is an image a later reader opens."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")


def quiet(_msg):
    """Swallow the importer's progress lines inside a test."""


def _staged(root, name="fish_photo_28feb25", with_pins=True):
    """A staged box dataset in the layout the generic stager writes.

    Writes images/ (STAGED_IMAGES tiny JPEGs), labels/ (one box each),
    classes.txt, data.yaml, names.csv and, by default, split_pins.json pinning
    every image to STAGED_COUNTS.

    Parameters:
      root: temp directory the dataset folder is made in.
      name: the staged folder name, which is also the split seed.
      with_pins: write split_pins.json as well.
    Returns:
      the absolute path of the staged dataset folder.
    """
    dest = os.path.join(root, "bbox_datasets", name)
    names = [f"img_{i:02d}.jpg" for i in range(1, STAGED_IMAGES + 1)]
    for image in names:
        write_jpeg(os.path.join(dest, "images", image))
        write(os.path.join(dest, "labels", os.path.splitext(image)[0] + ".txt"),
              "0 0.5 0.5 0.2 0.2\n")
    write(os.path.join(dest, "classes.txt"), "AAAA_Normal\nBBBB_TP\n")
    write(os.path.join(dest, "data.yaml"),
          "path: .\ntrain: train/images\nval: valid/images\n"
          "names:\n  0: AAAA_Normal\n  1: BBBB_TP\nnc: 2\n")
    write(os.path.join(dest, "names.csv"),
          'token,name\nAAAA_Normal,"Genus a, fish a"\n'
          'BBBB_TP,"Genus b, fish b, terminal phase"\n')
    if with_pins:
        pins = {}
        for image in names[:STAGED_COUNTS["train"]]:
            pins[image] = "train"
        valid_end = STAGED_COUNTS["train"] + STAGED_COUNTS["valid"]
        for image in names[STAGED_COUNTS["train"]:valid_end]:
            pins[image] = "valid"
        for image in names[valid_end:]:
            pins[image] = "test"
        write(os.path.join(dest, "split_pins.json"),
              json.dumps({"schema_version": 1,
                          "source": os.path.join(root, "split"),
                          "pins": pins}, indent=2))
    return dest


def _fake_run(root, name, task="detect", imgsz=640, results=True,
              weights_bytes=2048):
    """A finished ultralytics run folder, the shape the importer reads.

    Parameters:
      root: temp directory the run folder is made in.
      name: the run folder name.
      task: the value of the args.yaml task line ("" writes no task line).
      imgsz: the value of the args.yaml imgsz line.
      results: write results.csv as well.
      weights_bytes: how many bytes weights/best.pt holds.
    Returns:
      the absolute path of the run folder.
    """
    run_dir = os.path.join(root, "training_runs", name)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    with open(os.path.join(run_dir, "weights", "best.pt"), "wb") as fh:
        fh.write(b"\0" * weights_bytes if weights_bytes > 4096
                 else os.urandom(weights_bytes))
    task_line = f"task: {task}\n" if task else ""
    write(os.path.join(run_dir, "args.yaml"),
          f"{task_line}mode: train\nmodel: yolo11l.pt\n"
          f"data: /elsewhere/data.yaml\nsave_dir: /elsewhere/runs/{name}\n"
          f"imgsz: {imgsz}\n")
    if results:
        write(os.path.join(run_dir, "results.csv"), "epoch,time\n1,10\n")
    return run_dir


class _RefusingShutil:
    """A stand-in for the shutil module whose rmtree refuses.

    The importer reaches shutil through its own module global, so putting this
    object there exercises the rollback's report branch without a mode bit,
    which a root account would walk straight through.

    Parameters:
      real: the shutil module every other call is passed to.
      reason: the text the refusal carries.
    """

    def __init__(self, real, reason):
        self._real = real
        self._reason = reason

    def __getattr__(self, name):
        return getattr(self._real, name)

    def rmtree(self, path, *args, **kwargs):
        """Refuse to remove path, the way a read-only parent folder does."""
        raise OSError(f"{self._reason}: {path}")


def _refuse_open(*_args, **_kwargs):
    """Refuse a file read, standing in for a weights file no account can read.

    Returns:
      never; always raises.
    Raises:
      OSError carrying DENIED_BY_TEST.
    """
    raise OSError(DENIED_BY_TEST)


def _refused(fn, what):
    """Run fn, report whether it refused, and return the refusal message.

    Parameters:
      fn: a callable taking no argument.
      what: what the refusal is about, for the PASS or FAIL line.
    Returns:
      the refusal message, or "" when fn did not refuse.
    """
    try:
        fn()
    except imp.ImportRefused as exc:
        check(True, f"{what} refused")
        return str(exc)
    check(False, f"{what} must be refused")
    return ""


# -- layout and listing ---------------------------------------------------

def test_import_layout_and_listing():
    root = _mkdtemp("layout")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        r3 = _fake_run(root, "run3_heavy_augment")
        r6 = _fake_run(root, "run6_1024px_augment", imgsz=1024)
        out = imp.import_runs(inproc, src, "fish_paper", [r3, r6],
                              note="paper runs", log=quiet)
        lab = os.path.join(inproc, "_models", "fish_paper")
        check(out["lab_dir"] == lab, "lab dir returned")
        check(out["dataset_dir"] == os.path.join(lab, "dataset"),
              "dataset dir returned")
        for rel in ("lab.json", "dataset/data.yaml", "dataset/split_manifest.json",
                    "runs/run3_heavy_augment/weights/best.pt",
                    "runs/run3_heavy_augment/args.yaml",
                    "runs/run3_heavy_augment/results.csv",
                    "runs/run3_heavy_augment/provenance.json",
                    "runs/run6_1024px_augment/weights/best.pt"):
            check(os.path.isfile(os.path.join(lab, rel)), f"{rel} written")
        with open(os.path.join(lab, "lab.json")) as fh:
            labj = json.load(fh)
        check(labj["task"] == "detect" and labj["source_kind"] == "bbox"
              and labj["imported_runs"] == ["run3_heavy_augment", "run6_1024px_augment"],
              f"lab.json says what it holds: {labj}")
        check(labj["source_kind"] == imp.LAB_SOURCE_KIND == "bbox",
              "source_kind keeps the module's meaning, the kind of training "
              f"source: {labj['source_kind']}")
        check(labj["origin_kind"] == imp.LAB_ORIGIN_KIND == "imported",
              f"origin_kind says the runs were trained outside VEGA: {labj}")
        check(labj["lab_name"] == "fish_paper" and labj["name"] == "fish_paper",
              f"the lab name is written under both spellings on disk: {labj}")
        check(labj["created_at"] == labj["updated_at"]
              and len(labj["created_at"]) == AST_STAMP_CHARS,
              f"one AST stamp under both spellings: {labj.get('created_at')!r} "
              f"and {labj.get('updated_at')!r}")
        check(labj["model"] == "",
              f"an imported lab records no base weights of its own: {labj['model']!r}")
        check(labj["source"] == src, "lab.json names the staged dataset")
        with open(os.path.join(lab, "runs/run3_heavy_augment/provenance.json")) as fh:
            prov = json.load(fh)
        check(prov["imported_from"] == os.path.realpath(r3), "provenance names the source run")
        check(len(prov["best_pt_sha256"]) == SHA256_HEX_CHARS,
              "provenance carries the weights digest")
        check(prov["best_pt_bytes"] == 2048, "provenance carries the weights size")
        check(prov["args_data_original"] == "/elsewhere/data.yaml",
              "provenance keeps the historical data line")
        check(prov["args_save_dir_original"] == "/elsewhere/runs/run3_heavy_augment",
              "provenance keeps the historical save_dir line")
        check(prov["results_csv"] is True and prov["note"] == "paper runs",
              f"provenance records the optional file and the note: {prov}")
        with open(os.path.join(lab, "dataset/split_manifest.json")) as fh:
            man = json.load(fh)
        check(man["pins_source"] == "staged split_pins.json"
              and man["counts"] == STAGED_COUNTS,
              f"dataset built on the staged pins {man['counts']}")
        check(out["split_fingerprint"] == man["split_fingerprint"],
              "the returned fingerprint is the one on disk")
        rows = labruns.list_all_runs(inproc)
        seen = sorted((r["run_name"], r["origin"], r["task"]) for r in rows)
        check(seen == [("run3_heavy_augment", "lab:fish_paper", "detect"),
                       ("run6_1024px_augment", "lab:fish_paper", "detect")],
              f"the runs card sees both: {seen}")
        check(all(r["has_weights"] for r in rows), "both rows carry weights")
        fp = labruns.split_fingerprint_for_run(
            os.path.join(lab, "runs/run3_heavy_augment"))
        check(fp[0] == man["split_fingerprint"],
              "fingerprint readable through the run")
        check(labruns.run_imgsz(os.path.join(lab, "runs/run6_1024px_augment")) == 1024,
              "the run's own image size survives the copy")
        comparable, why = labruns.runs_comparable(
            os.path.join(lab, "runs", "run3_heavy_augment"),
            os.path.join(lab, "runs", "run6_1024px_augment"))
        check(comparable and "same split" in why,
              f"two imported runs are comparable with each other: {why}")
        check(not os.path.exists(os.path.join(lab, "rounds.csv")),
              "no ledger row is invented")
        check(not os.path.exists(os.path.join(inproc, "_models", "champion.json")),
              "no champion is invented")
        check(len(out["runs"]) == 2
              and out["runs"][0]["run_name"] == "run3_heavy_augment"
              and out["runs"][0]["run_dir"] == os.path.join(lab, "runs", "run3_heavy_augment")
              and len(out["runs"][0]["best_pt_sha256"]) == SHA256_HEX_CHARS,
              f"every imported run is reported back: {out['runs']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fingerprint_matches_a_training_project_split():
    """The whole point of the pins: the lab and a VEGA retrain hold the same
    images out, so the two runs are comparable."""
    root = _mkdtemp("fingerprint")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3_heavy_augment")
        out = imp.import_runs(inproc, src, "fish_paper", [run], log=quiet)
        project = labruns.build_detect_dataset(
            src, os.path.join(root, "project", "step6_trainModel", "dataset"),
            os.path.basename(src), 0.1, 0.1)
        check(project["pins_source"] == "staged split_pins.json",
              "a training project replays the same staged pins")
        check(project["split_fingerprint"] == out["split_fingerprint"],
              "the imported lab and a training project carry one fingerprint")
        check(project["counts"] == STAGED_COUNTS,
              "the project's counts are the pinned counts, not the fractions")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_relative_paths_are_recorded_absolute():
    """A relative invocation still writes absolute paths. provenance.json is
    the archival record of where a model came from, so a source path that can
    only be resolved from the folder the operator happened to stand in is not a
    record at all."""
    root = _mkdtemp("relative")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3_heavy_augment")
        held = os.getcwd()
        os.chdir(root)
        try:
            out = imp.import_runs("inprocess", os.path.relpath(src, root),
                                  "rel_lab", [os.path.relpath(run, root)],
                                  log=quiet)
        finally:
            os.chdir(held)
        lab = os.path.join(root, "inprocess", "_models", "rel_lab")
        for key in ("lab_dir", "dataset_dir"):
            check(os.path.isabs(out[key]), f"the returned {key} is absolute: {out[key]}")
        check(os.path.realpath(out["lab_dir"]) == os.path.realpath(lab),
              f"the lab is where the relative root points: {out['lab_dir']}")
        check(os.path.isabs(out["runs"][0]["run_dir"]),
              f"the returned run dir is absolute: {out['runs'][0]['run_dir']}")
        with open(os.path.join(out["runs"][0]["run_dir"], "provenance.json")) as fh:
            prov = json.load(fh)
        check(prov["imported_from"] == os.path.realpath(run),
              f"provenance resolves the source run: {prov['imported_from']}")
        check(os.path.isabs(prov["imported_from_given"])
              and os.path.realpath(prov["imported_from_given"]) == os.path.realpath(run),
              "the path the operator typed is recorded absolute: "
              f"{prov['imported_from_given']}")
        with open(os.path.join(out["lab_dir"], "lab.json")) as fh:
            labj = json.load(fh)
        check(os.path.isabs(labj["source"])
              and os.path.realpath(labj["source"]) == os.path.realpath(src),
              f"lab.json names the staged dataset absolutely: {labj['source']}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- refusals -------------------------------------------------------------

def test_refusals():
    root = _mkdtemp("refuse")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        r3 = _fake_run(root, "run3")
        for lab, why in (("_eval", "reserved"), ("_inference", "reserved second"),
                         ("bad/name", "slash"), ("", "empty"),
                         ("../escape", "traversal")):
            _refused(lambda lab=lab: imp.import_runs(inproc, src, lab, [r3], log=quiet),
                     f"{why} lab name")
        os.remove(os.path.join(r3, "weights", "best.pt"))
        msg = _refused(lambda: imp.import_runs(inproc, src, "lab1", [r3], log=quiet),
                       "run without weights")
        check("best.pt" in msg, "the refusal names the missing file")
        check(not os.path.exists(os.path.join(inproc, "_models", "lab1")),
              "the refused lab folder was never claimed")
        r4 = _fake_run(root, "run4")
        imp.import_runs(inproc, src, "lab2", [r4], log=quiet)
        msg = _refused(lambda: imp.import_runs(inproc, src, "lab2", [r4], log=quiet),
                       "existing lab")
        check("lab2" in msg, "the refusal names the lab")
        check(os.path.isfile(os.path.join(inproc, "_models", "lab2", "lab.json")),
              "the refused second import left the first lab whole")
        os.remove(os.path.join(src, "split_pins.json"))
        msg = _refused(lambda: imp.import_runs(inproc, src, "lab3", [r4], log=quiet),
                       "fresh split without the override")
        check("fresh" in msg, "the refusal explains the split")
        check(not os.path.exists(os.path.join(inproc, "_models", "lab3")),
              "the half-written lab was removed")
        out = imp.import_runs(inproc, src, "lab3", [r4], allow_fresh_split=True,
                              log=quiet)
        check(os.path.isdir(out["lab_dir"]), "the override allows a fresh split")
        with open(os.path.join(out["dataset_dir"], "split_manifest.json")) as fh:
            check(json.load(fh)["pins_source"] == "fresh",
                  "the override records the split as fresh")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_argument_refusals():
    root = _mkdtemp("args")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3")
        cases = (
            ("empty projects folder", lambda: imp.import_runs("", src, "lab1", [run], log=quiet)),
            ("no run folder at all", lambda: imp.import_runs(inproc, src, "lab1", [], log=quiet)),
            ("a run list that is one string",
             lambda: imp.import_runs(inproc, src, "lab1", run, log=quiet)),
            ("a note that is not text",
             lambda: imp.import_runs(inproc, src, "lab1", [run], note={"a": 1}, log=quiet)),
            ("a note longer than the record keeps",
             lambda: imp.import_runs(inproc, src, "lab1", [run],
                                     note="x" * (imp.NOTE_MAX_CHARS + 1), log=quiet)),
            ("an override that is not a true or false",
             lambda: imp.import_runs(inproc, src, "lab1", [run],
                                     allow_fresh_split="yes", log=quiet)),
            ("a dataset folder that is not there",
             lambda: imp.import_runs(inproc, os.path.join(root, "gone"), "lab1",
                                     [run], log=quiet)),
            ("a dataset folder with no data.yaml",
             lambda: imp.import_runs(inproc, root, "lab1", [run], log=quiet)),
            ("a run folder that is a file",
             lambda: imp.import_runs(inproc, src, "lab1",
                                     [os.path.join(src, "data.yaml")], log=quiet)),
        )
        for what, fn in cases:
            _refused(fn, what)
        msg = _refused(lambda: imp.import_runs(inproc, src, "frac_lab", [run],
                                               valid_frac=1.5, log=quiet),
                       "a validation fraction out of range")
        check("valid_frac" in msg, f"the refusal names the fraction: {msg}")
        check(not os.path.exists(os.path.join(inproc, "_models", "frac_lab")),
              "a refused fraction leaves no lab folder")
        check(not os.path.isdir(os.path.join(inproc, "_models", "lab1")),
              "no lab folder was claimed by any refused call")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_malformed_split_pins_stops_the_import():
    """A staged dataset whose pins cannot be read is refused by name, not
    quietly re-split: a silent re-split is the mistake the pins exist to stop."""
    root = _mkdtemp("pins")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3")
        write(os.path.join(src, "split_pins.json"), "not json at all")
        msg = _refused(lambda: imp.import_runs(inproc, src, "lab1", [run], log=quiet),
                       "a split_pins.json that is not JSON")
        check("split_pins.json" in msg, f"the refusal names the file: {msg}")
        check(not os.path.exists(os.path.join(inproc, "_models", "lab1")),
              "the half-written lab was removed")
        write(os.path.join(src, "split_pins.json"),
              json.dumps({"pins": {"img_01.jpg": "validation"}}))
        msg = _refused(lambda: imp.import_runs(inproc, src, "lab1", [run], log=quiet),
                       "a pin naming a split that is not train, valid or test")
        check("validation" in msg, f"the refusal quotes the bad pin: {msg}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_name_refusals():
    root = _mkdtemp("names")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        spaced = _fake_run(root, "run with spaces")
        msg = _refused(lambda: imp.import_runs(inproc, src, "name_lab", [spaced], log=quiet),
                       "a run folder name the ledger could not carry")
        check("run with spaces" in msg, f"the refusal names the folder: {msg}")
        msg = _refused(lambda: imp.import_runs(inproc, src, "name_lab", ["/"], log=quiet),
                       "a run path with no folder name")
        twin_root = os.path.join(root, "other_parent")
        twin = _fake_run(twin_root, "run3")
        mine = _fake_run(root, "run3")
        msg = _refused(lambda: imp.import_runs(inproc, src, "name_lab", [mine, twin],
                                               log=quiet),
                       "two run folders that would share one name")
        check("run3" in msg, f"the refusal names the shared name: {msg}")
        check(not os.path.exists(os.path.join(inproc, "_models", "name_lab")),
              "no lab folder was claimed by a refused name")
        longest = "r" * imp.RUN_NAME_MAX_CHARS
        out = imp.import_runs(inproc, src, "long_lab",
                              [_fake_run(root, longest)], log=quiet)
        check(os.path.isdir(os.path.join(out["lab_dir"], "runs", longest)),
              f"a run name of {imp.RUN_NAME_MAX_CHARS} characters is imported")
        too_long = _fake_run(root, "r" * (imp.RUN_NAME_MAX_CHARS + 1))
        msg = _refused(lambda: imp.import_runs(inproc, src, "long_lab2", [too_long],
                                               log=quiet),
                       f"a run name longer than {imp.RUN_NAME_MAX_CHARS} characters")
        check(str(imp.RUN_NAME_MAX_CHARS) in msg,
              f"the refusal names the limit the module keeps: {msg}")
        check(imp.RUN_NAME_MAX_CHARS == LAB_NAME_MAX_CHARS,
              "a run name and a lab name are capped alike, so an evaluation "
              "folder name is never truncated into a twin")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_task_refusals():
    root = _mkdtemp("task")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        seg = _fake_run(root, "seg_run", task="segment")
        msg = _refused(lambda: imp.import_runs(inproc, src, "task_lab", [seg], log=quiet),
                       "a run trained for segmentation")
        check("segment" in msg, f"the refusal names the task it read: {msg}")
        blank = _fake_run(root, "no_task_line", task="")
        out = imp.import_runs(inproc, src, "task_lab", [blank], log=quiet)
        rows = labruns.list_all_runs(inproc)
        check([r["task"] for r in rows] == ["detect"],
              "a run whose args.yaml names no task is read as the lab's task")
        check(os.path.isdir(out["lab_dir"]), "that run was imported")
        binary = _fake_run(root, "unreadable_args")
        with open(os.path.join(binary, "args.yaml"), "wb") as fh:
            fh.write(b"task: detect\n\xff\xfe\x00\n")
        msg = _refused(lambda: imp.import_runs(inproc, src, "arg_lab", [binary], log=quiet),
                       "an args.yaml that is not text")
        check("args.yaml" in msg, f"the refusal names the file: {msg}")
        missing = _fake_run(root, "no_args")
        os.remove(os.path.join(missing, "args.yaml"))
        msg = _refused(lambda: imp.import_runs(inproc, src, "arg_lab", [missing], log=quiet),
                       "a run with no args.yaml")
        check("args.yaml" in msg, "the refusal names the missing file")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- hostile shapes -------------------------------------------------------

def test_symlinked_run_folder():
    """A source run folder reached through a symlink out of the module root is
    copied as a real folder, so the runs card lists it instead of skipping it."""
    root = _mkdtemp("symlink")
    outside = _mkdtemp("symlink_target")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        real = _fake_run(outside, "run3_heavy_augment")
        link = os.path.join(root, "linked_run")
        os.symlink(real, link)
        out = imp.import_runs(inproc, src, "sym_lab", [link], log=quiet)
        copied = os.path.join(out["lab_dir"], "runs", "linked_run")
        check(os.path.isdir(copied) and not os.path.islink(copied),
              "the imported run folder is a real folder, not a link")
        check(os.path.isfile(os.path.join(copied, "weights", "best.pt"))
              and not os.path.islink(os.path.join(copied, "weights", "best.pt")),
              "the weights were copied, not linked")
        rows = labruns.list_all_runs(inproc)
        check([(r["run_name"], r["origin"]) for r in rows]
              == [("linked_run", "lab:sym_lab")],
              f"the runs card lists the imported copy: {rows}")
        with open(os.path.join(copied, "provenance.json")) as fh:
            prov = json.load(fh)
        check(prov["imported_from"] == os.path.realpath(real),
              "provenance records where the link pointed")
        check(prov["imported_from_given"] == link,
              "provenance also records the path the operator named")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def test_missing_results_csv():
    root = _mkdtemp("optional")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run_no_results", results=False)
        out = imp.import_runs(inproc, src, "opt_lab", [run], log=quiet)
        run_dir = out["runs"][0]["run_dir"]
        check(not os.path.exists(os.path.join(run_dir, "results.csv")),
              "no results.csv is invented")
        with open(os.path.join(run_dir, "provenance.json")) as fh:
            check(json.load(fh)["results_csv"] is False,
                  "provenance records that the run had no results.csv")
        rows = labruns.list_all_runs(inproc)
        check(len(rows) == 1 and rows[0]["map50"] is None,
              "the runs card lists the run with no score of its own")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_large_weights_are_hashed_in_chunks():
    root = _mkdtemp("hash")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run_big", weights_bytes=HUGE_WEIGHTS_BYTES)
        out = imp.import_runs(inproc, src, "hash_lab", [run], log=quiet)
        copied = os.path.join(out["runs"][0]["run_dir"], "weights", "best.pt")
        check(os.path.getsize(copied) == HUGE_WEIGHTS_BYTES,
              "the whole weights file was copied")
        digest = imp.sha256_of(copied)
        check(digest == out["runs"][0]["best_pt_sha256"],
              "the reported digest is the digest of the copy")
        check(digest == imp.sha256_of(os.path.join(run, "weights", "best.pt")),
              "the copy has the digest of the source")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rollback_and_hash_faults():
    """The two failures that must never pass in silence: a lab folder the
    rollback cannot remove, and a weights file that cannot be read. Both are
    staged twice, once by refusing the call the importer makes (which holds on
    any account, root included) and once with a mode bit (skipped for an
    account no mode bit denies)."""
    root = _mkdtemp("rollback")
    models = os.path.join(root, "_models")
    lab = os.path.join(models, "stuck")
    os.makedirs(lab)
    weights = os.path.join(root, "best.pt")
    write(weights, "weights")
    try:
        lines = []
        imp._remove_claimed_lab(os.path.join(models, "never_made"), lines.append)
        check(not lines, "removing a lab folder that is not there says nothing")
        imp.shutil = _RefusingShutil(shutil, DENIED_BY_TEST)
        try:
            imp._remove_claimed_lab(lab, lines.append)
        finally:
            imp.shutil = shutil
        check(os.path.isdir(lab), "the folder that cannot be removed is still there")
        check(any("could not be removed" in line and lab in line for line in lines),
              f"the rollback says which folder was left behind: {lines}")
        imp.open = _refuse_open
        try:
            msg = _refused(lambda: imp.sha256_of(weights),
                           "weights whose read is refused")
        finally:
            del imp.open
        check(weights in msg and DENIED_BY_TEST in msg,
              f"the refusal names the file and why: {msg}")
        if os.geteuid() == 0:
            check(True, "mode-bit faults skipped: no mode bit denies this account")
        else:
            os.chmod(models, 0o500)
            mode_lines = []
            imp._remove_claimed_lab(lab, mode_lines.append)
            check(os.path.isdir(lab) and any("could not be removed" in line
                                             for line in mode_lines),
                  f"a folder in an unwritable parent is reported: {mode_lines}")
            os.chmod(weights, 0o000)
            msg = _refused(lambda: imp.sha256_of(weights),
                           "weights an unreadable mode bit hides")
            check(weights in msg, f"the refusal names the file: {msg}")
    finally:
        os.chmod(models, 0o700)
        os.chmod(weights, 0o600)
        shutil.rmtree(root, ignore_errors=True)


def test_concurrent_import_claims_the_lab_once():
    """Two threads, one lab name: the folder creation is the claim, so exactly
    one import writes the lab and the other refuses without touching it."""
    root = _mkdtemp("race")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3_heavy_augment")
        both_ready = threading.Barrier(2)
        outcomes = []
        guard = threading.Lock()

        def attempt():
            both_ready.wait()
            try:
                result = imp.import_runs(inproc, src, "race_lab", [run], log=quiet)
                with guard:
                    outcomes.append(("imported", result["lab_dir"]))
            except imp.ImportRefused as exc:
                with guard:
                    outcomes.append(("refused", str(exc)))

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        check(not any(t.is_alive() for t in threads), "both threads finished")
        kinds = sorted(kind for kind, _ in outcomes)
        check(kinds == ["imported", "refused"],
              f"exactly one import won the name: {kinds}")
        lab = os.path.join(inproc, "_models", "race_lab")
        for rel in ("lab.json", "dataset/split_manifest.json",
                    "runs/run3_heavy_augment/weights/best.pt",
                    "runs/run3_heavy_augment/provenance.json"):
            check(os.path.isfile(os.path.join(lab, rel)),
                  f"the winner's {rel} survived the loser's refusal")
        rows = labruns.list_all_runs(inproc)
        check(len(rows) == 1, f"one lab run is listed, not two: {rows}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- command line ---------------------------------------------------------

def test_cli_end_to_end():
    root = _mkdtemp("cli")
    inproc = os.path.join(root, "inprocess")
    try:
        src = _staged(root)
        run = _fake_run(root, "run3_heavy_augment")
        proc = subprocess.run(
            [PYTHON, SCRIPT, "--inprocess-root", inproc, "--dataset", src + os.sep,
             "--lab", "cli_lab", "--runs", run, "--note", "paper runs", "--quiet"],
            capture_output=True, text=True, timeout=300)
        check(proc.returncode == 0,
              f"the command line imported the runs (exit {proc.returncode}, "
              f"stderr {proc.stderr.strip()[:200]})")
        lab = os.path.join(inproc, "_models", "cli_lab")
        check(os.path.isfile(os.path.join(lab, "lab.json")), "the lab was written")
        with open(os.path.join(lab, "lab.json")) as fh:
            labj = json.load(fh)
        check(labj["source"] == src,
              f"a dataset path with a trailing slash is recorded without it: {labj['source']}")
        with open(os.path.join(lab, "dataset", "split_manifest.json")) as fh:
            man = json.load(fh)
        check(man["seed"] == os.path.basename(src),
              f"the split seed is the staged folder name: {man['seed']}")
        check(man["counts"] == STAGED_COUNTS, "the pinned counts survived the CLI")
        loud = subprocess.run(
            [PYTHON, SCRIPT, "--inprocess-root", inproc, "--dataset", src,
             "--lab", "_eval", "--runs", run],
            capture_output=True, text=True, timeout=300)
        check(loud.returncode == 2, f"a refusal exits 2 (got {loud.returncode})")
        check("import stopped:" in loud.stderr,
              f"the refusal reaches stderr: {loud.stderr.strip()[:200]}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- cleanup --------------------------------------------------------------

def test_temp_fixtures_removed():
    left = [p for p in _TEMP_DIRS if os.path.exists(p)]
    check(not left, f"temp fixtures removed (still present: {left})")
    own_source = open(os.path.abspath(__file__), encoding="utf-8").read()
    # Spelled in two pieces so this line is not itself one of the matches.
    needle = "tempfile." + "mkdtemp("
    check(own_source.count(needle) == 1,
          "every fixture in this suite is made through _mkdtemp, so this check "
          "sees all of them")
    models_root = os.path.join(os.path.dirname(_REPO), "inprocess", "_models")
    if os.path.isdir(models_root):
        intruders = sorted(set(os.listdir(models_root)).intersection(LAB_NAMES))
        check(not intruders,
              f"no lab this suite named appears in the module tree (found {intruders})")
    else:
        check(True, "the module has no inprocess/_models folder to check")


TESTS = [
    test_import_layout_and_listing,
    test_fingerprint_matches_a_training_project_split,
    test_relative_paths_are_recorded_absolute,
    test_refusals,
    test_argument_refusals,
    test_malformed_split_pins_stops_the_import,
    test_run_name_refusals,
    test_task_refusals,
    test_symlinked_run_folder,
    test_missing_results_csv,
    test_large_weights_are_hashed_in_chunks,
    test_rollback_and_hash_faults,
    test_concurrent_import_claims_the_lab_once,
    test_cli_end_to_end,
    test_temp_fixtures_removed,
]


def main():
    print("external run importer:")
    for test in TESTS:
        print(f"{test.__name__}:")
        test()
    print(f"\n{_pass} passed, {_fail} failed")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
