#!/usr/bin/env python
"""Every route of the Bounding Boxes blueprint (scripts/_boxes).

No-pytest house harness. subprocess.run and subprocess.Popen are replaced at
MODULE level before the blueprint is imported, so no test here can start an
ingest or a SAM3 conversion: the fake records the command instead. Nothing
binds a port; the Flask test client drives the app in-process. Every dataset
the tests read is a synthetic fixture staged into a temp dir.

Covers:
  page       five section markers in DOM order (Train in VEGA sits second,
             and the three that follow it live inside the collapsed advanced
             block), tooltips on every control, cache-bust string, the
             coverage-matrix note, the footer, and no reference anywhere to
             the retired Train + Infer page or its mount
  handoff    GET /api/training-projects lists the projects minted for a
             staged dataset and refuses a name the page did not list;
             POST /api/create-project mints a project on the first press,
             answers the existing one on the next, mints a second only when
             asked, and always answers the main-window URL to open
  datasets   staged dataset cards from ingest_manifest.json, license and
             citation rendered verbatim, the source answered as a folder name
             with no path anywhere in the answer, no page-level citation,
             terms, coverage note or root, the facts a card reads from the
             sidecars (source kind, names table, split pins, background
             images), the two texts the generic stager writes for nothing
             recorded normalised to nothing at all, a pins file the split
             builder would refuse reported as unreadable rather than as
             absent, damaged and oversized sidecars read as absent, the
             page's copies of three writers' constants held against those
             writers, and the Coral Intel citation filling a gap for Coral
             Intel alone
  ingest     command and destination, no source path echoed to the browser,
             refused when the source is missing without echoing the path, 409
             while a job is running
  convert    dataset and limit validation, command with --limit, output path
             under inprocess/_bbox, never inside a project
  status     liveness reconcile flips a dead process to error with its code,
             and there is no separate log route for the page to leave unused
  stop       the stop path signalling only our own process group, and the
             refusal to signal a group that is not ours
  outputs    converted datasets listed from inprocess/_bbox
  assets     the page script calls only this blueprint's routes and opens
             the answered URL in the one VEGA window

Run with:
  env/bin/python scripts/_boxes/tests/test_boxes_routes.py
"""

import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_LAUNCHES = []

# A pid past /proc/sys/kernel/pid_max, so even an unstubbed signal call could
# not name a real process. The stop path also has os.getpgid/os.killpg stubbed
# below: a plausible pid here would let the stop test signal whatever process
# on this box happens to hold it.
_IMPOSSIBLE_PID = 2 ** 22 + 9999

_SIGNALLED = []
_PGID_OF = {"mode": "own"}

# Prose the page must not carry: the retired page, the banned word in every
# form, the old institution string, and typographic dashes.
_BANNED_WORD_RE = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
# The retired Train + Infer mount, assembled from parts so that the grep
# sweep proving this package independent of it ("/" + "models" under
# scripts/_boxes) finds nothing, this file included.
_RETIRED_MOUNT = "/" + "models"


def _fake_getpgid(pid):
    # start_new_session=True makes our own child its own group leader, so the
    # real call returns the pid itself. "foreign" simulates a pid that is not
    # ours, which stop() must refuse to signal.
    return pid if _PGID_OF["mode"] == "own" else 1


def _fake_killpg(pgid, sig):
    _SIGNALLED.append((pgid, sig))


_REAL_GETPGID = os.getpgid
_REAL_KILLPG = os.killpg
os.getpgid = _fake_getpgid
os.killpg = _fake_killpg


class _FakeProc:
    """Stands in for a launched ingest or conversion process."""

    def __init__(self, cmd, cwd=None, env=None, **_kw):
        self.cmd = list(cmd)
        self.cwd = cwd
        self.env = dict(env or {})
        self.pid = _IMPOSSIBLE_PID
        self.returncode = None
        self.stdout = io.StringIO("first line\nsecond line\n")
        self.stderr = None
        self.killed = False
        _LAUNCHES.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -15


def _blocked_run(*_a, **_k):
    raise RuntimeError("subprocess.run blocked in tests")


_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen
subprocess.run = _blocked_run
subprocess.Popen = _FakeProc

from flask import Flask  # noqa: E402

from _boxes import blueprint as bpmod  # noqa: E402
from _boxes import ingest_coral_intel as ing  # noqa: E402
from _boxes import ingest_yolo_boxes as ing_yolo  # noqa: E402
from _boxes.tests.test_ingest_coral_intel import build_source, write  # noqa: E402

_fail = 0


def check(cond, msg):
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def tag_for(html, cid):
    """The opening tag carrying id="<cid>", so a tooltip check reads the
    control itself rather than any title= anywhere on the page."""
    at = html.find(f'id="{cid}"')
    if at < 0:
        return ""
    start = html.rfind("<", 0, at)
    end = html.find(">", at)
    return html[start:end + 1] if start >= 0 and end > start else ""


def run_dirs(inprocess):
    return sorted(d for d in os.listdir(inprocess) if d.startswith("run_"))


# ── the honest dataset card (T5) ───────────────────────────────────────
# Three facts every card gains from the manifest, and the rule that keeps
# the Coral Intel paper's citation on the Coral Intel dataset only.

# The scale case: a pinned dataset far larger than the fish one, to prove the
# per-split counts are read from the file rather than the manifest.
_MANY_PINS = 60000


def check_second_dataset_card_facts(c, bbox_root):
    """A second staged dataset shows its own facts and borrows none."""
    d = os.path.join(bbox_root, "fish_test")
    os.makedirs(os.path.join(d, "images"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    write(os.path.join(d, "data.yaml"), "path: .\nnames:\n  0: X_Normal\nnc: 1\n")
    write(os.path.join(d, "names.csv"), 'token,name\nX_Normal,"Genus x, fish x"\n')
    write(os.path.join(d, "split_pins.json"),
          '{"schema_version": 1, "source": "s", '
          '"pins": {"a.jpg": "train", "b.jpg": "test"}}')
    write(os.path.join(d, "ingest_manifest.json"), json.dumps(
        {"dataset": "fish_test", "n_images": 2, "n_boxes": 3, "n_classes": 1,
         "classes": ["X_Normal"], "source_kind": "yolo",
         "names_csv": "names.csv", "split_pins": True, "n_background": 1,
         "ingested_at": "2026-09-15 10:00:00"}))
    entries = {e["name"]: e for e in c.get("/boxes/api/datasets").get_json()["datasets"]}
    e = entries["fish_test"]
    check(e["citation"] == "" and e["license_note"] == "",
          f"a second dataset borrows no citation and no license "
          f"(got {e['citation']!r} and {e['license_note'][:40]!r})")
    check(e["source_kind"] == "yolo" and e["has_names"] is True and e["n_names"] == 1,
          f"the card carries the source kind and the names count (got {e})")
    check(e["has_split_pins"] is True
          and e["split_counts"] == {"train": 1, "valid": 0, "test": 1}
          and e["split_pins_unreadable"] is False,
          f"the card carries the per-split pin counts (got {e})")
    check(e["n_background"] == 1,
          f"the card carries the count of images with no box (got "
          f"{e['n_background']!r})")
    coral = entries[ing.DATASET_NAME]
    check(coral["citation"] == ing.CITATION
          and coral["license_note"] == ing.LICENSE_NOTE,
          "Coral Intel keeps its own citation and its own terms")
    check(coral["source_kind"] == "coral_intel" and coral["has_names"] is False
          and coral["n_names"] == 0 and coral["has_split_pins"] is False
          and coral["split_counts"] is None
          and coral["split_pins_unreadable"] is False,
          f"Coral Intel reads as its own stager with no sidecars (got "
          f"{coral['source_kind']!r}, {coral['n_names']}, {coral['split_counts']})")
    check(coral["n_background"] is None,
          f"a stager that counts no background images answers null, never a "
          f"zero the card would read as a fact (got {coral['n_background']!r})")


def check_citation_fallback_is_coral_intel_only():
    """The paper's citation and terms fill a gap for one dataset name only."""
    root = tempfile.mkdtemp(prefix="boxes_routes_fallback_")
    try:
        for name in (ing.DATASET_NAME, "fish_photo_28feb25"):
            d = os.path.join(root, name)
            write(os.path.join(d, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
            write(os.path.join(d, "ingest_manifest.json"), json.dumps(
                {"dataset": name, "n_images": 1, "classes": ["A"]}))
        listed = {x["name"]: x for x in bpmod.list_datasets(root)}
        coral = listed[ing.DATASET_NAME]
        check(coral["citation"] == ing.CITATION
              and coral["license_note"] == ing.LICENSE_NOTE,
              "a Coral Intel manifest without terms still shows the paper's")
        check(coral["source_kind"] == "coral_intel",
              f"a Coral Intel manifest without a source kind reads as its own "
              f"stager (got {coral['source_kind']!r})")
        other = listed["fish_photo_28feb25"]
        check(other["citation"] == "" and other["license_note"] == "",
              f"another dataset with nothing recorded shows nothing "
              f"(got {other['citation']!r})")
        check(other["source_kind"] == "",
              f"an unrecorded source kind stays empty (got {other['source_kind']!r})")
        # What the generic stager writes when the operator supplies no
        # citation and no license: the two sentinels, which mean the dataset
        # records nothing. The page answers nothing at all for both, so no
        # card can claim terms for a dataset whose own text says there are
        # none.
        d = os.path.join(root, "staged_with_no_terms")
        write(os.path.join(d, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
        write(os.path.join(d, "ingest_manifest.json"), json.dumps(
            {"dataset": "staged_with_no_terms",
             "citation": ing_yolo.NO_CITATION,
             "license_note": ing_yolo.NO_LICENSE_NOTE,
             "source_kind": "yolo"}))
        # Terms the operator did supply pass through word for word.
        real = os.path.join(root, "staged_with_terms")
        write(os.path.join(real, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
        write(os.path.join(real, "ingest_manifest.json"), json.dumps(
            {"dataset": "staged_with_terms", "citation": "Reyes et al. 2025",
             "license_note": "CC BY 4.0", "source_kind": "yolo"}))
        listed = {x["name"]: x for x in bpmod.list_datasets(root)}
        plain = listed["staged_with_no_terms"]
        check(plain["citation"] == "" and plain["license_note"] == "",
              f"the stager's words for nothing recorded answer nothing at all "
              f"(got {plain['citation']!r} and {plain['license_note'][:40]!r})")
        kept = listed["staged_with_terms"]
        check(kept["citation"] == "Reyes et al. 2025"
              and kept["license_note"] == "CC BY 4.0",
              f"terms the operator supplied pass through word for word "
              f"(got {kept['citation']!r} and {kept['license_note']!r})")
        # Whitespace around a sentinel is still the sentinel: a manifest
        # edited by hand must not turn nothing into terms.
        padded = os.path.join(root, "staged_with_padded_sentinel")
        write(os.path.join(padded, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
        write(os.path.join(padded, "ingest_manifest.json"), json.dumps(
            {"dataset": "staged_with_padded_sentinel",
             "citation": "  " + ing_yolo.NO_CITATION + "\n",
             "license_note": " " + ing_yolo.NO_LICENSE_NOTE + " ",
             "source_kind": "yolo"}))
        padded_entry = {x["name"]: x for x in bpmod.list_datasets(root)}[
            "staged_with_padded_sentinel"]
        check(padded_entry["citation"] == "" and padded_entry["license_note"] == "",
              f"a padded sentinel is still nothing recorded "
              f"(got {padded_entry['citation']!r})")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check(not os.path.exists(root), "the fallback fixture is removed")


def check_page_constants_match_their_writers():
    """The page copies three writers' constants rather than importing them,
    so that a stager which fails to import can never stop the Bounding Boxes
    page from rendering. This test holds the copies against the originals: a
    rename in either writer fails here rather than making a card read none
    for a dataset that carries the fact."""
    detect = _load_detect_dataset()
    check(bpmod.NAMES_CSV_HEADER == ing_yolo.NAMES_HEADER
          == detect.NAMES_CSV_COLUMNS,
          f"the names-table header matches both readers of it (page "
          f"{bpmod.NAMES_CSV_HEADER}, stager {ing_yolo.NAMES_HEADER}, "
          f"split builder {detect.NAMES_CSV_COLUMNS})")
    check(bpmod.SPLIT_NAMES == ing_yolo.SPLIT_DIRS == detect.SPLIT_DIRS,
          f"the three split names match the stager and the split builder "
          f"(page {bpmod.SPLIT_NAMES}, stager {ing_yolo.SPLIT_DIRS}, "
          f"split builder {detect.SPLIT_DIRS})")
    check(bpmod.NAMES_CSV == detect.NAMES_CSV_NAME
          and bpmod.SPLIT_PINS_JSON == detect.SOURCE_PINS_NAME,
          f"the two sidecar file names match the split builder (page "
          f"{bpmod.NAMES_CSV!r} and {bpmod.SPLIT_PINS_JSON!r})")
    check(bpmod.NO_CITATION_RECORDED == ing_yolo.NO_CITATION
          and bpmod.NO_LICENSE_RECORDED == ing_yolo.NO_LICENSE_NOTE,
          f"the two texts that mean nothing recorded match the stager that "
          f"writes them (page {bpmod.NO_CITATION_RECORDED!r})")


def _load_detect_dataset():
    """The split builder module, loaded from its path.

    The page never imports it (an import failure must not stop a card from
    rendering), so this test loads it the way the orchestrator does, by file
    path, and reads the constants the page mirrors.

    Returns: the detect_dataset module object.
    """
    import importlib.util
    path = os.path.join(_SCRIPTS, "TCRMPtrain_oceankindCV", "src",
                        "detect_dataset.py")
    spec = importlib.util.spec_from_file_location("t5_detect_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage_sidecar_case(root, name, names_csv, pins_json):
    """One staged dataset carrying the sidecars a case wants, or neither."""
    d = os.path.join(root, name)
    write(os.path.join(d, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
    write(os.path.join(d, "ingest_manifest.json"), json.dumps(
        {"dataset": name, "n_images": 1, "classes": ["A"], "source_kind": "yolo",
         "split_pins": True}))
    if names_csv is not None:
        write(os.path.join(d, "names.csv"), names_csv)
    if pins_json is not None:
        write(os.path.join(d, "split_pins.json"), pins_json)
    return d


def check_damaged_sidecars_never_break_the_listing():
    """A sidecar that is malformed, hostile or huge never breaks the listing,
    and a pins file the split builder would refuse reads as unreadable rather
    than as a dataset that pins nothing.

    The two readers of split_pins.json have to agree on what a valid file is:
    detect_dataset.read_source_pins refuses a pin naming anything outside
    train, valid and test, so the card must not count such a file as pins and
    must not call it absent either. Absent means VEGA splits the dataset
    fresh, which is a state the operator needs do nothing about.
    """
    root = tempfile.mkdtemp(prefix="boxes_routes_sidecars_")
    # A sidecar bigger than what the page will read, built from one long JSON
    # string so the fixture costs one write rather than a hundred thousand pins.
    over_cap_pad = "x" * (bpmod.SIDECAR_MAX_BYTES + 1)
    many = json.dumps({"schema_version": 1, "pins": {
        f"img_{i:06d}.jpg": ("train" if i % 3 else "test")
        for i in range(_MANY_PINS)}})
    # name, names.csv, split_pins.json, expected n_names, expected
    # split_counts, expected split_pins_unreadable
    cases = (
        ("none_at_all", None, None, 0, None, False),
        ("names_header_only", "token,name\n", None, 0, None, False),
        ("names_wrong_header", "code,species\nA,Alpha\n", None, 0, None, False),
        ("names_no_header", "A,Alpha\nB,Beta\n", None, 0, None, False),
        ("names_blank_name", 'token,name\nA,\nB,"Beta b"\n', None, 1, None, False),
        ("names_short_row", "token,name\nA\nB,Beta\n", None, 1, None, False),
        ("names_over_cap", "token,name\nA,Alpha\n" + over_cap_pad, None, 0, None, False),
        ("pins_is_a_list", None, '[{"a.jpg": "train"}]', 0, None, True),
        ("pins_not_json", None, "not json at all", 0, None, True),
        ("pins_empty_file", None, "", 0, None, True),
        ("pins_not_an_object", None, '{"pins": "train"}', 0, None, True),
        ("pins_no_pins_key", None, '{"schema_version": 1, "source": "s"}', 0, None, True),
        # Valid and pinning nothing: the split builder reads it, replays no
        # pin, and splits fresh, so the card says the dataset pins none.
        ("pins_empty_object", None, '{"schema_version": 1, "pins": {}}', 0, None, False),
        ("pins_all_unknown", None, '{"pins": {"a.jpg": "validation"}}', 0, None, True),
        ("pins_one_unknown", None,
         '{"pins": {"a.jpg": "valid", "b.jpg": "validation"}}', 0, None, True),
        ("pins_whitespace_split", None, '{"pins": {"a.jpg": " test "}}', 0, None, True),
        ("pins_non_string_split", None, '{"pins": {"a.jpg": 3, "b.jpg": "train"}}',
         0, None, True),
        ("pins_over_cap", None,
         '{"pins": {"a.jpg": "train"}, "pad": "' + over_cap_pad + '"}', 0, None, True),
        ("pins_sixty_thousand", None, many, 0,
         {"train": 40000, "valid": 0, "test": 20000}, False),
    )
    # Fixtures that are not a sidecar text at all, each asserted on its own
    # below: a folder and a named pipe where a sidecar belongs, bytes that are
    # not UTF-8, a manifest that is a list, and a manifest that does not parse.
    extras = ("names_is_a_folder", "names_is_binary", "names_is_a_fifo",
              "pins_is_a_fifo", "manifest_is_a_list", "manifest_not_json")
    try:
        for name, names_csv, pins_json, _n, _counts, _unreadable in cases:
            _stage_sidecar_case(root, name, names_csv, pins_json)
        # A folder where a sidecar file belongs, and a names table that is not
        # text at all: both are read through the same guard.
        os.makedirs(os.path.join(root, "names_is_a_folder", "names.csv"), exist_ok=True)
        _stage_sidecar_case(root, "names_is_a_folder", None, None)
        binary = _stage_sidecar_case(root, "names_is_binary", None, None)
        with open(os.path.join(binary, "names.csv"), "wb") as fh:
            fh.write(b"token,name\n\xff\xfe\x00A,Alpha\n")
        # A named pipe where a sidecar belongs. Opening one blocks until a
        # writer arrives, which inside a Flask request thread would hang the
        # whole page, so the page must never open it: the fact this test
        # finishes at all is the assertion. os.mkfifo is POSIX only, which
        # this box is.
        for name, sidecar in (("names_is_a_fifo", "names.csv"),
                              ("pins_is_a_fifo", "split_pins.json")):
            case_dir = _stage_sidecar_case(root, name, None, None)
            os.mkfifo(os.path.join(case_dir, sidecar))
        # A manifest that parses but is not an object, and one that does not
        # parse at all: both name none of the keys the card reads, and the
        # listing still has to answer.
        odd = os.path.join(root, "manifest_is_a_list")
        write(os.path.join(odd, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
        write(os.path.join(odd, "ingest_manifest.json"), "[1, 2]")
        broken = os.path.join(root, "manifest_not_json")
        write(os.path.join(broken, "data.yaml"), "path: .\nnames:\n  0: A\nnc: 1\n")
        write(os.path.join(broken, "ingest_manifest.json"), "{not json")
        listed = {x["name"]: x for x in bpmod.list_datasets(root)}
        for name, _csv, _pins, n_names, counts, unreadable in cases:
            got = listed[name]
            check(got["n_names"] == n_names and got["has_names"] is (n_names > 0),
                  f"{name}: names table reads {n_names} "
                  f"(got {got['n_names']}, has_names {got['has_names']})")
            check(got["split_counts"] == counts
                  and got["has_split_pins"] is (counts is not None),
                  f"{name}: split counts read {counts} (got {got['split_counts']})")
            check(got["split_pins_unreadable"] is unreadable,
                  f"{name}: an unreadable pins file reads {unreadable} "
                  f"(got {got['split_pins_unreadable']})")
        for name in ("names_is_a_folder", "names_is_binary", "names_is_a_fifo"):
            got = listed[name]
            check(got["n_names"] == 0 and got["has_names"] is False,
                  f"{name}: an unreadable names table reads as none "
                  f"(got {got['n_names']})")
        fifo_pins = listed["pins_is_a_fifo"]
        check(fifo_pins["split_counts"] is None
              and fifo_pins["has_split_pins"] is False
              and fifo_pins["split_pins_unreadable"] is False,
              f"a named pipe where the pins file belongs reads as absent, the "
              f"same as the split builder reads it (got "
              f"{fifo_pins['split_counts']}, "
              f"{fifo_pins['split_pins_unreadable']})")
        for name in ("manifest_is_a_list", "manifest_not_json"):
            odd_entry = listed[name]
            check(odd_entry["n_images"] == 0 and odd_entry["citation"] == ""
                  and odd_entry["license_note"] == ""
                  and odd_entry["source_kind"] == ""
                  and odd_entry["n_background"] is None,
                  f"{name}: a manifest the page cannot read still lists its "
                  f"folder (got {odd_entry['n_images']}, "
                  f"{odd_entry['citation']!r}, {odd_entry['source_kind']!r})")
        check(len(listed) == len(cases) + len(extras),
              f"every damaged dataset is still listed (got {len(listed)}, "
              f"wanted {len(cases) + len(extras)})")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    check(not os.path.exists(root), "the sidecar fixtures are removed")


def main():
    src = tempfile.mkdtemp(prefix="boxes_routes_src_")
    bbox_root = tempfile.mkdtemp(prefix="boxes_routes_bbox_")
    seg_root = tempfile.mkdtemp(prefix="boxes_routes_seg_")
    scripts_dir = tempfile.mkdtemp(prefix="boxes_routes_scripts_")
    inprocess = tempfile.mkdtemp(prefix="boxes_routes_inprocess_")
    try:
        build_source(src)
        dataset = os.path.join(bbox_root, "coral_intel")
        ing.ingest(src, dataset, log=lambda m: None)
        # A half-staged folder beside it: offered nowhere, refused everywhere.
        os.makedirs(os.path.join(bbox_root, "halfway"), exist_ok=True)
        write(os.path.join(scripts_dir, "_boxes", "ingest_coral_intel.py"), "")
        write(os.path.join(scripts_dir, "_boxes", "bbox_to_seg.py"), "")
        py = os.path.join(scripts_dir, "python")
        write(py, "")

        source_path = [src]

        app = Flask("boxes_routes_test")
        app.register_blueprint(
            bpmod.make_blueprint(
                lambda: {"source_path": source_path[0], "bbox_root": bbox_root,
                         "seg_root": seg_root, "scripts_dir": scripts_dir,
                         "python": py, "inprocess_root": inprocess},
                log_fn=lambda m: None),
            url_prefix="/boxes")
        c = app.test_client()

        # ---- page ---------------------------------------------------------
        r = c.get("/boxes/")
        html = r.get_data(as_text=True)
        check(r.status_code == 200, f"GET /boxes/ is 200 (got {r.status_code})")
        for marker in ("bx-sec-datasets", "bx-sec-project", "bx-sec-convert",
                       "bx-sec-progress", "bx-sec-outputs"):
            check(marker in html, f"page carries the {marker} section marker")
        check(html.index("bx-sec-datasets") < html.index("bx-sec-project")
              < html.index("bx-sec-convert")
              < html.index("bx-sec-progress") < html.index("bx-sec-outputs"),
              "the five sections appear in the pinned DOM order")
        for gone in ("bx-sec-train", "bx-sec-trainprogress", "bx-sec-detectors",
                     "bx-train-", "bx-detectors", "BOXES_MODELS"):
            check(gone not in html, f"the quick trainer is gone from the page ({gone})")
        check("20260915-review-fixes" in html,
              "assets carry the cache-bust string of the review-fix build")
        for stale in ("20260904-train-in-vega", "20260915-honest-cards"):
            check(stale not in html,
                  f"the older cache-bust string {stale} is gone, so a browser "
                  f"holding the page open fetches the new script and stylesheet")
        check("never appear in the label coverage matrix" in html,
              "the page carries the coverage-matrix note")
        check(_RETIRED_MOUNT not in html, "the page has no reference to the retired mount")
        check("Train + Infer" not in html and "Train and Infer" not in html,
              "the page never names the retired Train + Infer page")
        check("UVI" not in html, "the page never carries the old institution string")
        check(not _BANNED_WORD_RE.search(html), "page copy has no form of the banned word")
        check("—" not in html and "–" not in html, "page copy has no em or en dashes")
        check("Lauren Olinger" in html, "the page footer credits Lauren Olinger")
        check("<title>Bounding Boxes - VEGA</title>" in html, "the page title carries the VEGA wordmark")
        check(html.count("title=") >= 10,
              f"every control is tooltipped (got {html.count('title=')} title attributes)")
        for cid in ("bx-datasets", "bx-ingest", "bx-refresh", "bx-convert-dataset",
                    "bx-convert-limit", "bx-convert", "bx-status", "bx-stop",
                    "bx-log", "bx-outputs-body"):
            check(f'id="{cid}"' in html, f"page has the {cid} control")
        check('value="200"' in html, "the conversion limit defaults to 200")
        # The staged-datasets tooltip promises one action, and Refresh is the
        # control that performs it: a tooltip that named a refresh with no
        # control to press sent the reader hunting for a button that was not
        # on the page. No tooltip points at the screen ("here", "above").
        ds_tip = tag_for(html, "bx-datasets")
        check("scripts/_boxes/ingest_yolo_boxes.py" in ds_tip
              and "when Refresh is pressed" in ds_tip,
              "the staged-datasets tooltip names the command-line stager and "
              "the control that reads the folder again")
        check("appears here" not in html and "shown here" not in html,
              "no tooltip points at the screen with the word here")
        check("title=" in tag_for(html, "bx-refresh"), "bx-refresh carries a tooltip")
        check('id="bx-ingest-row"' in html and 'id="bx-list-row"' in html,
              "the Coral Intel action row and the list row carry the ids the "
              "page script moves the first one by")

        # ---- the Train in VEGA section ----------------------------------------
        for cid in ("bx-project-dataset", "bx-project-by", "bx-project-existing",
                    "bx-project-result", "bx-project-link", "bx-project-new",
                    "bx-project-open"):
            check(f'id="{cid}"' in html, f"page has the {cid} control")
            check("title=" in tag_for(html, cid), f"{cid} carries a tooltip")
        check(">Train in VEGA<" in html, "the handoff button is labelled Train in VEGA")
        check('target="reefpointseg_orchestrator"' in tag_for(html, "bx-project-link"),
              "the fallback link opens in the one VEGA window")
        check("hidden" in tag_for(html, "bx-project-new"),
              "the second-project button is hidden until a project exists")
        check("Advanced: convert boxes to masks" in html,
              "the advanced summary names only the conversion")
        for tip_word in ("Pick the staged", "Type one to four letters", "Tick this to also",
                         "Set how many records", "Name the output folder"):
            check(tip_word in html, f"a tooltip opens with an imperative ({tip_word})")
        check("You " not in html.replace("Your initials", ""),
              "no tooltip opens with You")

        # ---- datasets -------------------------------------------------------
        r = c.get("/boxes/api/datasets")
        body = r.get_json()
        check(r.status_code == 200 and body["ok"] is True,
              f"/api/datasets 200 ok (got {r.status_code})")
        check([d["name"] for d in body["datasets"]] == ["coral_intel"],
              f"only fully staged datasets are listed (got {body['datasets']})")
        check("models_url" not in body, "the datasets answer no longer carries a models_url")
        card = body["datasets"][0]
        check(card["n_images"] == 4 and card["n_boxes"] == 6 and card["n_degenerate"] == 2,
              f"the card carries the staged counts (got {card})")
        check(card["n_classes"] == 3 and card["ingested_at"],
              f"the card carries the class count and the ingest stamp (got {card})")
        check(card["citation"] == ing.CITATION,
              "the citation line is rendered verbatim from the ingest module")
        check(card["license_note"] == ing.LICENSE_NOTE,
              "the license note is rendered verbatim from the ingest module")
        check(body["source"]["present"] is True
              and body["source"]["name"] == os.path.basename(src),
              f"the source answers its folder name and is reported present "
              f"(got {body['source']})")
        # The page renders the folder name alone, so the path is never sent.
        # The real one runs through a personal cloud-drive tree and carries
        # the old institution string, which no VICARIUS answer may show.
        check("path" not in body["source"],
              f"the source answer carries no path (got {body['source']})")
        check(src not in json.dumps(body),
              "no absolute source path reaches the browser anywhere in the answer")
        check("source_path" not in card,
              f"a card carries no path to the folder it was staged from "
              f"(got {sorted(card)})")
        # Terms and a citation belong to a dataset, never to the page: a
        # page-level pair held the Coral Intel paper, was read by nothing, and
        # was the obvious field for a later reader to credit that paper from
        # on a dataset with no connection to it.
        for gone in ("citation", "license_note", "coverage_note", "bbox_root",
                     "seg_root"):
            check(gone not in body,
                  f"the datasets answer carries no page-level {gone}")

        source_path[0] = os.path.join(src, "gone")
        body = c.get("/boxes/api/datasets").get_json()
        check(body["source"]["present"] is False,
              "a missing source folder is reported so the button can be disabled")

        check_second_dataset_card_facts(c, bbox_root)
        check_citation_fallback_is_coral_intel_only()
        check_damaged_sidecars_never_break_the_listing()
        check_page_constants_match_their_writers()

        # ---- handoff: training projects for a dataset ---------------------------
        r = c.get("/boxes/api/training-projects?dataset=coral_intel")
        body = r.get_json()
        check(r.status_code == 200 and body["ok"] is True and body["projects"] == [],
              f"a dataset with no project lists none (got {r.status_code} {body})")
        huge = "a" * 10000
        for bad, why in (("", "a blank name"), ("../coral_intel", "a traversing name"),
                         (dataset, "a full path"), ("halfway", "a half-staged folder"),
                         (huge, "a ten-thousand-character name"),
                         ("coral_intel/../../etc", "a name with separators"),
                         ("coral_intel\x00", "a name with a NUL byte")):
            r = c.get("/boxes/api/training-projects", query_string={"dataset": bad})
            err = (r.get_json() or {}).get("error") or ""
            check(r.status_code == 400 and err, f"training-projects refuses {why} (got {r.status_code})")
            check(len(err) < 200, f"the refusal for {why} does not echo the whole name")
            if bbox_root not in bad:
                check(bbox_root not in err, f"the refusal for {why} reveals no staged path")

        # The first press mints; the answer is the URL the main window honours.
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "LO"})
        body = r.get_json() or {}
        check(r.status_code == 200 and body.get("reused") is False,
              f"the first press mints a project (got {r.status_code} {body})")
        run_dir = body.get("run_dir") or ""
        check(os.path.dirname(run_dir) == os.path.abspath(inprocess)
              and os.path.isfile(os.path.join(run_dir, "project.json")),
              "the minted project sits directly under the projects folder with a project.json, "
              "which is what the main window's auto-open guard requires")
        check(body.get("open_url") == "/?project_dir=" + quote(run_dir, safe=""),
              f"the answer carries the main-window URL (got {body.get('open_url')!r})")
        check(body.get("n_existing") == 0, f"the first press found no earlier project (got {body})")

        # The second press reuses: nothing minted, same folder answered.
        before = run_dirs(inprocess)
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "LO"})
        body = r.get_json() or {}
        check(r.status_code == 200 and body.get("reused") is True and body.get("run_dir") == run_dir,
              f"a second press answers the existing project (got {r.status_code} {body})")
        check(run_dirs(inprocess) == before, "the second press minted nothing")
        check(body.get("open_url") == "/?project_dir=" + quote(run_dir, safe=""),
              "the reused answer carries the same main-window URL")
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel"})
        check(r.status_code == 200 and (r.get_json() or {}).get("reused") is True,
              f"opening an existing project needs no initials (got {r.status_code})")

        # Neighbours that must not count: a composite project, a box project
        # on another dataset, an unreadable project.json, a run dir with none.
        write(os.path.join(inprocess, "run_26-09-04_LO_other", "project.json"),
              json.dumps({"project_type": "composite_training", "name": "TRAIN_other"}))
        write(os.path.join(inprocess, "run_26-09-04_LO_train_elsewhere", "project.json"),
              json.dumps({"project_type": "bbox_training", "name": "TRAIN_elsewhere",
                          "training_source": {"kind": "bbox", "path": seg_root}}))
        write(os.path.join(inprocess, "run_26-09-04_LO_junk", "project.json"), "{not json")
        write(os.path.join(inprocess, "run_26-09-04_LO_list", "project.json"), "[1, 2]")
        os.makedirs(os.path.join(inprocess, "run_26-09-04_LO_empty"), exist_ok=True)
        write(os.path.join(inprocess, "run_26-09-04_LO_badsource", "project.json"),
              json.dumps({"project_type": "bbox_training", "training_source": "not a dict"}))
        # A planted source path with a NUL byte cannot be resolved; it is
        # skipped, and neither route may die on it.
        write(os.path.join(inprocess, "run_26-09-04_LO_nul", "project.json"),
              json.dumps({"project_type": "bbox_training", "name": "TRAIN_nul",
                          "training_source": {"kind": "bbox", "path": dataset + "\x00x"}}))
        # A run_* link pointing outside the projects folder matches the
        # dataset and would pass the main window's unresolved-path guard, so
        # it must be skipped; a link to the real project inside the folder
        # is the same project and must not list twice.
        outside = os.path.join(seg_root, "planted_outside")
        write(os.path.join(outside, "project.json"),
              json.dumps({"project_type": "bbox_training", "name": "TRAIN_outside",
                          "created_at": "2099-01-01T00:00:00-04:00",
                          "training_source": {"kind": "bbox", "path": dataset}}))
        os.symlink(outside, os.path.join(inprocess, "run_26-09-04_LO_link_out"))
        os.symlink(run_dir, os.path.join(inprocess, "run_26-09-04_LO_alias"))
        r = c.get("/boxes/api/training-projects?dataset=coral_intel")
        body = r.get_json() or {}
        names = [p["run_dir_name"] for p in body.get("projects") or []]
        check(r.status_code == 200 and names == [os.path.basename(run_dir)],
              f"only the real box project on this dataset is listed (got {r.status_code} {names})")
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel", "by": "LO"})
        check(r.status_code == 200 and (r.get_json() or {}).get("run_dir") == run_dir,
              f"the handoff survives the planted neighbours (got {r.status_code})")
        proj = body["projects"][0]
        check(proj["open_url"] == "/?project_dir=" + quote(run_dir, safe="")
              and proj["display_name"] == "TRAIN_coral_intel" and proj["initials"] == "LO"
              and proj["created_at"],
              f"a listed project carries its URL, name, initials and stamp (got {proj})")
        check(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", proj["created_at"] or ""),
              f"the stamp is AST 24-hour (got {proj['created_at']!r})")

        # Asked for a second project, the handoff mints one and lists it first.
        r = c.post("/boxes/api/create-project",
                   json={"dataset": "coral_intel", "by": "LO", "reuse": False})
        body = r.get_json() or {}
        check(r.status_code == 200 and body.get("reused") is False
              and body.get("run_dir_name") == os.path.basename(run_dir) + "_2",
              f"reuse false mints a second project (got {r.status_code} {body})")
        check(body.get("n_existing") == 1, f"the mint reports how many existed (got {body})")
        listed = c.get("/boxes/api/training-projects?dataset=coral_intel").get_json()["projects"]
        check([p["run_dir_name"] for p in listed]
              == [os.path.basename(run_dir) + "_2", os.path.basename(run_dir)],
              f"two projects list newest first (got {[p['run_dir_name'] for p in listed]})")

        # Ordering is by the parsed stamp, then the numeric suffix: _10 sits
        # above _3 on a same-second tie, never under it as a string would.
        # A stamp that does not parse falls back to the folder's own time,
        # shows as blank, and never jumps ahead of a real stamp.
        base = os.path.basename(run_dir)
        with open(os.path.join(body["run_dir"], "project.json")) as fh:
            second_meta = json.load(fh)
        for suffix in ("_10", "_3"):
            planted = dict(second_meta, name="TRAIN_coral_intel" + suffix)
            write(os.path.join(inprocess, base + suffix, "project.json"), json.dumps(planted))
        day = 86400
        for name, stamp, age in (("run_26-09-04_LO_nodate", "not a date", 2 * day),
                                 ("run_26-09-04_LO_objdate", {"a": 1}, 1 * day)):
            folder = os.path.join(inprocess, name)
            write(os.path.join(folder, "project.json"),
                  json.dumps({"project_type": "bbox_training", "name": "TRAIN_" + name,
                              "created_at": stamp,
                              "training_source": {"kind": "bbox", "path": dataset}}))
            os.utime(folder, (time.time() - age, time.time() - age))
        listed = c.get("/boxes/api/training-projects?dataset=coral_intel").get_json()["projects"]
        order = [p["run_dir_name"] for p in listed]
        check(order == [base + "_10", base + "_3", base + "_2", base,
                        "run_26-09-04_LO_objdate", "run_26-09-04_LO_nodate"],
              f"stamp first, numeric suffix on a tie, unparseable stamps last (got {order})")
        check([p["created_at"] for p in listed[-2:]] == ["", ""],
              f"an unparseable stamp shows blank (got {[p['created_at'] for p in listed[-2:]]})")
        r = c.post("/boxes/api/create-project", json={"dataset": "coral_intel"})
        check((r.get_json() or {}).get("run_dir_name") == base + "_10",
              f"reuse opens the newest by that order (got {(r.get_json() or {}).get('run_dir_name')})")
        r = c.post("/boxes/api/create-project",
                   json={"dataset": "coral_intel", "by": "LO", "reuse": "yes"})
        check(r.status_code == 400, f"a non-boolean reuse is refused (got {r.status_code})")
        check(not _LAUNCHES, "the handoff launched no process")

        # ---- ingest refused with no source -----------------------------------
        r = c.post("/boxes/api/ingest", json={})
        check(r.status_code == 400 and "not on this machine" in r.get_json()["error"],
              f"ingest with no source folder is refused (got {r.get_json()})")
        check(src not in r.get_json()["error"]
              and "TCRMP_CORAL_INTEL_SOURCE" in r.get_json()["error"],
              "the refusal names the way out without echoing the source path")
        check(not _LAUNCHES, "the refused ingest launched nothing")
        source_path[0] = src

        # ---- ingest happy path -------------------------------------------------
        del _LAUNCHES[:]
        r = c.post("/boxes/api/ingest", json={})
        body = r.get_json()
        check(r.status_code == 200 and body["ok"] is True,
              f"ingest launches (got {r.status_code} {body})")
        cmd = _LAUNCHES[0].cmd
        check(cmd[0] == py and cmd[1] == os.path.join(scripts_dir, "_boxes",
                                                      "ingest_coral_intel.py"),
              f"ingest drives the ingest script with the module interpreter (got {cmd[:2]})")
        check(cmd[cmd.index("--source") + 1] == src
              and cmd[cmd.index("--dest") + 1] == dataset,
              f"ingest passes the source and the staged destination (got {cmd})")
        check("source" not in body and src not in json.dumps(body),
              f"the ingest answer echoes no source path back to the browser "
              f"(got {sorted(body)})")

        # ---- busy guard ---------------------------------------------------------
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5})
        check(r.status_code == 409, f"a convert while the ingest runs is 409 (got {r.status_code})")
        r = c.post("/boxes/api/ingest", json={})
        check(r.status_code == 409, "a second ingest while one runs is 409 too")

        # ---- status --------------------------------------------------------
        st = c.get("/boxes/api/status").get_json()
        check(st["phase"] == "running" and st["kind"] == "ingest" and st["started_at"],
              f"status reports the running job (got {st})")
        check(isinstance(st["log_tail"], list) and len(st["log_tail"]) <= 12,
              f"status tail is at most 12 lines (got {len(st['log_tail'])})")
        # The log pane is filled from the status answer's log_tail. A separate
        # log route was here and no control on the page reached it, which is a
        # live address serving nothing; it is gone.
        r = c.get("/boxes/api/log?lines=1")
        check(r.status_code == 404,
              f"there is no separate log route for the page to leave unused "
              f"(got {r.status_code})")
        _LAUNCHES[0].returncode = 4
        st = c.get("/boxes/api/status").get_json()
        check(st["phase"] == "error" and st["exit_code"] == 4,
              f"a dead process reconciles to error with its exit code (got {st})")

        # ---- convert validation -----------------------------------------------------
        r = c.post("/boxes/api/convert", json={"dataset": os.path.join(bbox_root, "halfway"),
                                               "limit": 5})
        check(r.status_code == 400 and "dataset" in r.get_json()["error"].lower(),
              f"convert refuses a dataset that is not staged (got {r.get_json()})")
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 0})
        check(r.status_code == 400, "convert refuses a limit below one")
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5,
                                               "name": "../evil"})
        check(r.status_code == 400, "convert refuses a traversal output name")

        # ---- convert happy path --------------------------------------------------------
        del _LAUNCHES[:]
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5,
                                               "name": "smoke5"})
        body = r.get_json()
        check(r.status_code == 200 and body["ok"] is True,
              f"convert launches (got {r.status_code} {body})")
        cmd = _LAUNCHES[0].cmd
        check(cmd[1] == os.path.join(scripts_dir, "_boxes", "bbox_to_seg.py"),
              f"convert drives bbox_to_seg.py (got {cmd[:2]})")
        check(cmd[cmd.index("--dataset") + 1] == dataset
              and cmd[cmd.index("--limit") + 1] == "5",
              f"convert passes the dataset and the limit (got {cmd})")
        check(body["out_dir"] == os.path.join(seg_root, "smoke5"),
              f"convert writes under the bbox output root (got {body['out_dir']})")
        check("--keep-degenerate" not in cmd,
              "whole-frame boxes are skipped unless asked for")
        check("run_" not in os.path.relpath(body["out_dir"], seg_root),
              "the output folder is never a pipeline project folder")

        # ---- stop ----------------------------------------------------------------------
        del _SIGNALLED[:]
        r = c.post("/boxes/api/stop")
        check(r.status_code == 200 and _SIGNALLED == [(_IMPOSSIBLE_PID, signal.SIGTERM)],
              f"stop signals our own process group and nothing else (got {_SIGNALLED})")
        check(c.get("/boxes/api/status").get_json()["phase"] == "idle",
              "a deliberate stop reports idle, not a crash")

        # A pid whose group is not ours is never signalled: fall back to
        # killing the process we hold, never a group we did not start.
        _PGID_OF["mode"] = "foreign"
        del _LAUNCHES[:]
        del _SIGNALLED[:]
        c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5,
                                           "name": "foreign_group"})
        c.post("/boxes/api/stop")
        check(_SIGNALLED == [] and _LAUNCHES[0].killed,
              f"a foreign process group is never signalled (got {_SIGNALLED})")
        _PGID_OF["mode"] = "own"

        # ---- keep degenerate -------------------------------------------------------------
        del _LAUNCHES[:]
        r = c.post("/boxes/api/convert", json={"dataset": dataset, "limit": 5,
                                               "keep_degenerate": True})
        check(r.status_code == 200 and "--keep-degenerate" in _LAUNCHES[0].cmd,
              "keep_degenerate adds the flag")
        _LAUNCHES[0].returncode = 0
        c.get("/boxes/api/status")

        # ---- seg outputs -------------------------------------------------------------------
        r = c.get("/boxes/api/seg_outputs")
        check(r.status_code == 200 and r.get_json()["outputs"] == [],
              "no converted datasets yet")
        out = os.path.join(seg_root, "coral_intel_smoke5")
        os.makedirs(os.path.join(out, "all_labels"), exist_ok=True)
        write(os.path.join(out, "all_labels", "a.txt"), "0 0.1 0.1 0.2 0.2 0.3 0.1\n")
        write(os.path.join(out, "manifest.csv"),
              "basename,class_id,class_name,box_x_min,box_y_min,box_x_max,box_y_max,"
              "polygon_points,sam3_score,source_type,status\n"
              "a.jpg,0,Alpha-beta,1,1,9,9,12,0.83,bbox_sam3,ok\n")
        write(os.path.join(out, "README.md"), "# Boxes converted to masks\n")
        body = c.get("/boxes/api/seg_outputs").get_json()
        check([o["name"] for o in body["outputs"]] == ["coral_intel_smoke5"],
              f"a converted dataset is listed (got {body['outputs']})")
        o = body["outputs"][0]
        check(o["n_labels"] == 1 and o["n_polygons"] == 1 and o["path"] == out,
              f"the listing counts label files and ledger rows (got {o})")
        check(o["built_at"], "the listing carries a timestamp")

        # ---- a dataset outside the staged root is refused --------------------------------------
        r = c.post("/boxes/api/convert", json={"dataset": seg_root, "limit": 5})
        check(r.status_code == 400,
              "convert refuses a dataset path outside supporting_data/bbox_datasets")

        # ---- page script ------------------------------------------------------------------------
        js = c.get("/boxes/static/boxes.js").get_data(as_text=True)
        check(_RETIRED_MOUNT not in js, "the page script never reaches the retired mount")
        check("Train + Infer" not in js and "Train and Infer" not in js,
              "the page script never names the retired page")
        for gone in ("mlApi", "mlPoll", "MODELS", "loadDetectors", "startDetectorTraining"):
            check(gone not in js, f"the quick trainer is gone from the page script ({gone})")
        check("'/api/training-projects?dataset='" in js,
              "the page script asks this blueprint which projects exist for the dataset")
        check("'/api/create-project'" in js, "the page script hands off through create-project")
        check("body.open_url" in js and "reefpointseg_orchestrator" in js,
              "the page script opens the answered URL in the one VEGA window")
        check("reuse: reuse" in js and "trainInVega(false)" in js,
              "the second-project button asks for a fresh mint")
        check("TCRMP_CORAL_INTEL_SOURCE" in js,
              "the page script names the environment override when the source is missing")
        check("var polling = false" in js and "if (polling) return" in js
              and "polling = false; });" in js,
              "the dataset status poll has an in-flight guard that clears when the read settles")
        for tip in (
                "VICARIUS shows which tool wrote this dataset: the Coral Intel stager, the generic YOLO box stager, or the subset tool that narrows another staged dataset to the classes carrying at least so many boxes.",
                "This dataset's manifest names no stager, so VICARIUS cannot say which one wrote it.",
                "VICARIUS counts the species names the dataset carries for its class codes, which the evaluation report and the inference overlay show beside each code.",
                "VICARIUS shows how many images the dataset pins to each split, in the order train, valid, test, so a training project and an imported model score on the same held-out images.",
                "VICARIUS could not read the split pins in this dataset and cannot say which images are held out.",
                "Check split_pins.json in the dataset folder before training, because a pins file the split builder cannot read stops a training project.",
                "Classes counts the species names derived from the filenames and verified against every annotation file.",
                "Classes counts the class codes this dataset declares in its data.yaml, which VICARIUS checks against every annotation file before staging.",
                "Classes counts the class codes this dataset declares, as the stager that wrote it recorded them.",
                "Whole frame counts the boxes that cover the whole picture, so they say only that the species is somewhere in it.",
                "Background counts the staged images whose label file holds no box, so they show the detector what a frame with nothing to find looks like.",
                "VICARIUS finds no split pins in this dataset, so a training project built from it draws its own train, valid and test split, and two projects on this dataset hold out different images.",
                "VICARIUS shows no citation recorded for this dataset. Record one by staging it again with the citation argument.",
                "VICARIUS shows no license terms recorded for this dataset. Record them by staging it again with the license argument."):
            check(tip in js, f"the card carries the tooltip {tip[:46]!r}")
        # Every tooltip on a card opens with the actor, never with the fact.
        for actorless in ("'No citation is recorded", "'No license terms are recorded"):
            check(actorless not in js,
                  f"no card tooltip opens with an actor-less sentence ({actorless})")
        check("' VICARIUS drops them from the labels_localized folder inside the dataset and skips them when converting to masks.'" in js,
              "the labels_localized sentence is on the card that carries that folder")
        check("' VICARIUS skips them when converting to masks.'" in js,
              "a dataset with no labels_localized folder is promised none")
        check("'none recorded'" in js,
              "a fact the dataset does not carry reads none recorded on the card")
        check("'unknown'" not in js and "'none'" not in js,
              "one phrase covers every absent fact, so terms, a citation, a "
              "names table and a background count no longer read as three "
              "different states")
        check("'unreadable'" in js,
              "a sidecar that is there and cannot be read keeps its own "
              "phrase, because the operator has to act on it")
        check("'Coral Intel'" in js and "'YOLO'" in js
              and "'Class-count subset'" in js,
              "the source stat shows each of the three writers by name, never "
              "a bare token")
        check("splitPins" in js and "' / '" in js,
              "the split-pin stat shows the three per-split counts")
        check("text: ' ' + k, dim: true" in js,
              "each pin count carries the name of its split, so the cell says "
              "which images train and which are held out")
        check("'grid-column: span 2'" in js,
              "the stager name holds two grid columns, so it does not break "
              "after Coral and drop its label out of the row")
        check("'grid-column: 1 / -1'" in js,
              "the split pins hold the whole row, which the three counts and "
              "their split names need at 946 px")
        check("'scripts/_boxes/ingest_yolo_boxes.py'" in js,
              "a card staged from the command line names the stager that "
              "writes it again, with the path it sits at")
        check("'scripts/_boxes/subset_by_class_count.py'" in js,
              "a subset card names the tool that makes it again, with the "
              "path it sits at: the YOLO stager run over a subset folder "
              "would reproduce neither the class-count rule nor class_map.csv")
        check("function restageText" in js and "function restageTip" in js
              and "SUBSET_KIND" in js,
              "the restage line and its tooltip branch on which tool wrote "
              "the dataset, so neither names the wrong one")
        check("'Made again by running '" in js
              and "'Staged again by running '" in js,
              "a subset and a staged dataset each get their own restage line")
        check("VICARIUS wrote this dataset from the command line: the tool at "
              in js,
              "the subset restage tooltip names the tool and what it reads")
        check("function placeIngestRow" in js and "is-in-card" in js,
              "the Coral Intel action row is moved into the Coral Intel card, "
              "so each card carries the action that restages it")
        check("function refreshDatasets" in js
              and "elRefresh.addEventListener('click', refreshDatasets)" in js,
              "Refresh reads the staged datasets again")
        check("function hasText" not in js,
              "the card second-guesses no server text: an empty citation or "
              "license is the one signal that nothing is recorded")
        check(not _BANNED_WORD_RE.search(js), "page script copy has no form of the banned word")
        check("—" not in js and "–" not in js,
              "page script copy has no em or en dashes")
        css = c.get("/boxes/static/boxes.css").get_data(as_text=True)
        check("bx-btn-row" not in css and "[hidden]" in css and ".bx-foot" in css,
              "the stylesheet drops the row-button rule and gains the hidden and footer rules")
        cards_rule = css.split(".bx-cards {", 1)[-1].split("}", 1)[0]
        check("align-items: start" in cards_rule,
              f"the dataset cards size to their own content, so a card with "
              f"fewer facts does not end in a third of blank surface (got "
              f"{cards_rule.strip()!r})")
        for rule in (".bx-stat-sub", ".bx-actions.is-in-card", ".bx-actions-end",
                     ".bx-ds-restage"):
            check(rule in css, f"the stylesheet carries the {rule} rule")
    finally:
        subprocess.Popen = _REAL_POPEN
        subprocess.run = _REAL_RUN
        os.getpgid = _REAL_GETPGID
        os.killpg = _REAL_KILLPG
        for d in (src, bbox_root, seg_root, scripts_dir, inprocess):
            shutil.rmtree(d, ignore_errors=True)
    check(not any(os.path.exists(d) for d in (src, bbox_root, seg_root, scripts_dir, inprocess)),
          "temp fixtures removed")

    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
