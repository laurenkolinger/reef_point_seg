"""T1: the Refine loop's held-out guard in seed_from_predictions.py.

No-pytest harness: run with env/bin/python <this file>.

The guard decides which frames a model-assisted review round may seed. A frame
the training split assigned to validation or test must never reach a reviewer,
because a reviewer's accepted mask is exported back into the training set and
the held-out evaluation stops being held out.

Covers:
  detection manifest  - a pin table written as "pins" (detect_dataset.py) is
                        honored. Until 2026-09-16 only "pinned" was read, so
                        every detection project seeded its own test split.
  segmentation manifest - a pin table written as "pinned" (pinned_split.py) is
                        honored, unchanged.
  transect manifest   - holdout_mode "transect" still excludes by transect
                        number, with an empty pin table.
  fail closed         - a manifest naming no image and no transect holdout is
                        refused, and so are a corrupt one, a manifest that is
                        not UTF-8 text at all, a missing one, one that is not a
                        JSON object, and one whose pins carry a split the
                        splitters never write.
  predictions file    - a missing, half-written, non-UTF-8 or wrongly shaped
                        predictions.json is refused with a sentence naming it,
                        never an unhandled exception mid-round.
  box-only round      - a predictions file whose every detection is a bounding
                        box is refused, because a detection model's output
                        cannot become a reviewable mask and the round would
                        otherwise report success with nothing on any frame.
  geometry counters   - on a mixed round, a detection carrying a box and no
                        outline, and one carrying neither, are counted in the
                        summary instead of being dropped in silence.

The seeder runs as a subprocess, exactly as the orchestrator route runs it, so
this suite reaches no port, no project and no global state.
"""
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
PY = os.path.join(_REPO, "env", "bin", "python")
SEEDER = os.path.join(_REPO, "scripts", "TCRMPclip_combinedAnnotate", "src",
                      "seed_from_predictions.py")

# Exit status seed_from_predictions.py uses for every refusal.
EXIT_REFUSED = 2
# Fixture frame size, and a triangle big enough to survive mask_ops' 500-pixel
# minimum fragment.
FRAME_W, FRAME_H = 200, 100
TRIANGLE_XYN = [0.05, 0.1, 0.95, 0.1, 0.5, 0.9]
# Pin tables in the huge-manifest case. 50,000 entries is well past any real
# project (the fish dataset pins about 7,500).
HUGE_PIN_COUNT = 50000

_RESULTS = []


def check(cond, msg):
    """Raise AssertionError with ``msg`` when ``cond`` is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one test function, record its outcome, print a per-test line."""
    import traceback
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── fixtures ────────────────────────────────────────────────────────────

def _make_frame(path):
    """Write one small JPEG with a light triangle on a dark ground."""
    im = Image.new("RGB", (FRAME_W, FRAME_H), (10, 20, 30))
    ImageDraw.Draw(im).polygon([(10, 10), (190, 10), (100, 90)], fill=(200, 200, 200))
    im.save(path, "JPEG", quality=90)


def _polygon_det(confidence=0.9, polygon=None):
    """One predictions.json detection carrying an outline."""
    return {"class_id": 0, "class": "AA", "confidence": confidence,
            "polygon_xyn": list(TRIANGLE_XYN if polygon is None else polygon)}


def _box_det(confidence=0.9):
    """One predictions.json detection carrying box corners and no outline,
    which is what step 8 writes for a detection (bounding-box) model."""
    return {"class_id": 0, "class": "AA", "confidence": confidence,
            "box_xyxyn": [0.1, 0.1, 0.6, 0.8]}


def _no_geometry_det(confidence=0.9):
    """One detection carrying neither an outline nor box corners."""
    return {"class_id": 0, "class": "AA", "confidence": confidence}


def build_fixture(root, manifest_doc, frames, write_manifest=True,
                  manifest_text=None, manifest_bytes=None):
    """Build an export dir, raw frames, a codes CSV, a manifest and predictions.

    Parameters:
        root: the temp dir everything is written under.
        manifest_doc: the split_manifest.json document, or None with
            manifest_text to write something that is not a JSON object.
        frames: {filename: [detection, ...]} for predictions.json, in order.
        write_manifest: False leaves the manifest path pointing at no file.
        manifest_text: raw text to write instead of json.dump(manifest_doc).
        manifest_bytes: raw bytes to write instead of any text, for the file
            that is not UTF-8 at all.

    Returns:
        dict of the paths the seeder and the assertions need.
    """
    export_dir = os.path.join(root, "export")
    seg_dir = os.path.join(export_dir, "segmentations")
    raw_src_dir = os.path.join(root, "raw_src")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(raw_src_dir, exist_ok=True)

    for name in frames:
        _make_frame(os.path.join(raw_src_dir, name))

    seg_path = os.path.join(seg_dir, "segmentations.json")
    with open(seg_path, "w") as f:
        json.dump({}, f)

    codes_csv = os.path.join(root, "master_codes.csv")
    with open(codes_csv, "w") as f:
        f.write("code,category,name\nAA,Coral,Agaricia agaricites\n")

    split_manifest = os.path.join(root, "split_manifest.json")
    if write_manifest:
        if manifest_bytes is not None:
            with open(split_manifest, "wb") as f:
                f.write(manifest_bytes)
        else:
            with open(split_manifest, "w") as f:
                if manifest_text is not None:
                    f.write(manifest_text)
                else:
                    json.dump(manifest_doc, f)

    predictions_path = os.path.join(root, "predictions.json")
    with open(predictions_path, "w") as f:
        json.dump({
            "schema_version": 2,
            "generated_at": "2026-09-16T09:00:00-04:00",
            "class_names": {"0": "AA"},
            "items": [{"filename": name,
                       "raw": os.path.join(raw_src_dir, name),
                       "width": FRAME_W, "height": FRAME_H,
                       "detections": dets}
                      for name, dets in frames.items()],
        }, f)

    return {"export_dir": export_dir, "seg_path": seg_path, "codes_csv": codes_csv,
            "split_manifest": split_manifest, "predictions_path": predictions_path}


def run_seeder(fx, extra_args=None):
    """Invoke the seeder the way the orchestrator route does."""
    cmd = [PY, SEEDER,
           "--predictions", fx["predictions_path"],
           "--export_dir", fx["export_dir"],
           "--codes_csv", fx["codes_csv"],
           "--split_manifest", fx["split_manifest"]]
    if extra_args:
        cmd += extra_args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def summary_of(result):
    """The seeder's one-line JSON summary, parsed from the last stdout line."""
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    check(lines, f"seeder printed no summary (stderr: {result.stderr})")
    return json.loads(lines[-1])


def seeded_names(fx):
    """Frame names present in the export's segmentations.json."""
    with open(fx["seg_path"]) as f:
        return set(json.load(f))


# Manifest shapes, named once so every test reads the same way.
def detection_manifest(pins):
    """A split_manifest.json as detect_dataset.build_detect_dataset writes it."""
    return {"schema_version": 2, "holdout_mode": "lab-random",
            "pins_source": "fresh", "seed": "fixture",
            "valid_frac": 0.1, "test_frac": 0.1,
            "split_fingerprint": "0123456789abcdef",
            "split_fingerprint_scheme": "sha256-16/netstring-utf8-sorted-v1",
            "counts": {"train": 1, "valid": 1, "test": 1},
            "pins": pins}


def segmentation_manifest(pinned, holdout_mode="pinned-random"):
    """A split_manifest.json as pinned_split.build_manifest writes it."""
    return {"schema_version": 2, "created_at": "2026-09-16 09:00:00",
            "holdout_mode": holdout_mode,
            "transect_rule": "stem _T + 3 digits, first digit = transect",
            "holdout_transects": {"val": [5], "test": [6]},
            "seed": "1", "valid_frac": 0.2, "test_frac": 0.1,
            "split_fingerprint": "fedcba9876543210",
            "split_fingerprint_scheme": "sha256-16/netstring-utf8-sorted-v1",
            "counts": {"train": 1, "valid": 1, "test": 1},
            "pinned": pinned}


# ── detection manifest: the leak this task closes ───────────────────────

TRAIN_FRAME = "TCRMP20240101_clip_AAA_T118.jpeg"
VALID_FRAME = "TCRMP20240101_clip_AAA_T119.jpeg"
TEST_FRAME = "TCRMP20240101_clip_AAA_T120.jpeg"


def test_detection_pins_table_holds_out_valid_and_test():
    """A "pins" table keeps its valid and test frames out of the round."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest(
            {TRAIN_FRAME: "train", VALID_FRAME: "valid", TEST_FRAME: "test"}),
            {TRAIN_FRAME: [_polygon_det()], VALID_FRAME: [_polygon_det()],
             TEST_FRAME: [_polygon_det()]})
        result = run_seeder(fx)
        check(result.returncode == 0, f"seeder exited {result.returncode}: {result.stderr}")
        s = summary_of(result)
        check(s["skipped_holdout"] == 2,
              f"both held-out frames skipped (got {s['skipped_holdout']})")
        check(s["seeded_frames"] == 1, f"only the train frame seeded (got {s['seeded_frames']})")
        names = seeded_names(fx)
        check(VALID_FRAME not in names, "the validation frame never reached segmentations.json")
        check(TEST_FRAME not in names, "the test frame never reached segmentations.json")
        check(TRAIN_FRAME in names, "the train frame did reach segmentations.json")


def test_detection_pins_table_seeds_nothing_it_pins_to_train_only():
    """Every pin reading train leaves every frame seedable."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest(
            {TRAIN_FRAME: "train", VALID_FRAME: "train"}),
            {TRAIN_FRAME: [_polygon_det()], VALID_FRAME: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 0, f"nothing held out (got {s['skipped_holdout']})")
        check(s["seeded_frames"] == 2, f"both frames seeded (got {s['seeded_frames']})")


def test_both_pin_tables_together_hold_out_on_either_spelling():
    """A manifest carrying both spellings is read as one table, and the
    held-out answer wins wherever the two disagree."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        doc = detection_manifest({TRAIN_FRAME: "train", TEST_FRAME: "train"})
        doc["pinned"] = {TEST_FRAME: "test"}
        fx = build_fixture(root, doc,
                           {TRAIN_FRAME: [_polygon_det()], TEST_FRAME: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 1, f"the disputed frame is held out (got {s['skipped_holdout']})")
        check(TEST_FRAME not in seeded_names(fx), "the disputed frame was not seeded")


# ── segmentation manifest: unchanged behaviour ──────────────────────────

def test_segmentation_pinned_table_holds_out_valid_and_test():
    """A "pinned" table keeps working exactly as before."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, segmentation_manifest(
            {TRAIN_FRAME: "train", VALID_FRAME: "valid", TEST_FRAME: "test"}),
            {TRAIN_FRAME: [_polygon_det()], VALID_FRAME: [_polygon_det()],
             TEST_FRAME: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 2,
              f"both held-out frames skipped (got {s['skipped_holdout']})")
        check(seeded_names(fx) == {TRAIN_FRAME},
              f"only the train frame is in the export (got {sorted(seeded_names(fx))})")


def test_transect_holdout_mode_excludes_by_transect_number():
    """holdout_mode "transect" with an empty pin table still excludes T5 and T6."""
    t1, t5, t6 = ("TCRMP20240101_clip_AAA_T118.jpeg",
                  "TCRMP20240101_clip_AAA_T501.jpeg",
                  "TCRMP20240101_clip_AAA_T601.jpeg")
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, segmentation_manifest({}, holdout_mode="transect"),
                           {t1: [_polygon_det()], t5: [_polygon_det()], t6: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 2,
              f"the validation and test transects are skipped (got {s['skipped_holdout']})")
        check(seeded_names(fx) == {t1}, f"only transect 1 seeded (got {sorted(seeded_names(fx))})")


def test_transect_numbers_written_as_whole_floats_are_still_held_out():
    """A JSON writer that renders 5 as 5.0 must not shrink the holdout. Until
    2026-09-16 a float was skipped outright, so a manifest with pins plus
    float transects passed its validation and then seeded transects 5 and 6."""
    t1, t5, t6 = ("TCRMP20240101_clip_AAA_T118.jpeg",
                  "TCRMP20240101_clip_AAA_T501.jpeg",
                  "TCRMP20240101_clip_AAA_T601.jpeg")
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        doc = segmentation_manifest({}, holdout_mode="transect")
        doc["holdout_transects"] = {"val": [5.0], "test": ["6"]}
        fx = build_fixture(root, doc,
                           {t1: [_polygon_det()], t5: [_polygon_det()], t6: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 2,
              f"the float and the digit string both hold out (got {s['skipped_holdout']})")
        check(seeded_names(fx) == {t1}, f"only transect 1 seeded (got {sorted(seeded_names(fx))})")


# ── fail closed ─────────────────────────────────────────────────────────

def _refusal(fx, result, must_mention_path=True):
    """Assert one refusal: exit status, stderr, and an untouched export."""
    check(result.returncode == EXIT_REFUSED,
          f"refusal exits {EXIT_REFUSED} (got {result.returncode}; stderr: {result.stderr})")
    check("ERROR" in result.stderr, f"refusal says ERROR on stderr (got {result.stderr!r})")
    if must_mention_path:
        check(fx["split_manifest"] in result.stderr,
              f"refusal names the manifest path (got {result.stderr!r})")
    check("Traceback" not in result.stderr,
          f"refusal is a message, not a traceback (got {result.stderr!r})")
    check(seeded_names(fx) == set(), "a refused run seeds nothing")


def test_manifest_with_neither_key_and_no_holdout_mode_is_refused():
    """The shape at the heart of the defect: nothing says what is held out."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, {"schema_version": 2, "seed": "fixture",
                                  "counts": {"train": 1, "valid": 0, "test": 0}},
                           {TRAIN_FRAME: [_polygon_det()]})
        result = run_seeder(fx)
        _refusal(fx, result)
        check("holds nothing out" in result.stderr,
              f"refusal says the manifest holds nothing out (got {result.stderr!r})")
        check("training step" in result.stderr,
              f"refusal says what to do next (got {result.stderr!r})")


def test_empty_pins_table_is_refused():
    """An empty table names no image, so it can hold no image out."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({}), {TRAIN_FRAME: [_polygon_det()]})
        _refusal(fx, run_seeder(fx))


def test_empty_pinned_table_without_transect_mode_is_refused():
    """The segmentation spelling of the same empty table is refused too."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, segmentation_manifest({}, holdout_mode="adopted-random"),
                           {TRAIN_FRAME: [_polygon_det()]})
        _refusal(fx, run_seeder(fx))


def test_missing_manifest_file_is_refused():
    """A manifest path pointing at nothing refuses before any frame is read."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, None, {TRAIN_FRAME: [_polygon_det()]}, write_manifest=False)
        _refusal(fx, run_seeder(fx))


def test_corrupt_json_manifest_is_refused_with_a_message():
    """A truncated manifest refuses with a message, not a stack trace."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, None, {TRAIN_FRAME: [_polygon_det()]},
                           manifest_text='{"pins": {"a.jpeg": "train"')
        result = run_seeder(fx)
        _refusal(fx, result)
        check("not valid JSON" in result.stderr,
              f"refusal names the JSON problem (got {result.stderr!r})")


def test_manifest_that_is_not_utf8_is_refused_with_a_message():
    """A manifest holding binary bytes fails in the codec, inside the read,
    before the JSON parser runs. UnicodeDecodeError is not a JSONDecodeError,
    so until 2026-09-16 this shape escaped both except clauses and reached the
    orchestrator as a stack trace with exit 1."""
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\xfd"
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, None, {TRAIN_FRAME: [_polygon_det()]},
                           manifest_bytes=png_header)
        result = run_seeder(fx)
        _refusal(fx, result)
        check("not valid JSON" in result.stderr,
              f"refusal names the JSON problem (got {result.stderr!r})")
        check("training step" in result.stderr,
              f"refusal says what to do next (got {result.stderr!r})")


def test_manifest_that_is_not_an_object_is_refused():
    """A JSON array where the split record belongs is refused."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, None, {TRAIN_FRAME: [_polygon_det()]},
                           manifest_text='["pins", "pinned"]')
        result = run_seeder(fx)
        _refusal(fx, result)
        check("JSON object" in result.stderr,
              f"refusal says a JSON object was expected (got {result.stderr!r})")


def test_pins_table_that_is_not_an_object_is_refused():
    """"pins" holding a string names no image and is refused."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, None, {TRAIN_FRAME: [_polygon_det()]},
                           manifest_text='{"pins": "train"}')
        result = run_seeder(fx)
        _refusal(fx, result)
        check("object of image basename" in result.stderr,
              f"refusal says what a pin table must be (got {result.stderr!r})")


def test_non_string_pin_value_is_refused():
    """A pin whose split is a number could be hiding a holdout, so it refuses."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train", TEST_FRAME: 3}),
                           {TRAIN_FRAME: [_polygon_det()], TEST_FRAME: [_polygon_det()]})
        result = run_seeder(fx)
        _refusal(fx, result)
        check("outside" in result.stderr,
              f"refusal says the split is outside the three names (got {result.stderr!r})")


def test_pin_value_that_is_a_list_is_refused():
    """An unhashable pin value is a refusal, never a crash."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TEST_FRAME: ["test"]}),
                           {TEST_FRAME: [_polygon_det()]})
        _refusal(fx, run_seeder(fx))


def test_huge_manifest_is_read_and_honored():
    """A 50,000-pin manifest still excludes the frame it holds out."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        pins = {f"filler_{i:06d}.jpeg": "train" for i in range(HUGE_PIN_COUNT)}
        pins[TRAIN_FRAME] = "train"
        pins[TEST_FRAME] = "test"
        fx = build_fixture(root, detection_manifest(pins),
                           {TRAIN_FRAME: [_polygon_det()], TEST_FRAME: [_polygon_det()]})
        size_mb = os.path.getsize(fx["split_manifest"]) / (1024 * 1024)
        s = summary_of(run_seeder(fx))
        check(s["skipped_holdout"] == 1,
              f"the held-out frame is still excluded in a {size_mb:.1f} MB manifest "
              f"(got {s['skipped_holdout']})")
        check(seeded_names(fx) == {TRAIN_FRAME},
              f"only the train frame seeded (got {sorted(seeded_names(fx))})")


def test_missing_predictions_file_is_refused_with_a_message():
    """A --predictions path pointing at nothing is the operator's mistake most
    likely to reach this tool, and it refuses with a sentence, not an
    unhandled FileNotFoundError."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det()]})
        os.remove(fx["predictions_path"])
        result = run_seeder(fx)
        _refusal(fx, result, must_mention_path=False)
        check(fx["predictions_path"] in result.stderr,
              f"refusal names the predictions path (got {result.stderr!r})")
        check("inference step" in result.stderr,
              f"refusal says what to do next (got {result.stderr!r})")


def test_predictions_file_that_is_not_json_is_refused():
    """A half-written predictions file refuses the same way the manifest
    does, including one that is not UTF-8 text at all."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det()]})
        for payload in (b'{"items": [', b"\x89PNG\r\n\x1a\n\xff\xfe\xfd"):
            with open(fx["predictions_path"], "wb") as f:
                f.write(payload)
            result = run_seeder(fx)
            _refusal(fx, result, must_mention_path=False)
            check("not valid JSON" in result.stderr,
                  f"refusal names the JSON problem (got {result.stderr!r})")


def test_predictions_items_of_the_wrong_shape_are_refused():
    """items holding anything but per-frame objects is refused before the
    seeding loop reaches it, so a hand-edited file is a message and not an
    AttributeError halfway through a round."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det()]})
        for payload, phrase in ((["a", "b"], "JSON object"),
                                ({"items": "nope"}, "must be a list"),
                                ({"items": [{"filename": TRAIN_FRAME}, 7]},
                                 "per-frame objects")):
            with open(fx["predictions_path"], "w") as f:
                json.dump(payload, f)
            result = run_seeder(fx)
            _refusal(fx, result, must_mention_path=False)
            check(phrase in result.stderr,
                  f"refusal explains the shape (got {result.stderr!r})")
        check(seeded_names(fx) == set(), "no refused shape seeded anything")


# ── box-only round: the refusal an operator can read ────────────────────

def test_all_box_only_predictions_are_refused():
    """A detection (bounding-box) model's predictions carry no outline, so the
    round would open the review tool with no mask on any frame. Until
    2026-09-16 it exited 0 and the operator read a green "Seeded N frame(s), 0
    mask(s) for review." with nothing said, because the Refine route shows the
    seeder's stderr only on a non-zero exit."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_box_det(), _box_det()]})
        result = run_seeder(fx)
        _refusal(fx, result, must_mention_path=False)
        check(fx["predictions_path"] in result.stderr,
              f"refusal names the predictions file (got {result.stderr!r})")
        check("segmentation model" in result.stderr,
              f"refusal says what to do next (got {result.stderr!r})")
        check(not os.path.exists(os.path.join(fx["export_dir"], "routed_input")),
              "a refused round symlinks no frame into the export")


def test_box_only_frame_before_a_polygon_frame_still_seeds():
    """One outline anywhere in the file is a segmentation model's output, so
    the round runs even when the first frame read carries only boxes."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        other = "TCRMP20240101_clip_AAA_T117.jpeg"
        fx = build_fixture(root, detection_manifest(
            {TRAIN_FRAME: "train", other: "train"}),
            {TRAIN_FRAME: [_box_det()], other: [_polygon_det()]})
        result = run_seeder(fx)
        check(result.returncode == 0, f"seeder exited {result.returncode}: {result.stderr}")
        s = summary_of(result)
        check(s["seeded_masks"] == 1, f"the outline was seeded (got {s['seeded_masks']})")
        check(s["skipped_box_only"] == 1, f"the box was counted (got {s['skipped_box_only']})")


# ── geometry counters ───────────────────────────────────────────────────

def test_box_only_and_polygon_on_one_frame_are_counted_apart():
    """A mixed frame seeds its outline and counts its box."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det(), _box_det()]})
        result = run_seeder(fx)
        s = summary_of(result)
        check(s["seeded_masks"] == 1, f"the outline was seeded (got {s['seeded_masks']})")
        check(s["skipped_box_only"] == 1, f"the box was counted (got {s['skipped_box_only']})")
        check(fx["predictions_path"] in result.stderr,
              f"the warning names the file to open (got {result.stderr!r})")


def test_detection_with_no_geometry_is_counted_separately():
    """Neither outline nor box is its own count, not a box-only count."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_no_geometry_det()]})
        result = run_seeder(fx)
        s = summary_of(result)
        check(s["skipped_no_geometry"] == 1,
              f"the empty detection is counted (got {s.get('skipped_no_geometry')})")
        check(s["skipped_box_only"] == 0,
              f"it is not counted as box-only (got {s['skipped_box_only']})")
        check("neither an outline nor a" in result.stderr,
              f"the run warns about it on stderr (got {result.stderr!r})")


def test_malformed_polygons_are_counted_and_never_crash():
    """A truncated outline, an odd coordinate count and a text coordinate are
    all counted rather than raising or seeding a garbage mask."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [
                               _polygon_det(polygon=[0.1, 0.1, 0.2, 0.2]),
                               _polygon_det(polygon=[0.1, 0.1, 0.2, 0.2, 0.3]),
                               _polygon_det(polygon=["a", "b", "c", "d", "e", "f"]),
                               _polygon_det(polygon="abcdef"),
                           ]})
        result = run_seeder(fx)
        check(result.returncode == 0, f"seeder exited {result.returncode}: {result.stderr}")
        s = summary_of(result)
        check(s["seeded_masks"] == 0, f"no mask from a malformed outline (got {s['seeded_masks']})")
        check(s["skipped_no_geometry"] == 4,
              f"all four malformed detections counted (got {s['skipped_no_geometry']})")


def test_summary_carries_both_new_counters_on_an_ordinary_round():
    """The counters are always present, so a reader never has to guess."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det()]})
        s = summary_of(run_seeder(fx))
        check(s["skipped_box_only"] == 0, "skipped_box_only is reported as 0")
        check(s["skipped_no_geometry"] == 0, "skipped_no_geometry is reported as 0")
        check(s["seeded_masks"] == 1, f"the ordinary round still seeds (got {s['seeded_masks']})")


# ── CLI surface ─────────────────────────────────────────────────────────

def test_retired_transect_flags_are_gone():
    """--val_transects and --test_transects were declared, documented as
    unused, and read by nothing; the holdout has only ever come from
    --split_manifest. They are removed, and nothing in the module passes them."""
    with tempfile.TemporaryDirectory(prefix="seed_holdout_guard_") as root:
        fx = build_fixture(root, detection_manifest({TRAIN_FRAME: "train"}),
                           {TRAIN_FRAME: [_polygon_det()]})
        for flag in ("--val_transects", "--test_transects"):
            result = run_seeder(fx, extra_args=[flag, "5"])
            check(result.returncode != 0, f"{flag} is not accepted (exited 0)")
            check("unrecognized arguments" in result.stderr,
                  f"{flag} is reported as unknown (got {result.stderr!r})")
        check(seeded_names(fx) == set(), "a rejected command line seeds nothing")


def test_split_manifest_help_is_plain_language():
    """The help a student reads carries no internal shorthand."""
    result = subprocess.run([PY, SEEDER, "--help"], capture_output=True, text=True, timeout=120)
    check(result.returncode == 0, f"--help exited {result.returncode}: {result.stderr}")
    check("split_manifest.json" in result.stdout, "the help names the file to point at")
    check("D8" not in result.stdout, f"the help spells out its terms (got {result.stdout!r})")


if __name__ == "__main__":
    print("T1 -- Refine loop held-out guard (seed_from_predictions.py)\n")
    if not os.path.isfile(PY):
        print(f"FATAL: env python not found at {PY}")
        sys.exit(2)
    if not os.path.isfile(SEEDER):
        print(f"FATAL: seeder not found at {SEEDER}")
        sys.exit(2)

    print("DETECTION MANIFEST (pin table 'pins'):")
    run(test_detection_pins_table_holds_out_valid_and_test)
    run(test_detection_pins_table_seeds_nothing_it_pins_to_train_only)
    run(test_both_pin_tables_together_hold_out_on_either_spelling)
    print("SEGMENTATION MANIFEST (pin table 'pinned'):")
    run(test_segmentation_pinned_table_holds_out_valid_and_test)
    run(test_transect_holdout_mode_excludes_by_transect_number)
    run(test_transect_numbers_written_as_whole_floats_are_still_held_out)
    print("FAIL CLOSED:")
    run(test_manifest_with_neither_key_and_no_holdout_mode_is_refused)
    run(test_empty_pins_table_is_refused)
    run(test_empty_pinned_table_without_transect_mode_is_refused)
    run(test_missing_manifest_file_is_refused)
    run(test_corrupt_json_manifest_is_refused_with_a_message)
    run(test_manifest_that_is_not_utf8_is_refused_with_a_message)
    run(test_manifest_that_is_not_an_object_is_refused)
    run(test_pins_table_that_is_not_an_object_is_refused)
    run(test_non_string_pin_value_is_refused)
    run(test_pin_value_that_is_a_list_is_refused)
    run(test_huge_manifest_is_read_and_honored)
    run(test_missing_predictions_file_is_refused_with_a_message)
    run(test_predictions_file_that_is_not_json_is_refused)
    run(test_predictions_items_of_the_wrong_shape_are_refused)
    print("BOX-ONLY ROUND:")
    run(test_all_box_only_predictions_are_refused)
    run(test_box_only_frame_before_a_polygon_frame_still_seeds)
    print("GEOMETRY COUNTERS:")
    run(test_box_only_and_polygon_on_one_frame_are_counted_apart)
    run(test_detection_with_no_geometry_is_counted_separately)
    run(test_malformed_polygons_are_counted_and_never_crash)
    run(test_summary_carries_both_new_counters_on_an_ordinary_round)
    print("CLI SURFACE:")
    run(test_retired_transect_flags_are_gone)
    run(test_split_manifest_help_is_plain_language)

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed.")
    if failed:
        print("\nFAILURES:")
        for name, _, detail in failed:
            print(f"--- {name} ---\n{detail}")
        sys.exit(1)
    sys.exit(0)
