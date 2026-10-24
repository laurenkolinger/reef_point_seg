"""No-re-screening at Step 3 (2026-09-27 VEGA annotator tidy-up plan):
--prior-screening drops a frame already screened, in another project, for
every one of this project's target labels, and never touches a frame
screened for only some of them.

Run: env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_prior_screening_exclusion.py

Covers:
  load_prior_screening   missing path (silent: no --prior-screening flag is
                         normal, not an error), a given path that cannot be
                         read or does not parse (each logs one [WARN] line to
                         stderr naming the path and the reason), a non-dict
                         document (same), a non-list value under a key, and
                         the valid case.
  fully_screened_frame_ids
                         full match drops the frame id; a partial match, an
                         empty species_list, or an empty prior_screening
                         mapping all drop nothing.
  build_selection_summary
                         omitting skipped_fully_screened/partly_screened
                         (the default, no --prior-screening) produces the
                         exact same key set as before this feature existed
                         (byte-identical behaviour without the flag); passing
                         them adds exactly those two keys, shaped as the
                         contract describes.
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

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import select_images as si  # noqa: E402

_R = []


def check(c, m):
    if not c:
        raise AssertionError(m)


def run(fn):
    try:
        fn()
        _R.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:
        _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


def _binary():
    rows = [
        {"frame_id": "f_full", "year_int": 2022, "date": "2022-06-01",
         "site": "FLC", "transect": 3, "frame": 7, "PA": 1, "SS": 1},
        {"frame_id": "f_partial", "year_int": 2022, "date": "2022-06-01",
         "site": "FLC", "transect": 3, "frame": 8, "PA": 1, "SS": 1},
        {"frame_id": "f_none", "year_int": 2022, "date": "2022-06-01",
         "site": "FLC", "transect": 3, "frame": 9, "PA": 1, "SS": 1},
    ]
    return pd.DataFrame(rows).set_index("frame_id")


def _basename(binary, fid):
    r = binary.loc[fid]
    return si.build_image_filename(str(r["date"]), r["site"], int(r["transect"]), int(r["frame"]))


# ── load_prior_screening ──────────────────────────────────────────────────

def _call_capturing_stderr(fn, *args):
    """Run fn(*args), returning (result, stderr_text)."""
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        result = fn(*args)
    return result, buf.getvalue()


def test_load_prior_screening_no_flag_is_silent():
    """No path at all (the --prior-screening flag was never given) is the
    normal, expected case: {} with no warning at all."""
    for blank in (None, ""):
        result, err = _call_capturing_stderr(si.load_prior_screening, blank)
        check(result == {}, f"blank path -> {{}}: {result!r}")
        check(err == "", f"no --prior-screening flag must never warn: {err!r}")


def test_load_prior_screening_nonexistent_path_warns_and_returns_empty():
    """A path WAS given (the flag was used) but nothing is there: this is a
    real problem (the orchestrator's write presumably failed or the path is
    wrong), so it must be visible on stderr, naming the path."""
    path = "/no/such/file.json"
    result, err = _call_capturing_stderr(si.load_prior_screening, path)
    check(result == {}, "nonexistent path -> {}, never raises")
    check("[WARN]" in err, f"a given-but-missing path must warn: {err!r}")
    check(path in err, f"the warning must name the path: {err!r}")


def test_load_prior_screening_malformed_json_warns_and_returns_empty():
    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "bad.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        result, err = _call_capturing_stderr(si.load_prior_screening, path)
        check(result == {}, "malformed JSON -> {}, never raises")
        check("[WARN]" in err, f"malformed JSON must warn: {err!r}")
        check(path in err, f"the warning must name the path: {err!r}")


def test_load_prior_screening_non_dict_document_warns_and_returns_empty():
    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "list.json")
        with open(path, "w") as f:
            json.dump(["a", "b"], f)
        result, err = _call_capturing_stderr(si.load_prior_screening, path)
        check(result == {}, "a JSON array is not a valid mapping -> {}")
        check("[WARN]" in err, f"a non-object top level must warn: {err!r}")
        check(path in err, f"the warning must name the path: {err!r}")


def test_load_prior_screening_drops_non_list_values():
    """A valid, readable, well-shaped file with one odd inner value is not a
    read/parse error, so it must stay silent (no warning)."""
    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "mixed.json")
        with open(path, "w") as f:
            json.dump({"stem_a": ["PA", "SS"], "stem_b": "not a list", "stem_c": None}, f)
        got, err = _call_capturing_stderr(si.load_prior_screening, path)
        check(got == {"stem_a": ["PA", "SS"]},
              f"a non-list value under a key must be dropped, not raise: {got}")
        check(err == "", f"a readable, well-shaped file must never warn: {err!r}")


def test_load_prior_screening_valid_file():
    with tempfile.TemporaryDirectory() as t:
        path = os.path.join(t, "prior.json")
        with open(path, "w") as f:
            json.dump({"TCRMP20220601_clip_FLC_T307": ["PA"]}, f)
        got, err = _call_capturing_stderr(si.load_prior_screening, path)
        check(got == {"TCRMP20220601_clip_FLC_T307": ["PA"]}, got)
        check(err == "", f"a valid file must never warn: {err!r}")


# ── fully_screened_frame_ids ──────────────────────────────────────────────

def test_fully_screened_frame_dropped():
    binary = _binary()
    prior = {
        _basename(binary, "f_full"): ["PA", "SS"],
        _basename(binary, "f_partial"): ["PA"],
    }
    drop = si.fully_screened_frame_ids(binary, si.build_image_filename, prior, ["PA", "SS"])
    check(drop == {"f_full"}, f"only the fully-screened frame drops: {drop}")


def test_partially_screened_frame_never_dropped():
    binary = _binary()
    prior = {_basename(binary, "f_partial"): ["PA"]}
    drop = si.fully_screened_frame_ids(binary, si.build_image_filename, prior, ["PA", "SS"])
    check(drop == set(), f"a partial match must never be dropped: {drop}")


def test_empty_species_list_drops_nothing():
    binary = _binary()
    prior = {_basename(binary, fid): ["PA", "SS"] for fid in binary.index}
    drop = si.fully_screened_frame_ids(binary, si.build_image_filename, prior, [])
    check(drop == set(), f"no target labels -> nothing to screen against: {drop}")


def test_empty_prior_screening_drops_nothing_byte_identical_to_no_flag():
    """The exact scenario a run WITHOUT --prior-screening hits: prior_screening
    is {} (load_prior_screening's own default for a missing path), so this
    must return set() no matter how many labels every frame carries -
    proving the gate is a true no-op absent the flag."""
    binary = _binary()
    drop = si.fully_screened_frame_ids(binary, si.build_image_filename, {}, ["PA", "SS"])
    check(drop == set(), f"empty prior_screening must drop nothing: {drop}")


def test_frame_absent_from_prior_screening_never_dropped():
    binary = _binary()
    prior = {"some_other_stem_entirely": ["PA", "SS"]}
    drop = si.fully_screened_frame_ids(binary, si.build_image_filename, prior, ["PA", "SS"])
    check(drop == set(), f"a frame prior_screening never mentions is never dropped: {drop}")


# ── build_selection_summary: byte-identical without the flag ─────────────

def _base_kwargs():
    return dict(
        species_list=["PA"], achieved={"PA": 40}, target=40,
        per_site={"CRB": 20, "BPT": 20}, pool_totals={"PA": 500},
        alloc_totals={"PA": 40}, cap_frac=0.25, n_selected=40)


def test_summary_omits_new_keys_when_not_given():
    doc = si.build_selection_summary(**_base_kwargs())
    check("skipped_fully_screened" not in doc,
          f"no --prior-screening -> no skipped_fully_screened key: {doc}")
    check("partly_screened" not in doc,
          f"no --prior-screening -> no partly_screened key: {doc}")
    check(set(doc.keys()) == {"per_site", "per_species", "cap_frac", "cap_ok",
                              "cap_satisfiable", "n_selected", "shortfalls"},
          f"key set must match the pre-existing contract exactly: {sorted(doc.keys())}")


def test_summary_carries_new_keys_when_given():
    doc = si.build_selection_summary(
        **_base_kwargs(), skipped_fully_screened=12,
        partly_screened=["TCRMP20220601_clip_FLC_T307", "TCRMP20220601_clip_FLC_T308"])
    check(doc["skipped_fully_screened"] == 12, doc.get("skipped_fully_screened"))
    check(doc["partly_screened"] == {
        "count": 2,
        "stems": ["TCRMP20220601_clip_FLC_T307", "TCRMP20220601_clip_FLC_T308"],
    }, doc.get("partly_screened"))


def test_summary_skipped_fully_screened_zero_is_still_written():
    """A run WITH --prior-screening that happens to drop nothing must still
    report 0, not omit the key: the panel line needs to tell 0 apart from
    'the flag was never used'."""
    doc = si.build_selection_summary(**_base_kwargs(), skipped_fully_screened=0,
                                     partly_screened=[])
    check(doc["skipped_fully_screened"] == 0, doc)
    check(doc["partly_screened"] == {"count": 0, "stems": []}, doc)


TESTS = [
    test_load_prior_screening_no_flag_is_silent,
    test_load_prior_screening_nonexistent_path_warns_and_returns_empty,
    test_load_prior_screening_malformed_json_warns_and_returns_empty,
    test_load_prior_screening_non_dict_document_warns_and_returns_empty,
    test_load_prior_screening_drops_non_list_values,
    test_load_prior_screening_valid_file,
    test_fully_screened_frame_dropped,
    test_partially_screened_frame_never_dropped,
    test_empty_species_list_drops_nothing,
    test_empty_prior_screening_drops_nothing_byte_identical_to_no_flag,
    test_frame_absent_from_prior_screening_never_dropped,
    test_summary_omits_new_keys_when_not_given,
    test_summary_carries_new_keys_when_given,
    test_summary_skipped_fully_screened_zero_is_still_written,
]


if __name__ == "__main__":
    for fn in TESTS:
        run(fn)
    ok = sum(1 for _, p, _ in _R if p)
    print(f"\n{ok}/{len(_R)} passed")
    if ok != len(_R):
        for name, p, err in _R:
            if not p:
                print(f"\n--- {name} ---\n{err}")
    sys.exit(0 if ok == len(_R) else 1)
