#!/usr/bin/env python
"""No-pytest harness for the composite export (composite.py): resolution
(preference wins, strongest outcome, latest-at tie), negatives, review-only
skip, unlabeled skip, multi-part join, manifest + data.yaml + class_map,
the labels filter, the name guard and the listing.

Run with:
  env/bin/python scripts/_matrix/tests/test_composite.py
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

import csv
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from datetime import datetime  # noqa: E402

from _matrix import composite as composite_mod  # noqa: E402
from _matrix.composite import (  # noqa: E402
    CompositeExists,
    _at_key,
    _fmt_ast,
    _join_polygons_fallback,
    build_composite,
    join_polygons,
    list_composites,
    resolve,
    sanitize_name,
)
from _matrix.preferences import Preferences  # noqa: E402
from _matrix.tests._fixture_cell import (  # noqa: E402
    RING_A, RING_B, T101, T102, T105, T2015, T403, T404,
    build_root, make_image, make_project, mask, seg_entry, write,
)
from _reefreview.clock import now_ast  # noqa: E402
from _reefreview.mask_registry import MaskRegistry, build_registry_record  # noqa: E402

_FAILURES = []


def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def _stem(name):
    return os.path.splitext(name)[0]


def _label_lines(out_dir, stem):
    p = os.path.join(out_dir, "all_labels", stem + ".txt")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


def _manifest(out_dir):
    with open(os.path.join(out_dir, "composite_manifest.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _raise_eml(*_a, **_k):
    """Stand-in for composite.write_eml that always fails, used to prove a
    write_eml failure never breaks build_composite (spec Task 12)."""
    raise RuntimeError("simulated EML failure")


def main():
    root = tempfile.mkdtemp(prefix="matrix_composite_test_")
    try:
        fx = build_root(root)

        check(join_polygons is not _join_polygons_fallback,
              "join_polygons imported from TCRMPclip_combinedAnnotate/src/export_yolo.py")
        check(sanitize_name("my comp/1!") == "my_comp_1", f"name sanitized to [A-Za-z0-9._-] (got {sanitize_name('my comp/1!')})")
        check(sanitize_name("").startswith("composite_") and len(sanitize_name("")) == len("composite_26-08-26_2200"),
              f"blank name -> composite_<YY-MM-DD_HHMM> (got {sanitize_name('')})")

        # ---- L10: `at` normalization across the three stamp shapes ----
        reg_at = "2026-08-26 18:00:00"
        led_at = "2026-08-26T09:00:00-04:00"
        legacy_at = "2026-08-26T12:30:00"
        check(_at_key(reg_at) == datetime(2026, 8, 26, 18, 0, 0), "registry 'YYYY-MM-DD HH:MM:SS' parses")
        check(_at_key(led_at) == datetime(2026, 8, 26, 9, 0, 0), "ledger ISO with offset parses to naive wall clock")
        check(_at_key(legacy_at) == datetime(2026, 8, 26, 12, 30, 0), "legacy naive ISO parses")
        check(_at_key("") == datetime.min and _at_key(None) == datetime.min and _at_key("yesterday") == datetime.min,
              "blank / None / unparsable -> datetime.min")
        check(max(reg_at, led_at) == led_at, "raw string order would rank the ledger form above the later registry stamp")
        check(max([reg_at, led_at, legacy_at], key=_at_key) == reg_at, "parsed order ranks the true latest stamp first")
        check(_fmt_ast(led_at) == "2026-08-26 09:00:00" and _fmt_ast("2026-08-26T09:00:00+00:00") == "2026-08-26 05:00:00",
              "_fmt_ast converts an aware stamp to AST 'YYYY-MM-DD HH:MM:SS'")
        check(_fmt_ast(reg_at) == reg_at and _fmt_ast(legacy_at) == "2026-08-26 12:30:00" and _fmt_ast("") == ""
              and _fmt_ast("nope") == "", "_fmt_ast keeps the space form, converts naive ISO, blanks the rest")

        # ---- full build over the fixture ----
        res = build_composite(root, "first", None, by="lo")
        out = res["out_dir"]
        check(out == os.path.join(root, "_composite", "first"), f"out_dir under inprocess/_composite (got {out})")
        for f in ("all_images", "all_labels", "data.yaml", "class_map.json", "composite_manifest.csv", "README.md"):
            check(os.path.exists(os.path.join(out, f)), f"composite holds {f}")
        check(res["success"] is True and res["by"] == "LO", "result success + by uppercased")
        check(res["n_images"] == 4, f"n_images: T101, T2015, T403 (negative), T105 (negative) (got {res['n_images']})")
        check(res["n_masks"] == 3, f"n_masks: T101 PA, T101 OFAV, T2015 PA (got {res['n_masks']})")
        check(res["n_negatives"] == 2 and sorted(res["negatives"]) == sorted([_stem(T403), _stem(T105)]),
              f"negatives: reviewed with no exported mask, no pending review (got {res['negatives']})")
        check(res["skipped_review_only"] == 1, f"review-only T102 skipped (got {res['skipped_review_only']})")
        check(res["skipped_unreviewed"] == 1, f"never reviewed T404 skipped (got {res['skipped_unreviewed']})")
        check(res["blocked_unlabeled"] == 1, f"accepted mask without a label never exports (got {res['blocked_unlabeled']})")
        check(res["by_label"] == {"OFAV": 1, "PA": 2}, f"by_label (got {res['by_label']})")
        check(res["source_types"] is None, f"no source_types filter -> null (got {res['source_types']})")
        check(res["by_site"] == {"SCP": 2, "SSJ": 1, "BIT": 1},
              f"L14 (Task 6): per-site image count (SCP: T101+T105, SSJ: T403, BIT: T2015) (got {res['by_site']})")
        with open(os.path.join(out, "composite.json"), encoding="utf-8") as fh:
            meta_first = json.load(fh)
        check(meta_first.get("history") == [], f"a fresh build carries an empty history list (got {meta_first.get('history')})")
        with open(os.path.join(out, "README.md"), encoding="utf-8") as fh:
            readme_first = fh.read()
        check("Source filter: none, every source type is included." in readme_first,
              f"README states the (absent) source filter (got:\n{readme_first})")
        check("## Per-site image counts" in readme_first, "README gains a per-site table heading")
        for site, n in (("SCP", 2), ("SSJ", 1), ("BIT", 1)):
            check(f"| {site} | {n} |" in readme_first,
                  f"README per-site table lists {site}: {n} (got:\n{readme_first})")
        check(_label_lines(out, _stem(T102)) is None and _label_lines(out, _stem(T404)) is None,
              "skipped images get no label file")
        check(_label_lines(out, _stem(T403)) == [] and os.path.islink(os.path.join(out, "all_images", T403)),
              "negative: empty label file + image symlink")
        link = os.readlink(os.path.join(out, "all_images", T101))
        check(link in (fx["img_a"][T101], fx["img_b"][T101]) and os.path.isfile(link),
              f"image symlink points at a contributing project's image file (got {link})")

        with open(os.path.join(out, "class_map.json"), encoding="utf-8") as fh:
            cmap = json.load(fh)
        check(cmap == {"OFAV": 0, "PA": 1}, f"class ids follow sorted label order (got {cmap})")
        with open(os.path.join(out, "data.yaml"), encoding="utf-8") as fh:
            yaml_text = fh.read()
        check(f"path: {out}" in yaml_text and "nc: 2" in yaml_text and "  0: OFAV" in yaml_text and "  1: PA" in yaml_text
              and "train: all_images" in yaml_text, "data.yaml matches the export_yolo layout")

        lines = _label_lines(out, _stem(T101))
        check(len(lines) == 2, f"T101 has one line per exported mask (got {lines})")
        pa_line = [ln for ln in lines if ln.startswith("1 ")]
        ofav_line = [ln for ln in lines if ln.startswith("0 ")]
        check(len(pa_line) == 1 and len(pa_line[0].split()) == 1 + len(RING_A),
              f"PA line: class 1 + single ring (got {pa_line})")
        check(len(ofav_line) == 1 and len(ofav_line[0].split()) > 1 + len(RING_B),
              f"multi-part OFAV mask joined into ONE longer line (got {ofav_line})")

        rows = _manifest(out)
        check([r for r in rows][0].keys() >= {"basename", "label", "project_id", "uid", "status", "source_type",
                                              "reviewer", "chosen_by_preference", "at"},
              "manifest columns")
        pa101 = next(r for r in rows if r["basename"] == _stem(T101) and r["label"] == "PA")
        check(pa101["project_id"] == "projB" and pa101["reviewer"] == "CD" and pa101["chosen_by_preference"] == "",
              f"strongest outcome wins: projB found_expert beats projA found_manual (got {pa101})")
        check(pa101["uid"].startswith("SCP-20140930-T101-x15y25"), f"manifest uid minted with mask_uid (got {pa101['uid']})")
        ofav101 = next(r for r in rows if r["basename"] == _stem(T101) and r["label"] == "OFAV")
        check(ofav101["project_id"] == "projA",
              f"equal strength (found_ai both): latest at wins, projA registry row is newer (got {ofav101['project_id']})")
        check(all(r["status"] == "accepted" for r in rows), "only accepted masks in the manifest")

        # ---- Task 12: eml.xml companion file ----
        eml_path = os.path.join(out, "eml.xml")
        check(os.path.exists(eml_path), "composite holds eml.xml")
        if os.path.exists(eml_path):
            try:
                ET.parse(eml_path)
                eml_parses = True
            except ET.ParseError:
                eml_parses = False
            check(eml_parses, "eml.xml parses as well-formed XML")

        # ---- Task 12: a write_eml failure must never break the composite
        # build. Isolated throwaway root (a second composite in the shared
        # `root` fixture would break the len(listing) == 1 check just below):
        # monkeypatch the module attribute build_composite actually calls
        # (composite_mod.write_eml) to raise, and confirm every normal
        # output still lands with no eml.xml.
        eml_fail_root = tempfile.mkdtemp(prefix="matrix_composite_emlfail_")
        try:
            build_root(eml_fail_root)
            _orig_write_eml = composite_mod.write_eml
            composite_mod.write_eml = _raise_eml
            try:
                res_eml_fail = build_composite(eml_fail_root, "eml_fail", None, by="lo")
            finally:
                composite_mod.write_eml = _orig_write_eml
            out_fail = res_eml_fail["out_dir"]
            check(res_eml_fail["success"] is True,
                  "build_composite still succeeds when write_eml raises")
            for f in ("all_images", "all_labels", "data.yaml", "class_map.json",
                       "composite_manifest.csv", "README.md"):
                check(os.path.exists(os.path.join(out_fail, f)),
                      f"composite still holds {f} when write_eml raises")
            check(not os.path.exists(os.path.join(out_fail, "eml.xml")),
                  "no eml.xml written when write_eml raises")
        finally:
            shutil.rmtree(eml_fail_root, ignore_errors=True)
        check(not os.path.exists(eml_fail_root), "eml-failure fixture removed")

        # ---- 409 on rebuild, listing ----
        try:
            build_composite(root, "first")
            check(False, "rebuilding an existing name raises CompositeExists")
        except CompositeExists:
            check(True, "rebuilding an existing name raises CompositeExists")
        listing = list_composites(root)
        check(len(listing) == 1 and listing[0]["name"] == "first" and listing[0]["n_images"] == 4
              and listing[0]["n_masks"] == 3 and listing[0]["labels"] == ["OFAV", "PA"]
              and listing[0]["path"] == out and listing[0]["created_at"],
              f"list_composites entry shape (got {listing})")
        check(len(listing[0]["created_at"]) == 19 and listing[0]["created_at"][10] == " ",
              f"L13: created_at 'YYYY-MM-DD HH:MM:SS' (got {listing[0]['created_at']!r})")

        # ---- L13: symlink out of _composite skipped; ISO created_at normalized ----
        outside = tempfile.mkdtemp(prefix="matrix_composite_outside_")
        try:
            write(os.path.join(outside, "data.yaml"), "nc: 1\n")
            sneaky = os.path.join(root, "_composite", "sneaky")
            os.symlink(outside, sneaky)
            iso_dir = os.path.join(root, "_composite", "iso_meta")
            write(os.path.join(iso_dir, "data.yaml"), "nc: 1\n")
            write(os.path.join(iso_dir, "composite.json"),
                  json.dumps({"created_at": "2026-08-26T09:00:00-04:00", "n_images": 1, "n_masks": 0, "labels": ["PA"]}))
            names = [c["name"] for c in list_composites(root)]
            check("sneaky" not in names and "iso_meta" in names and "first" in names,
                  f"list_composites skips the symlink that resolves outside _composite (got {names})")
            iso_entry = next(c for c in list_composites(root) if c["name"] == "iso_meta")
            check(iso_entry["created_at"] == "2026-08-26 09:00:00",
                  f"ISO created_at in composite.json listed as AST space form (got {iso_entry['created_at']!r})")
            os.unlink(sneaky)
            shutil.rmtree(iso_dir)
        finally:
            shutil.rmtree(outside, ignore_errors=True)
        check(not os.path.exists(outside) and [c["name"] for c in list_composites(root)] == ["first"],
              "listing fixtures removed")

        # ---- L6: empty label set raises before anything is written ----
        empty_root = tempfile.mkdtemp(prefix="matrix_composite_empty_")
        try:
            try:
                build_composite(empty_root, "void")
                check(False, "empty root: build_composite raises ValueError")
            except ValueError as e:
                check("nothing to export" in str(e), f"empty root: build_composite raises ValueError (got {e})")
            check(not os.path.exists(os.path.join(empty_root, "_composite")), "empty root: no _composite dir created")
        finally:
            shutil.rmtree(empty_root, ignore_errors=True)
        check(not os.path.exists(empty_root), "empty root removed")

        # ---- preference wins ----
        Preferences(root).set(_stem(T101), "OFAV", "projB", [], by="LO")
        res2 = build_composite(root, "second")
        rows2 = _manifest(res2["out_dir"])
        ofav2 = next(r for r in rows2 if r["basename"] == _stem(T101) and r["label"] == "OFAV")
        check(ofav2["project_id"] == "projB" and ofav2["chosen_by_preference"] == "1",
              f"preference row overrides the ladder (got {ofav2})")
        check(res2["n_preferred"] == 1 and res2["preference_unresolved"] == 0, "preference counters")
        lines2 = _label_lines(res2["out_dir"], _stem(T101))
        ofav_line2 = [ln for ln in lines2 if ln.startswith("0 ")]
        check(len(ofav_line2[0].split()) == 1 + len(RING_B), "preferred projB OFAV mask is the single ring")

        # preference narrowed by uid: choose projA's PA mask explicitly
        Preferences(root).set(_stem(T101), "PA", "projA", [fx["uids_a"][(T101, 0)]], by="LO")
        res3 = build_composite(root, "third")
        pa3 = next(r for r in _manifest(res3["out_dir"]) if r["basename"] == _stem(T101) and r["label"] == "PA")
        check(pa3["project_id"] == "projA" and pa3["uid"] == fx["uids_a"][(T101, 0)] and pa3["chosen_by_preference"] == "1",
              f"preference with uids narrows to that mask (got {pa3})")

        # preference pointing at a project with no accepted mask of that label: ladder fallback, counted
        Preferences(root).set(_stem(T101), "PA", "projC", [], by="LO")
        res4 = build_composite(root, "fourth")
        pa4 = next(r for r in _manifest(res4["out_dir"]) if r["basename"] == _stem(T101) and r["label"] == "PA")
        check(pa4["project_id"] == "projB" and res4["preference_unresolved"] == 1,
              f"unresolvable preference falls back to the ladder and is counted (got {pa4}, {res4['preference_unresolved']})")
        Preferences(root).clear(_stem(T101), "PA")
        Preferences(root).clear(_stem(T101), "OFAV")

        # ---- labels filter ----
        res5 = build_composite(root, "pa_only", ["PA"])
        with open(os.path.join(res5["out_dir"], "class_map.json"), encoding="utf-8") as fh:
            cmap5 = json.load(fh)
        check(cmap5 == {"PA": 0} and res5["by_label"] == {"PA": 2}, f"labels filter limits classes (got {cmap5}, {res5['by_label']})")
        check(len(_label_lines(res5["out_dir"], _stem(T101))) == 1, "filtered label lines only")

        # ---- Task 6: source_types filter excludes an accepted auto mask ----
        # T101 OFAV is auto in both projA and projB; T2015 PA (projB, its
        # only candidate) is also auto. A manual-only filter drops OFAV
        # entirely (zero manual candidates anywhere) and flips T2015 from a
        # positive into a negative (reviewed, no manual PA candidate left).
        res_manual = build_composite(root, "manual_only", None, by="lo",
                                      source_types=["manual_click", "manual_box"])
        check(res_manual["by_label"] == {"PA": 1},
              f"manual-only filter excludes the accepted auto OFAV + T2015 PA masks (got {res_manual['by_label']})")
        check(res_manual["n_masks"] == 1 and res_manual["n_images"] == 4 and res_manual["n_negatives"] == 3,
              f"filtered-out T2015 becomes a negative (got n_masks={res_manual['n_masks']} "
              f"n_images={res_manual['n_images']} n_negatives={res_manual['n_negatives']})")
        check(_stem(T2015) in res_manual["negatives"],
              f"T2015's only PA candidate is auto-sourced, excluded, image becomes a negative (got {res_manual['negatives']})")
        check(res_manual["source_types"] == ["manual_box", "manual_click"],
              f"composite.json records the normalized, sorted source_types list (got {res_manual['source_types']})")
        with open(os.path.join(res_manual["out_dir"], "README.md"), encoding="utf-8") as fh:
            readme_manual = fh.read()
        check("Source filter: manual_box, manual_click." in readme_manual,
              f"README states the active source filter (got:\n{readme_manual})")

        res_all = build_composite(root, "all_sources", None, by="lo", source_types=None)
        check(res_all["source_types"] is None and res_all["by_label"].get("OFAV", 0) == 1,
              f"source_types=None keeps every source type, OFAV included again (got {res_all['by_label']})")

        try:
            build_composite(root, "empty_filter", None, by="lo", source_types=[])
            check(False, "an empty source_types list matches no mask -> ValueError")
        except ValueError as e:
            check("nothing to export" in str(e), f"empty source_types list -> nothing to export (got {e})")

        # ---- Task 6: missing/empty source_type defaults to manual_click ----
        blank_root = tempfile.mkdtemp(prefix="matrix_composite_blank_st_")
        try:
            pdir = make_project(blank_root, "run_blank_20260826_bbb1", "blankp", "Blank Project", target_species="PA")
            raw = os.path.join(pdir, "step4test_combinedAnnotate", "routed_input", "raw")
            img = make_image(raw, T101)
            m_blank = mask(0, "PA", sx=1, sy=2)
            m_blank["source_type"] = ""
            seg = {T101: seg_entry(img, [m_blank])}
            write(os.path.join(pdir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
                  json.dumps(seg))

            res_blank_manual = build_composite(blank_root, "blank_manual", None, by="lo",
                                                source_types=["manual_click"])
            check(res_blank_manual["n_masks"] == 1,
                  f"an empty source_type mask is treated as manual_click, kept by a manual_click filter "
                  f"(got {res_blank_manual['n_masks']})")
            with open(os.path.join(res_blank_manual["out_dir"], "composite.json"), encoding="utf-8") as fh:
                meta_blank = json.load(fh)
            check(meta_blank.get("source_type_missing_default") == "manual_click",
                  f"composite.json documents the missing-source_type rule "
                  f"(got {meta_blank.get('source_type_missing_default')})")
            with open(os.path.join(res_blank_manual["out_dir"], "README.md"), encoding="utf-8") as fh:
                readme_blank = fh.read()
            check("treated as manual_click for this filter" in readme_blank,
                  f"README states the missing-source_type rule (got:\n{readme_blank})")

            try:
                build_composite(blank_root, "blank_auto", None, by="lo", source_types=["auto"])
                check(False, "an auto-only filter excludes the empty-source_type mask -> nothing to export")
            except ValueError as e:
                check("nothing to export" in str(e),
                      f"empty-source_type mask NOT matched by an auto-only filter (got {e})")
        finally:
            shutil.rmtree(blank_root, ignore_errors=True)
        check(not os.path.exists(blank_root), "blank-source-type fixture removed")

        # ---- Task 6: rebuild atomically swaps in place; history accumulates ----
        res_v1 = build_composite(root, "rebuild_me", None, by="lo")
        try:
            build_composite(root, "rebuild_me", None, by="lo")
            check(False, "without rebuild, an existing name still raises CompositeExists (409 at the route)")
        except CompositeExists:
            check(True, "without rebuild, an existing name still raises CompositeExists (409 at the route)")

        res_v2 = build_composite(root, "rebuild_me", None, by="ab",
                                  source_types=["manual_click", "manual_box"], rebuild=True)
        check(res_v2["out_dir"] == res_v1["out_dir"], f"rebuild keeps the same out_dir (got {res_v2['out_dir']})")
        check(res_v2["by"] == "AB" and res_v2["source_types"] == ["manual_box", "manual_click"],
              f"rebuild actually rebuilt with the new params (got by={res_v2['by']} source_types={res_v2['source_types']})")
        check(res_v2["by_label"] != res_v1["by_label"],
              f"rebuild content differs under the new source_types filter (v1={res_v1['by_label']}, v2={res_v2['by_label']})")
        with open(os.path.join(res_v2["out_dir"], "composite.json"), encoding="utf-8") as fh:
            meta_v2 = json.load(fh)
        check(isinstance(meta_v2.get("history"), list) and len(meta_v2["history"]) == 1,
              f"composite.json history holds exactly the previous build (got {meta_v2.get('history')})")
        prev_entry = meta_v2["history"][0]
        check(prev_entry.get("by") == "LO" and prev_entry.get("n_masks") == res_v1["n_masks"],
              f"history entry carries the previous build's own metadata (got {prev_entry})")
        check(prev_entry.get("history") in (None, []),
              f"history entries do not nest their own history (got {prev_entry.get('history')})")
        check(not os.path.exists(res_v2["out_dir"] + ".rebuild-tmp")
              and not os.path.exists(res_v2["out_dir"] + ".rebuild-backup"),
              "no leftover tmp/backup swap directories after a rebuild")

        res_v3 = build_composite(root, "rebuild_me", None, by="cd", rebuild=True)
        with open(os.path.join(res_v3["out_dir"], "composite.json"), encoding="utf-8") as fh:
            meta_v3 = json.load(fh)
        check(len(meta_v3["history"]) == 2 and [h.get("by") for h in meta_v3["history"]] == ["LO", "AB"],
              f"a second rebuild appends to the accumulated history, oldest first (got {[h.get('by') for h in meta_v3['history']]})")
        rebuild_names = [c["name"] for c in list_composites(root)]
        check(rebuild_names.count("rebuild_me") == 1,
              f"list_composites shows one entry for a rebuilt name, not a duplicate (got {rebuild_names.count('rebuild_me')})")

        # ---- latest-at tie on a dedicated two-project root ----
        tie_root = tempfile.mkdtemp(prefix="matrix_composite_tie_")
        try:
            for pid, run, at in (("early", "run_early_20260826_e1", "2026-08-10T10:00:00"),
                                 ("late", "run_late_20260826_l1", "2026-08-12T10:00:00")):
                pdir = make_project(tie_root, run, pid, pid, target_species="PA")
                raw = os.path.join(pdir, "step4test_combinedAnnotate", "routed_input", "raw")
                img = make_image(raw, T101)
                seg = {T101: seg_entry(img, [mask(0, "PA", source_type="auto", sx=1, sy=2)], processed_at=at)}
                write(os.path.join(pdir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
                      json.dumps(seg))
            decisions, _images, _c = resolve(tie_root)
            d = next(x for x in decisions if x["label"] == "PA")
            check(d["candidate"]["project_id"] == "late" and d["candidate"]["outcome"] == "found_ai",
                  f"equal outcome: the project with the latest at wins (got {d['candidate']['project_id']})")
            # A stronger but older project still beats a newer weaker one.
            pdir = make_project(tie_root, "run_old_manual_20260826_m1", "oldmanual", "oldmanual", target_species="PA")
            raw = os.path.join(pdir, "step4test_combinedAnnotate", "routed_input", "raw")
            img = make_image(raw, T101)
            seg = {T101: seg_entry(img, [mask(0, "PA", source_type="manual_box", sx=1, sy=2)],
                                   processed_at="2026-08-01T10:00:00")}
            write(os.path.join(pdir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
                  json.dumps(seg))
            decisions, _images, _c = resolve(tie_root)
            d = next(x for x in decisions if x["label"] == "PA")
            check(d["candidate"]["project_id"] == "oldmanual",
                  f"found_manual beats found_ai regardless of at (got {d['candidate']['project_id']})")

            # L10: registry-backed vs ledger-only project on the same date.
            # The registry stamps 'YYYY-MM-DD HH:MM:SS' (now); the ledger row
            # carries ISO with an offset at 00:00:00 today. Raw string order
            # would pick the ledger ('T' > ' '); the real latest is the registry.
            today = now_ast()[:10]
            led_stamp = f"{today}T00:00:00-04:00"
            for pid, run in (("aled", "run_aled_20260826_a1"), ("zreg", "run_zreg_20260826_z1")):
                pdir = make_project(tie_root, run, pid, pid, target_species="PA")
                raw = os.path.join(pdir, "step4test_combinedAnnotate", "routed_input", "raw")
                img = make_image(raw, T2015)
                m = mask(0, "PA", source_type="auto", sx=5, sy=6)
                seg = {T2015: seg_entry(img, [m], processed_at="2026-01-01T10:00:00")}
                write(os.path.join(pdir, "step4test_combinedAnnotate", "segmentations", "segmentations.json"),
                      json.dumps(seg))
                if pid == "aled":
                    write(os.path.join(pdir, "step4test_combinedAnnotate", "label_provenance.csv"),
                          "basename,label,outcome,source,reviewer,at,project_id\n"
                          f"{_stem(T2015)},PA,found_ai,step4test,EF,{led_stamp},aled\n")
                else:
                    rec = build_registry_record(T2015, m, "zreg", "zreg")
                    MaskRegistry(os.path.join(tie_root, "_mask_registry")).upsert(rec)
            decisions, images, _c = resolve(tie_root)
            cands = {c["project_id"]: c for c in images[_stem(T2015)]["by_label"]["PA"]}
            check(cands["aled"]["at"] == led_stamp and cands["zreg"]["at"][10] == " " and len(cands["zreg"]["at"]) == 19,
                  f"candidates carry the ledger ISO form and the registry space form (got {cands['aled']['at']!r}, {cands['zreg']['at']!r})")
            check(cands["aled"]["strength"] == cands["zreg"]["strength"], "both candidates are found_ai (equal strength)")
            check(max(cands["aled"]["at"], cands["zreg"]["at"]) == led_stamp,
                  "raw string order would have picked the ledger project")
            d = next(x for x in decisions if x["stem"] == _stem(T2015) and x["label"] == "PA")
            check(d["candidate"]["project_id"] == "zreg",
                  f"parsed at: the registry-backed project (stamped now) beats the ledger row from 00:00 today (got {d['candidate']['project_id']})")
        finally:
            shutil.rmtree(tie_root, ignore_errors=True)
        check(not os.path.exists(tie_root), "tie fixture removed")
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
