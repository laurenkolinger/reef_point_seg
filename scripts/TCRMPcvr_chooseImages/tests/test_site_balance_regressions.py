"""Site-balance regressions: allocation convergence, cross-species crosstalk,
and honest shortfall reasons.

Three verified defects are pinned here:

  (a) allocate_by_site deadlock. The water-fill cap was recomputed each round
      from the RUNNING total. Right after the 1-per-site floor round the
      running total equals n_sites, so at cap_frac 0.25 any site count of 7 or
      fewer produced floor(0.25 * n_sites) == 1, zero headroom, and an
      immediate break: 2 to 7 sites allocated exactly n_sites frames no matter
      how large the target was.

  (b) Cross-species crosstalk in greedy_select. The per-label target was
      recovered as ONE scalar (the max allocation-sum over species). Inside
      site_balanced_select that scalar is the DOMINANT species' site target, so
      a low-target species at the same site was topped up to the other
      species' number and the dominant site swelled past its own share.

  (c) Dishonest shortfall reporting. Every shortfall printed "pool exhausted",
      including shortfalls caused by the cap or by the allocation, so an
      operator could not tell a real data limit from an algorithm limit.

No pytest: run with the unified env python.
    env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_site_balance_regressions.py
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

import os
import sys
import math
import traceback

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

import select_images as si  # noqa: E402


_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ── (a) allocation convergence for small site counts ────────────────────────

def test_small_site_counts_reach_target():
    """2 through 7 sites, 500 frames each, target 1000.

    Before the fix every one of these returned exactly n_sites frames (2..7)
    because the cap was seeded from the post-floor running total. After the fix
    each converges on the full target.

    The per-site bound is the tighter of the two rules that can apply:
    when there are enough sites for cap_frac to be satisfiable at all the
    cap_frac share binds; when there are not (2 or 3 sites at cap_frac 0.25),
    an even split is the most balanced allocation that exists, so the bound
    falls back to ceil(total / n_sites). One frame of slack absorbs
    largest-remainder rounding.
    """
    for n_sites in range(2, 8):
        avail = {f"S{i:02d}": 500 for i in range(n_sites)}
        target = 1000
        result = si.allocate_by_site(avail, target, cap_frac=0.25)

        total = sum(result.values())
        check(total == target,
              f"n_sites={n_sites}: allocated {total}, expected {target} "
              f"(availability is {n_sites * 500}, far above target)")
        check(set(result.keys()) == set(avail.keys()),
              f"n_sites={n_sites}: every site must appear: {result}")

        bound = max(math.ceil(0.25 * total), math.ceil(total / n_sites)) + 1
        worst = max(result.values())
        check(worst <= bound,
              f"n_sites={n_sites}: max site allocation {worst} exceeds "
              f"bound {bound}: {result}")


def test_eight_sites_unchanged_by_the_fix():
    """8 sites already converged before the fix; it must still converge."""
    avail = {f"S{i:02d}": 500 for i in range(8)}
    result = si.allocate_by_site(avail, 1000, cap_frac=0.25)
    total = sum(result.values())
    check(total == 1000, f"8 sites should allocate the full 1000: {total}")
    check(max(result.values()) <= math.floor(0.25 * total),
          f"8 sites: cap must bind: {result}")


def test_cap_binds_at_convergence_for_a_dominant_site():
    """A dominant site is still held to floor(cap_frac * FINAL total).

    The convergence fix seeds the cap from the intended total, so it must not
    let the dominant site keep a share that the final total no longer supports.
    """
    avail = {"CRB": 44, "BPT": 8, "CBS": 7, "KGC": 6}
    for i in range(24):
        avail[f"S{i:02d}"] = 1 + (i % 3)

    result = si.allocate_by_site(avail, 1000, cap_frac=0.25)
    total = sum(result.values())
    cap = math.floor(0.25 * total)
    check(result["CRB"] <= cap,
          f"CRB {result['CRB']} must stay at or under floor(0.25*{total})={cap}")
    for s, a in avail.items():
        check(1 <= result[s] <= a, f"{s}: {result[s]} outside [1, {a}]")


def test_allocation_is_still_deterministic_after_the_fix():
    avail = {f"S{i:02d}": 500 for i in range(5)}
    r1 = si.allocate_by_site(avail, 1000, cap_frac=0.25)
    r2 = si.allocate_by_site(avail, 1000, cap_frac=0.25)
    check(r1 == r2, f"allocate_by_site must stay deterministic: {r1} vs {r2}")


# ── (b) cross-species crosstalk ─────────────────────────────────────────────

def _two_species_matrix():
    """SPA is abundant at 11 sites; SPB is concentrated at CRB.

    SPA's CRB allocation runs to several frames; SPB's CRB allocation is small
    because SPB also occurs at three other sites, so the cap holds its dominant
    site down. Before the fix greedy_select recovered ONE scalar target for the
    CRB sub-frame (SPA's number) and topped SPB up to it, so CRB carried far
    more than its combined allocation of the selection.

    No frame carries both labels, so achieved counts are unambiguous.
    """
    rows = []
    fid = 0

    def add(year, site, transect, spa, spb):
        nonlocal fid
        rows.append({
            "frame_id": f"f{fid}", "year_int": year, "year": year,
            "date": f"{year}-01-01", "site": site, "transect": transect,
            "frame": fid, "SPA": spa, "SPB": spb,
        })
        fid += 1

    years = [2018, 2019]
    for i in range(60):
        add(years[i % 2], "CRB", 1, 1, 0)
    for i in range(20):
        add(years[i % 2], "CRB", 2, 0, 1)
    small_spb = {"BPT": 2, "CBS": 2, "KGC": 2}
    for site in ["BPT", "CBS", "KGC", "TEK", "SPR",
                 "SAL", "HAW", "GTB", "LSC", "YAW"]:
        for i in range(20):
            add(years[i % 2], site, 1, 1, 0)
        for i in range(small_spb.get(site, 0)):
            add(years[i % 2], site, 2, 0, 1)

    return pd.DataFrame(rows).set_index("frame_id"), ["SPA", "SPB"], years


def test_low_target_species_is_not_topped_up_to_another_species_target():
    binary, species_list, years = _two_species_matrix()
    target = 40

    site_targets = si.compute_site_targets(
        binary, species_list, target, cap_frac=0.25)
    spb_at_crb = site_targets["SPB"].get("CRB", 0)
    spa_at_crb = site_targets["SPA"].get("CRB", 0)
    check(spb_at_crb < spa_at_crb,
          f"fixture must give SPB a smaller CRB target than SPA: "
          f"SPB={spb_at_crb} SPA={spa_at_crb}")

    selected, achieved = si.site_balanced_select(
        binary, species_list, years, target, cap_frac=0.25)

    sel = binary.loc[selected]
    spb_selected_at_crb = int(sel[sel["site"] == "CRB"]["SPB"].sum())
    check(spb_selected_at_crb <= spb_at_crb,
          f"SPB must not exceed its own CRB allocation of {spb_at_crb}: "
          f"selected {spb_selected_at_crb}")


def test_dominant_site_stays_within_its_summed_allocation():
    binary, species_list, years = _two_species_matrix()
    target = 40

    site_targets = si.compute_site_targets(
        binary, species_list, target, cap_frac=0.25)
    crb_alloc = sum(site_targets[sp].get("CRB", 0) for sp in species_list)

    selected, achieved = si.site_balanced_select(
        binary, species_list, years, target, cap_frac=0.25)

    sel = binary.loc[selected]
    crb_frames = int((sel["site"] == "CRB").sum())
    check(crb_frames <= crb_alloc,
          f"CRB took {crb_frames} frames but was allocated {crb_alloc} "
          f"across all species: {sel['site'].value_counts().to_dict()}")


def _lopsided_alloc(binary, species_list, years):
    """An allocation whose per-label sums differ, which is exactly the shape
    site_balanced_select hands greedy_select for one site. SPA sums to 6, SPB
    to 1, so the scalar recovery (max allocation-sum) is 6 for both labels."""
    return {
        "SPA": {years[0]: 3, years[1]: 3},
        "SPB": {years[0]: 1, years[1]: 0},
    }


def test_global_path_still_recovers_one_scalar_target():
    """Without target_by_species the old scalar recovery must be untouched:
    SPB is pulled up to SPA's allocation-sum of 6. This is the pre-existing
    behavior that the global (no site balance) path depends on."""
    binary, species_list, years = _two_species_matrix()
    alloc = _lopsided_alloc(binary, species_list, years)
    selected, achieved = si.greedy_select(binary, species_list, alloc, years)
    check(achieved["SPB"] == 6,
          f"scalar recovery should top SPB up to 6: {achieved}")
    check(achieved["SPA"] >= 6, f"SPA reaches 6 as well: {achieved}")


def test_target_by_species_is_honored_per_label():
    """With per-label targets, SPB stops at ITS number instead of SPA's."""
    binary, species_list, years = _two_species_matrix()
    alloc = _lopsided_alloc(binary, species_list, years)
    selected, achieved = si.greedy_select(
        binary, species_list, alloc, years,
        target_by_species={"SPA": 6, "SPB": 1})
    check(achieved["SPB"] == 1,
          f"SPB must stop at its own target of 1, not SPA's 6: {achieved}")
    check(achieved["SPA"] >= 6, f"SPA should reach its own target 6: {achieved}")


# ── (c) honest shortfall reasons ────────────────────────────────────────────

def test_shortfall_reason_ok():
    check(si.shortfall_reason(achieved=10, target=10, pool_total=50,
                              alloc_total=10) == "ok",
          "target met should read ok")


def test_shortfall_reason_pool_exhausted():
    r = si.shortfall_reason(achieved=3, target=10, pool_total=3, alloc_total=3)
    check(r == "pool exhausted", f"every existing frame taken: got {r}")


def test_shortfall_reason_cap_limited():
    r = si.shortfall_reason(achieved=20, target=40, pool_total=500,
                            alloc_total=20)
    check(r == "cap-limited",
          f"pool is ample but the allocation fell short: got {r}")


def test_shortfall_reason_allocation_limited():
    r = si.shortfall_reason(achieved=30, target=40, pool_total=500,
                            alloc_total=40)
    check(r == "allocation-limited",
          f"allocation met target but selection did not: got {r}")


def test_summary_lines_render_the_reason():
    lines = si.build_summary_lines(
        candidates=100, eligible=70, selected=20, reserve=50,
        species_list=["SPA", "SPB"],
        achieved={"SPA": 40, "SPB": 20}, target=40,
        exclusions={"image_missing": 0, "cpc_missing": 0, "pts_missing": 0},
        reasons={"SPA": "ok", "SPB": "cap-limited"})
    txt = "\n".join(lines)
    check("cap-limited" in txt,
          f"cap-limited shortfall must be named in the summary:\n{txt}")
    check("pool exhausted" not in txt,
          f"a cap-limited shortfall must not be reported as exhaustion:\n{txt}")


def test_summary_lines_default_reason_is_unchanged():
    lines = si.build_summary_lines(
        candidates=10, eligible=8, selected=5, reserve=3,
        species_list=["SPA"], achieved={"SPA": 3}, target=5,
        exclusions={})
    txt = "\n".join(lines)
    check("pool exhausted" in txt,
          f"omitting reasons must keep the original wording:\n{txt}")


# ── selection_summary.json (Task 11 consumes this verbatim) ─────────────────

def test_selection_summary_json_shape():
    doc = si.build_selection_summary(
        species_list=["SPA", "SPB"],
        achieved={"SPA": 40, "SPB": 20},
        target=40,
        per_site={"CRB": 12, "BPT": 8, "CBS": 8},
        pool_totals={"SPA": 500, "SPB": 20},
        alloc_totals={"SPA": 40, "SPB": 20},
        cap_frac=0.25,
        n_selected=28)

    check(set(doc.keys()) >= {"per_site", "per_species", "cap_frac",
                              "cap_ok", "n_selected", "shortfalls"},
          f"selection_summary.json is missing keys: {sorted(doc.keys())}")
    check(doc["per_site"] == {"CRB": 12, "BPT": 8, "CBS": 8},
          f"per_site must be the selected count per site: {doc['per_site']}")
    check(doc["per_species"]["SPA"] == {"target": 40, "achieved": 40,
                                        "reason": "ok"},
          f"SPA entry wrong: {doc['per_species']['SPA']}")
    check(doc["per_species"]["SPB"]["reason"] == "pool exhausted",
          f"SPB took every frame that exists: {doc['per_species']['SPB']}")
    check(doc["n_selected"] == 28, f"n_selected wrong: {doc['n_selected']}")
    check(doc["cap_ok"] is False,
          f"CRB holds 12/28 = 43 percent, above the 25 percent cap: {doc}")
    check([s["species"] for s in doc["shortfalls"]] == ["SPB"],
          f"only SPB is short: {doc['shortfalls']}")


def test_selection_summary_json_cap_ok_true():
    doc = si.build_selection_summary(
        species_list=["SPA"], achieved={"SPA": 40}, target=40,
        per_site={f"S{i:02d}": 5 for i in range(8)},
        pool_totals={"SPA": 500}, alloc_totals={"SPA": 40},
        cap_frac=0.25, n_selected=40)
    check(doc["cap_ok"] is True,
          f"8 equal sites at 12.5 percent each are under the cap: {doc}")
    check(doc["shortfalls"] == [], f"nothing is short: {doc['shortfalls']}")


# ── reserve ordering feeds the site-aware Step 4 refill ─────────────────────

def _reserve_pool():
    """CRB holds most of the selection; CBS holds none of it. Both keep
    unselected frames in the reserve pool."""
    rows = []
    fid = 0
    for site, n in (("CRB", 30), ("BPT", 12), ("CBS", 12)):
        for i in range(n):
            rows.append({
                "frame_id": f"{site}{i:02d}", "year_int": 2022, "year": 2022,
                "date": "2022-06-01", "site": site, "transect": (i % 3) + 1,
                "frame": i + 1, "PA": 1,
            })
            fid += 1
    return pd.DataFrame(rows).set_index("frame_id")


def test_reserve_rows_lead_with_under_represented_sites():
    binary = _reserve_pool()
    # A deliberately skewed selection: CRB takes 20, BPT 4, CBS 0.
    selected = ([f"CRB{i:02d}" for i in range(20)]
                + [f"BPT{i:02d}" for i in range(4)])
    clip_index = {}

    rows = si.build_reserve_rows(binary, selected, ["PA"],
                                 si.build_image_filename, clip_index)

    check(len(rows) == len(binary) - len(selected),
          f"reserve is the eligible pool minus the selection: {len(rows)}")
    ranks = sorted(r["reserve_rank"] for r in rows)
    check(ranks == list(range(len(rows))), f"ranks must be 0..N-1: {ranks[:5]}")

    order = [r["site"] for r in sorted(rows, key=lambda r: r["reserve_rank"])]
    # CBS holds 0 percent of the selection, BPT 4/24, CRB 20/24, so every CBS
    # row must precede every BPT row and every BPT row every CRB row.
    first_crb = order.index("CRB")
    first_bpt = order.index("BPT")
    last_cbs = len(order) - 1 - order[::-1].index("CBS")
    last_bpt = len(order) - 1 - order[::-1].index("BPT")
    check(last_cbs < first_bpt,
          f"CBS holds none of the selection so it must rank first: {order}")
    check(last_bpt < first_crb,
          f"BPT is less represented than CRB so it must outrank it: {order}")


def test_reserve_rows_keep_random_order_within_a_site():
    """Two runs over the same skewed selection must agree on the site ordering
    but differ in the order of rows inside a site, so the fresh-random
    tie-break survives the new site-share sort."""
    binary = _reserve_pool()
    selected = ([f"CRB{i:02d}" for i in range(20)]
                + [f"BPT{i:02d}" for i in range(4)])

    site_orders = []
    within = []
    for _ in range(8):
        rows = sorted(si.build_reserve_rows(binary, selected, ["PA"],
                                            si.build_image_filename, {}),
                      key=lambda r: r["reserve_rank"])
        site_orders.append([r["site"] for r in rows])
        within.append([r["frame_id"] for r in rows if r["site"] == "CBS"])

    check(all(o == site_orders[0] for o in site_orders),
          "the site ordering must be deterministic across runs")
    check(any(w != within[0] for w in within),
          f"rows inside a site must still shuffle: {within[0]}")


# ── end to end: main() actually writes selection_summary.json ───────────────

def test_main_writes_selection_summary_json():
    """Drive main() over a synthetic pool and read the JSON back.

    Everything lives in a temp tree that is removed in the finally block; no
    project data is touched and no service is started.
    """
    import csv
    import json
    import shutil
    import tempfile

    scratch = os.environ.get(
        "TMPDIR", "/tmp/claude-1000/-mnt-rip-vicarius-drive/"
                  "ab8997e1-410f-4a89-a1a3-8a15baff3ddb/scratchpad")
    root = tempfile.mkdtemp(prefix="chooseimages_e2e_",
                            dir=scratch if os.path.isdir(scratch) else None)
    cfg = si.config
    saved = {k: getattr(cfg, k) for k in
             ("CLIP_DIR", "CPC_ALL_DIR", "OUTPUT_DIR", "_RECODE_DIR",
              "_REPO_DIR")}
    saved_argv = sys.argv[:]
    try:
        clip_dir = os.path.join(root, "clip")
        out_dir = os.path.join(root, "output")
        empty = os.path.join(root, "empty")
        for d in (clip_dir, out_dir, empty):
            os.makedirs(d, exist_ok=True)

        sites = [f"S{i:02d}" for i in range(8)]
        rows = []
        for si_idx, site in enumerate(sites):
            for f_idx in range(1, 13):
                year = 2022 if f_idx % 2 else 2023
                date = f"{year}-06-01"
                transect = (f_idx % 3) + 1
                base = si.build_image_filename(date, site, transect, f_idx)
                open(os.path.join(clip_dir, base + ".jpg"), "w").close()
                open(os.path.join(clip_dir, base + "_pts.jpg"), "w").close()
                rows.append({"date": date, "year": year, "site": site,
                             "transect": transect, "frame": f_idx,
                             "species_code": "PA", "category": "Coral",
                             "point_label": "PA", "x": 1000, "y": 700})
                if si_idx < 3:
                    rows.append({"date": date, "year": year, "site": site,
                                 "transect": transect, "frame": f_idx,
                                 "species_code": "OFAV", "category": "Coral",
                                 "point_label": "OFAV", "x": 1200, "y": 800})

        ap_path = os.path.join(root, "all_points.csv")
        with open(ap_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        mc_path = os.path.join(root, "master_codes.csv")
        with open(mc_path, "w", newline="") as f:
            f.write("code,name\nPA,Porites astreoides\nOFAV,Orbicella faveolata\n")

        cfg.CLIP_DIR = clip_dir
        cfg.CPC_ALL_DIR = empty
        cfg.OUTPUT_DIR = out_dir
        cfg._RECODE_DIR = empty
        cfg._REPO_DIR = empty

        sys.argv = ["select_images.py", "--all-points", ap_path,
                    "--master-codes", mc_path, "--species", "PA", "OFAV",
                    "--target", "40", "--min-year", "2022",
                    "--max-year", "2023", "--skip-image-check"]
        si.main()

        json_path = os.path.join(out_dir, "selection_summary.json")
        check(os.path.exists(json_path),
              f"selection_summary.json was not written: {json_path}")
        doc = json.load(open(json_path))

        check(set(doc.keys()) >= {"per_site", "per_species", "cap_frac",
                                  "cap_ok", "n_selected", "shortfalls"},
              f"schema keys missing: {sorted(doc.keys())}")
        check(doc["n_selected"] == sum(doc["per_site"].values()),
              f"per_site must sum to n_selected: {doc}")
        check(set(doc["per_species"].keys()) == {"PA", "OFAV"},
              f"both labels must appear: {doc['per_species']}")
        for sp, entry in doc["per_species"].items():
            check(entry["reason"] in si.SHORTFALL_REASONS,
                  f"{sp} carries an unknown reason: {entry}")
            check(entry["target"] == 40, f"{sp} target wrong: {entry}")
        check(doc["cap_ok"] is True,
              f"8 balanced sites should clear the 25 percent cap: {doc}")
        check(len(doc["per_site"]) == 8,
              f"every site must be represented: {doc['per_site']}")

        txt = open(os.path.join(out_dir, "selection_summary.txt")).read()
        check("=== Sites (8) ===" in txt,
              f"text summary should still list 8 sites:\n{txt[:400]}")
    finally:
        sys.argv = saved_argv
        for k, v in saved.items():
            setattr(cfg, k, v)
        shutil.rmtree(root, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} site-balance regression tests...\n")
    for fn in tests:
        run(fn)
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = [(n, e) for n, ok, e in _RESULTS if not ok]
    print(f"\n==== {passed}/{len(_RESULTS)} passed ====")
    if failed:
        print("\nFAILURES:")
        for n, e in failed:
            print(f"\n--- {n} ---\n{e}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
