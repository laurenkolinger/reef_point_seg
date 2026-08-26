"""Site-balanced selection: no single site may dominate Step 3 output.

Covers two units:
  (1) allocate_by_site - pure per-species site allocation (water-filling,
      inclusion floor of 1, 25% cap, deterministic).
  (2) site_balanced_select - end-to-end selection reusing greedy_select PER
      SITE so a dominant site (e.g. Coral Bay) cannot swamp the selected set.

No pytest dependency: run with the unified env python:
    env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_site_balance.py
"""

import os
import sys
import traceback
import math

import pandas as pd

# src/ on path so `import config` inside select_images resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

import select_images as si  # noqa: E402


# ── tiny test harness (mirrors test_selection_topup.py) ─────────────────────
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


# ── Part A: allocate_by_site ─────────────────────────────────────────────────

def test_dominant_site_is_capped():
    """CRB dwarfs 28 other sites; target vastly exceeds total availability, so
    everything-but-capped is taken. CRB must land <= floor(0.25*total)."""
    avail = {"CRB": 44, "BPT": 8, "CBS": 7, "KGC": 6}
    # pad out to 28 sites total (matches the spec's concrete case shape)
    for i in range(24):
        avail[f"S{i:02d}"] = 1 + (i % 3)
    target = 1000

    result = si.allocate_by_site(avail, target, cap_frac=0.25)

    check(set(result.keys()) == set(avail.keys()), "every site with avail>=1 present")
    for s, a in avail.items():
        check(result[s] >= 1, f"{s} must be included (>=1): got {result[s]}")
        check(result[s] <= a, f"{s} must not exceed avail {a}: got {result[s]}")

    total = sum(result.values())
    # Target vastly exceeds total availability, so every NON-capped site is
    # fully exhausted to its own avail. CRB alone is held below its avail by
    # the cap, so the total is necessarily less than sum(avail) by exactly
    # CRB's shortfall (that's the whole point of the cap).
    non_dominant_total = sum(v for s, v in result.items() if s != "CRB")
    non_dominant_avail = sum(a for s, a in avail.items() if s != "CRB")
    check(non_dominant_total == non_dominant_avail,
          f"every non-dominant site should be fully exhausted: "
          f"{non_dominant_total} == {non_dominant_avail}")

    cap = math.floor(0.25 * total)
    check(result["CRB"] <= cap,
          f"CRB must be capped at floor(0.25*total)={cap}, got {result['CRB']}")
    check(result["CRB"] < avail["CRB"],
          f"CRB should be capped well below its raw availability {avail['CRB']}: "
          f"got {result['CRB']}")


def test_target_below_n_sites_gives_everyone_floor_of_one():
    """30 sites, target=10 (< n_sites) -> every site still gets exactly 1;
    inclusion beats the numeric target."""
    avail = {f"S{i:02d}": 5 for i in range(30)}
    target = 10

    result = si.allocate_by_site(avail, target, cap_frac=0.25)

    check(set(result.keys()) == set(avail.keys()), "every site present")
    for s in avail:
        check(result[s] == 1, f"{s} should get exactly the floor of 1: got {result[s]}")
    check(sum(result.values()) == 30,
          f"total == n_sites (30) even though target was 10: got {sum(result.values())}")


def test_proportional_mid_case():
    """avail={A:100,B:50,C:50}, target=40, cap 0.25 -> A capped at 10 (25% of
    40); B and C share the rest proportionally; all >=1."""
    avail = {"A": 100, "B": 50, "C": 50}
    target = 40

    result = si.allocate_by_site(avail, target, cap_frac=0.25)

    check(set(result.keys()) == set(avail.keys()), "every site present")
    total = sum(result.values())
    cap = max(1, math.floor(0.25 * total))
    check(result["A"] <= cap, f"A capped at floor(0.25*total)={cap}, got {result['A']}")
    for s in avail:
        check(result[s] >= 1, f"{s} >= 1: got {result[s]}")
        check(result[s] <= avail[s], f"{s} <= avail {avail[s]}: got {result[s]}")
    check(result["B"] == result["C"],
          f"B and C have equal avail so should split evenly: {result['B']} vs {result['C']}")


def test_every_site_with_avail_gets_floor_one():
    avail = {"A": 1, "B": 2, "C": 3}
    result = si.allocate_by_site(avail, target=1, cap_frac=0.25)
    for s in avail:
        check(result[s] >= 1, f"{s} floor of 1: got {result[s]}")


def test_zero_avail_sites_excluded():
    avail = {"A": 5, "B": 0}
    result = si.allocate_by_site(avail, target=10, cap_frac=0.25)
    check("B" not in result or result.get("B", 0) == 0,
          f"a site with avail 0 should not receive frames: {result.get('B')}")
    check(result["A"] >= 1, "A included")


def test_never_exceeds_avail():
    avail = {"A": 3, "B": 2, "C": 1, "D": 50}
    result = si.allocate_by_site(avail, target=1000, cap_frac=0.25)
    for s, a in avail.items():
        check(result.get(s, 0) <= a, f"{s}: {result.get(s, 0)} must not exceed avail {a}")


def test_monotonic_in_avail():
    """Higher-abundance sites get >= lower-abundance sites (modulo cap/rounding)."""
    avail = {"LOW": 5, "MID": 20, "HIGH": 60}
    result = si.allocate_by_site(avail, target=50, cap_frac=0.25)
    check(result["HIGH"] >= result["MID"] >= result["LOW"],
          f"expected monotonic allocation, got {result}")


def test_deterministic():
    avail = {"A": 44, "B": 8, "C": 7, "D": 6, "E": 3, "F": 12, "G": 1}
    r1 = si.allocate_by_site(avail, target=200, cap_frac=0.25)
    r2 = si.allocate_by_site(avail, target=200, cap_frac=0.25)
    check(r1 == r2, f"allocate_by_site must be deterministic: {r1} vs {r2}")


def test_cap_unsatisfiable_with_few_sites_floors_win():
    """With only 2 sites, 25% cap is mathematically below the floor of 1
    (cap would be 0 or 1 for tiny totals) - floors must win, no site is ever
    forced below 1, and no exception is raised."""
    avail = {"A": 20, "B": 20}
    result = si.allocate_by_site(avail, target=10, cap_frac=0.25)
    check(result["A"] >= 1 and result["B"] >= 1,
          f"floors must win when cap is unsatisfiable: {result}")


# ── Part B: site_balanced_select ─────────────────────────────────────────────

def _build_site_matrix():
    """Synthetic binary matrix: CRB dominant (40 eligible PA frames) plus ~10
    small sites (3-6 frames each). Mirrors the real LO_test_PA imbalance."""
    rows = []
    fid = 0

    def add(year, site, transect, pa):
        nonlocal fid
        rows.append({
            "frame_id": f"f{fid}", "year_int": year, "year": year,
            "date": f"{year}-01-01", "site": site, "transect": transect,
            "frame": fid, "PA": pa,
        })
        fid += 1

    years = [2018, 2019, 2020, 2021]
    # Dominant site: 40 PA frames spread across years/transects.
    for i in range(40):
        add(years[i % len(years)], "CRB", (i % 3) + 1, 1)

    # ~10 small sites, 3-6 PA frames each.
    small_sites = {"BPT": 6, "CBS": 5, "KGC": 4, "TEK": 4, "SPR": 3,
                   "SAL": 3, "HAW": 5, "GTB": 4, "LSC": 3, "YAW": 6}
    for site, n in small_sites.items():
        for i in range(n):
            add(years[i % len(years)], site, (i % 3) + 1, 1)

    binary = pd.DataFrame(rows).set_index("frame_id")
    return binary, ["PA"], years, small_sites


def test_site_balanced_select_caps_dominant_site():
    binary, species_list, years, small_sites = _build_site_matrix()
    target = 40  # requested PA frame-instances

    selected, achieved = si.site_balanced_select(
        binary, species_list, years, target, cap_frac=0.25)

    check(len(selected) == len(set(selected)), "no frame double-counted")
    check(all(fid in binary.index for fid in selected), "all selected frames valid")

    sel_sites = binary.loc[selected, "site"]
    n_total = len(selected)
    check(n_total > 0, "selection non-empty")

    site_counts = sel_sites.value_counts()
    max_frac = site_counts.max() / n_total
    check(max_frac <= 0.25 + 1e-9,
          f"no site should exceed 25% of selection: {site_counts.to_dict()} "
          f"(max_frac={max_frac:.3f}, n_total={n_total})")

    check("CRB" in site_counts.index, "dominant site CRB still included")
    check(site_counts["CRB"] / n_total <= 0.25 + 1e-9,
          f"CRB specifically must be <= 25%: {site_counts['CRB']}/{n_total}")

    # Every small site with the label present must be included (floor >= 1).
    all_sites = set(binary["site"].unique())
    missing = all_sites - set(site_counts.index)
    check(not missing, f"every site with PA must be included: missing {missing}")


def test_site_balanced_select_achieved_matches_selection():
    binary, species_list, years, _ = _build_site_matrix()
    selected, achieved = si.site_balanced_select(
        binary, species_list, years, target=40, cap_frac=0.25)

    for sp in species_list:
        true_count = int(binary.loc[selected, sp].sum())
        check(achieved[sp] == true_count,
              f"achieved[{sp}] should match actual selected-frame sum: "
              f"{achieved[sp]} vs {true_count}")


def test_site_balanced_select_no_duplicate_frames_across_sites():
    binary, species_list, years, _ = _build_site_matrix()
    selected, achieved = si.site_balanced_select(
        binary, species_list, years, target=100, cap_frac=0.25)
    check(len(selected) == len(set(selected)),
          f"frame must never be selected by more than one site's pass: "
          f"{len(selected)} vs {len(set(selected))}")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} site-balance tests...\n")
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
