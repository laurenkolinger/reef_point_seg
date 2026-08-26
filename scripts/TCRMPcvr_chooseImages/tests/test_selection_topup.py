"""
Self-contained unit tests for the forgiving year-stratified top-up in
TCRMPcvr_chooseImages/src/select_images.py.

No pytest dependency: run with the unified env python:
    env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_selection_topup.py

The bug under test: greedy_select() is year-stratified — even_allocate() splits
each label's target across years, and greedy_select fills each year's quota
independently and BREAKS when a year holds no more frames with a needed label.
Deficits were never redistributed to OTHER years that still contain that label,
so a label could finish SHORT even though binary[sp].sum() >= target.

even_allocate() also under-allocates by rounding: for a label whose instances
are spread lumpily across many years, per_year = remaining/len(active) rounds
down each year and the leftover is never placed, so sum(alloc[sp]) < target even
though plenty of frames exist. That is precisely the "even per-year alloc cannot
fill it within quota" condition. These tests build such a matrix and assert:
  (a) WITHOUT the top-up the label would be short  -> documented via the alloc
      (sum(alloc[OFRA]) < target) and a replica of the original pass-0 selection
      that only reaches sum(alloc), i.e. short;
  (b) WITH greedy_select the label reaches target  -> top-up redistributes;
  (c) a genuinely scarce label (total < target) still reports SHORT and selects
      every available frame.
"""

import os
import sys
import traceback

import pandas as pd
import numpy as np

# src/ on path so `import config` inside select_images resolves.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
sys.path.insert(0, _SRC)

import select_images as si  # noqa: E402


# ── tiny test harness (mirrors _reefreview/tests/test_reefreview.py) ────────
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


# ── replica of the ORIGINAL year-stratified pass (pass 0, no top-up) ────────
# Used only to prove the pre-top-up shortfall actually occurs. This mirrors the
# code that ships in greedy_select's pass 0 but stops before the top-up loop.
def _pass0_only(binary, species_list, alloc, years):
    has_central = all(f"{sp}_central" in binary.columns for sp in species_list)
    selected, achieved, used = [], {sp: 0 for sp in species_list}, set()
    for pass_name in (["central", "any"] if has_central else ["any"]):
        for yr in years:
            yr_data = binary[(binary["year_int"] == yr) & (~binary.index.isin(used))]
            yr_rem = {sp: max(0, alloc[sp].get(yr, 0) - sum(
                int(binary.loc[fid, sp]) for fid in selected
                if binary.loc[fid, "year_int"] == yr
            )) for sp in species_list}
            avail = yr_data.copy()
            while any(r > 0 for r in yr_rem.values()) and len(avail) > 0:
                scores = pd.Series(0.0, index=avail.index)
                n_needed = sum(1 for sp in species_list if yr_rem[sp] > 0)
                for sp in species_list:
                    if yr_rem[sp] > 0:
                        scores += avail[sp]
                if n_needed > 1:
                    nsf = pd.Series(0, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            nsf += avail[sp]
                    scores += (nsf == 1).astype(float) * 0.5
                if pass_name == "central":
                    cm = pd.Series(False, index=avail.index)
                    for sp in species_list:
                        if yr_rem[sp] > 0:
                            cm |= avail[f"{sp}_central"] > 0
                    scores = scores.where(cm, 0)
                if scores.max() == 0:
                    break
                best = scores.idxmax()
                selected.append(best)
                used.add(best)
                for sp in species_list:
                    a = int(avail.loc[best, sp])
                    yr_rem[sp] = max(0, yr_rem[sp] - a)
                    achieved[sp] += a
                avail = avail.drop(best)
    return selected, achieved


# ── synthetic binary matrix builder ─────────────────────────────────────────
def _build_matrix(target):
    """Tiny binary frame x label matrix (the shape greedy_select consumes).

    Labels:
      OFRA — 25 instances spread lumpily across 9 years (2014-2022). With
             target=10 even_allocate hands each active year per_year = 10/9 and
             rounds down to 1, never placing the 1 leftover -> sum(alloc) == 9,
             so the stratified pass tops out at 9 (SHORT by 1) despite 25 frames.
      PA   — plentiful but concentrated in just 2 years, so even_allocate gives
             it 5+5 == target exactly. PA anchors the scalar target that
             greedy_select recovers from the allocation (max alloc-sum).
      SCAR — genuinely scarce: only 3 instances total, < target. Must stay SHORT
             and have all 3 frames selected.

    No frame carries more than one label, so achieved counts are unambiguous.
    """
    rows = []
    fid = 0

    def add(year, ofra, pa, scar):
        nonlocal fid
        rows.append({
            "frame_id": f"f{fid}", "year_int": year,
            "OFRA": ofra, "PA": pa, "SCAR": scar,
        })
        fid += 1

    ofra_counts = {2014: 3, 2015: 1, 2016: 4, 2017: 1,
                   2018: 4, 2019: 4, 2020: 2, 2021: 3, 2022: 3}  # 25 total
    for y, c in ofra_counts.items():
        for _ in range(c):
            add(y, 1, 0, 0)
    for y in (2014, 2015):            # PA concentrated -> alloc sum == target
        for _ in range(8):
            add(y, 0, 1, 0)
    for _ in range(3):                # SCAR scarce
        add(2016, 0, 0, 1)

    binary = pd.DataFrame(rows).set_index("frame_id")
    return binary, ["OFRA", "PA", "SCAR"]


# ── tests ───────────────────────────────────────────────────────────────────
def test_alloc_underfills_concentrated_label():
    """(a) Document the pre-top-up shortfall: even-per-year alloc < target and
    the original stratified pass only reaches that capped allocation."""
    target = 10
    binary, sl = _build_matrix(target)
    years = sorted(binary["year_int"].unique())
    alloc = si.even_allocate(binary, sl, years, target)

    ofra_alloc = sum(alloc["OFRA"].values())
    pa_alloc = sum(alloc["PA"].values())

    # Enough OFRA instances exist to meet target...
    check(int(binary["OFRA"].sum()) >= target,
          f"OFRA total {int(binary['OFRA'].sum())} should be >= target {target}")
    # ...but the even per-year allocation cannot place them all (rounding loss),
    # so the stratified pass would leave OFRA short.
    check(ofra_alloc < target,
          f"expected even_allocate to underfill OFRA: {ofra_alloc} < {target}")
    # PA's allocation anchors the true target (so greedy_select can recover it).
    check(pa_alloc == target,
          f"PA allocation should equal target to anchor it: {pa_alloc} == {target}")

    # The original pass-0 selection (no top-up) reaches only the capped alloc.
    _, a0 = _pass0_only(binary, sl, alloc, years)
    check(a0["OFRA"] == ofra_alloc,
          f"pass-0 reaches the capped alloc, not target: {a0['OFRA']} == {ofra_alloc}")
    check(a0["OFRA"] < target,
          f"pass-0 leaves OFRA SHORT: {a0['OFRA']} < {target}")


def test_topup_reaches_target_for_concentrated_label():
    """(b) WITH greedy_select the concentrated label reaches target."""
    target = 10
    binary, sl = _build_matrix(target)
    years = sorted(binary["year_int"].unique())
    alloc = si.even_allocate(binary, sl, years, target)

    selected, achieved = si.greedy_select(binary, sl, alloc, years)

    check(achieved["OFRA"] >= target,
          f"top-up should bring OFRA to target: {achieved['OFRA']} >= {target}")
    check(achieved["PA"] >= target,
          f"PA stays at target: {achieved['PA']} >= {target}")
    # 'used' dedupe: no frame selected twice.
    check(len(selected) == len(set(selected)),
          f"no frame selected twice: {len(selected)} == {len(set(selected))}")
    # Every selected frame is a real index in the matrix.
    check(all(fid in binary.index for fid in selected), "all selected frames valid")


def test_scarce_label_stays_short_and_exhausts():
    """(c) A genuinely scarce label (total < target) stays SHORT and every
    available frame for it is selected."""
    target = 10
    binary, sl = _build_matrix(target)
    years = sorted(binary["year_int"].unique())
    alloc = si.even_allocate(binary, sl, years, target)

    scar_total = int(binary["SCAR"].sum())
    check(scar_total < target, f"SCAR is scarce: {scar_total} < {target}")

    selected, achieved = si.greedy_select(binary, sl, alloc, years)

    # SHORT, but only because the data is exhausted — achieved == total avail.
    check(achieved["SCAR"] < target,
          f"SCAR genuinely SHORT: {achieved['SCAR']} < {target}")
    check(achieved["SCAR"] == scar_total,
          f"SCAR exhausts all available frames: {achieved['SCAR']} == {scar_total}")
    # All SCAR-bearing frames are in the selection.
    scar_frames = set(binary.index[binary["SCAR"] > 0])
    check(scar_frames.issubset(set(selected)),
          f"every SCAR frame selected: {scar_frames - set(selected)} missing")


def test_topup_with_central_columns_relaxes_and_dedupes():
    """The central-region rung must not be skipped or regressed. Build the same
    matrix WITH `{sp}_central` columns present (all zero, so the central rung
    relaxes to year+central) and confirm OFRA still reaches target, SCAR still
    exhausts, and no frame is selected twice."""
    target = 10
    binary, sl = _build_matrix(target)
    for sp in sl:
        binary[f"{sp}_central"] = 0          # no central points -> rung 2 relaxes
    years = sorted(binary["year_int"].unique())
    alloc = si.even_allocate(binary, sl, years, target)

    selected, achieved = si.greedy_select(binary, sl, alloc, years)

    check(achieved["OFRA"] >= target,
          f"OFRA reaches target with central cols: {achieved['OFRA']} >= {target}")
    check(achieved["SCAR"] == int(binary["SCAR"].sum()),
          f"SCAR exhausts with central cols: {achieved['SCAR']}")
    check(len(selected) == len(set(selected)),
          f"no dup selection with central cols: {len(selected)} vs {len(set(selected))}")


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} selection top-up tests...\n")
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
