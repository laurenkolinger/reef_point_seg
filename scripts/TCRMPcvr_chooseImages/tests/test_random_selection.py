"""Selection is fresh-random within score ties, and always hits invariants.
Run: env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_random_selection.py
"""
import os, sys, traceback
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
import select_images as si  # noqa: E402

_R = []
def check(c, m):
    if not c: raise AssertionError(m)
def run(fn):
    try: fn(); _R.append((fn.__name__, True, "")); print(f"  PASS {fn.__name__}")
    except Exception as e: _R.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}")); print(f"  FAIL {fn.__name__}: {e}")

def test_pick_best_is_random_among_ties():
    s = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0})
    picks = {si._pick_best(s) for _ in range(60)}
    check(picks <= {"a", "b", "c"}, f"never picks the loser d: {picks}")
    check(len(picks) >= 2, f"expected randomness among ties, always picked {picks}")

def test_pick_best_none_when_no_positive():
    check(si._pick_best(pd.Series({"a": 0.0, "b": 0.0})) is None, "zero scores -> None")

def test_selection_varies_but_meets_target():
    # 20 single-species frames all tie; target 5. Two runs should usually differ,
    # but both must select exactly 5 distinct valid frames.
    rows = [{"frame_id": f"f{i}", "year_int": 2022, "PA": 1} for i in range(20)]
    binary = pd.DataFrame(rows).set_index("frame_id")
    years = [2022]
    alloc = si.even_allocate(binary, ["PA"], years, 5)
    r1, a1 = si.greedy_select(binary, ["PA"], alloc, years)
    r2, a2 = si.greedy_select(binary, ["PA"], alloc, years)
    check(a1["PA"] == 5 and a2["PA"] == 5, f"both hit target: {a1}, {a2}")
    check(len(set(r1)) == 5 and len(set(r2)) == 5, "5 distinct frames each run")

if __name__ == "__main__":
    for fn in (test_pick_best_is_random_among_ties, test_pick_best_none_when_no_positive, test_selection_varies_but_meets_target):
        run(fn)
    ok = sum(1 for _, p, _ in _R if p)
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) else 1)
