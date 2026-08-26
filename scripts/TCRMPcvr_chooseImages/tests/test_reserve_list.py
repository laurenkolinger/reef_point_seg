"""Reserve = eligible pool minus selected, per-species tagged, randomly ranked.
Run: env/bin/python scripts/TCRMPcvr_chooseImages/tests/test_reserve_list.py
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

def _pool():
    rows = [{"frame_id": f"f{i}", "year_int": 2022, "date": "2022-06-01",
             "site": "FLC", "transect": 1, "frame": i, "PA": 1} for i in range(10)]
    return pd.DataFrame(rows).set_index("frame_id")

def test_reserve_is_eligible_minus_selected():
    b = _pool()
    clip_index = {si.build_image_filename("2022-06-01","FLC",1,i): f"/c/{i}.jpg" for i in range(10)}
    selected = ["f0", "f1", "f2"]
    rows = si.build_reserve_rows(b, selected, ["PA"], si.build_image_filename, clip_index)
    fids = {r["frame_id"] for r in rows}
    check(fids == {f"f{i}" for i in range(3, 10)}, f"reserve wrong: {fids}")
    ranks = sorted(r["reserve_rank"] for r in rows)
    check(ranks == list(range(len(rows))), f"ranks not 0..N-1: {ranks}")
    check(all("PA" in r["species"] for r in rows), "species tag present")

if __name__ == "__main__":
    run(test_reserve_is_eligible_minus_selected)
    ok = sum(1 for _, p, _ in _R if p)
    print(f"\n{ok}/{len(_R)} passed")
    sys.exit(0 if ok == len(_R) else 1)
