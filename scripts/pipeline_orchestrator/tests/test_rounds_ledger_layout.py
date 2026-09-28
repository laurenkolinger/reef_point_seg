"""
The Rounds ledger holds its Promote column at 946 and 1280 px without a
sideways scroll: its Gates chips wrap between words and never inside one, a
run name breaks at its underscores, the Base path is cut with an ellipsis
and carried whole on its tooltip, and the family chip reads at the table's
chip size with no inline size of its own.

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_rounds_ledger_layout.py

Drives the SHIPPED scripts/pipeline_orchestrator/static/orchestrator.js in the
Node stand-in that test_rail_family.py builds (imported, never copied). The
rendered pixels are the pixel review's business; this suite pins the markup
those pixels come from, and how hostile ledger values reach it.

Covers:
  CHIPS     a Gates chip allows wrapping, balances its lines, and keeps its
            words contiguous so "per class: 2 flagged" is one phrase; the
            Gates block carries the floor that keeps every word whole.
  NAMES     a run name gets a break opportunity after each underscore and
            nowhere else, keeps its whole text on the cell's tooltip, and a
            hostile name is escaped before the breaks go in.
  BASE      a path base shows its last three parts cut with an ellipsis, its
            width a share of the window, the whole path on the tooltip.
  SCORES    the family chip carries no inline font size and sits under its
            number; only the mAP cell carries the header's floor.
  PROMOTE   the action is the small button with the trimmed sides, the
            champion row a chip.
  SOURCE    no 10 px size is written anywhere in the ledger renderer.
"""

import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_rail_family as rail  # noqa: E402

_RESULTS = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run(fn):
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - a suite reports, never crashes
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


_LIVE_ROWS = """
[{ round: 1, run_name: 'finalverify_detect_e12', base_model: 'yolo11n.pt', metric_family: 'box',
   map50_95_B: 0.098, recall_B: null, gate_map: 'unpinned', gate_class: 'unpinned', gate_recall: 'unpinned' },
 { round: 2, run_name: 'TRAIN_coral_intel_20260902_224223', base_model: 'yolo11m.pt', metric_family: 'box',
   map50_95_B: 0.535, recall_B: 0.655, conf: 0.001, gate_map: 'pass', gate_class: 'flag:PA,SS', gate_recall: 'pass' },
 { round: 3, run_name: 'TRAIN_coral_intel_20260903_150509',
   base_model: '/x/runs/TRAIN_coral_intel_20260902_224223/weights/best.pt', metric_family: 'box',
   map50_95_B: 0.535, recall_B: 0.644, conf: 0.001, gate_map: 'pass', gate_class: 'flag:DL', gate_recall: 'pass' }]
"""
_HOSTILE_ROWS = """
[{ round: 1, run_name: '<img src=x onerror=alert(1)>_x', base_model: '"quoted"/<b>/weights/best.pt',
   metric_family: 'box', map50_95_B: 0.1, gate_map: 'pass', gate_class: 'pass', gate_recall: 'pass' }]
"""


def _cells(html):
    """Every <td ...>...</td> of the rendered rows, as (attributes, inner)."""
    return re.findall(r"<td([^>]*)>(.*?)</td>", html, flags=re.S)


def _render(rows):
    return rail._drive(rail._rounds_server(rows) + "roundsHtml();")


# ── CHIPS ───────────────────────────────────────────────────────────
def test_a_gates_chip_wraps_between_words_and_keeps_its_phrase_whole():
    out = _render(_LIVE_ROWS)
    chips = re.findall(r'<span class="badge badge-[a-z]+" style="([^"]*)" title="[^"]*">([^<]*)</span>', out)
    gate_chips = [(style, text) for style, text in chips if ": " in text]
    check(len(gate_chips) == 9, f"three gates per row, three rows: {len(gate_chips)}")
    for style, text in gate_chips:
        check("white-space:nowrap" not in style, f"a chip no longer refuses to wrap: {text}")
        check("white-space:normal" in style and "text-wrap:balance" in style,
              f"a chip wraps on balanced lines: {style}")
        check("max-width:100%" in style, f"a chip never overflows its cell: {style}")
    texts = [text for _, text in gate_chips]
    check("per class: 2 flagged" in texts and "mAP: split not pinned" in texts,
          f"the plain phrases are contiguous text, not split across markup: {texts}")


def test_the_gates_block_carries_the_word_floor():
    out = _render(_LIVE_ROWS)
    blocks = re.findall(r'<div style="display:flex;flex-direction:column;[^"]*min-width:([0-9a-z.]+)"', out)
    check(len(blocks) == 3, f"one gates block per row: {blocks}")
    check(all(b == "96px" for b in blocks), f"the floor is the named constant: {blocks}")


# ── NAMES ───────────────────────────────────────────────────────────
def test_a_run_name_breaks_after_its_underscores_only():
    out = _render(_LIVE_ROWS)
    check("TRAIN_<wbr>coral_<wbr>intel_<wbr>20260902_<wbr>224223" in out,
          f"a break opportunity follows each underscore: {out[:600]}")
    check('title="TRAIN_coral_intel_20260902_224223"' in out, "the whole name is on the tooltip")
    check("finalverify_<wbr>detect_<wbr>e12" in out, "a short name gets the same treatment")
    check(out.count("<wbr>") == 10, f"no break goes anywhere else: {out.count('<wbr>')}")
    check(re.search(r'<span style="display:inline-block;min-width:72px;overflow-wrap:anywhere">TRAIN_', out),
          "the name sits in a block with the floor a minted part fits")


def test_a_hostile_run_name_is_escaped_before_the_breaks_go_in():
    out = _render(_HOSTILE_ROWS)
    check("<img" not in out, f"no markup from the ledger reaches the page: {out[:400]}")
    check("&lt;img src=x onerror=alert(1)&gt;_<wbr>x" in out,
          f"the name is escaped and still breaks at its underscore: {out[:400]}")
    check('title="&lt;img src=x onerror=alert(1)&gt;_x"' in out, "the tooltip holds the escaped name")


# ── BASE ────────────────────────────────────────────────────────────
def test_a_path_base_shows_its_run_folder_whole_and_carries_the_path():
    """2026-09-17: the Base cell shows the one token that identifies the base (the
    run folder above weights/best.pt, else the weights file) and wraps at that
    token; it never cuts a word with an ellipsis, at any width."""
    out = _render(_LIVE_ROWS)
    base_cells = [(a, i) for a, i in _cells(out) if 'overflow-wrap:anywhere">' in i and "min-width" not in i]
    check(len(base_cells) == 3, f"one Base cell per row: {len(base_cells)}")
    attrs, inner = base_cells[2]
    check("text-overflow" not in attrs and "nowrap" not in attrs, f"the cell never cuts a word: {attrs}")
    check(inner.endswith("TRAIN_coral_intel_20260902_224223</span>"),
          f"a path base shows the run folder that owns the weights: {inner}")
    check('title="/x/runs/TRAIN_coral_intel_20260902_224223/weights/best.pt"' in attrs,
          f"the whole path is on the tooltip: {attrs}")
    check(base_cells[0][1].endswith("yolo11n.pt</span>"), f"a stock file shows as it is: {base_cells[0][1]}")


def test_a_hostile_base_is_escaped_in_the_cell_and_the_tooltip():
    out = _render(_HOSTILE_ROWS)
    attrs, inner = [(a, i) for a, i in _cells(out) if 'overflow-wrap:anywhere">' in i and "min-width" not in i][0]
    check("<b>" not in inner and "&lt;b&gt;" in inner, f"the cell is escaped: {inner}")
    check('title="&quot;quoted&quot;/&lt;b&gt;/weights/best.pt"' in attrs, f"the tooltip is escaped: {attrs}")


# ── SCORES ──────────────────────────────────────────────────────────
def test_the_family_chip_has_no_inline_size_and_sits_under_its_number():
    out = _render(_LIVE_ROWS)
    chips = re.findall(r'<span class="badge badge-info" style="([^"]*)" title="Scored on boxes\."', out)
    check(len(chips) == 6, f"two family chips per row: {len(chips)}")
    check(all("font-size" not in style for style in chips),
          f"no chip sets its own size; the stylesheet's 12 px stands: {chips}")
    check(all(style == "margin-top:2px" for style in chips), f"the chip sits under the number: {chips}")
    check("0.535<br><span class=\"badge badge-info\"" in out, "the number and the chip are on their own lines")
    check("n/a<br><span" in out, "a missing recall still carries its family")


def test_only_the_map_cell_carries_the_header_floor():
    out = _render(_LIVE_ROWS)
    floors = re.findall(r'<span style="display:inline-block;min-width:71px">', out)
    check(len(floors) == 3, f"one floor per row, on the mAP cell: {len(floors)}")
    recall_cells = [i for a, i in _cells(out) if "text-align:right" in a and "min-width" not in i
                    and "Scored on" in i]
    check(len(recall_cells) == 3, f"the Recall cell carries none: {len(recall_cells)}")


# ── PROMOTE ─────────────────────────────────────────────────────────
def test_promote_is_the_small_button_with_trimmed_sides():
    out = _render(_LIVE_ROWS)
    buttons = re.findall(r'<button class="btn btn-outline btn-sm" style="([^"]*)"', out)
    check(len(buttons) == 3, f"every row without a champion offers Promote: {len(buttons)}")
    check(all(b == "padding-left:10px;padding-right:10px" for b in buttons), f"the trim is the named constant: {buttons}")


def test_the_champion_row_shows_a_chip_instead():
    """The row whose run is the champion carries the chip; every other row
    offers Promote."""
    rows = _LIVE_ROWS.replace("run_name: 'finalverify_detect_e12',", "run_name: 'finalverify_detect_e12', run_dir: '/x/runs/r1',") \
        .replace("run_name: 'TRAIN_coral_intel_20260902_224223',", "run_name: 'TRAIN_coral_intel_20260902_224223', run_dir: '/x/runs/r2',") \
        .replace("run_name: 'TRAIN_coral_intel_20260903_150509',", "run_name: 'TRAIN_coral_intel_20260903_150509', run_dir: '/x/runs/r3',")
    out = rail._drive(rail._SETTLE + """
fetch = (url, opts) => {
  const answer = (body) => Promise.resolve({ status: 200, json: () => Promise.resolve(body) });
  if (url.indexOf('/labels/api/vocabulary') >= 0) return answer({ rows: [] });
  if (url.indexOf('/api/step/7/rounds') >= 0) return answer({ rounds: %s, champion: { run_dir: '/x/runs/r3', metric_family: 'box', map50_95_B: 0.535 } });
  return answer({});
};
(async () => { await refreshStep7Rounds(); await settle(); return document.getElementById('s7-rounds-tbody').innerHTML; })();
""" % rows)
    check(out.count("Promote</button>") == 2, f"the two rows that are not the champion offer Promote: {out.count('Promote</button>')}")
    check(out.count('title="This run is the current champion.">champion</span>') == 1, "the champion row carries the chip")


# ── SOURCE ──────────────────────────────────────────────────────────
def test_no_ten_pixel_size_is_written_in_the_ledger_renderer():
    js = open(rail._JS, encoding="utf-8").read()
    region = js[js.index("const ROUNDS_FAMILY_COLUMNS"):js.index("function confirmPromoteRun(")]
    check("font-size:10px" not in region and "font-size: 10px" not in region,
          "the 10 px chip size is gone from the renderer")
    for name in ("ROUNDS_MAP_MIN_WIDTH", "ROUNDS_RUN_MIN_WIDTH",
                 "ROUNDS_GATES_MIN_WIDTH", "ROUNDS_PROMOTE_PADDING", "GATE_CHIP_STYLE"):
        check(f"const {name} = " in js, f"{name} is a named constant")


if __name__ == "__main__":
    print("CHIPS:")
    run(test_a_gates_chip_wraps_between_words_and_keeps_its_phrase_whole)
    run(test_the_gates_block_carries_the_word_floor)
    print("NAMES:")
    run(test_a_run_name_breaks_after_its_underscores_only)
    run(test_a_hostile_run_name_is_escaped_before_the_breaks_go_in)
    print("BASE:")
    run(test_a_path_base_shows_its_run_folder_whole_and_carries_the_path)
    run(test_a_hostile_base_is_escaped_in_the_cell_and_the_tooltip)
    print("SCORES:")
    run(test_the_family_chip_has_no_inline_size_and_sits_under_its_number)
    run(test_only_the_map_cell_carries_the_header_floor)
    print("PROMOTE:")
    run(test_promote_is_the_small_button_with_trimmed_sides)
    run(test_the_champion_row_shows_a_chip_instead)
    print("SOURCE:")
    run(test_no_ten_pixel_size_is_written_in_the_ledger_renderer)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print(f"\n{passed}/{total} passed")
    for name, ok, err in _RESULTS:
        if not ok:
            print(f"\n--- {name} ---\n{err}")
    sys.exit(0 if passed == total else 1)
