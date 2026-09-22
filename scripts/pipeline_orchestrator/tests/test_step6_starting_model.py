"""
The Train panel's starting-model rebuild (2026-09-21): the template half of
[docs/superpowers/plans/2026-09-21-vega-starting-model-and-training-data.md]
(Task B), built against
[docs/superpowers/specs/2026-09-21-vega-starting-model-and-training-data-design.md].

No pytest. Run with the unified env python:
    env/bin/python scripts/pipeline_orchestrator/tests/test_step6_starting_model.py

Covers the template only (the JS half is Task C1's test_step6_continue_from_js.py,
the server half is Task A's test_step6_source_summary.py):

  RENDER    templates/index.html compiles through Jinja2 with StrictUndefined
            and the app's own render context, proving the reorganised panel
            parses.
  DATA      the Training data section (id s6-section-data) sits first inside
            panel 6, before the Starting model and Run settings sections.
  MODEL     the Starting model select (#s6-model) offers only the five YOLO11
            sizes per task plus the YOLO26 group, medium preselected for
            outlines, and carries no "Continue from a previous run" optgroup.
  CONTINUE  #s6-continue-from sits inside #s6-continue-details with a blank
            first option; #s6-freeze sits inside the same disclosure.
  PRESETS   the two built-in preset buttons (s6-preset-retrain,
            s6-preset-finetune) are gone.
  RUNS CARD the trained-runs include sits once, inside panel 7's
            #s7-all-models disclosure, and nowhere in panel 6.
  REFINE    the Refine after-export card's two buttons carry the renamed ids
            and onclick handlers.
  SECTIONS  the three section headings (data, model, run) appear in that
            order and each carries a rule-compliant tooltip.
  TOOLTIPS  every title= on panel 6 (and the two new panel-7 elements) is one
            or two complete sentences, never opens with "You", never
            references the screen, and ends with a full stop.
"""

import os
import re
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORCH = os.path.dirname(_HERE)
_TPL = os.path.join(_ORCH, "templates", "index.html")
_FRAGMENT = os.path.join(_ORCH, "templates", "_step6_runs.html")

sys.path.insert(0, _ORCH)

EM_DASH = "—"
EN_DASH = "–"
BANNED_WORD = re.compile(r"\bland(s|ed|ing)?\b", re.IGNORECASE)
SCREEN_REFERENCE = re.compile(r"\b(above|below|this tab|this window)\b", re.IGNORECASE)
# Split on a sentence terminator FOLLOWED BY WHITESPACE (or end of string), not
# on every literal period: a naive [^.!?]+[.!?] scan miscounts any tooltip
# that carries a decimal number ("0.02 to 0.05"), because each embedded "."
# reads as its own sentence boundary. This is the same approach
# test_step6_task.py's test_subject_select_sits_beside_the_task_control uses.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_RESULTS = []
_fail = 0


def check(cond, msg):
    """Raise AssertionError with ``msg`` when ``cond`` is false."""
    if not cond:
        raise AssertionError(msg)


def run(fn):
    """Run one named test, record PASS or FAIL, never stop the suite."""
    global _fail
    try:
        fn()
        _RESULTS.append((fn.__name__, True, ""))
        print(f"  PASS {fn.__name__}")
    except Exception as e:  # noqa: BLE001 - the harness reports every failure kind
        _fail += 1
        _RESULTS.append((fn.__name__, False, f"{e}\n{traceback.format_exc()}"))
        print(f"  FAIL {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_CACHE = {}


def _read(path):
    """Return the text of ``path``, cached, failing loudly when it is missing."""
    if path not in _CACHE:
        check(os.path.isfile(path), f"missing file: {path}")
        with open(path, encoding="utf-8") as fh:
            _CACHE[path] = fh.read()
    return _CACHE[path]


def _render():
    """Render templates/index.html through Jinja2 with StrictUndefined and the
    same stand-in context app.py's index() passes (no project loaded), proving
    the reorganised panel compiles. Cached across tests."""
    if "html" not in _CACHE:
        import jinja2

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(_ORCH, "templates")),
            undefined=jinja2.StrictUndefined,
            autoescape=False,
        )
        env.globals["url_for"] = lambda endpoint, **kw: f"/stub/{endpoint}/{kw.get('filename', '')}"
        tpl = env.get_template("index.html")
        _CACHE["html"] = tpl.render(
            asset_version="test-step6-starting-model",
            repo_root="/stub/repo",
            projects_dir="/stub/projects",
            supporting_data_dir="/stub/supporting_data",
            all_points_csv="/stub/all_points.csv",
            master_codes_csv="/stub/master_codes.csv",
            project_loaded=False,
            project_state={},
        )
    return _CACHE["html"]


def _panel(html, key):
    """The markup of one step panel, from its opening tag to the next panel's."""
    start = html.index(f'id="panel-{key}"')
    nxt = html.find('class="step-panel"', start + 1)
    return html[start:nxt if nxt > 0 else len(html)]


def _tag_with(html, marker, from_idx=0):
    """The whole opening tag that carries ``marker`` (an id= or the like)."""
    idx = html.index(marker, from_idx)
    start = html.rfind("<", 0, idx)
    end = html.index(">", idx)
    return html[start:end + 1]


def _titles_in(text):
    """Every title="..." value in ``text``, unescaped for the quote entities
    Jinja's autoescape (off here, but kept honest) would otherwise leave in."""
    found = re.findall(r'title="([^"]*)"', text)
    return [t.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&") for t in found]


def _assert_tooltip_rules(title, where):
    """The platform tooltip sentence rules (design section 7) for one title."""
    check(title.strip(), f"{where}: empty tooltip")
    check(not title.startswith("You"), f"{where}: tooltip opens with You: {title!r}")
    check(title.rstrip().endswith("."), f"{where}: tooltip does not end with a full stop: {title!r}")
    n = len([s for s in _SENTENCE_SPLIT.split(title.strip()) if s])
    check(1 <= n <= 2, f"{where}: tooltip has {n} sentences, want one or two: {title!r}")
    check(not SCREEN_REFERENCE.search(title), f"{where}: tooltip references the screen: {title!r}")
    check(EM_DASH not in title and EN_DASH not in title, f"{where}: tooltip carries a dash: {title!r}")


# ---------------------------------------------------------------------------
# RENDER
# ---------------------------------------------------------------------------

def test_template_renders_through_jinja():
    html = _render()
    check(len(html) > 10000, "rendered page looks truncated")
    check('id="panel-6"' in html and 'id="panel-7"' in html, "panels 6 and 7 must render")


def test_fragment_appears_exactly_once():
    tpl = _read(_TPL)
    check(tpl.count("{% include '_step6_runs.html' %}") == 1,
          "the trained-runs fragment must be included exactly once")
    html = _render()
    check(html.count('id="s6-runs-card"') == 1, "the rendered runs card must appear exactly once")


# ---------------------------------------------------------------------------
# DATA: the Training data section
# ---------------------------------------------------------------------------

def test_sections_appear_in_order_data_model_run():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    for sid in ("s6-section-data", "s6-section-model", "s6-section-run"):
        check(f'id="{sid}"' in panel, f"section heading {sid} missing from panel 6")
    data_at = panel.index('id="s6-section-data"')
    model_at = panel.index('id="s6-section-model"')
    run_at = panel.index('id="s6-section-run"')
    check(data_at < model_at < run_at,
          f"sections must read data, model, run (got offsets {data_at}, {model_at}, {run_at})")
    # Every heading carries the section-head shape used elsewhere (the Rounds
    # heading): field-label section-label text plus a sec() info glyph.
    for sid, label in (("s6-section-data", "Training data"),
                        ("s6-section-model", "Starting model"),
                        ("s6-section-run", "Run settings")):
        idx = panel.index(f'id="{sid}"')
        # class="section-head" precedes id= on the same opening tag, so the
        # window must start a little before it, not at it.
        block = panel[max(0, idx - 60):idx + 600]
        check('class="section-head"' in block, f"{sid} must carry class section-head: {block!r}")
        check(f'class="field-label section-label">{label}<' in block,
              f"{sid} must carry the visible label {label!r}")
        # This slices the raw (unrendered) template, so the sec() macro call
        # itself is what proves the info glyph is there (its rendered form,
        # class="tip", is proved separately by the tooltip-rule tests below,
        # which read the JINJA-RENDERED page).
        check("{{ sec(" in block, f"{sid} must carry an info glyph ({{{{ sec(...) }}}})")


def test_source_card_sits_inside_the_data_section_before_the_model_section():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    data_at = panel.index('id="s6-section-data"')
    model_at = panel.index('id="s6-section-model"')
    source_at = panel.index('id="s6-source-card"')
    classes_at = panel.index('id="s6-classes-card"')
    pictures_at = panel.index('id="s6-data-pictures"')
    split_at = panel.index('id="s6-data-split"')
    check(data_at < source_at < pictures_at < classes_at < split_at < model_at,
          "Training data must read: source, pictures, classes, split, all before Starting model "
          f"(got {data_at}, {source_at}, {pictures_at}, {classes_at}, {split_at}, {model_at})")
    # The holdout row (pin-split + the two fractions + min samples) also
    # belongs to Training data, after the split line.
    holdout_at = panel.index('id="s6-pin-split-group"')
    check(split_at < holdout_at < model_at,
          "the holdout row must sit after the split line and before Starting model")
    for gid in ("s6-valid-group", "s6-test-group", "s6-min-samples-group"):
        check(split_at < panel.index(f'id="{gid}"') < model_at,
              f"{gid} must sit in the Training data section")


def test_composite_banner_and_bbox_note_still_exist_once_each():
    """Folded into the Training source row (spec section 3.1), but the two
    ids the host JS fills must survive exactly once each, so
    applyCompositeTrainingMode()/applyStep6BoxDatasetMode() keep working."""
    tpl = _read(_TPL)
    check(tpl.count('id="s6-composite-banner"') == 1, "s6-composite-banner must appear exactly once")
    check(tpl.count('id="s6-bbox-source-note"') == 1, "s6-bbox-source-note must appear exactly once")
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    source_at = panel.index('id="s6-source-card"')
    banner_at = panel.index('id="s6-composite-banner"')
    note_at = panel.index('id="s6-bbox-source-note"')
    check(source_at < banner_at, "the composite banner must sit inside or after the source card")
    check(source_at < note_at, "the box-dataset note must sit at or after the source card")
    # The box note must NOT be nested inside s6-source-card, or hiding the
    # card for a box-dataset project (applyStep6BoxDatasetMode) would hide
    # the one line that names the dataset too.
    card_close = panel.index("</div>", source_at)
    # Walk to the matching close of the source-card div (depth-matched, since
    # the card itself nests a form-group div).
    depth, cursor = 0, source_at
    while True:
        nxt_open = panel.find("<div", cursor + 1)
        nxt_close = panel.find("</div>", cursor + 1)
        if 0 <= nxt_open < nxt_close:
            depth += 1
            cursor = nxt_open
            continue
        depth -= 1
        cursor = nxt_close
        if depth < 0:
            card_close = nxt_close
            break
    check(note_at > card_close,
          "s6-bbox-source-note must sit AFTER the source card closes, not nested inside it, "
          "so it stays visible when applyStep6BoxDatasetMode() hides the card")


def test_classes_count_span_present_in_the_heading_row():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check('id="s6-classes-count"' in panel, "the class picker heading row needs a tick count span")
    classes_at = panel.index('id="s6-classes-card"')
    count_at = panel.index('id="s6-classes-count"')
    header_close = panel.index("</div>", panel.index('class="field-head"', classes_at))
    check(classes_at < count_at < header_close,
          "s6-classes-count must sit inside the class picker's own header row")
    tag = _tag_with(panel, 'id="s6-classes-count"')
    check("ticked" in panel[count_at:count_at + 80], "the count span must read like \"N of M ticked\"")


def test_pictures_and_split_lines_present():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    for cid, label in (("s6-data-pictures", "Pictures"), ("s6-data-split", "Split")):
        check(f'id="{cid}"' in panel, f"{cid} missing")
        row = panel[max(0, panel.index(f'id="{cid}"') - 250):panel.index(f'id="{cid}"')]
        check(f">{label}<" in row, f"{cid} must sit under a {label!r} label")


# ---------------------------------------------------------------------------
# MODEL: the Starting model select
# ---------------------------------------------------------------------------

_OUTLINE_VALUES = ("yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt")
_BOX_VALUES = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
_YOLO26_VALUES = ("yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt")


def _model_select_block(panel):
    start = panel.index('<select id="s6-model"')
    end = panel.index("</select>", start) + len("</select>")
    return panel[start:end]


def test_starting_model_offers_only_yolo11_and_yolo26_no_runs_optgroup():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check(panel.count('id="s6-model"') == 1, "exactly one Starting model control")
    check('id="s6-model-runs-optgroup"' not in panel, "the previous-run optgroup must be gone")
    check("Continue from a previous run" not in panel,
          "the retired \"Continue from a previous run\" optgroup label must be gone")

    block = _model_select_block(panel)
    seg = block[block.index('id="s6-model-seg-optgroup"'):block.index("</optgroup>", block.index('id="s6-model-seg-optgroup"'))]
    det = block[block.index('id="s6-model-detect-optgroup"'):block.index("</optgroup>", block.index('id="s6-model-detect-optgroup"'))]
    y26 = block[block.index('id="s6-model-seg26-optgroup"'):block.index("</optgroup>", block.index('id="s6-model-seg26-optgroup"'))]

    for values, group, task_word in ((_OUTLINE_VALUES, seg, "outlines"), (_BOX_VALUES, det, "boxes")):
        found = re.findall(r'value="([a-z0-9_.-]+)"', group)
        check(tuple(found) == values, f"{task_word} group must offer exactly {values}, got {tuple(found)}")
        # The closed select shows about 27 characters at 946 px (the Device
        # select documents the same budget); "YOLO11 medium, outlines
        # (recommended)" clipped there on 2026-09-22, so the speed and
        # strength words sit on the group label and every option stays short.
        texts = re.findall(r'<option value="[a-z0-9_.-]+"[^>]*>([^<]*)</option>', group)
        check(len(texts) == len(values), f"every {task_word} option needs visible text: {texts}")
        for text in texts:
            check(len(text) <= 27, f"option text longer than the 946 px select budget: {text!r}")
            check(text.startswith("YOLO11 ") and text.endswith(f", {task_word}"),
                  f"option text must read 'YOLO11 <size>, {task_word}': {text!r}")
        check(f"YOLO11 medium, {task_word}" in texts, f"the medium {task_word} option is missing: {texts}")
        # The label attribute precedes the id in the tag, so read it from the
        # whole select rather than from the slice that starts at the id.
        gid = "s6-model-seg-optgroup" if task_word == "outlines" else "s6-model-detect-optgroup"
        tag = re.search(rf'<optgroup label="([^"]*)" id="{gid}"', block)
        label = tag.group(1) if tag else ""
        check("medium is recommended" in label and "nano is fastest" in label and "x-large is strongest" in label,
              f"the {task_word} group label must carry the speed, strength and recommendation words: {label!r}")

    check(re.search(r'value="yolo11m-seg\.pt"\s+selected', seg) is not None,
          "YOLO11 medium, outlines must be preselected")
    check("selected" not in det, "no detection option is statically selected (the fallback is JS-driven)")

    found26 = re.findall(r'value="([a-z0-9_.-]+)"', y26)
    check(tuple(found26) == _YOLO26_VALUES, f"the YOLO26 group must keep its five values, got {tuple(found26)}")
    check('id="s6-model-seg26-optgroup"' in block and "YOLO26, outlines" in block,
          "the YOLO26 group must be labelled per the spec")

    label_block = panel[panel.index('for="s6-model"'):panel.index('for="s6-model"') + 60]
    check(">Starting model<" in label_block, "the field label must read Starting model, not Base model")
    check(">Base model<" not in panel, "the retired label Base model must not survive on panel 6")


def test_model_hint_line_and_note_present():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check('id="s6-model-note"' in panel, "the always-visible hint line under Starting model is missing")
    note = panel[panel.index('id="s6-model-note"'):panel.index("</div>", panel.index('id="s6-model-note"'))]
    check("COCO" not in note and "everyday photographs" in note,
          f"the hint names what the base model was trained on: {note!r}")
    check("training data above" in note, f"the hint points at the training data section: {note!r}")


# ---------------------------------------------------------------------------
# CONTINUE: the Continue from a VEGA model disclosure
# ---------------------------------------------------------------------------

def _details_block(panel, details_id):
    """The exact text of one <details id="..."> ... </details> block, matched
    by nesting depth so an inner <details> (there are none here, but the
    pattern is shared with other tests in this suite) never confuses the
    close tag."""
    open_at = panel.index(f'<details class="advanced" id="{details_id}"')
    depth, cursor, close_at = 0, open_at, -1
    while cursor < len(panel):
        nxt_open = panel.find("<details", cursor + 1)
        nxt_close = panel.find("</details>", cursor + 1)
        if nxt_close < 0:
            break
        if 0 <= nxt_open < nxt_close:
            depth += 1
            cursor = nxt_open
            continue
        if depth == 0:
            close_at = nxt_close + len("</details>")
            break
        depth -= 1
        cursor = nxt_close
    check(close_at > open_at, f"{details_id}: no matching </details> found")
    return panel[open_at:close_at]


def test_continue_from_select_and_freeze_sit_inside_the_disclosure():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check('id="s6-continue-details"' in panel, "the Continue from a VEGA model disclosure is missing")
    block = _details_block(panel, "s6-continue-details")
    check('summary id="s6-continue-summary"' in block, "the disclosure summary needs its contract id")
    check(">Continue from a VEGA model (advanced)<" in block, "the summary text must match the contract")
    check('id="s6-continue-from"' in block, "s6-continue-from must sit inside the disclosure")
    check('id="s6-freeze"' in block, "s6-freeze must sit inside the disclosure")
    check('id="s6-continue-note"' in block, "the freeze/learning-rate note must sit inside the disclosure")

    select = block[block.index('<select id="s6-continue-from"'):block.index("</select>", block.index('<select id="s6-continue-from"'))]
    first_option = re.search(r'<option value="([^"]*)">([^<]*)</option>', select)
    check(first_option is not None, "the continue-from select needs at least one option")
    check(first_option.group(1) == "", f"the first option's value must be blank, got {first_option.group(1)!r}")
    check(first_option.group(2) == "None: start from the starting model above",
          f"the first option's text must match the spec, got {first_option.group(2)!r}")
    # Found on the live page 2026-09-22 00:15 AST: with no handler on the
    # select, choosing a run changed nothing on the panel (Freeze layers, the
    # learning rate, the disabled starting model and its hint all stayed put).
    check('onchange="onStep6ContinueChange()"' in select,
          "the continue-from select must call onStep6ContinueChange() on change")

    note = block[block.index('id="s6-continue-note"'):block.index("</div>", block.index('id="s6-continue-note"'))]
    check("Freeze layers to 10" in note and "0.001" in note,
          f"the note must state the auto freeze/learning-rate behaviour: {note!r}")
    check("puts both back when the choice is cleared" in note, f"the note must describe clearing: {note!r}")


def test_freeze_input_attributes_unchanged():
    tpl = _read(_TPL)
    check(tpl.count('id="s6-freeze"') == 1, "exactly one Freeze layers control")
    m = re.search(r'<input type="number" id="s6-freeze" value="" min="0" placeholder="none"', tpl)
    check(m is not None, "the Freeze layers input must keep its original attributes")


def test_starts_from_details_holds_the_three_glossary_lines():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    check('id="s6-starts-from-details"' in panel, "the What the model starts from disclosure is missing")
    block = _details_block(panel, "s6-starts-from-details")
    check(">What the model starts from<" in block, "the summary text must match the contract")
    for term in ("<strong>Base model.</strong>", "<strong>Train from a base model.</strong>",
                 "<strong>Continue from a VEGA model.</strong>"):
        check(term in block, f"the disclosure must define {term!r}")
    check("COCO" in block, "the disclosure must say Ultralytics trained the base model on COCO")


def test_the_two_disclosures_sit_between_model_select_and_run_settings():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    model_at = panel.index('<select id="s6-model"')
    starts_at = panel.index('id="s6-starts-from-details"')
    continue_at = panel.index('id="s6-continue-details"')
    run_at = panel.index('id="s6-section-run"')
    check(model_at < starts_at < continue_at < run_at,
          f"order must be model select, starts-from, continue, Run settings "
          f"(got {model_at}, {starts_at}, {continue_at}, {run_at})")


# ---------------------------------------------------------------------------
# PRESETS: the two built-in buttons are gone
# ---------------------------------------------------------------------------

def test_builtin_preset_buttons_are_gone():
    tpl = _read(_TPL)
    for gone in ('id="s6-preset-retrain"', 'id="s6-preset-finetune"',
                 "applyStep6BuiltinPreset", "Full retrain preset", "Fine-tune preset"):
        check(gone not in tpl, f"retired preset control {gone!r} must be gone from the template")
    # The rest of the preset loader (Load preset, Refresh, Import, applied
    # line, Why this recipe) survives untouched.
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    for survivor in ('id="s6-preset-select"', 'onclick="refreshStep6Presets()"',
                      'onclick="importStep6Preset()"', 'id="s6-preset-applied"'):
        check(survivor in panel, f"the preset loader must keep {survivor}")


def test_run_settings_holds_run_name_epochs_device_and_preset_loader():
    tpl = _read(_TPL)
    panel = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    run_at = panel.index('id="s6-section-run"')
    for cid in ("s6-run-name", "s6-epochs", "s6-imgsz", "s6-batch", "s6-patience",
                "s6-device-preset", "s6-preset-select"):
        check(run_at < panel.index(f'id="{cid}"'),
              f"{cid} must sit inside Run settings, after id=s6-section-run")
    name_at = panel.index('id="s6-run-name"')
    epochs_at = panel.index('id="s6-epochs"')
    patience_at = panel.index('id="s6-patience"')
    device_at = panel.index('id="s6-device-preset"')
    preset_at = panel.index('id="s6-preset-select"')
    check(name_at < epochs_at < patience_at < device_at < preset_at,
          "Run settings must read: run name, epochs row (with patience), device, load preset")


# ---------------------------------------------------------------------------
# RUNS CARD: moved to panel 7, inside a closed disclosure
# ---------------------------------------------------------------------------

def test_runs_include_lives_only_in_panel_7_inside_s7_all_models():
    tpl = _read(_TPL)
    panel6 = tpl[tpl.index('id="panel-6"'):tpl.index('id="panel-7"')]
    panel7 = tpl[tpl.index('id="panel-7"'):tpl.index('id="panel-8"')]
    check("_step6_runs.html" not in panel6, "the runs fragment must not be included on panel 6")
    check("_step6_runs.html" in panel7, "the runs fragment must be included on panel 7")
    check('id="s7-all-models"' in panel7, "the All trained models disclosure is missing from panel 7")

    all_models_at = panel7.index('id="s7-all-models"')
    include_at = panel7.index("_step6_runs.html")
    rounds_at = panel7.index('id="s7-rounds-block"')
    footer_at = panel7.index('id="nav-footer-7"')
    check(rounds_at < all_models_at < include_at < footer_at,
          "s7-all-models must sit after the Rounds ledger and before the nav footer, "
          f"and hold the include (got rounds={rounds_at}, disclosure={all_models_at}, "
          f"include={include_at}, footer={footer_at})")
    summary_tag = _tag_with(panel7, 'id="s7-all-models-summary"')
    check('title="' in summary_tag, "the s7-all-models summary needs a tooltip")
    check(">All trained models<" in panel7[panel7.index('id="s7-all-models-summary"'):
                                            panel7.index('id="s7-all-models-summary"') + 200],
          "the disclosure summary must read All trained models")


def test_fragment_header_comment_records_the_move():
    src = _read(_FRAGMENT)
    check("2026-09-21" in src, "the fragment's header comment must record the 2026-09-21 move")
    check("Evaluate" in src, "the fragment's header comment must name the Evaluate panel context")


def test_fragment_title_and_glyph_name_the_evaluate_context():
    src = _read(_FRAGMENT)
    check(">All trained models<" in src, "the fragment heading must read All trained models")
    glyph = re.search(r'class="vega-runs-tip"\s+title="([^"]*)"', src)
    check(glyph is not None, "the section glyph is missing")
    tip = glyph.group(1)
    _assert_tooltip_rules(tip, "fragment glyph")
    check("evaluated" in tip.lower() or "promoted" in tip.lower() or "continued from" in tip.lower(),
          f"the glyph must say what this card is for (evaluate/promote/continue from): {tip!r}")
    check("fine-tun" not in tip.lower(), f"the glyph must not use the retired word fine-tune: {tip!r}")


# ---------------------------------------------------------------------------
# REFINE: the after-export card
# ---------------------------------------------------------------------------

def test_refine_card_buttons_renamed():
    tpl = _read(_TPL)
    for gone in ('id="s4loop-goto-retrain"', 'id="s4loop-goto-finetune"', "gotoStep6Preset"):
        check(gone not in tpl, f"retired Refine-card control {gone!r} must be gone")
    base_tag = _tag_with(tpl, 'id="s4loop-goto-base"')
    champion_tag = _tag_with(tpl, 'id="s4loop-goto-champion"')
    check("onclick=\"gotoStep6Start('base')\"" in base_tag, f"got {base_tag!r}")
    check("onclick=\"gotoStep6Start('champion')\"" in champion_tag, f"got {champion_tag!r}")
    for tag in (base_tag, champion_tag):
        check('title="' in tag, f"Refine-card button missing a tooltip: {tag!r}")
    base_end = tpl.index("</button>", tpl.index('id="s4loop-goto-base"')) + len("</button>")
    champion_end = tpl.index("</button>", tpl.index('id="s4loop-goto-champion"')) + len("</button>")
    base_label = tpl[tpl.index('id="s4loop-goto-base"'):base_end]
    champion_label = tpl[tpl.index('id="s4loop-goto-champion"'):champion_end]
    check(">Train a new model</button>" in base_label, f"got {base_label!r}")
    check(">Continue from the champion</button>" in champion_label, f"got {champion_label!r}")


# ---------------------------------------------------------------------------
# ROUNDS HEADER
# ---------------------------------------------------------------------------

def test_rounds_header_reads_started_from():
    tpl = _read(_TPL)
    panel7 = tpl[tpl.index('id="panel-7"'):tpl.index('id="panel-8"')]
    check(">Started from<" in panel7, "the Rounds table header must read Started from")
    check(">Base<" not in panel7, "the retired header text Base must be gone from panel 7")


# ---------------------------------------------------------------------------
# TOOLTIPS: every title= on panel 6, plus the two new panel-7 elements
# ---------------------------------------------------------------------------

# Every title= this task wrote from scratch or reworded. A blanket sweep of
# every title on panel 6 would also grade tooltips this task left untouched
# (the augmentation and loss-weight subcards, carried over verbatim per the
# plan's "keep every surviving control's tooltip text"), which is not this
# task's mandate ("every NEW OR CHANGED title obeys the tooltip rules").
_NEW_OR_CHANGED_TOOLTIP_MARKERS = (
    'for="s6-model">Starting model',       # the Starting model select's ?
    'id="s6-continue-summary"',            # Continue from a VEGA model disclosure summary
    'for="s6-continue-from"',              # the Continue from select's ?
    'for="s6-freeze"',                     # the reworded Freeze layers ?
)


def test_every_new_or_changed_tooltip_follows_the_sentence_rules():
    html = _render()
    panel = _panel(html, 6)
    checked = 0
    for marker in _NEW_OR_CHANGED_TOOLTIP_MARKERS:
        idx = panel.index(marker)
        tag = _tag_with(panel, "title=", from_idx=idx)
        m = re.search(r'title="([^"]*)"', tag)
        check(m is not None, f"no title= found near {marker!r}")
        _assert_tooltip_rules(m.group(1), marker)
        checked += 1
    check(checked == len(_NEW_OR_CHANGED_TOOLTIP_MARKERS), "every marker must resolve to one tooltip")

    # The two new disclosure summaries (What the model starts from, and the
    # section glyphs) also carry new tooltips.
    for marker in ('id="s6-starts-from-details"', 'id="s6-section-data"',
                    'id="s6-section-model"', 'id="s6-section-run"'):
        idx = panel.index(marker)
        tag = _tag_with(panel, "title=", from_idx=idx)
        m = re.search(r'title="([^"]*)"', tag)
        check(m is not None, f"no title= found near {marker!r}")
        _assert_tooltip_rules(m.group(1), marker)

    # And the Refine card's two renamed buttons.
    for bid in ("s4loop-goto-base", "s4loop-goto-champion"):
        tag = _tag_with(_read(_TPL), f'id="{bid}"')
        m = re.search(r'title="([^"]*)"', tag)
        check(m is not None, f"{bid} needs a title=")
        _assert_tooltip_rules(m.group(1), bid)


def test_new_panel_7_tooltips_follow_the_sentence_rules():
    html = _render()
    panel7 = _panel(html, 7)
    for marker in ('id="s7-all-models-summary"',):
        tag = _tag_with(panel7, marker)
        m = re.search(r'title="([^"]*)"', tag)
        check(m is not None, f"{marker} needs a title=")
        _assert_tooltip_rules(m.group(1), marker)


def test_no_banned_prose_on_panel_6_or_the_fragment():
    html = _render()
    panel = _panel(html, 6)
    for text, where in ((panel, "panel 6"), (_read(_FRAGMENT), "_step6_runs.html")):
        check(EM_DASH not in text, f"{where}: em dash present")
        check(EN_DASH not in text, f"{where}: en dash present")
        hit = BANNED_WORD.search(text)
        check(hit is None, f"{where}: banned word {hit.group(0) if hit else ''!r}")
        check("UVI" not in text, f"{where}: the string UVI is present")


if __name__ == "__main__":
    print("RENDER:")
    run(test_template_renders_through_jinja)
    run(test_fragment_appears_exactly_once)
    print("DATA:")
    run(test_sections_appear_in_order_data_model_run)
    run(test_source_card_sits_inside_the_data_section_before_the_model_section)
    run(test_composite_banner_and_bbox_note_still_exist_once_each)
    run(test_classes_count_span_present_in_the_heading_row)
    run(test_pictures_and_split_lines_present)
    print("MODEL:")
    run(test_starting_model_offers_only_yolo11_and_yolo26_no_runs_optgroup)
    run(test_model_hint_line_and_note_present)
    print("CONTINUE:")
    run(test_continue_from_select_and_freeze_sit_inside_the_disclosure)
    run(test_freeze_input_attributes_unchanged)
    run(test_starts_from_details_holds_the_three_glossary_lines)
    run(test_the_two_disclosures_sit_between_model_select_and_run_settings)
    print("PRESETS:")
    run(test_builtin_preset_buttons_are_gone)
    run(test_run_settings_holds_run_name_epochs_device_and_preset_loader)
    print("RUNS CARD:")
    run(test_runs_include_lives_only_in_panel_7_inside_s7_all_models)
    run(test_fragment_header_comment_records_the_move)
    run(test_fragment_title_and_glyph_name_the_evaluate_context)
    print("REFINE:")
    run(test_refine_card_buttons_renamed)
    print("ROUNDS HEADER:")
    run(test_rounds_header_reads_started_from)
    print("TOOLTIPS:")
    run(test_every_new_or_changed_tooltip_follows_the_sentence_rules)
    run(test_new_panel_7_tooltips_follow_the_sentence_rules)
    run(test_no_banned_prose_on_panel_6_or_the_fragment)

    print(f"\n{len(_RESULTS) - _fail}/{len(_RESULTS)} passed.")
    if _fail:
        print("\nFAILURES:")
        for name, ok, detail in _RESULTS:
            if not ok:
                print(f"--- {name} ---\n{detail}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
