#!/usr/bin/env python
"""Theme contract check for the reef_point_seg windows rethemed on 2026-09-02.

The contract lives at
.superpowers/sdd/2026-09-01-reef-point-seg-quality-sweep/theme-contract.md:
one palette (VICARIUS near-black layers, T-Mobile magenta accent) and three
font roles (Inter prose, Syne display, Space Mono data and chrome). This suite
pins that contract for the windows in this batch:

  Label Manager           scripts/_labels (page render plus its stylesheet)
  Annotate boot page      scripts/pipeline_orchestrator/templates/annotate.html
  Add Expert IDs assets   scripts/_expertids (fragment CSS shipped on the
                          orchestrator page, so it must not clash)

The Train + Infer stylesheet was in this batch until the page was retired on
2026-09-05 (docs/adr/0002); its assertions went with it.

No-pytest house harness. Nothing binds a port; the Label Manager page is
rendered through the Flask test client with a paths_provider that returns an
empty dict, so no project file is read or written.

Run with:
  env/bin/python scripts/_labels/tests/test_theme_unify.py
"""

from __future__ import annotations

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

import sys
from pathlib import Path

THIS = Path(__file__).resolve()
SCRIPTS_DIR = THIS.parent.parent.parent  # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CACHE_BUST = "20260917-plain"

# Contract palette (name -> hex), every value verbatim from theme-contract.md.
CONTRACT_HEX = {
    "magenta": "#E20074",
    "magenta-dark": "#B8005C",
    "pink": "#FFB3D9",
    "bg": "#0a0a0f",
    "bg-deep": "#050508",
    "surface": "#11111a",
    "surface-light": "#191926",
    "input-bg": "#15151f",
    "text": "#F5EDF0",
    "text-dim": "#b0a7ab",
    "success": "#00FFB3",
    "warning": "#FFB800",
    "error": "#FF4D6D",
    "border": "#2a2a38",
    "border-dim": "#1d1d26",
}

# Hexes that must not survive anywhere in the rethemed windows: the old teal
# ground, the off-magenta and purple accents, and the stale greys.
BANNED_HEX = [
    "#0b1418", "#111e24", "#16252c", "#1f2f38", "#d7e3e8", "#7e96a0",
    "#ff5cc8", "#4bbf73", "#e0a23a", "#e0593a",
    "#12141a", "#c14bd6", "#e05555", "#3a3d4a", "#252833", "#9aa0ae",
    "#e8e8ea", "#0c0c14",
    "#e7eef2", "#93a4ad", "#36c275", "#e8a33d",
]

_fail = 0


def check(cond, msg):
    global _fail
    if cond:
        print(f"  ok   {msg}")
    else:
        _fail += 1
        print(f"  FAIL {msg}")


def read(rel):
    return (SCRIPTS_DIR / rel).read_text(encoding="utf-8")


def no_banned(text, where, allow=()):
    low = text.lower()
    for hexv in BANNED_HEX:
        if hexv in allow:
            continue
        check(hexv.lower() not in low, f"{where} has dropped {hexv}")


def no_dashes(text, where):
    check("\u2014" not in text and "\u2013" not in text,
          f"{where} has no em or en dashes")


# ── Label Manager ────────────────────────────────────────────────────────────

def test_label_manager():
    print("\n[label manager]")
    from flask import Flask

    from _labels import make_blueprint

    app = Flask("labels_theme_test")
    app.register_blueprint(make_blueprint(lambda: {}, log_fn=lambda m: None),
                           url_prefix="/labels")
    c = app.test_client()
    r = c.get("/labels/")
    html = r.get_data(as_text=True)
    check(r.status_code == 200, f"GET /labels/ is 200 (got {r.status_code})")

    for fam in ("family=Inter:wght@400;500;600;700",
                "family=Space+Mono:wght@400;700",
                "family=Syne:wght@600;700;800"):
        check(fam in html, f"page loads {fam.split('=')[1].split(':')[0]}")
    check("display=swap" in html, "the font request uses display=swap")
    check("fonts.gstatic.com" in html and "preconnect" in html,
          "the page preconnects to the font hosts")

    check(f"labels.css?v={CACHE_BUST}" in html,
          f"the stylesheet carries the {CACHE_BUST} cache-bust")
    check(f"labels.js?v={CACHE_BUST}" in html,
          f"the script carries the {CACHE_BUST} cache-bust")
    no_dashes(html, "label manager page")

    css = read("_labels/static/labels.css")
    for name in ("bg", "surface", "border", "text", "text-dim",
                 "magenta", "success", "warning", "error"):
        check(CONTRACT_HEX[name] in css,
              f"labels.css carries the contract {name} value {CONTRACT_HEX[name]}")
    no_banned(css, "labels.css")
    check("255, 92, 200" not in css, "labels.css has dropped the old accent rgba")
    check("rgba(4, 9, 11" not in css, "labels.css modal scrim is no longer teal-black")

    for token in ("--font-body:", "--font-display:", "--font-mono:"):
        check(token in css, f"labels.css declares {token.rstrip(':')}")
    check("'Inter'" in css and "'Syne'" in css and "'Space Mono'" in css,
          "labels.css names all three families in its font tokens")
    check("font-family: inherit" not in css,
          "labels.css has no font-family: inherit (prose can no longer resolve to mono)")
    check("font-family: var(--font-body)" in css,
          "labels.css sets the page body to the body font")
    check(css.count("var(--font-display)") >= 3,
          f"display font reaches the title, section heads and card heads "
          f"(got {css.count('var(--font-display)')} uses)")
    check(css.count("var(--font-mono)") >= 10,
          f"mono font reaches buttons, inputs, tables and micro-labels "
          f"(got {css.count('var(--font-mono)')} uses)")
    no_dashes(css, "labels.css")

    # The modal is display: flex, which beats the browser default for [hidden],
    # so the closed state has to be stated or the dialog and its scrim sit over
    # the page from first paint. Caught by the 1280 wide screenshot.
    check(".lb-modal[hidden]" in css,
          "the confirm dialog is hidden until the script opens it")

    js = read("_labels/static/labels.js")
    check("toLocaleString()" not in js,
          "the footer stamp does not use the locale 12 hour clock")
    check("stamp24(" in js and "(AST)" in js,
          "the footer stamp is 24 hour and names AST")


# ── Annotate boot page ───────────────────────────────────────────────────────

def test_annotate_boot():
    print("\n[annotate boot page]")
    html = read("pipeline_orchestrator/templates/annotate.html")
    check("family=Inter" in html, "the boot page loads Inter")
    check("'Inter'" in html, "the boot page sets Inter as the body font")
    check("system-ui" not in html, "the boot page no longer falls back to system-ui first")
    for name in ("bg", "surface", "magenta", "error", "border"):
        check(CONTRACT_HEX[name] in html,
              f"boot page carries the contract {name} value {CONTRACT_HEX[name]}")
    no_banned(html, "annotate.html")
    no_dashes(html, "annotate.html")
    check("title=" in html, "the retry control keeps its tooltip")


# ── Add Expert IDs fragment assets ───────────────────────────────────────────

def test_expertids():
    print("\n[add expert ids assets]")
    css = read("_expertids/static/expertids.css")
    check("'Inter'" in css, "expertids.css uses Inter for its fragment prose")
    check("'Syne'" in css, "expertids.css uses Syne for its step headings")
    check("'Space Mono'" in css, "expertids.css uses Space Mono for codes and paths")
    check("-apple-system, BlinkMacSystemFont" not in css,
          "expertids.css no longer leads with the system sans stack")
    for name in ("text", "text-dim", "success", "warning"):
        check(CONTRACT_HEX[name] in css,
              f"expertids.css standalone fallback uses the contract {name} value")
    no_banned(css, "expertids.css")
    no_dashes(css, "expertids.css")

    idx = read("_expertids/templates/expertids/index.html")
    check(CONTRACT_HEX["bg"] in idx, "the expert IDs standalone page uses the contract background")
    for fam in ("family=Inter", "family=Space+Mono", "family=Syne"):
        check(fam in idx, f"the standalone page requests {fam.split('=')[1]}")
    no_banned(idx, "expertids/index.html")
    no_dashes(idx, "expertids/index.html")


def main():
    print("theme contract, 2026-09-02 unify batch")
    test_label_manager()
    test_annotate_boot()
    test_expertids()
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
