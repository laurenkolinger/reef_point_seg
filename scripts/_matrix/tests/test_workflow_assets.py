#!/usr/bin/env python3
"""Shared workflow assets: the stylesheet and tooltip script every VICARIUS
window links (spec 2026-09-02-workflow-page-design.md, sections 2, 5, 7)."""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
GR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSS = os.path.join(GR, "scripts", "pipeline_orchestrator", "static", "vicarius_workflow.css")
JS = os.path.join(GR, "scripts", "pipeline_orchestrator", "static", "vicarius_tooltips.js")
FAILS = []
COUNT = 0
def check(cond, msg):
    global COUNT
    COUNT += 1
    print(("[PASS] " if cond else "[FAIL] ") + msg)
    if not cond: FAILS.append(msg)

css = open(CSS, encoding="utf-8").read() if os.path.exists(CSS) else ""
js = open(JS, encoding="utf-8").read() if os.path.exists(JS) else ""
check(bool(css), "vicarius_workflow.css exists")
check(bool(js), "vicarius_tooltips.js exists")
TOKENS = {"--magenta": "#E20074", "--magenta-dark": "#B8005C", "--pink": "#FFB3D9",
          "--pink-soft": "#FFE6F2", "--bg": "#0a0a0f", "--bg-deep": "#050508",
          "--surface": "#11111a", "--surface-light": "#191926", "--input-bg": "#15151f",
          "--text": "#F5EDF0", "--text-dim": "#b0a7ab", "--success": "#00FFB3",
          "--warning": "#FFB800", "--error": "#FF4D6D", "--border": "#2a2a38",
          "--border-dim": "#1d1d26", "--radius": "2px", "--radius-lg": "3px"}
for name, val in TOKENS.items():
    check(re.search(r"%s\s*:\s*%s\s*;" % (re.escape(name), re.escape(val)), css) is not None,
          "token %s is %s" % (name, val))
for fam in ("'Inter'", "'Syne'", "'Space Mono'"):
    check(fam in css, "font family %s declared" % fam)
for cls in (".page", ".head", ".title", ".tabs", ".tab", ".tip", ".bubble", ".stages", ".rail",
            ".stage", ".node", ".stage-name", ".stage-meta", ".pane", ".pane-head", ".choices",
            ".opt", ".opts", ".dd-list", ".id", ".count", ".mini-actions", ".btn", ".btn-primary",
            ".btn-mini", ".btn-danger", ".actions", ".result", ".rows", ".row", ".row-name",
            ".row-meta", ".row-state", ".row-act", "details.src", ".util", ".util-body", ".foot"):
    check(re.search(r"(^|[\s,}])" + re.escape(cls) + r"[\s,{:.\[>]", css, re.M) is not None,
          "class %s defined" % cls)
check(".btn:disabled" in css, "disabled buttons get their own state (I2, 2026-09-03 whole-batch review)")
check(".result.result-error" in css, "error variant of the result line defined")
check(".result:empty" in css, "empty result line hides its border and background")
check("prefers-reduced-motion" in css, "reduced motion respected")
check(":focus-visible" in css, "focus ring defined")
check("—" not in css and "—" not in js, "no em dash in shared assets")
check("window.VicariusTooltips" in js, "tooltip script exposes window.VicariusTooltips")
check("focusin" in js and "mouseover" in js and "Escape" in js, "bubble opens on hover and focus, Escape closes")
check("data-tip" in js and "removeAttribute('title')" in js, "title is parked in data-tip while the bubble is open")
check("aria-describedby" in js, "bubble is linked with aria-describedby")
check("vicarius-bubble" in js, "bubble carries a stable id")
print("%d checks, %d failed" % (COUNT, len(FAILS)))
sys.exit(1 if FAILS else 0)
