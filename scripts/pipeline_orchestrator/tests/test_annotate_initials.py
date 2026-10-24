#!/usr/bin/env python
"""/annotate passes the signed-in initials on to the Annotate Images window.

No-pytest house harness. Run with:
  env/bin/python scripts/pipeline_orchestrator/tests/test_annotate_initials.py

Runs the inline script of templates/annotate.html in node against a fake DOM
whose fetch answers a healthy annotator on port 5096, then reads the URL the
page navigates to:
  - a valid ?initials parameter is appended (upper-cased) to the 5096 URL
  - A, ABCDE, L0, <b>, empty and a hostile value are dropped
  - the orchestrated=1 flag is always kept
Exits nonzero on any failure.
"""
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", ".."))
TEMPLATE = os.path.join(_SCRIPTS, "pipeline_orchestrator", "templates", "annotate.html")

# Wraps the driver: fetch answers the run and status routes, then the
# navigation target is printed after the page's own timers are flushed.
NODE_WRAPPER = r"""
const fs = require('fs'), vm = require('vm');
const search = process.argv[2];
const html = fs.readFileSync(process.argv[3], 'utf8');
const m = /<script>([\s\S]*?)<\/script>/.exec(html);
const stub = () => new Proxy(function () {}, {get: (t, k) => (k === 'then' ? undefined : stub()), set: () => true, apply: () => stub()});
const loc = {search, hostname: 'host', href: ''};
const ctx = vm.createContext({
  document: {getElementById: () => stub()}, location: loc, URLSearchParams, Promise, String, RegExp, Math,
  fetch: (u) => Promise.resolve({json: () => Promise.resolve(
    u.indexOf('/status') >= 0 ? {healthy: true} : {success: true, ui_port: 5096})}),
  setTimeout: (f) => 0,
});
ctx.window = ctx;
vm.runInContext(m[1], ctx);
setTimeout(() => { process.stdout.write(loc.href + '\n'); }, 50);
"""


def navigated(search):
    """The URL the page sends the browser to for a given query string."""
    with tempfile.TemporaryDirectory() as tmp:
        wrapper = os.path.join(tmp, "wrap.js")
        with open(wrapper, "w", encoding="utf-8") as fh:
            fh.write(NODE_WRAPPER)
        r = subprocess.run(["node", wrapper, search, TEMPLATE], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"node failed: {r.stderr[:300]}")
    return r.stdout.strip()


_FAILURES = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def main():
    base = "http://host:5096/?orchestrated=1"
    for search, want in [("?initials=AT", base + "&initials=AT"), ("?initials=at", base + "&initials=AT"),
                         ("?initials=%20ab%20", base + "&initials=AB")]:
        got = navigated(search)
        check(got == want, f"{search} -> {got}")
    for search in ["", "?initials=A", "?initials=ABCDE", "?initials=L0", "?initials=%3Cb%3E",
                   "?initials=", "?initials=A&initials=AT", "?initials=AT%26x%3D1"]:
        got = navigated(search)
        check(got == base, f"{search!r} is dropped -> {got}")
    check("initials" in open(TEMPLATE, encoding="utf-8").read(), "template reads the initials parameter")
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
