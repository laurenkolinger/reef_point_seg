#!/usr/bin/env python
"""The Annotate Images initials boxes start on the signed-in initials.

No-pytest house harness. Run with:
  env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_initials_seed.py

Cuts the reviewer block out of src/templates/index.html (passedInitials,
upperAsTyped, paintReviewer, loadReviewer, setReviewer) and runs it in node
against a fake DOM and a fake /api/reviewer. Covers #reviewer-input and
#sa-initials:
  - the server's reviewer wins over a valid ?initials parameter
  - with no server reviewer a valid parameter fills both boxes and is
    pushed to /api/reviewer (so the accept guard is satisfied), upper-cased
  - A, ABCDE, L0, <b>, empty and an invalid first of a repeated parameter
    are ignored: nothing is filled and nothing is pushed
  - an sa-initials box the person already typed in is kept
  - the input handler rewrites lower case to upper case and keeps the caret
  - both inputs carry the oninput hook in the markup
Exits nonzero on any failure.
"""
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.abspath(os.path.join(_HERE, "..", "src", "templates", "index.html"))

START_MARK = "let _reviewerPushed = null;"
END_MARK = "// A server refusal for a missing identity."

NODE_WRAPPER = r"""
const fs = require('fs'), vm = require('vm');
const [search, server, html, typedText] = process.argv.slice(2);
const text = fs.readFileSync(html, 'utf8');
const a = text.indexOf(process.env.START_MARK), b = text.indexOf(process.env.END_MARK);
if (a < 0 || b < a) { console.error('markers not found'); process.exit(2); }
const block = text.slice(a, b);
const els = {};
const mk = (id) => ({id, value: '', selectionStart: 0, classList: {toggle() {}, add() {}},
  setSelectionRange(s) { this.selectionStart = s; }});
const posts = [];
const ctx = vm.createContext({
  document: {getElementById: (id) => els[id] || (els[id] = mk(id))},
  location: {search}, URLSearchParams, Promise, String, RegExp, JSON, Error,
  setStatus() {},
  fetch: (url, opts) => {
    if (opts && opts.method === 'POST') {
      const body = JSON.parse(opts.body); posts.push(body.reviewer);
      return Promise.resolve({json: () => Promise.resolve({reviewer: body.reviewer})});
    }
    return Promise.resolve({json: () => Promise.resolve({reviewer: server})});
  },
});
ctx.window = ctx;
vm.runInContext(block + '\nthis.loadReviewer = loadReviewer; this.upperAsTyped = upperAsTyped;', ctx);
const spec = JSON.parse(typedText);
if (spec.sa) ctx.document.getElementById('sa-initials').value = spec.sa;
ctx.loadReviewer().then(() => {
  const t = ctx.document.getElementById('reviewer-input');
  const out = {top: t.value, sa: ctx.document.getElementById('sa-initials').value, posts};
  t.value = spec.type || ''; t.selectionStart = spec.caret || 0;
  ctx.upperAsTyped(t);
  out.typed = t.value; out.caret = t.selectionStart;
  process.stdout.write(JSON.stringify(out) + '\n');
});
"""


def run(search, server="", sa="", typed="", caret=0):
    """Run the reviewer block for a query string and a server reviewer."""
    env = dict(os.environ, START_MARK=START_MARK, END_MARK=END_MARK)
    with tempfile.TemporaryDirectory() as tmp:
        wrapper = os.path.join(tmp, "wrap.js")
        with open(wrapper, "w", encoding="utf-8") as fh:
            fh.write(NODE_WRAPPER)
        r = subprocess.run(["node", wrapper, search, server, TEMPLATE,
                            json.dumps({"sa": sa, "type": typed, "caret": caret})],
                           capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"node failed: {r.stderr[:300]}")
    return json.loads(r.stdout)


_FAILURES = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        _FAILURES.append(msg)


def main():
    for search, want in [("?orchestrated=1&initials=AT", "AT"), ("?initials=at", "AT"),
                         ("?initials=%20ab%20", "AB")]:
        out = run(search)
        check(out["top"] == want and out["sa"] == want and out["posts"] == [want],
              f"{search}: both boxes start on {want} and the server is told: {out}")
    out = run("?initials=AT", server="XY")
    check(out["top"] == "XY" and out["sa"] == "" and out["posts"] == [],
          f"a server reviewer wins over the parameter and nothing is pushed: {out}")
    for search in ["", "?initials=A", "?initials=ABCDE", "?initials=L0", "?initials=%3Cb%3E",
                   "?initials=", "?initials=A&initials=AT"]:
        out = run(search)
        check(out["top"] == "" and out["sa"] == "" and out["posts"] == [],
              f"{search!r} is ignored: nothing filled, nothing pushed")
    out = run("?initials=AT", sa="JD")
    check(out["sa"] == "JD", "a value already typed in the session box is kept")
    out = run("", typed="abc", caret=2)
    check(out["typed"] == "ABC" and out["caret"] == 2, "typed 'abc' becomes ABC with the caret kept")
    html = open(TEMPLATE, encoding="utf-8").read()
    check(html.count('oninput="upperAsTyped(this)"') == 2, "both initials inputs carry the oninput hook")
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
