"""Target-only view must never hide a tentative (pending) mask.

Regression guard for the "good mask disappears when 'Target points only' is
ticked" bug. In the MANUAL_ANNOTATE combined annotator every mask is operator-
placed; a freshly-placed mask is born status='pending' with no species yet, so
the target-species filter cannot validly classify it as non-target. Pending
masks are the operator's active, unclassified work and must ALWAYS draw (same
rule as REVIEW masks), otherwise they vanish yet still block new placement.

Extracts the real isReviewMask/isTargetMask/shouldDrawMask trio from the
template and evaluates it under node. No pytest; run with the unified env:
    env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_target_only_shows_pending.py
"""
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), 'src')
_TPL = os.path.join(_SRC, 'templates', 'index.html')

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


def _find_node():
    for cand in ('node', '/usr/bin/node'):
        try:
            subprocess.run([cand, '--version'], capture_output=True, check=True)
            return cand
        except Exception:
            continue
    return None


def _extract_visibility_trio():
    """Pull the three pure visibility functions verbatim from the template.

    They live contiguously; `function toggleTargetOnly` immediately follows the
    last one, so slicing between those two markers is robust to internal braces.
    """
    src = open(_TPL).read()
    start = src.index('function isReviewMask')
    end = src.index('function toggleTargetOnly')
    return src[start:end]


def test_target_only_always_shows_pending_masks():
    node = _find_node()
    if node is None:
        print('    (node unavailable — skipping)')
        return

    trio = _extract_visibility_trio()
    harness = """
let showOnlyTarget = true;              // "Target points only" ticked (default)
const TARGET_SPECIES = ['AL', 'SS'];    // stand-in target codes
%s
const cases = [
  // Unlabeled masks always show — whether still pending or locked in unlabeled.
  {name: 'pending, empty species (freshly placed)', m: {status: 'pending',  species: ''},                    expect: true},
  {name: 'accepted, empty species (locked in unlabeled)', m: {status: 'accepted', species: ''},              expect: true},
  {name: 'pending, UNK placeholder',                m: {status: 'pending',  species: 'UNK'},                 expect: true},
  {name: 'accepted, ? placeholder',                 m: {status: 'accepted', species: '?'},                   expect: true},
  {name: 'target species',                          m: {status: 'accepted', species: 'AL'},                  expect: true},
  {name: 'review-flagged mask',                     m: {status: 'pending',  species: 'REVIEW', review: true},expect: true},
  // Classified NON-target masks stay decluttered under the filter, whether the
  // operator accepted them as such or they came pending from the auto-SAM path.
  {name: 'accepted non-target mask',                m: {status: 'accepted', species: 'MACA'},                expect: false},
  {name: 'pending non-target (auto path)',          m: {status: 'pending',  species: 'MACA'},                expect: false},
];
let fails = 0;
for (const c of cases) {
  const got = !!shouldDrawMask(c.m);
  if (got !== c.expect) { console.log('FAIL ' + c.name + ' got=' + got + ' expect=' + c.expect); fails++; }
  else { console.log('PASS ' + c.name); }
}
// With the filter OFF, everything draws.
showOnlyTarget = false;
if (shouldDrawMask({status: 'accepted', species: 'MACA'}) !== true) { console.log('FAIL filter-off shows all'); fails++; }
else { console.log('PASS filter-off shows all'); }
process.exit(fails ? 1 : 0);
""" % trio

    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(harness)
        path = f.name
    try:
        res = subprocess.run([node, path], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            print('    ' + line)
        if res.stderr.strip():
            print('    [stderr] ' + res.stderr.strip())
        check(res.returncode == 0, 'target-only keeps tentative masks visible')
    finally:
        os.unlink(path)


if __name__ == '__main__':
    test_target_only_always_shows_pending_masks()
    print('PASS' if _fail == 0 else f'{_fail} FAILED')
    sys.exit(1 if _fail else 0)
