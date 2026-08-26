"""Step 3 output informativeness: AST timestamps + missing-reason helper.

No-pytest harness: run with env/bin/python <this file>.
"""
import os, sys, re
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
import select_images as si


def check(c, m):
    if not c:
        raise AssertionError(m)


def test_missing_reason_helper():
    check(si.missing_reason(found=True, basename='X', clip_dir='/c') == '',
          'found frame should have empty reason')
    r = si.missing_reason(found=False, basename='TCRMP20251023_clip_FLC_T307', clip_dir='/c')
    check('TCRMP20251023_clip_FLC_T307' in r and '/c' in r,
          f'reason should name basename + clip dir: {r}')


def test_ast_timestamp_format():
    ts = si.ast_timestamp()
    check(re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} AST$', ts),
          f'bad AST timestamp: {ts}')


def test_summary_has_eligibility_funnel():
    # build_summary_lines(...) is a new pure helper returning the funnel block.
    lines = si.build_summary_lines(
        candidates=100, eligible=70, selected=55, reserve=15,
        species_list=["PA"], achieved={"PA": 55}, target=55,
        exclusions={"image_missing": 2, "cpc_missing": 25, "pts_missing": 3})
    txt = "\n".join(lines)
    check("candidates" in txt and "eligible" in txt, "funnel labels present")
    check("PA" in txt and "55/55" in txt, "per-species achieved/target present")
    check("cpc_missing" in txt and "25" in txt, "exclusion reasons present")


if __name__ == '__main__':
    fails = []
    for fn in (test_missing_reason_helper, test_ast_timestamp_format,
               test_summary_has_eligibility_funnel):
        try:
            fn(); print(f'  PASS {fn.__name__}')
        except Exception as e:
            fails.append((fn.__name__, e)); print(f'  FAIL {fn.__name__}: {e}')
    print(f'\ntest_outputs_informative: {3 - len(fails)} passed, {len(fails)} failed')
    sys.exit(1 if fails else 0)
