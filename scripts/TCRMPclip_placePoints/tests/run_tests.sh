#!/usr/bin/env bash
# Run the Place Points test suite: pure-logic (Node) + Flask launch/render (Python).
# Usage: scripts/TCRMPclip_placePoints/tests/run_tests.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"            # github_repo root
export REPO                                     # the python heredoc reads os.environ['REPO'] — must be exported so it works from ANY CWD
PY="$REPO/env/bin/python"; [ -x "$PY" ] || PY=python3

echo "== node --check on both app templates (Jinja stripped) =="
"$PY" - <<'PYEOF'
import re, subprocess, tempfile, os, sys
REPO=os.environ.get('REPO') or os.getcwd()
def check(rel,label):
    p=os.path.join(REPO,rel); html=open(p,encoding='utf-8').read()
    blocks=re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>',html,re.S)
    js="\n;\n".join(blocks)
    js=re.sub(r'\{%.*?%\}','',js,flags=re.S); js=re.sub(r'\{\{.*?\}\}','0',js,flags=re.S); js=re.sub(r'\{#.*?#\}','',js,flags=re.S)
    t=tempfile.NamedTemporaryFile('w',suffix='.js',delete=False,encoding='utf-8'); t.write(js); t.close()
    r=subprocess.run(['node','--check',t.name],capture_output=True,text=True); os.unlink(t.name)
    print(("  JS OK  : " if r.returncode==0 else "  JS FAIL: ")+label+("" if r.returncode==0 else "\n"+r.stderr))
    sys.exit(r.returncode) if r.returncode else None
check('scripts/TCRMPclip_placePoints/src/templates/index.html','place_points')
check('scripts/TCRMPclip_segmentImages/src/templates/index.html','segment_images')
PYEOF

echo "== py_compile backends =="
"$PY" -m py_compile \
  "$REPO/scripts/TCRMPclip_placePoints/src/app.py" \
  "$REPO/scripts/TCRMPclip_placePoints/src/config.py" \
  "$REPO/scripts/TCRMPclip_segmentImages/src/app.py" \
  "$REPO/scripts/TCRMPclip_segmentImages/src/config.py"
echo "  PY OK"

echo "== pure-logic unit tests (Node) =="
node "$HERE/test_pp_core.js"

echo "== Flask launch / render verification =="
"$PY" "$HERE/verify_apps.py"

echo "ALL TESTS PASSED"
