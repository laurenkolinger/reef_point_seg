"""Standalone mode: /api/standalone/start creates a session dir under
TCRMP_STANDALONE_ROOT, ingests images, stores labels, builds the queue; the
template renders the single card + Import button only in standalone mode.
Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_standalone_session.py
"""
import os, sys, io, json, tempfile, atexit, shutil
from PIL import Image
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, SRC)
ROOT = tempfile.mkdtemp(prefix="standalone_test_")
atexit.register(shutil.rmtree, ROOT, ignore_errors=True)
os.environ['TCRMP_STANDALONE_ROOT'] = ROOT
os.environ['TCRMP_SESSION_MODE'] = 'standalone'
os.environ.setdefault('TCRMP_TARGET_SPECIES', 'PA'); os.environ.setdefault('TCRMP_MANUAL_ANNOTATE', '1')
mc = os.path.join(ROOT, "master_codes.csv")
open(mc, "w").write("species_code,species_name,category\nPA,Porites astreoides,Coral\nMC,Montastraea cavernosa,Coral\nSS,Siderastrea siderea,Coral\n")
os.environ['TCRMP_MASTER_CODES'] = mc
import app as A
_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1
client = A.app.test_client()
html = client.get('/').data.decode()
# template: Task 8
check('id="standalone-card"' in html and 'id="btn-import-images"' in html, "standalone mode renders the single card + Import")
# template: Task 8
check('id="mode-new"' not in html, "two-mode startup card hidden in standalone mode")
j = client.get('/api/standalone/master_codes').get_json()
check([c["code"] for c in j["codes"]] == ["PA", "MC", "SS"], "master codes served for the label picker")
def img_bytes():
    b = io.BytesIO(); Image.new("RGB", (64, 48), (1, 2, 3)).save(b, "JPEG"); b.seek(0); return b
# Initials are required (the session has no project to inherit them from);
# test_reviewer_required covers the refusal when they are missing.
r = client.post('/api/standalone/start', data={"session": "26-08-26_LO", "labels": "MC,SS",
    "reviewer": "LO",
    "images": [(img_bytes(), "one.jpg"), (img_bytes(), "two.jpg")]}, content_type='multipart/form-data')
j = r.get_json()
check(r.status_code == 200 and j.get("ok") and len(j["imported"]) == 2, f"start ingests two images, got {j}")
check(j["session_dir"] == os.path.join(ROOT, "26-08-26_LO") and os.path.isdir(os.path.join(j["session_dir"], "custom_imports", "raw")), "session dir created under root")
check(j["labels"] == ["MC", "SS"] and A.session["target_labels"][0]["code"] == "MC" and A.session["target_labels"][0]["name"].startswith("Montastraea"), "labels stored with names")
check(A.session["configured"] and j["review_count"] == 2, "queue built")
s = client.get('/api/status').get_json()
check(s["session_mode"] == "standalone" and [l["code"] for l in s["target_labels"]] == ["MC", "SS"], "status exposes mode + labels")
check(os.path.isfile(os.path.join(j["session_dir"], "session.json")), "session.json written")
r = client.post('/api/standalone/start', data={"session": "../evil", "labels": "MC", "reviewer": "LO"}, content_type='multipart/form-data')
check(r.status_code == 400, "path traversal in session name is 400")
r = client.post('/api/standalone/start', data={"session": ".", "labels": "MC", "reviewer": "LO"}, content_type='multipart/form-data')
check(r.status_code == 400, "a session name of only dots is 400")
# project mode never shows the card
os.environ['TCRMP_SESSION_MODE'] = 'configure'
import importlib, config as C; importlib.reload(C); A.cfg = C
html = client.get('/').data.decode()
check('id="standalone-card"' not in html and 'id="btn-import-images"' not in html, "project mode: no card, no Import button")
print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
