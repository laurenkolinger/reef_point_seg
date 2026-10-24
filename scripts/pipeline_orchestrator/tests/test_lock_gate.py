"""Module-lock gate (VICARIUS platform contract, Task B1): while the lock
registry marks reef_point_seg locked, a browser navigation to any page of
this module serves the minimal vic-lock-page (never the SPA, never the
?project_dir= auto-open), and every API call refuses with 423 JSON, whether
it is under /api/ or under a dashboard's own /api/ prefix; /static assets
stay served.
Fail-open: a missing, empty, or corrupt registry means unlocked (one stderr
warning per distinct error), and VICARIUS_LOCK_BYPASS=1 forces unlocked.
Run with github_repo/env/bin/python.
"""
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

import contextlib
import io
import json
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

MSG = "Under development. Please come back later. Email Lauren for questions."

tmp = tempfile.mkdtemp(prefix="lock_gate_")
LOCKED = os.path.join(tmp, "locked.json")
UNLOCKED = os.path.join(tmp, "unlocked.json")
CORRUPT = os.path.join(tmp, "corrupt.json")
MISSING = os.path.join(tmp, "missing.json")
with open(LOCKED, "w") as fh:
    json.dump({"schema_version": 1,
               "locks": {"reef_point_seg":
                         {"locked": True, "by": "LO", "ts": "2026-08-16 12:00 AST"}}}, fh)
with open(UNLOCKED, "w") as fh:
    json.dump({"schema_version": 1, "locks": {"reef_point_seg": {"locked": False}}}, fh)
with open(CORRUPT, "wb") as fh:
    fh.write(b"not json{")

# Point the reader at the tmp registry BEFORE the app exists; is_locked() reads
# the env fresh on every call, so one app instance serves every registry state.
os.environ["VICARIUS_LOCKS_PATH"] = LOCKED
os.environ.pop("VICARIUS_LOCK_BYPASS", None)

import lock_status
import app as appmod

client = appmod.create_app().test_client()

# j. the constant itself (everything below depends on the exact copy)
check(lock_status.LOCK_MESSAGE == MSG, "LOCK_MESSAGE is the exact canonical copy")

# a. locked: GET / is the lock page, not the SPA
r = client.get("/")
body = r.get_data(as_text=True)
check(r.status_code == 200, f"locked GET / is 200 (got {r.status_code})")
check("vic-lock-page" in body, "locked GET / carries the vic-lock-page marker")
check("Under development. Please come back later." in body, "lock page carries the message")
check('href="mailto:lauren.olinger@uvi.edu"' in body, "Email Lauren is a mailto anchor")
check(">Email Lauren</a> for questions." in body, "anchor text is exactly 'Email Lauren'")
check("Reef Point Seg" in body, "lock page names the module")
check("VICARIUS module lock" in body, "muted footer line present")
check("s3-label-chips" not in body, "locked GET / never serves the SPA")

# b. locked: ?project_dir= auto-open never executes
r = client.get("/?project_dir=/tmp/nonexistent")
body = r.get_data(as_text=True)
check("vic-lock-page" in body and "s3-label-chips" not in body,
      "locked GET /?project_dir=... is still the lock page")
check(appmod.current_project is None, "current_project stays None (no auto-open while locked)")

# c. locked: POST /api/project/open refuses with 423 JSON
r = client.post("/api/project/open", json={"project_dir": "/tmp/nonexistent"})
check(r.status_code == 423, f"locked POST /api/project/open is 423 (got {r.status_code})")
j = r.get_json(silent=True) or {}
check(j.get("error") == "locked", f"423 body error is 'locked' (got {j.get('error')!r})")
check(j.get("message") == lock_status.LOCK_MESSAGE, "423 body message == LOCK_MESSAGE")

# d. locked: a person navigating to any dashboard mounted beside the main
# window gets the same lock page the contract requires, and the data routes
# behind those dashboards refuse with 423. Before 2026-09-16 the pages
# themselves answered a navigation with raw JSON.
for page in ("/labels/", "/matrix/", "/expertids/", "/boxes/", "/annotate"):
    r = client.get(page, headers={"Accept": "text/html,application/xhtml+xml"})
    check(r.status_code == 200 and b"vic-lock-page" in r.data,
          f"locked navigation to {page} is the lock page (got {r.status_code})")
for api in ("/labels/api/vocabulary", "/matrix/api/data", "/expertids/api/status",
            "/boxes/api/datasets"):
    r = client.get(api, headers={"Accept": "text/html,application/xhtml+xml"})
    check(r.status_code == 423, f"locked GET {api} is 423 (got {r.status_code})")
    body = r.get_json(silent=True) or {}
    check(body.get("error") == "locked", f"{api} 423 body error is 'locked' (got {body})")
# A fetch that asks for JSON is code, not a navigation, whatever the path.
r = client.get("/matrix/", headers={"Accept": "application/json"})
check(r.status_code == 423, f"locked JSON fetch of /matrix/ is 423 (got {r.status_code})")
# A POST is never a navigation.
r = client.post("/matrix/api/prefer", json={})
check(r.status_code == 423, f"locked POST /matrix/api/prefer is 423 (got {r.status_code})")

# e. locked: static assets stay served (the lock page may style itself)
r = client.get("/static/style.css")
check(r.status_code == 200, f"locked GET /static/style.css is 200 (got {r.status_code})")

# f. unlocked entry serves the normal SPA index
os.environ["VICARIUS_LOCKS_PATH"] = UNLOCKED
r = client.get("/")
check("s3-label-chips" in r.get_data(as_text=True),
      "unlocked entry ({'locked': false}) serves the normal index")

# g. missing registry file is fail-open
os.environ["VICARIUS_LOCKS_PATH"] = MISSING
r = client.get("/")
check("s3-label-chips" in r.get_data(as_text=True),
      "missing registry file serves the normal index (fail-open)")

# h. corrupt registry is fail-open, warning once per distinct error
os.environ["VICARIUS_LOCKS_PATH"] = CORRUPT
err = io.StringIO()
with contextlib.redirect_stderr(err):
    first = lock_status.is_locked()
    second = lock_status.is_locked()
check(first is False and second is False, "corrupt registry: is_locked() is False")
warnings = [ln for ln in err.getvalue().splitlines() if ln.strip()]
check(len(warnings) == 1, f"corrupt registry warns exactly once (got {len(warnings)})")
r = client.get("/")
check("s3-label-chips" in r.get_data(as_text=True),
      "corrupt registry serves the normal index (fail-open)")

# i. VICARIUS_LOCK_BYPASS=1 forces unlocked while the registry says locked
os.environ["VICARIUS_LOCKS_PATH"] = LOCKED
os.environ["VICARIUS_LOCK_BYPASS"] = "1"
r = client.get("/")
check("s3-label-chips" in r.get_data(as_text=True),
      "VICARIUS_LOCK_BYPASS=1 serves the normal index despite a locked registry")
del os.environ["VICARIUS_LOCK_BYPASS"]
r = client.get("/")
check("vic-lock-page" in r.get_data(as_text=True),
      "lock re-engages once the bypass env var is removed")


# k. backend runs pass the lock (Lauren, 2026-09-28): this machine plus the key
os.environ["VICARIUS_LOCKS_PATH"] = LOCKED
os.environ["VEGA_BACKEND_TOKEN_PATH"] = os.path.join(tmp, "cfg", "vega_backend_token")
key = lock_status.backend_token(create=True)
mode = os.stat(os.environ["VEGA_BACKEND_TOKEN_PATH"]).st_mode & 0o777
check(len(key) == 64 and mode == 0o600, f"a 64-character key is created owner-only (mode {oct(mode)})")
check(lock_status.backend_token(create=True) == key, "an existing key is reused, not replaced")
H = lock_status.BACKEND_HEADER
r = client.get("/api/project/state", headers={H: key})
check(r.status_code != 423, f"locked: a local call with the key passes (got {r.status_code})")
r = client.get("/api/project/state", headers={H: "0" * 64})
check(r.status_code == 423, f"locked: a wrong key is refused (got {r.status_code})")
r = client.get("/api/project/state")
check(r.status_code == 423, f"locked: no key is refused (got {r.status_code})")
r = client.get("/api/project/state", headers={H: key},
               environ_base={"REMOTE_ADDR": "10.0.0.7"})
check(r.status_code == 423, f"locked: the key from another machine is refused (got {r.status_code})")
r = client.get("/", headers={"Accept": "text/html"})
check("vic-lock-page" in r.get_data(as_text=True), "locked: a browser still gets the lock page")
with open(os.environ["VEGA_BACKEND_TOKEN_PATH"], "w") as fh:
    fh.write("")
r = client.get("/api/project/state", headers={H: ""})
check(r.status_code == 423, "locked: an empty key file never matches an empty header")
os.environ.pop("VEGA_BACKEND_TOKEN_PATH", None)
del os.environ["VICARIUS_LOCKS_PATH"]

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
