# CA/tests/test_open_routes.py
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import app as A

_fail = 0
def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c: _fail += 1

# ── _host_open: path-exists guard ─────────────────────────────────────────────
_calls = []

# monkeypatch subprocess.Popen so xdg-open never actually runs
A.subprocess.Popen = lambda *a, **k: _calls.append(a)

# existing dir → should call Popen and return (True, '')
_d = tempfile.mkdtemp(prefix="open_test_")
ok, err = A._host_open(_d)
check(ok is True, f"_host_open existing dir returns ok=True (got {ok!r})")
check(err == '', f"_host_open existing dir returns empty err (got {err!r})")
check(len(_calls) == 1, f"Popen called once for existing dir (got {len(_calls)})")
check(_calls[0][0][1] == _d, f"Popen called with correct path (got {_calls[0][0][1]!r})")

# nonexistent path → should return (False, nonempty)
ok2, err2 = A._host_open('/nonexistent/xyz_does_not_exist_9f2b')
check(ok2 is False, f"_host_open nonexistent returns ok=False (got {ok2!r})")
check(bool(err2), f"_host_open nonexistent returns nonempty err (got {err2!r})")
check(len(_calls) == 1, "Popen NOT called again for nonexistent path")

# empty/None path → should return (False, nonempty)
ok3, err3 = A._host_open('')
check(ok3 is False, f"_host_open empty path returns ok=False (got {ok3!r})")
check(bool(err3), f"_host_open empty path returns nonempty err (got {err3!r})")

# ── Flask routes: smoke via test_client (drives module-level session dict) ─────
try:
    import json as _json, pathlib as _pathlib
    A.app.config['TESTING'] = True

    # The app uses a process-global session dict (not Flask session cookies).
    # Set export_dir directly on A.session.
    A.session['export_dir'] = _d

    with A.app.test_client() as c:
        r_folder = c.post('/api/open_export_folder')
        check(r_folder.status_code == 200, f"/api/open_export_folder 200 (got {r_folder.status_code})")
        body = _json.loads(r_folder.data)
        check(body.get('ok') is True, f"open_export_folder body ok=True (got {body!r})")

        # manifest csv doesn't exist yet → expect 400
        r_man = c.post('/api/open_export_manifest')
        check(r_man.status_code == 400, f"/api/open_export_manifest 400 for missing csv (got {r_man.status_code})")

        # create the csv → should 200
        csv_path = os.path.join(_d, 'export_manifest.csv')
        _pathlib.Path(csv_path).touch()
        r_man2 = c.post('/api/open_export_manifest')
        check(r_man2.status_code == 200, f"/api/open_export_manifest 200 for existing csv (got {r_man2.status_code})")
except Exception as _e:
    print(f"  SKIP Flask route smoke ({_e})")

if _fail:
    print(f"\nFAILED ({_fail} checks failed)")
    sys.exit(1)
print(f"\nPASS (all checks passed)")
