"""Module-lock probe for the reef_point_seg orchestrator.

This file is the VICARIUS platform reference implementation for standalone
module web apps: a module app that serves its own UI (rather than being
rendered inside the desktop UI) copies this file, adjusts MODULE_SLUG, and
registers a before_request gate on it (see app.py create_app()).

Lock state lives in one platform registry, written ONLY by vicarius_ui_os
(the desktop UI); this module never writes it. Locked blocks everyone,
including admin sessions: the desktop UI's admin (dev) mode gates the toggle,
it is never a bypass for use. Fail-open by design: any problem reading the
registry means unlocked, because locks are the exception and a bad state file
must not brick the module.

Stdlib only, so any module app can embed it with zero dependencies.
"""

import json
import os
import sys
from pathlib import Path

MODULE_SLUG = "reef_point_seg"
LOCK_MESSAGE = "Under development. Please come back later. Email Lauren for questions."
# Default registry location derives from VICARIUS_ROOT (same env the run
# scripts honor) so an off-box install only sets one variable; the literal
# fallback is this box's platform root. VICARIUS_LOCKS_PATH (read at call
# time in is_locked) still overrides the whole path for tests.
LOCKS_PATH = os.path.join(
    os.environ.get("VICARIUS_ROOT") or "/mnt/rip/vicarius_drive/vicarius",
    "_METADATA", "module_locks.json")

# One stderr line per distinct read problem, so a bad registry does not spam
# every request.
_warned = set()


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(f"[lock_status] warning: {msg}; treating {MODULE_SLUG} as unlocked",
              file=sys.stderr)


def is_locked():
    """True when the platform lock registry marks this module locked.

    Reads the registry fresh on every call (no caching), so a toggle in the
    desktop UI takes effect on the very next request. Env overrides are read
    here at call time, not at import, so tests can flip them per request:

      VICARIUS_LOCKS_PATH     alternate registry path
      VICARIUS_LOCK_BYPASS=1  force unlocked (manual dev runs)

    Fail-open: missing/empty/corrupt registry, a malformed locks table or
    entry, or a non-bool "locked" value all mean unlocked, warning on stderr
    at most once per distinct error.
    """
    if os.environ.get("VICARIUS_LOCK_BYPASS") == "1":
        return False
    path = Path(os.environ.get("VICARIUS_LOCKS_PATH") or LOCKS_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        locks = data.get("locks") if isinstance(data, dict) else None
        if not isinstance(locks, dict):
            raise ValueError("registry has no 'locks' dict")
        entry = locks.get(MODULE_SLUG)
        if entry is None:
            return False  # no entry means unlocked (platform default)
        if not isinstance(entry, dict):
            raise ValueError(f"lock entry for {MODULE_SLUG!r} is not a dict")
        locked = entry["locked"]
        if not isinstance(locked, bool):
            raise ValueError(f"'locked' for {MODULE_SLUG!r} is not a bool")
        return locked
    except Exception as exc:
        _warn_once(f"{path}: {exc}")
        return False


# ── Backend runs pass the lock (Lauren, 2026-09-28) ─────────────────────────
# A lock keeps people out of the module's pages (students off the annotation
# tools); it must not stop a training study Lauren started headlessly. A
# request passes the gate while locked only when it comes from this machine
# and carries the backend key, a random secret kept outside the synced drive
# and readable only by this account. Browsers never send it, so every page
# still answers with the lock page and every other call with 423.
BACKEND_HEADER = "X-VEGA-Backend-Token"
BACKEND_TOKEN_PATH = os.path.join(os.path.expanduser("~"), ".config", "vicarius",
                                  "vega_backend_token")
BACKEND_TOKEN_BYTES = 32
LOOPBACK_ADDRS = ("127.0.0.1", "::1")


def backend_token_path():
    """Where the backend key lives; VEGA_BACKEND_TOKEN_PATH overrides it (tests)."""
    return Path(os.environ.get("VEGA_BACKEND_TOKEN_PATH") or BACKEND_TOKEN_PATH)


def backend_token(create=False):
    """The backend key, '' when there is none.

    With create=True a missing key is generated (32 random bytes, hex) and
    written with owner-only permissions. A key that cannot be read is '' and
    therefore never matches, so a broken key file keeps the lock closed.
    """
    import secrets  # noqa: PLC0415
    path = backend_token_path()
    try:
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    except OSError:
        pass
    if not create:
        return ""
    token = secrets.token_hex(BACKEND_TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token


def is_backend_request(headers, remote_addr):
    """True when a request may pass the lock: from this machine, with the key.

    headers: the request headers (any mapping with .get); remote_addr: the
    peer address. The comparison is constant-time, and an empty key on either
    side never matches.
    """
    import hmac  # noqa: PLC0415
    if remote_addr not in LOOPBACK_ADDRS:
        return False
    sent = (headers.get(BACKEND_HEADER) or "").strip()
    expected = backend_token()
    return bool(sent) and bool(expected) and hmac.compare_digest(sent, expected)

