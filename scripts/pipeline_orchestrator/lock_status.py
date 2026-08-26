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
LOCKS_PATH = "/mnt/rip/vicarius_drive/vicarius/_METADATA/module_locks.json"

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
