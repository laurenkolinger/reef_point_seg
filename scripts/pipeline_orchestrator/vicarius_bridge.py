"""
Thin bridge between the orchestrator and the VICARIUS event stream.

Emits process_start / process_end / user_note events via the VICARIUS
Python API (vicarius_log.VICARIUSLog). Degrades to no-ops when:
  - config/pipeline.yaml has `vicarius.enabled: false`, OR
  - the VICARIUS logging library is not importable, OR
  - initialization of the VICARIUSLog singleton raises.

Public API used by app.py:
    process_start(step_name, purpose, inputs=None, notes="") -> Optional[event_id]
    process_end(step_name, parent_event_id, status, duration_sec, outputs=None, notes="")
    note(message, study=None)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from orchestrator_config import PATHS, VICARIUS


_log: Optional[Any] = None
_MODULE: str = VICARIUS.get("module_name", "reef_point_seg")
_STUDY: str = VICARIUS.get("study_default", "")
_ENABLED: bool = bool(VICARIUS.get("enabled", False))


def _init():
    global _log
    if not _ENABLED or _log is not None:
        return
    lib_path = PATHS.get("vicarius_logging_lib", "")
    if not lib_path or not os.path.isdir(lib_path):
        print(f"[vicarius_bridge] lib path missing ({lib_path}) — disabled", file=sys.stderr)
        return
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    try:
        from vicarius_log import VICARIUSLog  # type: ignore
        _log = VICARIUSLog()
    except Exception as exc:
        print(f"[vicarius_bridge] disabled ({exc})", file=sys.stderr)


_init()


def is_enabled() -> bool:
    return _log is not None


def process_start(step_name: str, purpose: str,
                  inputs=None, notes: str = ""):
    """Record a process_start event. Returns the event_id for later linking."""
    if _log is None:
        return None
    try:
        return _log.process_start(
            module=f"{_MODULE}.{step_name}",
            purpose=purpose,
            study=_STUDY or None,
            inputs=inputs or [],
            notes=notes,
        )
    except Exception as exc:
        print(f"[vicarius_bridge] process_start failed: {exc}", file=sys.stderr)
        return None


def process_end(step_name: str, parent_event_id, status: str,
                duration_sec: float, outputs=None, notes: str = ""):
    """Record a process_end event, linked to the matching process_start."""
    if _log is None:
        return
    try:
        _log.process_end(
            module=f"{_MODULE}.{step_name}",
            status=status,
            duration_sec=duration_sec,
            parent_event_id=parent_event_id,
            outputs=outputs or [],
            notes=notes,
        )
    except Exception as exc:
        print(f"[vicarius_bridge] process_end failed: {exc}", file=sys.stderr)


def note(message: str, study: Optional[str] = None):
    """Record a user_note event."""
    if _log is None:
        return
    try:
        _log.user_note(message, study=study or _STUDY or None)
    except Exception as exc:
        print(f"[vicarius_bridge] note failed: {exc}", file=sys.stderr)
