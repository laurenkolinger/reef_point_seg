"""
Pipeline orchestrator configuration — loaded from config/pipeline.yaml.

This file is a thin shim over the YAML. It exposes the same module-level
constants the rest of the orchestrator (app.py, project_manager.py,
stage_runner.py) already imports, so no other code had to change.

Path of the YAML is resolved from the TCRMP_CONFIG_PATH env var if set,
otherwise from <repo_root>/config/pipeline.yaml where repo_root is two
directories above this file (scripts/pipeline_orchestrator -> repo root).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_CONFIG_PATH = Path(
    os.environ.get("TCRMP_CONFIG_PATH")
    or Path(__file__).resolve().parent.parent.parent / "config" / "pipeline.yaml"
)


def _collect_strings(obj):
    """Yield every string inside nested dict/list/scalar structure."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _collect_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _collect_strings(v)


def _interp(obj, scope):
    """Recursively replace ${var} with scope[var]. Untouched if var is missing."""
    if isinstance(obj, str):
        def _sub(m):
            key = m.group(1)
            val = scope.get(key)
            return str(val) if val is not None else m.group(0)
        # Iterate up to three passes so ${a} -> ${b} -> final value resolves
        for _ in range(3):
            new = re.sub(r"\$\{(\w+)\}", _sub, obj)
            if new == obj:
                break
            obj = new
        return obj
    if isinstance(obj, dict):
        return {k: _interp(v, scope) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, scope) for v in obj]
    return obj


def _load():
    raw = yaml.safe_load(_CONFIG_PATH.read_text())
    repo_root = str(_CONFIG_PATH.resolve().parent.parent)

    # First pass: resolve the paths block using ${repo_root} only.
    path_scope = {"repo_root": repo_root}
    resolved_paths = _interp(raw.get("paths", {}) or {}, path_scope)

    # Two passes over the paths block itself so paths that depend on other
    # paths (e.g. all_points_csv -> supporting_data_dir) resolve.
    for _ in range(3):
        new_paths = _interp(resolved_paths, {**path_scope, **resolved_paths})
        if new_paths == resolved_paths:
            break
        resolved_paths = new_paths

    # Full scope for the rest of the config
    scope = {"repo_root": repo_root, **resolved_paths}
    cfg = _interp(raw, scope)
    cfg["paths"] = resolved_paths  # reinstate the fully-resolved paths block

    # Fail-fast: no unresolved ${...} should remain
    leftover = [s for s in _collect_strings(cfg) if "${" in s]
    if leftover:
        raise RuntimeError(
            f"Unresolved ${{...}} placeholders in {_CONFIG_PATH}: {leftover[:5]}"
        )

    return cfg, scope


_cfg, _scope = _load()

# ── Public constants (matching the original hardcoded API) ─────────────────
REPO_DIR = str(_CONFIG_PATH.resolve().parent.parent)
CONFIG_PATH = str(_CONFIG_PATH)
PATHS = _scope  # dict of every resolved path variable
PIPELINE_NAME = _cfg.get("pipeline_name", "reef_point_seg")

ORCHESTRATOR_PORT = _cfg["orchestrator_port"]
STAGE_PORTS = {int(k): v for k, v in _cfg["stage_ports"].items()}
PYTHON_PATHS = {int(k): v for k, v in _cfg["python_paths"].items()}
ENTRY_POINTS = {int(k): v for k, v in _cfg["entry_points"].items()}
WORKING_DIRS = {int(k): v for k, v in _cfg["working_dirs"].items()}
DEFAULT_STEP_CONFIGS = _cfg["step_defaults"]
PRESET_DIRS = _cfg["preset_dirs"]
VICARIUS = _cfg.get("vicarius", {"enabled": False})

# ── Step 6 default overrides ────────────────────────────────────────────────
# Source of truth for step 6 form defaults (pipeline.yaml is owned by the
# module config, not the orchestrator). Applied here so new projects and
# orchestrator restarts inherit the current policy:
#   - run_name defaults to blank (user opts in to reusing a run-name)
#   - device defaults to "0,1" (DDP across both GPUs by default)
# Anything already present in pipeline.yaml is respected if explicitly set;
# these overrides only fill in missing keys / flip values the user requested.
_s6 = DEFAULT_STEP_CONFIGS.setdefault("6", {})
_s6["run_name"] = ""                 # always blank — see Step 6 UX rules
_s6["device"] = "0,1"                # DDP across cuda:0 + cuda:1 by default
