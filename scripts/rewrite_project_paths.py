#!/usr/bin/env python3
"""
Migrate absolute paths inside projects/*/project.json (and per-step data.yaml +
absolute symlinks) from OLD_ROOT to NEW_ROOT.

The copied projects/ dir carries hardcoded absolute paths to the source repo
location. Run this once (from bootstrap.sh) to point everything at the new
repo root.

Usage:
    python rewrite_project_paths.py --old /old/path --new /new/path           # dry-run
    python rewrite_project_paths.py --old /old/path --new /new/path --apply   # commit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def rewrite_json_file(path: Path, old: str, new: str, apply: bool) -> bool:
    """Return True if any change was made (or would be made)."""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return False
    if old not in text:
        return False
    new_text = text.replace(old, new)
    # Validate that the result is still valid JSON
    try:
        json.loads(new_text)
    except json.JSONDecodeError as e:
        print(f"  SKIP {path} — would break JSON: {e}", file=sys.stderr)
        return False
    if apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_text)
        os.replace(tmp, path)
    return True


def rewrite_text_file(path: Path, old: str, new: str, apply: bool) -> bool:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return False
    if old not in text:
        return False
    if apply:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text.replace(old, new))
        os.replace(tmp, path)
    return True


def rewrite_symlink(link_path: Path, old: str, new: str, apply: bool) -> bool:
    try:
        target = os.readlink(link_path)
    except OSError:
        return False
    if not target.startswith(old):
        return False
    new_target = new + target[len(old):]
    if apply:
        link_path.unlink()
        os.symlink(new_target, link_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old", required=True,
                    help="Old absolute path prefix (e.g. /mnt/rip/.../CVR_CLIP_forAI)")
    ap.add_argument("--new", required=True,
                    help="New absolute path prefix (e.g. /mnt/rip/.../seg_AI_img_full_april2026)")
    ap.add_argument("--apply", action="store_true",
                    help="Commit changes. Without this flag, print what would change.")
    ap.add_argument("--projects-dir", default=None,
                    help="Override the projects/ directory (default: <new>/projects)")
    args = ap.parse_args()

    old = args.old.rstrip("/")
    new = args.new.rstrip("/")
    projects = Path(args.projects_dir) if args.projects_dir \
               else Path(new) / "projects"

    if not projects.is_dir():
        print(f"No projects directory at {projects} — nothing to do.")
        return

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Rewriting paths inside {projects}")
    print(f"         {old}  ->  {new}")

    json_patched = 0
    yaml_patched = 0
    symlinks_patched = 0

    for project_dir in sorted(projects.iterdir()):
        if not project_dir.is_dir():
            continue

        # 1. project.json
        pj = project_dir / "project.json"
        if pj.is_file() and rewrite_json_file(pj, old, new, args.apply):
            print(f"  json   {pj.relative_to(projects)}")
            json_patched += 1

        # 2. Any *.yaml inside the project (mostly step5/data.yaml,
        #    sometimes step6 args.yaml with paths). Text replacement only.
        for yml in project_dir.rglob("*.yaml"):
            # Skip venv yaml files (we don't track envs in projects, but guard anyway)
            if "env/" in str(yml):
                continue
            if rewrite_text_file(yml, old, new, args.apply):
                print(f"  yaml   {yml.relative_to(projects)}")
                yaml_patched += 1

        # 3. Symlinks under step5_segmentImages/all_images/ and elsewhere
        for root, dirs, files in os.walk(project_dir, followlinks=False):
            for name in files:
                p = Path(root) / name
                if p.is_symlink():
                    if rewrite_symlink(p, old, new, args.apply):
                        print(f"  link   {p.relative_to(projects)}")
                        symlinks_patched += 1

    print(f"\nSummary: {json_patched} project.json, "
          f"{yaml_patched} yaml, {symlinks_patched} symlinks rewritten.")
    if not args.apply:
        print("(dry-run — rerun with --apply to commit)")


if __name__ == "__main__":
    main()
