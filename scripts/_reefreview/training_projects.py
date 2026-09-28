"""Minting a pipeline project that starts at Train.

A training project is a run_* folder that holds no annotation work of its own:
its steps 1 to 4 are skipped and it opens on Train (step 6). Two kinds of
training source mint one, and this module is the single place that writes the
run dir for both.

    composite   a YOLO-seg export assembled across projects on the coverage
                matrix, at <inprocess>/_composite/<name>. step4test_combinedAnnotate
                is a relative SYMLINK to it, so Train reads the all_images/ +
                all_labels/ + data.yaml layout it already knows and no image is
                copied. project_type "composite_training".

    bbox        a staged detection dataset (images/ + labels/), read where it
                is staged. NO symlink and no imagery of any kind enters the run
                dir: Step 6 builds its dataset_detect/ from the recorded source
                path. project_type "bbox_training". Non-TCRMP imagery never
                reaches the mask registry, the coverage matrix or STATS, and a
                run dir that held it would.

Every scan that gathers annotation work has to skip both types, which is what
is_training_project() and TRAINING_PROJECT_TYPES are for.

Nothing here imports the orchestrator or the matrix: it works off explicit
paths, so a caller (and its tests) can point at a temp tree.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

AST = timezone(timedelta(hours=-4))

# github_repo root: <repo>/scripts/_reefreview/training_projects.py. project.json
# records it the same way project_manager does.
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# project.json markers for a run dir that trains from a prepared source instead
# of holding annotation work. _matrix/builder.py mirrors the composite one.
COMPOSITE_PROJECT_TYPE = "composite_training"
BBOX_PROJECT_TYPE = "bbox_training"
TRAINING_PROJECT_TYPES = (COMPOSITE_PROJECT_TYPE, BBOX_PROJECT_TYPE)

SOURCE_KINDS = ("composite", "bbox")
_PROJECT_TYPE_FOR = {
    "composite": COMPOSITE_PROJECT_TYPE,
    "bbox": BBOX_PROJECT_TYPE,
}

# The step dir the Train step reads on the composite path. Symlinked into the
# composite so the export layout is the one Train already knows.
# project_manager.STEP_DIRS is the source of truth for these names; constraint:
# a pipeline step dir is never renamed.
TRAINING_LINK_DIRNAME = "step4test_combinedAnnotate"
COMPOSITE_DIRNAME = "_composite"

# Steps a training project never runs. Written as "skipped" so the sidebar
# reads them as deliberately bypassed rather than pending work. Steps 5 to 8
# are deliberately ABSENT from the written project.json: the orchestrator
# grafts them on first open with its own DEFAULT_STEP_CONFIGS, so the Train
# panel inherits every default instead of a copy that could drift.
SKIPPED_STEPS = [
    ("1", "step1_makeAllPoints", "Make All Points"),
    ("2", "step2_recodeSpecies", "Recode Species"),
    ("3", "step3_chooseImages", "Choose Images"),
    ("4", "step4_routeChosenImages", "Place Points"),
]

# Initials accepted for the `by` of a training project (the shape the matrix
# and the launcher use for reviewer initials).
_INITIALS_RE = re.compile(r"^[A-Za-z]{1,4}$")
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
# Collision suffixes tried for a same-day run dir, matching the launcher's
# 99-per-day-per-initials ceiling (vicarius_ui_os module_views._auto_run_id).
_MAX_RUN_SUFFIX = 99


class RunDirExhausted(Exception):
    """Every run dir name for these initials and this source is taken today
    (a route answers 409, same as the VICARIUS launcher's own ceiling).

    Same name as _matrix/composite.py's, so once that module delegates here
    the blueprint's status mapping can catch one exception for both."""


def now_ast():
    """'YYYY-MM-DD HH:MM:SS' in AST 24h, the way every VICARIUS surface writes
    a timestamp."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M:%S")


def sanitize_name(name):
    """Keep [A-Za-z0-9._-]; anything else becomes an underscore."""
    return _NAME_RE.sub("_", str(name or "").strip()).strip("._")


def is_training_project(meta):
    """True for a project.json that trains from a prepared source, of either
    kind. Every scan of annotation work skips these."""
    return (meta or {}).get("project_type") in TRAINING_PROJECT_TYPES


def _mint_run_dir(inprocess_root, base_name):
    """Create the first free run dir and return (path, basename): base_name,
    then base_name_2 .. base_name_99 (the same 99-per-day ceiling the VICARIUS
    launcher enforces).

    The mkdir IS the claim. An exists-precheck followed by makedirs lets two
    concurrent minters agree on the same free name and one of them crash with
    FileExistsError; here the loser simply takes the next suffix. Raises
    RunDirExhausted when every suffix is taken.
    """
    for n in range(1, _MAX_RUN_SUFFIX + 1):
        candidate = base_name if n == 1 else f"{base_name}_{n}"
        path = os.path.join(inprocess_root, candidate)
        try:
            os.makedirs(path)
        except FileExistsError:
            continue
        return path, candidate
    raise RunDirExhausted(f"no free run folder for {base_name} today "
                          f"({_MAX_RUN_SUFFIX} per day per initials); archive one and retry")


def _write_json_atomic(path, doc):
    """Write JSON so a concurrent reader sees either the old file or the whole
    new one, never a truncated one: a temp file in the SAME directory (so the
    rename cannot cross a filesystem) then os.replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".training-project-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _display_path(inprocess_root, source_path):
    """How project.json names the source: repo-relative for anything inside
    inprocess/ (which is how the composite contract already writes it), the
    absolute path for a source staged outside it."""
    real_root = os.path.abspath(inprocess_root)
    real_src = os.path.abspath(source_path)
    if real_src == real_root or real_src.startswith(real_root + os.sep):
        return os.path.join("inprocess", os.path.relpath(real_src, real_root))
    return real_src


def _composite_readme(source_name, sources, run_name, created_at):
    lines = [
        f"# {run_name}",
        "",
        f"Created {created_at} AST.",
        "",
        f"This project trains on the training set {source_name}, assembled on the",
        "coverage matrix from masks that were annotated in other projects. It holds no",
        "annotation work of its own: steps 1 to 4 are skipped, and the project opens on",
        "Train.",
        "",
        "## Where the training data lives",
        "",
        f"{TRAINING_LINK_DIRNAME} is a symlink to inprocess/{COMPOSITE_DIRNAME}/{source_name},",
        "so Train reads that set's all_images/, all_labels/ and data.yaml directly. No",
        "image is copied. Editing or rebuilding the training set changes what this",
        "project trains on, and deleting the set leaves this project with nothing to",
        "read.",
        "",
        "## Source projects",
        "",
    ]
    if sources:
        lines.append("| project | masks |")
        lines.append("| --- | --- |")
        for s in sources:
            lines.append(f"| {s.get('project_id', '')} | {s.get('n_masks', 0)} |")
    else:
        lines.append("No source project was recorded for this training set.")
    lines.extend([
        "",
        "The masks stay owned by the projects above. The coverage matrix keeps naming",
        "them as the place each annotation was made, and names this project as one of",
        "the training projects built from the set.",
        "",
    ])
    return "\n".join(lines)


def _bbox_readme(source_name, source_path, run_name, created_at, task_default):
    return "\n".join([
        f"# {run_name}",
        "",
        f"Created {created_at} AST.",
        "",
        f"This project trains on the staged detection dataset {source_name}. It holds no",
        "annotation work of its own: steps 1 to 4 are skipped, and the project opens on",
        "Train.",
        "",
        "## Where the training data lives",
        "",
        f"{source_path}",
        "",
        "The dataset is read where it is staged. Nothing is copied or symlinked into",
        "this folder: Step 6 builds its own train/valid/test split from that path, and",
        f"trains with task {task_default}. Imagery from outside TCRMP never enters the",
        "mask registry, the coverage matrix or STATS, and this project contributes",
        "nothing to any of them.",
        "",
    ])


def mint_training_project(inprocess_root, source_kind, source_name, source_path, by, extra=None):
    """Mint a run_* project that starts at Train, for either kind of source.

    Creates <inprocess_root>/run_{YY-MM-DD}_{INITIALS}_train_{name} (with a
    _2 .. _99 suffix on a same-day collision) holding project.json, README.md,
    the four skipped step folders, and (composite sources only) the relative
    step4test_combinedAnnotate symlink.

    ``extra`` is optional and may carry:
        source_projects   list of {project_id, n_masks} for a composite
        display_name      overrides the default TRAIN_<name>
        path_label        overrides how project.json names the source path
        task_default      overrides "detect" for a bbox source

    Returns {"run_dir" (absolute), "run_dir_name", "project_id",
    "display_name"}. Raises ValueError on bad initials or an unknown source
    kind, FileNotFoundError when the source directory is not there, and
    RunDirExhausted when no run dir is free today.

    Concurrency: the run dir is claimed by the mkdir itself (_mint_run_dir), so
    simultaneous calls cannot agree on the same name, and project.json is
    written with an atomic replace so a reader never sees a truncated file.

    The CALLER owns the source-specific guards, because what makes a source
    usable differs by kind and only the caller knows its own request shape.
    _matrix/composite.py::create_training_project holds two that any composite
    caller needs and this helper deliberately does not repeat:

      1. Resolve the name inside the composite root before calling: refuse a
         name holding a path separator, a traversal, or a symlink pointing out
         of the root, and require a data.yaml. A name arriving from a request
         body is otherwise a path-traversal surface.
      2. Refuse a set whose all_images/ is missing or empty, so a project is
         never minted around a training set that trains on nothing.

    A bbox caller owes the equivalent: the staged dataset must hold an
    image/label pair (detect_dataset.bbox_pairs answers that), and its path
    must stay inside the staging root the caller offered.

    The composite path also owes the composite.json linked_projects back link,
    which stays with the composite module (it is that module's file, written
    under its own lock).
    """
    extra = dict(extra or {})

    kind = str(source_kind or "").strip().lower()
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown training source kind: {source_kind!r}. "
                         f"Use one of {', '.join(SOURCE_KINDS)}.")

    initials = str(by or "").strip()
    if not _INITIALS_RE.match(initials):
        raise ValueError("initials must be 1 to 4 letters")
    initials = initials.upper()

    name = str(source_name or "").strip()
    if not name:
        raise ValueError("a training source needs a name")
    if not source_path or not os.path.isdir(source_path):
        raise FileNotFoundError(f"unknown training source: {source_path}")

    inprocess_root = os.path.abspath(inprocess_root)
    stamp = datetime.now(AST)
    base_name = "run_%s_%s_train_%s" % (stamp.strftime("%y-%m-%d"), initials, sanitize_name(name))
    run_dir, run_name = _mint_run_dir(inprocess_root, base_name)

    project_id = run_name[len("run_"):] if run_name.startswith("run_") else run_name
    display_name = str(extra.get("display_name") or f"TRAIN_{name}")
    created_at = now_ast()
    now_iso = stamp.replace(microsecond=0).isoformat()

    # Wording matches _matrix/composite.py::create_training_project on the
    # composite path, so a project minted here reads identically to one minted
    # there and the later delegation changes no file on disk.
    source_label = "training set" if kind == "composite" else "detection dataset"

    steps = {}
    for key, dirname, step_name in SKIPPED_STEPS:
        step_path = os.path.join(run_dir, dirname)
        os.makedirs(step_path, exist_ok=True)
        steps[key] = {
            "status": "skipped",
            "name": step_name,
            "dir": step_path,
            "config": {},
            "outputs": {},
            "note": f"from {source_label} {name}",
        }

    state = {
        "id": project_id,
        "name": display_name,
        "created_at": now_iso,
        "updated_at": now_iso,
        "project_dir": run_dir,
        "repo_dir": _REPO_DIR,
        "current_step": 6,
        "steps": steps,
        "access_log": [{
            "ts": stamp.isoformat(timespec="seconds"),
            "initials": initials,
            "action": "create",
            "purpose": f"training project for the {source_label} {name}",
        }],
        "initials": initials,
        "project_type": _PROJECT_TYPE_FOR[kind],
    }

    path_label = extra.get("path_label") or _display_path(inprocess_root, source_path)
    if kind == "composite":
        # Relative link so the pair survives a move of the whole inprocess
        # tree. The composite already holds all_images/ (themselves symlinks to
        # the source frames), all_labels/ and data.yaml, which is exactly the
        # layout Train reads, so no image is duplicated anywhere.
        os.symlink(os.path.relpath(os.path.abspath(source_path), run_dir),
                   os.path.join(run_dir, TRAINING_LINK_DIRNAME))
        sources = list(extra.get("source_projects") or [])
        state["composite"] = {
            "name": name,
            "path": path_label,
            "source_projects": sources,
            "created_at": created_at,
        }
        readme = _composite_readme(name, sources, run_name, created_at)
    else:
        task_default = str(extra.get("task_default") or "detect")
        state["training_source"] = {
            "kind": "bbox",
            "name": name,
            "path": os.path.abspath(source_path),
            "task_default": task_default,
        }
        readme = _bbox_readme(name, os.path.abspath(source_path), run_name, created_at,
                              task_default)

    _write_json_atomic(os.path.join(run_dir, "project.json"), state)
    with open(os.path.join(run_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme)

    return {
        "run_dir": run_dir,
        "run_dir_name": run_name,
        "project_id": project_id,
        "display_name": display_name,
    }
