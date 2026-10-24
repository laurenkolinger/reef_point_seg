#!/usr/bin/env python3
"""return_frames_to_review.py: send finished VEGA frames back to an annotator's review queue.

Module:  reef_point_seg / return_frames_to_review
Purpose: A frame marked exported never re-enters the combined annotator's
         review queue ("done is done", _build_review_list in
         TCRMPclip_combinedAnnotate/src/app.py). This tool sends chosen
         finished frames back: it sets each frame's `exported` flag to false
         and appends one dated event to the frame's `returned_to_review` list,
         so the next time the annotator opens the project the frames lead the
         queue for the person to label or reject the outline. Nothing else
         changes: not the masks, label_outcomes or reviewed on the frame, and
         not the ledger, the mask registry, the label files or the export
         manifest. The next export of a frame replaces its label file,
         manifest row, registry rows and ledger rows.
Inputs:  --project RUN        a run folder under --root (repeatable)
         --frame STEM         a frame named by its file stem (repeatable); a bare
                              STEM needs exactly one --project
         --frame RUN/STEM     the same, naming its project, which must be one of the
                              --project values; lets one run name frames in several
         --select unlabelled-hand-outlines
                              every exported, unscrapped frame holding a pending
                              hand-drawn outline with no species
         --reason TEXT        why the frames go back (5 to 300 characters)
         --by XX              initials of the person sending them back
         --apply              make the changes; without it the tool only lists them
         --root DIR           the inprocess/ tree (default: the module's own)
Outputs: dry run: the frame list on stdout; nothing is written.
         --apply, per project with frames to send back:
           step4test_combinedAnnotate/_superseded/<YYYYMMDD-HHMMSS>/before_return_to_review/
             segmentations.json  a byte-for-byte copy of the file before the change
           step4test_combinedAnnotate/segmentations/segmentations.json
             rewritten in place, same mode and the same JSON layout byte for byte
           step4test_combinedAnnotate/review_returns.jsonl
             one JSON line: frames, checksums before and after, backup, who, why, when (AST)
         "Step N complete" checkpoints and the counts on stdout. A refusal is
         one sentence on stderr and exit code 2; exit code 0 otherwise.

The tool refuses (exit 2), in the dry run as well, when a project folder or
its segmentations.json is missing, unreadable or a link; when the export or
segmentations folder is a link; when the file repeats a key or its JSON layout
cannot be reproduced byte for byte (a rewrite would then change more than the
frames sent back); when anything listens on the annotator ports (5080 to 5089
and 5096, read with `ss -ltn`, and also when ss cannot run); when a named frame
does not exist; when a RUN/STEM frame names a project not given with
--project; or when a name or the reason is not plain one-line text. A
named frame already out of the finished set (not exported) or scrapped is
skipped and reported, so a rerun is safe.

The write: the new file is written to a temporary file beside the original
and flushed to disk; the backup is written and its checksum confirmed; then,
holding the expert-ID importer's segmentations lock (the same lock file the
importer locks, polled for up to IMPORTER_LOCK_TIMEOUT_S, about a minute) and
the ledger lock, the tool checks the ports again, re-reads segmentations.json
and refuses if its checksum changed since the first read, os.replace()s the
temporary file over it and flushes the folder. An import saved meanwhile is
therefore never overwritten, and two runs of this tool never interleave. The
temporary file is removed on every way out, Ctrl-C included (exit 130).

Example:
    env/bin/python scripts/return_frames_to_review.py --project run_26-09-18_AT \\
        --select unlabelled-hand-outlines --by LO --reason "Outline saved without a species."
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

sys.dont_write_bytecode = True  # a dry run must leave nothing behind, not even __pycache__

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _reefreview.ledger_io import LedgerLockTimeout, ledger_lock  # noqa: E402
from _reefreview.manual_sources import MANUAL_SOURCE_TYPES  # noqa: E402
from _superseded import STAMP_FORMAT, SUPERSEDED_DIR_NAME  # noqa: E402

# ---------------------------------------------------------------- constants
TOOL_NAME = 'return_frames_to_review.py'
TOOL_VERSION = '1.3.0'
TOOL_ID = f'{TOOL_NAME} {TOOL_VERSION}'
MODULE_ROOT = SCRIPTS_DIR.parent.parent           # scripts -> github_repo -> reef_point_seg
DEFAULT_ROOT = MODULE_ROOT / 'inprocess'
ANNOTATOR_PROVENANCE = SCRIPTS_DIR / 'TCRMPclip_combinedAnnotate' / 'src' / 'provenance.py'

RUN_PREFIX = 'run_'
EXPORT_DIR_NAME = 'step4test_combinedAnnotate'
SEG_NAME = 'segmentations.json'
SEG_DIR_NAME = 'segmentations'
LOG_NAME = 'review_returns.jsonl'
BACKUP_LABEL = 'before_return_to_review'
MAX_BACKUP_NAMES = 1000

RETURN_KEY = 'returned_to_review'
EXPORTED_KEY = 'exported'
SCRAPPED_KEY = 'scrapped'
PENDING = 'pending'
SELECT_UNLABELLED_HAND = 'unlabelled-hand-outlines'
SELECTIONS = (SELECT_UNLABELLED_HAND,)

REASON_MIN_CHARS, REASON_MAX_CHARS = 5, 300
INITIALS_PATTERN = re.compile(r'[A-Z]{2,4}')      # used with fullmatch: nothing before or after
CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f]')
PATH_SEPARATORS = tuple({'/', '\\', os.sep} | ({os.altsep} if os.altsep else set()))
NOT_NAMES = ('.', '..')
AST_OFFSET = timedelta(hours=-4)

# Annotator ports: the combined annotator and Edit Masks sweep 5080 to 5089;
# the standalone Annotate Images window listens on 5096.
STANDALONE_ANNOTATE_PORT = 5096
ANNOTATOR_PORTS = tuple(range(5080, 5090)) + (STANDALONE_ANNOTATE_PORT,)
PORTS_TEXT = '5080 to 5089 and 5096'
SS_COMMAND = ('ss', '-ltn')
SS_TIMEOUT_S = 10
SS_HEADER_WORD = 'State'
SS_LISTEN_STATE = 'LISTEN'
SS_LOCAL_ADDRESS_COLUMN = 3

# JSON layouts tried, in order, to reproduce the file byte for byte: the
# annotator writes indent=1, other repair tools wrote compact JSON.
JSON_LAYOUTS = ({'indent': 1}, {'indent': None}, {'indent': 2},
                {'indent': 1, 'ensure_ascii': False}, {'indent': None, 'ensure_ascii': False})
JSON_ENDINGS = (b'', b'\n')
# The expert-ID importer (_expertids/importer.py, _seg_locked) takes an
# exclusive flock on <segmentations folder>/.segmentations.json.lock for every
# read-modify-write of segmentations.json; the tool locks the same file, but
# polls instead of blocking, so it can say it is waiting and give up in time.
IMPORTER_LOCK_NAME = f'.{SEG_NAME}.lock'
IMPORTER_LOCK_TIMEOUT_S = 60.0
IMPORTER_LOCK_POLL_S = 0.1
WAITING_LINE = 'waiting for the expert-ID importer to finish saving'
LOG_FILE_MODE = 0o664
BACKUP_CREATE_MODE = 0o600   # tightened at once to the original's mode
CHUNK = 1 << 20
EXIT_OK, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 2, 130


# ---------------------------------------------------------------- data model
class Refusal(Exception):
    """A precondition failed. The message is the plain sentence the person reads."""


class RepeatedKey(ValueError):
    """segmentations.json writes one key twice in the same object."""


@dataclass
class Pick:
    """One frame chosen to go back to the queue, and why it was chosen."""
    key: str
    why: str


@dataclass
class ProjectPlan:
    """One project as read: its file bytes, parsed frames, JSON layout and the frames chosen."""
    name: str
    export_dir: Path
    seg_path: Path
    raw: bytes
    sha_before: str
    data: dict
    layout: dict
    ending: bytes
    picks: list = field(default_factory=list)
    skipped: list = field(default_factory=list)   # [(frame key, why)]


@dataclass(frozen=True)
class Change:
    """Who sends the frames back, why, and when (AST ISO time)."""
    at: str
    by: str
    reason: str

    def event(self) -> dict:
        """The returned_to_review event appended to each frame."""
        return {'at': self.at, 'by': self.by, 'reason': self.reason, 'tool': TOOL_ID,
                'previous_exported': True}


# ---------------------------------------------------------------- small helpers
def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file read in chunks (raises OSError when unreadable)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def printable(text) -> str:
    """Text as one printable line: control characters as '?', undecodable bytes as escapes."""
    return CONTROL_CHARS.sub('?', str(text)).encode('utf-8', 'backslashreplace').decode('utf-8')


def blank(value) -> bool:
    """True when value is not a string or holds only whitespace."""
    return not isinstance(value, str) or not value.strip()


def unique(items) -> list:
    """Items in their first order, each once."""
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def plural(n: int, word: str) -> str:
    """'1 frame', '2 frames'."""
    return f'{n} {word}' if n == 1 else f'{n} {word}s'


def say(line: str) -> None:
    """Print one progress line at once (the person may be watching a wait)."""
    print(line, flush=True)


def is_utf8_text(value: str) -> bool:
    """True when value encodes as UTF-8 (argv bytes that were not UTF-8 arrive as lone surrogates)."""
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


# ---------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    """Command-line parser; argparse itself exits 2 on a usage error."""
    ap = argparse.ArgumentParser(prog=TOOL_NAME, description=(
        'Send finished VEGA frames back to the combined annotator review queue. '
        'Without --apply the tool only lists what it would change.'))
    ap.add_argument('--project', action='append', required=True, metavar='RUN',
                    help='run folder under --root, such as run_26-09-18_AT (repeatable)')
    ap.add_argument('--frame', action='append', default=[], metavar='[RUN/]STEM',
                    help='frame file stem, such as TCRMP20141208_clip_CRB_T318, or RUN/STEM to name its '
                         'project, which must be one of the --project values (repeatable); a bare STEM '
                         'needs exactly one --project')
    ap.add_argument('--select', choices=SELECTIONS, default=None,
                    help='pick every exported, unscrapped frame holding a pending hand-drawn '
                         'outline with no species')
    ap.add_argument('--reason', required=True,
                    help=f'why the frames go back, {REASON_MIN_CHARS} to {REASON_MAX_CHARS} characters')
    ap.add_argument('--by', required=True, metavar='XX', help='initials, two to four capital letters')
    ap.add_argument('--apply', action='store_true', help='make the changes (default: dry run)')
    ap.add_argument('--root', default=str(DEFAULT_ROOT), help='the inprocess/ tree (default: %(default)s)')
    return ap


def name_problem(value: str) -> Optional[str]:
    """Plain reason a project or frame name is unsafe, or None when it is a plain name."""
    if not value:
        return 'is empty'
    if not is_utf8_text(value):
        return 'is not valid text (it holds bytes that are not UTF-8)'
    if CONTROL_CHARS.search(value):
        return 'holds a control character'
    if any(sep in value for sep in PATH_SEPARATORS):
        return 'holds a path separator'
    if value in NOT_NAMES:
        return 'is not a name'
    return None


def frame_target(value: str, projects: list) -> tuple:
    """(project or None, stem) for one --frame value: RUN/STEM or a bare STEM.

    Raises:
        Refusal: either part is not a plain name (empty, a control character,
            another separator, not UTF-8), or RUN is not among the --project values.
    """
    run_part, slash, stem = value.partition('/')
    if not slash:
        problem = name_problem(value)
        if problem:
            raise Refusal(f'--frame {printable(repr(value))} {problem}.')
        return None, value
    problem = name_problem(run_part)
    if problem:
        raise Refusal(f'--frame {printable(repr(value))}: its project part {problem}.')
    if run_part not in projects:
        raise Refusal(f'--frame {printable(repr(value))} names project {printable(run_part)}, which is not among '
                      f'the --project values ({", ".join(printable(p) for p in unique(projects))}).')
    problem = name_problem(stem)
    if problem:
        raise Refusal(f'--frame {printable(repr(value))}: its frame part {problem}.')
    return run_part, stem


def check_arguments(args: argparse.Namespace) -> None:
    """Validate every argument; strips the reason and sets args.frame_targets in place.

    args.frame_targets is the list of (project or None, stem) the --frame
    values name; None means the one --project given.

    Raises:
        Refusal: a bad reason, bad initials, no selection, an unsafe name, a
            RUN/STEM whose RUN was not given, or a bare STEM with more than one project.
    """
    reason = args.reason.strip()
    if not is_utf8_text(reason):
        raise Refusal('--reason is not valid text (it holds bytes that are not UTF-8); type it again.')
    if not REASON_MIN_CHARS <= len(reason) <= REASON_MAX_CHARS:
        raise Refusal(f'--reason must be {REASON_MIN_CHARS} to {REASON_MAX_CHARS} characters; '
                      f'it has {len(reason)}.')
    if CONTROL_CHARS.search(reason):
        raise Refusal('--reason holds a control character (a line break or a tab); write it on one line.')
    args.reason = reason
    if not INITIALS_PATTERN.fullmatch(args.by):
        raise Refusal(f'--by must be two to four capital letters, such as LO; got {printable(repr(args.by))}.')
    if not args.frame and not args.select:
        raise Refusal(f'name frames with --frame or pick them with --select {SELECT_UNLABELLED_HAND}.')
    for name in args.project:
        problem = name_problem(name)
        if problem:
            raise Refusal(f'--project {printable(repr(name))} {problem}.')
        if not name.startswith(RUN_PREFIX):
            raise Refusal(f'--project {printable(repr(name))} is not a project folder (it must start with '
                          f'{RUN_PREFIX}).')
    args.frame_targets = unique(frame_target(value, args.project) for value in args.frame)
    if any(run_name is None for run_name, _ in args.frame_targets) and len(set(args.project)) != 1:
        raise Refusal('a bare --frame STEM needs exactly one --project, so a frame is never sent back in the '
                      'wrong project; with several projects name it RUN/STEM.')


# ---------------------------------------------------------------- port check
def listening_ports(command=SS_COMMAND) -> set:
    """Every TCP port something listens on, read from `ss -ltn`.

    Raises:
        Refusal: ss is missing, fails, times out, or prints something that is
            not its listener table. The tool never guesses that a port is free.
    """
    try:
        done = subprocess.run(list(command), capture_output=True, text=True, timeout=SS_TIMEOUT_S, check=False)
    except FileNotFoundError:
        raise Refusal('the annotator ports cannot be checked: the ss command is not installed or not on PATH.')
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        raise Refusal(f'the annotator ports cannot be checked: ss did not run ({type(exc).__name__}: {exc}).')
    if done.returncode != 0:
        raise Refusal(f'the annotator ports cannot be checked: ss exited with code {done.returncode} '
                      f'({printable(done.stderr.strip())[:200]}).')
    lines = [line.split() for line in done.stdout.splitlines() if line.strip()]
    if not lines or lines[0][0] != SS_HEADER_WORD:
        raise Refusal('the annotator ports cannot be checked: ss did not print its listener table.')
    ports = set()
    for tokens in lines[1:]:
        port = tokens[SS_LOCAL_ADDRESS_COLUMN].rpartition(':')[2] \
            if len(tokens) > SS_LOCAL_ADDRESS_COLUMN and tokens[0] == SS_LISTEN_STATE else ''
        if not port.isdigit():
            raise Refusal(f'the annotator ports cannot be checked: ss printed a line the tool cannot read '
                          f'({printable(" ".join(tokens))[:120]!r}).')
        ports.add(int(port))
    return ports


def require_no_annotator() -> None:
    """Refuse while anything listens on an annotator port (an annotator may hold the file)."""
    busy = sorted(p for p in listening_ports() if p in ANNOTATOR_PORTS)
    if busy:
        raise Refusal(f'something is listening on port{"s" if len(busy) > 1 else ""} '
                      f'{", ".join(str(p) for p in busy)} (the annotator ports {PORTS_TEXT}). '
                      f'Stop every annotator tile in VEGA (closing the browser window leaves the '
                      f'annotator running), then run the tool again.')


# ---------------------------------------------------------------- reading and choosing
def _object_without_repeats(pairs: list) -> dict:
    """json object_pairs_hook: build the object, refusing a key written twice."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise RepeatedKey(key)
        out[key] = value
    return out


def json_layout(raw: bytes, data: dict) -> Optional[tuple]:
    """(dump keyword arguments, ending bytes) that reproduce raw exactly from data, or None."""
    for layout in JSON_LAYOUTS:
        try:
            body = json.dumps(data, **layout).encode('utf-8')
        except UnicodeEncodeError:  # a lone surrogate cannot be written unescaped: not this layout
            continue
        for ending in JSON_ENDINGS:
            if len(body) + len(ending) == len(raw) and body + ending == raw:
                return layout, ending
    return None


def _refuse_links(export_dir: Path, seg_path: Path) -> None:
    """Refuse when anything the tool writes through is a link.

    That is the export folder, the segmentations folder, segmentations.json,
    the _superseded holding folder and review_returns.jsonl.
    """
    for path in (export_dir, seg_path.parent, seg_path, export_dir / SUPERSEDED_DIR_NAME, export_dir / LOG_NAME):
        if path.is_symlink():
            raise Refusal(f'{path} is a link; the tool changes only real folders and files inside the project.')


def read_project(root: Path, name: str) -> ProjectPlan:
    """Read one project's segmentations.json: bytes, checksum, frames and JSON layout.

    Raises:
        Refusal: the folder is missing or leads outside root; the export
            folder, segmentations folder or file is a link; the file is
            missing, unreadable, not UTF-8 JSON, not an object, repeats a key,
            or is laid out in a way the tool cannot reproduce byte for byte.
    """
    folder = root / name
    if not folder.is_dir():
        raise Refusal(f'project {name} is not a folder under {root}.')
    if folder.resolve().parent != root.resolve():
        raise Refusal(f'project {name} is a link that leads outside {root}.')
    export_dir = folder / EXPORT_DIR_NAME
    seg_path = export_dir / SEG_DIR_NAME / SEG_NAME
    _refuse_links(export_dir, seg_path)
    try:
        raw = seg_path.read_bytes()
    except FileNotFoundError:
        raise Refusal(f'{seg_path} does not exist.')
    except OSError as exc:
        raise Refusal(f'{seg_path} cannot be read ({exc.strerror or exc}).')
    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_object_without_repeats)
    except RepeatedKey as exc:
        raise Refusal(f'{seg_path} writes the key {printable(repr(exc.args[0]))} twice in one object; a rewrite '
                      f'would silently drop one copy, so the tool will not change this file.')
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise Refusal(f'{seg_path} is not valid JSON ({type(exc).__name__}: {printable(str(exc))[:120]}).')
    if not isinstance(data, dict):
        raise Refusal(f'{seg_path} is not an object keyed by frame.')
    found = json_layout(raw, data)
    if found is None:
        raise Refusal(f'the JSON layout of {seg_path} cannot be reproduced byte for byte, so a rewrite would '
                      f'change more than the frames sent back; the tool will not change this file.')
    return ProjectPlan(name, export_dir, seg_path, raw, sha256_bytes(raw), data, found[0], found[1])


def find_frame(plan: ProjectPlan, stem: str) -> str:
    """The frame key a --frame value names: the exact key, else the one key with that stem.

    Raises:
        Refusal: no frame, or more than one frame, has that stem.
    """
    if stem in plan.data:
        return stem
    keys = sorted(k for k in plan.data if os.path.splitext(k)[0] == stem)
    if not keys:
        raise Refusal(f'frame {printable(stem)} is not in {plan.name}.')
    if len(keys) > 1:
        raise Refusal(f'frame {printable(stem)} matches {len(keys)} entries in {plan.name} '
                      f'({", ".join(printable(k) for k in keys)}); name it by its full file name.')
    return keys[0]


def unlabelled_hand_outlines(seg: dict) -> int:
    """How many pending hand-drawn masks (MANUAL_SOURCE_TYPES) on a frame have no species."""
    masks = seg.get('masks')
    if not isinstance(masks, list):
        return 0
    return sum(1 for m in masks if isinstance(m, dict) and m.get('status') == PENDING
               and m.get('source_type') in MANUAL_SOURCE_TYPES and blank(m.get('species')))


def choose_frames(plan: ProjectPlan, stems: list, select: Optional[str]) -> None:
    """Fill plan.picks and plan.skipped from the named stems and the selection rule.

    A named frame that is scrapped or not exported is skipped with a reason;
    --select picks only exported, unscrapped frames, so it never lists one.

    Raises:
        Refusal: a named frame does not exist or is not an object, or a chosen
            frame's returned_to_review value is not a list.
    """
    chosen = {}
    for stem in stems:
        key = find_frame(plan, stem)
        if not isinstance(plan.data[key], dict):
            raise Refusal(f'frame {printable(key)} in {plan.name} is not an object.')
        chosen.setdefault(key, 'named with --frame')
    if select == SELECT_UNLABELLED_HAND:
        for key, seg in plan.data.items():
            if not isinstance(seg, dict) or not seg.get(EXPORTED_KEY) or seg.get(SCRAPPED_KEY):
                continue
            n = unlabelled_hand_outlines(seg)
            if n:
                why = f'{plural(n, "hand outline")} with no species'
                chosen[key] = f'{chosen[key]}; {why}' if key in chosen else why
    for key in sorted(chosen):
        seg = plan.data[key]
        if seg.get(SCRAPPED_KEY):
            plan.skipped.append((key, 'scrapped, so it stays out of the queue'))
        elif not seg.get(EXPORTED_KEY):
            plan.skipped.append((key, 'already out of the finished set (not exported)'))
        elif not isinstance(seg.get(RETURN_KEY, []), list):
            raise Refusal(f'frame {printable(key)} in {plan.name} has a {RETURN_KEY} value that is not a list; '
                          f'it needs a person to look at it.')
        else:
            plan.picks.append(Pick(key, chosen[key]))


# ---------------------------------------------------------------- clock and edit
def load_ast_clock() -> Callable[[], str]:
    """The annotator's own AST clock (provenance.ast_now), loaded read-only.

    Raises:
        Refusal: the annotator's provenance.py cannot be loaded.
    """
    try:
        spec = importlib.util.spec_from_file_location('rftr_annotator_provenance', ANNOTATOR_PROVENANCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ast_now
    except Exception as exc:  # any failure to load the clock stops the tool before it writes
        raise Refusal(f'the annotator clock in {ANNOTATOR_PROVENANCE} cannot be loaded '
                      f'({type(exc).__name__}: {exc}).')


def make_change(by: str, reason: str, clock: Callable[[], str]) -> tuple:
    """(Change, holding-folder stamp) for this run, both from one AST reading.

    Raises:
        Refusal: the clock did not give an AST ISO time.
    """
    at = clock()
    try:
        when = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        when = None
    if when is None or when.utcoffset() != AST_OFFSET:
        raise Refusal(f'the annotator clock gave {at!r}, not an AST time.')
    return Change(at=at, by=by, reason=reason), when.strftime(STAMP_FORMAT)


def edited_bytes(plan: ProjectPlan, event: dict) -> bytes:
    """The new file: every picked frame copied with exported false and one more event.

    plan.data itself is never changed, and every frame not picked is the
    same object, so the result differs from plan.raw (which read_project
    proved this layout reproduces) only inside the picked frames.
    """
    edited = dict(plan.data)
    for pick in plan.picks:
        seg = dict(plan.data[pick.key])
        seg[EXPORTED_KEY] = False
        seg[RETURN_KEY] = list(seg.get(RETURN_KEY, [])) + [dict(event)]
        edited[pick.key] = seg
    return json.dumps(edited, **plan.layout).encode('utf-8') + plan.ending


# ---------------------------------------------------------------- writing
def fsync_folder(folder: Path) -> None:
    """Flush a folder's entries to disk so a rename survives a power cut."""
    fd = os.open(folder, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_own_temp(path: str) -> None:
    """Remove the temporary file this tool created; say so when that fails."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f'note: the temporary file {printable(path)} could not be removed ({exc.strerror or exc}).',
              file=sys.stderr)


def write_temp(plan: ProjectPlan, new_bytes: bytes) -> str:
    """Write new_bytes to a temporary file beside segmentations.json, with its mode, flushed to disk.

    Raises:
        Refusal: the folder cannot be written; segmentations.json is untouched.
    """
    tmp = None
    try:
        mode = stat.S_IMODE(plan.seg_path.stat().st_mode)
        fd, tmp = tempfile.mkstemp(dir=plan.seg_path.parent, prefix=f'.{SEG_NAME}.', suffix='.tmp')
        with os.fdopen(fd, 'wb') as f:
            f.write(new_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
    except OSError as exc:
        if tmp:
            _remove_own_temp(tmp)
        raise Refusal(f'a temporary file beside {plan.seg_path} cannot be written ({exc.strerror or exc}); '
                      f'{SEG_NAME} in {plan.name} was not changed.')
    except BaseException:  # Ctrl-C mid-write: remove the partial file, then stop
        if tmp:
            _remove_own_temp(tmp)
        raise
    return tmp


def _claim_backup_path(folder: Path) -> tuple:
    """(open descriptor, path) for the first free backup name in folder, created exclusively.

    A name already taken is never written over: the next try adds -1, -2, ...
    """
    stem, suffix = Path(SEG_NAME).stem, Path(SEG_NAME).suffix
    for n in range(MAX_BACKUP_NAMES):
        target = folder / (SEG_NAME if n == 0 else f'{stem}-{n}{suffix}')
        try:
            return os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, BACKUP_CREATE_MODE), target
        except FileExistsError:
            continue
    raise Refusal(f'no free backup name in {folder} after {MAX_BACKUP_NAMES} tries; nothing was changed.')


def write_backup(plan: ProjectPlan, stamp: str) -> Path:
    """Copy the file as read, byte for byte, into the dated holding folder; confirm its checksum.

    Returns:
        The backup path, <export>/_superseded/<stamp>/before_return_to_review/segmentations.json.
    Raises:
        Refusal: the folder or copy cannot be written, or the copy's checksum
            differs from the file as read.
    """
    folder = plan.export_dir / SUPERSEDED_DIR_NAME / stamp / BACKUP_LABEL
    try:
        mode = stat.S_IMODE(plan.seg_path.stat().st_mode)
        folder.mkdir(parents=True, exist_ok=True)
        fd, target = _claim_backup_path(folder)
        with os.fdopen(fd, 'wb') as f:
            f.write(plan.raw)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(target, mode)
        fsync_folder(folder)
        copied = sha256_file(target)
    except OSError as exc:
        raise Refusal(f'the backup in {folder} cannot be written ({exc.strerror or exc}); '
                      f'nothing was changed in {plan.name}.')
    if copied != plan.sha_before:
        raise Refusal(f'the backup {target} does not match the file it copies; nothing was changed in {plan.name}.')
    return target


def _try_lock(handle) -> bool:
    """Take an exclusive flock on an open file without waiting; True when it was taken."""
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _wait_for_importer(handle, plan: ProjectPlan, log: Callable[[str], None]) -> None:
    """Poll the importer's lock until it is free or IMPORTER_LOCK_TIMEOUT_S runs out.

    Raises:
        Refusal: the deadline passed; segmentations.json was not changed.
    """
    deadline = time.monotonic() + IMPORTER_LOCK_TIMEOUT_S
    told = False
    while not _try_lock(handle):
        if not told:
            log(WAITING_LINE)
            told = True
        if time.monotonic() >= deadline:
            raise Refusal(f'the expert-ID importer kept {plan.seg_path} locked for {IMPORTER_LOCK_TIMEOUT_S:g} '
                          f'seconds; {SEG_NAME} in {plan.name} was not changed. Run the tool again once the '
                          f'import has finished.')
        time.sleep(IMPORTER_LOCK_POLL_S)


@contextlib.contextmanager
def importer_lock(plan: ProjectPlan, log: Callable[[str], None]):
    """Hold the expert-ID importer's segmentations lock (same lock file), polled up to the deadline.

    Raises:
        Refusal: the lock file cannot be opened, or the deadline passed.
    """
    lock_path = plan.seg_path.parent / IMPORTER_LOCK_NAME
    try:
        handle = open(lock_path, 'a')
    except OSError as exc:
        raise Refusal(f'the expert-ID importer lock {lock_path} cannot be opened ({exc.strerror or exc}); '
                      f'{SEG_NAME} in {plan.name} was not changed.')
    with handle:
        _wait_for_importer(handle, plan, log)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def project_locks(plan: ProjectPlan, log: Callable[[str], None]):
    """Hold the expert-ID importer's segmentations lock, then the ledger lock.

    The importer saves segmentations.json only under its own lock (and never
    waits on the ledger lock while holding it); the annotators write the
    ledger under the ledger lock. Taking them in this order cannot deadlock.

    Raises:
        Refusal: either lock stayed busy past its bound.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(importer_lock(plan, log))
        try:
            stack.enter_context(ledger_lock(str(plan.export_dir)))
        except LedgerLockTimeout as exc:
            raise Refusal(f'the project lock of {plan.name} stayed busy ({exc}); nothing was changed.')
        yield


def _replace_if_unchanged(plan: ProjectPlan, tmp: str) -> None:
    """Under the project locks: check the ports, compare the checksum, replace, flush the folder."""
    require_no_annotator()
    if sha256_file(plan.seg_path) != plan.sha_before:
        raise Refusal(f'{plan.seg_path} changed while the tool ran (another program saved it); it was not '
                      f'written. Run the tool again.')
    os.replace(tmp, plan.seg_path)
    fsync_folder(plan.seg_path.parent)


def guarded_replace(plan: ProjectPlan, tmp: str, new_bytes: bytes, log: Callable[[str], None] = say) -> str:
    """Swap the temporary file in under the locks, only if nobody changed the file meanwhile.

    Returns:
        The new file's sha256.
    Raises:
        Refusal: the file changed, an annotator is listening, a lock stayed
            busy, or the swap failed. The temporary file is removed on every
            way out, a KeyboardInterrupt included.
    """
    try:
        with project_locks(plan, log):
            _replace_if_unchanged(plan, tmp)
    except OSError as exc:
        if not os.path.lexists(tmp):
            raise Refusal(f'{plan.seg_path} was replaced, but its folder could not be flushed to disk '
                          f'({exc.strerror or exc}); check it against the backup.')
        raise Refusal(f'{plan.seg_path} could not be replaced ({exc.strerror or exc}); it was not changed.')
    finally:
        _remove_own_temp(tmp)   # gone after a good swap; removed here after a refusal or Ctrl-C
    try:
        sha_after = sha256_file(plan.seg_path)
    except OSError as exc:
        raise Refusal(f'{plan.seg_path} was replaced but cannot be read back ({exc.strerror or exc}); '
                      f'check it against the backup.')
    if sha_after != sha256_bytes(new_bytes):
        raise Refusal(f'{plan.seg_path} does not hold what the tool wrote; restore it from the backup.')
    return sha_after


def log_line(record: dict) -> str:
    """One JSON line for review_returns.jsonl, readable text where it can be.

    A frame key holding a lone surrogate cannot be written as UTF-8 text; the
    line then uses JSON escapes throughout, which read back to the same text.
    """
    line = json.dumps(record, ensure_ascii=False)
    if not is_utf8_text(line):
        line = json.dumps(record)
    return line + '\n'


def append_log(plan: ProjectPlan, line: str, backup: Path) -> Path:
    """Append the prepared line to review_returns.jsonl and flush it to disk.

    Raises:
        Refusal: the log cannot be written. segmentations.json has already
            changed by then, and the message says so and carries the line.
    """
    path = plan.export_dir / LOG_NAME
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, LOG_FILE_MODE)
        with os.fdopen(fd, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except (OSError, UnicodeError) as exc:
        raise Refusal(f'{SEG_NAME} in {plan.name} WAS changed ({plural(len(plan.picks), "frame")} sent back to '
                      f'the review queue; the file before the change is {backup}), but its line in {path} is '
                      f'missing because the log cannot be written ({getattr(exc, "strerror", None) or exc}). '
                      f'Add this line to the log by hand: {line.strip()}')
    return path


def apply_plan(plan: ProjectPlan, change: Change, stamp: str, log: Callable[[str], None] = say) -> Optional[dict]:
    """Write the temporary file, back up, swap under the locks, then log. Returns the log record.

    Returns None when nothing is chosen in this project.

    Raises:
        Refusal: the temporary file, the backup, a lock, the guard, the swap
            or the log failed.
    """
    if not plan.picks:
        log(f'{plan.name}: nothing to change.')
        return None
    new_bytes = edited_bytes(plan, change.event())
    tmp = write_temp(plan, new_bytes)
    try:
        backup = write_backup(plan, stamp)
        log(f'Step 5 complete: {plan.name}: backup written and its checksum confirmed: {backup}')
        record = _log_record(plan, change, new_bytes, backup)
        line = log_line(record)
        guarded_replace(plan, tmp, new_bytes, log)
    finally:
        _remove_own_temp(tmp)   # on every way out, Ctrl-C included; already gone after the swap
    log(f'Step 6 complete: {plan.name}: segmentations.json rewritten, '
        f'{plural(len(plan.picks), "frame")} back in the review queue')
    path = append_log(plan, line, backup)
    log(f'Step 7 complete: {plan.name}: one line added to {path}')
    return record


def _log_record(plan: ProjectPlan, change: Change, new_bytes: bytes, backup: Path) -> dict:
    """The review_returns.jsonl record of one project's change."""
    return {'at': change.at, 'by': change.by, 'reason': change.reason, 'tool': TOOL_ID,
              'project': plan.name, 'segmentations': str(plan.seg_path),
              'frames': [p.key for p in plan.picks],
              'skipped': [{'frame': k, 'why': w} for k, w in plan.skipped],
              'sha256_before': plan.sha_before, 'sha256_after': sha256_bytes(new_bytes),
              'backup': str(backup), 'layout': dict(plan.layout)}


# ---------------------------------------------------------------- output
def print_plan(plans: list, apply: bool) -> None:
    """List every frame that goes back, and every frame skipped, per project."""
    for plan in plans:
        verb = ('goes back' if len(plan.picks) == 1 else 'go back') if apply else 'would go back'
        print(f'{plan.name}: {plural(len(plan.picks), "frame")} {verb} to the review queue'
              f'{", " + plural(len(plan.skipped), "frame") + " skipped" if plan.skipped else ""} '
              f'({plural(len(plan.data), "frame")} in the project)')
        for pick in plan.picks:
            print(f'  {printable(pick.key)}: exported true -> false, one {RETURN_KEY} event added '
                  f'({printable(pick.why)})')
        for key, why in plan.skipped:
            print(f'  {printable(key)}: skipped, {why}')


def counts_line(plans: list, apply: bool) -> str:
    """The closing counts: frames sent back per project and frames skipped."""
    per = ', '.join(f'{p.name} {len(p.picks)}' for p in plans)
    total = sum(len(p.picks) for p in plans)
    skipped = sum(len(p.skipped) for p in plans)
    label = 'Frames sent back' if apply else 'Frames that would go back'
    return f'{label}: {total} ({per}); frames skipped: {skipped}.'


# ---------------------------------------------------------------- main
def run(args: argparse.Namespace) -> int:
    """Check, read, choose, list, and (with --apply) write. Returns the exit code."""
    check_arguments(args)
    projects, targets = unique(args.project), args.frame_targets
    mode = 'apply' if args.apply else 'dry run'
    print(f'Return frames to review ({mode}), {TOOL_ID}')
    print(f'Step 1 complete: arguments checked ({plural(len(projects), "project")}, '
          f'{plural(len(targets), "named frame")}, selection: {args.select or "none"})')
    root = Path(args.root)
    if not root.is_dir():
        raise Refusal(f'--root {root} is not a folder.')
    plans = [read_project(root, name) for name in projects]
    print(f'Step 2 complete: {plural(len(plans), "project")} read, each file\'s JSON layout reproducible '
          f'byte for byte ({", ".join(f"{p.name} {len(p.data)} frames" for p in plans)})')
    for plan in plans:
        stems = [stem for run_name, stem in targets if run_name in (None, plan.name)]
        choose_frames(plan, stems, args.select)
    print(f'Step 3 complete: {plural(sum(len(p.picks) for p in plans), "frame")} chosen, '
          f'{sum(len(p.skipped) for p in plans)} skipped')
    require_no_annotator()
    print(f'Step 4 complete: nothing is listening on ports {PORTS_TEXT}')
    print_plan(plans, args.apply)
    if not args.apply:
        print('Dry run: nothing was written. Run the same command with --apply to make these changes.')
        print(counts_line(plans, False))
        return EXIT_OK
    change, stamp = make_change(args.by, args.reason, load_ast_clock())
    done = []
    try:
        for plan in plans:
            apply_plan(plan, change, stamp)
            done.append(plan.name)
    except Refusal:
        if done:
            print(f'Projects finished before the refusal: {", ".join(done)}', file=sys.stderr)
        raise
    print(counts_line(plans, True))
    return EXIT_OK


def _never_fail_on_odd_text() -> None:
    """Make stdout and stderr escape text they cannot encode instead of crashing mid-run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(errors='backslashreplace')


def main(argv: Optional[list] = None) -> int:
    """Parse the command line and run; a refusal prints one sentence and returns 2, Ctrl-C returns 130."""
    _never_fail_on_odd_text()
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Refusal as exc:
        print(f'Refused: {printable(exc)}', file=sys.stderr)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print('Stopped by Ctrl-C. A project whose "Step 6 complete" line is printed above was changed; '
              'no other project was, and no temporary file is left.', file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == '__main__':
    sys.exit(main())
