#!/usr/bin/env python3
"""check_project_integrity.py: read-only integrity checker for every VEGA project.

Module:  reef_point_seg / check_project_integrity
Purpose: Audit every project under inprocess/ (project.json, Step 1 to 3 files,
         segmentations, exported images and YOLO labels, export manifest,
         label_provenance.csv ledger) and the shared stores (_mask_registry,
         _composite) against the 14 rules of the 2026-10-15 integrity audit.
         The checker never modifies anything it reads: every file is opened
         read-only, no application code that writes is called, and the only
         files it creates are its own <YYYYMMDD-HHMMSS>.txt and .json reports
         inside --report-dir.
Inputs:  --root         inprocess/ tree (default: the module's own inprocess/)
         --archive-dir  archive/ tree (default: <root>/../archive)
         --vocab        species vocabulary CSV (tcrmp_species_codes.csv)
         --pipeline-config  config/pipeline.yaml (names the hopper mirror)
         --snapshot-dir nightly snapshot folder (default /mnt/tear/vega_run_snapshots/daily)
         --exceptions   integrity_exceptions.yaml: frames a person ruled out of the
                        coverage matrix (default <VICARIUS_ROOT>/_METADATA/)
Outputs: findings on stdout (text, or one JSON document with --json),
         Step checkpoints on stderr, the two report files in --report-dir,
         exit 0 (no FAIL), 1 (any FAIL), 2 (usage or I/O error).

Rules (each finding carries rule id, PASS/INFO/WARN/FAIL, project, path, sentence):
  R01 project.json shape, ids, step folders, file permissions
  R02 Step 1 entries and Step 2 outputs resolve and parse
  R03 Step 3 selection: unique basenames, known targets, stale selection
  R04 segmentations.json structure, blank species and returned_to_review records
  R05 all_images links, image counts, image and label pairing
  R06 YOLO label lines, class_map.json, data.yaml
  R07 export_manifest.csv against frames, masks and label files
  R08 label_provenance.csv ledger against stored and computed outcomes
  R09 _mask_registry/manifest.csv against each project's exported masks
  R10 _composite training sets
  R11 species vocabulary, hopper mirror, codes in use
  R12 manual-source constants include "merged"
  R13 stray .tmp files, conflict files, prior_screening.json
  R14 nightly snapshot age and archive/ runs outside the nightly copy
  R15 integrity_exceptions.yaml parses and names live projects and existing frames

A frame named in integrity_exceptions.yaml is kept out of the coverage matrix by a
person's ruling: its "no ledger rows" (R08) and "no registry row" (R09) findings
are reported as INFO citing the ruling, and any ledger or registry row it gains
is a FAIL, because that breaks the ruling. Every other finding on it is unchanged.

A frame that return_frames_to_review.py sent back for review (it is neither
exported nor scrapped, and the last event in its returned_to_review list is
newer than the export stamp of its export_manifest.csv row) is waiting for its
next export:
the findings that compare its manifest row and ledger rows with its stored
state (R07 frame set and counts, R08 ledger and computed outcomes) are INFO
citing the event. A malformed event is an R04 FAIL and covers nothing. Once
the frame is exported again the event is history and every rule applies.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

sys.dont_write_bytecode = True  # importing app code must not write __pycache__

# ---------------------------------------------------------------- constants
AST = timezone(timedelta(hours=-4))  # Atlantic Standard Time, no DST
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
DEFAULT_ROOT = os.path.join(MODULE_ROOT, 'inprocess')
DEFAULT_REPORT_DIR = '/mnt/tear/vega_run_snapshots/integrity'
DEFAULT_SNAPSHOT_DIR = '/mnt/tear/vega_run_snapshots/daily'
DEFAULT_VOCAB = ('/mnt/rip/vicarius_drive/vicarius/_METADATA/library/definitions/'
                 'tcrmp_species_codes.csv')
DEFAULT_PIPELINE_CONFIG = os.path.join(os.path.dirname(SCRIPTS_DIR), 'config', 'pipeline.yaml')
MIRROR_KEY = 'duplicate_master_codes'
# modules/reef_point_seg -> modules -> the VICARIUS root, unless the environment names it.
VICARIUS_ROOT = os.environ.get('VICARIUS_ROOT') or os.path.dirname(os.path.dirname(MODULE_ROOT))
DEFAULT_EXCEPTIONS = os.path.join(VICARIUS_ROOT, '_METADATA', 'integrity_exceptions.yaml')
EXCEPTIONS_VERSION = 1
EXCEPTION_TEXT_KEYS = ('project_id', 'ruling', 'by')
EXCEPTION_SHORT_KEYS = ('project_id', 'by')  # names, never prose
MAX_SHORT_FIELD_CHARS = 200
EXCEPTION_DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
EXCEPTION_DATE_FORMAT = '%Y-%m-%d'
MAX_RULING_CHARS = 400  # a ruling longer than this is cut in findings, never in the file
# Frames sent back for review by return_frames_to_review.py (one event per return).
RETURN_KEY = 'returned_to_review'
RETURN_EVENT_KEYS = ('at', 'by', 'reason', 'tool', 'previous_exported')
RETURN_INITIALS = re.compile(r'[A-Z]{2,4}')  # used with fullmatch: 'LO\n' is not initials
RETURN_REASON_CHARS = (5, 300)  # the bounds return_frames_to_review.py enforces
AST_OFFSET = timedelta(hours=-4)
# export_manifest.csv batch_name: batch_<NNN>_<YYYYMMDD-HHMMSS>, stamped by the
# annotator with the box's local clock, which is AST on the VEGA box.
BATCH_STAMP = re.compile(r'_(\d{8}-\d{6})$')

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
PASS, INFO, WARN, FAIL = 'PASS', 'INFO', 'WARN', 'FAIL'
LEVELS = (PASS, INFO, WARN, FAIL)
MAX_EXAMPLES = 5
STALE_TMP_SECONDS = 3600
SNAPSHOT_MAX_AGE = timedelta(hours=26)
CSV_FIELD_LIMIT = 1 << 27
GROUP_READ_BIT = stat.S_IRGRP

EXPORT_DIR_NAME = 'step4test_combinedAnnotate'
STEP_DIRS = {'1': 'step1_makeAllPoints', '2': 'step2_recodeSpecies', '3': 'step3_chooseImages'}
STEP1_FILES = ('all_points.csv', 'tcrmp_species_codes.csv')
STEP2_FILES = ('all_points_recoded.csv', 'master_codes_recoded.csv', 'remap_log.json')
RUN_PREFIX = 'run_'
# Folder name _superseded.py moves content into instead of deleting it; walks skip it.
SUPERSEDED_DIR_NAME = '_superseded'
SHARED_STORES = ('_mask_registry', '_composite', '_expert_id_library', '_bbox',
                 '_standalone', '_models')
REVIEW_CODE = 'REVIEW'

PROJECT_KEYS = {'id': str, 'name': str, 'created_at': str, 'updated_at': str,
                'project_dir': str, 'current_step': int, 'steps': dict, 'access_log': list}
ACCESS_LOG_KEYS = ('ts', 'initials', 'action')
ACTIVE_STEP_STATES = ('completed', 'running')
TRAINING_PROJECT_TYPES = ('composite_training', 'bbox_training')
ANNOTATION_RULES = ('R04', 'R05', 'R06', 'R07', 'R08')
MASK_KEYS = ('id', 'species', 'status')

MANIFEST_BASE = ['basename', 'n_masks', 'label_file', 'image_file', 'outcome',
                 'preview_file', 'batch_name', 'image_path']
EXPORTED_OUTCOMES = ('exported_with_masks', 'exported_empty')
MANIFEST_OUTCOMES = EXPORTED_OUTCOMES + ('scrapped', 'review_only', 'image_missing')
LEDGER_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']
LEDGER_OUTCOMES = ('found_expert', 'found_manual', 'found_ai', 'pending_expert', 'not_found')
MIN_POLYGON_COORDS = 6
CONFLICT_GLOB = re.compile(r'^segmentations\.conflict-.*\.json$')
PRIOR_SCREENING_VERSION = 1
SNAPSHOT_STAMP = re.compile(r'^\d{8}-\d{6}$')
SNAPSHOT_STAMP_FORMAT = '%Y%m%d-%H%M%S'
REPORT_STAMP_FORMAT = '%Y%m%d-%H%M%S'

KNOWN_MANUAL_SOURCE_FILES = (
    'TCRMPclip_combinedAnnotate/src/provenance.py',
    'TCRMPclip_editMasks/src/provenance.py',
    'TCRMPclip_segmentImages/src/provenance.py',
    '_matrix/builder.py',
    '_matrix/composite.py',
    '_matrix/blueprint.py',
)
MANUAL_BOX_LITERAL = re.compile(r'''['"]manual_box['"]''')
MANUAL_CLICK_LITERAL = re.compile(r'''['"]manual_click['"]''')
MERGED_LITERAL = re.compile(r'''['"]merged['"]''')
CONSTANT_CONTEXT_LINES = 2


# ---------------------------------------------------------------- data model
@dataclass
class Finding:
    """One result: which rule, how serious, where, and a plain sentence."""
    rule: str
    level: str
    project: str
    path: str
    message: str


@dataclass
class Config:
    """Where everything lives; every default can be overridden for tests."""
    root: str = DEFAULT_ROOT
    archive_dir: Optional[str] = None
    vocab_path: str = DEFAULT_VOCAB
    pipeline_config: str = DEFAULT_PIPELINE_CONFIG
    scripts_dir: str = SCRIPTS_DIR
    snapshot_dir: str = DEFAULT_SNAPSHOT_DIR
    manual_source_files: Optional[list] = None
    exceptions_path: str = DEFAULT_EXCEPTIONS
    now: Optional[datetime] = None
    progress: bool = False  # print 'Step N complete' lines to stderr


@dataclass
class Report:
    """All findings of one run plus the counts the summary needs."""
    root: str
    started: str
    findings: list = field(default_factory=list)
    projects: int = 0

    def add(self, rule, level, project, path, message):
        """Append one finding; messages are kept to a single clean line."""
        self.findings.append(Finding(rule, level, project, str(path), _one_line(message)))

    def counts(self):
        """{PASS: n, INFO: n, WARN: n, FAIL: n}."""
        c = Counter(f.level for f in self.findings)
        return {level: c[level] for level in LEVELS}


@dataclass(frozen=True)
class Exclusion:
    """One ruling that keeps named frames of one project out of the coverage matrix."""
    project_id: str
    ruling: str
    date: str
    by: str

    def cite(self):
        """'ruling of <date> by <who>: "<text>"', the text cut to MAX_RULING_CHARS."""
        text = self.ruling if len(self.ruling) <= MAX_RULING_CHARS \
            else self.ruling[:MAX_RULING_CHARS].rstrip() + '...'
        return f'ruling of {self.date} by {self.by}: "{text}"'


@dataclass(frozen=True)
class ReturnEvent:
    """The last returned_to_review event of a frame waiting for its next export."""
    at: str
    by: str
    reason: str
    tool: str

    def cite(self):
        """'sent back for review on <at> by <by> (<tool>): "<reason>"'."""
        reason = self.reason if len(self.reason) <= MAX_RULING_CHARS \
            else self.reason[:MAX_RULING_CHARS].rstrip() + '...'
        return f'sent back for review on {self.at} by {self.by} ({self.tool}): "{reason}"'


def _one_line(text):
    """Collapse control characters so every finding is exactly one line."""
    return re.sub(r'[\x00-\x1f\x7f]+', ' ', str(text)).strip()


def _examples(items, limit=MAX_EXAMPLES):
    """'a, b, c (+N more)' for a list of things, so a message stays short."""
    items = [str(i) for i in items]
    shown = ', '.join(items[:limit])
    return shown + (f' (+{len(items) - limit} more)' if len(items) > limit else '')


def split_excepted(items, excluded, stem=lambda item: item):
    """Separate findings items that a ruling covers from the rest.

    Parameters:
        items: the things a finding lists (frame stems, mask uids, ...).
        excluded: {frame stem: Exclusion} for one project.
        stem: maps an item to its frame stem.
    Returns:
        (items no ruling covers, {Exclusion: [items it covers]}), order kept.
    """
    kept, covered = [], {}
    for item in items:
        exc = excluded.get(stem(item))
        if exc is None:
            kept.append(item)
        else:
            covered.setdefault(exc, []).append(item)
    return kept, covered


def excepted_findings(covered, path, text):
    """One INFO per ruling: '<n> <text>; excepted by ruling of ...: <items>.'"""
    return [(INFO, path, f'{len(items)} {text}; excepted by {exc.cite()}: {_examples(items)}.')
            for exc, items in covered.items()]


def broken_rulings(stems, excluded, path, text):
    """One FAIL per ruling whose excluded frames appear in stems (a ledger or registry row)."""
    _, covered = split_excepted(sorted(set(stems)), excluded)
    return [(FAIL, path, f'{len(items)} frames kept out of the coverage matrix have {text}, which '
                         f'breaks the {exc.cite()}: {_examples(items)}.')
            for exc, items in covered.items()]


def returned_findings(covered, path, text):
    """One INFO per return event: '<n> <text>; <event cite>, not exported again yet: <items>.'"""
    return [(INFO, path, f'{len(items)} {text}; {ev.cite()}, not exported again yet: {_examples(items)}.')
            for ev, items in covered.items()]


def frame_of(item):
    """Frame stem of a finding item: 'stem' or 'stem/label'."""
    return str(item).rsplit('/', 1)[0]


# ---------------------------------------------------------------- returned frames
def parse_ast_time(value):
    """datetime for an ISO time carrying the AST offset (-04:00), else None."""
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    return when if when.utcoffset() == AST_OFFSET else None


def return_event_problem(event):
    """Plain reason one returned_to_review event is malformed, or None when it is sound."""
    if not isinstance(event, dict):
        return 'an event is not an object'
    missing = [k for k in RETURN_EVENT_KEYS if k not in event]
    if missing:
        return f'an event lacks {", ".join(missing)}'
    if parse_ast_time(event['at']) is None:
        return f'an event has date {str(event["at"])[:40]!r}, not an AST ISO time'
    if not isinstance(event['by'], str) or not RETURN_INITIALS.fullmatch(event['by']):
        return f'an event has initials {str(event["by"])[:20]!r}, not two to four capital letters'
    low, high = RETURN_REASON_CHARS
    if not isinstance(event['reason'], str) or not low <= len(event['reason'].strip()) <= high:
        return f'an event reason is not {low} to {high} characters of text'
    if blank(event['tool']):
        return 'an event names no tool'
    if event['previous_exported'] is not True:
        return 'an event does not record that the frame was exported before'
    return None


def return_problems(seg):
    """Plain reasons a frame's returned_to_review list is malformed ([] when sound or absent)."""
    if RETURN_KEY not in seg:
        return []
    events = seg[RETURN_KEY]
    if not isinstance(events, list):
        return [f'{RETURN_KEY} is not a list']
    return [why for why in (return_event_problem(e) for e in events) if why]


def batch_time(batch_name):
    """AST datetime of a manifest batch_name stamp, or None when it has none."""
    m = BATCH_STAMP.search(str(batch_name or ''))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), SNAPSHOT_STAMP_FORMAT).replace(tzinfo=AST)
    except ValueError:
        return None


def returned_frames(p):
    """{frame stem: ReturnEvent} for frames sent back for review and not exported since.

    A frame counts when it is not exported and not scrapped, its
    returned_to_review list is sound, and its last event is newer than the
    batch stamp of its manifest row. A frame with no manifest row or no
    readable stamp does not count. A scrapped frame is never exported again,
    so its INFO window ends: its old label file and manifest row are then
    reported by the ordinary rules.
    """
    _, rows, err = load_csv(os.path.join(p.exp, 'export_manifest.csv'))
    exported_at = {} if err else {r.get('basename', ''): batch_time(r.get('batch_name')) for r in rows}
    out = {}
    for key, seg in p.frames().items():
        events = seg.get(RETURN_KEY)
        if seg.get('exported') or seg.get('scrapped') or not isinstance(events, list) or not events \
                or return_problems(seg):
            continue
        last, stem = events[-1], stem_of(key)
        export_time = exported_at.get(stem)
        if export_time is None or parse_ast_time(last['at']) <= export_time:
            continue
        out[stem] = ReturnEvent(last['at'], last['by'], last['reason'].strip(), str(last['tool']).strip())
    return out


# ---------------------------------------------------------------- readers
def load_json(path):
    """Read-only JSON load. Returns (object, error string or None)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, 'file not found'
    except (OSError, ValueError) as exc:
        return None, f'{type(exc).__name__}: {str(exc)[:120]}'


def load_csv(path):
    """Read-only CSV load. Returns (header list, row dicts, error or None)."""
    csv.field_size_limit(CSV_FIELD_LIMIT)
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            return header, list(reader), None
    except FileNotFoundError:
        return [], [], 'file not found'
    except (OSError, ValueError, csv.Error) as exc:
        return [], [], f'{type(exc).__name__}: {str(exc)[:120]}'


def csv_header_only(path):
    """First row of a CSV (header) without reading the rest; (header, error)."""
    csv.field_size_limit(CSV_FIELD_LIMIT)
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            return next(csv.reader(f), []), None
    except (OSError, ValueError, csv.Error) as exc:
        return [], f'{type(exc).__name__}: {str(exc)[:120]}'


def stem_of(name):
    """File name without directory or final extension."""
    return os.path.splitext(os.path.basename(str(name)))[0]


def blank(value):
    """True when value is not a string or is only whitespace."""
    return not isinstance(value, str) or not value.strip()


def parse_data_yaml(text):
    """Minimal parser for the data.yaml the exporters write.

    Returns {'path': str|None, 'nc': int|None, 'names': {int: str}}.
    Handles the block form ('names:' then '  0: PA') and the list form.
    """
    out = {'path': None, 'nc': None, 'names': {}}
    in_names = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if line[0] in ' \t':
            m = re.match(r'\s+(\d+):\s*(.*)$', line)
            if in_names and m:
                out['names'][int(m.group(1))] = m.group(2).strip().strip('\'"')
            continue
        in_names = False
        key, _, val = line.partition(':')
        key, val = key.strip(), val.strip()
        if key == 'path':
            out['path'] = val.strip('\'"')
        elif key == 'nc' and val.isdigit():
            out['nc'] = int(val)
        elif key == 'names':
            if val.startswith('['):
                items = [v.strip().strip('\'"') for v in val.strip('[]').split(',') if v.strip()]
                out['names'] = dict(enumerate(items))
            else:
                in_names = True
    return out


def read_vocabulary(path):
    """(list of codes, error). Codes are the 'code' column of the vocabulary."""
    header, rows, err = load_csv(path)
    if err:
        return [], err
    if 'code' not in header:
        return [], "no 'code' column"
    return [r.get('code', '') for r in rows], None


def mirror_path(pipeline_config):
    """The duplicate_master_codes path named in config/pipeline.yaml, or None."""
    try:
        with open(pipeline_config, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'\s*' + MIRROR_KEY + r':\s*(\S+)', line)
                if m:
                    return m.group(1).strip('\'"')
    except OSError:
        return None
    return None


def sha256_of(path):
    """Hex sha256 of a file read in chunks, or None when unreadable."""
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def load_app_code(scripts_dir):
    """Import the registry record builder and outcome calculator read-only.

    Returns (module namespace dict, error or None). The imports are pure
    functions; nothing in them is called that writes.
    """
    ns = {}
    try:
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from _reefreview import mask_registry as mr
        ns.update(REGISTRY_FIELDS=list(mr.REGISTRY_FIELDS),
                  build_registry_record=mr.build_registry_record,
                  is_test_residue=mr.is_test_residue_project_id)
        path = os.path.join(scripts_dir, 'TCRMPclip_combinedAnnotate', 'src', 'provenance.py')
        spec = importlib.util.spec_from_file_location('integrity_provenance_ro', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ns['compute_label_outcomes'] = mod.compute_label_outcomes
    except Exception as exc:  # report, never hide: the caller turns this into a finding
        return ns, f'{type(exc).__name__}: {str(exc)[:160]}'
    return ns, None


# ---------------------------------------------------------------- project context
class Project:
    """One run_* folder with its lazily read, cached files."""

    def __init__(self, folder):
        self.folder = os.path.abspath(folder)
        self.name = os.path.basename(self.folder)
        self.expected_id = self.name[len(RUN_PREFIX):] if self.name.startswith(RUN_PREFIX) else self.name
        self.pj_path = os.path.join(self.folder, 'project.json')
        self.pj, self.pj_error = load_json(self.pj_path)
        self.pid = self.pj.get('id') if isinstance(self.pj, dict) and isinstance(self.pj.get('id'), str) \
            else self.expected_id
        self.exp = os.path.join(self.folder, EXPORT_DIR_NAME)
        self.seg_path = os.path.join(self.exp, 'segmentations', 'segmentations.json')
        self._segs = None
        self._label_stats = None
        self.excluded = {}  # {frame stem: Exclusion}, filled by rule_r15
        self._returned = None

    @property
    def training(self):
        """True for a training project: it holds no annotation work of its own."""
        return isinstance(self.pj, dict) and self.pj.get('project_type') in TRAINING_PROJECT_TYPES

    @property
    def label(self):
        """Project name used in findings (the id Lauren sees)."""
        return self.expected_id

    @property
    def returned(self):
        """{frame stem: ReturnEvent} for frames waiting for their next export, read once."""
        if self._returned is None:
            self._returned = returned_frames(self)
        return self._returned

    @property
    def segs(self):
        """(segmentations dict or None, error or None), read once."""
        if self._segs is None:
            self._segs = load_json(self.seg_path)
        return self._segs

    def frames(self):
        """{filename: seg dict} for well-formed frames only."""
        data, _ = self.segs
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    def exported_stems(self):
        """Stems of frames marked exported."""
        return {stem_of(k) for k, v in self.frames().items() if v.get('exported')}

    def targets(self):
        """Target species codes from the Step 3 config, as a list."""
        try:
            raw = self.pj['steps']['3']['config']['target_species']
        except (TypeError, KeyError):
            return []
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        return [c for c in re.split(r'[,\s]+', str(raw)) if c]

    def step_status(self, number):
        """Status string of a step in project.json, or ''."""
        try:
            return str(self.pj['steps'][number].get('status', ''))
        except (TypeError, KeyError, AttributeError):
            return ''


def is_review(mask):
    """True for a REVIEW mask (the flag or the literal species)."""
    return bool(mask.get('review')) or mask.get('species') == REVIEW_CODE


def labelled_accepted(mask):
    """Accepted, non-REVIEW mask with a real species: what an export counts."""
    return (mask.get('status') == 'accepted' and not is_review(mask)
            and (mask.get('species', '') or '') != '')


def masks_of(seg):
    """The dict masks of one frame, ignoring malformed entries."""
    ms = seg.get('masks')
    return [m for m in ms if isinstance(m, dict)] if isinstance(ms, list) else []


# ---------------------------------------------------------------- R01
def rule_r01(p):
    """project.json parses, ids agree, keys are typed, step folders exist, files readable."""
    out = []
    pj = p.pj
    if p.pj_error or not isinstance(pj, dict):
        return [(FAIL, p.pj_path, f'project.json cannot be used ({p.pj_error or "not an object"}).')]
    if not p.name.startswith(RUN_PREFIX):
        out.append((FAIL, p.folder, f'Folder name does not start with "{RUN_PREFIX}".'))
    if pj.get('id') != p.expected_id:
        out.append((FAIL, p.pj_path, f'id is {pj.get("id")!r} but the folder name gives {p.expected_id!r}.'))
    pdir = pj.get('project_dir')
    if isinstance(pdir, str) and os.path.abspath(pdir) != p.folder \
            and os.path.realpath(pdir) != os.path.realpath(p.folder):
        out.append((FAIL, p.pj_path, f'project_dir points to {pdir} instead of this folder.'))
    for key, typ in PROJECT_KEYS.items():
        if key not in pj:
            out.append((FAIL, p.pj_path, f'Key "{key}" is missing.'))
        elif not isinstance(pj[key], typ) or (typ is int and isinstance(pj[key], bool)):
            out.append((FAIL, p.pj_path, f'Key "{key}" should be {typ.__name__}, found {type(pj[key]).__name__}.'))
    out += _r01_access_log(p)
    out += _r01_steps(p)
    out += _r01_permissions(p)
    return out


def _r01_access_log(p):
    """access_log entries each carry a non-blank ts, initials and action."""
    log = p.pj.get('access_log')
    if not isinstance(log, list):
        return []
    bad = [i for i, e in enumerate(log)
           if not isinstance(e, dict) or any(blank(e.get(k)) for k in ACCESS_LOG_KEYS)]
    if bad:
        return [(FAIL, p.pj_path, f'{len(bad)} access_log entries lack ts, initials or action '
                                  f'(positions {_examples(bad)}).')]
    return []


def _r01_steps(p):
    """A step marked completed or running has its folder on disk."""
    steps = p.pj.get('steps')
    if not isinstance(steps, dict):
        return []
    out = []
    for num, st in steps.items():
        if not isinstance(st, dict):
            out.append((FAIL, p.pj_path, f'Step {num} entry is not an object.'))
            continue
        if st.get('status') not in ACTIVE_STEP_STATES:
            continue
        d = st.get('dir')
        if blank(d):
            out.append((FAIL, p.pj_path, f'Step {num} is {st["status"]} but names no folder.'))
        elif not os.path.isdir(os.path.join(p.folder, os.path.basename(d.rstrip('/')))):
            out.append((FAIL, os.path.join(p.folder, os.path.basename(d.rstrip('/'))),
                        f'Step {num} is {st["status"]} but its folder is missing.'))
    return out


def _r01_permissions(p):
    """WARN for project.json and annotation files the group cannot read (mode 0600)."""
    paths = [p.pj_path]
    for d in () if p.training else (p.exp, os.path.join(p.exp, 'segmentations')):
        try:
            paths += [os.path.join(d, n) for n in sorted(os.listdir(d))
                      if os.path.isfile(os.path.join(d, n))]
        except OSError:
            continue
    out = []
    for path in paths:
        try:
            mode = os.lstat(path).st_mode
        except OSError:
            continue
        if not mode & GROUP_READ_BIT:
            out.append((WARN, path, f'File is not group-readable (mode {stat.S_IMODE(mode):04o}).'))
    return out


# ---------------------------------------------------------------- R02
def _resolve_issue(path):
    """Plain reason a path does not resolve, or None when it does."""
    if os.path.exists(path):
        return None
    return 'is a broken link' if os.path.islink(path) else 'does not exist'


def _parse_issue(path):
    """Plain reason a Step output does not parse, or None."""
    if path.endswith('.json'):
        _, err = load_json(path)
        return err
    header, err = csv_header_only(path)
    return err or (None if header else 'has no header row')


def step_in_use(p, number, folder):
    """True when a step ran: marked completed or running, or (no status) its folder holds files.

    A skipped, pending or locked step (training projects skip Steps 1 to 4) has
    nothing to check.
    """
    status = p.step_status(number)
    if status:
        return status in ACTIVE_STEP_STATES
    return bool(_list_dir(folder))


def rule_r02(p):
    """Step 1 entries resolve (links too); the three Step 2 outputs exist and parse."""
    out = []
    step1 = os.path.join(p.folder, STEP_DIRS['1'])
    if step_in_use(p, '1', step1):
        listed = []
        try:
            listed = [v for v in p.pj['steps']['1'].get('outputs', {}).values() if isinstance(v, str)]
        except (TypeError, KeyError, AttributeError):
            pass
        for path in [os.path.join(step1, n) for n in STEP1_FILES] + listed:
            why = _resolve_issue(path) or _parse_issue(path)
            if why:
                out.append((FAIL, path, f'Step 1 file {why}.'))
    step2 = os.path.join(p.folder, STEP_DIRS['2'])
    if step_in_use(p, '2', step2):
        for n in STEP2_FILES:
            path = os.path.join(step2, n)
            why = _resolve_issue(path) or _parse_issue(path)
            if why:
                out.append((FAIL, path, f'Step 2 output {why}.'))
    return out


# ---------------------------------------------------------------- R03
def rule_r03(p, vocab):
    """Step 3 selection: unique basenames, known targets, not newer than the prompts."""
    out = []
    sel = os.path.join(p.folder, STEP_DIRS['3'], 'selected_frames.csv')
    if not step_in_use(p, '3', os.path.dirname(sel)):
        return out
    header, rows, err = load_csv(sel)
    if err:
        out.append((FAIL, sel, f'selected_frames.csv cannot be read ({err}).'))
    elif 'basename' not in header:
        out.append((FAIL, sel, 'selected_frames.csv has no basename column.'))
    else:
        counts = Counter(r.get('basename', '') for r in rows)
        dups = [b for b, n in counts.items() if n > 1]
        if dups:
            out.append((FAIL, sel, f'{len(dups)} basenames appear more than once: {_examples(dups)}.'))
    unknown = [c for c in p.targets() if c not in vocab and c != REVIEW_CODE]
    if unknown:
        out.append((FAIL, p.pj_path, f'Target codes not in the vocabulary: {_examples(unknown)}.'))
    out += _r03_stale(p, sel)
    return out


def _r03_stale(p, sel):
    """WARN when the selection is newer than the click prompts while frames are reviewed."""
    prompts = os.path.join(p.exp, 'routed_input', 'ids', 'sam_click_prompts.json')
    try:
        newer = os.stat(sel).st_mtime > os.stat(prompts).st_mtime
    except OSError:
        return []
    worked = [k for k, v in p.frames().items() if v.get('reviewed') or v.get('exported')]
    if newer and worked:
        return [(WARN, sel, f'selected_frames.csv is newer than sam_click_prompts.json while '
                            f'{len(worked)} frames are reviewed or exported; a re-run of Step 3 may '
                            f'have changed the frame list.')]
    return []


# ---------------------------------------------------------------- R04
def rule_r04(p):
    """segmentations.json: object frames, masks lists, mask keys, species, unique ids."""
    data, err = p.segs
    siblings = [os.path.join(p.exp, n) for n in ('export_manifest.csv', 'label_provenance.csv', 'all_labels')]
    if err == 'file not found':
        if any(os.path.exists(s) for s in siblings):
            return [(FAIL, p.seg_path, 'segmentations.json is missing although exports exist.')]
        return []
    if err:
        return [(FAIL, p.seg_path, f'segmentations.json cannot be read ({err}).')]
    if not isinstance(data, dict):
        return [(FAIL, p.seg_path, 'segmentations.json is not an object keyed by image.')]
    return _r04_frames(p, data)


def _r04_frames(p, data):
    """Frame-by-frame structure checks, aggregated into a few findings."""
    shape, keys, blank_sp, dup_ids, bad_returns = [], [], [], [], []
    for fn, seg in data.items():
        problems = return_problems(seg) if isinstance(seg, dict) else []
        if problems:
            bad_returns.append(f'{fn} ({problems[0]})')
        if not isinstance(seg, dict) or not isinstance(seg.get('masks'), list):
            shape.append(fn)
            continue
        seen = set()
        for m in seg['masks']:
            if not isinstance(m, dict) or any(k not in m for k in MASK_KEYS):
                keys.append(fn)
                continue
            if m['id'] in seen:
                dup_ids.append(f'{fn}#{m["id"]}')
            seen.add(m['id'])
            if m.get('status') == 'accepted' and not is_review(m) and blank(m.get('species')):
                blank_sp.append(f'{fn}#{m["id"]}')
    out = []
    for items, text in ((shape, 'frames are not objects with a masks list'),
                        (keys, 'frames hold masks missing id, species or status'),
                        (blank_sp, 'accepted masks have a blank species'),
                        (dup_ids, 'mask ids repeat inside a frame'),
                        (bad_returns, f'frames carry a malformed {RETURN_KEY} record')):
        if items:
            out.append((FAIL, p.seg_path, f'{len(set(items))} {text}: {_examples(sorted(set(items)))}.'))
    return out


# ---------------------------------------------------------------- R05 / R06
def _list_dir(path):
    """Sorted names in a directory, or [] when it is absent."""
    try:
        return sorted(os.listdir(path))
    except OSError:
        return []


def rule_r05(p):
    """all_images links resolve; image count matches the manifest; images pair with labels."""
    img_dir = os.path.join(p.exp, 'all_images')
    lab_dir = os.path.join(p.exp, 'all_labels')
    if not os.path.isdir(img_dir) and not os.path.isdir(lab_dir):
        return []
    out = []
    images = _list_dir(img_dir)
    broken = [n for n in images if not os.path.exists(os.path.join(img_dir, n))]
    if broken:
        out.append((FAIL, img_dir, f'{len(broken)} image links do not resolve: {_examples(broken)}.'))
    header, rows, err = load_csv(os.path.join(p.exp, 'export_manifest.csv'))
    if not err:
        n_rows = sum(1 for r in rows if r.get('outcome') in EXPORTED_OUTCOMES)
        if len(images) != n_rows:
            out.append((FAIL, img_dir, f'{len(images)} images but {n_rows} manifest rows are '
                                       f'exported_with_masks or exported_empty.'))
    img_stems = {stem_of(n) for n in images}
    lab_stems = {stem_of(n) for n in _list_dir(lab_dir) if n.endswith('.txt')}
    no_label, no_image = sorted(img_stems - lab_stems), sorted(lab_stems - img_stems)
    if no_label:
        out.append((FAIL, lab_dir, f'{len(no_label)} images have no label file: {_examples(no_label)}.'))
    if no_image:
        out.append((FAIL, img_dir, f'{len(no_image)} label files have no image: {_examples(no_image)}.'))
    return out


def check_label_file(path, class_ids):
    """Validate one YOLO label file. Returns (n_lines, Counter by class, problems)."""
    problems, counts, n = [], Counter(), 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                n += 1
                problem = _label_line_problem(line.split(), class_ids)
                if problem:
                    problems.append(f'line {ln} {problem}')
                else:
                    counts[int(line.split()[0])] += 1
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f'unreadable ({type(exc).__name__})')
    return n, counts, problems


def _label_line_problem(tokens, class_ids):
    """Plain reason a split label line is invalid, or None."""
    if not tokens[0].isdigit():
        return 'class is not an integer'
    if class_ids is not None and int(tokens[0]) not in class_ids:
        return f'class {tokens[0]} is outside class_map'
    coords = tokens[1:]
    if len(coords) < MIN_POLYGON_COORDS or len(coords) % 2:
        return f'has {len(coords)} coordinates (need an even number, at least {MIN_POLYGON_COORDS})'
    try:
        if any(not 0.0 <= float(c) <= 1.0 for c in coords):
            return 'has a coordinate outside 0 to 1'
    except ValueError:
        return 'has a non-numeric coordinate'
    return None


def read_class_map(path):
    """(class_map dict or None, list of problems)."""
    cm, err = load_json(path)
    if err:
        return None, [f'class_map.json cannot be read ({err})']
    if not isinstance(cm, dict) or any(isinstance(v, bool) or not isinstance(v, int) for v in cm.values()):
        return None, ['class_map.json is not a name-to-integer object']
    problems = []
    blanks = [k for k in cm if not k.strip()]
    if blanks:
        problems.append(f'class_map.json has {len(blanks)} blank class names')
    if sorted(cm.values()) != list(range(len(cm))):
        problems.append(f'class ids {sorted(cm.values())} do not run 0..{len(cm) - 1}')
    return cm, problems


def rule_r06(p):
    """YOLO label lines, class_map.json (0..n-1, named) and data.yaml agree."""
    lab_dir = os.path.join(p.exp, 'all_labels')
    cm_path = os.path.join(p.exp, 'class_map.json')
    yaml_path = os.path.join(p.exp, 'data.yaml')
    if not (os.path.isdir(lab_dir) or os.path.exists(cm_path) or os.path.exists(yaml_path)):
        return []
    out = []
    cm, cm_problems = read_class_map(cm_path)
    out += [(FAIL, cm_path, f'{t}.') for t in cm_problems]
    ids = set(cm.values()) if cm is not None else None
    bad, stats = [], {}
    for n in _list_dir(lab_dir):
        if not n.endswith('.txt'):
            continue
        lines, counts, problems = check_label_file(os.path.join(lab_dir, n), ids)
        stats[stem_of(n)] = (lines, counts)
        bad += [f'{n} {t}' for t in problems]
    p._label_stats = stats
    if bad:
        out.append((FAIL, lab_dir, f'{len(bad)} label problems: {_examples(bad)}.'))
    out += _r06_yaml(p, cm, yaml_path)
    return out


def _r06_yaml(p, cm, yaml_path):
    """data.yaml path, nc and names agree with class_map.json."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            y = parse_data_yaml(f.read())
    except (OSError, UnicodeDecodeError) as exc:
        return [(FAIL, yaml_path, f'data.yaml cannot be read ({type(exc).__name__}).')]
    out = []
    if y['path'] and os.path.realpath(y['path']) != os.path.realpath(p.exp):
        out.append((FAIL, yaml_path, f'data.yaml path is {y["path"]}, not this export folder.'))
    if cm is not None:
        if y['nc'] != len(cm):
            out.append((FAIL, yaml_path, f'data.yaml nc is {y["nc"]} but class_map has {len(cm)} classes.'))
        if y['names'] != {v: k for k, v in cm.items()}:
            out.append((FAIL, yaml_path, 'data.yaml names differ from class_map.json.'))
    return out


# ---------------------------------------------------------------- R07
def label_stats(p):
    """{stem: (n lines, Counter by class id)} from R06, computed on demand."""
    if p._label_stats is None:
        rule_r06(p)
        p._label_stats = p._label_stats or {}
    return p._label_stats


def expected_counts(seg):
    """Counter species -> accepted, labelled, non-REVIEW masks of one frame."""
    return Counter(m['species'] for m in masks_of(seg) if labelled_accepted(m))


def rule_r07(p):
    """export_manifest.csv matches the exported frames, mask counts and label files."""
    path = os.path.join(p.exp, 'export_manifest.csv')
    header, rows, err = load_csv(path)
    if err == 'file not found':
        if p.exported_stems():
            return [(FAIL, path, f'{len(p.exported_stems())} frames are marked exported but there is no export_manifest.csv.')]
        return []
    if err:
        return [(FAIL, path, f'export_manifest.csv cannot be read ({err}).')]
    missing_cols = [c for c in MANIFEST_BASE if c not in header]
    if missing_cols:
        return [(FAIL, path, f'export_manifest.csv lacks columns {_examples(missing_cols)}.')]
    out = []
    stems = [r.get('basename', '') for r in rows]
    dups = [s for s, n in Counter(stems).items() if n > 1]
    if dups:
        out.append((FAIL, path, f'{len(dups)} manifest basenames repeat: {_examples(dups)}.'))
    exported = p.exported_stems()
    if p.segs[1] is None:
        only_m, only_f = sorted(set(stems) - exported), sorted(exported - set(stems))
        only_m, sent_back = split_excepted(only_m, p.returned)
        out += returned_findings(sent_back, path, 'manifest rows are for frames not marked exported')
        if only_m:
            out.append((FAIL, path, f'{len(only_m)} manifest rows are for frames not marked exported: {_examples(only_m)}.'))
        if only_f:
            out.append((FAIL, path, f'{len(only_f)} exported frames have no manifest row: {_examples(only_f)}.'))
    missing = [r['basename'] for r in rows if r.get('outcome') == 'image_missing']
    if missing:
        out.append((FAIL, path, f'{len(missing)} rows are image_missing, so their frames export no label: {_examples(missing)}.'))
    unknown = sorted({r.get('outcome', '') for r in rows} - set(MANIFEST_OUTCOMES))
    if unknown:
        out.append((FAIL, path, f'Unknown outcomes in manifest: {_examples(unknown)}.'))
    out += _r07_counts(p, header, rows, path)
    return out


def _r07_counts(p, header, rows, path):
    """Per-row n_masks, species columns and label-file line counts."""
    frames = {stem_of(k): v for k, v in p.frames().items()}
    stats = label_stats(p)
    species_cols = [c for c in header if c not in MANIFEST_BASE]
    bad_species, bad_n, bad_lines = [], [], []
    for r in rows:
        if r.get('outcome') not in EXPORTED_OUTCOMES or r.get('basename') not in frames:
            continue
        stem = r['basename']
        exp = expected_counts(frames[stem])
        got = {c: _int(r.get(c)) for c in species_cols}
        if any(got.get(c, 0) != exp.get(c, 0) for c in set(exp) | set(got)):
            bad_species.append(stem)
        n_manifest = _int(r.get('n_masks'))
        if n_manifest != sum(exp.values()):
            bad_n.append(stem)
        lines = stats.get(stem, (None, None))[0]
        if lines is not None and lines != n_manifest:
            bad_lines.append(stem)
    out = []
    for items, text, stored_state in (
            (bad_species, 'rows whose species counts differ from the accepted labelled masks', True),
            (bad_n, 'rows whose n_masks differs from the accepted labelled masks', True),
            (bad_lines, 'rows whose label file has a different number of lines than n_masks', False)):
        if stored_state:  # a frame sent back for review is expected to differ until its next export
            items, sent_back = split_excepted(items, p.returned)
            out += returned_findings(sent_back, path, text)
        if items:
            out.append((FAIL, path, f'{len(items)} {text}: {_examples(items)}.'))
    return out


def _int(value):
    """int(value) or -1 for blank or non-numeric manifest cells."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return -1 if str(value or '').strip() else 0


# ---------------------------------------------------------------- R08
def rule_r08(p, vocab, app):
    """label_provenance.csv: header, keys, values, and agreement with outcomes."""
    path = os.path.join(p.exp, 'label_provenance.csv')
    header, rows, err = load_csv(path)
    frames = {stem_of(k): v for k, v in p.frames().items()}
    scrapped = {s for s, v in frames.items() if v.get('scrapped')}
    stored = {s: v.get('label_outcomes') for s, v in frames.items()
              if s not in scrapped and isinstance(v.get('label_outcomes'), dict) and v['label_outcomes']}
    if err == 'file not found':
        unledgered, covered = split_excepted(sorted(stored), p.excluded)
        out = excepted_findings(covered, path, 'frames hold label outcomes and there is no ledger')
        if unledgered:
            out.append((FAIL, path, f'{len(unledgered)} frames hold label outcomes but there is no ledger.'))
        return out
    if err:
        return [(FAIL, path, f'label_provenance.csv cannot be read ({err}).')]
    if header != LEDGER_HEADER:
        return [(FAIL, path, f'Ledger header is {",".join(header)}, expected {",".join(LEDGER_HEADER)}.')]
    kept = [r for r in rows if r['basename'] not in scrapped]
    out = _r08_values(p, kept, frames, vocab, path)
    out += broken_rulings([r['basename'] for r in rows], p.excluded, path, 'ledger rows')
    out += _r08_against_outcomes(kept, stored, path, p.excluded, p.returned)
    out += _r08_computed(p, frames, stored, app, path)
    out += _r08_frame_flags(frames, scrapped, rows, path)
    return out


def _r08_frame_flags(frames, scrapped, rows, path):
    """WARN for ledger rows left on scrapped frames and for unconfirmed writes.

    A scrapped frame should have no ledger rows. A frame still carrying
    'ledger_pending' (ledger write not confirmed) or 'image_missing_at'
    (exported without its image) needs a person to look at it.
    """
    out = []
    left = sorted({r['basename'] for r in rows if r['basename'] in scrapped})
    if left:
        out.append((WARN, path, f'{len(left)} scrapped frames still have ledger rows: {_examples(left)}.'))
    for key, text in (('ledger_pending', 'frames have an unconfirmed ledger write'),
                      ('image_missing_at', 'frames were exported without their image')):
        flagged = sorted(s for s, v in frames.items() if v.get(key))
        if flagged:
            out.append((WARN, path, f'{len(flagged)} {text} ({key}): {_examples(flagged)}.'))
    return out


def _r08_values(p, rows, frames, vocab, path):
    """Unique keys, existing reviewed frames, known labels and outcomes, own project."""
    keys = Counter((r['basename'], r['label']) for r in rows)
    dups = [f'{b}/{lab}' for (b, lab), n in keys.items() if n > 1]
    no_frame = sorted({r['basename'] for r in rows if r['basename'] not in frames and frames})
    not_done = sorted({r['basename'] for r in rows if r['basename'] in frames
                       and not (frames[r['basename']].get('reviewed') or frames[r['basename']].get('exported'))})
    blank_rows = [r['basename'] for r in rows if blank(r['label'])]
    bad_label = sorted({r['label'] for r in rows if not blank(r['label'])
                        and r['label'] not in vocab and r['label'] != REVIEW_CODE})
    bad_outcome = sorted({r['outcome'] for r in rows if r['outcome'] not in LEDGER_OUTCOMES})
    bad_proj = sorted({r['project_id'] for r in rows if r['project_id'] != p.pid})
    # A frame sent back for review keeps the ledger rows of its last export until the next one.
    not_done, sent_back = split_excepted(not_done, p.returned)
    out = returned_findings(sent_back, path, 'ledger rows are for frames neither reviewed nor exported')
    for items, text in ((dups, 'duplicate (basename, label) rows'),
                        (no_frame, 'rows for frames that do not exist'),
                        (not_done, 'rows for frames neither reviewed nor exported'),
                        (blank_rows, 'rows with a blank label (frames)'),
                        (bad_label, 'labels outside the vocabulary'),
                        (bad_outcome, 'unknown outcomes'),
                        (bad_proj, 'project_id values that are not this project')):
        if items:
            out.append((FAIL, path, f'{len(items)} {text}: {_examples(items)}.'))
    return out


def _r08_against_outcomes(rows, stored, path, excluded=None, returned=None):
    """Ledger rows equal the frame's stored label_outcomes.

    A frame a ruling keeps out of the coverage matrix (excluded: {stem:
    Exclusion}) is expected to have no ledger rows; that finding is INFO. A
    frame sent back for review (returned: {stem: ReturnEvent}) may differ
    until its next export; those findings are INFO too.
    """
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r['basename'], {})[r['label']] = r['outcome']
    unledgered, covered = split_excepted(sorted(s for s in stored if s not in by_frame), excluded or {})
    missing, extra, differs = [], [], []
    for s, outcomes in stored.items():
        if s not in by_frame:
            continue
        have = by_frame[s]
        for code, o in outcomes.items():
            if not isinstance(o, dict):
                continue
            if code not in have:
                missing.append(f'{s}/{code}')
            elif have[code] != o.get('outcome'):
                differs.append(f'{s}/{code}')
        extra += [f'{s}/{c}' for c in have if c not in outcomes]
    out = excepted_findings(covered, path, 'frames hold label outcomes but have no ledger rows')
    for items, text in ((unledgered, 'frames hold label outcomes but have no ledger rows'),
                        (missing, 'stored outcomes have no ledger row'),
                        (extra, 'ledger rows have no stored outcome'),
                        (differs, 'ledger outcomes differ from the stored outcome')):
        items, sent_back = split_excepted(items, returned or {}, frame_of)
        out += returned_findings(sent_back, path, text)
        if items:
            out.append((FAIL, path, f'{len(items)} {text}: {_examples(items)}.'))
    return out


def _r08_computed(p, frames, stored, app, path):
    """WARN when stored outcomes differ from what compute_label_outcomes gives now."""
    fn = app.get('compute_label_outcomes')
    targets = p.targets()
    if fn is None or not targets or not stored:
        return []
    ps, _ = load_json(os.path.join(p.exp, 'prior_screening.json'))
    pframes = ps.get('frames', {}) if isinstance(ps, dict) and isinstance(ps.get('frames'), dict) else {}
    differ, errors = [], []
    for stem, outcomes in stored.items():
        screened = pframes.get(stem, {}).get('screened', {}) if isinstance(pframes.get(stem), dict) else {}
        try:
            calc = fn(frames[stem], targets, 'integrity-check', '', screened_elsewhere=list(screened))
        except Exception as exc:  # a frame the app code cannot score is itself a finding
            errors.append(f'{stem} ({type(exc).__name__})')
            continue
        have = {c: o.get('outcome') for c, o in outcomes.items() if isinstance(o, dict)}
        if have != {c: o['outcome'] for c, o in calc.items()}:
            differ.append(stem)
    text = 'frames whose stored outcomes differ from a fresh computation from their masks'
    differ, sent_back = split_excepted(differ, p.returned)
    out = returned_findings(sent_back, path, text)
    if differ or errors:
        out.append((WARN, path, f'{len(differ) + len(errors)} {text}: {_examples(differ + errors)}.'))
    return out


# ---------------------------------------------------------------- R09
def project_ids(root):
    """{id: folder} for run_* folders directly under root."""
    return {n[len(RUN_PREFIX):]: os.path.join(root, n) for n in _list_dir(root)
            if n.startswith(RUN_PREFIX) and os.path.isdir(os.path.join(root, n))}


def rule_r09(cfg, projects, live, archived, app):
    """Registry header, unique uids, live project ids and per-project agreement.

    Returns {project label or '': [(level, path, message)]}.
    """
    path = os.path.join(cfg.root, '_mask_registry', 'manifest.csv')
    header, rows, err = load_csv(path)
    if err == 'file not found':
        return {'': [(FAIL, path, 'The mask registry manifest.csv is missing.')]}
    if err:
        return {'': [(FAIL, path, f'The mask registry cannot be read ({err}).')]}
    fields = app.get('REGISTRY_FIELDS')
    if fields is None:
        return {'': [(FAIL, path, 'The registry header could not be loaded from mask_registry.py.')]}
    if header != fields:
        return {'': [(FAIL, path, f'Registry header is {",".join(header)}, expected {",".join(fields)}.')]}
    res = {'': _r09_global(rows, live, archived, app, path)}
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r['project_id'], []).append(r)
    for p in projects:
        res[p.label] = _r09_project(p, by_pid.get(p.pid, []), app, path)
    return res


def _r09_global(rows, live, archived, app, path):
    """Unique uids and project ids that are live, archived, fixture residue or unknown."""
    out = []
    dups = [u for u, n in Counter(r['uid'] for r in rows).items() if n > 1]
    if dups:
        out.append((FAIL, path, f'{len(dups)} registry uids appear more than once: {_examples(dups)}.'))
    counts = Counter(r['project_id'] for r in rows)
    is_fixture = app.get('is_test_residue', lambda _x: False)
    fixture = sorted(i for i in counts if i not in live and i not in archived and is_fixture(i))
    arch = sorted(i for i in counts if i not in live and i in archived)
    unknown = sorted(i for i in counts if i not in live and i not in archived and i not in fixture)
    if fixture:
        out.append((FAIL, path, f'{sum(counts[i] for i in fixture)} registry rows belong to test-fixture '
                                f'projects: {_examples([f"{i} ({counts[i]})" for i in fixture])}.'))
    if arch:
        out.append((WARN, path, f'{sum(counts[i] for i in arch)} registry rows belong to archived projects: '
                                f'{_examples([f"{i} ({counts[i]})" for i in arch])}.'))
    if unknown:
        out.append((WARN, path, f'{sum(counts[i] for i in unknown)} registry rows belong to projects found '
                                f'nowhere: {_examples([f"{i} ({counts[i]})" for i in unknown])}.'))
    return out


def _r09_project(p, reg_rows, app, path):
    """A project's exported masks equal its registry rows, field by field."""
    build = app.get('build_registry_record')
    exported = {k: v for k, v in p.frames().items() if v.get('exported')}
    if not exported:
        return []
    if build is None:
        return [(FAIL, path, 'The registry record builder could not be loaded.')]
    expected = {}
    for fn, seg in exported.items():
        for m in masks_of(seg):
            try:
                rec = build(fn, m, p.pid, p.pid, reviewer=m.get('reviewer') or '')
            except Exception:  # a mask the builder rejects has no registry row to expect
                rec = None
            if rec:
                expected[rec['uid']] = rec
    stems = {stem_of(k) for k in exported}
    have = {r['uid']: r for r in reg_rows if r['basename'] in stems}
    missing, covered = split_excepted(sorted(set(expected) - set(have)), p.excluded,
                                      lambda u: expected[u]['basename'])
    extra = sorted(set(have) - set(expected))
    differs = [u for u in set(expected) & set(have)
               if any(str(expected[u].get(k, '')) != have[u].get(k, '')
                      for k in ('species', 'status', 'source_type', 'review', 'expert_mode'))]
    out = broken_rulings([r['basename'] for r in reg_rows], p.excluded, path, 'registry rows')
    out += excepted_findings(covered, path, 'exported masks have no registry row')
    for items, text in ((missing, 'exported masks have no registry row'),
                        (extra, 'registry rows have no exported mask'),
                        (sorted(differs), 'registry rows differ from the mask in species, status, source, review or expert mode')):
        if items:
            out.append((FAIL, path, f'{len(items)} {text}: {_examples(items)}.'))
    return out


# ---------------------------------------------------------------- R10
def rule_r10(cfg, live, archived):
    """Each _composite training set: links, labels, line counts, source projects.

    Returns {composite name: [(level, path, message)]}.
    """
    base = os.path.join(cfg.root, '_composite')
    res = {}
    for name in _list_dir(base):
        d = os.path.join(base, name)
        if os.path.isdir(d):
            res[name] = _r10_one(d, live, archived)
    return res


def _r10_one(d, live, archived):
    """Checks for one composite directory."""
    out = []
    img_dir = os.path.join(d, 'all_images')
    broken = [n for n in _list_dir(img_dir) if not os.path.exists(os.path.join(img_dir, n))]
    if broken:
        out.append((FAIL, img_dir, f'{len(broken)} composite links do not resolve: {_examples(broken)}.'))
    cm, cm_problems = read_class_map(os.path.join(d, 'class_map.json'))
    out += [(FAIL, os.path.join(d, 'class_map.json'), f'{t}.') for t in cm_problems]
    mpath = os.path.join(d, 'composite_manifest.csv')
    header, rows, err = load_csv(mpath)
    if err:
        return out + [(FAIL, mpath, f'composite_manifest.csv cannot be read ({err}).')]
    if cm is not None:
        per_class = Counter()
        for n in _list_dir(os.path.join(d, 'all_labels')):
            _, counts, _ = check_label_file(os.path.join(d, 'all_labels', n), None)
            per_class.update(counts)
        rows_by = Counter(r.get('label', '') for r in rows)
        off = [f'{lab} (lines {per_class.get(cid, 0)}, rows {rows_by.get(lab, 0)})'
               for lab, cid in cm.items() if per_class.get(cid, 0) != rows_by.get(lab, 0)]
        if off:
            out.append((FAIL, d, f'Label lines differ from manifest rows for {_examples(off)}.'))
    ghosts = sorted({r.get('project_id', '') for r in rows} - set(live))
    dead = [g for g in ghosts if g not in archived]
    arch = [g for g in ghosts if g in archived]
    if dead:
        out.append((FAIL, mpath, f'Source projects not live: {_examples(dead)}.'))
    if arch:
        out.append((WARN, mpath, f'Source projects only in the archive: {_examples(arch)}.'))
    return out


# ---------------------------------------------------------------- R11
def collect_codes(projects, cfg, ids_seen):
    """Add every species code used anywhere to ids_seen: {code: set of places}."""
    for p in projects:
        for fn, seg in p.frames().items():
            for m in masks_of(seg):
                if isinstance(m.get('species'), str) and m['species'].strip():
                    ids_seen.setdefault(m['species'], set()).add(f'{p.label} masks')
        cm, _ = read_class_map(os.path.join(p.exp, 'class_map.json'))
        for k in (cm or {}):
            ids_seen.setdefault(k, set()).add(f'{p.label} class_map')
        _, rows, _ = load_csv(os.path.join(p.exp, 'label_provenance.csv'))
        for r in rows:
            if r.get('label'):
                ids_seen.setdefault(r['label'], set()).add(f'{p.label} ledger')
    _, rows, _ = load_csv(os.path.join(cfg.root, '_mask_registry', 'manifest.csv'))
    for r in rows:
        if r.get('species'):
            ids_seen.setdefault(r['species'], set()).add('registry')
    comp = os.path.join(cfg.root, '_composite')
    for n in _list_dir(comp):
        cm, _ = read_class_map(os.path.join(comp, n, 'class_map.json'))
        for k in (cm or {}):
            ids_seen.setdefault(k, set()).add(f'composite {n}')


def rule_r11(cfg, vocab_codes, vocab_err, used):
    """Vocabulary unique, mirror identical, every code in use known."""
    out = []
    if vocab_err:
        return [(FAIL, cfg.vocab_path, f'The species vocabulary cannot be read ({vocab_err}).')]
    dups = [c for c, n in Counter(vocab_codes).items() if n > 1]
    blanks = [c for c in vocab_codes if blank(c)]
    if dups:
        out.append((FAIL, cfg.vocab_path, f'Vocabulary codes repeat: {_examples(dups)}.'))
    if blanks:
        out.append((FAIL, cfg.vocab_path, f'{len(blanks)} vocabulary rows have a blank code.'))
    mirror = mirror_path(cfg.pipeline_config)
    if not mirror:
        out.append((FAIL, cfg.pipeline_config, f'pipeline.yaml does not name {MIRROR_KEY}.'))
    elif sha256_of(mirror) is None:
        out.append((FAIL, mirror, 'The hopper mirror of the vocabulary cannot be read.'))
    elif sha256_of(mirror) != sha256_of(cfg.vocab_path):
        out.append((FAIL, mirror, 'The hopper mirror differs from the vocabulary file.'))
    unknown = {c: w for c, w in used.items() if c not in set(vocab_codes) and c != REVIEW_CODE}
    if unknown:
        shown = [f'{c!r} ({_examples(sorted(w), 2)})' for c, w in sorted(unknown.items())]
        out.append((FAIL, cfg.vocab_path, f'{len(unknown)} codes in use are not in the vocabulary: '
                                          f'{_examples(shown)}.'))
    return out


# ---------------------------------------------------------------- R12
def rule_r12(cfg):
    """WARN for any manual-source list in the known files that omits "merged"."""
    files = cfg.manual_source_files
    if files is None:
        files = [os.path.join(cfg.scripts_dir, rel) for rel in KNOWN_MANUAL_SOURCE_FILES]
    out = []
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            out.append((WARN, path, f'Cannot read the file to check its manual-source list ({type(exc).__name__}).'))
            continue
        for i, line in enumerate(lines):
            window = '\n'.join(lines[max(0, i - CONSTANT_CONTEXT_LINES): i + CONSTANT_CONTEXT_LINES + 1])
            if MANUAL_CLICK_LITERAL.search(line) and MANUAL_BOX_LITERAL.search(window) \
                    and not MERGED_LITERAL.search(window):
                out.append((WARN, f'{path}:{i + 1}', 'A manual-source list names manual_box and manual_click but not "merged".'))
    return out


# ---------------------------------------------------------------- R13
def stray_tmp_files(root, now_ts):
    """Paths of .tmp files under root older than an hour (links not followed)."""
    found = []
    for dp, dns, fns in os.walk(root, followlinks=False):
        # Moved-aside folders are history, not live files (images are never deleted).
        dns[:] = [d for d in dns if d != SUPERSEDED_DIR_NAME]
        for n in fns:
            if n.endswith('.tmp') or '.tmp.' in n:
                full = os.path.join(dp, n)
                try:
                    if now_ts - os.lstat(full).st_mtime > STALE_TMP_SECONDS:
                        found.append(full)
                except OSError:
                    continue
    return found


def rule_r13(p, now_ts):
    """Stale .tmp files, conflict files and prior_screening.json for one project."""
    out = []
    stale = stray_tmp_files(p.folder, now_ts)
    if stale:
        out.append((FAIL, p.folder, f'{len(stale)} .tmp files are over an hour old: {_examples([os.path.relpath(s, p.folder) for s in stale])}.'))
    segdir = os.path.join(p.exp, 'segmentations')
    for n in _list_dir(segdir):
        if CONFLICT_GLOB.match(n):
            data, err = load_json(os.path.join(segdir, n))
            frames = sorted(data) if isinstance(data, dict) else []
            out.append((WARN, os.path.join(segdir, n),
                        f'Conflict file from a lost write race holds {len(frames)} frames: {_examples(frames)}'
                        + (f' (unreadable: {err})' if err else '') + '.'))
    ps_path = os.path.join(p.exp, 'prior_screening.json')
    if os.path.exists(ps_path):
        ps, err = load_json(ps_path)
        if err:
            out.append((FAIL, ps_path, f'prior_screening.json cannot be read ({err}).'))
        elif not isinstance(ps, dict) or ps.get('version') != PRIOR_SCREENING_VERSION:
            out.append((FAIL, ps_path, f'prior_screening.json is not version {PRIOR_SCREENING_VERSION}.'))
        elif ps.get('project_id') != p.pid:
            out.append((FAIL, ps_path, f'prior_screening.json names project {ps.get("project_id")!r}, not {p.pid!r}.'))
    return out


def rule_r13_stores(cfg, now_ts):
    """Stale .tmp files in the shared stores."""
    out = []
    for name in SHARED_STORES:
        d = os.path.join(cfg.root, name)
        if os.path.isdir(d):
            stale = stray_tmp_files(d, now_ts)
            if stale:
                out.append((FAIL, d, f'{len(stale)} .tmp files are over an hour old: {_examples(stale)}.'))
    return out


# ---------------------------------------------------------------- R14
def rule_r14_snapshots(cfg, now):
    """Newest nightly snapshot is under 26 hours old."""
    names = [n for n in _list_dir(cfg.snapshot_dir) if SNAPSHOT_STAMP.match(n)]
    if not names:
        return [(FAIL, cfg.snapshot_dir, 'No nightly snapshot folders found.')]
    newest = names[-1]
    stamp = datetime.strptime(newest, SNAPSHOT_STAMP_FORMAT).replace(tzinfo=AST)
    age = now - stamp
    if age > SNAPSHOT_MAX_AGE:
        return [(FAIL, os.path.join(cfg.snapshot_dir, newest),
                 f'Newest snapshot is {age.total_seconds() / 3600:.1f} hours old (limit 26).')]
    return []


def rule_r14_archive(cfg):
    """{run id: [(WARN, path, message)]} for archive/ runs outside the nightly copy."""
    out = {}
    for pid, path in project_ids(cfg.archive_dir).items():
        out[pid] = [(WARN, path, 'This archived run is not covered by the nightly snapshot copy.')]
    return out


# ---------------------------------------------------------------- R15
def load_exceptions(path):
    """Read integrity_exceptions.yaml read-only and check each entry's own fields.

    The file is {version: 1, exceptions: [{project_id, ruling, date, by,
    frames: [stem, ...]}, ...]}; a date is YYYY-MM-DD.

    Parameters:
        path: the exceptions file.
    Returns:
        (entries, problems): entries is a list of (entry number, Exclusion,
        [frame stems]) for entries whose fields are sound; problems is a list
        of plain sentences. A missing file is ([], []): nothing is excepted.
    """
    if not os.path.exists(path):
        return [], []
    try:
        import yaml  # PyYAML is in the module env; imported here so the other rules never need it
    except ImportError:
        return [], ['PyYAML is not installed, so the exceptions file cannot be read']
    try:
        with open(path, 'r', encoding='utf-8') as f:
            doc = yaml.safe_load(f)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError, RecursionError) as exc:
        # ValueError and RecursionError: YAML that parses but cannot be built
        # (a date no calendar holds, an integer too long, nesting too deep).
        return [], [f'The exceptions file cannot be read ({type(exc).__name__}: {str(exc)[:120]})']
    if not isinstance(doc, dict) or type(doc.get('version')) is not int \
            or doc['version'] != EXCEPTIONS_VERSION or not isinstance(doc.get('exceptions'), list):
        return [], [f'The exceptions file is not a version {EXCEPTIONS_VERSION} document '
                    f'with an exceptions list']
    entries, problems = [], []
    for n, raw in enumerate(doc['exceptions'], 1):
        exc, frames, why = _exception_entry(raw)
        if why:
            problems.append(f'Entry {n} {why}')
        else:
            entries.append((n, exc, frames))
    return entries, problems


def _exception_entry(raw):
    """(Exclusion, frame stems, None) for a sound entry, else (None, [], reason)."""
    if not isinstance(raw, dict):
        return None, [], 'is not a mapping'
    bad = [k for k in EXCEPTION_TEXT_KEYS if blank(raw.get(k))]
    if bad:
        return None, [], f'needs non-blank text for {", ".join(bad)}'
    long = [k for k in EXCEPTION_SHORT_KEYS if len(raw[k].strip()) > MAX_SHORT_FIELD_CHARS]
    if long:
        return None, [], f'has {", ".join(long)} longer than {MAX_SHORT_FIELD_CHARS} characters'
    date = raw.get('date')
    date = str(date.isoformat() if hasattr(date, 'isoformat') else date)
    try:
        if not EXCEPTION_DATE_PATTERN.match(date):
            raise ValueError(date)
        datetime.strptime(date, EXCEPTION_DATE_FORMAT)
    except ValueError:
        return None, [], f'has date {date[:40]!r}, not YYYY-MM-DD'
    frames = raw.get('frames')
    if not isinstance(frames, list) or not frames or any(blank(f) for f in frames):
        return None, [], 'needs frames: a non-empty list of frame stems'
    exc = Exclusion(raw['project_id'].strip(), raw['ruling'].strip(), date, raw['by'].strip())
    return exc, [f.strip() for f in frames], None


def rule_r15(cfg, projects):
    """Exceptions file sound; attach each ruling's frames to its project as p.excluded.

    Returns {project label or '': [(level, path, message)]}. An entry naming a
    project that is not live, or a frame its project does not hold, is a
    FAIL; the entry's other frames still apply. A frame named twice is a FAIL.
    """
    path = cfg.exceptions_path
    entries, problems = load_exceptions(path)
    res = {'': [(FAIL, path, f'{t}.') for t in problems]}
    # The folder name is the project's identity here, as for every live-project
    # check: a copied folder whose project.json kept the original id (an R01
    # FAIL) must never take over the original's ruling.
    by_id = {p.expected_id: p for p in projects}
    for n, exc, frames in entries:
        p = by_id.get(exc.project_id)
        if p is None:
            res[''].append((FAIL, path, f'Entry {n} names project {exc.project_id!r}, '
                                        f'which is not a live project.'))
            continue
        known = {stem_of(k) for k in p.frames()}
        absent = [f for f in frames if f not in known]
        listed = Counter(frames)
        repeated = sorted(f for f, c in listed.items() if c > 1 or f in p.excluded)
        for f in frames:
            if f in known:
                p.excluded.setdefault(f, exc)
        found = res.setdefault(p.label, [])
        if absent:
            found.append((FAIL, path, f'Entry {n} names {len(absent)} frames that {p.label} does not '
                                      f'hold: {_examples(absent)}.'))
        if repeated:
            found.append((FAIL, path, f'Entry {n} names {len(repeated)} frames already listed: '
                                      f'{_examples(repeated)}.'))
    return res


# ---------------------------------------------------------------- orchestration
def emit(report, rule, project, results, pass_path, pass_message):
    """Add results as findings, plus one PASS when none of them is a WARN or FAIL."""
    if not any(level in (WARN, FAIL) for level, _, _ in results):
        report.add(rule, PASS, project, pass_path, pass_message)
    for level, path, message in results:
        report.add(rule, level, project, path, message)


def guarded(report, rule, project, path, fn, *args):
    """Run a rule; an unexpected exception becomes a FAIL finding, never a crash."""
    try:
        return fn(*args)
    except Exception as exc:  # the checker must report its own faults, not stop
        report.add(rule, FAIL, project, path, f'Checker error in {rule}: {type(exc).__name__}: {str(exc)[:160]}')
        return None


def _checkpoint(cfg, n, text):
    """Print a 'Step N complete' line to stderr when cfg.progress is set."""
    if cfg.progress:
        print(f'Step {n} complete: {text}', file=sys.stderr)


def run_checks(cfg):
    """Run all 15 rules against cfg and return a Report. Writes nothing."""
    now = cfg.now or datetime.now(AST)
    now_ts = now.timestamp()
    if cfg.archive_dir is None:
        cfg = Config(**{**asdict(cfg), 'archive_dir': os.path.join(os.path.dirname(cfg.root.rstrip('/')), 'archive')})
    report = Report(root=cfg.root, started=now.isoformat(timespec='seconds'))
    folders = [os.path.join(cfg.root, n) for n in _list_dir(cfg.root)
               if n.startswith(RUN_PREFIX) and os.path.isdir(os.path.join(cfg.root, n))]
    projects = [Project(f) for f in folders]
    report.projects = len(projects)
    live, archived = set(project_ids(cfg.root)), set(project_ids(cfg.archive_dir))
    _checkpoint(cfg, 1, f'{len(projects)} projects found under {cfg.root}')
    vocab_codes, vocab_err = read_vocabulary(cfg.vocab_path)
    vocab = set(vocab_codes)
    app, app_err = load_app_code(cfg.scripts_dir)
    if app_err:
        report.add('R09', FAIL, '', cfg.scripts_dir, f'Application helpers could not be imported ({app_err}).')
    exceptions = guarded(report, 'R15', '', cfg.exceptions_path, rule_r15, cfg, projects)
    for label, res in (exceptions or {}).items():
        emit(report, 'R15', label, res, cfg.exceptions_path,
             'Integrity exceptions name live projects and frames they hold (or there are none).')
    for p in projects:
        _per_project(report, cfg, p, vocab, app, now_ts)
    _checkpoint(cfg, 2, 'exceptions R15, per-project rules R01 to R08 and R13')
    _shared_rules(report, cfg, projects, live, archived, app, vocab_codes, vocab_err, now, now_ts)
    _checkpoint(cfg, 3, 'shared stores and backups R09 to R12, R14')
    return report


def _per_project(report, cfg, p, vocab, app, now_ts):
    """Rules R01 to R08 and R13 for one project."""
    name = p.label
    steps = [
        ('R01', rule_r01, (p,), p.pj_path, 'project.json is well formed.'),
        ('R02', rule_r02, (p,), p.folder, 'Step 1 and Step 2 files resolve and parse.'),
        ('R03', rule_r03, (p, vocab), p.folder, 'Step 3 selection is consistent.'),
        ('R04', rule_r04, (p,), p.seg_path, 'Segmentations are well formed.'),
        ('R05', rule_r05, (p,), p.exp, 'Images, links and labels pair up.'),
        ('R06', rule_r06, (p,), p.exp, 'YOLO labels, class map and data.yaml agree.'),
        ('R07', rule_r07, (p,), p.exp, 'Export manifest matches the frames and labels.'),
        ('R08', rule_r08, (p, vocab, app), p.exp, 'Label ledger matches the stored outcomes.'),
        ('R13', rule_r13, (p, now_ts), p.folder, 'No stale temp files, conflicts or bad screening file.'),
    ]
    for rule, fn, args, path, ok in steps:
        if p.training and rule in ANNOTATION_RULES:
            report.add(rule, PASS, name, path, 'Training project: it holds no annotation work of its own.')
            continue
        res = guarded(report, rule, name, path, fn, *args)
        if res is not None:
            emit(report, rule, name, res, path, ok)


def _shared_rules(report, cfg, projects, live, archived, app, vocab_codes, vocab_err, now, now_ts):
    """Rules that span projects: R09, R10, R11, R12, R14 and shared-store temp files."""
    reg = guarded(report, 'R09', '', cfg.root, rule_r09, cfg, projects, live, archived, app)
    if reg is not None:
        for label, res in reg.items():
            emit(report, 'R09', label, res, os.path.join(cfg.root, '_mask_registry', 'manifest.csv'),
                 'Registry agrees with exported masks.' if label else 'Registry header, uids and project ids are sound.')
    comps = guarded(report, 'R10', '', cfg.root, rule_r10, cfg, live, archived)
    if comps is not None:
        if not comps:
            report.add('R10', PASS, '', os.path.join(cfg.root, '_composite'), 'No composite training sets to check.')
        for name, res in comps.items():
            emit(report, 'R10', f'_composite/{name}', res, os.path.join(cfg.root, '_composite', name),
                 'Composite training set is consistent.')
    used = {}
    guarded(report, 'R11', '', cfg.root, collect_codes, projects, cfg, used)
    res = guarded(report, 'R11', '', cfg.vocab_path, rule_r11, cfg, vocab_codes, vocab_err, used)
    if res is not None:
        emit(report, 'R11', '', res, cfg.vocab_path, 'Vocabulary is unique, mirrored, and covers every code in use.')
    res = guarded(report, 'R12', '', cfg.scripts_dir, rule_r12, cfg)
    if res is not None:
        emit(report, 'R12', '', res, cfg.scripts_dir, 'Every manual-source list includes "merged".')
    res = guarded(report, 'R13', '', cfg.root, rule_r13_stores, cfg, now_ts)
    if res is not None:
        emit(report, 'R13', '', res, cfg.root, 'No stale temp files in the shared stores.')
    res = guarded(report, 'R14', '', cfg.snapshot_dir, rule_r14_snapshots, cfg, now)
    if res is not None:
        emit(report, 'R14', '', res, cfg.snapshot_dir, 'The newest nightly snapshot is under 26 hours old.')
    arch = guarded(report, 'R14', '', cfg.archive_dir, rule_r14_archive, cfg)
    for pid, res in (arch or {}).items():
        emit(report, 'R14', pid, res, cfg.archive_dir, '')


def exit_code(report):
    """0 when no FAIL, 1 otherwise."""
    return EXIT_FAIL if report.counts()[FAIL] else EXIT_OK


# ---------------------------------------------------------------- output
def render_text(report, verbose=False):
    """Plain-text report: WARN and FAIL lines (PASS too when verbose) and a summary."""
    lines = [f'VEGA project integrity check, {report.started} AST', f'Root: {report.root}', '']
    for f in report.findings:
        if f.level == PASS and not verbose:
            continue
        lines.append(f'[{f.level}] {f.rule} {f.project or "(shared)"} | {f.path} | {f.message}')
    c = report.counts()
    lines += ['', f'Projects checked: {report.projects}',
              f'Summary: {c[PASS]} PASS, {c[INFO]} INFO, {c[WARN]} WARN, {c[FAIL]} FAIL']
    by_rule = Counter((f.rule, f.level) for f in report.findings if f.level != PASS)
    for (rule, level), n in sorted(by_rule.items()):
        lines.append(f'  {rule} {level}: {n}')
    return '\n'.join(lines) + '\n'


def render_json(report):
    """JSON document with the summary and every finding."""
    return json.dumps({'root': report.root, 'started': report.started,
                       'projects': report.projects, 'summary': report.counts(),
                       'findings': [asdict(f) for f in report.findings]}, indent=1, ensure_ascii=False)


def write_reports(report, report_dir, now):
    """Write <stamp>.txt and <stamp>.json into report_dir; returns the two paths."""
    os.makedirs(report_dir, exist_ok=True)
    stamp = now.strftime(REPORT_STAMP_FORMAT)
    paths = []
    for ext, body in (('txt', render_text(report, verbose=True)), ('json', render_json(report))):
        path = os.path.join(report_dir, f'{stamp}.{ext}')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(body)
        paths.append(path)
    return paths


def build_parser():
    """Command-line parser."""
    ap = argparse.ArgumentParser(description='Read-only integrity checker for VEGA projects.')
    ap.add_argument('--root', default=DEFAULT_ROOT, help='inprocess/ tree to check')
    ap.add_argument('--archive-dir', default=None, help='archive/ tree (default: <root>/../archive)')
    ap.add_argument('--report-dir', default=DEFAULT_REPORT_DIR, help='where <stamp>.txt and .json are written')
    ap.add_argument('--vocab', default=DEFAULT_VOCAB, help='species vocabulary CSV')
    ap.add_argument('--pipeline-config', default=DEFAULT_PIPELINE_CONFIG, help='config/pipeline.yaml')
    ap.add_argument('--snapshot-dir', default=DEFAULT_SNAPSHOT_DIR, help='nightly snapshot folder')
    ap.add_argument('--manual-source-file', action='append', default=None,
                    help='file to scan for manual-source lists (repeatable; default: the known files)')
    ap.add_argument('--exceptions', default=DEFAULT_EXCEPTIONS,
                    help='integrity_exceptions.yaml naming frames a ruling keeps out of the coverage matrix')
    ap.add_argument('--now', default=None, help='ISO time to use as now (tests)')
    ap.add_argument('--json', action='store_true', help='print JSON instead of text')
    ap.add_argument('--verbose', action='store_true', help='also print PASS lines in text mode')
    return ap


def main(argv=None):
    """Parse arguments, run the checks, write reports, print, return the exit code."""
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.root):
        print(f'error: --root {args.root} is not a directory', file=sys.stderr)
        return EXIT_USAGE
    try:
        now = datetime.fromisoformat(args.now) if args.now else datetime.now(AST)
    except ValueError:
        print(f'error: --now {args.now!r} is not an ISO time', file=sys.stderr)
        return EXIT_USAGE
    cfg = Config(root=os.path.abspath(args.root), archive_dir=args.archive_dir, vocab_path=args.vocab,
                 pipeline_config=args.pipeline_config, snapshot_dir=args.snapshot_dir,
                 manual_source_files=args.manual_source_file, exceptions_path=args.exceptions,
                 now=now, progress=True)
    report = run_checks(cfg)
    try:
        paths = write_reports(report, args.report_dir, now)
    except OSError as exc:
        print(f'error: cannot write reports to {args.report_dir}: {exc}', file=sys.stderr)
        return EXIT_USAGE
    _checkpoint(cfg, 4, f'reports written: {", ".join(paths)}')
    print(render_json(report) if args.json else render_text(report, args.verbose))
    return exit_code(report)


if __name__ == '__main__':
    sys.exit(main())
