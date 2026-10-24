#!/usr/bin/env python3
"""Tests for return_frames_to_review.py: send finished VEGA frames back to review.

Module:  reef_point_seg / scripts/tests/test_return_frames_to_review.py
Purpose: Prove the tool that sends exported frames back to the combined
         annotator's review queue. The default run builds every project under
         a fresh temporary folder and never reads or writes the real
         inprocess/ tree. It covers:
           * the dry run writes nothing (checksum, time, mode and listing sweep);
           * --apply changes only the chosen frames' exported flag and adds one
             returned_to_review event, keeps the file's mode and layout, writes
             a confirmed backup and one log line;
           * a rerun changes nothing; a scrapped or unexported frame is skipped
             and reported;
           * every refusal exits 2 with a sentence and writes nothing: missing
             project, missing frame, live listener, ss failing, missing or
             unreadable, malformed JSON, a file changed mid-run, a second run
             racing the first, hostile names, a bad reason, bad initials;
           * --select over a fixture holding every mask kind; a 5,000-frame file;
           * the annotator round trip, in a child process started with the
             settings the orchestrator gives a step4test launch: the returned
             frames are in the review queue and in "Frames with a mask missing
             its species", a save and a save merged with an outside change keep
             returned_to_review, and labelling, rejecting and exporting rewrite
             the label file, manifest row, registry rows and ledger rows; the
             integrity checker reads the project clean before, INFO while the
             frames wait, and clean again after.
         With --e2e-real WORKDIR it also copies AT's two PA projects into
         WORKDIR (read only from the real tree), runs the same flow on the
         copies, proves the real projects did not change, and deletes WORKDIR.
Inputs:  optional --e2e-real WORKDIR (a folder that does not exist yet)
Outputs: PASS and FAIL lines, a tally, exit 0 when every check passed.

Run: /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python \
        scripts/tests/test_return_frames_to_review.py [--e2e-real WORKDIR]
"""
# Keep this suite off the live mask registry, expert library and clip index:
# _isolation.py points the three stores at temp folders and blocks any write
# to the live copies, in this process and in the annotator child processes.
import os as _iso_os
import sys as _iso_sys
_iso_sys.path.insert(0, _iso_os.path.dirname(_iso_os.path.abspath(__file__)))
import _isolation  # noqa: E402
_isolation.install()

import atexit  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import stat  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.parse import quote  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(SCRIPTS))

try:
    import return_frames_to_review as rftr  # noqa: E402
    TOOL_IMPORT_ERROR = ''
except ImportError as _exc:  # the tool is written after these tests; report, never hide
    rftr = None
    TOOL_IMPORT_ERROR = f'{type(_exc).__name__}: {_exc}'
import check_project_integrity as cic  # noqa: E402
from _reefreview.mask_registry import build_registry_record  # noqa: E402

# ---------------------------------------------------------------- constants
TOOL = SCRIPTS / 'return_frames_to_review.py'
ANNOTATOR_SRC = SCRIPTS / 'TCRMPclip_combinedAnnotate' / 'src'
PYTHON = sys.executable
AST = timezone(timedelta(hours=-4))
AST_ZONE = 'America/St_Thomas'  # the box's zone; the export manifest's batch stamps use local time
PID = '26-10-16_XX'
RUN = 'run_' + PID
EXPORT = 'step4test_combinedAnnotate'
RETURN_KEY = 'returned_to_review'
LOG_NAME = 'review_returns.jsonl'
BACKUP_LABEL = 'before_return_to_review'
LEDGER_LOCK = 'label_provenance.csv.lock'
REASON = 'Outline saved without a species by the label popup bug; please label or reject it.'
BY = 'LO'
SELECT = 'unlabelled-hand-outlines'
STAMP_RE = re.compile(r'^\d{8}-\d{6}$')
FRAME_LINE = re.compile(r'^  (\S+): exported true -> false')
LEDGER_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']
MANIFEST_BASE = ['basename', 'n_masks', 'label_file', 'image_file', 'outcome',
                 'preview_file', 'batch_name', 'image_path']
LARGE_FRAMES = 5000
LARGE_EVERY = 10              # every tenth frame of the large file holds an unlabelled outline
LARGE_SECONDS = 120           # generous bound for a dry run plus an apply on 5,000 frames
CHILD_TIMEOUT_S = 900
EXPORT_SECOND = 1.1           # an event must be newer than the batch stamp (second precision)
HOLD_S = 2.0                  # how long the stand-in importer holds its lock in the gap test
WAIT_S = 30                   # longest wait for a helper process to signal it is ready
EDITMASKS_SRC = SCRIPTS / 'TCRMPclip_editMasks' / 'src'
IMPORT_NOTE = 'saved by the importer'
# A stand-in for the expert-ID importer: it takes the importer's own lock,
# reads segmentations.json, holds the lock while "working", then saves an
# import with the importer's own _save, exactly as _expertids/importer.py does.
HOLDER_CODE = '''
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from _expertids.importer import _seg_locked, _load, _save
path, ready, key, hold, modify = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), sys.argv[6] == '1'
with _seg_locked(path):
    data = _load(path)
    open(ready, 'w').close()
    time.sleep(hold)
    if modify:
        data[key]['imported_note'] = 'saved by the importer'
        _save(path, data)
'''
WAITING_LINE = 'waiting for the expert-ID importer to finish saving'
SHORT_DEADLINE_S = 1.0        # the importer-lock deadline the timeout test sets
LONG_HOLD_S = 20.0            # a stand-in importer that outlasts the test's patience
SS_HEADER = 'State  Recv-Q Send-Q Local Address:Port Peer Address:Port Process'
SS_SCRIPTS = {
    'quiet': f'printf "{SS_HEADER}\\nLISTEN 0 128 127.0.0.1:5090 0.0.0.0:*\\n'
             f'LISTEN 0 4096 [::]:22 [::]:*\\nLISTEN 0 128 127.0.0.53%%lo:53 0.0.0.0:*\\n"',
    'busy': f'printf "{SS_HEADER}\\nLISTEN 0 128 0.0.0.0:5081 0.0.0.0:*\\n'
            f'LISTEN 0 128 [::]:5089 [::]:*\\nLISTEN 0 128 127.0.0.1:5090 0.0.0.0:*\\n"',
    'busy5096': f'printf "{SS_HEADER}\\nLISTEN 0 128 0.0.0.0:5096 0.0.0.0:*\\n"',
    'fail': 'echo "ss: netlink failure" >&2; exit 3',
    'garbage': 'echo "this is not a listener table"',
    'empty': 'exit 0',
}
REAL_INPROCESS = SCRIPTS.parent.parent / 'inprocess'
REAL_PROJECTS = ('run_26-09-18_AT', 'run_26-10-04_AT')
REAL_COPY_FILES = ('segmentations/segmentations.json', 'label_provenance.csv', 'export_manifest.csv',
                   'class_map.json', 'data.yaml', 'prior_screening.json',
                   'routed_input/ids/sam_click_prompts.json')
REAL_HASHED_FILES = REAL_COPY_FILES + ('.export_batch_seq', 'export_log.txt', 'eml.xml', 'config_log.json')
EXPECTED_REAL = {
    'run_26-09-18_AT': [
        'TCRMP20141208_clip_CRB_T318', 'TCRMP20141208_clip_CRB_T312', 'TCRMP20151003_clip_CST_T118',
        'TCRMP20161020_clip_BWR_T306', 'TCRMP20161024_clip_CRB_T416', 'TCRMP20161024_clip_CRB_T411',
        'TCRMP20161024_clip_CRB_T206', 'TCRMP20161014_clip_EGR_T619', 'TCRMP20161013_clip_LBP_T406',
        'TCRMP20171207_clip_BPT_T408', 'TCRMP20171204_clip_SVN_T622', 'TCRMP20181121_clip_BPT_T203',
        'TCRMP20181025_clip_CRB_T508', 'TCRMP20191018_clip_CBS_T210', 'TCRMP20191017_clip_MTS_T105',
        'TCRMP20200706_clip_BPT_T313', 'TCRMP20200224_clip_CRB_T503', 'TCRMP20200204_clip_SCP_T414',
        'TCRMP20220908_clip_CRB_T506'],
    'run_26-10-04_AT': [
        'TCRMP20180226_clip_SPH_T501', 'TCRMP20241111_clip_BIX_T215', 'TCRMP20241025_clip_SWT_T204',
        'TCRMP20250328_clip_SWT_T211'],
}

# The second run: frames that read PA not found although a PA outline over a
# PA survey point was rejected or never accepted. Run after the first.
REASON_SECOND = ('PA outline over a PA survey point, but the frame reads not found; '
                 'please accept, redraw or confirm the rejection.')
EXPECTED_SECOND = {
    'run_26-10-04_AT': [
        'TCRMP20150930_clip_JKB_T622', 'TCRMP20150930_clip_JKB_T610', 'TCRMP20151001_clip_BIX_T223',
        'TCRMP20151003_clip_EGR_T120', 'TCRMP20161003_clip_BPT_T603'],
    'run_26-09-18_AT': [
        'TCRMP20141202_clip_HBE_T614', 'TCRMP20141208_clip_CRB_T102', 'TCRMP20161024_clip_FSB_T514',
        'TCRMP20181025_clip_CRB_T312', 'TCRMP20241113_clip_JKB_T312'],
}

PASSES, FAILS = [], []


# ---------------------------------------------------------------- harness
def check(name, cond, detail=''):
    """Record one assertion as PASS or FAIL and print it."""
    if cond:
        PASSES.append(name)
        print(f'PASS  {name}')
    else:
        FAILS.append(name)
        print(f'FAIL  {name}  {str(detail)[:1500]}')


def group(fn):
    """Run one test group; a crash is one FAIL naming the group, never a stop."""
    try:
        fn()
    except Exception as exc:  # a crashed group is a failed group, reported in full
        FAILS.append(f'{fn.__name__} crashed')
        print(f'FAIL  {fn.__name__} crashed: {type(exc).__name__}: {exc}')
        traceback.print_exc()


def need_tool():
    """Raise when the tool module could not be imported."""
    if rftr is None:
        raise RuntimeError(f'return_frames_to_review.py could not be imported ({TOOL_IMPORT_ERROR})')


_BIN_ROOT = Path(tempfile.mkdtemp(prefix='rftr_bin_'))
atexit.register(shutil.rmtree, _BIN_ROOT, True)


def ss_dir(kind):
    """A folder whose only command is a fake `ss` of this kind ('missing': no ss at all)."""
    folder = _BIN_ROOT / kind
    if folder.is_dir():
        return str(folder)
    folder.mkdir()
    if kind != 'missing':
        script = folder / 'ss'
        script.write_text('#!/bin/sh\n' + SS_SCRIPTS[kind] + '\n', encoding='utf-8')
        script.chmod(0o755)
    return str(folder)


@contextlib.contextmanager
def fake_ss(kind):
    """Run the block with PATH holding only the fake ss of this kind."""
    old = os.environ.get('PATH', '')
    os.environ['PATH'] = ss_dir(kind)
    try:
        yield
    finally:
        os.environ['PATH'] = old


def cli(args, ss='quiet', path=None):
    """Run the tool as Lauren does, in a subprocess; returns CompletedProcess."""
    env = dict(os.environ, PATH=path if path is not None else ss_dir(ss), CUDA_VISIBLE_DEVICES='')
    return subprocess.run([PYTHON, str(TOOL), *[str(a) for a in args]], capture_output=True,
                          text=True, env=env, timeout=CHILD_TIMEOUT_S)


def in_process(argv, ss='quiet'):
    """Run rftr.main(argv) in this process; returns (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with fake_ss(ss), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rftr.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def hash_tree(root):
    """{path: (sha256 or link target or 'dir', mtime_ns, mode)} for every entry under root."""
    found = {}
    for dp, dns, fns in os.walk(root):
        for n in dns + fns:
            p = os.path.join(dp, n)
            st = os.lstat(p)
            if os.path.islink(p):
                digest = 'link:' + os.readlink(p)
            elif os.path.isfile(p):
                with open(p, 'rb') as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
            else:
                digest = 'dir'
            found[p] = (digest, st.st_mtime_ns, stat.S_IMODE(st.st_mode))
    return found


def tree_diff(before, after):
    """(changed paths, new paths, removed paths) between two hash_tree results."""
    changed = sorted(p for p in set(before) & set(after) if before[p] != after[p])
    return changed, sorted(set(after) - set(before)), sorted(set(before) - set(after))


def sha256(path):
    """Hex sha256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    """JSON object from a file."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- fixtures
def outline(mid, species='', status='pending', source='manual_click', **extra):
    """One mask dict with its own geometry, shaped like the annotator's."""
    x0, y0 = 4 + 3 * mid, 4 + 2 * mid
    m = {'id': mid, 'label': chr(ord('A') + mid % 26), 'species': species,
         'name': 'Porites astreoides' if species == 'PA' else '', 'category': 'Coral' if species else '',
         'source_x': x0 + 5, 'source_y': y0 + 5,
         'polygon_px': [[x0, y0, x0 + 20, y0, x0 + 20, y0 + 14]],
         'polygon_norm': [[x0 / 64, y0 / 48, (x0 + 20) / 64, y0 / 48, (x0 + 20) / 64, (y0 + 14) / 48]],
         'bbox': [x0, y0, 20, 14], 'area': 140 + mid, 'score': 1.0, 'refinement_clicks': [],
         'status': status, 'source_type': source, 'reviewer': 'AT'}
    m.update(extra)
    return m


def frame(exp, key, masks, exported=True, **extra):
    """One frame entry for segmentations.json."""
    seg = {'image_path': 'raw/' + key, 'image_path_abs': str(exp / 'routed_input' / 'raw' / key),
           'image_width': 64, 'image_height': 48, 'masks': masks, 'reference_points': [],
           'processed_at': '2026-10-01T09:00:00', 'reviewed': True, 'exported': exported}
    seg.update(extra)
    return seg


def key_of(stem):
    """Frame key (file name) for a stem, the way the annotator stores it."""
    return stem + '.jpeg'


KIND_STEMS = {
    'manual_click blank': ('TCRMP20200101_clip_BPT_T101', True),
    'manual_box blank': ('TCRMP20200101_clip_BPT_T102', True),
    'merged blank': ('TCRMP20200101_clip_BPT_T103', True),
    'whitespace species': ('TCRMP20200101_clip_BPT_T104', True),
    'species null': ('TCRMP20200101_clip_BPT_T105', True),
    'no species key': ('TCRMP20200101_clip_BPT_T106', True),
    'auto blank': ('TCRMP20200101_clip_BPT_T107', False),
    'exemplar blank': ('TCRMP20200101_clip_BPT_T108', False),
    'pending labelled': ('TCRMP20200101_clip_BPT_T109', False),
    'rejected blank': ('TCRMP20200101_clip_BPT_T110', False),
    'accepted labelled': ('TCRMP20200101_clip_BPT_T111', False),
    'review tag': ('TCRMP20200101_clip_BPT_T112', False),
    'not exported': ('TCRMP20200101_clip_BPT_T113', False),
    'scrapped': ('TCRMP20200101_clip_BPT_T114', False),
    'no masks': ('TCRMP20200101_clip_BPT_T115', False),
    'mixed': ('TCRMP20200101_clip_BPT_T116', True),
    'masks missing': ('TCRMP20200101_clip_BPT_T117', False),
    'no source type': ('TCRMP20200101_clip_BPT_T118', False),
}


def kind_frames(exp):
    """{key: frame} holding one frame per mask kind the selection must tell apart."""
    s = {name: key_of(stem) for name, (stem, _) in KIND_STEMS.items()}
    no_species = outline(0)
    del no_species['species']
    no_source = outline(0)
    del no_source['source_type']
    masks_missing = frame(exp, s['masks missing'], [])
    del masks_missing['masks']
    return {
        s['manual_click blank']: frame(exp, s['manual_click blank'], [outline(0)]),
        s['manual_box blank']: frame(exp, s['manual_box blank'], [outline(0, source='manual_box')]),
        s['merged blank']: frame(exp, s['merged blank'], [outline(0, source='merged')]),
        s['whitespace species']: frame(exp, s['whitespace species'], [outline(0, species='   ')]),
        s['species null']: frame(exp, s['species null'], [outline(0, species=None)]),
        s['no species key']: frame(exp, s['no species key'], [no_species]),
        s['auto blank']: frame(exp, s['auto blank'], [outline(0, source='auto')]),
        s['exemplar blank']: frame(exp, s['exemplar blank'], [outline(0, source='exemplar')]),
        s['pending labelled']: frame(exp, s['pending labelled'], [outline(0, species='PA')]),
        s['rejected blank']: frame(exp, s['rejected blank'], [outline(0, status='rejected')]),
        s['accepted labelled']: frame(exp, s['accepted labelled'], [outline(0, 'PA', 'accepted')]),
        s['review tag']: frame(exp, s['review tag'], [outline(0, species='REVIEW', review=True)]),
        s['not exported']: frame(exp, s['not exported'], [outline(0)], exported=False),
        s['scrapped']: frame(exp, s['scrapped'], [outline(0)], scrapped=True),
        s['no masks']: frame(exp, s['no masks'], []),
        s['mixed']: frame(exp, s['mixed'], [outline(0, 'PA', 'accepted'), outline(1),
                                             outline(2, status='rejected', source='auto')]),
        s['masks missing']: masks_missing,
        s['no source type']: frame(exp, s['no source type'], [no_source]),
    }


def expected_kind_keys():
    """Keys --select must pick from kind_frames()."""
    return sorted(key_of(stem) for stem, picked in KIND_STEMS.values() if picked)


def kind_key(name):
    """Key of the kind_frames() frame with this kind name."""
    return key_of(KIND_STEMS[name][0])


def write_json(path, obj, indent=1):
    """Write obj as JSON (the annotator's indent=1 layout by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding='utf-8')


def write_csv(path, header, rows):
    """Write row dicts with header to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})


def make_project(root, frames_for, pid=PID, indent=1):
    """Write one project under root; frames_for(exp) returns its frames. Returns the paths."""
    proj = Path(root) / ('run_' + pid)
    exp = proj / EXPORT
    frames = frames_for(exp)
    write_json(proj / 'project.json', {
        'id': pid, 'name': 'Return fixture', 'created_at': '2026-10-01T08:00:00-04:00',
        'updated_at': '2026-10-01T09:00:00-04:00', 'project_dir': str(proj), 'current_step': 4,
        'steps': {}, 'access_log': [{'ts': '2026-10-01T08:00:00-04:00', 'initials': 'AT',
                                     'action': 'create', 'purpose': 'fixture'}]})
    write_json(exp / 'segmentations' / 'segmentations.json', frames, indent=indent)
    write_csv(exp / 'label_provenance.csv', LEDGER_HEADER,
              [{'basename': os.path.splitext(k)[0], 'label': 'PA', 'outcome': 'not_found',
                'source': 'step4test', 'reviewer': 'AT', 'at': '2026-10-01T09:00:00-04:00',
                'project_id': pid} for k in sorted(frames)])
    write_csv(exp / 'export_manifest.csv', MANIFEST_BASE + ['PA'],
              [{'basename': os.path.splitext(k)[0], 'n_masks': 0, 'outcome': 'exported_empty',
                'batch_name': 'batch_001_20261001-090000', 'PA': 0} for k in sorted(frames)])
    for k in frames:
        p = exp / 'all_labels' / (os.path.splitext(k)[0] + '.txt')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('', encoding='utf-8')
    write_json(exp / 'class_map.json', {'PA': 0})
    return proj, exp, exp / 'segmentations' / 'segmentations.json'


def tool_args(root, *extra, projects=(RUN,), select=True, frames=(), apply=False, by=BY, reason=REASON):
    """Command-line arguments for the tool."""
    args = ['--root', root]
    for p in projects:
        args += ['--project', p]
    for f in frames:
        args += ['--frame', f]
    if select:
        args += ['--select', SELECT]
    args += ['--by', by, '--reason', reason]
    if apply:
        args.append('--apply')
    return args + list(extra)


def listed_frames(stdout):
    """Frame keys the tool's output lists as going back to the queue."""
    return sorted(m.group(1) for m in (FRAME_LINE.match(line) for line in stdout.splitlines()) if m)


def strip_returns(data, keys):
    """Copy of data with the tool's two changes undone on keys (for comparing the rest)."""
    out = json.loads(json.dumps(data))
    for k in keys:
        out[k]['exported'] = True
        events = out[k].get(RETURN_KEY) or []
        if events:
            events.pop()
        if not events:
            out[k].pop(RETURN_KEY, None)
    return out


# ---------------------------------------------------------------- tool groups
def test_dry_run_writes_nothing():
    """A dry run reads, lists and writes nothing anywhere under the tree."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames)
        before = hash_tree(tmp)
        r = cli(tool_args(root))
        after = hash_tree(tmp)
        check('dry run exits 0', r.returncode == 0, r.stdout[-600:] + r.stderr[-600:])
        changed, new, gone = tree_diff(before, after)
        check('dry run: no file, link, time or mode changed', not changed, changed[:5])
        check('dry run: no file created or removed', not new and not gone, (new + gone)[:5])
        check('dry run lists exactly the frames --select picks',
              listed_frames(r.stdout) == expected_kind_keys(),
              f'{listed_frames(r.stdout)} vs {expected_kind_keys()}')
        check('dry run says nothing was written and how to apply',
              'Dry run' in r.stdout and '--apply' in r.stdout, r.stdout[-400:])
        check('dry run prints Step 1 to Step 4 checkpoints',
              all(f'Step {n} complete' in r.stdout for n in range(1, 5)), r.stdout[:600])
        check('dry run prints no apply checkpoints', 'Step 5 complete' not in r.stdout)


def test_apply_changes_only_named():
    """--apply on one named frame changes its exported flag and event, nothing else."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        proj, exp, seg = make_project(root, kind_frames)
        original = load(seg)
        before = hash_tree(tmp)
        target = kind_key('manual_click blank')
        r = cli(tool_args(root, select=False, frames=[KIND_STEMS['manual_click blank'][0]], apply=True))
        check('apply exits 0', r.returncode == 0, r.stdout[-600:] + r.stderr[-600:])
        check('one frame reads "1 frame goes back"', '1 frame goes back to the review queue' in r.stdout
              and '1 frame go back' not in r.stdout, r.stdout[:800])
        after = hash_tree(tmp)
        changed, new, gone = tree_diff(before, after)
        check('apply changes only segmentations.json (and the folder times holding new files)',
              [p for p in changed if not os.path.isdir(p)] == [str(seg)], changed)
        allowed_new = {str(exp / LOG_NAME), str(exp / LEDGER_LOCK),
                       str(seg.parent / ('.' + seg.name + '.lock'))}
        backups = [p for p in new if p.endswith('segmentations.json')]
        check('apply creates one backup, the log and the two project locks, and nothing else',
              len(backups) == 1 and set(p for p in new if not os.path.isdir(p)) - set(backups) <= allowed_new
              and str(exp / LOG_NAME) in new, new)
        check('apply removes nothing', not gone, gone)
        data = load(seg)
        check('the named frame is no longer exported', data[target]['exported'] is False)
        events = data[target].get(RETURN_KEY)
        check('the named frame carries one returned_to_review event',
              isinstance(events, list) and len(events) == 1, events)
        check('every other part of the file is unchanged', strip_returns(data, [target]) == original)
        check('masks, label_outcomes and reviewed are untouched on the named frame',
              data[target]['masks'] == original[target]['masks']
              and data[target].get('reviewed') == original[target].get('reviewed'))
        check('no temporary file is left beside segmentations.json',
              not [n for n in os.listdir(seg.parent) if n.endswith('.tmp')], os.listdir(seg.parent))


def test_backup_and_log():
    """The backup equals the file before; the log line and the event carry who, why, when."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        proj, exp, seg = make_project(root, kind_frames)
        raw_before = seg.read_bytes()
        r = cli(tool_args(root, apply=True))
        check('apply with --select exits 0', r.returncode == 0, r.stdout[-600:] + r.stderr[-600:])
        holding = exp / '_superseded'
        stamps = sorted(os.listdir(holding)) if holding.is_dir() else []
        check('backup sits in _superseded/<YYYYMMDD-HHMMSS>/before_return_to_review/',
              len(stamps) == 1 and STAMP_RE.match(stamps[0])
              and (holding / stamps[0] / BACKUP_LABEL / 'segmentations.json').is_file(), stamps)
        backup = holding / stamps[0] / BACKUP_LABEL / 'segmentations.json' if stamps else None
        check('backup is byte for byte the file before', backup is not None and backup.read_bytes() == raw_before)
        lines = (exp / LOG_NAME).read_text(encoding='utf-8').splitlines() if (exp / LOG_NAME).exists() else []
        check('the log holds one line', len(lines) == 1, lines)
        rec = json.loads(lines[0]) if lines else {}
        at = rec.get('at', '')
        try:
            when = datetime.fromisoformat(at)
        except (TypeError, ValueError):
            when = None
        check('log time is AST ISO with the -04:00 offset',
              when is not None and when.utcoffset() == timedelta(hours=-4), at)
        check('backup folder stamp is the log time', bool(stamps) and when is not None
              and stamps[0] == when.strftime('%Y%m%d-%H%M%S'), (stamps, at))
        check('log names who, why, the tool and the project',
              rec.get('by') == BY and rec.get('reason') == REASON
              and str(rec.get('tool', '')).startswith('return_frames_to_review.py')
              and rec.get('project') == RUN, rec)
        check('log lists the frames sent back', rec.get('frames') == expected_kind_keys(), rec.get('frames'))
        check('log checksums match the backup and the new file',
              backup is not None and rec.get('sha256_before') == sha256(backup)
              and rec.get('sha256_after') == sha256(seg), rec)
        check('log names the backup path', backup is not None and rec.get('backup') == str(backup), rec.get('backup'))
        data = load(seg)
        ev = data[expected_kind_keys()[0]][RETURN_KEY][-1]
        check('event holds exactly at, by, reason, tool and previous_exported',
              set(ev) == {'at', 'by', 'reason', 'tool', 'previous_exported'}, ev)
        check('event matches the log line and records the earlier export',
              ev['at'] == at and ev['by'] == BY and ev['reason'] == REASON and ev['tool'] == rec.get('tool')
              and ev['previous_exported'] is True, ev)
        check('apply prints Step 5 to Step 7 checkpoints and the counts',
              all(f'Step {n} complete' in r.stdout for n in range(5, 8))
              and f'{len(expected_kind_keys())}' in r.stdout.splitlines()[-1], r.stdout[-800:])


def test_rerun_makes_no_change():
    """A second identical apply, by name or by rule, writes nothing."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames)
        cli(tool_args(root, apply=True))
        first = hash_tree(tmp)
        r = cli(tool_args(root, apply=True))
        check('rerun with --select exits 0', r.returncode == 0, r.stderr[-400:])
        check('rerun with --select writes nothing', tree_diff(first, hash_tree(tmp)) == ([], [], []),
              tree_diff(first, hash_tree(tmp)))
        check('rerun with --select lists no frame', listed_frames(r.stdout) == [], r.stdout[-400:])
        stem = KIND_STEMS['manual_click blank'][0]
        r = cli(tool_args(root, select=False, frames=[stem], apply=True))
        check('rerun by name exits 0', r.returncode == 0, r.stderr[-400:])
        check('rerun by name writes nothing', tree_diff(first, hash_tree(tmp)) == ([], [], []))
        check('rerun by name reports the frame as already out of the finished set',
              'not exported' in r.stdout and key_of(stem) in r.stdout, r.stdout[-400:])


def test_skips_scrapped_and_unexported():
    """A named scrapped or unexported frame is skipped and reported; nothing is written."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames)
        before = hash_tree(tmp)
        for name, phrase in (('scrapped', 'scrapped'), ('not exported', 'not exported')):
            r = cli(tool_args(root, select=False, frames=[KIND_STEMS[name][0]], apply=True))
            check(f'a {name} frame exits 0', r.returncode == 0, r.stderr[-400:])
            check(f'a {name} frame is reported as skipped',
                  'skipped' in r.stdout and phrase in r.stdout and listed_frames(r.stdout) == [],
                  r.stdout[-400:])
        check('skipped frames write nothing', tree_diff(before, hash_tree(tmp)) == ([], [], []))


def test_select_every_mask_kind():
    """--select picks exported, unscrapped frames holding a pending blank hand outline only."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        proj, exp, seg = make_project(root, kind_frames)
        r = cli(tool_args(root, apply=True))
        data = load(seg)
        returned = sorted(k for k, v in data.items() if RETURN_KEY in v)
        check('--select returns exactly the hand outlines with no species', returned == expected_kind_keys(),
              f'{returned} vs {expected_kind_keys()}')
        check('frames not picked keep their exported flag',
              all(v.get('exported') == load_original_exported(k) for k, v in data.items() if k not in returned))
        r2 = cli(tool_args(root, select=True, frames=[KIND_STEMS['accepted labelled'][0]], apply=True))
        data = load(seg)
        check('--select and --frame together add the named frame',
              r2.returncode == 0 and data[kind_key('accepted labelled')]['exported'] is False, r2.stdout[-400:])


def load_original_exported(key):
    """The exported flag kind_frames() gives a frame."""
    return key != kind_key('not exported')


def test_two_projects():
    """Two projects in one run: each gets its own backup and log line."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp1, _ = make_project(root, kind_frames, pid='26-10-16_AA')
        _, exp2, _ = make_project(root, kind_frames, pid='26-10-16_BB')
        r = cli(tool_args(root, projects=('run_26-10-16_AA', 'run_26-10-16_BB'), apply=True))
        check('two projects exit 0', r.returncode == 0, r.stderr[-400:])
        for exp in (exp1, exp2):
            lines = (exp / LOG_NAME).read_text(encoding='utf-8').splitlines() if (exp / LOG_NAME).exists() else []
            check(f'{exp.parent.name} has one log line and one backup',
                  len(lines) == 1 and len(list((exp / '_superseded').glob('*/' + BACKUP_LABEL + '/*.json'))) == 1)
        check('the counts name both projects', 'run_26-10-16_AA' in r.stdout.splitlines()[-1]
              and 'run_26-10-16_BB' in r.stdout.splitlines()[-1], r.stdout[-300:])


def test_mode_and_layout_kept():
    """A 0600 file stays 0600; an indent=1 and a compact file keep their layout byte for byte."""
    need_tool()
    for indent, label in ((1, 'indent=1'), (None, 'compact')):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inprocess'
            _, _, seg = make_project(root, kind_frames, indent=indent)
            os.chmod(seg, 0o600)
            r = cli(tool_args(root, apply=True))
            check(f'{label}: apply exits 0', r.returncode == 0, r.stderr[-400:])
            check(f'{label}: mode 0600 is kept', stat.S_IMODE(os.stat(seg).st_mode) == 0o600,
                  oct(stat.S_IMODE(os.stat(seg).st_mode)))
            data = load(seg)
            check(f'{label}: the bytes are the same layout as before',
                  seg.read_bytes() == json.dumps(data, indent=indent).encode('utf-8'))
    for label, write_odd in (
            ('indent=4', lambda seg: seg.write_text(json.dumps(load(seg), indent=4), encoding='utf-8')),
            ('a frame key written twice', lambda seg: seg.write_text(
                seg.read_text(encoding='utf-8')[:-2] + ',\n "' + kind_key('no masks') + '": {"masks": []}\n}',
                encoding='utf-8')),
            ('a mask key written twice', lambda seg: seg.write_text(
                seg.read_text(encoding='utf-8').replace('"label": "A",', '"label": "A", "label": "B",', 1),
                encoding='utf-8'))):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inprocess'
            _, exp, seg = make_project(root, kind_frames)
            write_odd(seg)
            before = hash_tree(tmp)
            dry, r = cli(tool_args(root)), cli(tool_args(root, apply=True))
            check(f'{label}: the dry run refuses and says why', dry.returncode == 2
                  and ('layout' in dry.stderr or 'twice' in dry.stderr), dry.stderr[-300:])
            check(f'{label}: --apply refuses (exit 2)', r.returncode == 2, r.stdout[-300:] + r.stderr[-300:])
            check(f'{label}: nothing is written', tree_diff(before, hash_tree(tmp)) == ([], [], []))


def test_large_file():
    """5,000 frames: the dry run and the apply finish quickly and pick every tenth frame."""
    need_tool()

    def many(exp):
        return {key_of(f'TCRMP20200101_clip_LRG_T{n:05d}'):
                frame(exp, key_of(f'TCRMP20200101_clip_LRG_T{n:05d}'),
                      [outline(0, 'PA', 'accepted'), outline(1) if n % LARGE_EVERY == 0
                       else outline(1, 'PA', 'accepted')]) for n in range(LARGE_FRAMES)}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, _, seg = make_project(root, many)
        original = load(seg)
        t0 = time.monotonic()
        dry = cli(tool_args(root))
        r = cli(tool_args(root, apply=True))
        took = time.monotonic() - t0
        picked = sorted(k for k, v in load(seg).items() if v.get(RETURN_KEY))
        want = sorted(key_of(f'TCRMP20200101_clip_LRG_T{n:05d}') for n in range(0, LARGE_FRAMES, LARGE_EVERY))
        check('5,000 frames: dry run and apply exit 0', dry.returncode == 0 and r.returncode == 0,
              dry.stderr[-300:] + r.stderr[-300:])
        check('5,000 frames: every tenth frame goes back', picked == want, f'{len(picked)} picked')
        check('5,000 frames: the rest of the file is unchanged', strip_returns(load(seg), picked) == original)
        check(f'5,000 frames: both runs take under {LARGE_SECONDS} s', took < LARGE_SECONDS, f'{took:.1f}s')


REFUSAL_CASES = (
    # (name, setup(root, exp, seg) -> None, args builder(root) -> list, ss kind, phrase in stderr)
    ('a missing project', None, lambda root: tool_args(root, projects=('run_nope',)), 'quiet', 'run_nope'),
    ('a project with no segmentations.json',
     lambda root, exp, seg: seg.unlink(), lambda root: tool_args(root), 'quiet', 'does not exist'),
    ('malformed JSON', lambda root, exp, seg: seg.write_text('{"a": ', encoding='utf-8'),
     lambda root: tool_args(root), 'quiet', 'not valid JSON'),
    ('JSON that is a list', lambda root, exp, seg: seg.write_text('[1, 2]', encoding='utf-8'),
     lambda root: tool_args(root), 'quiet', 'not an object'),
    ('bytes that are not UTF-8', lambda root, exp, seg: seg.write_bytes(b'\xff\xfe\x00garbage'),
     lambda root: tool_args(root), 'quiet', 'not valid JSON'),
    ('a segmentations.json that is a link',
     lambda root, exp, seg: (seg.rename(seg.with_name('real.json')), os.symlink('real.json', seg)),
     lambda root: tool_args(root), 'quiet', 'is a link'),
    ('a missing frame', None,
     lambda root: tool_args(root, select=False, frames=['TCRMP20990101_clip_XXX_T999']), 'quiet',
     'TCRMP20990101_clip_XXX_T999'),
    ('a live listener in 5080 to 5089', None, lambda root: tool_args(root), 'busy', '5081'),
    ('a live listener on 5096 (standalone Annotate Images)', None, lambda root: tool_args(root),
     'busy5096', '5096'),
    ('initials with a trailing line break', None, lambda root: tool_args(root, by='LO\n'), 'quiet', 'capital'),
    ('an export folder that is a link',
     lambda root, exp, seg: (shutil.move(str(exp), str(exp.parent / 'export_real')),
                             os.symlink('export_real', str(exp))),
     lambda root: tool_args(root), 'quiet', 'is a link'),
    ('a _superseded folder that is a link',
     lambda root, exp, seg: ((root.parent / 'elsewhere_superseded').mkdir(),
                             os.symlink(str(root.parent / 'elsewhere_superseded'), str(exp / '_superseded'))),
     lambda root: tool_args(root), 'quiet', 'is a link'),
    ('a review_returns.jsonl that is a link',
     lambda root, exp, seg: ((root.parent / 'elsewhere.jsonl').write_text('', encoding='utf-8'),
                             os.symlink(str(root.parent / 'elsewhere.jsonl'), str(exp / LOG_NAME))),
     lambda root: tool_args(root), 'quiet', 'is a link'),
    ('a segmentations folder that is a link',
     lambda root, exp, seg: (shutil.move(str(seg.parent), str(exp / 'segmentations_real')),
                             os.symlink('segmentations_real', str(seg.parent))),
     lambda root: tool_args(root), 'quiet', 'is a link'),
    ('ss failing', None, lambda root: tool_args(root), 'fail', 'ss'),
    ('ss missing', None, lambda root: tool_args(root), 'missing', 'ss'),
    ('ss printing something unreadable', None, lambda root: tool_args(root), 'garbage', 'ss'),
    ('ss printing nothing', None, lambda root: tool_args(root), 'empty', 'ss'),
    ('a qualified frame name with a second slash', None,
     lambda root: tool_args(root, select=False, frames=[RUN + '/a/b']), 'quiet', 'path separator'),
    ('a qualified frame name climbing out', None,
     lambda root: tool_args(root, select=False, frames=[RUN + '/../TCRMP20200101_clip_BPT_T101']),
     'quiet', 'path separator'),
    ('a frame name whose project part is ..', None,
     lambda root: tool_args(root, select=False, frames=['../TCRMP20200101_clip_BPT_T101']), 'quiet', 'not a name'),
    ('a frame name naming a project not given with --project', None,
     lambda root: tool_args(root, select=False, frames=['run_other/TCRMP20200101_clip_BPT_T101']),
     'quiet', 'not among'),
    ('a frame name with an empty project part', None,
     lambda root: tool_args(root, select=False, frames=['/TCRMP20200101_clip_BPT_T101']), 'quiet', 'empty'),
    ('a qualified frame name with an empty frame part', None,
     lambda root: tool_args(root, select=False, frames=[RUN + '/']), 'quiet', 'empty'),
    ('a qualified frame name with a line break', None,
     lambda root: tool_args(root, select=False, frames=[RUN + '/TCRMP\nX']), 'quiet', 'control character'),
    ('a qualified frame name with a backslash', None,
     lambda root: tool_args(root, select=False, frames=[RUN + '/a\\b']), 'quiet', 'path separator'),
    ('a frame name with a backslash', None, lambda root: tool_args(root, select=False, frames=['a\\b']),
     'quiet', 'path separator'),
    ('a frame name with a line break', None,
     lambda root: tool_args(root, select=False, frames=['TCRMP\nX']), 'quiet', 'control character'),
    ('an empty frame name', None, lambda root: tool_args(root, select=False, frames=['']), 'quiet', 'empty'),
    ('a frame name of ..', None, lambda root: tool_args(root, select=False, frames=['..']), 'quiet', 'name'),
    ('a project name climbing out', None, lambda root: tool_args(root, projects=('../run_x',)),
     'quiet', 'path separator'),
    ('a project name that is absolute', None, lambda root: tool_args(root, projects=('/etc',)),
     'quiet', 'path separator'),
    ('a project name without run_', None, lambda root: tool_args(root, projects=('_mask_registry',)),
     'quiet', 'run_'),
    ('a reason of 301 characters', None, lambda root: tool_args(root, reason='x' * 301), 'quiet', '300'),
    ('a reason of 4 characters', None, lambda root: tool_args(root, reason='fix!'), 'quiet', '5 to 300'),
    ('an empty reason', None, lambda root: tool_args(root, reason=''), 'quiet', '5 to 300'),
    ('a reason of spaces', None, lambda root: tool_args(root, reason='       '), 'quiet', '5 to 300'),
    ('a reason with a line break', None, lambda root: tool_args(root, reason='first line\nsecond'),
     'quiet', 'control character'),
    ('lower-case initials', None, lambda root: tool_args(root, by='lo'), 'quiet', 'capital'),
    ('one-letter initials', None, lambda root: tool_args(root, by='L'), 'quiet', 'capital'),
    ('five-letter initials', None, lambda root: tool_args(root, by='LOLOL'), 'quiet', 'capital'),
    ('initials with a digit', None, lambda root: tool_args(root, by='L0'), 'quiet', 'capital'),
    ('no selection', None, lambda root: tool_args(root, select=False), 'quiet', '--select'),
    ('--frame with two projects', None,
     lambda root: tool_args(root, projects=(RUN, 'run_other'), select=False,
                            frames=['TCRMP20200101_clip_BPT_T101']), 'quiet', 'exactly one --project'),
    ('an unknown --select rule', None,
     lambda root: tool_args(root, select=False) + ['--select', 'everything'], 'quiet', 'invalid choice'),
    ('a returned_to_review value that is not a list',
     lambda root, exp, seg: seg.write_text(json.dumps(
         {k: dict(v, returned_to_review='oops') if k == kind_key('manual_click blank') else v
          for k, v in json.loads(seg.read_text(encoding='utf-8')).items()}, indent=1), encoding='utf-8'),
     lambda root: tool_args(root), 'quiet', 'not a list'),
    ('a stem naming two frames',
     lambda root, exp, seg: seg.write_text(json.dumps(dict(
         json.loads(seg.read_text(encoding='utf-8')),
         **{'TCRMP20200101_clip_BPT_T101.jpg': frame(exp, 'TCRMP20200101_clip_BPT_T101.jpg', [])}),
         indent=1), encoding='utf-8'),
     lambda root: tool_args(root, select=False, frames=['TCRMP20200101_clip_BPT_T101']), 'quiet', 'matches 2'),
    ('a named frame entry that is not an object',
     lambda root, exp, seg: seg.write_text(json.dumps(dict(
         json.loads(seg.read_text(encoding='utf-8')), **{'TCRMP20200101_clip_BPT_T199.jpeg': ['odd']}),
         indent=1), encoding='utf-8'),
     lambda root: tool_args(root, select=False, frames=['TCRMP20200101_clip_BPT_T199']), 'quiet',
     'not an object'),
    ('a project folder that is a link out of the root',
     lambda root, exp, seg: (shutil.move(str(exp.parent), str(root.parent / 'elsewhere')),
                             os.symlink(str(root.parent / 'elsewhere'), str(exp.parent))),
     lambda root: tool_args(root), 'quiet', 'outside'),
    ('a root that is not a folder', None, lambda root: tool_args(root / 'nope'), 'quiet', 'not a folder'),
)


def test_refusals():
    """Every refusal exits 2, says why in a sentence, and writes nothing."""
    need_tool()
    for name, setup, build, ss, phrase in REFUSAL_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'inprocess'
            proj, exp, seg = make_project(root, kind_frames)
            if setup:
                setup(root, exp, seg)
            before = hash_tree(tmp)
            r = cli(build(root) + ['--apply'], ss=ss)
            check(f'refuses {name} with exit 2', r.returncode == 2, f'rc={r.returncode} {r.stdout[-300:]}')
            check(f'refusal for {name} says why ({phrase!r})', phrase in r.stderr, r.stderr[-400:])
            check(f'refusal for {name} writes nothing', tree_diff(before, hash_tree(tmp)) == ([], [], []),
                  tree_diff(before, hash_tree(tmp)))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames)
        before = hash_tree(tmp)
        code, out, err = in_process(tool_args(root, select=False, frames=['TCRMP\x00X'], apply=True))
        check('refuses a frame name holding a NUL byte', code == 2 and 'control character' in err, err)
        check('a NUL frame name writes nothing', tree_diff(before, hash_tree(tmp)) == ([], [], []))
        r = cli(tool_args(root, reason='x' * 301))
        check('a dry run refuses a bad reason too', r.returncode == 2, r.returncode)
        r = cli(tool_args(root), ss='busy')
        check('a live listener refusal says to stop every annotator tile in VEGA',
              'Stop every annotator tile in VEGA' in r.stderr, r.stderr)
        r = cli(tool_args(root))
        check('the port checkpoint names 5096 too', 'ports 5080 to 5089 and 5096' in r.stdout, r.stdout[:600])


def plan_for(root, stems=(), select=SELECT):
    """Read the fixture project and choose frames the way main() does (no write)."""
    plan = rftr.read_project(Path(root), RUN)
    rftr.choose_frames(plan, list(stems), select)
    return plan


def test_changed_mid_run():
    """A file another program saves between the read and the write is never overwritten."""
    need_tool()
    change = rftr.Change(at=datetime.now(AST).isoformat(timespec='seconds'), by=BY, reason=REASON)
    stamp = datetime.now(AST).strftime('%Y%m%d-%H%M%S')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, _, seg = make_project(root, kind_frames)
        plan = plan_for(root)
        data = load(seg)
        data[kind_key('no masks')]['outside_note'] = 'saved by another program'
        write_json(seg, data)
        outside = seg.read_bytes()
        refused = ''
        with fake_ss('quiet'):
            try:
                rftr.apply_plan(plan, change, stamp, log=lambda _line: None)
            except rftr.Refusal as exc:
                refused = str(exc)
        check('a file changed mid-run is refused', 'changed' in refused, refused)
        check('the outside change is kept, not overwritten', seg.read_bytes() == outside)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, _, seg = make_project(root, kind_frames)
        first, second = plan_for(root), plan_for(root, select=None,
                                                 stems=[KIND_STEMS['accepted labelled'][0]])
        refused = ''
        with fake_ss('quiet'):
            rftr.apply_plan(first, change, stamp, log=lambda _line: None)
            after_first = seg.read_bytes()
            try:
                rftr.apply_plan(second, change, stamp, log=lambda _line: None)
            except rftr.Refusal as exc:
                refused = str(exc)
        check('a second run racing the first is refused, never merged blind', 'changed' in refused, refused)
        check('the first run\'s write stands', seg.read_bytes() == after_first)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, _, seg = make_project(root, kind_frames)
        plan = plan_for(root)
        raw = seg.read_bytes()
        refused = ''
        with fake_ss('busy'):
            try:
                rftr.apply_plan(plan, change, stamp, log=lambda _line: None)
            except rftr.Refusal as exc:
                refused = str(exc)
        check('an annotator opened between the check and the write stops the write', '5081' in refused, refused)
        check('segmentations.json is untouched when the second port check fails', seg.read_bytes() == raw)


def test_ss_parsing():
    """The listener list is read from the real ss output shapes."""
    need_tool()
    with fake_ss('quiet'):
        ports = rftr.listening_ports()
    check('ss output is parsed into ports (IPv4, IPv6, interface suffix)', {5090, 22, 53} <= ports, ports)
    with fake_ss('busy'):
        ports = rftr.listening_ports()
    check('a listener on 5081 and 5089 is seen', {5081, 5089} <= ports, ports)


def cli_bytes(args, ss='quiet'):
    """Run the tool with raw argument bytes (as a shell passes undecodable text)."""
    env = dict(os.environ, PATH=ss_dir(ss), CUDA_VISIBLE_DEVICES='')
    return subprocess.run([PYTHON, str(TOOL), *args], capture_output=True, env=env, timeout=CHILD_TIMEOUT_S)


def test_crash_paths():
    """Undecodable text, an unwritable folder, a failed log and an odd frame key end in a sentence, not a crash."""
    need_tool()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames)
        before = hash_tree(tmp)
        base = [b'--root', str(root).encode(), b'--project', RUN.encode(), b'--by', b'LO']
        r = cli_bytes(base + [b'--select', SELECT.encode(), b'--reason', b'bad \xff reason text', b'--apply'])
        err = r.stderr.decode('utf-8', 'replace')
        check('a reason that is not valid text is refused (exit 2) without a crash',
              r.returncode == 2 and 'not valid text' in err and 'Traceback' not in err, err[-400:])
        r = cli_bytes(base + [b'--frame', b'TCRMP\xffX', b'--reason', REASON.encode(), b'--apply'])
        err = r.stderr.decode('utf-8', 'replace')
        check('a frame name that is not valid text is refused (exit 2) without a crash',
              r.returncode == 2 and 'not valid text' in err and 'Traceback' not in err, err[-400:])
        check('undecodable arguments write nothing', tree_diff(before, hash_tree(tmp)) == ([], [], []))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp, seg = make_project(root, kind_frames)
        raw = seg.read_bytes()
        os.chmod(seg.parent, 0o555)
        try:
            r = cli(tool_args(root, apply=True))
        finally:
            os.chmod(seg.parent, 0o775)
        check('a segmentations folder the tool cannot write to is a plain refusal (exit 2)',
              r.returncode == 2 and 'Traceback' not in r.stderr and 'cannot' in r.stderr, r.stderr[-500:])
        check('segmentations.json is unchanged when its folder cannot be written', seg.read_bytes() == raw)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp, seg = make_project(root, kind_frames)
        (exp / LOG_NAME).mkdir()
        r = cli(tool_args(root, apply=True))
        check('a log that cannot be written after the replace is exit 2, not a crash',
              r.returncode == 2 and 'Traceback' not in r.stderr, r.stderr[-500:])
        check('that refusal says plainly the file changed and the log line is missing',
              'changed' in r.stderr and 'missing' in r.stderr, r.stderr[-500:])
        check('the frames were sent back all the same', all(
            load(seg)[k]['exported'] is False for k in expected_kind_keys()))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        odd = 'TCRMP20200101_clip_BPT_T120\udcff.jpeg'
        _, exp, seg = make_project(root, kind_frames)
        write_json(seg, dict(load(seg), **{odd: frame(exp, odd, [outline(0)])}))
        dry, r = cli(tool_args(root)), cli(tool_args(root, apply=True))
        check('a frame key holding a lone surrogate is listed by the dry run without a crash',
              dry.returncode == 0 and 'Traceback' not in dry.stderr, dry.stderr[-400:])
        lines = (exp / LOG_NAME).read_text(encoding='utf-8').splitlines() if (exp / LOG_NAME).exists() else []
        check('that frame goes back and its log line names it',
              r.returncode == 0 and lines and odd in json.loads(lines[0])['frames']
              and load(seg)[odd]['exported'] is False, r.stderr[-400:])


def start_holder(seg, ready, hold, modify):
    """Start the stand-in importer and wait until it holds its lock; returns the process."""
    holder = subprocess.Popen([PYTHON, '-c', HOLDER_CODE, str(SCRIPTS), str(seg), str(ready),
                               kind_key('no masks'), str(hold), '1' if modify else '0'])
    deadline = time.monotonic() + WAIT_S
    while not Path(ready).exists() and time.monotonic() < deadline and holder.poll() is None:
        time.sleep(0.02)
    return holder


def test_importer_lock_deadline():
    """A held importer lock: the tool says it waits, refuses at its deadline, and leaves no temp file."""
    need_tool()
    change = rftr.Change(at=datetime.now(AST).isoformat(timespec='seconds'), by=BY, reason=REASON)
    stamp = datetime.now(AST).strftime('%Y%m%d-%H%M%S')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp, seg = make_project(root, kind_frames)
        raw = seg.read_bytes()
        plan = plan_for(root)
        holder = start_holder(seg, Path(tmp) / 'ready', LONG_HOLD_S, modify=False)
        saved = getattr(rftr, 'IMPORTER_LOCK_TIMEOUT_S', None)
        rftr.IMPORTER_LOCK_TIMEOUT_S = SHORT_DEADLINE_S
        out, refused = io.StringIO(), ''
        t0 = time.monotonic()
        try:
            with fake_ss('quiet'), contextlib.redirect_stdout(out):
                rftr.apply_plan(plan, change, stamp, log=print)
        except rftr.Refusal as exc:
            refused = str(exc)
        finally:
            took = time.monotonic() - t0
            if saved is not None:
                rftr.IMPORTER_LOCK_TIMEOUT_S = saved
            holder.kill()
            holder.wait(WAIT_S)
        check('the tool says it is waiting for the expert-ID importer', WAITING_LINE in out.getvalue(),
              out.getvalue()[-300:])
        check('at the deadline it refuses in a plain sentence naming the importer',
              'expert-ID importer' in refused and 'second' in refused, refused)
        check('the deadline is kept, not the importer\'s whole hold', took < LONG_HOLD_S / 2, f'{took:.1f}s')
        check('the original file is untouched at the deadline', seg.read_bytes() == raw)
        check('no temporary file is left at the deadline',
              not [n for n in os.listdir(seg.parent) if n.endswith('.tmp')], os.listdir(seg.parent))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp, seg = make_project(root, kind_frames)
        raw = seg.read_bytes()
        holder = start_holder(seg, Path(tmp) / 'ready', LONG_HOLD_S, modify=False)
        out_path = Path(tmp) / 'tool.out'
        env = dict(os.environ, PATH=ss_dir('quiet'), CUDA_VISIBLE_DEVICES='')
        with open(out_path, 'w') as out_file:
            tool = subprocess.Popen([PYTHON, str(TOOL), *map(str, tool_args(root, apply=True))],
                                    stdout=out_file, stderr=subprocess.STDOUT, env=env)
        deadline = time.monotonic() + 15
        while WAITING_LINE not in out_path.read_text() and time.monotonic() < deadline and tool.poll() is None:
            time.sleep(0.05)
        tool.send_signal(signal.SIGINT)
        try:
            rc = tool.wait(30)
        finally:
            holder.kill()
            holder.wait(WAIT_S)
        text = out_path.read_text()
        check('Ctrl-C while waiting for the importer stops the tool with exit 130 and a sentence',
              rc == 130 and 'Stopped' in text and 'Traceback' not in text, f'rc={rc} {text[-400:]}')
        check('Ctrl-C leaves no temporary file beside segmentations.json',
              not [n for n in os.listdir(seg.parent) if n.endswith('.tmp')], os.listdir(seg.parent))
        check('Ctrl-C leaves segmentations.json untouched', seg.read_bytes() == raw)


def test_importer_lock_gap():
    """An import saved under the importer's own lock is never overwritten by the tool."""
    need_tool()
    change = rftr.Change(at=datetime.now(AST).isoformat(timespec='seconds'), by=BY, reason=REASON)
    stamp = datetime.now(AST).strftime('%Y%m%d-%H%M%S')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp, seg = make_project(root, kind_frames)
        plan = plan_for(root)
        ready = Path(tmp) / 'holder_ready'
        holder = start_holder(seg, ready, HOLD_S, modify=True)
        result, saw_tmp = {}, False

        def apply():
            try:
                result['record'] = rftr.apply_plan(plan, change, stamp, log=lambda _line: None)
            except rftr.Refusal as exc:
                result['refused'] = str(exc)
        with fake_ss('quiet'):
            worker = threading.Thread(target=apply)
            worker.start()
            while worker.is_alive():
                saw_tmp = saw_tmp or any(n.endswith('.tmp') for n in os.listdir(seg.parent))
                time.sleep(0.02)
            worker.join()
        holder.wait(WAIT_S)
        final = load(seg)
        check('lock test: the stand-in importer held its lock and saved', ready.exists() and holder.returncode == 0)
        check('an import saved while the tool waits is never overwritten',
              final[kind_key('no masks')].get('imported_note') == IMPORT_NOTE)
        check('the tool never reports a change that is not in the file',
              'record' not in result or all(final[k]['exported'] is False for k in expected_kind_keys()), result)
        check('the tool refuses because the file changed under the importer lock',
              'changed' in result.get('refused', ''), result)
        check('the tool writes its temporary file before waiting for the importer lock', saw_tmp)
        check('no temporary file is left after the refusal',
              not [n for n in os.listdir(seg.parent) if n.endswith('.tmp')], os.listdir(seg.parent))


QUEUE_FRAMES = [('F01.jpeg', True, False), ('F02.jpeg', False, True), ('F03.jpeg', False, False),
                ('F04.jpeg', False, True), ('F05.jpeg', False, False), ('F06.jpeg', True, True)]


def test_returned_frames_lead_the_queue():
    """Both annotators put unexported returned frames first, sorted or shuffled; the rest keep their order."""
    need_tool()
    with tempfile.TemporaryDirectory(prefix='rftr_q_') as tmp:
        base = Path(tmp)
        exp = base / 'inprocess' / RUN / EXPORT
        (exp / 'routed_input').mkdir(parents=True)
        env = step4test_env(exp, PID, base / 'registry')
        combined = run_child({'scenario': 'queue_order', 'frames': QUEUE_FRAMES}, env, base)
        check('combined annotator: returned frames first, then the rest in name order',
              combined['sorted'] == ['F02.jpeg', 'F04.jpeg', 'F03.jpeg', 'F05.jpeg'], combined['sorted'])
        check('combined annotator: returned frames first when shuffle is on',
              all(set(order[:2]) == {'F02.jpeg', 'F04.jpeg'} and set(order[2:]) == {'F03.jpeg', 'F05.jpeg'}
                  for order in combined['shuffled']), combined['shuffled'])
        env_edit = dict(env, TCRMP_SESSION_MODE='edit', TCRMP_PROVENANCE_SOURCE='edit')
        edit = run_child({'scenario': 'queue_order', 'frames': QUEUE_FRAMES, 'app_src': str(EDITMASKS_SRC)},
                         env_edit, base)
        check('Edit Masks: returned unexported frames first, then every other frame in name order',
              edit['sorted'] == ['F02.jpeg', 'F04.jpeg', 'F01.jpeg', 'F03.jpeg', 'F05.jpeg', 'F06.jpeg'],
              edit['sorted'])
        check('Edit Masks: returned unexported frames first when shuffle is on',
              all(set(order[:2]) == {'F02.jpeg', 'F04.jpeg'} for order in edit['shuffled']), edit['shuffled'])


def kind_stem(name):
    """Stem of the kind_frames() frame with this kind name."""
    return KIND_STEMS[name][0]


def returned_in(seg_path):
    """Sorted keys of frames that carry a returned_to_review event."""
    return sorted(k for k, v in load(seg_path).items() if isinstance(v, dict) and v.get(RETURN_KEY))


def test_qualified_frames():
    """--frame RUN/STEM names a frame in one of several projects; refusals for the rest."""
    need_tool()
    a, b = 'run_26-10-16_AA', 'run_26-10-16_BB'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, _, seg_a = make_project(root, kind_frames, pid=a[len('run_'):])
        _, _, seg_b = make_project(root, kind_frames, pid=b[len('run_'):])
        args = tool_args(root, projects=(a, b), select=False,
                         frames=[f'{a}/{kind_stem("accepted labelled")}', f'{b}/{kind_stem("pending labelled")}'])
        dry = cli(args)
        check('qualified names across two projects: the dry run exits 0', dry.returncode == 0, dry.stderr[-400:])
        sections, current = {}, None
        for line in dry.stdout.splitlines():
            if line.startswith('run_'):
                current = line.split(':')[0]
            m = FRAME_LINE.match(line)
            if m and current:
                sections.setdefault(current, []).append(m.group(1))
        check('the dry run lists each qualified frame under its own project only',
              sections == {a: [kind_key('accepted labelled')], b: [kind_key('pending labelled')]}, sections)
        r = cli(args + ['--apply'])
        check('qualified names across two projects: apply exits 0', r.returncode == 0, r.stderr[-400:])
        check('each project sends back only the frame named with its own project',
              returned_in(seg_a) == [kind_key('accepted labelled')]
              and returned_in(seg_b) == [kind_key('pending labelled')],
              (returned_in(seg_a), returned_in(seg_b)))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        make_project(root, kind_frames, pid=a[len('run_'):])
        make_project(root, kind_frames, pid=b[len('run_'):])
        before = hash_tree(tmp)
        for label, frames, phrase in (
                ('a bare frame name with two projects', [kind_stem('accepted labelled')], 'exactly one --project'),
                ('a qualified frame whose project was not given', [f'run_26-10-16_CC/{kind_stem("no masks")}'],
                 'not among'),
                ('a qualified frame its project does not hold', [f'{a}/TCRMP20990101_clip_XXX_T999'], 'is not in')):
            r = cli(tool_args(root, projects=(a, b), select=False, frames=frames, apply=True))
            check(f'refuses {label} (exit 2, says why)', r.returncode == 2 and phrase in r.stderr, r.stderr[-400:])
        check('the refused qualified names wrote nothing', tree_diff(before, hash_tree(tmp)) == ([], [], []))
        r = cli(tool_args(root, projects=(a,), select=False,
                          frames=[kind_stem('accepted labelled'), f'{a}/{kind_stem("accepted labelled")}'],
                          apply=True))
        check('a bare and a qualified name for the same frame with one project send it back once',
              r.returncode == 0 and listed_frames(r.stdout) == [kind_key('accepted labelled')], r.stdout[-400:])


def test_two_runs_in_order():
    """The --select run, then a qualified-name run with its own reason; an overlap is skipped and named."""
    need_tool()
    a, b = 'run_26-10-16_AA', 'run_26-10-16_BB'
    second_reason = 'PA outline over a PA survey point; please check it again.'
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / 'inprocess'
        _, exp_a, seg_a = make_project(root, kind_frames, pid=a[len('run_'):])
        _, exp_b, seg_b = make_project(root, kind_frames, pid=b[len('run_'):])
        first = cli(tool_args(root, projects=(a, b), apply=True))
        check('two runs: the first (--select) run applies', first.returncode == 0, first.stderr[-400:])
        overlap = f'{a}/{kind_stem("manual_click blank")}'
        second = tool_args(root, projects=(a, b), select=False, reason=second_reason,
                           frames=[f'{a}/{kind_stem("accepted labelled")}', f'{b}/{kind_stem("rejected blank")}',
                                   overlap])
        dry = cli(second)
        check('two runs: the second dry run works after the first apply', dry.returncode == 0, dry.stderr[-400:])
        check('two runs: the second dry run lists its new frames',
              listed_frames(dry.stdout) == sorted([kind_key('accepted labelled'), kind_key('rejected blank')]),
              dry.stdout[-600:])
        check('two runs: a frame the first run already sent back is skipped and named',
              f'{kind_key("manual_click blank")}: skipped' in dry.stdout and 'not exported' in dry.stdout,
              dry.stdout[-600:])
        r = cli(second + ['--apply'])
        check('two runs: the second run applies', r.returncode == 0, r.stderr[-400:])
        data_a, data_b = load(seg_a), load(seg_b)
        check('two runs: the overlapping frame keeps only the first run\'s event',
              [e['reason'] for e in data_a[kind_key('manual_click blank')][RETURN_KEY]] == [REASON])
        check('two runs: the second run\'s frames carry the second reason',
              [e['reason'] for e in data_a[kind_key('accepted labelled')][RETURN_KEY]] == [second_reason]
              and [e['reason'] for e in data_b[kind_key('rejected blank')][RETURN_KEY]] == [second_reason])
        for exp in (exp_a, exp_b):
            lines = (exp / LOG_NAME).read_text(encoding='utf-8').splitlines()
            check(f'two runs: {exp.parent.name} logs one line per run, the second with its own reason',
                  len(lines) == 2 and json.loads(lines[1])['reason'] == second_reason, lines)
        check('two runs: each project keeps two backups, one per run',
              all(len(list((exp / '_superseded').glob('*/' + BACKUP_LABEL + '/*.json'))) == 2
                  for exp in (exp_a, exp_b)))


# ---------------------------------------------------------------- annotator round trip
def step4test_env(export_dir, project_id, registry_dir, reviewer='AT', targets='PA'):
    """The environment the orchestrator gives a step4test annotator launch.

    Mirrors pipeline_orchestrator/app.py (the annotator launch, around lines
    6158 to 6232): MANUAL_ANNOTATE 1, target-only 1, step4test provenance,
    batch size 10, the reviewer's initials. The stores point at test folders,
    the SAM3 devices at cpu and CUDA_VISIBLE_DEVICES is empty, so no card is
    touched; the queue, save and export paths never read the device.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('TCRMP_') or k in ('TCRMP_EXPERT_LIBRARY_DIR', 'TCRMP_CLIP_INDEX_DIR')}
    env.update({
        'TCRMP_INPUT_DIR': str(Path(export_dir) / 'routed_input'),
        'TCRMP_EXPORT_DIR': str(export_dir),
        'TCRMP_PROJECT_ID': project_id,
        'TCRMP_PROJECT_NAME': project_id,
        'TCRMP_TARGET_SPECIES': targets,
        'TCRMP_TARGET_SPECIES_ONLY': '1',
        'TCRMP_REFERENCE_DEFAULT': '1',
        'TCRMP_REVIEW_BATCH_SIZE': '10',
        'TCRMP_SYMLINK_IMAGES': '1',
        'TCRMP_REVIEWER': reviewer,
        'TCRMP_MANUAL_ANNOTATE': '1',
        'TCRMP_PROVENANCE_SOURCE': 'step4test',
        'TCRMP_MASK_REGISTRY_DIR': str(registry_dir),
        'TCRMP_REVIEW_DIR': str(Path(registry_dir).parent / '_review_dir_unused'),
        'TCRMP_REVIEW_GIT_PUSH': '0',
        'TCRMP_SAM3_DEVICE_TRACKER': 'cpu',
        'TCRMP_SAM3_DEVICE_EXEMPLAR': 'cpu',
        'CUDA_VISIBLE_DEVICES': '',
        'VICARIUS_HOST_OPEN': '0',
        'VICARIUS_LOCK_BYPASS': '1',
        'TZ': AST_ZONE,
    })
    return env


def run_child(spec, env, workdir):
    """Run one annotator scenario in a child process; returns its result dict."""
    spec_path = Path(workdir) / f'child_{spec["scenario"]}_{time.monotonic_ns()}.json'
    result_path = spec_path.with_suffix('.result.json')
    spec = dict(spec, result=str(result_path))
    spec_path.write_text(json.dumps(spec), encoding='utf-8')
    r = subprocess.run([PYTHON, str(Path(__file__).resolve()), '--annotator-child', str(spec_path)],
                       capture_output=True, text=True, env=env, timeout=CHILD_TIMEOUT_S)
    if r.returncode != 0 or not result_path.exists():
        raise RuntimeError(f'annotator child {spec["scenario"]} failed (exit {r.returncode}):\n'
                           f'{r.stdout[-3000:]}\n{r.stderr[-3000:]}')
    return load(result_path)


def disk_frames(seg_path):
    """segmentations.json as it is on disk now."""
    return load(seg_path)


def scenario_export_all(app, client, spec):
    """Open the project the way the page does and export every batch (AT's first pass)."""
    out = {'configure': client.post('/api/configure', data='{}', content_type='application/json').status_code}
    exports = []
    for _ in range(spec.get('max_batches', 50)):
        if not app.session['review_files']:
            break
        r = client.post('/api/export_batch')
        exports.append(r.status_code)
    out['exports'] = exports
    out['queue_after'] = list(app.session['review_files'])
    return out


def scenario_queue(app, client, spec):
    """Open the project and report the queue, the unlabelled set and the batch of each frame."""
    r = client.post('/api/configure', data='{}', content_type='application/json')
    queue = list(app.session['review_files'])
    size = app.session['review_batch_size']
    return {'configure': r.status_code, 'queue': queue, 'batch_size': size,
            'unlabeled': client.get('/api/unlabeled').get_json().get('frames', []),
            'batch_of': {fn: queue.index(fn) // size + 1 for fn in spec.get('frames', []) if fn in queue},
            'settings': settings_of(app)}


def settings_of(app):
    """The annotator settings that decide the queue."""
    return {'target_species_only': int(app.cfg.TARGET_SPECIES_ONLY), 'manual_annotate': bool(app.cfg.MANUAL_ANNOTATE),
            'source': app.cfg.PROVENANCE_SOURCE, 'session_mode': app.cfg.SESSION_MODE,
            'targets': app._target_species(), 'reviewer': app._current_reviewer(),
            'batch_size': app.session['review_batch_size']}


def frame_state(data, fn, mask_id=None):
    """The parts of one frame the round trip checks."""
    seg = data.get(fn, {})
    state = {'exported': seg.get('exported'), 'events': seg.get(RETURN_KEY), 'outside_note': seg.get('outside_note')}
    if mask_id is not None:
        m = next((m for m in seg.get('masks', []) if m.get('id') == mask_id), {})
        state.update(species=m.get('species'), status=m.get('status'))
    return state


def scenario_roundtrip(app, client, spec):
    """Queue, label and accept, save, save merged with an outside change, reject, export."""
    seg_path = Path(spec['export_dir']) / 'segmentations' / 'segmentations.json'
    lab, rej, other = spec['label'], spec['reject'], spec['outside']
    out = {'configure': client.post('/api/configure', data='{}', content_type='application/json').status_code}
    out['settings'] = settings_of(app)
    out['queue'] = list(app.session['review_files'])
    out['unlabeled'] = client.get('/api/unlabeled').get_json().get('frames', [])
    start = disk_frames(seg_path)
    out['before'] = {'label': frame_state(start, lab['frame']), 'reject': frame_state(start, rej['frame']),
                     'outside': frame_state(start, other)}
    body = {'relabel': {str(lab['mask']): {'species': 'PA', 'name': 'Porites astreoides', 'category': 'Coral'}},
            'updates': {str(lab['mask']): 'accepted'}}
    out['label_put'] = client.put(f"/api/image/{quote(lab['frame'])}/masks", json=body).get_json()
    out['flush_save'] = client.post('/api/flush').get_json()
    out['after_save'] = frame_state(disk_frames(seg_path), lab['frame'], lab['mask'])
    data = disk_frames(seg_path)
    for fn in (other, lab['frame']):
        data[fn]['outside_note'] = 'written by another program'
    write_json(seg_path, data)
    out['reject_put'] = client.put(f"/api/image/{quote(rej['frame'])}/masks",
                                   json={'updates': {str(rej['mask']): 'rejected'}}).get_json()
    out['relabel_again_put'] = client.put(f"/api/image/{quote(lab['frame'])}/masks", json=body).get_json()
    out['flush_merge'] = client.post('/api/flush').get_json()
    merged = disk_frames(seg_path)
    out['after_merge'] = {'label': frame_state(merged, lab['frame'], lab['mask']),
                          'reject': frame_state(merged, rej['frame'], rej['mask']),
                          'outside': frame_state(merged, other)}
    r = client.post('/api/export_batch')
    out['export'] = {'status': r.status_code, 'body': r.get_json()}
    exported = disk_frames(seg_path)
    out['after_export'] = {'label': frame_state(exported, lab['frame'], lab['mask']),
                           'reject': frame_state(exported, rej['frame'], rej['mask'])}
    out['queue_after'] = list(app.session['review_files'])
    out['masks_after'] = {fn: exported[fn]['masks'] for fn in (lab['frame'], rej['frame'])}
    return out


def scenario_queue_order(app, client, spec):
    """Seed frames (name, exported, returned) and read the queue sorted, then shuffled ten times."""
    event = {'at': '2026-10-16T09:00:00-04:00', 'by': BY, 'reason': REASON, 'tool': 'test',
             'previous_exported': True}
    app.session['segmentations'] = {
        name: dict({'image_path': name, 'image_path_abs': '', 'image_width': 10, 'image_height': 10,
                    'masks': [], 'reviewed': True, 'exported': exported},
                   **({RETURN_KEY: [dict(event)]} if returned else {}))
        for name, exported, returned in spec['frames']}
    app.session['shuffle'] = False
    app._build_review_list()
    out = {'sorted': list(app.session['review_files']), 'shuffled': []}
    app.session['shuffle'] = True
    for _ in range(10):
        app._build_review_list()
        out['shuffled'].append(list(app.session['review_files']))
    return out


SCENARIOS = {'export_all': scenario_export_all, 'queue': scenario_queue, 'roundtrip': scenario_roundtrip,
             'queue_order': scenario_queue_order}


def child_main(spec_path):
    """Entry point of an annotator child process (one scenario, results to a JSON file)."""
    spec = load(spec_path)
    sys.path.insert(0, spec.get('app_src', str(ANNOTATOR_SRC)))
    import app as annotator  # noqa: E402  (imported only here, with the launch environment set)
    annotator.app.config['TESTING'] = True
    with annotator.app.test_client() as client:
        result = SCENARIOS[spec['scenario']](annotator, client, spec)
    # The real annotator saves once more when it quits (SIGTERM, Ctrl+C or a
    # normal exit run flush_before_exit); the child does the same, so the
    # cleared ledger marker of the last export is on disk as it would be.
    annotator.flush_before_exit()
    Path(spec['result']).write_text(json.dumps(result, default=str), encoding='utf-8')
    return 0


def checker_config(base, root, registry_vocab=None):
    """A checker Config for a fixture tree (vocabulary, mirror, snapshot and constants in base)."""
    base = Path(base)
    vocab = base / 'vocab.csv'
    if not vocab.exists():
        write_csv(vocab, ['code', 'category', 'name'], [{'code': 'PA', 'category': 'Coral',
                                                         'name': 'Porites astreoides'}])
        shutil.copy(vocab, base / 'mirror.csv')
        (base / 'pipeline.yaml').write_text(f'paths:\n  duplicate_master_codes: {base / "mirror.csv"}\n',
                                            encoding='utf-8')
        (base / 'const.py').write_text("_M = {'manual_box', 'manual_click', 'merged'}\n", encoding='utf-8')
        (base / 'archive').mkdir(exist_ok=True)
    daily = base / 'daily'
    shutil.rmtree(daily, ignore_errors=True)
    (daily / (datetime.now(AST) - timedelta(hours=1)).strftime('%Y%m%d-%H%M%S')).mkdir(parents=True)
    return cic.Config(root=str(root), archive_dir=str(base / 'archive'), vocab_path=str(vocab),
                      pipeline_config=str(base / 'pipeline.yaml'), scripts_dir=str(SCRIPTS),
                      snapshot_dir=str(daily), manual_source_files=[str(base / 'const.py')],
                      exceptions_path=str(base / 'no_exceptions.yaml'), now=datetime.now(AST))


@contextlib.contextmanager
def full_examples():
    """Make the checker list every item in a finding, so no frame hides in '(+N more)'."""
    original = cic._examples
    cic._examples = lambda items, limit=None: ', '.join(str(i) for i in items)
    try:
        yield
    finally:
        cic._examples = original


def run_checker(cfg):
    """Run the integrity checker with every item listed; returns its report."""
    with full_examples():
        return cic.run_checks(cfg)


def findings_naming(report, stems, levels=('FAIL', 'WARN')):
    """Findings at these levels whose message names one of the stems."""
    return [f for f in report.findings if f.level in levels and any(s in f.message for s in stems)]


def project_issues(report, label):
    """(rule, level) pairs of every non-PASS finding for one project."""
    return sorted({(f.rule, f.level) for f in report.findings if f.project == label and f.level != 'PASS'})


ROUNDTRIP_FRAMES = {
    'TCRMP20200101_clip_BPT_T101.jpeg': [outline(0), outline(1, 'PA', 'accepted')],
    'TCRMP20200101_clip_BPT_T102.jpeg': [outline(0, source='manual_box')],
    'TCRMP20200101_clip_BPT_T103.jpeg': [outline(0, source='merged')],
    'TCRMP20200101_clip_BPT_T104.jpeg': [outline(0, 'PA', 'accepted')],
    'TCRMP20200101_clip_BPT_T105.jpeg': [],
}
ROUNDTRIP_RETURNED = sorted(ROUNDTRIP_FRAMES)[:3]


def roundtrip_project(root):
    """A project of five unexported frames with images and routed prompts, as Step 4 leaves it."""
    from PIL import Image

    def frames(exp):
        raw = exp / 'routed_input' / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        for k in ROUNDTRIP_FRAMES:
            Image.new('RGB', (64, 48), (10, 60, 90)).save(raw / k)
        write_json(exp / 'routed_input' / 'ids' / 'sam_click_prompts.json',
                   {k: {'raw': 'raw/' + k, 'points': [{'x': 10, 'y': 10, 'species': 'PA', 'label': 'A'}]}
                    for k in ROUNDTRIP_FRAMES})
        return {k: frame(exp, k, json.loads(json.dumps(m)), exported=False, reviewed=False)
                for k, m in ROUNDTRIP_FRAMES.items()}
    proj, exp, seg = make_project(root, frames)
    for name in ('label_provenance.csv', 'export_manifest.csv', 'class_map.json'):
        (exp / name).unlink()
    shutil.rmtree(exp / 'all_labels')
    return proj, exp, seg


def registry_rows(registry_dir, stem):
    """Registry manifest rows for one frame stem."""
    path = Path(registry_dir) / 'manifest.csv'
    with open(path, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['basename'] == stem]


def csv_rows(path, stem):
    """Rows of a CSV whose basename is stem."""
    with open(path, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['basename'] == stem]


def check_roundtrip_result(tag, res, exp, registry_dir, returned, lab, rej, before_files, pid):
    """Assertions shared by the fixture and the real-copy round trip."""
    check(f'{tag}: the annotator opens the project (POST /api/configure)', res['configure'] == 200, res['configure'])
    s = res['settings']
    check(f'{tag}: the launch settings are a step4test launch',
          s['target_species_only'] == 1 and s['manual_annotate'] and s['source'] == 'step4test'
          and s['session_mode'] == 'configure' and s['targets'] == ['PA'] and s['batch_size'] == 10, s)
    check(f'{tag}: every returned frame is in the review queue', set(returned) <= set(res['queue']),
          sorted(set(returned) - set(res['queue'])))
    check(f'{tag}: every returned frame is in "Frames with a mask missing its species"',
          set(returned) <= set(res['unlabeled']), sorted(set(returned) - set(res['unlabeled'])))
    lab_fn, rej_fn = lab['frame'], rej['frame']
    events_before = res['before']['label']['events']
    check(f'{tag}: the frames start with one returned_to_review event each',
          all(isinstance(res['before'][k]['events'], list) and len(res['before'][k]['events']) == 1
              and res['before'][k]['exported'] is False for k in ('label', 'reject', 'outside')), res['before'])
    check(f'{tag}: labelling PA and accepting is answered ok',
          res['label_put'].get('ok') and not res['label_put'].get('needs_species'), res['label_put'])
    a = res['after_save']
    check(f'{tag}: a save keeps returned_to_review', a['events'] == events_before and a['exported'] is False, a)
    check(f'{tag}: the save holds the label and the accept', a['species'] == 'PA' and a['status'] == 'accepted', a)
    m = res['after_merge']
    check(f'{tag}: a save merged with an outside change keeps returned_to_review on every frame',
          m['label']['events'] == events_before and m['reject']['events'] == res['before']['reject']['events']
          and m['outside']['events'] == res['before']['outside']['events'], m)
    check(f'{tag}: the merge keeps the outside change on a frame this window did not edit',
          m['outside']['outside_note'] == 'written by another program', m['outside'])
    check(f'{tag}: the merge keeps this window\'s label, accept and reject',
          m['label']['species'] == 'PA' and m['label']['status'] == 'accepted'
          and m['reject']['status'] == 'rejected', m)
    check(f'{tag}: the merge is reported to the page', res['flush_merge'].get('conflict')
          or res['flush_merge'].get('merged'), res['flush_merge'])
    e = res['export']
    check(f'{tag}: exporting the batch succeeds', e['status'] == 200 and e['body'].get('ok'), e)
    after = res['after_export']
    check(f'{tag}: the exported frames are marked exported again',
          after['label']['exported'] is True and after['reject']['exported'] is True, after)
    check(f'{tag}: returned_to_review stays on the re-exported frames as history',
          after['label']['events'] == events_before
          and after['reject']['events'] == res['before']['reject']['events'], after)
    check(f'{tag}: the re-exported frames leave the queue',
          lab_fn not in res['queue_after'] and rej_fn not in res['queue_after'], res['queue_after'][:12])
    lab_stem, rej_stem = os.path.splitext(lab_fn)[0], os.path.splitext(rej_fn)[0]
    masks = res['masks_after']
    accepted_lab = [mm for mm in masks[lab_fn] if mm.get('status') == 'accepted' and (mm.get('species') or '')]
    label_file = exp / 'all_labels' / (lab_stem + '.txt')
    lines = [ln for ln in label_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    class_map = load(exp / 'class_map.json')
    check(f'{tag}: the label file is rewritten with the new PA line',
          len(lines) == len(accepted_lab) and any(ln.split()[0] == str(class_map['PA']) for ln in lines)
          and label_file.read_bytes() != before_files['label'], lines)
    lab_row = csv_rows(exp / 'export_manifest.csv', lab_stem)
    rej_row = csv_rows(exp / 'export_manifest.csv', rej_stem)
    check(f'{tag}: the export manifest row is rewritten (counts, outcome, new batch)',
          len(lab_row) == 1 and int(lab_row[0]['n_masks']) == len(accepted_lab)
          and int(lab_row[0]['PA']) == sum(1 for mm in accepted_lab if mm['species'] == 'PA')
          and lab_row[0]['outcome'] == 'exported_with_masks'
          and lab_row[0]['batch_name'] != before_files['lab_batch'], lab_row)
    check(f'{tag}: the rejected frame\'s manifest row is rewritten too',
          len(rej_row) == 1 and rej_row[0]['batch_name'] != before_files['rej_batch'], rej_row)
    lab_mask = next(mm for mm in masks[lab_fn] if mm['id'] == lab['mask'])
    rej_mask = next(mm for mm in masks[rej_fn] if mm['id'] == rej['mask'])
    lab_uid = build_registry_record(lab_fn, lab_mask, pid, pid)['uid']
    rej_uid = build_registry_record(rej_fn, rej_mask, pid, pid)['uid']
    lab_reg = {r['uid']: r for r in registry_rows(registry_dir, lab_stem)}
    rej_reg = {r['uid']: r for r in registry_rows(registry_dir, rej_stem)}
    check(f'{tag}: the registry row of the labelled outline says PA, accepted (isolated store)',
          lab_reg.get(lab_uid, {}).get('species') == 'PA' and lab_reg[lab_uid]['status'] == 'accepted',
          lab_reg.get(lab_uid))
    check(f'{tag}: the registry row of the rejected outline says rejected',
          rej_reg.get(rej_uid, {}).get('status') == 'rejected', rej_reg.get(rej_uid))
    ledger_lab = {r['label']: r for r in csv_rows(exp / 'label_provenance.csv', lab_stem)}
    ledger_rej = {r['label']: r for r in csv_rows(exp / 'label_provenance.csv', rej_stem)}
    check(f'{tag}: the ledger row of the labelled frame says found_manual, reviewer AT',
          ledger_lab.get('PA', {}).get('outcome') == 'found_manual'
          and ledger_lab['PA'].get('reviewer') == 'AT', ledger_lab)
    check(f'{tag}: the ledger rows of the two frames were rewritten after the return',
          ledger_lab.get('PA', {}).get('at', '') > before_files['return_at']
          and all(r.get('at', '') > before_files['return_at'] for r in ledger_rej.values()),
          (ledger_lab, ledger_rej, before_files['return_at']))


def test_annotator_roundtrip_fixture():
    """Fixture: AT's first export, the return, the queue, a save, a merged save, an export."""
    need_tool()
    with tempfile.TemporaryDirectory(prefix='rftr_rt_') as tmp:
        base = Path(tmp)
        root = base / 'inprocess'
        registry = root / '_mask_registry'
        proj, exp, seg = roundtrip_project(root)
        env = step4test_env(exp, PID, registry)
        first = run_child({'scenario': 'export_all', 'export_dir': str(exp)}, env, base)
        check('fixture: AT\'s first pass exports every frame', first['configure'] == 200
              and first['exports'] and all(c == 200 for c in first['exports']) and not first['queue_after'], first)
        rep = run_checker(checker_config(base, root))
        check('fixture: the checker reads the exported project clean before the return',
              project_issues(rep, PID) == [], project_issues(rep, PID))
        time.sleep(EXPORT_SECOND)
        r = cli(tool_args(root, apply=True))
        check('fixture: the tool sends the three outlines back', r.returncode == 0
              and listed_frames(r.stdout) == ROUNDTRIP_RETURNED, r.stdout[-600:] + r.stderr[-400:])
        rep = run_checker(checker_config(base, root))
        issues = project_issues(rep, PID)
        check('fixture: while the frames wait, the checker shows INFO only (no FAIL or WARN)',
              issues and all(level == 'INFO' for _, level in issues), issues)
        infos = [f.message for f in rep.findings if f.project == PID and f.level == 'INFO']
        check('fixture: the INFO cites the return (who and why)', infos and all(BY in m and REASON[:30] in m
                                                                                for m in infos), infos)
        data = load(seg)
        lab = {'frame': ROUNDTRIP_RETURNED[0], 'mask': 0}
        rej = {'frame': ROUNDTRIP_RETURNED[1], 'mask': 0}
        rec = json.loads((exp / LOG_NAME).read_text(encoding='utf-8').splitlines()[-1])
        before_files = {
            'label': (exp / 'all_labels' / (os.path.splitext(lab['frame'])[0] + '.txt')).read_bytes(),
            'lab_batch': csv_rows(exp / 'export_manifest.csv', os.path.splitext(lab['frame'])[0])[0]['batch_name'],
            'rej_batch': csv_rows(exp / 'export_manifest.csv', os.path.splitext(rej['frame'])[0])[0]['batch_name'],
            'return_at': rec['at']}
        check('fixture: the labelled frame starts with a blank pending hand outline',
              data[lab['frame']]['masks'][0]['species'] == '' and data[lab['frame']]['masks'][0]['status'] == 'pending')
        time.sleep(EXPORT_SECOND)
        res = run_child({'scenario': 'roundtrip', 'export_dir': str(exp), 'label': lab, 'reject': rej,
                         'outside': ROUNDTRIP_RETURNED[2]}, env, base)
        check('fixture: the queue holds exactly the returned frames', res['queue'] == ROUNDTRIP_RETURNED,
              res['queue'])
        check_roundtrip_result('fixture', res, exp, registry, ROUNDTRIP_RETURNED, lab, rej, before_files, PID)
        rep = run_checker(checker_config(base, root))
        left = [(f.rule, f.level, f.message) for f in rep.findings if f.project == PID and f.level != 'PASS']
        conflict = [x for x in left if x[:2] == ('R13', 'WARN') and 'Conflict file' in x[2]
                    and lab['frame'] in x[2]]
        check('fixture: after the export the checker reads the project clean again, apart from the '
              'conflict file the deliberate both-sides edit leaves',
              len(conflict) == 1 and len(left) == 1, [(r, lv, m[:200]) for r, lv, m in left])


# ---------------------------------------------------------------- real copies (opt-in)
def real_fingerprint():
    """Read-only fingerprint of AT's two real projects and the live registry manifest."""
    found = {}
    for name in REAL_PROJECTS:
        exp = REAL_INPROCESS / name / EXPORT
        for rel in REAL_HASHED_FILES:
            p = exp / rel
            found[str(p)] = sha256(p) if p.exists() else None
        for dp, dns, fns in os.walk(exp):
            for n in dns + fns:
                p = os.path.join(dp, n)
                st = os.lstat(p)
                found[p + '#stat'] = (st.st_size, st.st_mtime_ns, stat.S_IMODE(st.st_mode),
                                      os.readlink(p) if os.path.islink(p) else '')
        found[str(REAL_INPROCESS / name / 'project.json')] = sha256(REAL_INPROCESS / name / 'project.json')
    reg = REAL_INPROCESS / '_mask_registry' / 'manifest.csv'
    found[str(reg)] = sha256(reg)
    return found


def copy_real_project(name, dest_root):
    """Copy one real project's annotation files into dest_root (reads the real tree only)."""
    src = REAL_INPROCESS / name / EXPORT
    dst = Path(dest_root) / name / EXPORT
    for rel in REAL_COPY_FILES:
        if (src / rel).exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dst / rel)
    shutil.copytree(src / 'all_labels', dst / 'all_labels')
    shutil.copytree(src / 'all_images', dst / 'all_images', symlinks=True)
    raw = dst / 'routed_input' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    for n in sorted(os.listdir(src / 'routed_input' / 'raw')):
        os.symlink(str(src / 'routed_input' / 'raw' / n), str(raw / n))
    pj = load(REAL_INPROCESS / name / 'project.json')
    pj['project_dir'] = str(Path(dest_root) / name)
    write_json(Path(dest_root) / name / 'project.json', pj, indent=2)
    return dst


def copy_real_registry(dest_root):
    """Copy the two projects' rows of the live registry into dest_root/_mask_registry."""
    ids = {n[len('run_'):] for n in REAL_PROJECTS}
    with open(REAL_INPROCESS / '_mask_registry' / 'manifest.csv', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, [r for r in reader if r['project_id'] in ids]
    write_csv(Path(dest_root) / '_mask_registry' / 'manifest.csv', header, rows)
    return Path(dest_root) / '_mask_registry'


def listed_by_project(stdout):
    """{project: [frame stems the output lists as going back]} from the tool's output."""
    found, current = {}, None
    for line in stdout.splitlines():
        if line.startswith('run_'):
            current = line.split()[0].rstrip(':')
        m = FRAME_LINE.match(line)
        if m and current:
            found.setdefault(current, []).append(os.path.splitext(m.group(1))[0])
    return found


def unlabelled_among(exp, keys):
    """The keys whose frame holds a mask with no species (the gear menu's find)."""
    data = load(exp / 'segmentations' / 'segmentations.json')
    return sorted(k for k in keys if any(not (m.get('species') or '') for m in data[k].get('masks', [])))


def second_run_args(root):
    """The second command: the 10 frames named RUN/STEM across both projects, with their own reason."""
    args = ['--root', root, '--project', REAL_PROJECTS[0], '--project', REAL_PROJECTS[1]]
    for name in ('run_26-10-04_AT', 'run_26-09-18_AT'):
        for stem in EXPECTED_SECOND[name]:
            args += ['--frame', f'{name}/{stem}']
    return args + ['--by', BY, '--reason', REASON_SECOND]


def e2e_second_run(root, exps, first_keys, real_path):
    """After the first run's apply: the second command's dry run and apply on the copies."""
    args = second_run_args(root)
    dry = cli(args, path=real_path)
    print('---- second dry run on the copies, after the first apply (real ss) ----')
    print(dry.stdout + dry.stderr)
    print('---- end of second dry run ----')
    check('e2e: the second dry run exits 0 after the first apply', dry.returncode == 0, dry.stderr[-400:])
    listed = listed_by_project(dry.stdout)
    for name in REAL_PROJECTS:
        check(f'e2e: the second dry run lists exactly the 5 frames for {name}',
              sorted(listed.get(name, [])) == sorted(EXPECTED_SECOND[name]),
              sorted(set(listed.get(name, [])) ^ set(EXPECTED_SECOND[name])))
    check('e2e: the second dry run skips nothing (no overlap with the first run)',
          'skipped' not in dry.stdout.split('Step 4 complete')[-1].split('Dry run')[0], dry.stdout[-800:])
    r = cli(args + ['--apply'], path=real_path)
    print('---- second apply on the copies ----')
    print(r.stdout + r.stderr)
    print('---- end of second apply ----')
    check('e2e: the second --apply exits 0', r.returncode == 0, r.stderr[-400:])
    second_keys = {}
    for name in REAL_PROJECTS:
        d = load(exps[name] / 'segmentations' / 'segmentations.json')
        by_stem = {os.path.splitext(k)[0]: k for k in d}
        second_keys[name] = sorted(by_stem[s] for s in EXPECTED_SECOND[name])
        got = sorted(k for k, v in d.items() if v.get(RETURN_KEY))
        check(f'e2e: {name} now holds the first run\'s and the second run\'s frames, nothing else',
              got == sorted(first_keys[name] + second_keys[name]),
              sorted(set(got) ^ set(first_keys[name] + second_keys[name])))
        check(f'e2e: {name}: each second-run frame carries one event, with the second reason',
              all([e['reason'] for e in d[k][RETURN_KEY]] == [REASON_SECOND] for k in second_keys[name]))
        check(f'e2e: {name}: each first-run frame still carries only the first reason',
              all([e['reason'] for e in d[k][RETURN_KEY]] == [REASON] for k in first_keys[name]))
    return second_keys


def e2e_real(workdir):
    """Copies of AT's two projects: both runs in order (23, then 10), queue, round trip, checker, cleanup."""
    need_tool()
    workdir = Path(workdir)
    check('e2e: the work folder does not exist yet', not workdir.exists(), workdir)
    fp_before = real_fingerprint()
    try:
        root = workdir / 'inprocess'
        exps = {name: copy_real_project(name, root) for name in REAL_PROJECTS}
        registry = copy_real_registry(root)
        (workdir / 'archive').mkdir()
        args = ['--root', root, '--project', REAL_PROJECTS[0], '--project', REAL_PROJECTS[1],
                '--select', SELECT, '--by', BY, '--reason', REASON]
        real_path = os.environ.get('PATH', '')
        dry = cli(args, path=real_path)
        print('---- dry run on the copies (real ss) ----')
        print(dry.stdout + dry.stderr)
        print('---- end of dry run ----')
        check('e2e: the dry run exits 0', dry.returncode == 0, dry.stderr[-400:])
        by_project = listed_by_project(dry.stdout)
        for name in REAL_PROJECTS:
            check(f'e2e: the dry run lists exactly the plan\'s {len(EXPECTED_REAL[name])} frames for {name}',
                  sorted(by_project.get(name, [])) == sorted(EXPECTED_REAL[name]),
                  sorted(set(by_project.get(name, [])) ^ set(EXPECTED_REAL[name])))
        cfg = checker_config(workdir, root)
        cfg.vocab_path, cfg.pipeline_config, cfg.manual_source_files = cic.DEFAULT_VOCAB, cic.DEFAULT_PIPELINE_CONFIG, None
        stems = [s for v in list(EXPECTED_REAL.values()) + list(EXPECTED_SECOND.values()) for s in v]
        check('e2e: none of the second run\'s 10 frames is among the first run\'s 23',
              all(not set(EXPECTED_SECOND[n]) & set(EXPECTED_REAL[n]) for n in REAL_PROJECTS)
              and len(set(stems)) == 33, sorted({s for s in stems if stems.count(s) > 1}))
        baseline = run_checker(cfg)
        check('e2e: before the return no FAIL names the 33 frames',
              not findings_naming(baseline, stems, levels=('FAIL',)),
              [(f.rule, f.message[:200]) for f in findings_naming(baseline, stems, levels=('FAIL',))])
        for f in findings_naming(baseline, stems, levels=('WARN',)):
            print(f'e2e: already on the copy before the return: [WARN] {f.rule} {f.project} | {f.message[:240]}')
        before_files = {}
        data = load(exps[REAL_PROJECTS[0]] / 'segmentations' / 'segmentations.json')
        returned_keys = {}
        for name in REAL_PROJECTS:
            keys = load(exps[name] / 'segmentations' / 'segmentations.json')
            by_stem = {os.path.splitext(k)[0]: k for k in keys}
            returned_keys[name] = sorted(by_stem[s] for s in EXPECTED_REAL[name] if s in by_stem)
        lab_fn, rej_fn = returned_keys[REAL_PROJECTS[0]][0], returned_keys[REAL_PROJECTS[0]][1]

        def blank_hand(fn):
            return next(m['id'] for m in data[fn]['masks'] if m.get('status') == 'pending'
                        and not (m.get('species') or '').strip() and m.get('source_type') in
                        ('manual_click', 'manual_box', 'merged'))
        lab, rej = {'frame': lab_fn, 'mask': blank_hand(lab_fn)}, {'frame': rej_fn, 'mask': blank_hand(rej_fn)}
        exp0 = exps[REAL_PROJECTS[0]]
        before_files.update(
            label=(exp0 / 'all_labels' / (os.path.splitext(lab_fn)[0] + '.txt')).read_bytes(),
            lab_batch=csv_rows(exp0 / 'export_manifest.csv', os.path.splitext(lab_fn)[0])[0]['batch_name'],
            rej_batch=csv_rows(exp0 / 'export_manifest.csv', os.path.splitext(rej_fn)[0])[0]['batch_name'])
        r = cli(args + ['--apply'], path=real_path)
        print('---- apply on the copies ----')
        print(r.stdout + r.stderr)
        print('---- end of apply ----')
        check('e2e: --apply on the copies exits 0', r.returncode == 0, r.stderr[-400:])
        for name in REAL_PROJECTS:
            d = load(exps[name] / 'segmentations' / 'segmentations.json')
            got = sorted(k for k, v in d.items() if v.get(RETURN_KEY))
            check(f'e2e: {name} has exactly the expected frames sent back', got == returned_keys[name],
                  sorted(set(got) ^ set(returned_keys[name])))
        second_keys = e2e_second_run(root, exps, returned_keys, real_path)
        all_returned = {n: sorted(returned_keys[n] + second_keys[n]) for n in REAL_PROJECTS}
        before_files['return_at'] = json.loads(
            (exp0 / LOG_NAME).read_text(encoding='utf-8').splitlines()[-1])['at']
        waiting = run_checker(cfg)
        check('e2e: while the frames wait, no FAIL or WARN names them', not findings_naming(waiting, stems),
              [(f.rule, f.level, f.message[:200]) for f in findings_naming(waiting, stems)])
        q = run_child({'scenario': 'queue', 'export_dir': str(exps[REAL_PROJECTS[1]]),
                       'frames': all_returned[REAL_PROJECTS[1]]},
                      step4test_env(exps[REAL_PROJECTS[1]], REAL_PROJECTS[1][len('run_'):], registry), workdir)
        print(f'e2e: {REAL_PROJECTS[1]} queue holds {len(q["queue"])} frames; '
              f'"Frames with a mask missing its species" holds {len(q["unlabeled"])}; '
              f'batch of each returned frame: {q["batch_of"]}')
        check(f'e2e: {REAL_PROJECTS[1]}: the 9 returned frames lead the queue (all in the first batch)',
              q['queue'][:len(all_returned[REAL_PROJECTS[1]])] == all_returned[REAL_PROJECTS[1]]
              and set(q['batch_of'].values()) == {1}, q['batch_of'])
        check(f'e2e: {REAL_PROJECTS[1]}: "Frames with a mask missing its species" is exactly the returned '
              f'frames holding an outline with no species',
              sorted(q['unlabeled']) == unlabelled_among(exps[REAL_PROJECTS[1]], all_returned[REAL_PROJECTS[1]])
              and set(returned_keys[REAL_PROJECTS[1]]) <= set(q['unlabeled']), q['unlabeled'])
        time.sleep(EXPORT_SECOND)
        res = run_child({'scenario': 'roundtrip', 'export_dir': str(exp0), 'label': lab, 'reject': rej,
                         'outside': returned_keys[REAL_PROJECTS[0]][2]},
                        step4test_env(exp0, REAL_PROJECTS[0][len('run_'):], registry), workdir)
        print(f'e2e: {REAL_PROJECTS[0]} queue holds {len(res["queue"])} frames; '
              f'"Frames with a mask missing its species" holds {len(res["unlabeled"])}')
        check(f'e2e: {REAL_PROJECTS[0]}: the queue is exactly the 24 returned frames (19 and 5), in name order',
              res['queue'] == all_returned[REAL_PROJECTS[0]], len(res['queue']))
        check_roundtrip_result('e2e', res, exp0, registry, returned_keys[REAL_PROJECTS[0]], lab, rej,
                               before_files, REAL_PROJECTS[0][len('run_'):])
        after = run_checker(cfg)
        named = findings_naming(after, stems, levels=('FAIL',))
        check('e2e: after the round trip no FAIL names any of the 33 frames', not named,
              [(f.rule, f.message[:300]) for f in named])
        c = after.counts()
        print(f'e2e: checker on the copies after the round trip: {c}')
        for f in after.findings:
            if f.level in ('FAIL', 'WARN', 'INFO') and f.project in {n[len('run_'):] for n in REAL_PROJECTS}:
                print(f'  [{f.level}] {f.rule} {f.project} | {f.message[:240]}')
    finally:
        fp_after = real_fingerprint()
        changed = sorted(k for k in set(fp_before) | set(fp_after) if fp_before.get(k) != fp_after.get(k))
        check('e2e: the real projects and the live registry did not change', not changed, changed[:10])
        shutil.rmtree(workdir, ignore_errors=True)
        check('e2e: the work folder is deleted', not workdir.exists(), workdir)


# ---------------------------------------------------------------- main
def main(argv):
    """Run every group (and the opt-in real-copy end to end); exit nonzero on any failure."""
    if len(argv) == 3 and argv[1] == '--annotator-child':
        return child_main(argv[2])
    groups = (test_dry_run_writes_nothing, test_apply_changes_only_named, test_backup_and_log,
              test_rerun_makes_no_change, test_skips_scrapped_and_unexported, test_select_every_mask_kind,
              test_two_projects, test_mode_and_layout_kept, test_large_file, test_refusals,
              test_changed_mid_run, test_ss_parsing, test_crash_paths, test_importer_lock_gap,
              test_importer_lock_deadline,
              test_qualified_frames, test_two_runs_in_order,
              test_returned_frames_lead_the_queue, test_annotator_roundtrip_fixture)
    for fn in groups:
        group(fn)
    if len(argv) == 3 and argv[1] == '--e2e-real':
        def test_e2e_real_copies():
            """The opt-in end to end on copies of AT's two projects."""
            e2e_real(argv[2])
        group(test_e2e_real_copies)
    print(f'\n{len(PASSES)} passed, {len(FAILS)} failed')
    for name in FAILS:
        print(f'  FAILED: {name}')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
