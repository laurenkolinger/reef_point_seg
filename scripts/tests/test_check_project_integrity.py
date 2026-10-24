"""Tests for check_project_integrity.py (the read-only VEGA project checker).

Every fixture lives under a fresh tempfile.TemporaryDirectory(); nothing here
reads or writes the real inprocess/ tree, archive/, or /mnt/tear.

Covers: a clean fixture project passing every rule; one fixture per rule that
breaks it (broken link, blank species, ledger drift, duplicate ledger row,
0600 file, stale .tmp, image_missing manifest row, bad YOLO line, class id out
of range, registry drift, composite links, vocabulary, stale backups, ...)
producing exactly the expected FAIL/WARN set; hostile names (spaces, quotes,
unicode); an empty project; missing files; malformed JSON; the CLI exit codes
and report files; a hash sweep proving the checker never changes any input; and
the integrity exceptions file (frames a ruling keeps out of the coverage matrix
read as INFO, a broken or stale exception is an R15 FAIL).

Run: /mnt/rip/vicarius_drive/vicarius/modules/reef_point_seg/github_repo/env/bin/python \
        scripts/tests/test_check_project_integrity.py
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, SCRIPTS)

import check_project_integrity as cic  # noqa: E402
from _reefreview.mask_registry import REGISTRY_FIELDS, build_registry_record  # noqa: E402

AST = timezone(timedelta(hours=-4))
NOW = datetime.now(AST).replace(microsecond=0)  # real clock, so fixture times never depend on the date
FRAME_A = 'TCRMP20201010_clip_BPT_T101.jpg'
FRAME_B = 'TCRMP20201011_clip_BPT_T102.jpg'
PID = '26-10-15_XX'
LEDGER_HEADER = ['basename', 'label', 'outcome', 'source', 'reviewer', 'at', 'project_id']
MANIFEST_BASE = ['basename', 'n_masks', 'label_file', 'image_file', 'outcome',
                 'preview_file', 'batch_name', 'image_path']
VOCAB = [('PA', 'Coral', 'Pseudodiploria strigosa'), ('SS', 'Coral', 'Siderastrea siderea'),
         ('AC', 'Coral', 'Acropora cervicornis')]
FAILS = []
PASSES = []


def check(name, cond, detail=''):
    """Record one assertion as PASS or FAIL and print it."""
    if cond:
        PASSES.append(name)
        print(f'PASS  {name}')
    else:
        FAILS.append(name)
        print(f'FAIL  {name}  {detail}')


def write(path, text):
    """Write text to path, creating parents."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        f.write(text)


def write_json(path, obj):
    """Write obj as JSON to path."""
    write(path, json.dumps(obj, indent=1))


def write_csv(path, header, rows):
    """Write rows (dicts) with header to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})


def mask(mid, species, status, source_type, sx, sy, review=False):
    """One mask dict with distinct geometry so each mask gets its own uid."""
    return {'id': mid, 'label': 'A', 'species': species, 'name': species,
            'category': 'Coral', 'source_x': sx, 'source_y': sy,
            'polygon_norm': [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]],
            'bbox': [sx, sy, sx + 10 + mid, sy + 10], 'area': 10, 'score': 0.9,
            'status': status, 'source_type': source_type, 'reviewer': 'XX',
            'review': review}


class Fixture:
    """A complete clean VEGA tree under a temp dir, with knobs for tests."""

    def __init__(self, base, pid=PID, frame_a=FRAME_A, frame_b=FRAME_B):
        self.base = base
        self.pid = pid
        self.fa, self.fb = frame_a, frame_b
        self.root = os.path.join(base, 'inprocess')
        self.archive = os.path.join(base, 'archive')
        self.vocab = os.path.join(base, 'vocab.csv')
        self.mirror = os.path.join(base, 'mirror_codes.csv')
        self.pipeline = os.path.join(base, 'pipeline.yaml')
        self.daily = os.path.join(base, 'daily')
        self.exceptions = os.path.join(base, 'integrity_exceptions.yaml')  # absent unless a test writes it
        self.scripts = os.path.join(base, 'const')
        self.proj = os.path.join(self.root, 'run_' + pid)
        self.exp = os.path.join(self.proj, 'step4test_combinedAnnotate')
        os.makedirs(self.archive)
        self.build()

    def cfg(self, **kw):
        """A Config pointing at this fixture."""
        args = dict(root=self.root, archive_dir=self.archive, vocab_path=self.vocab,
                    pipeline_config=self.pipeline, scripts_dir=SCRIPTS,
                    snapshot_dir=self.daily, manual_source_files=[self.const_file],
                    exceptions_path=self.exceptions, now=NOW)
        args.update(kw)
        return cic.Config(**args)

    def segs(self):
        """The two-frame segmentations dict."""
        a = {'image_path': 'raw/' + self.fa,
             'image_path_abs': os.path.join(self.exp, 'routed_input', 'raw', self.fa),
             'image_width': 64, 'image_height': 48, 'reviewed': True, 'exported': True,
             'masks': [mask(0, 'PA', 'accepted', 'manual_click', 100, 100),
                       mask(1, 'SS', 'accepted', 'manual_box', 300, 200),
                       mask(2, 'PA', 'rejected', 'auto', 500, 500)],
             'label_outcomes': {c: {'outcome': 'found_manual', 'reviewed': True,
                                    'source': 'step4test', 'reviewer': 'XX',
                                    'at': '2026-10-15T09:00:00-04:00'}
                                for c in ('PA', 'SS')}}
        b = {'image_path': 'raw/' + self.fb,
             'image_path_abs': os.path.join(self.exp, 'routed_input', 'raw', self.fb),
             'image_width': 64, 'image_height': 48, 'reviewed': True, 'exported': True,
             'masks': [],
             'label_outcomes': {c: {'outcome': 'not_found', 'reviewed': True,
                                    'source': 'step4test', 'reviewer': 'XX',
                                    'at': '2026-10-15T09:00:00-04:00'}
                                for c in ('PA', 'SS')}}
        return {self.fa: a, self.fb: b}

    def build(self):
        """Write the whole clean tree."""
        proj, exp = self.proj, self.exp
        step = lambda n, name, d: {'status': 'completed', 'name': name,  # noqa: E731
                                   'dir': os.path.join(proj, d)}
        steps = {'1': step(1, 'Make All Points', 'step1_makeAllPoints'),
                 '2': step(2, 'Recode Species', 'step2_recodeSpecies'),
                 '3': step(3, 'Choose Images', 'step3_chooseImages'),
                 '4': step(4, 'Place Points', 'step4test_combinedAnnotate')}
        steps['3']['config'] = {'target_species': 'PA, SS'}
        steps['1']['outputs'] = {'all_points': os.path.join(proj, 'step1_makeAllPoints', 'all_points.csv')}
        write_json(os.path.join(proj, 'project.json'), {
            'id': self.pid, 'name': 'Fixture project', 'created_at': '2026-10-15T08:00:00',
            'updated_at': '2026-10-15T09:00:00-04:00', 'project_dir': proj,
            'current_step': 4, 'steps': steps,
            'access_log': [{'ts': '2026-10-15T08:00:00-04:00', 'initials': 'XX',
                            'action': 'create', 'purpose': 'fixture'}]})
        write(os.path.join(proj, 'step1_makeAllPoints', 'all_points.csv'), 'a,b\n1,2\n')
        write(os.path.join(proj, 'step1_makeAllPoints', 'tcrmp_species_codes.csv'), 'code,category,name\nPA,Coral,x\n')
        write(os.path.join(proj, 'step2_recodeSpecies', 'all_points_recoded.csv'), 'a,b\n1,2\n')
        write(os.path.join(proj, 'step2_recodeSpecies', 'master_codes_recoded.csv'), 'code,category,name\nPA,Coral,x\n')
        write_json(os.path.join(proj, 'step2_recodeSpecies', 'remap_log.json'), {'remaps': []})
        write_csv(os.path.join(proj, 'step3_chooseImages', 'selected_frames.csv'),
                  ['basename', 'year'],
                  [{'basename': os.path.splitext(f)[0], 'year': 2020} for f in (self.fa, self.fb)])
        # Prompts file written after selection so the staleness check is quiet.
        write_json(os.path.join(exp, 'routed_input', 'ids', 'sam_click_prompts.json'), {})
        future = NOW.timestamp() - 3600
        os.utime(os.path.join(exp, 'routed_input', 'ids', 'sam_click_prompts.json'), (future, future))
        past = NOW.timestamp() - 7200
        os.utime(os.path.join(proj, 'step3_chooseImages', 'selected_frames.csv'), (past, past))
        for f in (self.fa, self.fb):
            write(os.path.join(exp, 'routed_input', 'raw', f), 'img')
            link = os.path.join(exp, 'all_images', f)
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(os.path.join(exp, 'routed_input', 'raw', f), link)
        coords = '0.1 0.1 0.5 0.1 0.5 0.5'
        write(os.path.join(exp, 'all_labels', os.path.splitext(self.fa)[0] + '.txt'),
              f'0 {coords}\n1 {coords}\n')
        write(os.path.join(exp, 'all_labels', os.path.splitext(self.fb)[0] + '.txt'), '')
        write_json(os.path.join(exp, 'class_map.json'), {'PA': 0, 'SS': 1})
        write(os.path.join(exp, 'data.yaml'),
              f'path: {exp}\ntrain: all_images\nval: all_images\nnc: 2\nnames:\n  0: PA\n  1: SS\n')
        stem_a, stem_b = (os.path.splitext(f)[0] for f in (self.fa, self.fb))
        rows = [
            {'basename': stem_a, 'n_masks': 2, 'label_file': stem_a + '.txt', 'image_file': self.fa,
             'outcome': 'exported_with_masks', 'preview_file': '', 'batch_name': 'b1',
             'image_path': os.path.join(exp, 'routed_input', 'raw', self.fa), 'PA': 1, 'SS': 1},
            {'basename': stem_b, 'n_masks': 0, 'label_file': stem_b + '.txt', 'image_file': self.fb,
             'outcome': 'exported_empty', 'preview_file': '', 'batch_name': 'b1',
             'image_path': os.path.join(exp, 'routed_input', 'raw', self.fb), 'PA': 0, 'SS': 0}]
        write_csv(os.path.join(exp, 'export_manifest.csv'), MANIFEST_BASE + ['PA', 'SS'], rows)
        ledger = []
        for stem, outcome in ((stem_a, 'found_manual'), (stem_b, 'not_found')):
            for code in ('PA', 'SS'):
                ledger.append({'basename': stem, 'label': code, 'outcome': outcome,
                               'source': 'step4test', 'reviewer': 'XX',
                               'at': '2026-10-15T09:00:00-04:00', 'project_id': self.pid})
        write_csv(os.path.join(exp, 'label_provenance.csv'), LEDGER_HEADER, ledger)
        write_json(os.path.join(exp, 'segmentations', 'segmentations.json'), self.segs())
        write_json(os.path.join(exp, 'prior_screening.json'),
                   {'version': 1, 'project_id': self.pid, 'frames': {}})
        self.write_registry()
        self.write_composite()
        write_csv(self.vocab, ['code', 'category', 'name'],
                  [dict(code=c, category=g, name=n) for c, g, n in VOCAB])
        shutil.copy(self.vocab, self.mirror)
        write(self.pipeline, f'paths:\n  duplicate_master_codes: {self.mirror}\n')
        stamp = (NOW - timedelta(hours=9)).strftime('%Y%m%d-%H%M%S')
        os.makedirs(os.path.join(self.daily, stamp))
        self.const_file = os.path.join(self.scripts, 'provenance.py')
        write(self.const_file, "_MANUAL_SOURCES = {'manual_box', 'manual_click', 'merged'}\n")

    def write_registry(self, mutate=None):
        """Write the registry manifest with one row per mask of both frames."""
        recs = []
        for fn, seg in self.segs().items():
            for m in seg['masks']:
                rec = build_registry_record(fn, m, self.pid, self.pid, reviewer='XX')
                rec['created_at'] = rec['updated_at'] = '2026-10-15 09:00:00'
                recs.append(rec)
        if mutate:
            recs = mutate(recs)
        write_csv(os.path.join(self.root, '_mask_registry', 'manifest.csv'), REGISTRY_FIELDS, recs)

    def write_composite(self):
        """A one-image composite training set whose source project is live."""
        comp = os.path.join(self.root, '_composite', 'comp1')
        stem = os.path.splitext(self.fa)[0]
        os.makedirs(os.path.join(comp, 'all_images'), exist_ok=True)
        os.symlink(os.path.join(self.exp, 'routed_input', 'raw', self.fa),
                   os.path.join(comp, 'all_images', self.fa))
        write(os.path.join(comp, 'all_labels', stem + '.txt'), '0 0.1 0.1 0.5 0.1 0.5 0.5\n')
        write_json(os.path.join(comp, 'class_map.json'), {'PA': 0})
        write_csv(os.path.join(comp, 'composite_manifest.csv'),
                  ['basename', 'label', 'project_id', 'uid', 'status', 'source_type',
                   'reviewer', 'chosen_by_preference', 'at'],
                  [{'basename': stem, 'label': 'PA', 'project_id': self.pid, 'uid': 'u1',
                    'status': 'accepted', 'source_type': 'manual_click', 'reviewer': 'XX',
                    'at': '2026-10-15T09:00:00-04:00'}])

    def write_exceptions(self, entries, version=1):
        """Write the integrity exceptions file as YAML."""
        write(self.exceptions, yaml.safe_dump({'version': version, 'exceptions': entries},
                                              sort_keys=False, allow_unicode=True))

    def stem(self, frame):
        """File stem of one of the fixture frames."""
        return os.path.splitext(frame)[0]

    def drop_records_of(self, frame):
        """Remove a frame's ledger rows and registry rows, as the 2nd July test left 20 frames."""
        stem = self.stem(frame)
        self.edit_csv(os.path.join(self.exp, 'label_provenance.csv'),
                      lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['basename'] != stem]))
        self.write_registry(lambda recs: [r for r in recs if r['basename'] != stem])

    def rewrite_segs(self, fn):
        """Load, mutate in place with fn(segs), and write back segmentations.json."""
        path = os.path.join(self.exp, 'segmentations', 'segmentations.json')
        with open(path, encoding='utf-8') as f:
            segs = json.load(f)
        fn(segs)
        write_json(path, segs)

    def edit_csv(self, path, fn):
        """Load a CSV, mutate rows with fn(rows), write back with the same header."""
        before = os.stat(path)
        with open(path, newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            header, rows = rd.fieldnames, list(rd)
        fn(rows)
        write_csv(path, header, rows)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))  # an edit keeps the file's age


def run(fx, **kw):
    """Run the checker against a fixture and return its report."""
    return cic.run_checks(fx.cfg(**kw))


def issues(report):
    """Set of (rule, level) pairs for every non-PASS finding."""
    return {(f.rule, f.level) for f in report.findings if f.level != 'PASS'}


def hash_tree(root):
    """{path: (sha256 or link target, mtime_ns, mode)} for every entry under root."""
    out = {}
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
            out[p] = (digest, st.st_mtime_ns, st.st_mode)
    return out


def case(name, mutate, expected, **kw):
    """Build a clean fixture, apply mutate(fx), assert the exact non-PASS set."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        mutate(fx)
        got = issues(run(fx, **kw))
        check(name, got == set(expected), f'expected {sorted(expected)} got {sorted(got)}')


def clean_cases():
    """The clean fixture passes everything; each rule reports a PASS."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        rep = run(fx)
        check('clean: no WARN or FAIL', issues(rep) == set(), str(sorted(issues(rep))))
        rules = {f.rule for f in rep.findings if f.level == 'PASS'}
        want = {f'R{n:02d}' for n in range(1, 16)}
        check('clean: every rule reports a PASS', want <= rules, str(sorted(want - rules)))
        check('clean: exit code 0', cic.exit_code(rep) == 0)
        f0 = rep.findings[0]
        check('clean: finding carries rule, level, project, path, message',
              all(hasattr(f0, a) for a in ('rule', 'level', 'project', 'path', 'message')))


def rule_cases():
    """One fixture per rule that breaks it."""
    exp_of = lambda fx, *p: os.path.join(fx.exp, *p)  # noqa: E731

    # R01 project.json
    case('R01 malformed project.json', lambda fx: write(os.path.join(fx.proj, 'project.json'), '{"id": '),
         {('R01', 'FAIL')})

    def id_mismatch(fx):
        p = os.path.join(fx.proj, 'project.json')
        d = json.load(open(p))
        d['id'] = 'other'
        d['project_dir'] = '/somewhere/else'
        write_json(p, d)
    case('R01 id and project_dir mismatch (ledger, registry and screening file then name another id)',
         id_mismatch, {('R01', 'FAIL'), ('R08', 'FAIL'), ('R09', 'FAIL'), ('R13', 'FAIL')})

    def bad_types(fx):
        p = os.path.join(fx.proj, 'project.json')
        d = json.load(open(p))
        d['current_step'] = 'four'
        d['access_log'] = [{'ts': ' ', 'initials': 'XX', 'action': 'open'}]
        del d['name']
        write_json(p, d)
    case('R01 wrong types, missing key, blank ts', bad_types, {('R01', 'FAIL')})

    def missing_step_dir(fx):
        shutil.rmtree(os.path.join(fx.proj, 'step3_chooseImages'))
    case('R01 completed step folder missing', missing_step_dir,
         {('R01', 'FAIL'), ('R03', 'FAIL')})

    case('R01 project.json mode 0600 is a WARN',
         lambda fx: os.chmod(os.path.join(fx.proj, 'project.json'), 0o600), {('R01', 'WARN')})
    case('R01 segmentations.json mode 0600 is a WARN',
         lambda fx: os.chmod(exp_of(fx, 'segmentations', 'segmentations.json'), 0o600),
         {('R01', 'WARN')})

    # R02 steps 1 and 2
    def broken_step1_link(fx):
        t = os.path.join(fx.proj, 'step1_makeAllPoints', 'all_points.csv')
        os.remove(t)
        os.symlink('/nonexistent/all_points.csv', t)
    case('R02 broken Step 1 symlink', broken_step1_link, {('R02', 'FAIL')})
    case('R02 missing Step 2 output',
         lambda fx: os.remove(os.path.join(fx.proj, 'step2_recodeSpecies', 'remap_log.json')),
         {('R02', 'FAIL')})
    case('R02 Step 2 remap_log not JSON',
         lambda fx: write(os.path.join(fx.proj, 'step2_recodeSpecies', 'remap_log.json'), '{{'),
         {('R02', 'FAIL')})

    # R03 selection
    def dup_selected(fx):
        fx.edit_csv(os.path.join(fx.proj, 'step3_chooseImages', 'selected_frames.csv'),
                    lambda rows: rows.append(dict(rows[0])))
    case('R03 duplicate basename in selected_frames.csv', dup_selected, {('R03', 'FAIL')})

    def unknown_target(fx):
        p = os.path.join(fx.proj, 'project.json')
        d = json.load(open(p))
        d['steps']['3']['config']['target_species'] = 'PA, ZZZ'
        write_json(p, d)
    case('R03 target code not in vocabulary', unknown_target, {('R03', 'FAIL'), ('R08', 'WARN')})

    def stale_selection(fx):
        sel = os.path.join(fx.proj, 'step3_chooseImages', 'selected_frames.csv')
        t = NOW.timestamp() - 60
        os.utime(sel, (t, t))
    case('R03 selection newer than prompts while frames reviewed', stale_selection,
         {('R03', 'WARN')})

    # R04 segmentations
    def blank_species(fx):
        def f(s):
            s[fx.fa]['masks'][0]['species'] = ''
        fx.rewrite_segs(f)
    case('R04 accepted mask with blank species', blank_species,
         {('R04', 'FAIL'), ('R07', 'FAIL'), ('R08', 'WARN'), ('R09', 'FAIL')})

    def dup_mask_id(fx):
        def f(s):
            s[fx.fa]['masks'][1]['id'] = 0
        fx.rewrite_segs(f)
    case('R04 duplicate mask id in a frame', dup_mask_id, {('R04', 'FAIL')})

    def bad_value(fx):
        def f(s):
            s[fx.fb] = ['not', 'an', 'object']
        fx.rewrite_segs(f)
    case('R04 frame value not an object', bad_value,
         {('R04', 'FAIL'), ('R07', 'FAIL'), ('R08', 'FAIL')})

    def mask_missing_key(fx):
        def f(s):
            del s[fx.fa]['masks'][0]['status']
        fx.rewrite_segs(f)
    case('R04 mask missing status', mask_missing_key,
         {('R04', 'FAIL'), ('R07', 'FAIL'), ('R08', 'WARN'), ('R09', 'FAIL')})

    case('R04 malformed segmentations.json',
         lambda fx: write(exp_of(fx, 'segmentations', 'segmentations.json'), '{"a": '),
         {('R04', 'FAIL')})

    # R05 images and labels
    def broken_link(fx):
        link = exp_of(fx, 'all_images', fx.fa)
        os.remove(link)
        os.symlink('/nonexistent/' + fx.fa, link)
    case('R05 broken all_images link', broken_link, {('R05', 'FAIL')})

    case('R05 image without a label file',
         lambda fx: os.remove(exp_of(fx, 'all_labels', os.path.splitext(fx.fb)[0] + '.txt')),
         {('R05', 'FAIL')})

    def extra_image(fx):
        write(exp_of(fx, 'all_images', 'extra.jpg'), 'x')
        write(exp_of(fx, 'all_labels', 'extra.txt'), '')
    case('R05 image count differs from manifest rows', extra_image, {('R05', 'FAIL')})

    # R06 YOLO labels
    def odd_coords(fx):
        write(exp_of(fx, 'all_labels', os.path.splitext(fx.fa)[0] + '.txt'),
              '0 0.1 0.1 0.5 0.1 0.5\n1 0.1 0.1 0.5 0.1 0.5 0.5\n')
    case('R06 odd coordinate count', odd_coords, {('R06', 'FAIL')})

    def out_of_unit(fx):
        write(exp_of(fx, 'all_labels', os.path.splitext(fx.fa)[0] + '.txt'),
              '0 0.1 0.1 1.5 0.1 0.5 0.5\n1 0.1 0.1 0.5 0.1 0.5 0.5\n')
    case('R06 coordinate outside 0..1', out_of_unit, {('R06', 'FAIL')})

    def class_range(fx):
        write(exp_of(fx, 'all_labels', os.path.splitext(fx.fa)[0] + '.txt'),
              '0 0.1 0.1 0.5 0.1 0.5 0.5\n7 0.1 0.1 0.5 0.1 0.5 0.5\n')
    case('R06 class id out of range', class_range, {('R06', 'FAIL')})

    case('R06 class_map with a blank name',
         lambda fx: write_json(exp_of(fx, 'class_map.json'), {'PA': 0, '': 1}),
         {('R06', 'FAIL'), ('R11', 'FAIL')})
    case('R06 class_map with a gap in ids',
         lambda fx: write_json(exp_of(fx, 'class_map.json'), {'PA': 0, 'SS': 2}),
         {('R06', 'FAIL')})
    case('R06 data.yaml nc disagrees',
         lambda fx: write(exp_of(fx, 'data.yaml'),
                          f'path: {fx.exp}\nnc: 3\nnames:\n  0: PA\n  1: SS\n'),
         {('R06', 'FAIL')})
    case('R06 missing class_map.json', lambda fx: os.remove(exp_of(fx, 'class_map.json')),
         {('R06', 'FAIL')})

    # R07 export manifest
    def image_missing_row(fx):
        def f(rows):
            rows[1]['outcome'] = 'image_missing'
            rows[1]['label_file'] = rows[1]['image_file'] = ''
        fx.edit_csv(exp_of(fx, 'export_manifest.csv'), f)
    case('R07 image_missing manifest row', image_missing_row, {('R07', 'FAIL'), ('R05', 'FAIL')})

    def species_count(fx):
        fx.edit_csv(exp_of(fx, 'export_manifest.csv'), lambda rows: rows[0].update(PA='3'))
    case('R07 species count differs from accepted masks', species_count, {('R07', 'FAIL')})

    def frame_set(fx):
        fx.edit_csv(exp_of(fx, 'export_manifest.csv'), lambda rows: rows.pop())
    case('R07 manifest frame set differs from exported frames', frame_set,
         {('R07', 'FAIL'), ('R05', 'FAIL')})

    def n_masks(fx):
        fx.edit_csv(exp_of(fx, 'export_manifest.csv'), lambda rows: rows[0].update(n_masks='5'))
    case('R07 n_masks differs from label lines', n_masks, {('R07', 'FAIL')})

    # R08 ledger
    def drift(fx):
        fx.edit_csv(exp_of(fx, 'label_provenance.csv'), lambda rows: rows[0].update(outcome='found_ai'))
    case('R08 ledger outcome drifts from stored outcome', drift, {('R08', 'FAIL')})

    def dup_row(fx):
        fx.edit_csv(exp_of(fx, 'label_provenance.csv'), lambda rows: rows.append(dict(rows[0])))
    case('R08 duplicate ledger row', dup_row, {('R08', 'FAIL')})

    case('R08 ledger header wrong',
         lambda fx: write_csv(exp_of(fx, 'label_provenance.csv'), LEDGER_HEADER[:-1], []),
         {('R08', 'FAIL')})

    def bad_values(fx):
        def f(rows):
            rows[0].update(outcome='bogus', project_id='someone_else')
            rows[1].update(label='')
        fx.edit_csv(exp_of(fx, 'label_provenance.csv'), f)
    case('R08 bad outcome, project_id and blank label', bad_values, {('R08', 'FAIL')})

    def unledgered(fx):
        fx.edit_csv(exp_of(fx, 'label_provenance.csv'), lambda rows: rows.__delitem__(slice(2, 4)))
    case('R08 frame with outcomes but no ledger rows', unledgered, {('R08', 'FAIL')})

    def warn_compare(fx):
        def f(s):
            s[fx.fa]['masks'][0]['status'] = 'pending'
        fx.rewrite_segs(f)
    case('R08 computed outcomes differ from stored (WARN) with registry drift',
         warn_compare, {('R07', 'FAIL'), ('R08', 'WARN'), ('R09', 'FAIL')})

    # R08 scrapped frames and unconfirmed writes
    def scrap(fx, keep_rows):
        fx.rewrite_segs(lambda s: s[fx.fb].update(scrapped=True))
        stem = os.path.splitext(fx.fb)[0]
        os.remove(exp_of(fx, 'all_images', fx.fb))
        os.remove(exp_of(fx, 'all_labels', stem + '.txt'))

        def man(rows):
            rows[1].update(outcome='scrapped', label_file='', image_file='')
        fx.edit_csv(exp_of(fx, 'export_manifest.csv'), man)
        if not keep_rows:
            fx.edit_csv(exp_of(fx, 'label_provenance.csv'),
                        lambda rows: rows.__delitem__(slice(2, 4)))
    case('R08 scrapped frame with its ledger rows removed is clean', lambda fx: scrap(fx, False), set())
    case('R08 scrapped frame whose ledger rows remain is a WARN', lambda fx: scrap(fx, True),
         {('R08', 'WARN')})
    case('R08 frame still carrying ledger_pending is a WARN',
         lambda fx: fx.rewrite_segs(lambda s: s[fx.fa].update(ledger_pending=True)), {('R08', 'WARN')})
    case('R08 frame still carrying image_missing_at is a WARN',
         lambda fx: fx.rewrite_segs(lambda s: s[fx.fa].update(image_missing_at='2026-10-15T09:00:00-04:00')),
         {('R08', 'WARN')})

    # R09 registry
    def reg_species(fx):
        fx.write_registry(lambda recs: [dict(r, species='SS') if i == 0 else r
                                        for i, r in enumerate(recs)])
    case('R09 registry species differs', reg_species, {('R09', 'FAIL')})

    case('R09 registry row missing', lambda fx: fx.write_registry(lambda recs: recs[:-1]),
         {('R09', 'FAIL')})

    def reg_dup(fx):
        fx.write_registry(lambda recs: recs + [dict(recs[0])])
    case('R09 duplicate registry uid', reg_dup, {('R09', 'FAIL')})

    def reg_fixture_row(fx):
        def f(recs):
            r = dict(recs[0])
            r.update(uid='FIX-x1y1', project_id='e2e_abc123')
            return recs + [r]
        fx.write_registry(f)
    case('R09 test-fixture project id in registry', reg_fixture_row, {('R09', 'FAIL')})

    def reg_archived(fx):
        os.makedirs(os.path.join(fx.archive, 'run_old_one'))

        def f(recs):
            r = dict(recs[0])
            r.update(uid='OLD-x1y1', project_id='old_one')
            return recs + [r]
        fx.write_registry(f)
    case('R09 archived project id is a WARN (plus the archive listing)', reg_archived,
         {('R09', 'WARN'), ('R14', 'WARN')})

    case('R09 registry header wrong',
         lambda fx: write_csv(os.path.join(fx.root, '_mask_registry', 'manifest.csv'),
                              REGISTRY_FIELDS[:-1], []),
         {('R09', 'FAIL')})

    # R10 composites
    def comp_link(fx):
        link = os.path.join(fx.root, '_composite', 'comp1', 'all_images', fx.fa)
        os.remove(link)
        os.symlink('/nonexistent/x.jpg', link)
    case('R10 broken composite link', comp_link, {('R10', 'FAIL')})

    def comp_lines(fx):
        write(os.path.join(fx.root, '_composite', 'comp1', 'all_labels',
                           os.path.splitext(fx.fa)[0] + '.txt'),
              '0 0.1 0.1 0.5 0.1 0.5 0.5\n0 0.1 0.1 0.5 0.1 0.5 0.5\n')
    case('R10 label lines differ from manifest rows', comp_lines, {('R10', 'FAIL')})

    def comp_project(fx):
        fx.edit_csv(os.path.join(fx.root, '_composite', 'comp1', 'composite_manifest.csv'),
                    lambda rows: rows[0].update(project_id='ghost'))
    case('R10 composite source project not live', comp_project, {('R10', 'FAIL')})

    # R11 vocabulary
    def dup_code(fx):
        fx.edit_csv(fx.vocab, lambda rows: rows.append(dict(rows[0])))
    case('R11 duplicate vocabulary code (and a mirror that differs)', dup_code, {('R11', 'FAIL')})

    case('R11 mirror differs', lambda fx: write(fx.mirror, 'code,category,name\nPA,Coral,x\n'),
         {('R11', 'FAIL')})
    case('R11 mirror missing', lambda fx: os.remove(fx.mirror), {('R11', 'FAIL')})

    def unknown_species(fx):
        def f(s):
            s[fx.fa]['masks'][2]['species'] = 'ZZZ'
        fx.rewrite_segs(f)
    case('R11 species used but not in vocabulary', unknown_species,
         {('R11', 'FAIL'), ('R09', 'FAIL')})

    # R12 code constants
    case('R12 manual-source list without merged is a WARN',
         lambda fx: write(fx.const_file, "_MANUAL_SOURCES = {'manual_box', 'manual_click'}\n"),
         {('R12', 'WARN')})

    # R13 strays
    def stale_tmp(fx):
        p = exp_of(fx, 'label_provenance.csv.tmp')
        write(p, 'x')
        t = NOW.timestamp() - 7200
        os.utime(p, (t, t))
    case('R13 .tmp older than an hour', stale_tmp, {('R13', 'FAIL')})

    def fresh_tmp(fx):
        p = exp_of(fx, 'label_provenance.csv.tmp')
        write(p, 'x')
        t = NOW.timestamp() - 60
        os.utime(p, (t, t))
    case('R13 .tmp younger than an hour is fine', fresh_tmp, set())

    def conflict(fx):
        write_json(exp_of(fx, 'segmentations', 'segmentations.conflict-20261015-090000.json'),
                   {fx.fa: [{'at': 'x', 'content': {}}]})
    case('R13 conflict file is a WARN', conflict, {('R13', 'WARN')})

    case('R13 prior_screening wrong version',
         lambda fx: write_json(exp_of(fx, 'prior_screening.json'),
                               {'version': 2, 'project_id': fx.pid, 'frames': {}}),
         {('R13', 'FAIL')})
    case('R13 prior_screening names another project',
         lambda fx: write_json(exp_of(fx, 'prior_screening.json'),
                               {'version': 1, 'project_id': 'other', 'frames': {}}),
         {('R13', 'FAIL')})

    # R14 backups
    def old_snapshot(fx):
        shutil.rmtree(fx.daily)
        stamp = (NOW - timedelta(hours=40)).strftime('%Y%m%d-%H%M%S')
        os.makedirs(os.path.join(fx.daily, stamp))
    case('R14 newest snapshot older than 26 hours', old_snapshot, {('R14', 'FAIL')})
    case('R14 no snapshot folder', lambda fx: shutil.rmtree(fx.daily), {('R14', 'FAIL')})
    case('R14 archive runs listed as WARN',
         lambda fx: os.makedirs(os.path.join(fx.archive, 'run_old_one')), {('R14', 'WARN')})


def structure_cases():
    """Empty, partial and hostile projects."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        empty = os.path.join(fx.root, 'run_empty')
        write_json(os.path.join(empty, 'project.json'), {
            'id': 'empty', 'name': 'e', 'created_at': 'x', 'updated_at': 'x',
            'project_dir': empty, 'current_step': 1, 'steps': {}, 'access_log': []})
        rep = run(fx)
        mine = [f for f in rep.findings if f.project == 'empty' and f.level != 'PASS']
        check('empty project (valid project.json, no steps) has no WARN or FAIL', not mine,
              str([(f.rule, f.level, f.message) for f in mine]))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        tp = os.path.join(fx.root, 'run_train1')
        steps = {str(n): {'status': 'skipped', 'name': 'x', 'dir': os.path.join(tp, d)}
                 for n, d in enumerate(('step1_makeAllPoints', 'step2_recodeSpecies',
                                        'step3_chooseImages'), 1)}
        steps['6'] = {'status': 'completed', 'name': 'Train', 'dir': os.path.join(tp, 'step6_trainModel')}
        write_json(os.path.join(tp, 'project.json'), {
            'id': 'train1', 'name': 't', 'created_at': 'x', 'updated_at': 'x', 'project_dir': tp,
            'current_step': 6, 'steps': steps, 'access_log': [], 'project_type': 'composite_training'})
        for d in ('step1_makeAllPoints', 'step2_recodeSpecies', 'step3_chooseImages', 'step6_trainModel'):
            os.makedirs(os.path.join(tp, d))
        os.symlink(os.path.join(fx.root, '_composite', 'comp1'), os.path.join(tp, 'step4test_combinedAnnotate'))
        rep = run(fx)
        mine = [f for f in rep.findings if f.project == 'train1' and f.level != 'PASS']
        check('training project with skipped steps and a symlinked composite has no WARN or FAIL',
              not mine, str([(f.rule, f.level, f.message) for f in mine]))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        write(fx.const_file, '"""Docs: manual_box/manual_click prompts, merged elsewhere."""\nX = 1\n')
        check('R12 prose that mentions manual_click is not a list', issues(run(fx)) == set())

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)

        def blanks(rows):
            rows[0]['label'] = ''
            rows[1]['label'] = ' '
        fx.edit_csv(os.path.join(fx.exp, 'label_provenance.csv'), blanks)
        msgs = [f.message for f in run(fx).findings if f.rule == 'R08' and f.level == 'FAIL']
        check('R08 counts blank ledger labels by row', any(m.startswith('2 rows with a blank label') for m in msgs),
              str(msgs))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        os.makedirs(os.path.join(fx.root, 'run_nothing'))
        rep = run(fx)
        r01 = [f for f in rep.findings if f.project == 'nothing' and f.rule == 'R01'
               and f.level == 'FAIL']
        check('folder with no project.json is an R01 FAIL', bool(r01))
        check('exit code 1 when any FAIL', cic.exit_code(rep) == 1)

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp, pid="odd name 'q' é中", frame_a="TCRMP20201010_clip_BPT_T10 1 é.jpg",
                     frame_b="TCRMP20201011_clip_BPT_T1'02.jpg")
        rep = run(fx)
        check('hostile names (spaces, quotes, unicode) pass clean', issues(rep) == set(),
              str(sorted(issues(rep))))
        fx.rewrite_segs(lambda s: s[fx.fa].update(masks=[]))
        rep = run(fx)
        check('hostile names: a real fault is still found', ('R07', 'FAIL') in issues(rep))
        txt = cic.render_text(rep, verbose=True)
        check('hostile names: text report is one line per finding',
              all('\n' not in f.message for f in rep.findings) and isinstance(txt, str))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        shutil.rmtree(os.path.join(fx.proj, 'step4test_combinedAnnotate', 'routed_input'))
        os.remove(os.path.join(fx.exp, 'segmentations', 'segmentations.json'))
        got = issues(run(fx))
        check('segmentations.json deleted under an exported project is a FAIL',
              ('R04', 'FAIL') in got, str(sorted(got)))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        write(os.path.join(fx.exp, 'segmentations', 'segmentations.json'), '\x00\xff not json')
        got = issues(run(fx))
        check('binary garbage in segmentations.json is a FAIL, not a crash', ('R04', 'FAIL') in got)

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        write_csv(os.path.join(fx.exp, 'label_provenance.csv'), LEDGER_HEADER, [])
        got = issues(run(fx))
        check('empty ledger over reviewed frames is a FAIL', ('R08', 'FAIL') in got)

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        big = os.path.join(fx.exp, 'all_labels', os.path.splitext(fx.fa)[0] + '.txt')
        with open(big, 'w') as f:
            f.write('0 ' + ' '.join(['0.5'] * 400000) + '\n1 0.1 0.1 0.5 0.1 0.5 0.5\n')
        got = issues(run(fx))
        check('a very long label line is handled', ('R06', 'FAIL') not in got, str(sorted(got)))


def read_only_cases():
    """The checker changes nothing it reads."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        os.chmod(os.path.join(fx.exp, 'segmentations', 'segmentations.json'), 0o600)
        write(os.path.join(fx.exp, 'a.tmp'), 'x')
        os.utime(os.path.join(fx.exp, 'a.tmp'), (1, 1))
        write(os.path.join(fx.proj, 'step2_recodeSpecies', 'remap_log.json'), '{{')
        before = hash_tree(tmp)
        run(fx)
        run(fx, manual_source_files=[])
        after = hash_tree(tmp)
        check('hash sweep: no file, link, mtime or mode changed', before == after,
              str([p for p in set(before) | set(after) if before.get(p) != after.get(p)][:5]))
        check('hash sweep: no file created or removed', set(before) == set(after))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        before = hash_tree(tmp)
        report_dir = os.path.join(tmp, 'reports')
        rc = cic.main(['--root', fx.root, '--archive-dir', fx.archive, '--vocab', fx.vocab,
                       '--pipeline-config', fx.pipeline, '--snapshot-dir', fx.daily,
                       '--report-dir', report_dir, '--manual-source-file', fx.const_file,
                       '--exceptions', fx.exceptions, '--now', NOW.isoformat()])
        after = hash_tree(tmp)
        new = sorted(set(after) - set(before))
        check('CLI on clean fixture exits 0', rc == 0, str(rc))
        only_reports = all(p == report_dir or p.startswith(report_dir + os.sep) for p in new)
        check('CLI writes only inside --report-dir', only_reports, str(new[:5]))
        names = sorted(os.listdir(report_dir)) if os.path.isdir(report_dir) else []
        check('report dir holds <stamp>.txt and <stamp>.json',
              [os.path.splitext(n)[1] for n in names] == ['.json', '.txt']
              and os.path.splitext(names[0])[0] == os.path.splitext(names[1])[0] == NOW.strftime('%Y%m%d-%H%M%S'),
              str(names))
        check('input files untouched by CLI run',
              all(before[p] == after[p] for p in before))


def cli_cases():
    """Exit codes and output formats through a real subprocess."""
    exe = sys.executable
    script = os.path.join(SCRIPTS, 'check_project_integrity.py')
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        base = [exe, script, '--root', fx.root, '--archive-dir', fx.archive, '--vocab', fx.vocab,
                '--pipeline-config', fx.pipeline, '--snapshot-dir', fx.daily,
                '--report-dir', os.path.join(tmp, 'rep'), '--manual-source-file', fx.const_file,
                '--exceptions', fx.exceptions, '--now', NOW.isoformat()]
        r = subprocess.run(base + ['--json'], capture_output=True, text=True)
        try:
            doc = json.loads(r.stdout)
            ok = isinstance(doc.get('findings'), list) and 'summary' in doc
        except ValueError:
            ok = False
        check('--json prints one JSON document on stdout', ok and r.returncode == 0,
              r.stdout[:200] + r.stderr[-200:])
        check('Step checkpoints go to stderr', 'Step 1 complete' in r.stderr)
        r = subprocess.run(base, capture_output=True, text=True)
        check('plain text mode shows the summary line', 'FAIL' in r.stdout and 'PASS' in r.stdout
              and r.returncode == 0, r.stdout[-300:])
        os.remove(os.path.join(fx.exp, 'class_map.json'))
        r = subprocess.run(base, capture_output=True, text=True)
        check('exit 1 when any FAIL', r.returncode == 1, str(r.returncode))
        r = subprocess.run([exe, script, '--root', os.path.join(tmp, 'nope'), '--report-dir',
                            os.path.join(tmp, 'rep2')], capture_output=True, text=True)
        check('exit 2 when --root does not exist', r.returncode == 2, str(r.returncode))
        r = subprocess.run([exe, script, '--bogus-flag'], capture_output=True, text=True)
        check('exit 2 on a usage error', r.returncode == 2, str(r.returncode))
        blocker = os.path.join(tmp, 'blocker')
        write(blocker, 'x')
        r = subprocess.run(base[:base.index('--report-dir')] + ['--report-dir',
                           os.path.join(blocker, 'sub')], capture_output=True, text=True)
        check('exit 2 when the report dir cannot be created', r.returncode == 2, str(r.returncode))


RULING = 'Finished frames with outcomes but no ledger rows stay out of the coverage matrix.'
RULING_DATE = '2026-09-27'
RULING_BY = 'Lauren Olinger (LO)'


def entry(fx, stems, project_id=None, **over):
    """One exceptions entry for the fixture project; `over` replaces or adds keys."""
    e = {'project_id': fx.pid if project_id is None else project_id, 'ruling': RULING,
         'date': RULING_DATE, 'by': RULING_BY, 'frames': list(stems)}
    e.update(over)
    return e


def findings_of(report, rule, level, project=None):
    """Findings with this rule and level (and project label, when given)."""
    return [f for f in report.findings if f.rule == rule and f.level == level
            and (project is None or f.project == project)]


def excepted_case(name, mutate, expected):
    """Fixture whose frame A lost its ledger and registry rows, then mutate(fx); exact non-PASS set."""
    def both(fx):
        fx.drop_records_of(fx.fa)
        mutate(fx)
    case(name, both, expected)


def exceptions_cases():
    """Frames excluded by a ruling read as INFO; the ruling is scoped and enforced."""
    excepted_case('frame without ledger or registry rows FAILs when no ruling names it',
                  lambda fx: None, {('R08', 'FAIL'), ('R09', 'FAIL')})
    excepted_case('frame named by a ruling reads as INFO, not FAIL',
                  lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])]),
                  {('R08', 'INFO'), ('R09', 'INFO')})

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        fx.drop_records_of(fx.fa)
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
        with open(fx.exceptions, 'rb') as f:
            before = hashlib.sha256(f.read()).hexdigest()
        rep = run(fx)
        with open(fx.exceptions, 'rb') as f:
            after = hashlib.sha256(f.read()).hexdigest()
        infos = findings_of(rep, 'R08', 'INFO') + findings_of(rep, 'R09', 'INFO')
        check('excepted findings cite the ruling text, date and who',
              len(infos) == 2 and all('excepted by ruling' in f.message and RULING in f.message
                                      and RULING_DATE in f.message and RULING_BY in f.message
                                      for f in infos), str([f.message for f in infos]))
        check('excepted findings name the frame and its masks',
              fx.stem(fx.fa) in infos[0].message and '3 exported masks' in infos[1].message,
              str([f.message for f in infos]))
        check('only excepted findings leave exit code 0', cic.exit_code(rep) == 0)
        check('a rule with only INFO findings still reports a PASS',
              bool(findings_of(rep, 'R08', 'PASS', fx.pid)) and bool(findings_of(rep, 'R09', 'PASS', fx.pid)))
        check('counts carry INFO', rep.counts().get('INFO') == 2, str(rep.counts()))
        txt = cic.render_text(rep)
        check('text report shows INFO lines and an INFO count',
              '[INFO] R08' in txt and ' INFO, ' in txt, txt[-400:])
        check('the exceptions file is read only', before == after)

    excepted_case('a ruling covers only its named frames: another unledgered frame still FAILs',
                  lambda fx: (fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])]),
                              fx.edit_csv(os.path.join(fx.exp, 'label_provenance.csv'),
                                          lambda rows: rows.__setitem__(slice(None), []))),
                  {('R08', 'INFO'), ('R09', 'INFO'), ('R08', 'FAIL')})

    def other_fault(fx):
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
        fx.rewrite_segs(lambda s: s[fx.fa]['masks'][0].update(species=''))
    excepted_case('a ruling never hides another fault on the same frame', other_fault,
                  {('R04', 'FAIL'), ('R07', 'FAIL'), ('R08', 'INFO'), ('R08', 'WARN'), ('R09', 'INFO')})

    case('an excluded frame that has ledger and registry rows breaks the ruling (FAIL)',
         lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])]),
         {('R08', 'FAIL'), ('R09', 'FAIL')})

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
        rep = run(fx)
        msgs = [f.message for f in findings_of(rep, 'R08', 'FAIL') + findings_of(rep, 'R09', 'FAIL')]
        check('a broken ruling says which frame and which ruling',
              len(msgs) == 2 and all(fx.stem(fx.fa) in m and RULING_DATE in m for m in msgs), str(msgs))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        twin = '26-10-15_YY'
        dst = os.path.join(fx.root, 'run_' + twin)
        shutil.copytree(fx.proj, dst, symlinks=True)
        pj = os.path.join(dst, 'project.json')
        d = json.load(open(pj))
        d.update(id=twin, project_dir=dst)
        write_json(pj, d)
        stem = fx.stem(fx.fa)
        for proj_dir in (fx.proj, dst):
            fx.edit_csv(os.path.join(proj_dir, 'step4test_combinedAnnotate', 'label_provenance.csv'),
                        lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['basename'] != stem]))
        fx.write_exceptions([entry(fx, [stem])])
        rep = run(fx)
        twin_fail = [f for f in findings_of(rep, 'R08', 'FAIL', twin) if 'no ledger rows' in f.message]
        check('a ruling is scoped to its project: the same frame in another project still FAILs',
              bool(findings_of(rep, 'R08', 'INFO', fx.pid)) and bool(twin_fail),
              str([(f.project, f.level, f.message) for f in rep.findings if f.rule == 'R08']))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        stem = fx.stem(fx.fa)
        copy_dir = os.path.join(fx.root, 'run_26-10-15_ZZ')  # a copied folder that kept the original id
        shutil.copytree(fx.proj, copy_dir, symlinks=True)
        for proj_dir in (fx.proj, copy_dir):
            fx.edit_csv(os.path.join(proj_dir, 'step4test_combinedAnnotate', 'label_provenance.csv'),
                        lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['basename'] != stem]))
        fx.write_exceptions([entry(fx, [stem])])
        rep = run(fx)
        copy_fail = [f for f in findings_of(rep, 'R08', 'FAIL', '26-10-15_ZZ') if 'no ledger rows' in f.message]
        check('a ruling matches the project folder, not a copy that kept its project.json id',
              bool(findings_of(rep, 'R08', 'INFO', fx.pid)) and not findings_of(rep, 'R08', 'INFO', '26-10-15_ZZ')
              and bool(copy_fail),
              str([(f.project, f.level, f.message[:80]) for f in rep.findings if f.rule == 'R08']))

    def partial_registry(fx):
        stem = fx.stem(fx.fa)
        fx.edit_csv(os.path.join(fx.exp, 'label_provenance.csv'),
                    lambda rows: rows.__setitem__(slice(None), [r for r in rows if r['basename'] != stem]))
        fx.write_registry(lambda recs: [r for i, r in enumerate(recs) if r['basename'] != stem or i == 0])
        fx.write_exceptions([entry(fx, [stem])])
    case('an excluded frame with some registry rows gives a FAIL and an INFO', partial_registry,
         {('R08', 'INFO'), ('R09', 'FAIL'), ('R09', 'INFO')})

    def no_ledger_file(fx):
        os.remove(os.path.join(fx.exp, 'label_provenance.csv'))
        fx.rewrite_segs(lambda s: s[fx.fb].pop('label_outcomes'))
        fx.write_registry(lambda recs: [r for r in recs if r['basename'] != fx.stem(fx.fa)])
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
    case('no ledger file and only excluded frames hold outcomes reads as INFO', no_ledger_file,
         {('R08', 'INFO'), ('R09', 'INFO')})

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        fx.drop_records_of(fx.fa)
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)], ruling='x' * 250 + ' end of ruling')])
        info = findings_of(run(fx), 'R08', 'INFO')
        check('a ruling of 264 characters is cited in full', bool(info) and 'end of ruling' in info[0].message,
              str([f.message[-80:] for f in info]))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp, pid="odd name 'q' é中", frame_a="TCRMP20201010_clip_BPT_T10 1 é.jpg")
        fx.drop_records_of(fx.fa)
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
        got = issues(run(fx))
        check('hostile project and frame names match their ruling',
              got == {('R08', 'INFO'), ('R09', 'INFO')}, str(sorted(got)))


def exceptions_file_cases():
    """R15: the exceptions file parses and names live projects and existing frames."""
    case('no exceptions file: R15 passes and nothing is excepted', lambda fx: None, set())
    case('an exception naming a project that does not exist FAILs',
         lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)], project_id='no_such_project')]),
         {('R15', 'FAIL')})
    case('an exception naming a frame the project does not hold FAILs',
         lambda fx: fx.write_exceptions([entry(fx, ['TCRMP20990101_clip_XXX_T999'])]),
         {('R15', 'FAIL')})
    excepted_case('valid frames of an entry still apply when another frame in it is stale',
                  lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa), 'TCRMP20990101_clip_XXX_T999'])]),
                  {('R08', 'INFO'), ('R09', 'INFO'), ('R15', 'FAIL')})
    excepted_case('an unquoted YAML date is accepted',
                  lambda fx: write(fx.exceptions,
                                   f'version: 1\nexceptions:\n  - project_id: {fx.pid}\n'
                                   f'    ruling: "{RULING}"\n    date: {RULING_DATE}\n'
                                   f'    by: {RULING_BY}\n    frames:\n      - {fx.stem(fx.fa)}\n'),
                  {('R08', 'INFO'), ('R09', 'INFO')})

    bad_files = (
        ('malformed YAML', 'exceptions: [\n  - {project_id: x,\n'),
        ('an empty file', ''),
        ('a top level that is a list', '- a\n- b\n'),
        ('binary garbage', '\x00\xff\xfe'),
        ('a python object tag', '!!python/object/apply:os.system ["echo hostile"]\n'),
    )
    for label, text in bad_files:
        excepted_case(f'{label} FAILs R15 and excepts nothing',
                      lambda fx, t=text: write(fx.exceptions, t),
                      {('R08', 'FAIL'), ('R09', 'FAIL'), ('R15', 'FAIL')})

    bad_entries = (
        ('version 2', lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])], version=2)),
        ('exceptions as a mapping', lambda fx: write(fx.exceptions, yaml.safe_dump(
            {'version': 1, 'exceptions': {'a': 1}}))),
        ('an entry that is not a mapping', lambda fx: fx.write_exceptions(['just text'])),
        ('a missing ruling', lambda fx: fx.write_exceptions(
            [{k: v for k, v in entry(fx, [fx.stem(fx.fa)]).items() if k != 'ruling'}])),
        ('a blank by', lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)], by='  ')])),
        ('a date that is not YYYY-MM-DD', lambda fx: fx.write_exceptions(
            [entry(fx, [fx.stem(fx.fa)], date='27/09/2026')])),
        ('a numeric project_id', lambda fx: fx.write_exceptions(
            [entry(fx, [fx.stem(fx.fa)], project_id=123)])),
        ('frames as a string', lambda fx: fx.write_exceptions([entry(fx, [], frames=fx.stem(fx.fa))])),
        ('an empty frames list', lambda fx: fx.write_exceptions([entry(fx, [])])),
        ('non-string frames', lambda fx: fx.write_exceptions([entry(fx, [1, None])])),
        ('a date without leading zeros', lambda fx: fx.write_exceptions(
            [entry(fx, [fx.stem(fx.fa)], date='2026-9-7')])),
        ('version true', lambda fx: write(fx.exceptions, yaml.safe_dump(
            {'version': True, 'exceptions': [entry(fx, [fx.stem(fx.fa)])]}))),
        ('version 1.0', lambda fx: write(fx.exceptions, yaml.safe_dump(
            {'version': 1.0, 'exceptions': [entry(fx, [fx.stem(fx.fa)])]}))),
        ('a 200 KB by', lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)], by='L' * 200000)])),
        ('a 200 KB project_id', lambda fx: fx.write_exceptions(
            [entry(fx, [fx.stem(fx.fa)], project_id='P' * 200000)])),
    )
    for label, mutate in bad_entries:
        excepted_case(f'{label} FAILs R15 and excepts nothing', mutate,
                      {('R08', 'FAIL'), ('R09', 'FAIL'), ('R15', 'FAIL')})

    excepted_case('a frame listed twice FAILs R15 (the frame is still excepted)',
                  lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)] * 2)]),
                  {('R08', 'INFO'), ('R09', 'INFO'), ('R15', 'FAIL')})
    excepted_case('a frame named in two entries FAILs R15 (the frame is still excepted)',
                  lambda fx: fx.write_exceptions([entry(fx, [fx.stem(fx.fa)]),
                                                  entry(fx, [fx.stem(fx.fa)], ruling='A second ruling.')]),
                  {('R08', 'INFO'), ('R09', 'INFO'), ('R15', 'FAIL')})

    unbuildable = (
        ('a date YAML parses but no calendar holds',
         lambda fx: write(fx.exceptions, f'version: 1\nexceptions:\n  - project_id: {fx.pid}\n'
                                         f'    ruling: r\n    date: 2026-02-30\n    by: LO\n'
                                         f'    frames: [{fx.stem(fx.fa)}]\n')),
        ('an integer of 5000 digits', lambda fx: write(fx.exceptions, 'version: ' + '9' * 5000 + '\n')),
        ('nesting 20 000 deep', lambda fx: write(fx.exceptions, '[' * 20000 + ']' * 20000 + '\n')),
    )
    for label, mutate in unbuildable:
        def readable(fx, m=mutate):
            fx.drop_records_of(fx.fa)
            m(fx)
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(tmp)
            readable(fx)
            rep = run(fx)
            r15 = findings_of(rep, 'R15', 'FAIL')
            check(f'{label} is a readable R15 FAIL that excepts nothing',
                  len(r15) == 1 and not r15[0].message.startswith('Checker error')
                  and not findings_of(rep, 'R08', 'INFO'),
                  str([f.message[:120] for f in r15]))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        many = [f'TCRMP2099{n:08d}_clip_XXX_T999' for n in range(20000)]
        fx.write_exceptions([entry(fx, many)])
        t0 = time.monotonic()
        rep = run(fx)
        took = time.monotonic() - t0
        r15 = findings_of(rep, 'R15', 'FAIL')
        check('20 000 stale frames give one aggregated R15 FAIL, quickly',
              len(r15) == 1 and '20000 frames' in r15[0].message and took < 30,
              f'{len(r15)} findings, {took:.1f}s')

    check('the default exceptions file sits in _METADATA',
          cic.DEFAULT_EXCEPTIONS.endswith(os.path.join('_METADATA', 'integrity_exceptions.yaml')),
          cic.DEFAULT_EXCEPTIONS)


def exceptions_cli_cases():
    """--exceptions through a real subprocess."""
    exe = sys.executable
    script = os.path.join(SCRIPTS, 'check_project_integrity.py')
    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        fx.drop_records_of(fx.fa)
        fx.write_exceptions([entry(fx, [fx.stem(fx.fa)])])
        base = [exe, script, '--root', fx.root, '--archive-dir', fx.archive, '--vocab', fx.vocab,
                '--pipeline-config', fx.pipeline, '--snapshot-dir', fx.daily,
                '--report-dir', os.path.join(tmp, 'rep'), '--manual-source-file', fx.const_file,
                '--now', NOW.isoformat()]
        r = subprocess.run(base + ['--exceptions', fx.exceptions], capture_output=True, text=True)
        check('CLI with --exceptions exits 0 and prints INFO lines',
              r.returncode == 0 and '[INFO] R08' in r.stdout, r.stdout[-400:] + r.stderr[-300:])
        r = subprocess.run(base + ['--exceptions', os.path.join(tmp, 'absent.yaml')],
                           capture_output=True, text=True)
        check('CLI without the ruling exits 1', r.returncode == 1, str(r.returncode))


RETURN_REASON = 'Outline saved without a species by the label popup bug; please label or reject it.'
RETURN_TOOL = 'return_frames_to_review.py 1.0.0'
EXPORT_STAMP = (NOW - timedelta(days=1)).strftime('%Y%m%d-%H%M%S')
RETURN_AT = (NOW - timedelta(minutes=5)).isoformat(timespec='seconds')


def return_event(**over):
    """One returned_to_review event as return_frames_to_review.py writes it; `over` replaces keys."""
    e = {'at': RETURN_AT, 'by': 'LO', 'reason': RETURN_REASON, 'tool': RETURN_TOOL, 'previous_exported': True}
    e.update(over)
    return e


def send_back(fx, frame, events=None, batch=None, exported=False):
    """Stamp the manifest's batch names and mark one frame as sent back for review."""
    stamp = f'batch_001_{EXPORT_STAMP}' if batch is None else batch
    fx.edit_csv(os.path.join(fx.exp, 'export_manifest.csv'),
                lambda rows: [r.update(batch_name=stamp) for r in rows])
    evs = [return_event()] if events is None else events
    fx.rewrite_segs(lambda s: s[frame].update(exported=exported, returned_to_review=evs))


def rework_a(fx):
    """Frame A being reworked: its PA outline (mask 0) rejected; stored outcomes and ledger not yet updated."""
    fx.rewrite_segs(lambda s: s[fx.fa]['masks'][0].update(status='rejected'))


def lag_ledger_a(fx):
    """Frame A's ledger rows lag its stored outcomes (PA reads found_ai in the ledger)."""
    stem = fx.stem(fx.fa)
    fx.edit_csv(os.path.join(fx.exp, 'label_provenance.csv'),
                lambda rows: [r.update(outcome='found_ai') for r in rows
                              if r['basename'] == stem and r['label'] == 'PA'])


def returned_cases():
    """A frame sent back for review reads as INFO until it is exported again."""
    case('a returned frame reads as INFO, not FAIL', lambda fx: send_back(fx, fx.fa), {('R07', 'INFO')})

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        send_back(fx, fx.fa)
        rep = run(fx)
        infos = findings_of(rep, 'R07', 'INFO')
        check('the INFO cites the return: when, who, the tool and the reason',
              len(infos) == 1 and all(t in infos[0].message for t in
                                      (RETURN_AT, 'LO', RETURN_TOOL, RETURN_REASON, fx.stem(fx.fa))),
              str([f.message for f in infos]))
        check('a returned frame alone leaves exit code 0', cic.exit_code(rep) == 0)
        check('R07 still reports a PASS beside the INFO', bool(findings_of(rep, 'R07', 'PASS', fx.pid)))

    case('a returned frame being reworked: manifest counts and computed outcomes read as INFO',
         lambda fx: (send_back(fx, fx.fa), rework_a(fx)), {('R07', 'INFO'), ('R08', 'INFO')})
    case('a returned frame whose ledger lags its outcomes reads as INFO',
         lambda fx: (send_back(fx, fx.fa), lag_ledger_a(fx)), {('R07', 'INFO'), ('R08', 'INFO')})
    case('the same lag on a frame never sent back is a FAIL', lag_ledger_a, {('R08', 'FAIL')})
    case('a returned frame that was exported without being marked reviewed reads as INFO',
         lambda fx: (send_back(fx, fx.fa), fx.rewrite_segs(lambda s: s[fx.fa].update(reviewed=False))),
         {('R07', 'INFO'), ('R08', 'INFO')})
    case('ledger rows on an unreviewed, unexported frame never sent back are a FAIL',
         lambda fx: fx.rewrite_segs(lambda s: s[fx.fa].update(reviewed=False, exported=False)),
         {('R07', 'FAIL'), ('R08', 'FAIL')})
    case('a returned frame exported again: the event is history and every rule applies',
         lambda fx: (send_back(fx, fx.fa, exported=True), rework_a(fx)),
         {('R07', 'FAIL'), ('R08', 'WARN'), ('R09', 'FAIL')})
    case('an event older than the export does not cover the frame',
         lambda fx: send_back(fx, fx.fa, events=[return_event(
             at=(NOW - timedelta(days=2)).isoformat(timespec='seconds'))]), {('R07', 'FAIL')})
    case('an export stamp the checker cannot read does not cover the frame',
         lambda fx: send_back(fx, fx.fa, batch='b1'), {('R07', 'FAIL')})
    case('the last event decides: an older event then a newer one covers the frame',
         lambda fx: send_back(fx, fx.fa, events=[
             return_event(at=(NOW - timedelta(days=3)).isoformat(timespec='seconds')), return_event()]),
         {('R07', 'INFO')})
    case('a returned frame that is then scrapped is no longer covered: every rule applies',
         lambda fx: (send_back(fx, fx.fa), fx.rewrite_segs(lambda s: s[fx.fa].update(scrapped=True))),
         {('R07', 'FAIL'), ('R08', 'WARN')})
    case('an empty returned_to_review list on an exported frame is clean',
         lambda fx: fx.rewrite_segs(lambda s: s[fx.fa].update(returned_to_review=[])), set())

    def blank_on_returned(fx):
        send_back(fx, fx.fa)
        fx.rewrite_segs(lambda s: s[fx.fa]['masks'][0].update(species=''))
    case('a return never hides another fault on the same frame', blank_on_returned,
         {('R04', 'FAIL'), ('R07', 'INFO'), ('R08', 'INFO')})

    no_by = return_event()
    del no_by['by']
    malformed = (
        ('a missing field', [no_by]),
        ('a date that is not ISO', [return_event(at='16/10/2026 21:00')]),
        ('a date with no offset', [return_event(at=NOW.replace(tzinfo=None).isoformat(timespec='seconds'))]),
        ('a UTC date', [return_event(at=NOW.astimezone(timezone.utc).isoformat(timespec='seconds'))]),
        ('lower-case initials', [return_event(by='lo')]),
        ('five-letter initials', [return_event(by='LOLOL')]),
        ('initials ending in a line break', [return_event(by='LO\n')]),
        ('a reason of 3 characters', [return_event(reason='fix')]),
        ('a reason that is not text', [return_event(reason=42)]),
        ('a blank tool', [return_event(tool='  ')]),
        ('previous_exported false', [return_event(previous_exported=False)]),
        ('an event that is not an object', ['returned']),
        ('a value that is not a list', 'returned'),
        ('a good event after a malformed one', [return_event(by='x'), return_event()]),
    )
    for label, events in malformed:
        case(f'a returned frame with {label} is an R04 FAIL and is not covered',
             lambda fx, e=events: send_back(fx, fx.fa, events=e), {('R04', 'FAIL'), ('R07', 'FAIL')})
    case('a malformed event on an exported frame is an R04 FAIL',
         lambda fx: fx.rewrite_segs(lambda s: s[fx.fa].update(returned_to_review=[return_event(by='x')])),
         {('R04', 'FAIL')})

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        send_back(fx, fx.fa, events=[return_event(at='not a date')])
        msgs = [f.message for f in findings_of(run(fx), 'R04', 'FAIL')]
        check('the R04 FAIL names the frame and what is wrong',
              len(msgs) == 1 and fx.fa in msgs[0] and 'date' in msgs[0], str(msgs))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp, pid="odd name 'q' é中", frame_a="TCRMP20201010_clip_BPT_T10 1 é.jpg")
        send_back(fx, fx.fa)
        got = issues(run(fx))
        check('hostile project and frame names: a returned frame reads as INFO', got == {('R07', 'INFO')},
              str(sorted(got)))

    with tempfile.TemporaryDirectory() as tmp:
        fx = Fixture(tmp)
        send_back(fx, fx.fa)
        rework_a(fx)
        before = hash_tree(tmp)
        run(fx)
        check('the checker stays read-only with returned frames present', hash_tree(tmp) == before)


def main():
    """Run every group and exit nonzero on any failure."""
    for fn in (clean_cases, rule_cases, structure_cases, read_only_cases, cli_cases,
               exceptions_cases, exceptions_file_cases, exceptions_cli_cases, returned_cases):
        fn()
    print(f'\n{len(PASSES)} passed, {len(FAILS)} failed')
    for name in FAILS:
        print(f'  FAILED: {name}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
