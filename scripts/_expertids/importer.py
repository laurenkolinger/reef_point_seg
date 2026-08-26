"""
Core expert-ID import logic (no Flask, no SAM3).

ROLLING ingest model (CONTRACTS §4,§5,§6, C5/C6):

Import (a reviewer's returned CSV) does NOT prune the repo and does NOT set
expert_id. For every row it:
  1. Groups rows by their `project_id` column.
  2. Resolves each project's step-5 export dir via
     project_manager.resolve_step_dir(project_id, projects_root); falls back to
     the library record's project_id, then the caller's open export_dir.
     Every UID that cannot be routed is RETURNED/REPORTED (never silent).
  3. UPSERTs the reviewer's tentative ID into the manifest item's reviews[]
     (review_repo.add_review, keyed by `reviewer`) AND the library
     reviews/<uid>.json — so tentative IDs show on the site immediately.
  4. Pushes the manifest so the live site reflects the tentative reviews.
The overlap-tentative cascade runs PER resolved export_dir ONLY (identical frame
filenames across projects must not cross-contaminate).

Acceptance (C6, the consensus builder) is the ONLY path that sets expert_id,
relabels the project's step-5 segmentations.json mask (mode EXPERT), removes the
item from the review repo, and prunes.

Both paths finish with a BEST-EFFORT provenance refresh: each DISTINCT
segmentations.json they wrote gets its project's label_provenance.csv ledger
regenerated (via the annotator's stdlib-only provenance module) so the coverage
matrix shows pending_expert / found_expert immediately instead of waiting for
the annotator to re-touch the frame. A refresh failure logs and never fails the
import/accept.

Shared logic for the Add-Expert-IDs blueprint ([[_expertids.blueprint]]); mounted
both by the standalone tool and natively by the reef_point_seg orchestrator.
"""

import glob
import json
import os
import sys
import tempfile
from contextlib import contextmanager

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:          # pragma: no cover — non-POSIX fallback
    _HAVE_FCNTL = False

# scripts/ on path for the shared package (_expertids lives directly under it).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _reefreview import codes as codes_mod
from _reefreview import filename_parse
from _reefreview.uid import make_uid, mask_uid, is_safe_uid
from _reefreview import mask_geom
from _reefreview.library import Library
from _reefreview.library import replace_preserving_mode as _replace_preserving_mode
from _reefreview.review_repo import ReviewRepo
from _reefreview.clock import now_ast, now_ast_iso

REVIEW_CODES = {'', 'REVIEW'}

# CSV columns the importer understands (CONTRACTS §4). Extra trailing
# human-readable columns (project_name, site, frame, name) are ignored.
_CSV_FIELDS = ('uid', 'code', 'confidence', 'reviewer', 'labeler', 'project_id')


def _pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def default_projects_root():
    """`<module>/inprocess` resolved from this package (CONTRACTS §1,§5).

    Mirrors _reefreview.library.default_dir's walk so the routing is robust no
    matter the CWD: scripts/_expertids -> scripts -> github_repo -> module."""
    pkg = _pkg_dir()                                  # scripts/_expertids
    scripts = os.path.dirname(pkg)                    # scripts
    repo = os.path.dirname(scripts)                   # github_repo
    return os.path.join(os.path.dirname(repo), 'inprocess')


def _resolve_step_dir(project_id, projects_root, step=None):
    """Resolve a project_id to the export dir whose segmentations/ the expert-id
    round trip relabels, via the orchestrator's project_manager. Import here (not
    at module top) so the importer keeps working in contexts where
    pipeline_orchestrator isn't importable; a failure just yields '' and the
    caller falls back.

    When `step` is given, resolve that exact step. When `step` is None (the
    default), prefer the 4.test combined-annotator export (step4test, which
    replaces Steps 4+5) if that project actually has a segmentations/ tree, else
    fall back to the legacy Step-5 dir so older Step-5-annotated projects still
    round-trip."""
    if not project_id or not projects_root:
        return ''
    try:
        scripts = os.path.dirname(_pkg_dir())
        po = os.path.join(scripts, 'pipeline_orchestrator')
        if po not in sys.path:
            sys.path.insert(0, po)
        import project_manager  # noqa: E402
        if step is not None:
            return project_manager.resolve_step_dir(project_id, projects_root, step=step) or ''
        d4 = project_manager.resolve_step_dir(project_id, projects_root, step='step4test') or ''
        # Prefer 4.test only when it actually HOLDS segmentation data: an empty or
        # aborted-routing step4test/segmentations/ dir must NOT shadow a real
        # Step-5 annotation (else import/accept would silently write nowhere).
        # Match _iter_segmentation_files' exact glob so the preference and the
        # write target can never disagree.
        if d4 and (os.path.isfile(os.path.join(d4, 'segmentations', 'segmentations.json'))
                   or glob.glob(os.path.join(d4, 'segmentations', '*', 'segmentations.json'))):
            return d4
        return project_manager.resolve_step_dir(project_id, projects_root, step='5') or ''
    except Exception:
        return ''


def parse_csv_text(text):
    """Parse CSV text into [{uid, code, confidence, reviewer, labeler,
    project_id}]. Tolerates a header row (CONTRACTS §4 header
    `uid,code,confidence,reviewer,project_id` plus ignored trailing columns) or
    a headerless positional uid,code[,confidence,reviewer/labeler] CSV. Blank
    rows are skipped."""
    import csv
    import io
    rows = []
    # Excel round-trips prepend a UTF-8 BOM; without stripping it the first
    # header cell reads '\ufeffuid', header detection fails, and the file is
    # parsed positionally (header leaks as a phantom data row).
    text = (text or '')
    if text.startswith('\ufeff'):
        text = text[1:]
    # A CSV cell can never legitimately exceed the whole input text, so lift the
    # csv module's per-field cap (default 131072) to cover this input instead of
    # letting reader iteration crash with _csv.Error on an oversized field
    # (CONTRACTS §4: the importer is tolerant — bad rows are skipped/reported
    # downstream by is_safe_uid, never a crash). The limit is only ever RAISED
    # (monotonic), so concurrent parses under a threaded Flask host can never
    # see it shrink mid-iteration.
    needed = len(text) + 16
    if csv.field_size_limit() < needed:
        csv.field_size_limit(needed)
    reader = csv.reader(io.StringIO(text))
    header = None
    for raw in reader:
        if not raw or all(not c.strip() for c in raw):
            continue
        cells = [c.strip() for c in raw]
        low = [c.lower() for c in cells]
        if header is None and ('uid' in low and 'code' in low):
            header = low
            continue
        if header:
            idx = {name: header.index(name) for name in header}

            def _g(name, _idx=idx, _cells=cells):
                i = _idx.get(name)
                return _cells[i] if (i is not None and i < len(_cells)) else ''
            # `reviewer` (CONTRACTS §4) is the identity gate name. `labeler` is
            # the legacy column name; accept either, reviewer wins.
            reviewer = _g('reviewer') or _g('labeler')
            rows.append({'uid': _g('uid'), 'code': _g('code'),
                         'confidence': _g('confidence'),
                         'reviewer': reviewer, 'labeler': _g('labeler'),
                         'project_id': _g('project_id')})
        else:
            # Headerless: positional uid,code[,confidence,reviewer]
            rows.append({'uid': cells[0] if len(cells) > 0 else '',
                         'code': cells[1] if len(cells) > 1 else '',
                         'confidence': cells[2] if len(cells) > 2 else '',
                         'reviewer': cells[3] if len(cells) > 3 else '',
                         'labeler': cells[3] if len(cells) > 3 else '',
                         'project_id': cells[4] if len(cells) > 4 else ''})
    return [r for r in rows if r['uid']]


def _iter_segmentation_files(export_dir):
    flat = os.path.join(export_dir, 'segmentations', 'segmentations.json')
    if os.path.isfile(flat):
        return [flat]
    pat = os.path.join(export_dir, 'segmentations', '*', 'segmentations.json')
    return sorted(glob.glob(pat))


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@contextmanager
def _seg_locked(path):
    """Exclusive cross-process/cross-thread lock for ONE segmentations.json
    read-modify-write cycle (mirrors ReviewRepo._locked's flock pattern, but
    per seg file). Two concurrent CSV imports for the same project each
    load-mutate-save the same segmentations.json; without this lock the second
    writer clobbers the first reviewer's freshly upserted reviews[] row (lost
    update) while the flock-protected manifest keeps both, so manifest and
    segmentations silently diverge. flock is per open file description, so
    separate open() handles exclude both threads and processes. The hidden
    sibling lock file never matches _iter_segmentation_files' globs."""
    if not _HAVE_FCNTL:
        yield
        return
    d = os.path.dirname(os.path.abspath(path))
    lock_path = os.path.join(d, '.' + os.path.basename(path) + '.lock')
    try:
        fh = open(lock_path, 'w')
    except OSError:
        # Read-only/foreign tree: the lock file cannot be created. Proceed
        # unlocked — a dir we cannot create a lock file in is a dir _save
        # cannot write either, so lost-update protection is moot there, and
        # load-only iteration over read-only projects must keep working
        # (pre-lock behavior: failures surfaced only at an actual write).
        yield
        return
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _save(path, data):
    """Atomic JSON write via a UNIQUE temp file. A fixed '<path>.tmp' name let
    two concurrent writers race: both wrote the same tmp, the first os.replace
    consumed it, and the second crashed with FileNotFoundError mid-import
    (surfacing as an HTTP 500 with partial state)."""
    d = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=1)
        _replace_preserving_mode(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── ACCEPT path: set the FINAL expert_id on a mask (C6 only) ───────────
def _apply_to_mask(mask, code, info, labeler, via, basis='operator'):
    """Stamp the accepted EXPERT id onto a mask. Only the consensus/accept path
    (C6) calls this — rolling import never sets expert_id (CONTRACTS §5)."""
    mask['species'] = code
    mask['name'] = info.get('name', '') if info else ''
    mask['category'] = info.get('category', '') if info else mask.get('category', '')
    mask['review'] = False
    # export_yolo trains status=='accepted' masks only, and provenance counts
    # found_* only for accepted non-review masks - an expert accept must flip
    # a still-'pending' mask or it never reaches training or the matrix.
    mask['status'] = 'accepted'
    mask['expert_id'] = {'code': code, 'mode': 'EXPERT', 'labeler': labeler,
                         'via': via, 'basis': basis, 'at': now_ast()}


# ── ROLLING path: append a tentative reviewer ID to a mask (C5) ────────
def _upsert_tentative_on_mask(mask, reviewer, code, confidence, at):
    """UPSERT one reviewer's tentative ID into mask.reviews[] keyed by reviewer
    (CONTRACTS §3,§5). Does NOT set expert_id and does NOT clear review state —
    the mask stays pending until acceptance."""
    row = {'reviewer': reviewer, 'code': code or '',
           'confidence': confidence or '', 'at': at}
    reviews = [r for r in (mask.get('reviews') or [])
               if (r.get('reviewer') or '').strip() != reviewer]
    reviews.append(row)
    mask['reviews'] = reviews


# ── provenance ledger refresh (best-effort, after any seg mutation) ────
def _in_expert_flow(mask):
    """True when a mask participates in the expert-review pathway: still
    review-flagged, carrying tentative reviews[], or stamped with an
    expert_id. Frames with none of these are the annotator's business only."""
    return (bool(mask.get('review')) or mask.get('species') == 'REVIEW'
            or bool(mask.get('reviews'))
            or isinstance(mask.get('expert_id'), dict))


def _refresh_provenance_ledger(seg_path, reviewer='', log_fn=None):
    """Best-effort regeneration of the label_provenance.csv ledger after this
    importer mutated seg_path (a segmentations.json it just wrote), so the
    coverage matrix (scripts/_matrix/builder.py) reflects found_expert /
    pending_expert immediately instead of staying stale until the annotator's
    next save/accept/export re-stamps the frame.

    Mirrors the annotator's ledger conventions
    (TCRMPclip_combinedAnnotate/src/app.py _stamp_provenance):
      - the ledger lives at <export_dir>/label_provenance.csv where export_dir
        is the parent of the segmentations/ tree holding seg_path;
      - rows are keyed (splitext(frame filename), label) and upserted in place
        by provenance.write_provenance_csv;
      - project_id is the nearest run_* path part, else the export dir's
        parent folder name (annotator _derive_project_id).
    Only frames already in the expert flow are refreshed, and with NO target
    species, so not_found bookkeeping stays the annotator's job (it knows the
    project's target list; the importer does not).

    Never raises: any failure is logged and swallowed — imports/accepts must
    never fail because of a provenance refresh. Returns True when the ledger
    was written."""
    log = log_fn or (lambda _m: None)
    try:
        # <export_dir>/segmentations[/<year>]/segmentations.json -> export_dir.
        d = os.path.dirname(os.path.abspath(seg_path))
        while os.path.basename(d) != 'segmentations':
            parent = os.path.dirname(d)
            if parent == d:
                return False   # never found a segmentations/ ancestor
            d = parent
        export_dir = os.path.dirname(d)
        # Cross-package import of the annotator's stdlib-only provenance module
        # (same lazy sys.path pattern as _resolve_step_dir's orchestrator
        # import; the e2e suite imports it the same way).
        src = os.path.join(os.path.dirname(_pkg_dir()),
                           'TCRMPclip_combinedAnnotate', 'src')
        if src not in sys.path:
            sys.path.insert(0, src)
        import provenance  # noqa: E402
        data = _load(seg_path)
        if not data:
            return False
        project_id = ''
        for part in reversed(os.path.abspath(export_dir).split(os.sep)):
            if part.startswith('run_'):
                project_id = part
                break
        if not project_id:
            project_id = os.path.basename(
                os.path.dirname(os.path.abspath(export_dir))) or 'project'
        wrote = False
        fresh = {}   # frame stem -> set of freshly computed labels
        for fn, seg in data.items():
            if not isinstance(seg, dict):
                continue
            masks = [m for m in (seg.get('masks') or []) if isinstance(m, dict)]
            if not any(_in_expert_flow(m) for m in masks):
                continue
            outcomes = provenance.compute_label_outcomes(
                seg, [], 'expertids', reviewer=reviewer)
            fresh[os.path.splitext(fn)[0]] = set(outcomes)
            if not outcomes:
                continue
            provenance.write_provenance_csv(
                export_dir, os.path.splitext(fn)[0], outcomes, project_id)
            wrote = True
        # write_provenance_csv only upserts incoming (basename, label) pairs,
        # so a tentative code that vanished from the mask (corrected review,
        # or the literal REVIEW placeholder superseded by a real code) would
        # leave its pending_expert row behind. Regenerating means the pending
        # set for each refreshed frame is exact: prune the leftovers.
        pruned = _prune_stale_pending_rows(export_dir, fresh,
                                           provenance.CSV_HEADER)
        return wrote or pruned
    except Exception as e:
        log(f"[provenance] ledger refresh failed for {seg_path} (non-fatal): {e}")
        return False


def _prune_stale_pending_rows(export_dir, fresh, header):
    """Drop pending_expert ledger rows for just-refreshed frames whose label no
    longer appears in that frame's freshly computed outcomes. found_* and
    not_found rows are never touched (they stay the annotator's business).
    `fresh` maps frame stem -> set of current labels. Returns True if the
    ledger changed. Caller holds the try/except."""
    import csv
    path = os.path.join(export_dir, 'label_provenance.csv')
    if not fresh or not os.path.isfile(path):
        return False
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows
            if not (r.get('outcome') == 'pending_expert'
                    and r.get('basename') in fresh
                    and r.get('label') not in fresh[r.get('basename')])]
    if len(kept) == len(rows):
        return False
    # Unique tmp name (same rationale as _save): concurrent refreshes must not
    # consume each other's tmp file mid-replace.
    fd, tmp = tempfile.mkstemp(dir=export_dir,
                               prefix='label_provenance.csv.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for r in kept:
                writer.writerow({k: r.get(k, '') for k in header})
        _replace_preserving_mode(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True


def _refresh_provenance_ledgers(paths, reviewer='', log_fn=None):
    """Refresh the ledger once per DISTINCT touched segmentations file.
    Doubly guarded (the helper already never raises) so a broken helper can
    never fail an import/accept either. Returns the number refreshed."""
    log = log_fn or (lambda _m: None)
    refreshed = 0
    for path in sorted(set(paths)):
        try:
            if _refresh_provenance_ledger(path, reviewer=reviewer, log_fn=log):
                refreshed += 1
        except Exception as e:
            log(f"[provenance] refresh crashed for {path} (non-fatal): {e}")
    return refreshed


def _mask_matches_uid(fn, mask, parsed, target_uid):
    """True if this mask is the one identified by target_uid (persisted
    review_uid, content-stable mask_uid, or the legacy point uid)."""
    u = mask.get('review_uid') or mask_uid(fn, mask, parsed)
    if u == target_uid:
        return True
    return make_uid(fn, mask.get('source_x', 0), mask.get('source_y', 0),
                    parsed) == target_uid


def _tentative_in_segmentations(export_dir, source_image, target_uid,
                                reviewer, code, confidence, at, touched=None):
    """Append a tentative reviewer ID to the matching mask in the project's
    segmentations (no expert_id). Returns True if a mask was updated. Same
    frame can appear in multiple segmentation files; update them all.
    When `touched` (a set) is given, every saved file path is added to it so
    the caller can refresh the provenance ledger once per file."""
    base = os.path.basename(source_image or '')
    found_any = False
    for path in _iter_segmentation_files(export_dir):
        # Lock spans the WHOLE load-mutate-save so a concurrent import's
        # freshly written reviewer row is re-read, upserted alongside, and
        # never clobbered (lost update).
        with _seg_locked(path):
            data = _load(path)
            if not data:
                continue
            changed = False
            for fn, seg in data.items():
                if base and os.path.basename(fn) != base:
                    continue
                parsed = filename_parse.parse(fn)
                for mask in seg.get('masks', []):
                    if _mask_matches_uid(fn, mask, parsed, target_uid):
                        _upsert_tentative_on_mask(mask, reviewer, code,
                                                  confidence, at)
                        changed = True
            if changed:
                _save(path, data)
                found_any = True
                if touched is not None:
                    touched.add(path)
    return found_any


def _relabel_accepted_in_segmentations(export_dir, source_image, target_uid,
                                       code, info, labeler, basis='operator',
                                       touched=None):
    """ACCEPT (C6): find the mask whose UID == target_uid and stamp the final
    EXPERT id. Returns True if a mask was updated. Restricted to ONE export_dir
    so identical frame names in other projects are never touched. When
    `touched` (a set) is given, every saved file path is added to it so the
    caller can refresh the provenance ledger once per file."""
    base = os.path.basename(source_image or '')
    found_any = False
    for path in _iter_segmentation_files(export_dir):
        # Same locked read-modify-write as the tentative path: an accept
        # racing a concurrent import must not clobber (or be clobbered by)
        # that import's reviews[] upsert.
        with _seg_locked(path):
            data = _load(path)
            if not data:
                continue
            changed = False
            for fn, seg in data.items():
                if base and os.path.basename(fn) != base:
                    continue
                parsed = filename_parse.parse(fn)
                for mask in seg.get('masks', []):
                    if _mask_matches_uid(fn, mask, parsed, target_uid):
                        _apply_to_mask(mask, code, info, labeler, 'consensus',
                                       basis)
                        changed = True
            if changed:
                _save(path, data)
                found_any = True
                if touched is not None:
                    touched.add(path)
    return found_any


def _overlap_tentative_pass(export_dir, lib, thresh, reviewer, log, touched=None):
    """PER-export_dir cascade (CONTRACTS §5): any remaining TO-REVIEW mask in
    THIS project that overlaps a now-EXPERT-labeled library mask >thresh inherits
    that code as a TENTATIVE review (not an expert_id). Returns list of
    (uid, code, confidence) it touched so the caller can mirror to the repo +
    library. Never crosses export_dir boundaries. When `touched` (a set) is
    given, every saved file path is added to it so the caller can refresh the
    provenance ledger once per file."""
    resolved = []
    for path in _iter_segmentation_files(export_dir):
        # Locked read-modify-write: the cascade re-reads the file AFTER any
        # concurrent import's row upserts land, so it never writes a stale
        # copy over them.
        with _seg_locked(path):
            data = _load(path)
            if not data:
                continue
            changed = False
            for fn, seg in data.items():
                parsed = filename_parse.parse(fn)
                for mask in seg.get('masks', []):
                    is_review = (mask.get('review')
                                 or mask.get('species') == 'REVIEW')
                    if not is_review or mask.get('status') == 'rejected':
                        continue
                    # Already carries this reviewer's tentative row? skip.
                    have = any((r.get('reviewer') or '').strip() == reviewer
                               for r in (mask.get('reviews') or []))
                    if have:
                        continue
                    uid = mask.get('review_uid') or mask_uid(fn, mask, parsed)
                    match = lib.find_overlap(fn, mask.get('rle'), thresh=thresh,
                                             exclude_uid=uid)
                    if not match:
                        continue
                    code = match.get('code', '')
                    conf = match.get('confidence', '')
                    _upsert_tentative_on_mask(mask, reviewer, code, conf,
                                              now_ast_iso())
                    resolved.append((uid, code, conf))
                    changed = True
                    log(f"[import] {uid}: tentative {code} via overlap from "
                        f"{match.get('uid')} (overlap {match.get('_overlap')})")
            if changed:
                _save(path, data)
                if touched is not None:
                    touched.add(path)
    return resolved


# ── operator setup persistence (C2/C3) ────────────────────────────────
def _operator_setup_path(review_dir):
    return os.path.join(review_dir, 'operator_setup.json')


def load_operator_setup(review_dir):
    """Return {email, candidate_codes:[...]} the operator last saved (C2/C3),
    or an empty shell. Never raises."""
    empty = {'email': '', 'candidate_codes': []}
    try:
        with open(_operator_setup_path(review_dir)) as f:
            obj = json.load(f)
        return {'email': (obj.get('email') or '').strip(),
                'candidate_codes': [c for c in (obj.get('candidate_codes') or []) if c]}
    except Exception:
        return empty


def _parse_codes_list(raw):
    """'OFAV, OFRA' (or a list) -> ['OFAV','OFRA'], de-duped, upper-cased,
    order-preserving."""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace('\n', ',').split(',')]
    else:
        parts = [str(p).strip() for p in (raw or [])]
    out = []
    for p in parts:
        c = p.upper()
        if c and c not in out:
            out.append(c)
    return out


def save_operator_setup(*, review_dir, repo_url, master_codes,
                        email='', candidate_codes='', site_codes=None,
                        git_push=False, log_fn=None):
    """C2/C3: persist the operator's review-site setup — the contact EMAIL
    (review_repo.set_contacts) and extra candidate 'potential IDs'
    (codes.json candidate_codes). The extras are stored in a sidecar so they
    survive a later write_codes() re-seed, then merged into codes.json.

    Returns {email, candidate_codes}. Idempotent."""
    log = log_fn or (lambda _m: None)
    repo = ReviewRepo(review_dir, remote_url=repo_url, master_codes_path=master_codes,
                      site_codes_path=site_codes, log_fn=log)
    email = (email or '').strip()
    extras = _parse_codes_list(candidate_codes)
    # Persist operator intent to the sidecar FIRST (source of truth on re-seed):
    # ensure_repo()'s write_codes() reads the sidecar, so both additions AND
    # removals of extras take effect in this same call.
    setup = {'email': email, 'candidate_codes': extras}
    os.makedirs(review_dir, exist_ok=True)   # sidecar may precede the repo tree
    tmp = _operator_setup_path(review_dir) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(setup, f, indent=1)
    os.replace(tmp, _operator_setup_path(review_dir))
    repo.ensure_repo()
    # Reflect the contact email into the manifest (viewer acknowledgment list).
    repo.set_contacts([email] if email else [])
    # Belt+braces union into codes.json's candidate_codes (write_codes already
    # honors the sidecar; this heals a codes.json written by an older seed).
    _merge_candidate_codes(repo, extras)
    if git_push:
        repo.commit_push('operator setup: contacts + candidate codes', push=True)
    return setup


def _merge_candidate_codes(repo, extras):
    """Union the operator's extra candidate codes into the repo's codes.json
    (atomic). write_codes() already wrote the master-flagged base set plus the
    sidecar extras; this heals older codes.json files and skips the rewrite
    entirely when nothing would change (idempotency: never dirty a clean tree)."""
    path = os.path.join(repo.root, 'codes.json')
    try:
        with open(path) as f:
            cj = json.load(f)
    except Exception:
        return
    base = cj.get('candidate_codes') or []
    merged = list(base)
    for c in extras:
        if c not in merged:
            merged.append(c)
    if merged == base:
        return
    cj['candidate_codes'] = merged
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(cj, f, indent=1)
    os.replace(tmp, path)


# ── consensus classifier (C6) ─────────────────────────────────────────
def classify_reviews(reviews):
    """Classify a uid's reviews[] (CONTRACTS §6).

    Returns {status, code, codes, n_reviewers} where status is:
      'none'      zero reviewers
      'single'    exactly one reviewer (real or blank code)
      'consensus' >=2 reviewers AND all non-blank codes agree (>=2 real codes)
      'conflict'  >=2 DISTINCT non-blank codes
    Blank/IDK codes are ignored for agreement. `code` is the suggested final
    code (the agreed code on consensus, the lone code on single, else '')."""
    rows = reviews or []
    n = len(rows)

    def _real(c):
        c = (c or '').strip()
        return c if c not in ('', 'IDK') else ''

    real_codes = [_real(r.get('code')) for r in rows]
    distinct = []
    for c in real_codes:
        if c and c not in distinct:
            distinct.append(c)
    if n == 0:
        return {'status': 'none', 'code': '', 'codes': [], 'n_reviewers': 0}
    if n == 1:
        return {'status': 'single', 'code': real_codes[0],
                'codes': distinct, 'n_reviewers': 1}
    if len(distinct) >= 2:
        return {'status': 'conflict', 'code': '', 'codes': distinct, 'n_reviewers': n}
    # n>=2 and 0 or 1 distinct real codes.
    n_real = sum(1 for c in real_codes if c)
    if len(distinct) == 1 and n_real >= 2:
        return {'status': 'consensus', 'code': distinct[0],
                'codes': distinct, 'n_reviewers': n}
    # >=2 reviewers but <2 real agreeing codes (e.g. one real + blanks) — treat
    # as single-real (the one real code suggested, not yet a 2-way consensus).
    return {'status': 'single', 'code': distinct[0] if distinct else '',
            'codes': distinct, 'n_reviewers': n}


# ── rolling import (CONTRACTS §4,§5, C5) ───────────────────────────────
def import_rows(rows, *, export_dir, review_dir, repo_url, library_dir, master_codes,
                projects_root=None, overlap_thresh=0.5, git_push=True,
                default_reviewer='', default_labeler='', site_codes=None,
                log_fn=None):
    """ROLLING ingest. Group rows by project_id, resolve each project's step-5
    export dir, UPSERT each row into the item's reviews[] (manifest + library),
    run the per-project overlap-tentative cascade, then push. Does NOT prune and
    does NOT set expert_id (CONTRACTS §5).

    Returns a stats dict that ALWAYS includes `unrouted` — the list of UIDs that
    could not be routed to any export_dir (CONTRACTS §5: never silent)."""
    log = log_fn or (lambda _m: None)
    projects_root = projects_root or default_projects_root()
    lib = Library(library_dir or None)
    lib.ensure()
    repo = ReviewRepo(review_dir, remote_url=repo_url, master_codes_path=master_codes,
                      site_codes_path=site_codes, log_fn=log)
    repo.ensure_repo()

    by_code = codes_mod.load_codes(master_codes)['by_code']
    stats = {'rows': 0, 'reviews_added': 0, 'unknown_code': 0, 'unsafe_uid': 0,
             'projects': [], 'unrouted': [], 'auto_tentative': 0, 'uids': [],
             'provenance_refreshed': 0, 'seg_write_failed': []}
    touched_seg = set()   # distinct segmentations.json files this import wrote

    # Group rows by project_id (CONTRACTS §5).
    groups = {}
    for row in rows:
        uid = (row.get('uid') or '').strip()
        if not uid:
            continue
        groups.setdefault((row.get('project_id') or '').strip(), []).append(row)

    routed_dirs = set()
    for pid, prows in groups.items():
        # Resolve the export dir for this project's rows.
        export_for = _resolve_step_dir(pid, projects_root) if pid else ''
        proj_stat = {'project_id': pid, 'export_dir': export_for,
                     'resolved_via': 'project_id' if export_for else '',
                     'rows': 0, 'routed': 0, 'unrouted': 0}

        for row in prows:
            uid = (row.get('uid') or '').strip()
            if not is_safe_uid(uid):
                stats['unsafe_uid'] += 1
                log(f"[import] skipping unsafe UID from CSV: {uid!r}")
                continue
            stats['rows'] += 1
            proj_stat['rows'] += 1
            code = (row.get('code') or '').strip()
            reviewer = ((row.get('reviewer') or '').strip()
                        or (row.get('labeler') or '').strip() or default_reviewer)
            conf = (row.get('confidence') or '').strip()
            if not reviewer:
                # CONTRACTS §4/§8: reviewer is REQUIRED. Report (don't crash).
                stats['unrouted'].append(uid)
                proj_stat['unrouted'] += 1
                log(f"[import] {uid}: blank reviewer — skipped (CONTRACTS §4)")
                continue
            if code and code not in by_code and code not in ('IDK',):
                stats['unknown_code'] += 1  # kept anyway

            # Per-row fallback routing: project_id dir -> library record's
            # project_id -> caller's open export_dir.
            row_export = export_for
            via = proj_stat['resolved_via']
            rec = lib.lookup(uid)
            if not row_export and rec and rec.get('project_id'):
                row_export = _resolve_step_dir(rec['project_id'], projects_root)
                via = 'library_project_id' if row_export else via
            if not row_export and export_dir:
                row_export = export_dir
                via = 'open_export_dir'
            if not row_export:
                stats['unrouted'].append(uid)
                proj_stat['unrouted'] += 1
                log(f"[import] {uid}: UNROUTED (project_id={pid!r}) — reported")
                continue
            routed_dirs.add(row_export)
            proj_stat['routed'] += 1
            proj_stat['resolved_via'] = proj_stat['resolved_via'] or via

            at = now_ast_iso()
            # 1) Manifest: rolling upsert keyed by reviewer (tentative shows).
            updated = repo.add_review(uid, reviewer, code, conf, at=at)
            if updated is not None:
                stats['reviews_added'] += 1
            # 2) Library per-uid detail mirror (survives cross-project).
            # Locked upsert: two concurrent imports for the same uid must not
            # lose each other's reviews/<uid>.json rows (same lost-update the
            # manifest lock and _seg_locked already prevent).
            lib.upsert_review(uid, reviewer, code=code, confidence=conf, at=at)
            # 3) Segmentations: tentative review on the routed project's mask.
            src = (rec or {}).get('source_image', '')
            try:
                _tentative_in_segmentations(row_export, src, uid, reviewer, code,
                                            conf, at, touched=touched_seg)
            except OSError as e:
                # The manifest + library rows already landed; a read-only or
                # full segmentations tree must not crash the import with
                # partial state - report per uid instead (never silent).
                stats['seg_write_failed'].append({'uid': uid, 'error': str(e)})
                log(f"[import] {uid}: segmentations write failed ({e}) - reported")
            stats['uids'].append(uid)

        stats['projects'].append(proj_stat)

    # Overlap-tentative cascade PER resolved export_dir ONLY (CONTRACTS §5).
    # Attribute inherited tentative IDs to a synthetic 'overlap' reviewer so they
    # never collide with a real reviewer's row.
    for d in sorted(routed_dirs):
        try:
            auto = _overlap_tentative_pass(d, lib, overlap_thresh, 'overlap', log,
                                           touched=touched_seg)
        except OSError as e:
            stats['seg_write_failed'].append({'uid': '', 'error': f'overlap cascade {d}: {e}'})
            log(f"[import] overlap cascade failed in {d} ({e}) - reported")
            continue
        for uid, code, conf in auto:
            stats['auto_tentative'] += 1
            repo.add_review(uid, 'overlap', code, conf)
            lib.upsert_review(uid, 'overlap', code=code, confidence=conf,
                              at=now_ast_iso())

    # Provenance ledger refresh, once per DISTINCT touched segmentations file
    # (best-effort: a failure logs and never fails the import).
    stats['provenance_refreshed'] = _refresh_provenance_ledgers(
        touched_seg, log_fn=log)

    # Push the manifest so tentative reviews are visible. NO pruning (CONTRACTS §5).
    if stats['reviews_added'] or stats['auto_tentative']:
        msg = (f"import: +{stats['reviews_added']} tentative reviews "
               f"({stats['auto_tentative']} via overlap) across "
               f"{len(stats['projects'])} project(s)")
        stats['pushed'] = repo.commit_push(msg, push=git_push)
    else:
        stats['pushed'] = False
    stats['pending_total'] = len(repo.pending_uids())
    return stats


# ── acceptance (C6) ────────────────────────────────────────────────────
def accept_uid(uid, code, *, review_dir, repo_url, library_dir, master_codes,
               projects_root=None, labeler='', basis='consensus',
               git_push=True, site_codes=None, open_export_dir=None, log_fn=None):
    """Operator acceptance (C6, the ONLY expert_id path). Sets item.accepted +
    removes from the repo (review_repo.accept_item), stamps the library record
    EXPERT, and relabels the routed project's step-5 segmentations.json mask.

    Returns {ok, uid, code, accepted, relabeled_seg, export_dir, pushed} or
    {ok:False, error} when the UID isn't safe / not pending."""
    log = log_fn or (lambda _m: None)
    projects_root = projects_root or default_projects_root()
    if not is_safe_uid(uid):
        return {'ok': False, 'error': f'unsafe uid {uid!r}'}
    repo = ReviewRepo(review_dir, remote_url=repo_url, master_codes_path=master_codes,
                      site_codes_path=site_codes, log_fn=log)
    repo.ensure_repo()
    lib = Library(library_dir or None)
    lib.ensure()
    by_code = codes_mod.load_codes(master_codes)['by_code']
    info = by_code.get(code, {})

    # Resolve the routed export_dir BEFORE accept_item prunes the manifest item.
    item = next((it for it in repo.load_manifest().get('items', [])
                 if it.get('uid') == uid), None)
    if item is None:
        return {'ok': False, 'error': f'uid not pending: {uid}'}
    pid = item.get('project_id') or ''
    rec = lib.lookup(uid) or {}
    src = item.get('source_image') or rec.get('source_image', '')
    export_dir = _resolve_step_dir(pid, projects_root) if pid else ''
    if not export_dir and rec.get('project_id'):
        export_dir = _resolve_step_dir(rec['project_id'], projects_root)
    # Third tier mirrors import_rows' open_export_dir fallback: a UID routed at
    # import-time only via the active project's open export dir (blank or
    # unresolvable project_id) would otherwise resolve to '' here and accept
    # would silently NOT relabel the on-disk mask that feeds training.
    if not export_dir and open_export_dir:
        export_dir = open_export_dir

    # 1) Manifest: set accepted + remove (the ONLY removal path).
    accepted = repo.accept_item(uid, code, labeler=labeler, basis=basis)
    if accepted is None:
        return {'ok': False, 'error': f'accept failed: {uid}'}

    # 2) Library record -> EXPERT.
    lib.upsert({'uid': uid, 'code': code, 'name': info.get('name', ''),
                'category': info.get('category', ''),
                'labeler': labeler, 'mode': 'EXPERT'})
    # Locked read-modify-write: an accept racing a concurrent import's
    # upsert_review on the same uid must not clobber its row.
    lib.set_review_accepted(uid, accepted)

    # 3) Segmentations relabel — PER resolved export_dir ONLY (CONTRACTS §5).
    relabeled = False
    relabel_error = ''
    touched_seg = set()
    if export_dir:
        try:
            relabeled = _relabel_accepted_in_segmentations(
                export_dir, src, uid, code, info, labeler, basis=basis,
                touched=touched_seg)
        except OSError as e:
            # The repo item is already accepted + pruned; crashing here leaves
            # the same seg inconsistency PLUS a 500. Report so the operator can
            # fix perms and re-run the relabel.
            relabel_error = str(e)
            log(f"[accept] {uid}: segmentations relabel failed ({e}) - reported")
    else:
        log(f"[accept] {uid}: no export_dir resolved (project_id={pid!r}); "
            f"manifest accepted but segmentations not relabeled")

    # Provenance ledger refresh for the relabeled tree so the coverage matrix
    # shows found_expert immediately (best-effort, never fails the accept).
    refreshed = 0
    if relabeled:
        refreshed = _refresh_provenance_ledgers(
            touched_seg, reviewer=labeler, log_fn=log)

    msg = f"accept {uid} -> {code} ({basis})"
    pushed = repo.commit_push(msg, push=git_push)
    return {'ok': True, 'uid': uid, 'code': code, 'accepted': accepted,
            'relabeled_seg': relabeled, 'relabel_error': relabel_error,
            'export_dir': export_dir,
            'provenance_refreshed': refreshed, 'pushed': pushed}


def consensus_rows(review_dir, repo_url=None, master_codes=None, library_dir=None,
                   site_codes=None, log_fn=None):
    """Build the consensus-builder table (C6): one row per pending uid with its
    reviews[] + computed status/suggested code. Pure read; no mutation."""
    repo = ReviewRepo(review_dir, remote_url=repo_url or '',
                      master_codes_path=master_codes, site_codes_path=site_codes,
                      log_fn=log_fn)
    by_code = codes_mod.load_codes(master_codes)['by_code']
    out = []
    for it in repo.load_manifest().get('items', []):
        uid = it.get('uid')
        reviews = it.get('reviews') or []
        cls = classify_reviews(reviews)
        out.append({
            'uid': uid,
            'project_id': it.get('project_id', ''),
            'project_name': it.get('project_name', '') or it.get('project_id', ''),
            'site': it.get('site_full') or it.get('site', ''),
            'frame': it.get('frame', ''),
            'source_image': it.get('source_image', ''),
            'crop': it.get('crop', ''), 'mask': it.get('mask', ''),
            'full': it.get('full', ''),
            'reviews': reviews,
            'status': cls['status'], 'suggested': cls['code'],
            'codes': cls['codes'], 'n_reviewers': cls['n_reviewers'],
            'name': by_code.get(cls['code'], {}).get('name', ''),
        })
    # Conflicts + consensus first (they need operator attention), then by uid.
    order = {'conflict': 0, 'consensus': 1, 'single': 2, 'none': 3}
    out.sort(key=lambda r: (order.get(r['status'], 9), r['uid'] or ''))
    return out


# ── back-compat aliases ────────────────────────────────────────────────
# The standalone tool's compatibility shim
# (TCRMPclip_addExpertIDs/src/importer.py) re-exports these names; keep them
# importable after the rolling-ingest refactor renamed the internals.
_relabel_in_segmentations = _relabel_accepted_in_segmentations
_overlap_relabel_pass = _overlap_tentative_pass
