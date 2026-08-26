"""
The expert-review GitHub-Pages repo.

Backed by a local working tree (default /mnt/tear/REVIEW_reefpointseg) that is a
git clone of the standalone public repo laurenkolinger/reefpointseg-review with
Pages enabled. The emailed reviewer link points at the Pages site, which is the
static viewer in viewer/ rendering review_manifest.json + codes.json.

Lifecycle:
  ensure_repo()   git init (if needed) + remote + .nojekyll + viewer assets
  add_item()      copy a flagged mask's crop/mask/full into items/<uid>/ and
                  record it in the manifest (one living site, all pending UIDs)
  remove_item()   drop a UID once an expert ID has been imported
  write_manifest()/commit_push()  rewrite the manifest and push (best-effort)

All git calls are best-effort: a slow/unreachable remote or missing gh token
logs a warning and never blocks the caller's export/import.

Idempotency contract: ensure_repo() and every manifest/codes write compares
content BEFORE writing (ignoring the volatile generated_at stamp) and skips
the write when nothing semantic changed. A no-op preamble (ensure_repo +
set_contacts with unchanged contacts, e.g. an export run that queues zero
masks) must leave a clean working tree byte-identical: no dirty
codes.json/review_manifest.json, no README/viewer/.git/config churn.
generated_at therefore moves only when the content it stamps moved.
"""

import filecmp
import json
import os
import shutil
import glob
import subprocess
from contextlib import contextmanager

from . import codes as codes_mod
from .uid import is_safe_uid
from .clock import now_ast_iso

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_VIEWER_DIR = os.path.join(_PKG_DIR, 'viewer')
_VIEWER_FILES = ['index.html', 'viewer.js', 'viewer.css']

MANIFEST_NAME = 'review_manifest.json'
CODES_NAME = 'codes.json'
# Operator setup sidecar written by the Expert IDs panel
# (importer.save_operator_setup): {email, candidate_codes}. write_codes()
# honors its candidate_codes so a re-seed can never drop the operator's extras.
OPERATOR_SETUP_NAME = 'operator_setup.json'
GIT = '/usr/bin/git' if os.path.exists('/usr/bin/git') else 'git'


def _log(log_fn, msg):
    (log_fn or (lambda _m: None))(msg)


def _atomic_write(path, text):
    """Write text to path via a temp file + os.replace so a concurrent reader
    (or a racing `git add -A`) never sees a half-written file."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def _json_semantics_match(path, obj, ignore=('generated_at',)):
    """True when the JSON file at path already carries obj's content, ignoring
    the volatile keys in `ignore` (timestamps). Used to make the repo seeds
    idempotent: a no-op ensure_repo()/set_contacts() preamble must NEVER dirty
    a clean working tree with pure generated_at churn. Any read/parse failure
    returns False so the caller rewrites (self-healing on corrupt files)."""
    try:
        with open(path) as f:
            cur = json.load(f)
    except Exception:
        return False
    if not isinstance(cur, dict):
        return False
    try:
        # Round-trip through JSON so the in-memory obj compares in its
        # serialized form (tuples -> lists etc.), same as what a write yields.
        want = json.loads(json.dumps(obj))
    except Exception:
        return False
    a = {k: v for k, v in cur.items() if k not in ignore}
    b = {k: v for k, v in want.items() if k not in ignore}
    return a == b


def _run(args, cwd=None, log_fn=None, timeout=60):
    """Run a subprocess, return (ok, stdout). Never raises."""
    try:
        r = subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            _log(log_fn, f"[review_repo] {' '.join(args[:2])} -> rc={r.returncode}: "
                         f"{(r.stderr or r.stdout).strip()[:300]}")
            return False, (r.stdout or '')
        return True, (r.stdout or '')
    except Exception as e:
        _log(log_fn, f"[review_repo] {' '.join(args[:2])} failed: {e}")
        return False, ''


def _token_from_hosts_yml(paths):
    """Best-effort oauth_token from a gh hosts.yml. Naive line parse on purpose:
    no yaml dependency, and the file is a flat two-level mapping."""
    for path in paths:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('oauth_token:'):
                        tok = line.split(':', 1)[1].strip().strip('"\'')
                        if tok:
                            return tok
        except OSError:
            continue
    return ''


def _gh_token(log_fn=None):
    # GH_TOKEN/GITHUB_TOKEN env (gh CLI convention) beats a subprocess call.
    for var in ('GH_TOKEN', 'GITHUB_TOKEN'):
        tok = os.environ.get(var, '').strip()
        if tok:
            return tok
    try:
        r = subprocess.run(['gh', 'auth', 'token'], capture_output=True,
                           text=True, timeout=15)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception as e:
        _log(log_fn, f"[review_repo] gh auth token failed: {e}")
    # snap gh cannot exec in capability-restricted process trees (snap-confine
    # needs cap_dac_override), so fall back to reading its stored hosts.yml.
    home = os.path.expanduser('~')
    candidates = [os.path.join(home, '.config', 'gh', 'hosts.yml'),
                  os.path.join(home, 'snap', 'gh', 'current', '.config', 'gh', 'hosts.yml')]
    candidates += sorted(glob.glob(os.path.join(home, 'snap', 'gh', '*', '.config', 'gh', 'hosts.yml')), reverse=True)
    tok = _token_from_hosts_yml(candidates)
    if tok:
        return tok
    _log(log_fn, "[review_repo] no gh token (env, gh CLI, hosts.yml all empty)")
    return ''


def _token_url(remote_url, token):
    """https://github.com/o/r.git -> https://x-access-token:TOK@github.com/o/r.git"""
    if not remote_url or not token:
        return remote_url
    if remote_url.startswith('https://'):
        return 'https://x-access-token:' + token + '@' + remote_url[len('https://'):]
    return remote_url


class ReviewRepo:
    def __init__(self, root, remote_url='', master_codes_path=None,
                 site_codes_path=None, log_fn=None):
        self.root = os.path.abspath(root)
        self.remote_url = remote_url or ''
        self.master_codes_path = master_codes_path
        self.site_codes_path = site_codes_path
        self.log_fn = log_fn
        self.items_dir = os.path.join(self.root, 'items')
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)
        self._lock_path = os.path.join(self.root, '.reefreview.lock')
        self._lock_depth = 0   # reentrancy guard for the in-process holder

    # ── cross-process lock (step5 export + importer share this repo) ──
    @contextmanager
    def _locked(self):
        """Serialize manifest writes + git index ops so a concurrent step5
        export and importer can't interleave and corrupt the commit."""
        os.makedirs(self.root, exist_ok=True)
        if not _HAVE_FCNTL:
            yield
            return
        # Reentrant within one instance: ensure_repo()/add_item() nested inside an
        # already-held lock must not re-block on the same flock fd. Cross-PROCESS
        # exclusion is still enforced by fcntl on the shared lock file.
        if self._lock_depth > 0:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        fh = open(self._lock_path, 'w')
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            self._lock_depth = 1
            yield
        finally:
            self._lock_depth = 0
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def _scrub_remote(self):
        """Ensure origin is the CLEAN (token-free) URL. Defensive: guarantees no
        gh token ever lingers in .git/config even if a prior push crashed."""
        if not self.remote_url:
            return
        ok, cur = _run([GIT, 'remote', 'get-url', 'origin'], cwd=self.root)
        if not ok:
            _run([GIT, 'remote', 'add', 'origin', self.remote_url],
                 cwd=self.root, log_fn=self.log_fn)
        elif cur.strip() != self.remote_url:
            _run([GIT, 'remote', 'set-url', 'origin', self.remote_url],
                 cwd=self.root, log_fn=self.log_fn)

    # ── setup ────────────────────────────────────────────────────────
    def ensure_repo(self):
        # All mutating file seeds run under the lock so a concurrent
        # commit_push's `git add -A` (or the other process's ensure_repo) can't
        # observe a half-written codes.json / clobber a populated manifest.
        with self._locked():
            os.makedirs(self.items_dir, exist_ok=True)
            is_repo = os.path.isdir(os.path.join(self.root, '.git'))
            if not is_repo:
                _run([GIT, 'init'], cwd=self.root, log_fn=self.log_fn)
                _run([GIT, 'symbolic-ref', 'HEAD', 'refs/heads/main'],
                     cwd=self.root, log_fn=self.log_fn)
            # Identity (local) so commits never fail in a fresh tree. Probe
            # first: `git config <key> <val>` rewrites .git/config even when
            # the value is unchanged, and a no-op ensure_repo() must not touch
            # the production repo's files.
            for key, want in (('user.name', 'reef_point_seg'),
                              ('user.email', 'reefpointseg@vicar.local')):
                ok, cur = _run([GIT, 'config', key], cwd=self.root)
                if not ok or cur.strip() != want:
                    _run([GIT, 'config', key, want], cwd=self.root, log_fn=self.log_fn)
            self._scrub_remote()
            # Static assets.
            nojekyll = os.path.join(self.root, '.nojekyll')
            if not os.path.exists(nojekyll):
                open(nojekyll, 'w').close()
            # Never commit the local lock/temp files to the public Pages repo.
            # Re-assert unconditionally so a tree that predates this logic (no
            # .gitignore) is healed before any `git add -A` can stage the lock.
            gi = os.path.join(self.root, '.gitignore')
            want = '.reefreview.lock\n*.tmp\n'
            if not os.path.exists(gi) or '.reefreview.lock' not in open(gi).read():
                _atomic_write(gi, want)
            self._copy_viewer()
            self._write_readme()
            # Seed codes.json always (atomic); seed an empty manifest only if
            # absent — never re-seed an existing one (would clobber a populated
            # manifest a concurrent writer just produced).
            self.write_codes()
            if not os.path.exists(self.manifest_path):
                self._write_manifest_obj({'generated_at': now_ast_iso(),
                                          'contacts': [], 'reviewer_names': [],
                                          'count': 0, 'items': []})

    def _copy_viewer(self):
        if not os.path.isdir(_VIEWER_DIR):
            return
        for fn in _VIEWER_FILES:
            src = os.path.join(_VIEWER_DIR, fn)
            if os.path.exists(src):
                dst = os.path.join(self.root, fn)
                try:
                    if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
                        continue   # identical asset already in place: no churn
                except OSError:
                    pass
                tmp = dst + '.tmp'
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)   # atomic: never a half-copied asset for `git add -A`

    def _write_readme(self):
        path = os.path.join(self.root, 'README.md')
        text = (
            "# TCRMP reef — expert review\n\n"
            "Static viewer for taxonomic expert review of benthic/reef "
            "segmentation masks (coral, algae, sponge, gorgonian, substrate, "
            "and more) flagged REVIEW. Open the GitHub Pages site, label each "
            "mask, export the CSV, and email it back per the on-page "
            "instructions. Generated by the reef_point_seg pipeline; do not "
            "edit by hand.\n")
        try:
            with open(path) as f:
                if f.read() == text:
                    return   # unchanged: don't rewrite (mtime/inode churn)
        except OSError:
            pass
        _atomic_write(path, text)

    def _operator_candidate_extras(self):
        """Candidate codes the operator saved via the Expert IDs setup sidecar
        (operator_setup.json, written by importer.save_operator_setup). Read
        here so write_codes() regeneration can never drop them. [] on any
        missing/corrupt sidecar."""
        try:
            with open(os.path.join(self.root, OPERATOR_SETUP_NAME)) as f:
                obj = json.load(f)
            return [c for c in (obj.get('candidate_codes') or []) if c]
        except Exception:
            return []

    def write_codes(self):
        data = codes_mod.load_codes(self.master_codes_path)
        # candidate codes the operator wants surfaced (CONTRACTS §2): the
        # master-codes 'candidate' flag harvest, unioned with the operator's
        # sidecar extras so an ensure_repo() re-seed preserves them.
        cands = self._candidate_codes(data)
        for c in self._operator_candidate_extras():
            if c not in cands:
                cands.append(c)
        slim = {
            'generated_at': now_ast_iso(),
            'codes': data['codes'],
            'groups': data['groups'],
            'confidence': data['confidence'],
            'default_confidence': data['default_confidence'],
            'idk': data['idk'],
            'something_else': data['something_else'],
            # site_code -> full name (CONTRACTS §7); viewer maps for display.
            'sites': codes_mod.load_sites(self.site_codes_path),
            'candidate_codes': cands,
        }
        path = os.path.join(self.root, CODES_NAME)
        # Idempotent: identical content (ignoring generated_at) means the seed
        # is already in place — never dirty a clean tree with timestamp churn.
        if _json_semantics_match(path, slim):
            return
        _atomic_write(path, json.dumps(slim, indent=1))

    @staticmethod
    def _candidate_codes(data):
        """A list of codes flagged candidate in the master codes, else []. The
        canonical master_codes.csv has no such column today, so this is normally
        empty; supports a future 'candidate' truthy column without a schema bump."""
        out = []
        for entry in data.get('codes', []):
            flag = entry.get('candidate')
            if isinstance(flag, str):
                flag = flag.strip().lower() in ('1', 'true', 'yes', 'y', 'candidate')
            if flag:
                out.append(entry.get('code'))
        return [c for c in out if c]

    # ── manifest ─────────────────────────────────────────────────────
    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return {'generated_at': now_ast_iso(), 'contacts': [],
                    'reviewer_names': [], 'count': 0, 'items': []}
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except Exception:
            return {'generated_at': now_ast_iso(), 'contacts': [],
                    'reviewer_names': [], 'count': 0, 'items': []}

    def _write_manifest_obj(self, obj):
        obj['count'] = len(obj.get('items', []))
        # Idempotent: when nothing but generated_at would change (a no-op
        # set_contacts()/remove_project() on an unknown project, or the
        # export_flagged_masks preamble with nothing new), keep the file — and
        # its generated_at — untouched so the working tree stays clean.
        if _json_semantics_match(self.manifest_path, obj):
            return
        obj['generated_at'] = now_ast_iso()
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, self.manifest_path)

    def set_contacts(self, contacts):
        with self._locked():
            obj = self.load_manifest()
            obj['contacts'] = [c for c in (contacts or []) if c]
            self._write_manifest_obj(obj)

    def add_item(self, uid, item_meta, crop_src=None, mask_src=None, full_src=None):
        """Copy imagery into items/<uid>/ and upsert the manifest entry."""
        if not is_safe_uid(uid):
            _log(self.log_fn, f"[review_repo] refusing unsafe UID for add_item: {uid!r}")
            return None
        self.ensure_repo()
        with self._locked():
            dest = os.path.join(self.items_dir, uid)
            os.makedirs(dest, exist_ok=True)
            rel = {}
            for src, name, key in ((crop_src, 'crop.jpg', 'crop'),
                                   (mask_src, 'mask.png', 'mask'),
                                   (full_src, 'full.jpg', 'full')):
                if src and os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest, name))
                    rel[key] = f'items/{uid}/{name}'
            entry = dict(item_meta)
            entry['uid'] = uid
            entry.update(rel)
            # Per-reviewer accumulation + final acceptance start empty
            # (CONTRACTS §2). A re-add (refreshed imagery) carries forward any
            # tentative reviews/accepted already collected for this UID.
            obj = self.load_manifest()
            prior = next((it for it in obj.get('items', [])
                          if it.get('uid') == uid), None)
            entry.setdefault('reviews', (prior or {}).get('reviews', []) or [])
            entry.setdefault('accepted', (prior or {}).get('accepted', None))
            items = [it for it in obj.get('items', []) if it.get('uid') != uid]
            items.append(entry)
            obj['items'] = items
            self._write_manifest_obj(obj)
        return entry

    def remove_item(self, uid):
        """Drop a UID from the manifest and delete its items/<uid>/ folder."""
        if not is_safe_uid(uid):
            _log(self.log_fn, f"[review_repo] refusing unsafe UID for remove_item: {uid!r}")
            return 0
        with self._locked():
            obj = self.load_manifest()
            before = len(obj.get('items', []))
            obj['items'] = [it for it in obj.get('items', []) if it.get('uid') != uid]
            self._write_manifest_obj(obj)
            dest = os.path.join(self.items_dir, uid)
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
            return before - len(obj['items'])

    def remove_project(self, project_id):
        """Drop EVERY manifest item whose project_id matches and delete each
        items/<uid>/ folder, all in ONE locked pass (single manifest write,
        count recomputed). reviewer_names, contacts, and the cross-project
        library are untouched. Unknown project -> {'removed': 0, 'uids': []}."""
        pid = (project_id or '').strip()
        with self._locked():
            obj = self.load_manifest()
            keep, dropped = [], []
            for it in obj.get('items', []):
                if (it.get('project_id') or '') == pid:
                    dropped.append(it)
                else:
                    keep.append(it)
            obj['items'] = keep
            self._write_manifest_obj(obj)
            uids = [it.get('uid') for it in dropped if it.get('uid')]
            for uid in uids:
                if not is_safe_uid(uid):
                    _log(self.log_fn,
                         f"[review_repo] remove_project skipping unsafe UID: {uid!r}")
                    continue
                dest = os.path.join(self.items_dir, uid)
                if os.path.isdir(dest):
                    shutil.rmtree(dest, ignore_errors=True)
            return {'removed': len(dropped), 'uids': uids}

    # ── per-reviewer accumulation + acceptance (CONTRACTS §2,§5,§6) ───
    @staticmethod
    def _track_reviewer_names(obj, reviewer):
        """Add a distinct reviewer to the manifest top-level reviewer_names[]
        (CONTRACTS §2,§8). Mutates obj in place; no-op for blanks/dupes."""
        name = (reviewer or '').strip()
        if not name:
            return
        names = obj.get('reviewer_names') or []
        if name not in names:
            names.append(name)
        obj['reviewer_names'] = names

    def reviewer_names(self):
        """Distinct reviewers seen across the manifest (for acknowledgment)."""
        return list(self.load_manifest().get('reviewer_names') or [])

    def add_review(self, uid, reviewer, code, confidence='', at=None):
        """UPSERT a reviewer's tentative ID into item.reviews[] keyed by reviewer
        (CONTRACTS §2,§5). Rolling: the same reviewer re-dropping their CSV
        overwrites THEIR row only; other reviewers' rows are untouched. Does NOT
        prune and does NOT set accepted/expert_id. Tracks the reviewer into the
        manifest's reviewer_names[]. Returns the updated item, or None if the UID
        is unsafe / not in the manifest."""
        if not is_safe_uid(uid):
            _log(self.log_fn, f"[review_repo] refusing unsafe UID for add_review: {uid!r}")
            return None
        reviewer = (reviewer or '').strip()
        if not reviewer:
            _log(self.log_fn, "[review_repo] add_review requires a non-empty reviewer")
            return None
        row = {'reviewer': reviewer, 'code': code or '',
               'confidence': confidence or '', 'at': at or now_ast_iso()}
        with self._locked():
            obj = self.load_manifest()
            item = next((it for it in obj.get('items', [])
                         if it.get('uid') == uid), None)
            if item is None:
                _log(self.log_fn, f"[review_repo] add_review: uid not in manifest: {uid!r}")
                return None
            reviews = [r for r in (item.get('reviews') or [])
                       if (r.get('reviewer') or '').strip() != reviewer]
            reviews.append(row)
            item['reviews'] = reviews
            self._track_reviewer_names(obj, reviewer)
            self._write_manifest_obj(obj)
        return item

    def accept_item(self, uid, code, labeler='', basis='consensus'):
        """Operator acceptance (CONTRACTS §2,§5,§6): set item.accepted then remove
        the item from the repo. Acceptance is the ONLY removal path. Returns the
        accepted dict (so the caller can mirror it to the library/segmentations),
        or None if the UID is unsafe / not present."""
        if not is_safe_uid(uid):
            _log(self.log_fn, f"[review_repo] refusing unsafe UID for accept_item: {uid!r}")
            return None
        accepted = {'code': code or '', 'mode': 'EXPERT',
                    'labeler': labeler or '', 'at': now_ast_iso(),
                    'basis': basis or 'consensus'}
        with self._locked():
            obj = self.load_manifest()
            item = next((it for it in obj.get('items', [])
                         if it.get('uid') == uid), None)
            if item is None:
                _log(self.log_fn, f"[review_repo] accept_item: uid not in manifest: {uid!r}")
                return None
            item['accepted'] = accepted
            self._write_manifest_obj(obj)
        # accept is the ONLY removal (CONTRACTS §2): drop from the live site.
        self.remove_item(uid)
        return accepted

    def pending_by_project(self):
        """Sorted [{project_id, project_name, count}] buckets of the pending
        manifest items (CONTRACTS §2). Ordered by project_name then project_id."""
        buckets = {}
        for it in self.load_manifest().get('items', []):
            pid = it.get('project_id') or ''
            pname = it.get('project_name') or pid
            b = buckets.get(pid)
            if b is None:
                buckets[pid] = {'project_id': pid, 'project_name': pname, 'count': 1}
            else:
                b['count'] += 1
        return sorted(buckets.values(),
                      key=lambda b: ((b['project_name'] or '').lower(), b['project_id']))

    def pending_uids(self):
        return [it.get('uid') for it in self.load_manifest().get('items', [])]

    def remote_pending_uids(self, fetch=True):
        """Best-effort: the set of UIDs currently published on origin/main.

        Returns (uids:set | None, source:str). None => the remote state is
        unknown (no remote configured, or fetch failed and there is no local
        tracking ref to fall back on). All git calls are best-effort and never
        raise — a slow/blocked remote just yields None."""
        if not self.remote_url:
            return None, 'no-remote'
        if fetch:
            token = _gh_token(self.log_fn)
            url = _token_url(self.remote_url, token) if token else self.remote_url
            # Update FETCH_HEAD without touching the working tree. Pushes use a
            # bare URL (not the 'origin' remote name), so the origin/main
            # tracking ref can lag; FETCH_HEAD is the freshest baseline we have.
            _run([GIT, 'fetch', url, 'main'], cwd=self.root,
                 log_fn=self.log_fn, timeout=60)
        for ref in ('FETCH_HEAD', 'refs/remotes/origin/main', 'origin/main'):
            ok, out = _run([GIT, 'show', f'{ref}:{MANIFEST_NAME}'], cwd=self.root)
            if ok and out.strip():
                try:
                    obj = json.loads(out)
                    uids = {it.get('uid') for it in obj.get('items', [])
                            if it.get('uid')}
                    return uids, ref
                except Exception:
                    pass
        return None, 'unavailable'

    def push_preview(self, fetch=True):
        """Summarize what an 'update site + push' would publish, WITHOUT pushing.

        Compares the local working-tree manifest against what's on the remote:
          additions   - UIDs pending locally but not yet on the published site
          completions - UIDs on the site but no longer pending locally (resolved)
        Also reports the count of uncommitted working-tree changes so the caller
        can tell there's something to push even when the pending set is
        unchanged (e.g. refreshed imagery or codes.json)."""
        self.ensure_repo()
        current = {it.get('uid') for it in self.load_manifest().get('items', [])
                   if it.get('uid')}
        remote, source = self.remote_pending_uids(fetch=fetch)
        ok, status = _run([GIT, 'status', '--porcelain'], cwd=self.root)
        dirty = [ln for ln in (status or '').splitlines() if ln.strip()]
        if remote is None:
            additions, completions, remote_known = sorted(current), [], False
        else:
            additions = sorted(current - remote)
            completions = sorted(remote - current)
            remote_known = True
        return {
            'pending_total': len(current),
            'additions': additions,
            'completions': completions,
            'remote_known': remote_known,
            'remote_source': source,
            'dirty_count': len(dirty),
            'has_changes': bool(additions or completions or dirty),
        }

    # ── git push (best-effort) ───────────────────────────────────────
    def commit_push(self, message, push=True):
        """git add -A + commit; push to origin main with a gh token. Returns
        True if the push succeeded (commit-only success returns False).

        The token is used ONLY as an ephemeral push-URL argument and is never
        written to .git/config (origin stays the clean URL, re-asserted here)."""
        if not os.path.isdir(os.path.join(self.root, '.git')):
            self.ensure_repo()
        # The INDEX work (scrub remote, add, commit) is the only part that races
        # the importer/export, so it runs under the lock. The network push only
        # reads committed history (HEAD:main) — hold it OUTSIDE the lock so a
        # slow/unreachable remote (lab DPI, 180s timeout) can't block a
        # concurrent add_item/remove_item for the whole push duration.
        with self._locked():
            if push and self.remote_url:
                # Origin maintenance belongs to the PUSH path only. A local-only
                # commit (push=False) must never create or rewrite the working
                # tree's origin: a delete_project/import/accept request that
                # pointed review_dir at a caller-supplied directory while
                # suppressing the push used to leave that directory branded with
                # the production remote URL (hammering-pass finding, 2026-07-09).
                # ensure_repo still seeds origin when it initializes a tree, and
                # every pushing commit re-asserts the clean token-free URL here.
                self._scrub_remote()
            _run([GIT, 'add', '-A'], cwd=self.root, log_fn=self.log_fn)
            _run([GIT, 'commit', '-m', message], cwd=self.root, log_fn=self.log_fn)
        if not push or not self.remote_url:
            return False
        token = _gh_token(self.log_fn)
        if not token:
            _log(self.log_fn, "[review_repo] no gh token; skipped push")
            return False
        url = _token_url(self.remote_url, token)
        ok, _ = _run([GIT, 'push', url, 'HEAD:main'], cwd=self.root,
                     log_fn=self.log_fn, timeout=180)
        if ok:
            _log(self.log_fn, "[review_repo] pushed to origin/main")
        return ok
