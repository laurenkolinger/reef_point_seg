"""
Add-Expert-IDs as a self-contained Flask blueprint.

One shared implementation mounted by BOTH the standalone tool
(TCRMPclip_addExpertIDs) and natively by the reef_point_seg orchestrator — no
iframe, no subprocess, no duplicated logic. The host supplies a `paths_provider`
callable that returns the current effective paths; everything else (routes, the
UI fragment, its CSS/JS) lives here.

    from _expertids import make_blueprint
    app.register_blueprint(make_blueprint(paths_provider), url_prefix='/expertids')

paths_provider() -> dict with keys:
    export_dir, review_dir, library_dir, review_repo_url, master_codes,
    overlap_thresh, git_push

The routes are stateless: the UI sends the path fields it is showing with each
action (and they fall back to the provider's defaults), so a host never has to
keep server-side session state in sync with the form. Mutating routes also
accept per-request REMOTE overrides, review_repo_url ('' = no remote) and
git_push/push, so a caller that redirects review_dir to a scratch tree is
fully isolated from the provider's production Pages repo (CONTRACTS §5).
"""

import os
import re
import sys
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file, abort, render_template

# scripts/ on path for the shared review package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _reefreview.review_repo import ReviewRepo
from _reefreview.library import Library
from _reefreview.uid import is_safe_uid

from . import importer

_PKG = os.path.dirname(os.path.abspath(__file__))


def pages_url(repo_url):
    """github.com/owner/repo(.git) -> https://owner.github.io/repo/ (the public
    Pages site the expert browses). Empty string if it can't be parsed."""
    m = re.search(r'github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$', repo_url or '')
    if not m:
        return ''
    return f'https://{m.group(1).lower()}.github.io/{m.group(2)}/'


def _parse_push_flag(raw, default):
    """Tolerant boolean for a per-request push override (delete_project's
    long-standing parse): None/'' -> default, strings match 1/true/yes/y/on,
    anything else through bool()."""
    if raw is None or raw == '':
        return bool(default)
    if isinstance(raw, str):
        return raw.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(raw)


def _remote_overrides(src, p):
    """Effective (repo_url, git_push) for a mutating route (CONTRACTS §5).

    Presence-based per-request overrides falling back to the provider, so a
    caller that redirects review_dir to a scratch tree can ALSO detach the git
    remote: an EXPLICIT review_repo_url of '' means "no remote", and
    git_push/push false suppresses the push. Without these, a path-overridden
    request would still git-init the override dir with the PRODUCTION remote
    and push to it. Absent keys keep production behavior (provider repo_url,
    provider git_push). `src` is the request's form/json mapping."""
    if 'review_repo_url' in src:
        repo_url = (src.get('review_repo_url') or '').strip()
    elif 'repo_url' in src:
        repo_url = (src.get('repo_url') or '').strip()
    else:
        repo_url = p.get('review_repo_url', '')
    raw = src.get('git_push')
    if raw is None or raw == '':
        raw = src.get('push')
    return repo_url, _parse_push_flag(raw, p.get('git_push', True))


def make_blueprint(paths_provider, *, name='expertids', log_fn=None):
    log = log_fn or (lambda m: print(f"[expertids] {m}"))
    bp = Blueprint(name, __name__,
                   template_folder=os.path.join(_PKG, 'templates'),
                   static_folder=os.path.join(_PKG, 'static'),
                   static_url_path='static')
    state = {'history': []}   # per-process import history (matches old behavior)

    def P():
        try:
            return paths_provider() or {}
        except Exception as e:
            log(f"paths_provider failed: {e}")
            return {}

    def _repo(review_dir=None, repo_url=None):
        p = P()
        return ReviewRepo(review_dir or p.get('review_dir', ''),
                          remote_url=(p.get('review_repo_url', '')
                                      if repo_url is None else repo_url),
                          master_codes_path=p.get('master_codes'),
                          log_fn=log)

    def _lib(library_dir=None):
        p = P()
        chosen = library_dir if library_dir is not None else p.get('library_dir')
        return Library(chosen or None)

    def _projects_root():
        """The `<module>/inprocess` projects root (CONTRACTS §1,§5). Prefer an
        explicit provider value; otherwise compute it robustly from the package
        (mirrors library.default_dir) so routing works regardless of CWD."""
        p = P()
        return p.get('projects_root') or importer.default_projects_root()

    # ── Standalone index (so the blueprint renders GET / on its own) ──
    # When the host includes panel.html itself it never hits this; it exists so
    # the blueprint is independently smoke-testable (CONTRACTS §9 — GET / 200).
    @bp.route('/')
    def index():
        # url_for the static dir gives the blueprint's mount-aware base
        # (e.g. /expertids/static); strip the trailing /static for EXPERTIDS_BASE.
        from flask import url_for
        static_base = url_for(f'{name}.static', filename='_')
        base = static_base.rsplit('/static/', 1)[0]
        return render_template('expertids/index.html', base=base)

    # ── Mask catalog browser (preview tiles) ─────────────────────────
    @bp.route('/api/catalog')
    def api_catalog():
        lib = _lib(request.args.get('lib') or None)
        records = lib.load()
        facet = {'code': set(), 'mode': set(), 'site': set(), 'year': set()}
        rows = []
        for rec in records.values():
            facet['code'].add((rec.get('code') or '').strip())
            facet['mode'].add((rec.get('mode') or '').strip())
            facet['site'].add((rec.get('site') or '').strip())
            facet['year'].add(str(rec.get('year') or '').strip())
            rows.append(rec)

        code = (request.args.get('code') or '').strip()
        mode = (request.args.get('mode') or '').strip()
        site = (request.args.get('site') or '').strip()
        year = (request.args.get('year') or '').strip()
        q = (request.args.get('q') or '').strip().lower()

        def keep(r):
            if code and (r.get('code') or '').strip() != code: return False
            if mode and (r.get('mode') or '').strip() != mode: return False
            if site and (r.get('site') or '').strip() != site: return False
            if year and str(r.get('year') or '').strip() != year: return False
            if q and q not in (r.get('uid', '') + ' ' + r.get('source_image', '')).lower(): return False
            return True

        # Drop unsafe UIDs BEFORE counting/paging so total == sum over pages.
        filtered = [r for r in rows if keep(r) and is_safe_uid(r.get('uid', ''))]
        filtered.sort(key=lambda r: (r.get('updated_at') or ''), reverse=True)
        total = len(filtered)
        try:
            limit = max(1, min(1000, int(request.args.get('limit', 200))))
            offset = max(0, int(request.args.get('offset', 0)))
        except ValueError:
            limit, offset = 200, 0
        page = filtered[offset:offset + limit]

        tiles = []
        for r in page:
            uid = r.get('uid', '')
            geom = lib.load_polygon(uid) or {}
            tiles.append({
                'uid': uid, 'code': r.get('code', ''), 'name': r.get('name', ''),
                'category': r.get('category', ''), 'mode': r.get('mode', ''),
                'confidence': r.get('confidence', ''), 'labeler': r.get('labeler', ''),
                'site': r.get('site', ''), 'transect': r.get('transect', ''),
                'frame': r.get('frame', ''), 'year': r.get('year', ''),
                'source_image': r.get('source_image', ''), 'project_id': r.get('project_id', ''),
                'area': geom.get('area'), 'bbox': geom.get('bbox'),
                'has_image': os.path.exists(lib.image_path(uid)),
                'has_overlay': os.path.exists(lib.preview_mask_path(uid)),
                'updated_at': r.get('updated_at', ''),
            })

        return jsonify({'total': total, 'offset': offset, 'limit': limit,
                        'count': len(tiles), 'tiles': tiles,
                        'facets': {k: sorted(v - {''}) for k, v in facet.items()}})

    def _catalog_send(uid, which):
        if not is_safe_uid(uid):
            abort(404)
        lib = _lib(request.args.get('lib') or None)
        path = {'image': lib.image_path, 'overlay': lib.preview_mask_path,
                'mask': lib.mask_path}.get(which, lib.image_path)(uid)
        if not os.path.exists(path):
            abort(404)
        return send_file(path)

    @bp.route('/catalog/<uid>/image.jpg')
    def catalog_image(uid):
        return _catalog_send(uid, 'image')

    @bp.route('/catalog/<uid>/overlay.png')
    def catalog_overlay(uid):
        return _catalog_send(uid, 'overlay')

    # ── Status (counts + the path defaults the UI populates from) ────
    @bp.route('/api/status')
    def status():
        review_dir = request.args.get('review_dir') or None
        library_dir = request.args.get('lib') or None
        p = P()
        try:
            pending = _repo(review_dir).pending_uids()
        except Exception as e:
            pending = []
            log(f"status: repo unavailable: {e}")
        try:
            records = _lib(library_dir).load()
            n_expert = sum(1 for r in records.values() if r.get('mode') == 'EXPERT')
        except Exception:
            records, n_expert = {}, 0
        return jsonify({
            'export_dir': p.get('export_dir', ''),
            'review_dir': p.get('review_dir', ''),
            'library_dir': p.get('library_dir', '') or '',
            'review_site_url': pages_url(p.get('review_repo_url', '')),
            'pending_review': len(pending),
            'library_total': len(records),
            'library_expert': n_expert,
            'history': state['history'],
        })

    # ── Import an expert CSV (ROLLING ingest, CONTRACTS §4,§5 / C5) ───
    @bp.route('/api/import', methods=['POST'])
    def do_import():
        """Accept CSV as an uploaded file or as JSON {csv_text}, with optional
        path overrides (export_dir / review_dir / library_dir / reviewer) and
        remote overrides (review_repo_url / git_push; CONTRACTS §5: a
        review_dir override alone must not couple the caller to the provider's
        production remote).

        Rolling: groups rows by their project_id, resolves each project's step-5
        dir, UPSERTs each row into the item's reviews[] (manifest + library), runs
        the per-project overlap-tentative cascade, and pushes. Never prunes, never
        sets expert_id (that is acceptance's job). Every unrouted UID is reported."""
        p = P()
        csv_text = reviewer = ''
        ovr, src = {}, {}
        if request.files:
            f = next(iter(request.files.values()))
            csv_text = f.read().decode('utf-8', errors='replace')
            src = request.form
            reviewer = ((src.get('reviewer') or '')
                        or (src.get('labeler') or '')).strip()
            ovr = {k: (src.get(k) or '').strip()
                   for k in ('export_dir', 'review_dir', 'library_dir')}
        elif request.is_json:
            src = request.json or {}
            csv_text = src.get('csv_text', '')
            reviewer = ((src.get('reviewer') or '')
                        or (src.get('labeler') or '')).strip()
            ovr = {k: (src.get(k) or '').strip()
                   for k in ('export_dir', 'review_dir', 'library_dir')}
        if not csv_text.strip():
            return jsonify({'error': 'No CSV provided'}), 400
        repo_url, git_push = _remote_overrides(src, p)

        try:
            rows = importer.parse_csv_text(csv_text)
        except Exception as e:
            # A malformed CSV must come back as a JSON 400, never a raw HTML
            # 500 (parse_csv_text is tolerant, but any residual csv.Error from
            # pathological input still needs the JSON error contract).
            log(f"import failed: CSV parse error: {e}")
            return jsonify({'error': f'CSV parse failed: {e}'}), 400
        if not rows:
            return jsonify({'error': 'No usable rows (need a uid,code,reviewer,project_id CSV)'}), 400

        try:
            stats = importer.import_rows(
                rows,
                export_dir=ovr.get('export_dir') or p.get('export_dir', ''),
                review_dir=ovr.get('review_dir') or p.get('review_dir', ''),
                repo_url=repo_url,
                library_dir=(ovr.get('library_dir') or p.get('library_dir')) or None,
                master_codes=p.get('master_codes'),
                projects_root=_projects_root(),
                overlap_thresh=p.get('overlap_thresh', 0.5),
                git_push=git_push,
                default_reviewer=reviewer,
                site_codes=p.get('site_codes'),
                log_fn=log,
            )
        except Exception as e:
            log(f"import failed: {e}")
            return jsonify({'error': str(e)}), 500

        summary = {'at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), **stats}
        summary.pop('uids', None)
        state['history'].insert(0, summary)
        return jsonify({'ok': True, **stats})

    # ── Per-project pending counts (CONTRACTS §2) ────────────────────
    @bp.route('/api/pending_by_project')
    def api_pending_by_project():
        review_dir = request.args.get('review_dir') or None
        try:
            buckets = _repo(review_dir).pending_by_project()
        except Exception as e:
            log(f"pending_by_project failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'ok': True, 'projects': buckets,
                        'total': sum(b['count'] for b in buckets)})

    # ── Remove a whole project's pending items from the review site ──
    @bp.route('/api/delete_project', methods=['POST'])
    def api_delete_project():
        """Drop every pending item for one project_id (manifest entries +
        items/<uid>/ assets) and commit/push. Accepted IDs and the
        cross-project library are untouched.

        Stateless per-request remote isolation (CONTRACTS §5): the body may
        override review_dir plus review_repo_url ('' detaches the remote) and
        git_push/push (false commits locally without pushing). Absent keys keep
        the provider defaults — the panel UI relies on that to push the
        production site after a delete. A local-only commit never creates or
        rewrites the target directory's git origin (commit_push scrubs origin
        only on the push path)."""
        p = P()
        body = (request.json or {}) if request.is_json else request.form
        project_id = (body.get('project_id') or '').strip()
        if not project_id:
            return jsonify({'error': 'project_id required'}), 400
        review_dir = (body.get('review_dir') or '').strip() or None
        repo_url, do_push = _remote_overrides(body, p)
        try:
            repo = _repo(review_dir, repo_url=repo_url)
            res = repo.remove_project(project_id)
            n = res.get('removed', 0)
            pushed = False
            if n:
                pushed = repo.commit_push(
                    f"remove project {project_id} ({n} items) from review site",
                    push=do_push)
            pending_total = len(repo.pending_uids())
        except Exception as e:
            log(f"delete_project failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'ok': True, 'project_id': project_id, 'removed': n,
                        'pushed': bool(pushed), 'pending_total': pending_total})

    # ── Operator setup: contact email + extra candidate codes (C2/C3) ─
    @bp.route('/api/operator_setup', methods=['GET', 'POST'])
    def api_operator_setup():
        p = P()
        review_dir = None
        if request.method == 'GET':
            review_dir = request.args.get('review_dir') or None
            setup = importer.load_operator_setup(review_dir or p.get('review_dir', ''))
            return jsonify({'ok': True, **setup})
        body = (request.json or {}) if request.is_json else request.form
        review_dir = (body.get('review_dir') or '').strip() or None
        repo_url, git_push = _remote_overrides(body, p)
        try:
            setup = importer.save_operator_setup(
                review_dir=review_dir or p.get('review_dir', ''),
                repo_url=repo_url,
                master_codes=p.get('master_codes'),
                email=(body.get('email') or '').strip(),
                candidate_codes=(body.get('candidate_codes') or ''),
                site_codes=p.get('site_codes'),
                git_push=git_push,
                log_fn=log,
            )
        except Exception as e:
            log(f"operator_setup failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'ok': True, **setup})

    # ── Per-project email-request text (C3) ──────────────────────────
    @bp.route('/api/email_requests')
    def api_email_requests():
        """One paste-ready email per project: subject line, greeting, the
        review-site link, numbered steps, and the operator's return address
        (saved operator_setup email, falling back to lauren.olinger@uvi.edu)."""
        p = P()
        review_dir = request.args.get('review_dir') or None
        try:
            repo = _repo(review_dir)
            buckets = repo.pending_by_project()
        except Exception as e:
            log(f"email_requests failed: {e}")
            return jsonify({'error': str(e)}), 500
        url = pages_url(p.get('review_repo_url', ''))
        setup = importer.load_operator_setup(review_dir or p.get('review_dir', ''))
        operator_email = setup.get('email') or 'lauren.olinger@uvi.edu'
        out = []
        for b in buckets:
            name = b['project_name'] or b['project_id'] or 'this project'
            body = (
                f"Subject: TCRMP expert IDs requested: {name}\n"
                "\n"
                "Hello,\n"
                "\n"
                f"Please review the {b['count']} pending mask(s) for project {name}.\n"
                "\n"
                f"1. Open {url or '(review site URL unavailable)'}\n"
                "2. Enter your name.\n"
                f"3. In the Project menu choose \"{name}\".\n"
                "4. Click a code for each mask (IDK is fine).\n"
                f"5. Download the CSV and email it back to {operator_email}.\n"
                "\n"
                "Thank you!"
            )
            out.append({'project_id': b['project_id'], 'project_name': b['project_name'],
                        'count': b['count'], 'text': body})
        return jsonify({'ok': True, 'pages_url': url, 'requests': out})

    # ── Consensus builder: per-uid reviews + status (C6) ─────────────
    @bp.route('/api/consensus')
    def api_consensus():
        p = P()
        review_dir = request.args.get('review_dir') or None
        try:
            rows = importer.consensus_rows(
                review_dir or p.get('review_dir', ''),
                repo_url=p.get('review_repo_url', ''),
                master_codes=p.get('master_codes'),
                library_dir=(request.args.get('lib') or p.get('library_dir')) or None,
                site_codes=p.get('site_codes'),
                log_fn=log)
        except Exception as e:
            log(f"consensus failed: {e}")
            return jsonify({'error': str(e)}), 500
        counts = {'consensus': 0, 'conflict': 0, 'single': 0, 'none': 0}
        for r in rows:
            counts[r['status']] = counts.get(r['status'], 0) + 1
        return jsonify({'ok': True, 'rows': rows, 'counts': counts, 'total': len(rows)})

    @bp.route('/api/consensus_asset/<uid>/<which>')
    def api_consensus_asset(uid, which):
        """Serve a pending item's crop/mask/full thumbnail from the review repo's
        items/<uid>/ dir (the consensus table's preview)."""
        if not is_safe_uid(uid) or which not in ('crop', 'mask', 'full'):
            abort(404)
        review_dir = request.args.get('review_dir') or None
        repo = _repo(review_dir)
        fname = {'crop': 'crop.jpg', 'mask': 'mask.png', 'full': 'full.jpg'}[which]
        path = os.path.join(repo.items_dir, uid, fname)
        if not os.path.exists(path):
            abort(404)
        return send_file(path)

    # ── Accept a final code for a uid (C6 — the ONLY expert_id path) ─
    @bp.route('/api/accept', methods=['POST'])
    def api_accept():
        """Stateless like import (module docstring): the body may override
        review_dir / library_dir / export_dir / projects_root plus the remote
        (review_repo_url / git_push or push), each falling back to the
        provider's default. The acceptance then reads and WRITES the same
        paths the caller is looking at — the consensus table loaded from an
        overridden library must never accept into a different (e.g.
        production) library, and a push:false caller must never push."""
        p = P()
        body = (request.json or {}) if request.is_json else request.form
        uid = (body.get('uid') or '').strip()
        code = (body.get('code') or '').strip()
        review_dir = (body.get('review_dir') or '').strip() or None
        basis = (body.get('basis') or 'consensus').strip() or 'consensus'
        labeler = (body.get('labeler') or '').strip()
        library_dir = (body.get('library_dir') or '').strip()
        export_dir = (body.get('export_dir') or '').strip()
        projects_root = (body.get('projects_root') or '').strip()
        repo_url, git_push = _remote_overrides(body, p)
        if not uid:
            return jsonify({'error': 'uid required'}), 400
        try:
            res = importer.accept_uid(
                uid, code,
                review_dir=review_dir or p.get('review_dir', ''),
                repo_url=repo_url,
                library_dir=(library_dir or p.get('library_dir')) or None,
                master_codes=p.get('master_codes'),
                projects_root=projects_root or _projects_root(),
                labeler=labeler, basis=basis,
                git_push=git_push,
                site_codes=p.get('site_codes'),
                open_export_dir=export_dir or p.get('export_dir', ''),
                log_fn=log)
        except Exception as e:
            log(f"accept failed: {e}")
            return jsonify({'error': str(e)}), 500
        status = 200 if res.get('ok') else 400
        return jsonify(res), status

    # ── Publish: preview, then push ──────────────────────────────────
    def _publish_overrides():
        """review_dir + remote-URL overrides for the publish routes (the same
        stateless per-request contract as import/accept). site_push always
        pushes by design, so the review_repo_url override is the isolation
        switch: '' detaches the repo from the provider's production remote."""
        src = (request.json or {}) if request.is_json else request.form
        review_dir = (src.get('review_dir') or '').strip() or None
        return review_dir, _remote_overrides(src, P())[0]

    @bp.route('/api/site_preview', methods=['POST'])
    def site_preview():
        review_dir, repo_url = _publish_overrides()
        try:
            preview = _repo(review_dir, repo_url=repo_url).push_preview(fetch=True)
        except Exception as e:
            log(f"site_preview failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'ok': True, **preview})

    @bp.route('/api/site_push', methods=['POST'])
    def site_push():
        review_dir, repo_url = _publish_overrides()
        try:
            repo = _repo(review_dir, repo_url=repo_url)
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pushed = repo.commit_push(f"Manual site update {ts}", push=True)
            preview = repo.push_preview(fetch=True)
        except Exception as e:
            log(f"site_push failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'ok': True, 'pushed': bool(pushed), **preview})

    return bp
