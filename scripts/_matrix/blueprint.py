"""
Cross-project image x label matrix as a self-contained Flask blueprint.

One shared implementation the reef_point_seg orchestrator mounts natively
(no iframe, no subprocess). The host supplies a `paths_provider` callable
that returns the current effective paths; everything else (routes, the UI
fragment, its CSS/JS) lives here.

    from _matrix import make_blueprint
    app.register_blueprint(make_blueprint(paths_provider), url_prefix='/matrix')

paths_provider() -> dict with keys:
    inprocess_root     the `<module>/inprocess` projects root that
                       build_matrix() rescans on every /api/data call.
    master_codes_csv   code,category,name CSV resolving label codes to full
                       names in the /api/data and /api/stats label_names map.
    site_codes_csv     site,site_code CSV resolving site codes to full names
                       in the /api/data and /api/stats site_names map.

The data core (builder.build_matrix) is pure-python and rescans the tree
on every call, so a purged project simply stops contributing.
"""

import os
import re
import subprocess

from flask import Blueprint, jsonify, render_template, request, send_file

from .builder import build_matrix, scan_projects
from .cellinfo import cell_info, image_path_for, _basename_stem
from .composite import (SOURCE_TYPES, CompositeExists, RunDirExhausted, build_composite,
                        composite_root, create_training_project, list_composites)
from . import names as _names
from .preferences import Preferences, validate_choice
from .stats import compute_stats, purge_test_rows

_PKG = os.path.dirname(os.path.abspath(__file__))

# Cache-bust for the shared assets (vicarius_workflow.css, vicarius_tooltips.js)
# and this blueprint's own static/ files. Bump on any template or asset edit.
MATRIX_ASSET_VERSION = "20260917-plain"

# Reviewer initials accepted by the routes that record who acted
# (create-project). composite.create_training_project applies the same rule,
# so a direct caller is guarded too.
_INITIALS_RE = re.compile(r"^[A-Za-z]{1,4}$")

# Configured master_codes_csv / site_codes_csv paths _name_maps() has already
# warned about (fix round 1: a misconfigured name table should leave one
# trace in the log, not one per request). Module-level, not per
# make_blueprint() call, so the warning fires once per process even if the
# blueprint is mounted more than once.
_LOGGED_UNREADABLE_NAME_TABLES = set()


def default_open_folder(path):
    """Spawn ``xdg-open <path>`` detached with stderr captured. A launch still
    running after 3 s counts as success (the file manager keeps the process
    alive); a fast non-zero exit surfaces its stderr. Returns (ok, err)."""
    env = dict(os.environ)
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p and p != "/usr/bin"]
    env["PATH"] = os.pathsep.join(["/usr/bin"] + parts)
    try:
        proc = subprocess.Popen(
            ["xdg-open", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
    except Exception as exc:
        return False, str(exc)
    try:
        _out, err = proc.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        return True, ""
    if proc.returncode not in (0, None):
        text = (err or b"").decode("utf-8", "replace").strip()
        return False, text[:400] or f"xdg-open exited with code {proc.returncode}"
    return True, ""


def _json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def make_blueprint(paths_provider, *, name='matrix', log_fn=None, open_folder_fn=None):
    log = log_fn or (lambda m: print(f"[matrix] {m}"))
    open_folder = open_folder_fn or default_open_folder
    bp = Blueprint(name, __name__,
                   template_folder=os.path.join(_PKG, 'templates'),
                   static_folder=os.path.join(_PKG, 'static'),
                   static_url_path='static')

    def P():
        try:
            return paths_provider() or {}
        except Exception as e:
            log(f"paths_provider failed: {e}")
            return {}

    # code -> name / site_code -> site, for the two payloads that show codes
    # (spec 2026-09-02-workflow-page-design.md section 6: never a bare code
    # when the registry can resolve it). Empty dicts when the paths are unset.
    def _name_maps():
        """Resolve the label-code and site-code name maps for the matrix's
        two JSON payloads (/api/data and /api/stats), from whatever paths
        the host's paths_provider() currently reports.

        Parameters:
            None. Reads master_codes_csv and site_codes_csv off P() (the
            live paths_provider() result) on every call.

        Returns:
            tuple[dict[str, str], dict[str, str]]: (label_names, site_names).
            Either dict is {} when its path is unset, or when names.py could
            not resolve any names from it (missing file, unreadable, or
            wrong columns): names.py never raises for those cases, it just
            returns {}. The first time a configured, non-blank path yields
            {}, this logs it once through the blueprint's log function so a
            misconfigured table leaves a trace instead of silently going
            bare-code everywhere; a path already logged is not logged again
            (tracked in the module-level _LOGGED_UNREADABLE_NAME_TABLES).
        """
        p = P()
        master_path = p.get('master_codes_csv', '')
        site_path = p.get('site_codes_csv', '')
        label_names = _names.read_label_names(master_path)
        site_names = _names.read_site_names(site_path)
        for cfg_path, resolved in ((master_path, label_names), (site_path, site_names)):
            if cfg_path and not resolved and cfg_path not in _LOGGED_UNREADABLE_NAME_TABLES:
                log(f"name table unreadable or missing columns: {cfg_path}")
                _LOGGED_UNREADABLE_NAME_TABLES.add(cfg_path)
        return label_names, site_names

    # The grid page. Stays mount-aware via url_for so it works under any
    # url_prefix the host chooses (the orchestrator mounts at /matrix).
    @bp.route('/')
    def index():
        from flask import url_for
        static_base = url_for(f'{name}.static', filename='_')
        base = static_base.rsplit('/static/', 1)[0]
        # The app-level static endpoint (not this blueprint's own), where the
        # host serves the shared vicarius_workflow.css / vicarius_tooltips.js.
        shared = url_for('static', filename='_').rsplit('/_', 1)[0]
        return render_template('matrix/index.html', base=base, shared=shared, v=MATRIX_ASSET_VERSION)

    # The whole matrix as JSON (the contract in builder.build_matrix).
    @bp.route('/api/data')
    def api_data():
        p = P()
        inprocess_root = p.get('inprocess_root', '')
        try:
            data = build_matrix(inprocess_root)
        except Exception as e:
            log(f"build_matrix failed: {e}")
            return jsonify({'error': str(e)}), 500
        label_names, site_names = _name_maps()
        data['label_names'] = label_names
        data['site_names'] = site_names
        return jsonify(data)

    def _root():
        return P().get('inprocess_root', '') or ''

    def _img_base():
        from flask import url_for
        return url_for(f'{name}.img', project_id='_', basename='_').rsplit('/_/_', 1)[0]

    def _find_project(root, project_id):
        pid = str(project_id or '')
        for info in scan_projects(root):
            if info['project_id'] == pid or info['run_dir'] == pid:
                return info
        return None

    def _project_masks(root, image, label, project_id):
        """The cellinfo mask records of one project for the image, or None
        when the cell or project is unknown."""
        info = cell_info(root, image, label, image_url_base=_img_base())
        if info is None:
            return None
        for p in info['projects']:
            if p['project_id'] == project_id:
                return p['masks']
        return None

    # One cell: outcome, sources, preference and every project's masks.
    @bp.route('/api/cell')
    def api_cell():
        image = (request.args.get('image') or '').strip()
        label = (request.args.get('label') or '').strip()
        if not image or not label:
            return jsonify({'error': 'image and label are required'}), 400
        try:
            info = cell_info(_root(), image, label, image_url_base=_img_base())
        except Exception as e:
            log(f"cell_info failed: {e}")
            return jsonify({'error': str(e)}), 500
        if info is None:
            return jsonify({'error': f'unknown image {image!r}', 'image': image, 'label': label}), 404
        return jsonify(info)

    # The image bytes, resolved from that project's segmentations.json only.
    @bp.route('/img/<project_id>/<path:basename>')
    def img(project_id, basename):
        path = image_path_for(_root(), project_id, basename)
        if not path or not os.path.isfile(path):
            return jsonify({'error': 'image not found', 'project_id': project_id,
                            'image': _basename_stem(basename), 'path': path or ''}), 404
        return send_file(path)

    # Open a folder in the host file manager: {"project_id"} opens that
    # project's run_* dir, {"path"} opens a composite export directly under
    # <inprocess_root>/_composite. Both go through the same open_folder_fn
    # and every error body carries the path.
    @bp.route('/api/open-folder', methods=['POST'])
    def api_open_folder():
        body = _json_body()
        project_id = str(body.get('project_id') or '').strip()
        raw_path = str(body.get('path') or '').strip()
        root = os.path.abspath(_root())
        if project_id:
            info = _find_project(root, project_id)
            if info is None:
                return jsonify({'error': f'unknown project {project_id!r}', 'path': ''}), 400
            path = os.path.abspath(info['project_dir'])
            if not os.path.isdir(path):
                return jsonify({'error': f'not a directory: {path}', 'path': path}), 400
            real_root = os.path.realpath(root)
            real_path = os.path.realpath(path)
            if (os.path.dirname(real_path) != real_root
                    or not os.path.basename(real_path).startswith('run_')):
                return jsonify({'error': 'path is not a run_* dir under inprocess', 'path': path}), 403
        elif raw_path:
            path = os.path.abspath(raw_path)
            if not os.path.isdir(path):
                return jsonify({'error': f'not a directory: {path}', 'path': path}), 400
            real_comp = os.path.realpath(composite_root(root))
            real_path = os.path.realpath(path)
            if os.path.dirname(real_path) != real_comp:
                return jsonify({'error': 'path is not a composite dir under inprocess/_composite',
                                'path': path}), 403
        else:
            return jsonify({'error': 'project_id or path is required', 'path': ''}), 400
        try:
            ok, err = open_folder(path)
        except Exception as e:
            ok, err = False, str(e)
        if not ok:
            return jsonify({'error': f'file manager did not open: {err}', 'path': path}), 500
        return jsonify({'success': True, 'path': path})

    # Mask preferences (which project's masks are canonical per cell).
    @bp.route('/api/preferences')
    def api_preferences():
        return jsonify({'preferences': Preferences(_root()).load()})

    @bp.route('/api/prefer', methods=['POST'])
    def api_prefer():
        body = _json_body()
        image = _basename_stem(str(body.get('image') or '').strip())
        label = str(body.get('label') or '').strip()
        project_id = str(body.get('project_id') or '').strip()
        by = str(body.get('by') or '').strip().upper()
        uids = body.get('uids') or []
        if isinstance(uids, str):
            uids = [uids]
        note = str(body.get('note') or '').strip()
        if not image or not label:
            return jsonify({'error': 'image and label are required'}), 400
        if not project_id:
            return jsonify({'error': 'project_id is required'}), 400
        if not by:
            return jsonify({'error': 'reviewer initials (by) are required'}), 400
        root = _root()
        # Resolve the id the way /api/open-folder does so a run_dir name
        # works too; the saved row always carries the project.json id.
        found = _find_project(root, project_id)
        if found is not None:
            project_id = found['project_id']
        masks = _project_masks(root, image, label, project_id)
        if masks is None:
            return jsonify({'error': f'project {project_id!r} does not hold image {image!r}'}), 400
        why = validate_choice(masks, label, uids)
        if why:
            return jsonify({'error': why}), 400
        try:
            row = Preferences(root).set(image, label, project_id, uids, by, note)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        return jsonify({'success': True, 'preference': row})

    @bp.route('/api/prefer/clear', methods=['POST'])
    def api_prefer_clear():
        body = _json_body()
        image = _basename_stem(str(body.get('image') or '').strip())
        label = str(body.get('label') or '').strip()
        if not image or not label:
            return jsonify({'error': 'image and label are required'}), 400
        removed = Preferences(_root()).clear(image, label)
        return jsonify({'success': True, 'removed': bool(removed)})

    # Overall statistics for the STATS tab.
    @bp.route('/api/stats')
    def api_stats():
        status = (request.args.get('status') or 'accepted').strip().lower()
        if status not in ('accepted', 'all'):
            return jsonify({'error': "status must be 'accepted' or 'all'"}), 400
        try:
            stats = compute_stats(_root(), status)
        except Exception as e:
            log(f"compute_stats failed: {e}")
            return jsonify({'error': str(e)}), 500
        label_names, site_names = _name_maps()
        stats['label_names'] = label_names
        stats['site_names'] = site_names
        return jsonify(stats)

    # Composite exports assembled across projects.
    @bp.route('/api/composites')
    def api_composites():
        return jsonify({'composites': list_composites(_root())})

    @bp.route('/api/composite', methods=['POST'])
    def api_composite():
        body = _json_body()
        labels = body.get('labels')
        if labels is not None and not isinstance(labels, list):
            return jsonify({'error': 'labels must be a list or null'}), 400
        source_types = body.get('source_types')
        if source_types is not None:
            if not isinstance(source_types, list):
                return jsonify({'error': 'source_types must be a list or null'}), 400
            unknown = sorted({str(s) for s in source_types if str(s) not in SOURCE_TYPES})
            if unknown:
                return jsonify({'error': f"unknown source_type: {', '.join(unknown)}",
                                 'allowed': sorted(SOURCE_TYPES)}), 400
        rebuild = body.get('rebuild')
        if rebuild is not None and not isinstance(rebuild, bool):
            return jsonify({'error': 'rebuild must be a boolean'}), 400
        # Project filter (Task H): which projects may supply masks. Unknown
        # ids are rejected by build_composite itself (ValueError -> 400), so
        # the route only checks the shape.
        project_ids = body.get('project_ids')
        if project_ids is not None and not isinstance(project_ids, list):
            return jsonify({'error': 'project_ids must be a list or null'}), 400
        try:
            result = build_composite(_root(), body.get('name'), labels, body.get('by') or '',
                                      source_types=source_types, rebuild=bool(rebuild),
                                      project_ids=project_ids)
        except CompositeExists as e:
            # Both keys: the UI reads err.path for every error, out_dir is
            # the documented success-shape name.
            return jsonify({'error': 'composite already exists', 'path': str(e), 'out_dir': str(e)}), 409
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            log(f"build_composite failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify(result)

    # Hand a composite to the pipeline as a project that starts at Train.
    # The run dir's step4test export is a SYMLINK to the composite, so the
    # training images are never duplicated, and composite.json remembers the
    # project it fed (linked_projects).
    @bp.route('/api/composite/create-project', methods=['POST'])
    def api_composite_create_project():
        body = _json_body()
        name = str(body.get('name') or '').strip()
        by = str(body.get('by') or '').strip()
        if not name:
            return jsonify({'error': 'name is required'}), 400
        if not _INITIALS_RE.match(by):
            return jsonify({'error': 'reviewer initials (by) must be 1 to 4 letters'}), 400
        try:
            result = create_training_project(_root(), name, by)
        except FileNotFoundError as e:
            return jsonify({'error': str(e), 'name': name}), 404
        except RunDirExhausted as e:
            # Same shape the VICARIUS launcher answers when a day's run
            # folders for one set of initials are used up.
            return jsonify({'error': str(e), 'name': name}), 409
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            # Never hand a raw exception (and the filesystem path inside it)
            # to the browser: the detail goes to the orchestrator log.
            log(f"create_training_project failed for {name!r}: {e}")
            return jsonify({'error': 'could not create the project. See the orchestrator '
                                     'log for what went wrong.', 'name': name}), 500
        out = {'success': True}
        out.update(result)
        return jsonify(out)

    # Registry hygiene: drop test-fixture rows (D7) after a backup.
    @bp.route('/api/registry/purge-test-rows', methods=['POST'])
    def api_purge_test_rows():
        body = _json_body()
        if body.get('confirm') is not True:
            return jsonify({'error': 'confirm: true is required'}), 400
        try:
            removed, backup = purge_test_rows(_root())
        except Exception as e:
            log(f"purge_test_rows failed: {e}")
            return jsonify({'error': str(e)}), 500
        return jsonify({'success': True, 'removed': removed, 'backup': backup})

    return bp
