"""
Label (species code) manager as a self-contained Flask blueprint.

One shared implementation the reef_point_seg orchestrator mounts natively
(no iframe, no subprocess). The host supplies a `paths_provider` callable
that returns the current effective paths; everything else (routes, the UI
fragment, its CSS/JS) lives here.

    from _labels import make_blueprint
    app.register_blueprint(make_blueprint(paths_provider), url_prefix='/labels')

paths_provider() -> dict with keys:
    master_codes_csv        the canonical species-codes CSV (source of truth)
    duplicate_master_codes  the byte-identical mirror in supporting_data/
    all_points_csv          the all-points CSV the recode reads
    recode_output_dir       where remap_log_*.json and recode outputs live
    pipeline_yaml           pipeline.yaml, so locations can be re-derived

The IO core (labels_io) is pure-python and does all the file work behind a
binding safety posture: add or edit only (never delete, never change a code),
back up the canonical file before every write, mirror the duplicate, and run
the recode only on an explicit, button-driven request. This blueprint is a
thin transport over that core; every mutating route returns {ok, ...} and
surfaces a validation failure as a 400 with a plain message.

No emoji, no em dashes anywhere.
"""

import os

from flask import Blueprint, jsonify, render_template, request

from . import labels_io

_PKG = os.path.dirname(os.path.abspath(__file__))


def make_blueprint(paths_provider, *, name='labels', log_fn=None):
    log = log_fn or (lambda m: print(f"[labels] {m}"))
    bp = Blueprint(name, __name__,
                   template_folder=os.path.join(_PKG, 'templates'),
                   static_folder=os.path.join(_PKG, 'static'),
                   static_url_path='static')

    def P():
        try:
            return paths_provider() or {}
        except Exception as e:  # noqa: BLE001 - report, never crash a request
            log(f"paths_provider failed: {e}")
            return {}

    def _body():
        """Request body as a plain dict, tolerant of JSON or form posts."""
        if request.is_json:
            return request.json or {}
        return request.form.to_dict() if request.form else {}

    # ── The page (stays mount-aware via url_for so any url_prefix works) ──
    @bp.route('/')
    def index():
        from flask import url_for
        static_base = url_for(f'{name}.static', filename='_')
        base = static_base.rsplit('/static/', 1)[0]
        return render_template('labels/index.html', base=base)

    # ── Section 1: Vocabulary ─────────────────────────────────────────
    @bp.route('/api/vocabulary')
    def api_vocabulary():
        """The full species-code vocabulary, in file order, plus the canonical
        path it was read from (so the page can show where it lives)."""
        p = P()
        master = p.get('master_codes_csv', '')
        try:
            rows = labels_io.read_vocabulary(master)
        except FileNotFoundError:
            return jsonify({'ok': True, 'rows': [], 'count': 0,
                            'master_codes_csv': master, 'missing': True})
        except Exception as e:  # noqa: BLE001
            log(f"read_vocabulary failed: {e}")
            return jsonify({'error': str(e)}), 500
        categories = sorted({(r.get('category') or '') for r in rows} - {''})
        return jsonify({'ok': True, 'rows': rows, 'count': len(rows),
                        'categories': categories,
                        'master_codes_csv': master,
                        'duplicate_master_codes': p.get('duplicate_master_codes', '')})

    @bp.route('/api/label', methods=['POST'])
    def api_label():
        """Add a new code or edit an existing code's name/category. Backs up the
        canonical file, writes it atomically, then mirrors the duplicate. A
        validation failure (blank field, lowercase code, duplicate on add,
        missing on edit) comes back as 400 with a plain message."""
        p = P()
        master = p.get('master_codes_csv', '')
        dup = p.get('duplicate_master_codes', '')
        body = _body()
        code = (body.get('code') or '').strip()
        category = (body.get('category') or '').strip()
        name = (body.get('name') or '').strip()
        is_new = bool(body.get('is_new'))
        if not master:
            return jsonify({'error': 'no master_codes_csv configured'}), 400
        try:
            result = labels_io.add_or_edit(
                master, dup,
                code=code, category=category, name=name, is_new=is_new,
            )
        except ValueError as e:
            # Expected validation failure: plain message, 400.
            return jsonify({'error': str(e)}), 400
        except Exception as e:  # noqa: BLE001
            log(f"add_or_edit failed: {e}")
            return jsonify({'error': str(e)}), 500
        log(f"{'added' if is_new else 'edited'} code {code}; "
            f"backup {result.get('backup_path')}")
        return jsonify({'ok': True, **result})

    # ── Section 2: Remaps ─────────────────────────────────────────────
    @bp.route('/api/remaps')
    def api_remaps():
        """List every remap_log_*.json newest first, and inline the full body of
        the active (newest) one so the page can show its old->new entries."""
        p = P()
        out_dir = p.get('recode_output_dir', '')
        try:
            logs = labels_io.list_remap_logs(out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"list_remap_logs failed: {e}")
            return jsonify({'error': str(e)}), 500
        active = None
        if logs:
            try:
                active = labels_io.read_remap_log(logs[0]['path'])
                active['path'] = logs[0]['path']
                active['name'] = logs[0]['name']
            except Exception as e:  # noqa: BLE001
                log(f"read active remap_log failed: {e}")
                active = None
        return jsonify({'ok': True, 'logs': logs, 'count': len(logs),
                        'active': active, 'recode_output_dir': out_dir})

    @bp.route('/api/remap', methods=['POST'])
    def api_remap():
        """Write a NEW timestamped remap log from {remaps, excludes}. Never
        overwrites an existing file. remaps is a list of {old, new} (or {from,
        to}) pairs; excludes is a list of codes to drop."""
        p = P()
        out_dir = p.get('recode_output_dir', '')
        body = _body()
        remaps = body.get('remaps') or []
        excludes = body.get('excludes') or []
        if not out_dir:
            return jsonify({'error': 'no recode_output_dir configured'}), 400
        if not isinstance(remaps, list) or not isinstance(excludes, list):
            return jsonify({'error': 'remaps and excludes must be lists'}), 400
        if not remaps and not excludes:
            return jsonify({'error': 'nothing to save: add at least one remap '
                                     'or exclude'}), 400
        source_note = (body.get('source_note') or 'composed in the label manager').strip()
        try:
            result = labels_io.write_new_remap_log(
                out_dir, remaps, excludes, source_note=source_note,
            )
        except Exception as e:  # noqa: BLE001
            log(f"write_new_remap_log failed: {e}")
            return jsonify({'error': str(e)}), 500
        log(f"wrote remap log {result.get('path')}")
        return jsonify({'ok': True, **result,
                        'n_remaps': len(remaps), 'n_excludes': len(excludes)})

    # ── Section 3: Locations + Recode ─────────────────────────────────
    @bp.route('/api/locations')
    def api_locations():
        """The canonical paths the manager touches, read-only, plus the latest
        remap log so the recode panel can name what it would consume. Derived
        from pipeline.yaml when present, falling back to the provider values."""
        p = P()
        loc = {}
        pipeline_yaml = p.get('pipeline_yaml', '')
        if pipeline_yaml and os.path.isfile(pipeline_yaml):
            try:
                loc = labels_io.get_locations(pipeline_yaml)
            except Exception as e:  # noqa: BLE001
                log(f"get_locations failed: {e}")
                loc = {}
        # Provider values win where present (they track the live project), and
        # fill any gap get_locations left.
        merged = {
            'master_codes_csv': p.get('master_codes_csv') or loc.get('master_codes_csv', ''),
            'duplicate_master_codes': p.get('duplicate_master_codes') or loc.get('duplicate_master_codes', ''),
            'all_points_csv': p.get('all_points_csv') or loc.get('all_points_csv', ''),
            'recode_output_dir': p.get('recode_output_dir') or loc.get('recode_output_dir', ''),
            'supporting_data_dir': loc.get('supporting_data_dir', ''),
            'pipeline_yaml': pipeline_yaml,
        }
        latest = None
        try:
            logs = labels_io.list_remap_logs(merged['recode_output_dir'])
            latest = logs[0] if logs else None
        except Exception as e:  # noqa: BLE001
            log(f"list_remap_logs (locations) failed: {e}")
        # Flag whether each path actually exists so the page can be honest.
        exists = {k: bool(v) and os.path.exists(v)
                  for k, v in merged.items() if k.endswith('_csv') or k.endswith('_dir')}
        return jsonify({'ok': True, 'locations': merged,
                        'latest_remap_log': latest, 'exists': exists})

    @bp.route('/api/recode', methods=['POST'])
    def api_recode():
        """Manually re-run the recode on the canonical inputs and a chosen remap
        log (defaults to the latest). Backs up the prior recode outputs first.
        Guarded: requires confirm=true in the body so it is never a silent or
        accidental run."""
        p = P()
        body = _body()
        if not body.get('confirm'):
            return jsonify({'error': 'recode not confirmed; this regenerates the '
                                     'recoded master codes and must be confirmed'}), 400
        all_points = p.get('all_points_csv', '')
        master = p.get('master_codes_csv', '')
        out_dir = p.get('recode_output_dir', '')
        if not (all_points and master and out_dir):
            return jsonify({'error': 'missing one of all_points_csv, '
                                     'master_codes_csv, recode_output_dir'}), 400

        remap_log_path = (body.get('remap_log_path') or '').strip()
        if not remap_log_path:
            try:
                logs = labels_io.list_remap_logs(out_dir)
            except Exception as e:  # noqa: BLE001
                return jsonify({'error': f'could not list remap logs: {e}'}), 500
            if not logs:
                return jsonify({'error': 'no remap log to recode from; save one '
                                         'in the Remaps section first'}), 400
            remap_log_path = logs[0]['path']
        if not os.path.isfile(remap_log_path):
            return jsonify({'error': f'remap log not found: {remap_log_path}'}), 400

        try:
            result = labels_io.run_recode(
                all_points, master, remap_log_path, out_dir, backup=True,
            )
        except Exception as e:  # noqa: BLE001
            log(f"run_recode crashed: {e}")
            return jsonify({'error': str(e)}), 500
        # run_recode reports failures as ok=False rather than raising; mirror
        # that into the right HTTP status so the page can show the message.
        if not result.get('ok'):
            log(f"recode failed: {result.get('error')}")
            return jsonify({'ok': False, **result}), 400
        log(f"recode ok from {os.path.basename(remap_log_path)}")
        return jsonify({'ok': True, 'remap_log_path': remap_log_path, **result})

    return bp
