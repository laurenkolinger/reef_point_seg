"""
Cross-project image x label matrix as a self-contained Flask blueprint.

One shared implementation the reef_point_seg orchestrator mounts natively
(no iframe, no subprocess). The host supplies a `paths_provider` callable
that returns the current effective paths; everything else (routes, the UI
fragment, its CSS/JS) lives here.

    from _matrix import make_blueprint
    app.register_blueprint(make_blueprint(paths_provider), url_prefix='/matrix')

paths_provider() -> dict with key:
    inprocess_root   the `<module>/inprocess` projects root that
                     build_matrix() rescans on every /api/data call.

The data core (builder.build_matrix) is pure-python and rescans the tree
on every call, so a purged project simply stops contributing.
"""

import os

from flask import Blueprint, jsonify, render_template

from .builder import build_matrix

_PKG = os.path.dirname(os.path.abspath(__file__))


def make_blueprint(paths_provider, *, name='matrix', log_fn=None):
    log = log_fn or (lambda m: print(f"[matrix] {m}"))
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

    # The grid page. Stays mount-aware via url_for so it works under any
    # url_prefix the host chooses (the orchestrator mounts at /matrix).
    @bp.route('/')
    def index():
        from flask import url_for
        static_base = url_for(f'{name}.static', filename='_')
        base = static_base.rsplit('/static/', 1)[0]
        return render_template('matrix/index.html', base=base)

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
        return jsonify(data)

    return bp
