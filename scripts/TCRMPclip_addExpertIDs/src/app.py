"""
TCRMP Add Expert IDs — standalone host for the shared Add-Expert-IDs blueprint.

The UI + all logic live in the shared `_expertids` blueprint (scripts/_expertids),
which the reef_point_seg orchestrator mounts natively too. This file is just a
thin standalone host: it registers that blueprint and serves a page that includes
its UI fragment. Run it directly to use Add Expert IDs without the orchestrator.

Usage:
    python app.py [--port 5075]
"""

import argparse
import os
import sys

from flask import Flask, render_template
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# scripts/ on path for the shared blueprint.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _expertids import make_blueprint


def _paths():
    """Effective paths from this tool's config (overridable per-request via the
    UI's path fields)."""
    return {
        'export_dir': cfg.EXPORT_DIR,
        'review_dir': cfg.REVIEW_DIR,
        'library_dir': cfg.EXPERT_LIBRARY_DIR or '',
        'review_repo_url': cfg.REVIEW_REPO_URL,
        'master_codes': cfg.MASTER_CODES_CSV,
        'overlap_thresh': cfg.REVIEW_OVERLAP_THRESH,
        'git_push': cfg.REVIEW_GIT_PUSH,
    }


app = Flask(__name__, template_folder='templates')
# Cap the expert-CSV upload so a stray huge file can't be read unbounded into
# memory. A UID,code CSV is tiny; 16 MB is generous headroom.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)
app.register_blueprint(make_blueprint(_paths), url_prefix='/expertids')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCRMP Add Expert IDs importer')
    parser.add_argument('--port', type=int, default=cfg.PORT)
    args = parser.parse_args()
    print(f"Add-Expert-IDs server ready on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
