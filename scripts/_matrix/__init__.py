"""Cross-project image x label matrix for Reef Point Seg.

The data core aggregates per-project label provenance (and a legacy
segmentation fallback) into a single sparse image x label matrix.

`make_blueprint(paths_provider)` mounts the coverage grid UI natively in
the orchestrator (GET / for the page, GET /api/data for the JSON).
"""

from .builder import build_matrix
from .blueprint import make_blueprint

__all__ = ["build_matrix", "make_blueprint"]
