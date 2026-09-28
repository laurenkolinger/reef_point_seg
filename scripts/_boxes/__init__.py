"""Bounding Boxes: external box datasets for Reef Point Seg.

Stages a third-party bounding-box dataset (Coral Intel) into
supporting_data/bbox_datasets/, converts its boxes to segmentation polygons
with SAM3, and hands training to VEGA by minting a training project that
opens in the main window.

    from _boxes import make_blueprint
    app.register_blueprint(make_blueprint(paths_provider), url_prefix='/boxes')

None of this imagery belongs to the TCRMP survey. Nothing here writes to
inprocess/run_*, the mask registry, label provenance, the coverage matrix, or
STATS: staged datasets live under supporting_data/bbox_datasets and derived
segmentation datasets under inprocess/_bbox.
"""

__all__ = ["make_blueprint"]


def __getattr__(name):
    """Import the blueprint (and Flask with it) only when a host asks for it.
    The two CLI tools in this package run under the module interpreter with no
    web server in the picture, so importing the package must not pull Flask."""
    if name == "make_blueprint":
        from .blueprint import make_blueprint
        return make_blueprint
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
