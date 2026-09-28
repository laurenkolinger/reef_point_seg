"""VEGA tutorial fixtures: the small datasets the guided tutorial trains on.

The guided tutorial walks a student through one whole train, evaluate and
infer cycle on a real VEGA surface, so the sets it trains on have to be real
and small at the same time. This package holds three tools:

    build_tutorial_set.py       the box fixture, tutorial_boxes, taken from an
                                already staged bounding-box dataset and written
                                under supporting_data/bbox_datasets/.
    build_tutorial_outlines.py  the outlines fixture, tutorial_outlines, taken
                                from a training set on the coverage matrix and
                                written under inprocess/_composite/ where every
                                training set lives.
    build_tutorial_projects.py  one ready-made training project per fixture,
                                minted by the platform's own minter, so the
                                tutorial opens on the work rather than on
                                setting the work up.

The box fixture is imagery from outside the TCRMP survey and never reaches
inprocess/run_*, the mask registry, label provenance, the coverage matrix or
STATS: it is read where it is staged. The outlines fixture is survey imagery
and lives in the survey's own store, and the two projects are training
projects, which the coverage matrix and STATS skip the way they skip every
training project.
"""

__all__ = []
