"""Shared run library for Reef Point Seg (VEGA).

This package holds ``labruns``, the cross-run library the VEGA main window
imports: run discovery under inprocess/_models and inprocess/run_*, champion
records, split comparability, and the detection-dataset wrappers. Import the
library by module path so nothing else comes along:

    from _modellab import labruns

The project-free Train + Infer page that this package once served (a Flask
blueprint at /models with its template, script and stylesheet) was retired
on 2026-09-05 per docs/adr/0002-one-training-surface-the-vega-main-window.md:
training, evaluation and inference happen in the VEGA main window only. This
file deliberately imports nothing, so importing the package never loads Flask.
"""
