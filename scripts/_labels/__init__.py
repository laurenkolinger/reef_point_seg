"""
_labels - the Reef Point Seg label (species code) manager.

Public surface:
  - read_vocabulary(master_codes_csv) -> list[dict]
  - add_or_edit(master_codes_csv, dup_csv, *, code, category, name, is_new) -> dict

The Flask blueprint factory (make_blueprint) is a future addition; it is
imported lazily so this package stays importable without Flask installed.
"""

from .labels_io import (
    add_or_edit,
    get_locations,
    list_remap_logs,
    read_remap_log,
    read_vocabulary,
    run_recode,
    validate_row,
    write_new_remap_log,
)

__all__ = [
    "read_vocabulary",
    "add_or_edit",
    "validate_row",
    "get_locations",
    "list_remap_logs",
    "read_remap_log",
    "write_new_remap_log",
    "run_recode",
    "make_blueprint",
]


def make_blueprint(*args, **kwargs):
    """Lazy import of the Flask blueprint factory.

    Kept out of import time so the testable IO core has no Flask dependency.
    """
    from .blueprint import make_blueprint as _mk  # noqa: WPS433

    return _mk(*args, **kwargs)
