"""Mask preferences: which project's masks are canonical for an (image, label).

Stored next to the mask registry (spec D3):

    <inprocess_root>/_mask_registry/preferences.csv
    columns: basename,label,project_id,uids,chosen_by,at,note

One row per (basename, label). `uids` is semicolon-joined and narrows the
choice to specific masks; empty means every accepted mask of that label in
the chosen project. The composite export reads these rows first.
"""

import csv
import os
import sys
from contextlib import contextmanager

_PKG_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_SCRIPTS not in sys.path:
    sys.path.insert(0, _PKG_SCRIPTS)

from _reefreview.clock import now_ast  # noqa: E402

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False

PREF_FIELDS = ["basename", "label", "project_id", "uids", "chosen_by", "at", "note"]
PREFERENCES_FILENAME = "preferences.csv"


def preferences_path(inprocess_root):
    return os.path.join(inprocess_root or "", "_mask_registry", PREFERENCES_FILENAME)


def _split_uids(value):
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value or "").split(";")
    out = []
    for u in items:
        u = str(u or "").strip()
        if u and u not in out:
            out.append(u)
    return out


def _row_out(raw):
    return {
        "basename": (raw.get("basename") or "").strip(),
        "label": (raw.get("label") or "").strip(),
        "project_id": (raw.get("project_id") or "").strip(),
        "uids": _split_uids(raw.get("uids")),
        "chosen_by": (raw.get("chosen_by") or "").strip(),
        "at": (raw.get("at") or "").strip(),
        "note": (raw.get("note") or "").strip(),
    }


class Preferences:
    def __init__(self, inprocess_root=None, path=None):
        self.path = os.path.abspath(path or preferences_path(inprocess_root))
        self.root = os.path.dirname(self.path)
        self._lock_path = os.path.join(self.root, ".preferences.lock")

    @contextmanager
    def _locked(self):
        os.makedirs(self.root, exist_ok=True)
        if not _HAVE_FCNTL:
            yield
            return
        fh = open(self._lock_path, "w")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    def load(self):
        """Every preference row, uids already split into a list."""
        if not os.path.exists(self.path):
            return []
        rows = []
        try:
            with open(self.path, newline="", encoding="utf-8") as fh:
                for raw in csv.DictReader(fh):
                    row = _row_out(raw)
                    if row["basename"] and row["label"]:
                        rows.append(row)
        except OSError:
            return []
        return rows

    def index(self):
        """{(basename, label): row}; a later duplicate wins."""
        return {(r["basename"], r["label"]): r for r in self.load()}

    def get(self, basename, label):
        return self.index().get((str(basename or "").strip(), str(label or "").strip()))

    def _write(self, rows):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=PREF_FIELDS)
            w.writeheader()
            for r in rows:
                out = dict(r)
                out["uids"] = ";".join(_split_uids(r.get("uids")))
                w.writerow({k: out.get(k, "") for k in PREF_FIELDS})
        os.replace(tmp, self.path)

    def set(self, basename, label, project_id, uids=None, by="", note=""):
        """Insert or replace the row for (basename, label). Raises ValueError
        when any required field is blank."""
        basename = str(basename or "").strip()
        label = str(label or "").strip()
        project_id = str(project_id or "").strip()
        by = str(by or "").strip().upper()
        if not basename or not label:
            raise ValueError("image and label are required")
        if not project_id:
            raise ValueError("project_id is required")
        if not by:
            raise ValueError("reviewer initials (by) are required")
        row = {
            "basename": basename,
            "label": label,
            "project_id": project_id,
            "uids": _split_uids(uids),
            "chosen_by": by,
            "at": now_ast(),
            "note": str(note or "").strip(),
        }
        with self._locked():
            rows = [r for r in self.load()
                    if not (r["basename"] == basename and r["label"] == label)]
            rows.append(row)
            self._write(rows)
        return row

    def clear(self, basename, label):
        """Remove the row for (basename, label). Returns True when one existed."""
        basename = str(basename or "").strip()
        label = str(label or "").strip()
        with self._locked():
            rows = self.load()
            kept = [r for r in rows if not (r["basename"] == basename and r["label"] == label)]
            if len(kept) == len(rows):
                return False
            self._write(kept)
        return True


def exportable_masks(masks, label):
    """The masks of a project that a preference may point at: accepted,
    not review, labeled with exactly this label (the export_yolo filter)."""
    return [m for m in masks
            if m.get("status") == "accepted" and not m.get("review")
            and (m.get("species") or "") == label]


def validate_choice(masks, label, uids):
    """'' when the choice is valid, else the reason. `masks` are the
    cellinfo mask records of the chosen project for the image."""
    ok = exportable_masks(masks, label)
    if not ok:
        return f"that project has no accepted {label} mask on this image"
    known = {m.get("uid") for m in ok}
    missing = [u for u in _split_uids(uids) if u not in known]
    if missing:
        return "unknown or not accepted mask uid: " + ", ".join(missing)
    return ""
