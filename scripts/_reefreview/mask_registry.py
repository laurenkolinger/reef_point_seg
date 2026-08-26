"""
The permanent, cross-project canonical mask registry.

Lives in the module where projects live, gitignored:
    reef_point_seg/inprocess/_mask_registry/
        manifest.csv          one row per mask UID, across every project

This is the master record that tracks every mask across all projects: its
species/category label, its review status, its source image and project
context. The label coverage matrix derives from this registry. Unlike the
expert-ID library (library.py), the registry is manifest-only: it carries
no polygon/image/mask geometry sidecar directories.
"""

import csv
import os
from contextlib import contextmanager

from .clock import now_ast
from . import uid as _uid
from . import filename_parse

try:
    import fcntl  # POSIX advisory file locking (this box is Linux)
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False


REGISTRY_FIELDS = [
    'uid', 'source_image', 'basename', 'species', 'category', 'source_type',
    'review', 'expert_mode', 'status', 'project_id', 'project_name', 'site',
    'year', 'transect', 'frame', 'created_at', 'updated_at',
]


def default_registry_dir():
    """<module>/inprocess/_mask_registry/ resolved from this package."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(pkg))     # scripts/.. = github_repo
    module_root = os.path.dirname(repo)              # github_repo/.. = reef_point_seg
    return os.path.join(module_root, 'inprocess', '_mask_registry')


def build_registry_record(filename, mask, project_id, project_name, parsed=None):
    """Turn an accepted mask into a canonical REGISTRY_FIELDS-shaped record.

    Mints the content-stable uid via uid.mask_uid(filename, mask, parsed) and
    returns None when that uid is not safe to use (so callers can skip the
    mask rather than write a bad row). Pure: no I/O, no registry access.
    """
    u = _uid.mask_uid(filename, mask, parsed)
    if not _uid.is_safe_uid(u):
        return None

    p = parsed or filename_parse.parse(filename)
    expert_id = mask.get('expert_id')
    expert_mode = isinstance(expert_id, dict) and expert_id.get('mode') == 'EXPERT'

    return {
        'uid': u,
        'source_image': os.path.basename(filename),
        'basename': os.path.splitext(os.path.basename(filename))[0],
        'species': mask.get('species', ''),
        'category': mask.get('category', ''),
        'source_type': mask.get('source_type', ''),
        'review': '1' if bool(mask.get('review')) else '',
        'expert_mode': '1' if expert_mode else '',
        'status': mask.get('status', ''),
        'project_id': project_id,
        'project_name': project_name,
        'site': p.get('site'),
        'year': p.get('year'),
        'transect': p.get('transect'),
        'frame': p.get('frame'),
    }


class MaskRegistry:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or default_registry_dir())
        self.manifest_path = os.path.join(self.root, 'manifest.csv')
        self._lock_path = os.path.join(self.root, '.manifest.lock')

    # ── cross-process lock (concurrent writers can't lose rows) ───────
    @contextmanager
    def _locked(self):
        """Serialize manifest read-modify-write across processes via flock so
        concurrent writers (e.g. two pipeline steps) can't lose rows."""
        os.makedirs(self.root, exist_ok=True)
        if not _HAVE_FCNTL:
            yield
            return
        fh = open(self._lock_path, 'w')
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()

    # ── setup ────────────────────────────────────────────────────────
    def ensure(self):
        os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=REGISTRY_FIELDS).writeheader()
        readme = os.path.join(self.root, 'README.md')
        if not os.path.exists(readme):
            with open(readme, 'w') as f:
                f.write(
                    "# reef_point_seg mask registry\n\n"
                    "Permanent, cross-project store of every mask UID. Gitignored.\n"
                    "`manifest.csv` is the master record (UID, species, category, "
                    "review status, project context). The label coverage matrix is "
                    "derived from this registry. Manifest-only: no polygon/image/mask "
                    "geometry sidecar directories (see the expert-ID library for that).\n"
                )

    # ── manifest I/O ─────────────────────────────────────────────────
    def load(self):
        """Return {uid: record}."""
        out = {}
        if not os.path.exists(self.manifest_path):
            return out
        with open(self.manifest_path, newline='') as f:
            for row in csv.DictReader(f):
                uid = (row.get('uid') or '').strip()
                if uid:
                    out[uid] = row
        return out

    def _write_all(self, records):
        self.ensure()
        tmp = self.manifest_path + '.tmp'
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
            w.writeheader()
            for rec in records.values():
                w.writerow({k: rec.get(k, '') for k in REGISTRY_FIELDS})
        os.replace(tmp, self.manifest_path)

    def lookup(self, uid):
        return self.load().get(uid)

    def rows(self):
        return list(self.load().values())

    def upsert(self, record):
        """Insert or update one manifest row (keyed on uid). Stamps timestamps.
        The read-modify-write is locked so concurrent writers can't lose each
        other's rows."""
        self.ensure()
        uid = record['uid']
        now = now_ast()
        with self._locked():
            records = self.load()
            if uid in records:
                merged = dict(records[uid])
                merged.update({k: v for k, v in record.items() if v not in (None,)})
                merged['updated_at'] = now
                merged.setdefault('created_at', records[uid].get('created_at', now))
            else:
                merged = {k: record.get(k, '') for k in REGISTRY_FIELDS}
                merged['created_at'] = merged.get('created_at') or now
                merged['updated_at'] = now
            records[uid] = merged
            self._write_all(records)
        return merged
