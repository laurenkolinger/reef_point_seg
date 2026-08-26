"""
The permanent, cross-project expert-ID library.

Lives in the module where projects live, gitignored:
    reef_point_seg/inprocess/_expert_id_library/
        manifest.csv          one row per reviewed mask UID
        polygons/<uid>.json   geometry (rle + polygon_px + bbox + image WxH)
        images/<uid>.jpg      the closeup crop (kept for future training data)
        masks/<uid>.png       binary mask cutout (kept for future training data)

This is the master record that ties a mask UID to its assigned code, its source
image, and the labeler + MODE (USER vs EXPERT). It is consulted across projects
so a colony segmented again in a future project inherits a past expert's ID when
the masks overlap >50% (find_overlap). The polygon store also lets future work
combine labelsets.
"""

import csv
import json
import os
import tempfile
from contextlib import contextmanager

from . import mask_geom
from .uid import is_safe_uid
from .clock import now_ast

try:
    import fcntl  # POSIX advisory file locking (this box is Linux)
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover
    _HAVE_FCNTL = False


class UnsafeUidError(ValueError):
    """Raised when a UID is not safe to use as a filesystem path component."""


def replace_preserving_mode(tmp, path):
    """os.replace that first restores sane permissions: mkstemp creates 0600
    files and replace carries the tmp's mode onto the target, which would
    silently flip a 0644 json to owner-only. Keep the destination's existing
    mode, else the umask default."""
    try:
        mode = os.stat(path).st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    try:
        os.chmod(tmp, mode)
    except OSError:
        pass
    os.replace(tmp, path)


def _atomic_write_json(path, obj, indent=1):
    """Atomic JSON write via a UNIQUE temp file. A fixed '<path>.tmp' name
    lets two concurrent writers race: both write the same tmp, the first
    os.replace consumes it, and the second crashes with FileNotFoundError."""
    d = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + '.',
                               suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=indent)
        replace_preserving_mode(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _check_uid(uid):
    if not is_safe_uid(uid):
        raise UnsafeUidError(f"unsafe UID for a filesystem path: {uid!r}")
    return uid

MANIFEST_FIELDS = [
    'uid', 'code', 'name', 'category', 'confidence', 'labeler', 'mode',
    'site', 'year', 'transect', 'frame', 'source_image', 'project_id', 'project_name',
    'polygon_file', 'image_file', 'mask_file', 'created_at', 'updated_at',
]


def default_dir():
    """<module>/inprocess/_expert_id_library/ resolved from this package."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(pkg))     # scripts/.. = github_repo
    module_root = os.path.dirname(repo)              # github_repo/.. = reef_point_seg
    return os.path.join(module_root, 'inprocess', '_expert_id_library')


class Library:
    def __init__(self, root=None):
        self.root = os.path.abspath(root or default_dir())
        self.manifest_path = os.path.join(self.root, 'manifest.csv')
        self.polygons_dir = os.path.join(self.root, 'polygons')
        self.images_dir = os.path.join(self.root, 'images')
        self.masks_dir = os.path.join(self.root, 'masks')
        self.previews_dir = os.path.join(self.root, 'previews')
        self.reviews_dir = os.path.join(self.root, 'reviews')
        self._lock_path = os.path.join(self.root, '.manifest.lock')

    # ── cross-process lock (step5 export + importer can run at once) ──
    @contextmanager
    def _locked(self):
        """Serialize manifest read-modify-write across processes via flock so a
        concurrent step5 export and importer (or two imports) can't lose rows."""
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
        for d in (self.root, self.polygons_dir, self.images_dir, self.masks_dir,
                  self.previews_dir, self.reviews_dir):
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()
        readme = os.path.join(self.root, 'README.md')
        if not os.path.exists(readme):
            with open(readme, 'w') as f:
                f.write(
                    "# reef_point_seg expert-ID library\n\n"
                    "Permanent, cross-project store of reviewed mask IDs. Gitignored.\n"
                    "`manifest.csv` is the master record (UID, code, labeler, mode "
                    "USER|EXPERT). `polygons/`, `images/`, `masks/` keep the geometry "
                    "and image/mask/label sidecar so labelsets can be combined and so "
                    "future projects can inherit a past expert's ID on overlapping masks.\n"
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
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for rec in records.values():
                w.writerow({k: rec.get(k, '') for k in MANIFEST_FIELDS})
        os.replace(tmp, self.manifest_path)

    def lookup(self, uid):
        return self.load().get(uid)

    def upsert(self, record):
        """Insert or update one manifest row (keyed on uid). Stamps timestamps.
        The read-modify-write is locked so a concurrent step5 export and importer
        (or two imports) can't lose each other's rows."""
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
                merged = {k: record.get(k, '') for k in MANIFEST_FIELDS}
                merged['created_at'] = merged.get('created_at') or now
                merged['updated_at'] = now
            records[uid] = merged
            self._write_all(records)
        return merged

    # ── geometry + image sidecar ─────────────────────────────────────
    # Path-building functions validate the UID first so a hostile/garbled UID
    # (e.g. from an expert's CSV) can never escape the library directory.
    def save_polygon(self, uid, geom):
        """geom: {rle, polygon_px, polygon_norm, bbox, area, width, height, source_image}."""
        _check_uid(uid)
        self.ensure()
        path = os.path.join(self.polygons_dir, f'{uid}.json')
        with open(path, 'w') as f:
            json.dump(geom, f)
        return os.path.relpath(path, self.root)

    def load_polygon(self, uid):
        if not is_safe_uid(uid):
            return None
        path = os.path.join(self.polygons_dir, f'{uid}.json')
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    # ── per-uid reviewer detail (CONTRACTS §3, §8) ───────────────────
    # reviews/<uid>.json = {uid, reviews:[{reviewer,code,confidence,at}], accepted}
    # Mirrors the manifest item so the accumulated per-reviewer IDs + the final
    # accepted code survive cross-project (the manifest item is pruned on accept).
    def reviews_path(self, uid):
        return os.path.join(self.reviews_dir, f'{_check_uid(uid)}.json')

    def load_reviews(self, uid):
        """Return {uid, reviews:[...], accepted} or a fresh empty shell. Never
        raises on a missing/garbled file or unsafe UID."""
        empty = {'uid': uid, 'reviews': [], 'accepted': None}
        if not is_safe_uid(uid):
            return empty
        path = os.path.join(self.reviews_dir, f'{uid}.json')
        if not os.path.exists(path):
            return empty
        try:
            with open(path) as f:
                obj = json.load(f)
        except Exception:
            return empty
        obj.setdefault('uid', uid)
        obj.setdefault('reviews', [])
        obj.setdefault('accepted', None)
        return obj

    def save_reviews(self, uid, obj):
        """Persist the per-uid reviewer detail atomically (unique tmp name so
        two concurrent writers never consume each other's tmp file mid
        os.replace). UID-safe path. Callers doing a read-modify-write should
        use upsert_review/set_review_accepted, which hold the library lock
        around the whole cycle."""
        _check_uid(uid)
        self.ensure()
        rec = dict(obj or {})
        rec['uid'] = uid
        rec.setdefault('reviews', [])
        rec.setdefault('accepted', None)
        path = os.path.join(self.reviews_dir, f'{uid}.json')
        _atomic_write_json(path, rec)
        return os.path.relpath(path, self.root)

    def upsert_review(self, uid, reviewer, code='', confidence='', at=''):
        """UPSERT one reviewer's row into reviews/<uid>.json keyed by reviewer
        (CONTRACTS §3, rolling ingest). The whole load-filter-append-save runs
        under the library lock so two concurrent imports for the same uid
        (different reviewers) can't lose each other's rows — the same lost
        update the manifest lock already prevents. Returns the saved detail."""
        reviewer = (reviewer or '').strip()
        with self._locked():
            detail = self.load_reviews(uid)
            rows = [r for r in (detail.get('reviews') or [])
                    if (r.get('reviewer') or '').strip() != reviewer]
            rows.append({'reviewer': reviewer, 'code': code,
                         'confidence': confidence, 'at': at or now_ast()})
            detail['reviews'] = rows
            self.save_reviews(uid, detail)
        return detail

    def set_review_accepted(self, uid, accepted):
        """Stamp the final accepted dict onto reviews/<uid>.json under the
        library lock (locked read-modify-write, same rationale as
        upsert_review). Returns the saved detail."""
        with self._locked():
            detail = self.load_reviews(uid)
            detail['accepted'] = accepted
            self.save_reviews(uid, detail)
        return detail

    def image_path(self, uid):
        return os.path.join(self.images_dir, f'{_check_uid(uid)}.jpg')

    def mask_path(self, uid):
        return os.path.join(self.masks_dir, f'{_check_uid(uid)}.png')

    def preview_mask_path(self, uid):
        """Crop-aligned alpha mask for the catalog tile's opacity overlay
        (distinct from mask_path's full-frame binary kept for training)."""
        return os.path.join(self.previews_dir, f'{_check_uid(uid)}.png')

    # ── >50% overlap auto-relabel lookup ─────────────────────────────
    def find_overlap(self, source_image, rle, thresh=0.5, require_expert=True,
                     exclude_uid=None):
        """Find a stored mask on the SAME source image whose footprint covers
        the given mask by > thresh (intersection / area-of-new-mask).

        Returns the best matching manifest record (with its overlap fraction
        attached as record['_overlap']) or None. By default only EXPERT-mode
        records with a real (non-REVIEW/IDK) code are eligible, so we only ever
        inherit a *past expert's* identification.
        """
        if not rle:
            return None
        target_name = os.path.basename(str(source_image or ''))
        records = self.load()
        best = None
        best_frac = thresh
        for uid, rec in records.items():
            if exclude_uid and uid == exclude_uid:
                continue
            if os.path.basename(rec.get('source_image', '')) != target_name:
                continue
            code = (rec.get('code') or '').strip()
            if require_expert and rec.get('mode') != 'EXPERT':
                continue
            if code in ('', 'REVIEW', 'IDK'):
                continue
            geom = self.load_polygon(uid)
            if not geom or 'rle' not in geom:
                continue
            new_m, other_m = mask_geom.decode_pair(rle, geom['rle'])
            if new_m is None:
                continue
            frac = mask_geom.intersection_over_area(new_m, other_m)
            if frac > best_frac:
                best_frac = frac
                best = dict(rec)
                best['_overlap'] = round(frac, 4)
        return best
