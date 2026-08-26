"""
End-of-batch export of REVIEW-flagged masks.

Called from the step-5 segmentation app's do_export(). For each REVIEW mask in
the just-exported review batch:

  1. mint a stable UID;
  2. if a PAST EXPERT mask in the library overlaps it >threshold (same source
     image), inherit that expert ID in place (mode EXPERT) and do NOT queue it
     for the expert again;
  3. otherwise render a closeup + full frame, add it to the review repo (the
     GitHub-Pages queue), and record it in the library with mode USER / code
     REVIEW (awaiting an expert ID).

Then commit + push the review repo once. Mutates the passed seg dicts in place
(auto-relabels), so the caller should persist segmentations afterward.
"""

import os
import shutil
import tempfile

import numpy as np
from PIL import Image

from . import filename_parse, render_review, mask_geom, codes as codes_mod
from .uid import make_uid, mask_uid
from .library import Library
from .review_repo import ReviewRepo
from .clock import now_ast


def _is_review_mask(m):
    return bool(m.get('review')) or m.get('species') == 'REVIEW'


def _geom_of(mask, width, height, source_image):
    # area = mask pixel footprint (from the mask dict, else recomputed from rle)
    # so the catalog tile can show it without decoding every tile.
    area = mask.get('area')
    if area is None and mask.get('rle'):
        try:
            area = int(mask_geom.area(
                mask_geom.rle_decode(mask['rle'], shape=tuple(mask['rle']['size']))))
        except Exception:
            area = None
    return {
        'rle': mask.get('rle'),
        'polygon_px': mask.get('polygon_px'),
        'polygon_norm': mask.get('polygon_norm'),
        'bbox': mask.get('bbox'),
        'area': area,
        'width': int(width or 0),
        'height': int(height or 0),
        'source_image': source_image,
        'source_x': mask.get('source_x'),
        'source_y': mask.get('source_y'),
    }


def _save_binary_png(rle, path):
    """Write the full-frame binary mask as a compact 1-bit PNG (training sidecar)."""
    if not rle:
        return
    binary = mask_geom.rle_decode(rle, shape=tuple(rle['size']))
    Image.fromarray((binary.astype(np.uint8) * 255), mode='L').convert('1').save(path)


def export_flagged_masks(segmentations, batch_files, *, export_dir, review_dir,
                         repo_url, library_dir, master_codes, contacts,
                         featured_codes, project_id, project_name='',
                         candidate_codes=None, site_codes=None,
                         pad_px=40, max_edge=720,
                         full_max_edge=1400, overlap_thresh=0.5, git_push=True,
                         log_fn=None):
    log = log_fn or (lambda _m: None)
    lib = Library(library_dir or None)
    lib.ensure()
    repo = ReviewRepo(review_dir, remote_url=repo_url,
                      master_codes_path=master_codes,
                      site_codes_path=site_codes, log_fn=log)
    repo.ensure_repo()
    if contacts:
        repo.set_contacts(contacts)
    # site_code -> full name, for stamping site_full on each item (CONTRACTS §7).
    site_map = codes_mod.load_sites(site_codes)
    candidate_codes = list(candidate_codes or [])

    stats = {'new': 0, 'auto_relabeled': 0, 'skipped_rejected': 0,
             'skipped_missing_image': 0, 'uids': []}
    # UIDs minted in THIS call, so two masks whose source click rounds to the
    # same pixel don't collide to one UID (the second would overwrite the first).
    seen_uids = set()
    tmp = tempfile.mkdtemp(prefix='reefreview_')
    try:
        for fn in batch_files:
            seg = segmentations.get(fn)
            if not seg:
                continue
            image_path = seg.get('image_path_abs')
            W, H = seg.get('image_width'), seg.get('image_height')
            parsed = filename_parse.parse(fn)
            for idx, mask in enumerate(seg.get('masks', [])):
                if not _is_review_mask(mask):
                    continue
                # Only pending/accepted masks are reviewable; rejected (or any
                # other non-pending state) is the reviewer's "deleted" — skip.
                if mask.get('status') not in (None, '', 'pending', 'accepted'):
                    stats['skipped_rejected'] += 1
                    continue

                # Stable, content-derived UID (re-derivable by the importer). The
                # content disc disambiguates two masks at the same rounded pixel
                # without depending on batch order. A residual collision (same
                # source pixel AND same geometry hash) gets a stable index tail.
                uid = mask_uid(fn, mask, parsed)
                if uid in seen_uids:
                    uid = f"{uid}-{idx}"
                seen_uids.add(uid)
                # Persist the minted UID onto the mask so the importer relabels by
                # exact stored value, never a fragile re-derivation.
                mask['review_uid'] = uid
                rle = mask.get('rle')
                geom = _geom_of(mask, W, H, fn)

                site_code = parsed.get('site', '') or ''
                base = {
                    'uid': uid, 'site': site_code,
                    'site_full': site_map.get(site_code.upper(), '') if site_code else '',
                    'year': parsed.get('year') or '', 'transect': parsed.get('transect'),
                    'frame': parsed.get('frame'), 'source_image': fn,
                    'project_id': project_id, 'project_name': project_name or project_id,
                }

                # 1) Inherit a past expert's ID on a heavily-overlapping mask.
                match = lib.find_overlap(fn, rle, thresh=overlap_thresh)
                if match:
                    mask['species'] = match.get('code', '')
                    mask['name'] = match.get('name', '')
                    mask['category'] = match.get('category', '')
                    mask['review'] = False
                    mask['expert_id'] = {
                        'code': match.get('code', ''), 'mode': 'EXPERT',
                        'labeler': match.get('labeler', ''), 'via': 'overlap',
                        'from_uid': match.get('uid', ''), 'overlap': match.get('_overlap'),
                    }
                    lib.save_polygon(uid, geom)
                    lib.upsert({**base, 'code': match.get('code', ''),
                                'name': match.get('name', ''),
                                'category': match.get('category', ''),
                                'confidence': match.get('confidence', ''),
                                'labeler': match.get('labeler', ''), 'mode': 'EXPERT',
                                'polygon_file': f'polygons/{uid}.json'})
                    stats['auto_relabeled'] += 1
                    log(f"[review] {uid}: auto-relabeled {match.get('code')} "
                        f"from {match.get('uid')} (overlap {match.get('_overlap')})")
                    continue

                # 2) Queue for the expert: render imagery, add to repo + library.
                # If the source frame is gone we can't render a closeup; still
                # record geometry in the library (useful for overlap/merges) but
                # DON'T queue it to the review site (nothing to show) and report it.
                if not image_path or not os.path.exists(image_path):
                    lib.save_polygon(uid, geom)
                    lib.upsert({**base, 'code': 'REVIEW', 'name': '', 'category': 'Review',
                                'confidence': '', 'labeler': (contacts[0] if contacts else ''),
                                'mode': 'USER', 'polygon_file': f'polygons/{uid}.json'})
                    stats['skipped_missing_image'] += 1
                    log(f"[review] {uid}: source image missing, stored geometry only "
                        f"(not queued): {image_path}")
                    continue

                crop = os.path.join(tmp, f'{uid}_crop.jpg')
                amask = os.path.join(tmp, f'{uid}_mask.png')
                full = os.path.join(tmp, f'{uid}_full.jpg')
                ci = render_review.render_closeup(image_path, mask, crop, amask,
                                                  pad_px=pad_px, max_edge=max_edge)
                render_review.render_full(image_path, mask, full, max_edge=full_max_edge)

                item_meta = {**base, 'featured_codes': list(featured_codes or []),
                             'candidate_codes': list(candidate_codes),
                             'reviews': [], 'accepted': None,
                             'added_at': now_ast()}
                if ci:
                    item_meta['crop_w'] = ci.get('crop_w')
                    item_meta['crop_h'] = ci.get('crop_h')
                repo.add_item(uid, item_meta,
                              crop_src=crop if os.path.exists(crop) else None,
                              mask_src=amask if os.path.exists(amask) else None,
                              full_src=full if os.path.exists(full) else None)

                # Permanent library record (pending: mode USER, code REVIEW).
                lib.save_polygon(uid, geom)
                if os.path.exists(crop):
                    shutil.copy2(crop, lib.image_path(uid))
                if os.path.exists(amask):
                    # crop-aligned alpha, for the catalog tile's opacity overlay
                    shutil.copy2(amask, lib.preview_mask_path(uid))
                _save_binary_png(rle, lib.mask_path(uid))
                lib.upsert({**base, 'code': 'REVIEW', 'name': '', 'category': 'Review',
                            'confidence': '', 'labeler': (contacts[0] if contacts else ''),
                            'mode': 'USER', 'polygon_file': f'polygons/{uid}.json',
                            'image_file': f'images/{uid}.jpg', 'mask_file': f'masks/{uid}.png'})
                stats['new'] += 1
                stats['uids'].append(uid)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if stats['new'] or stats['auto_relabeled']:
        msg = (f"export {project_name or project_id}: +{stats['new']} for review, "
               f"{stats['auto_relabeled']} auto-relabeled")
        pushed = repo.commit_push(msg, push=git_push)
        stats['pushed'] = pushed
    else:
        stats['pushed'] = False
    stats['pending_total'] = len(repo.pending_uids())
    return stats
