"""The expert-review lock in mask_ops.py: no new review tag reaches a mask.

Lauren Olinger's decision of 2026-09-06 is that VICARIUS accepts no new
expert-review tags. mask_ops.py carries the shared refusal every annotator
imports, so this suite exercises it directly rather than through a route:
the spelling test (review_tag_requested), the non-mutating scrub
(strip_review_tag), the one choke point every new mask passes (build_mask_dict)
and the merge helper (_merge_review_state). The last case proves the whole
reversal is one constant.

Masks flagged before that decision must keep their tag, so every case here
checks that preservation is untouched: review_uid, reviews[] and expert_id
still travel across a merge, and an already flagged survivor stays flagged.

Run: env/bin/python scripts/TCRMPclip_combinedAnnotate/tests/test_review_lock_core.py
"""
import copy
import os
import sys

import numpy as np

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)
import mask_ops  # noqa: E402

_fail = 0

# A mask this size clears clean_mask's 500 px component floor with room to spare.
_IMAGE_H, _IMAGE_W = 100, 300
_BLOB_SIDE = 60


def check(cond, msg):
    """Print one PASS or FAIL line and count the failures."""
    global _fail
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        _fail += 1


def blob(x0=20):
    """Return one solid rectangle as a boolean (H, W) array."""
    binary = np.zeros((_IMAGE_H, _IMAGE_W), dtype=bool)
    binary[20:20 + _BLOB_SIDE, x0:x0 + _BLOB_SIDE] = True
    return binary


def build(point_info, mask_id=0):
    """Build one mask through the shared choke point, failing loudly on None."""
    mask = mask_ops.build_mask_dict(
        mask_id=mask_id, binary_mask=blob(), score=0.9,
        point_info=point_info, source_type='auto',
    )
    if mask is None:
        raise AssertionError(
            "build_mask_dict returned None for a fixture mask; the blob no "
            "longer clears clean_mask's component floor"
        )
    return mask


def flagged(mask):
    """Report whether a built mask carries a review tag in any spelling."""
    return bool(mask.get('review')) or mask_ops.normalise_review_value(
        mask.get('species')) == mask_ops.REVIEW_TAG


# Every spelling of the tag a payload can arrive in, as (label, payload) pairs.
_TRUTHY_REVIEW_VALUES = (True, 1, "1", "no", "false")
_REVIEW_SPELLINGS = ("REVIEW", "review", " REVIEW ", "Review")


def test_review_tag_requested_catches_every_spelling():
    """Truthy review keys, both species keys, any case, any whitespace."""
    for value in _TRUTHY_REVIEW_VALUES:
        check(mask_ops.review_tag_requested({'review': value}),
              f"review={value!r} is a review tag request")
    for spelling in _REVIEW_SPELLINGS:
        check(mask_ops.review_tag_requested({'species': spelling}),
              f"species={spelling!r} is a review tag request")
        check(mask_ops.review_tag_requested({'species_code': spelling}),
              f"species_code={spelling!r} is a review tag request")
        check(mask_ops.review_tag_requested({'species': '', 'species_code': spelling}),
              f"species_code={spelling!r} beside an empty species is caught "
              "(tested independently, not as a fallback chain)")
        check(mask_ops.review_tag_requested({'species': spelling, 'species_code': 'PA'}),
              f"species={spelling!r} beside a real species_code is caught")
    negatives = (
        ({'review': False}, "review=False"),
        ({'review': 0}, "review=0"),
        ({'review': ''}, "review is an empty string"),
        ({}, "an empty payload"),
        ({'species': 'PA'}, "species=PA"),
        ({'species': ''}, "an empty species"),
        ({'species': 'REVIEWED'}, "a species that merely contains the word"),
        ({'species_code': 'PRE-REVIEW'}, "a species_code that merely contains the word"),
        ({'label': 'REVIEW'}, "a label of REVIEW, which is not a species key"),
    )
    for payload, label in negatives:
        check(not mask_ops.review_tag_requested(payload),
              f"{label} is not a review tag request")


def test_review_tag_requested_catches_the_category_spelling():
    """The third spelling: a category of Review, tested as its own key.

    Across the three segmentation files under inprocess/, a category of
    'Review' pairs one for one with the flag: 130 masks carry both, none
    carries either on its own. It is not a real category either (each app's
    config.py lists 'Target species only' and nothing else), so a payload whose
    only review spelling is the category is a request for the tag and has to be
    caught like the two species keys, not as a decoration on one of them.
    """
    check('category' in mask_ops.REVIEW_TAG_STRING_KEYS,
          f"the category key is in the one policy tuple, got "
          f"{mask_ops.REVIEW_TAG_STRING_KEYS!r}")
    for spelling in _REVIEW_SPELLINGS + ("ReViEw",):
        check(mask_ops.review_tag_requested({'category': spelling}),
              f"category={spelling!r} alone is a review tag request")
        check(mask_ops.review_tag_requested({'species': 'PA', 'category': spelling}),
              f"category={spelling!r} beside a real species is caught")
        check(mask_ops.review_tag_requested(
            {'species': '', 'species_code': '', 'category': spelling}),
            f"category={spelling!r} beside two empty species keys is caught")
    negatives = (
        ({'category': 'Coral'}, "a real category"),
        ({'category': 'Target species only'}, "the shipped default category"),
        ({'category': ''}, "an empty category"),
        ({'category': None}, "a category of None"),
        ({'category': 0}, "a category of 0"),
        ({'species': 'PA'}, "a payload with no category key at all"),
        ({'category': 'Reviewed'}, "a category that merely contains the word"),
        ({'category': 'Pre-review'}, "a category that merely ends in the word"),
        ({'category': 'Review pending'}, "a category that merely starts with it"),
    )
    for payload, label in negatives:
        check(not mask_ops.review_tag_requested(payload),
              f"{label} is not a review tag request")


def test_a_real_species_survives_a_review_category():
    """The scrub is applied per value: PA is kept, the Review category is lost."""
    payload = {'species': 'PA', 'name': 'Porites astreoides',
               'category': 'Review', 'x': 12}
    before = copy.deepcopy(payload)
    stripped = mask_ops.strip_review_tag(payload)
    check(payload == before, f"the argument is unchanged, got {payload}")
    check(stripped['species'] == 'PA',
          f"the real species survives, got {stripped['species']!r}")
    check(stripped['category'] == '',
          f"only the category is blanked, got {stripped['category']!r}")
    check(stripped['name'] == 'Porites astreoides' and stripped['x'] == 12,
          "every other key is copied across unchanged")
    check(not mask_ops.review_tag_requested(stripped),
          "the stripped payload no longer asks for a review tag")
    other_way = mask_ops.strip_review_tag({'species': 'REVIEW', 'category': 'Coral'})
    check(other_way['species'] == '' and other_way['category'] == 'Coral',
          f"and the other way round: the real category survives a REVIEW "
          f"species, got {other_way!r}")


def test_review_tag_requested_tolerates_junk():
    """A payload that is not dict-like answers False instead of raising."""
    junk = (None, [], ['REVIEW'], "REVIEW", 7, 0.0, set(), ('species', 'REVIEW'))
    for value in junk:
        try:
            answer = mask_ops.review_tag_requested(value)
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{value!r} raised {type(exc).__name__} instead of answering False")
            continue
        check(answer is False, f"{value!r} is not a review tag request")
    check(not mask_ops.review_tag_requested({'species': 0, 'review': 0}),
          "a dict with non-string values is not a review tag request")
    check(not mask_ops.review_tag_requested({'species': 0}),
          "a species of 0 is not a review tag request")
    check(not mask_ops.review_tag_requested({'species': None, 'species_code': None}),
          "species and species_code of None are not a review tag request")
    check(mask_ops.review_tag_requested({'species': 0, 'species_code': 'review'}),
          "a species of 0 beside a species_code of review is still caught")


def test_strip_review_tag_never_mutates_its_argument():
    """The scrub returns a new dict and leaves the caller's payload alone."""
    payload = {'species': 'REVIEW', 'species_code': ' review ', 'review': 1,
               'name': 'Flagged for expert review', 'x': 12}
    before = copy.deepcopy(payload)
    stripped = mask_ops.strip_review_tag(payload)
    check(payload == before, f"the argument is unchanged, got {payload}")
    check(stripped is not payload, "the return value is a different dict")
    check(stripped['species'] == '', f"species blanked, got {stripped['species']!r}")
    check(stripped['species_code'] == '', f"species_code blanked, got {stripped['species_code']!r}")
    check(stripped['review'] is False, f"review set to False, got {stripped['review']!r}")
    check(stripped['name'] == 'Flagged for expert review' and stripped['x'] == 12,
          "every other key is copied across unchanged")
    check(not mask_ops.review_tag_requested(stripped),
          "the stripped payload no longer asks for a review tag")
    ordinary = {'species': 'PA', 'category': 'Coral'}
    kept = mask_ops.strip_review_tag(ordinary)
    check(kept['species'] == 'PA' and kept['category'] == 'Coral',
          "an ordinary species survives the scrub untouched")
    try:
        mask_ops.strip_review_tag(['REVIEW'])
    except TypeError as exc:
        check('dict-like' in str(exc), f"a non-dict payload raises a TypeError that says why, got {exc}")
    else:
        check(False, "a non-dict payload must raise TypeError")


def test_build_mask_dict_never_births_a_flagged_mask():
    """Every spelling of the tag produces a mask born without it."""
    mask_ops.reset_review_tags_dropped()
    payloads = []
    for value in _TRUTHY_REVIEW_VALUES:
        payloads.append(({'label': 'A', 'species': 'REVIEW', 'category': 'Review',
                          'review': value, 'x': 50, 'y': 50},
                         f"review={value!r} with species REVIEW"))
    for spelling in _REVIEW_SPELLINGS:
        payloads.append(({'label': 'A', 'species': spelling, 'category': 'Review',
                          'x': 50, 'y': 50},
                         f"species={spelling!r} with no review key"))
        payloads.append(({'label': 'A', 'species': '', 'species_code': spelling,
                          'x': 50, 'y': 50},
                         f"species_code={spelling!r} beside an empty species"))
    for point_info, label in payloads:
        mask = build(point_info)
        check(not flagged(mask), f"{label}: the new mask carries no review tag")
        check(mask_ops.normalise_review_value(mask.get('category')) != mask_ops.REVIEW_TAG,
              f"{label}: the new mask carries no review category")
        check('review' not in mask, f"{label}: no review boolean is written at all")
    check(mask_ops.review_tags_dropped() == len(payloads),
          f"every drop is counted, expected {len(payloads)}, got {mask_ops.review_tags_dropped()}")
    check(mask_ops.reset_review_tags_dropped() == len(payloads),
          "the reset reports the count it cleared")
    check(mask_ops.review_tags_dropped() == 0, "the counter is back to zero after a reset")
    plain = build({'label': 'A', 'species': 'REVIEW', 'x': 50, 'y': 50})
    check(plain['species'] == '',
          f"a species of REVIEW is born blank, got {plain['species']!r}")


def test_build_mask_dict_leaves_an_ordinary_species_alone():
    """A real label passes through the choke point untouched and uncounted."""
    mask_ops.reset_review_tags_dropped()
    mask = build({'label': 'A', 'species': 'PA', 'name': 'Porites astreoides',
                  'category': 'Coral', 'x': 50, 'y': 50})
    check(mask['species'] == 'PA', f"species kept, got {mask['species']!r}")
    check(mask['name'] == 'Porites astreoides', "name kept")
    check(mask['category'] == 'Coral', f"category kept, got {mask['category']!r}")
    check(mask['label'] == 'A' and mask['status'] == 'pending',
          "the rest of the mask is built as before")
    check(mask_ops.review_tags_dropped() == 0, "an ordinary mask counts no drop")
    blank = build({'label': 'B', 'species': '', 'category': '', 'x': 50, 'y': 50})
    check(blank['species'] == '' and mask_ops.review_tags_dropped() == 0,
          "an unlabelled mask is untouched and counts no drop")


def test_build_mask_dict_blanks_a_category_only_review():
    """A point_info whose only review spelling is the category is born clean.

    build_mask_dict has always blanked a Review category, but behind a guard
    that read the flag and the two species keys only, so for a category-only
    point_info the branch never ran. With the category among the spellings the
    guard catches, the blanking is live and the drop is counted like any other.
    """
    mask_ops.reset_review_tags_dropped()
    category_only = build({'label': 'A', 'species': 'PA',
                           'name': 'Porites astreoides', 'category': 'Review',
                           'x': 50, 'y': 50})
    check(category_only['species'] == 'PA',
          f"the real species survives, got {category_only['species']!r}")
    check(category_only['name'] == 'Porites astreoides', "the name survives")
    check(category_only['category'] == '',
          f"the Review category is blanked, got {category_only['category']!r}")
    check('review' not in category_only, "no review boolean is written")
    check(mask_ops.review_tags_dropped() == 1,
          f"the drop is counted, got {mask_ops.review_tags_dropped()}")

    bare = build({'label': 'B', 'category': ' review ', 'x': 50, 'y': 50})
    check(mask_ops.normalise_review_value(bare['category']) != mask_ops.REVIEW_TAG,
          f"a padded lower-case category is blanked too, got {bare['category']!r}")
    check(mask_ops.review_tags_dropped() == 2,
          f"and counted, got {mask_ops.review_tags_dropped()}")
    mask_ops.reset_review_tags_dropped()


def test_merge_does_not_give_the_survivor_a_flag_it_lacked():
    """A merge may preserve review state, never add it."""
    survivor = {'id': 1, 'species': 'PA'}
    absorbed = {'id': 2, 'species': 'PA', 'review': True}
    ok = mask_ops._merge_review_state(survivor, absorbed)
    check(ok is True, "the merge is still allowed")
    check(not survivor.get('review'),
          f"the clean survivor gains no review boolean, got {survivor.get('review')!r}")
    promoted_survivor = {'id': 1, 'species': 'PA', 'category': 'Coral'}
    promoted_absorbed = {'id': 2, 'species': 'REVIEW', 'category': 'Review',
                         'name': 'Flagged for expert review',
                         'expert_id': {'mode': 'EXPERT', 'code': 'PA'}}
    ok = mask_ops._merge_review_state(promoted_survivor, promoted_absorbed)
    check(ok is True, "the expert-ID promotion merge is still allowed")
    check(promoted_survivor['species'] == 'PA',
          f"a REVIEW species is not promoted onto the survivor, got {promoted_survivor['species']!r}")
    check(promoted_survivor['category'] == 'Coral',
          f"a Review category is not promoted either, got {promoted_survivor['category']!r}")
    check(promoted_survivor['expert_id'] == {'mode': 'EXPERT', 'code': 'PA'},
          "the accepted expert ID is still promoted")
    check(promoted_survivor['name'] == 'Flagged for expert review',
          "a name that is not the tag is still promoted")


def test_merge_still_carries_review_history():
    """review_uid, reviews[] and expert_id travel exactly as before."""
    survivor = {'id': 1, 'species': 'PA',
                'reviews': [{'reviewer': 'LO', 'code': 'PA', 'confidence': 'high', 'at': 't0'}]}
    absorbed = {'id': 2, 'species': 'PA', 'review_uid': 'uid-b', 'review': True,
                'expert_id': {'mode': 'EXPERT', 'code': 'SS'},
                'reviews': [{'reviewer': 'AB', 'code': 'MC', 'confidence': 'low', 'at': 't1'}]}
    ok = mask_ops._merge_review_state(survivor, absorbed)
    check(ok is True, "the merge is allowed")
    check(survivor['review_uid'] == 'uid-b',
          f"review_uid carried, got {survivor.get('review_uid')!r}")
    check(survivor['expert_id'] == {'mode': 'EXPERT', 'code': 'SS'},
          f"expert_id carried, got {survivor.get('expert_id')!r}")
    check([r['reviewer'] for r in survivor['reviews']] == ['LO', 'AB'],
          f"reviews concatenated survivor first, got {survivor.get('reviews')}")
    check(not survivor.get('review'),
          "the history travels without the boolean the survivor lacked")
    conflict = mask_ops._merge_review_state(
        {'id': 1, 'expert_id': {'mode': 'EXPERT', 'code': 'PA'}},
        {'id': 2, 'expert_id': {'mode': 'EXPERT', 'code': 'SS'}})
    check(conflict is False, "two conflicting accepted expert IDs still refuse the merge")


def test_merge_of_two_already_flagged_masks_is_unchanged():
    """Both sides flagged already: the survivor keeps everything it had."""
    survivor = {'id': 1, 'species': 'REVIEW', 'category': 'Review', 'review': True,
                'review_uid': 'uid-a'}
    absorbed = {'id': 2, 'species': 'REVIEW', 'category': 'Review', 'review': True,
                'review_uid': 'uid-b',
                'expert_id': {'mode': 'EXPERT', 'code': 'REVIEW'}}
    ok = mask_ops._merge_review_state(survivor, absorbed)
    check(ok is True, "the merge is allowed")
    check(survivor['review'] is True, "the survivor's existing flag is preserved")
    check(survivor['species'] == 'REVIEW',
          f"the survivor's existing REVIEW species is preserved, got {survivor['species']!r}")
    check(survivor['category'] == 'Review', "the survivor's Review category is preserved")
    check(survivor['review_uid'] == 'uid-a', "the survivor's own review_uid wins")
    check(survivor['expert_id'] == {'mode': 'EXPERT', 'code': 'REVIEW'},
          "the accepted expert ID is still carried onto an already flagged survivor")


def test_lock_constant_is_the_single_switch():
    """Flipping REVIEW_TAGGING_ENABLED to True restores the old behaviour."""
    check(mask_ops.REVIEW_TAGGING_ENABLED is False,
          "the shipped default is locked")
    check(mask_ops.REVIEW_LOCKED_MSG == (
        "Flagging for expert review is temporarily unavailable. "
        "VICARIUS is accepting no new review tags."),
        f"the refusal message is the agreed text, got {mask_ops.REVIEW_LOCKED_MSG!r}")
    check(mask_ops.REVIEW_TAG == "REVIEW", "the tag constant is REVIEW")
    mask_ops.REVIEW_TAGGING_ENABLED = True
    try:
        mask_ops.reset_review_tags_dropped()
        mask = build({'label': 'A', 'species': 'REVIEW', 'category': 'Review',
                      'review': True, 'x': 50, 'y': 50})
        check(mask['species'] == 'REVIEW',
              f"unlocked: a REVIEW species is born as before, got {mask['species']!r}")
        check(mask['category'] == 'Review', "unlocked: the Review category is born as before")
        check(mask_ops.review_tags_dropped() == 0, "unlocked: nothing is counted as dropped")
        survivor = {'id': 1, 'species': 'PA'}
        mask_ops._merge_review_state(survivor, {'id': 2, 'species': 'PA', 'review': True})
        check(survivor.get('review') is True,
              "unlocked: the merge carries the pending-review flag as before")
        promoted = {'id': 1, 'species': 'PA'}
        mask_ops._merge_review_state(promoted, {'id': 2, 'species': 'REVIEW',
                                                'expert_id': {'mode': 'EXPERT', 'code': 'PA'}})
        check(promoted['species'] == 'REVIEW',
              "unlocked: the species promotion runs as before")
    finally:
        mask_ops.REVIEW_TAGGING_ENABLED = False
    relocked = build({'label': 'A', 'species': 'REVIEW', 'x': 50, 'y': 50})
    check(relocked['species'] == '', "locked again: the one constant is the whole switch")
    mask_ops.reset_review_tags_dropped()


# Near-spellings of the tag built out of characters that are invisible or that
# belong to another script. Each renders as REVIEW in the mask registry and the
# coverage matrix, so each would be a junk species in the canonical record if it
# were written there. Kept as escapes rather than literals so the character
# under test is readable in the source.
_UNICODE_NEAR_SPELLINGS = (
    ("REVIEW\u200b", "a trailing zero width space"),
    ("\u200bREVIEW", "a leading zero width space"),
    ("RE\u200bVIEW", "a zero width space inside the word"),
    ("REVIEW\ufeff", "a trailing byte order mark"),
    ("RE\u00adVIEW", "a soft hyphen inside the word"),
    ("REVIE\u2060W", "a word joiner inside the word"),
    ("\u0420EVIEW", "a Cyrillic Er standing in for the R"),
    ("R\u0415VIEW", "a Cyrillic Ie standing in for the first E"),
    ("REVIE\uff37", "a fullwidth W"),
    ("\uff32\uff25\uff36\uff29\uff25\uff37", "the whole word in fullwidth letters"),
    ("rev\u0456ew", "a Cyrillic dotted I in the lower case spelling"),
)

# Values that contain the word, or read a little like it, and are ordinary
# labels rather than the tag. Every one of these has to survive the refusal
# untouched, because refusing them would blank a real species.
_NOT_THE_TAG = (
    ("REVIEWED", "a species that ends in the word"),
    ("PREREVIEW", "a species that starts with the word"),
    ("REVIEW_PA", "a species that carries the word as a prefix"),
    ("RE VIEW", "the word split by a real space"),
    ("RE-VIEW", "the word split by a hyphen"),
    ("REVIEWS", "the plural"),
    ("PREVIEW", "a word one letter short of it"),
    ("PA", "an ordinary species code"),
    ("OFAV", "a longer ordinary species code"),
    ("UNK", "the unknown code"),
    ("Coral", "an ordinary category"),
    ("Target species only", "the shipped default category"),
)


def test_dropped_tag_blanks_the_name_as_well_as_species_and_category():
    """A mask whose tag the lock drops loses its REVIEW display name too.

    The name reaches no predicate and no registry column, so it is display
    only, but a mask reading 'Flagged for expert review' beside an empty
    species says the opposite of what the mask is. The guard that blanks the
    species and the category blanks the name in the same breath, and a name
    that is a real species name is left alone, exactly as a real species is.
    """
    mask_ops.reset_review_tags_dropped()
    flagged_mask = build({'label': 'A', 'species': 'REVIEW', 'category': 'Review',
                          'review': True, 'name': 'Flagged for expert review',
                          'x': 50, 'y': 50})
    check(flagged_mask['species'] == '' and flagged_mask['category'] == '',
          f"species and category are blanked, got {flagged_mask['species']!r} "
          f"and {flagged_mask['category']!r}")
    check(flagged_mask['name'] == '',
          f"the review display name is blanked with them, got "
          f"{flagged_mask['name']!r}")
    candidate = build({'label': 'B', 'species': 'REVIEW', 'name': 'REVIEW',
                       'x': 50, 'y': 50})
    check(candidate['name'] == '',
          f"an exemplar candidate name of REVIEW is blanked, got "
          f"{candidate['name']!r}")
    for spelling in _REVIEW_SPELLINGS + ('flagged for expert review',
                                         ' Flagged For Expert Review '):
        cased = build({'label': 'C', 'species': 'REVIEW', 'name': spelling,
                       'x': 50, 'y': 50})
        check(cased['name'] == '',
              f"name={spelling!r} is blanked whatever its case or padding, got "
              f"{cased['name']!r}")
    real_name = build({'label': 'D', 'species': 'PA', 'category': 'Review',
                       'name': 'Porites astreoides', 'x': 50, 'y': 50})
    check(real_name['species'] == 'PA' and real_name['name'] == 'Porites astreoides',
          f"a real species and its real name survive a Review category, got "
          f"{real_name['species']!r} and {real_name['name']!r}")
    mask_ops.reset_review_tags_dropped()
    mask_ops.REVIEW_TAGGING_ENABLED = True
    try:
        unlocked = build({'label': 'E', 'species': 'REVIEW', 'review': True,
                          'name': 'Flagged for expert review', 'x': 50, 'y': 50})
        check(unlocked['name'] == 'Flagged for expert review',
              f"unlocked: the review display name is written as before, got "
              f"{unlocked['name']!r}")
    finally:
        mask_ops.REVIEW_TAGGING_ENABLED = False
    mask_ops.reset_review_tags_dropped()


def test_unicode_near_spellings_are_caught():
    """Invisible characters and foreign letters do not smuggle the tag through.

    Python's str.strip() removes no zero width space, and NFKC folds a
    fullwidth letter but not a Cyrillic one, so each of these values used to
    reach the mask registry as a species that renders exactly like REVIEW.
    Each is now recognised as the tag and blanked like any other spelling.
    """
    mask_ops.reset_review_tags_dropped()
    for spelling, label in _UNICODE_NEAR_SPELLINGS:
        check(mask_ops.normalise_review_value(spelling) == mask_ops.REVIEW_TAG,
              f"{label} normalises onto the tag, got "
              f"{mask_ops.normalise_review_value(spelling)!r}")
        for key in ('species', 'species_code', 'category'):
            check(mask_ops.review_tag_requested({key: spelling}),
                  f"{label} in {key} is a review tag request")
            stripped = mask_ops.strip_review_tag({key: spelling})
            check(stripped[key] == '',
                  f"{label} in {key} is blanked by the scrub, got {stripped[key]!r}")
        mask = build({'label': 'A', 'species': spelling, 'x': 50, 'y': 50})
        check(mask['species'] == '',
              f"{label} as a species is born blank, got {mask['species']!r}")
    check(mask_ops.review_tags_dropped() == len(_UNICODE_NEAR_SPELLINGS),
          f"every near spelling counts one drop, expected "
          f"{len(_UNICODE_NEAR_SPELLINGS)}, got {mask_ops.review_tags_dropped()}")
    mask_ops.reset_review_tags_dropped()


def test_a_species_containing_the_word_is_still_allowed():
    """A label that merely contains the word is a real label, not the tag.

    The refusal compares the whole value with the tag, never a substring, so
    REVIEWED, PREREVIEW, REVIEW_PA and RE VIEW are ordinary species that pass
    through untouched. Widening the refusal to a substring test would blank
    them, which is why this case sits beside the near-spelling one.
    """
    mask_ops.reset_review_tags_dropped()
    for value, label in _NOT_THE_TAG:
        for key in ('species', 'species_code', 'category'):
            check(not mask_ops.review_tag_requested({key: value}),
                  f"{label} in {key} is not a review tag request")
            kept = mask_ops.strip_review_tag({key: value})
            check(kept[key] == value,
                  f"{label} in {key} survives the scrub, got {kept[key]!r}")
        mask = build({'label': 'A', 'species': value, 'name': value,
                      'category': 'Coral', 'x': 50, 'y': 50})
        check(mask['species'] == value and mask['name'] == value,
              f"{label} is born with its species and name intact, got "
              f"{mask['species']!r} and {mask['name']!r}")
    check(mask_ops.review_tags_dropped() == 0,
          f"not one of them counts as a dropped tag, got "
          f"{mask_ops.review_tags_dropped()}")


TESTS = (
    test_review_tag_requested_catches_every_spelling,
    test_review_tag_requested_catches_the_category_spelling,
    test_a_real_species_survives_a_review_category,
    test_review_tag_requested_tolerates_junk,
    test_strip_review_tag_never_mutates_its_argument,
    test_build_mask_dict_never_births_a_flagged_mask,
    test_build_mask_dict_blanks_a_category_only_review,
    test_build_mask_dict_leaves_an_ordinary_species_alone,
    test_dropped_tag_blanks_the_name_as_well_as_species_and_category,
    test_unicode_near_spellings_are_caught,
    test_a_species_containing_the_word_is_still_allowed,
    test_merge_does_not_give_the_survivor_a_flag_it_lacked,
    test_merge_still_carries_review_history,
    test_merge_of_two_already_flagged_masks_is_unchanged,
    test_lock_constant_is_the_single_switch,
)


if __name__ == "__main__":
    for test in TESTS:
        print(test.__name__)
        try:
            test()
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{test.__name__} raised {type(exc).__name__}: {exc}")
    print("PASS" if _fail == 0 else f"{_fail} FAILED")
    sys.exit(1 if _fail else 0)
