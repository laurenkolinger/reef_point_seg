"""
Module: model_card_hook.py
Purpose: Write the model card that sits beside a model VEGA's Train step just
         finished, and infer the two finding tags (subject and training data
         source) the Train form pre-fills. The card code is the platform's
         shared writer under VICARIUS_ROOT/_METADATA/model_card, imported by
         path; it needs only the standard library and PyYAML.
Inputs:  a finished run folder (<step 6 folder>/runs/<run>/ holding
         weights/best.pt and args.yaml), the step 6 config the launch saved,
         the creator's initials, and, for infer_tags, a training source folder
         (data.yaml and its image folders).
Outputs: <run>/weights/best.pt.card.md; a dict of two tags. A card failure is
         reported through the log callable and never raised.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Callable, Optional

DEFAULT_VICARIUS_ROOT = "/mnt/rip/vicarius_drive/vicarius"
MODEL_CARD_FOLDER = ("_METADATA", "model_card")
TRAIN_LAYOUT = "vega_run"
TRAIN_WRITER = "VEGA Train"
WEIGHTS_RELATIVE = ("weights", "best.pt")
HASH_CHUNK_BYTES = 1024 * 1024
CONFIG_SUBJECT_KEY = "card_subject"
CONFIG_DATA_SOURCE_KEY = "card_data_source"
# The Train form also saves card_subject_picked and card_data_source_picked (true when a person chose the row, Not set included); only the form reads them, this module ignores them.
SOURCE_IMAGE_FOLDERS = ("all_images", "images", "train/images", "valid/images", "test/images")
CARD_ERRORS = (OSError, ValueError, KeyError, TypeError)


def vicarius_root() -> Path:
    """The platform tree, from VICARIUS_ROOT when set, else the default install."""
    return Path(os.environ.get("VICARIUS_ROOT") or DEFAULT_VICARIUS_ROOT)


def load_card_modules(root: Optional[Path] = None):
    """Import the shared card writer and builder by path.

    Parameters:
        root: the platform tree; defaults to vicarius_root().
    Returns:
        (card, build) the two modules.
    Raises:
        ImportError naming the folder when it is missing or lacks a module.
    """
    folder = Path(root or vicarius_root()).joinpath(*MODEL_CARD_FOLDER)
    if not folder.is_dir():
        raise ImportError(f"the shared model card code is not at {folder}")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import build  # noqa: E402
    import card  # noqa: E402
    return card, build


def sha256_file(path: Path) -> str:
    """Lowercase hex SHA-256 of a file, read in 1 MiB chunks.

    Parameters:
        path: the file to hash.
    Returns:
        The digest. OSError propagates with the path in its message.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_image_names(source_dir: Path, build) -> list:
    """File names found in the source's image folders (capped by the builder)."""
    names: list = []
    for folder in SOURCE_IMAGE_FOLDERS:
        names += build._list_names(source_dir / folder)
    return names[: build.MAX_IMAGE_NAMES]


def infer_tags(step6_config: dict, source_dir) -> dict:
    """Infer the subject and training data source of a training source.

    The rules are the card builder's: subject from how the class names resolve
    (fish names or the TCRMP species table), data source from how many image
    names follow the TCRMP pattern.

    Parameters:
        step6_config: the Train form's config (kept for future rules; unused
            today because the recipe subject defaults to coral for every run).
        source_dir: the training source folder (data.yaml, image folders).
    Returns:
        {"subject": "fish" | "benthic" | None, "data_source": "tcrmp" | "other" | None}.
        A tag is None when no rule applies. Never raises for an unreadable
        source; it returns both tags None.

    Example: a folder of TCRMP20140926_clip_BWR_T106.jpg frames with coral
    species codes gives {"subject": "benthic", "data_source": "tcrmp"}.
    """
    empty = {"subject": None, "data_source": None}
    try:
        _, build = load_card_modules()
        refs = build.load_references(vicarius_root())
        folder = Path(source_dir)
        data = build._read_data_yaml(folder / "data.yaml")
        classes = _classes_of(data, build, refs)
        subject, _ = build.infer_subject(classes, None, TRAIN_LAYOUT)
        source, _ = build.infer_data_source(_source_image_names(folder, build), None, TRAIN_LAYOUT)
    except (ImportError, ValueError, OSError, KeyError, TypeError):
        return empty
    return {"subject": subject, "data_source": source}


def _classes_of(data: dict, build, refs) -> list:
    """Grouped classes of a data.yaml reading; [] when it names none."""
    if not data or not data.get("names"):
        return []
    return build._classes(data, refs)


def _set_person_tags(card, weights: Path, config: dict, initials: str) -> list:
    """Write the form's two tags to the card's person block.

    Returns:
        A list of short problem texts, one per tag that could not be written.
    """
    problems = []
    for field, key in (("subject", CONFIG_SUBJECT_KEY), ("data_source", CONFIG_DATA_SOURCE_KEY)):
        value = str((config or {}).get(key) or "").strip().lower()
        if not value:
            continue
        try:
            card.edit_person(weights, field, value, initials)
        except card.CardError as exc:
            problems.append(f"{field}: {exc}")
    return problems


def write_train_card(run_dir, step6_config: dict, initials: str,
                     log: Callable[[str], None] = print) -> Optional[Path]:
    """Write the card beside a finished run's best.pt.

    Builds the machine-filled record, sets the weights checksum, writes it with
    refreshed_by "VEGA Train", then writes the person's subject and data source
    tags with the creator's initials. A failure is reported as one line through
    `log` and the function returns None (or the card path when only a tag could
    not be written); it never raises, so a card problem cannot fail the step.

    Parameters:
        run_dir: <step 6 folder>/runs/<run>.
        step6_config: the step 6 config saved at launch.
        initials: the creator's initials.
        log: called with one line on a failure.
    Returns:
        The card path, or None when no card was written.
    """
    weights = Path(run_dir).joinpath(*WEIGHTS_RELATIVE)
    if not weights.is_file():
        log(f"[model_card] no card written: {weights} is not on disk")
        return None
    try:
        card, build = load_card_modules()
    except ImportError as exc:
        log(f"[model_card] no card written for {weights}: {exc}")
        return None
    try:
        record = build.build_record(weights, TRAIN_LAYOUT, build.load_references(vicarius_root()))
        record.setdefault("weights", {})["sha256"] = sha256_file(weights)
        path = card.write_record(weights, record, TRAIN_WRITER)
        problems = _set_person_tags(card, weights, step6_config, initials)
    except CARD_ERRORS + (card.CardError,) as exc:
        log(f"[model_card] no card written for {weights}: {exc}")
        return None
    if problems:
        log(f"[model_card] card written at {path}, but a tag was not saved ({'; '.join(problems)})")
    return path

