"""Full names for the codes the matrix shows (spec section 6: never a bare code
when the registry can resolve it). Two tiny CSV readers with an mtime cache."""
import csv
import os
import threading

_LOCK = threading.Lock()
_CACHE = {}   # path -> (mtime, size, dict)


def _read(path, key_col, val_col):
    """Read a two-column code -> name table out of a CSV file, caching the
    result in-process so a call made on every request only re-parses the
    file when it actually changed on disk.

    Parameters:
        path (str): path to the CSV file. A falsy path, or a path that is
            not an existing regular file, resolves to {} without raising.
        key_col (str): name of the CSV column supplying each dict key.
        val_col (str): name of the CSV column supplying each dict value.

    Returns:
        dict[str, str]: key_col value -> val_col value for every row where
        both are non-blank after stripping surrounding whitespace (values
        are otherwise returned verbatim: embedded HTML, tabs, and NUL bytes
        are preserved). A duplicate key is overwritten by the last matching
        row. Always {} rather than a raised exception when the file is
        missing, unreadable (including permission-denied), has no header
        row, or lacks key_col or val_col; a UTF-8 byte-order mark on the
        header line is stripped rather than treated as part of the first
        column's name.
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        st = os.stat(path)
    except OSError:
        return {}
    stamp = (st.st_mtime, st.st_size)
    with _LOCK:
        hit = _CACHE.get(path)
        if hit and hit[0] == stamp:
            return dict(hit[1])
    out = {}
    try:
        # utf-8-sig: strips a leading BOM when one is present (some tools,
        # notably Excel, save CSV with one), and is otherwise identical to
        # plain utf-8 when there is no BOM. Without this, a BOM'd header
        # leaves the first column named '﻿code' instead of 'code' and
        # every row in the file silently resolves to {}.
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or key_col not in reader.fieldnames or val_col not in reader.fieldnames:
                return {}
            for row in reader:
                k = (row.get(key_col) or "").strip()
                v = (row.get(val_col) or "").strip()
                if k and v:
                    out[k] = v
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    with _LOCK:
        _CACHE[path] = (stamp, dict(out))
    return out


def read_label_names(path):
    """code -> name from master_codes.csv (columns code,category,name)."""
    return _read(path, "code", "name")


def read_site_names(path):
    """site_code -> site from site_codes.csv (columns site,site_code)."""
    return _read(path, "site_code", "site")
