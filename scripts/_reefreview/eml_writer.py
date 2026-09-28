"""EML (Ecological Metadata Language) writer for reef_point_seg exports.

Emits a minimal EML 2.2.0 document (eml.xml) beside a YOLO-seg export or a
composite export, modeled on GR/EML_SAMPLE.xml. Stdlib only: the XML is
built by hand with proper escaping (xml.sax.saxutils.escape) rather than
pulling in a templating dependency, and parses back cleanly with
xml.etree.ElementTree.

Callers whose export must never fail on a metadata problem (export_yolo's
export_batch, _matrix/composite.py's build_composite) wrap the write_eml
call in their own try/except and log a warning instead of raising.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

# Atlantic Standard Time: fixed UTC-4, no DST (platform rule: every VICARIUS
# timestamp is AST 24h). A local constant rather than importing clock.py so
# this module has zero dependencies beyond the stdlib.
AST = timezone(timedelta(hours=-4))

DEFAULT_CREATOR = "Lauren Olinger"
DEFAULT_EMAIL = "lauren.olinger@uvi.edu"

_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _now_ast_date():
    """'YYYY-MM-DD' in AST, for pubDate."""
    return datetime.now(AST).strftime("%Y-%m-%d")


def _now_ast_display():
    """'YYYY-MM-DD HH:MM AST' in AST 24h, for the generation stamp."""
    return datetime.now(AST).strftime("%Y-%m-%d %H:%M") + " AST"


def _short_name(dataset_name):
    """Collapse dataset_name to an EML shortName/alternateIdentifier: only
    letters, digits and underscore survive; a blank or all-punctuation
    input falls back to a generic name so the document is always valid."""
    cleaned = _NAME_RE.sub("_", str(dataset_name or "").strip()).strip("_")
    return cleaned or "reef_point_seg_export"


def _split_creator(creator):
    """'Lauren Olinger' -> ('Lauren', 'Olinger'); a middle name goes with
    the given name ('A B Smith' -> ('A B', 'Smith')). A single-word creator
    keeps the whole string as surName with an empty givenName rather than
    guessing at a shape it cannot parse."""
    parts = str(creator or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


def _package_id(short_name):
    return "vicarius.%s.%s" % (short_name, uuid.uuid4().hex[:8])


def _individual_block(indent, given, sur, email):
    given_line = "%s  <givenName>%s</givenName>\n" % (indent, escape(given)) if given else ""
    return (
        "%s<individualName>\n"
        "%s%s  <surName>%s</surName>\n"
        "%s</individualName>\n"
        "%s<electronicMailAddress>%s</electronicMailAddress>\n"
    ) % (indent, given_line, indent, escape(sur), indent, indent, escape(email))


def _rows_data_table(short_name, rows):
    """A minimal counts-only dataTable block: the row count plus the sorted
    field names seen across the rows, with no per-row data retained in the
    metadata document. Returns '' when rows is falsy (no dataTable emitted)."""
    if not rows:
        return ""
    fields = sorted({k for row in rows if isinstance(row, dict) for k in row.keys()})
    if fields:
        attr_lines = "\n".join(
            "        <attribute>\n"
            "          <attributeName>%s</attributeName>\n"
            "        </attribute>" % escape(str(f))
            for f in fields
        )
    else:
        attr_lines = (
            "        <attribute>\n"
            "          <attributeName>value</attributeName>\n"
            "        </attribute>"
        )
    return (
        "    <dataTable>\n"
        "      <entityName>%s_rows</entityName>\n"
        "      <entityDescription>%d rows summarized from this export "
        "(counts only; no row-level data retained in this metadata document).</entityDescription>\n"
        "      <attributeList>\n"
        "%s\n"
        "      </attributeList>\n"
        "    </dataTable>\n"
    ) % (escape(short_name), len(rows), attr_lines)


def write_eml(out_dir, dataset_name, description, creator=DEFAULT_CREATOR, rows=None):
    """Write <out_dir>/eml.xml describing one reef_point_seg export.

    Args:
        out_dir: directory the export already lives in; created if absent.
        dataset_name: short identifier for the export (project id, run
            name, composite name); used for shortName/title/packageId.
        description: one-line human summary (mask counts etc.); becomes
            the <abstract>.
        creator: 'Given Sur' (default 'Lauren Olinger'); split on the last
            space into givenName/surName.
        rows: optional list of dicts, summarized as a minimal counts-only
            dataTable block (row count + sorted field names, no per-row
            detail). Omitted entirely from the document when falsy.

    Returns:
        the path written (<out_dir>/eml.xml).

    Raises on a write failure (permissions, missing parent, etc). Callers
    that must never let an EML problem break their export wrap this call
    in their own try/except and log a warning instead.
    """
    os.makedirs(out_dir, exist_ok=True)

    short_name = _short_name(dataset_name)
    given, sur = _split_creator(creator)
    package_id = _package_id(short_name)
    title = escape(str(dataset_name or short_name).strip() or short_name)
    abstract = escape(str(description or "").strip())
    creator_block = _individual_block("      ", given, sur, DEFAULT_EMAIL)
    contact_block = _individual_block("      ", given, sur, DEFAULT_EMAIL)
    data_table = _rows_data_table(short_name, rows)

    xml = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<eml xmlns="https://eml.ecoinformatics.org/eml-2.2.0" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="https://eml.ecoinformatics.org/eml-2.2.0 '
        'https://eml.ecoinformatics.org/eml-2.2.0/eml.xsd" '
        'packageId="%s" system="vicarius" scope="system">\n'
        "  <dataset>\n"
        "    <alternateIdentifier>%s</alternateIdentifier>\n"
        "    <shortName>%s</shortName>\n"
        "    <title>%s</title>\n"
        "    <creator>\n"
        "%s"
        "    </creator>\n"
        "    <contact>\n"
        "%s"
        "    </contact>\n"
        "    <pubDate>%s</pubDate>\n"
        "    <abstract>\n"
        "      <para>%s</para>\n"
        "    </abstract>\n"
        "%s"
        "  </dataset>\n"
        "  <additionalMetadata>\n"
        "    <metadata>\n"
        "      <generatedAt>%s</generatedAt>\n"
        "      <generatedBy>reef_point_seg</generatedBy>\n"
        "    </metadata>\n"
        "  </additionalMetadata>\n"
        "</eml>\n"
    ) % (
        escape(package_id), escape(short_name), escape(short_name), title,
        creator_block, contact_block, _now_ast_date(), abstract, data_table,
        _now_ast_display(),
    )

    path = os.path.join(out_dir, "eml.xml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path
