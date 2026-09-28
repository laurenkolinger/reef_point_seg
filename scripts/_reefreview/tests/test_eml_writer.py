"""eml_writer.write_eml: well-formed XML, required fields, AST timestamp,
no em/en dashes, rows summarized as counts-only.

Run: env/bin/python scripts/_reefreview/tests/test_eml_writer.py
"""
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))        # scripts/_reefreview
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # scripts/ (for _reefreview pkg)

import eml_writer as EW  # noqa: E402

_fail = 0


def check(c, m):
    global _fail
    print(("  PASS " if c else "  FAIL ") + m)
    if not c:
        _fail += 1


NS = {"eml": "https://eml.ecoinformatics.org/eml-2.2.0"}

d = tempfile.mkdtemp(prefix="eml_writer_")
try:
    path = EW.write_eml(
        d,
        "reef_point_seg_test_export",
        "3 images, 5 masks exported for T101.",
        rows=[{"basename": "a", "label": "PA"}, {"basename": "b", "label": "OFAV"}],
    )
    check(os.path.isfile(path), "write_eml returns a path that exists")
    check(path == os.path.join(d, "eml.xml"), "eml.xml written directly under out_dir")

    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    try:
        root = ET.fromstring(raw)
        parsed_ok = True
    except ET.ParseError:
        parsed_ok = False
        root = None
    check(parsed_ok, "eml.xml parses back as well-formed XML")

    check("—" not in raw and "–" not in raw,
          "no em dash or en dash anywhere in eml.xml")

    if root is not None:
        dataset = root.find("eml:dataset", NS)
        check(dataset is not None, "eml/dataset present")

        title = dataset.find("eml:title", NS) if dataset is not None else None
        check(title is not None and (title.text or "").strip() != "", "title present and non-empty")

        given = dataset.find("eml:creator/eml:individualName/eml:givenName", NS)
        sur = dataset.find("eml:creator/eml:individualName/eml:surName", NS)
        check(given is not None and given.text == "Lauren",
              f"default creator givenName == Lauren (got {given.text if given is not None else None})")
        check(sur is not None and sur.text == "Olinger",
              f"default creator surName == Olinger (got {sur.text if sur is not None else None})")
        email = dataset.find("eml:creator/eml:electronicMailAddress", NS)
        check(email is not None and "@" in (email.text or ""), "creator electronicMailAddress present")

        contact_sur = dataset.find("eml:contact/eml:individualName/eml:surName", NS)
        check(contact_sur is not None and contact_sur.text == "Olinger", "contact surName present")

        abstract = dataset.find("eml:abstract/eml:para", NS)
        check(abstract is not None and "3 images" in (abstract.text or ""),
              "abstract carries the description text")

        pubdate = dataset.find("eml:pubDate", NS)
        check(pubdate is not None and re.match(r"^\d{4}-\d{2}-\d{2}$", pubdate.text or "") is not None,
              f"pubDate is YYYY-MM-DD (got {pubdate.text if pubdate is not None else None})")

        gen = root.find("eml:additionalMetadata/eml:metadata/eml:generatedAt", NS)
        check(gen is not None, "additionalMetadata/metadata/generatedAt present")
        if gen is not None:
            check(re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} AST$", gen.text or "") is not None,
                  f"generatedAt formatted 'YYYY-MM-DD HH:MM AST' (got {gen.text})")

        dt = dataset.find("eml:dataTable", NS)
        check(dt is not None, "dataTable present when rows given")
        if dt is not None:
            desc = dt.find("eml:entityDescription", NS)
            check(desc is not None and "2 rows" in (desc.text or ""),
                  f"entityDescription states the row count (got {desc.text if desc is not None else None})")
            attr_names = [a.text for a in dt.findall("eml:attributeList/eml:attribute/eml:attributeName", NS)]
            check(sorted(attr_names) == ["basename", "label"],
                  f"attributeList lists the sorted field names, no row values (got {attr_names})")
            check("PA" not in raw.split("<dataTable>")[1] if "<dataTable>" in raw else True,
                  "dataTable holds no actual row values (counts only)")

    # No rows -> no dataTable; custom creator splits on the last space;
    # out_dir is created if it does not already exist.
    sub = os.path.join(d, "sub", "nested")
    path2 = EW.write_eml(sub, "composite_test", "composite built.", creator="A B Smith")
    check(os.path.isfile(path2), "write_eml creates a missing out_dir")
    with open(path2, encoding="utf-8") as fh:
        raw2 = fh.read()
    root2 = ET.fromstring(raw2)
    dt2 = root2.find("eml:dataset/eml:dataTable", NS)
    check(dt2 is None, "no dataTable block when rows is None")
    given2 = root2.find("eml:dataset/eml:creator/eml:individualName/eml:givenName", NS)
    sur2 = root2.find("eml:dataset/eml:creator/eml:individualName/eml:surName", NS)
    check(given2 is not None and given2.text == "A B",
          f"multi-word creator splits on the last space for givenName (got {given2.text if given2 is not None else None})")
    check(sur2 is not None and sur2.text == "Smith",
          f"multi-word creator splits on the last space for surName (got {sur2.text if sur2 is not None else None})")

    # Dataset name with characters outside [A-Za-z0-9_] is sanitized for
    # shortName/alternateIdentifier/packageId but the raw text still shows
    # in <title>.
    path3 = EW.write_eml(os.path.join(d, "sub2"), "run 42 (T101/T102)", "sanitize check.")
    with open(path3, encoding="utf-8") as fh:
        raw3 = fh.read()
    root3 = ET.fromstring(raw3)
    short = root3.find("eml:dataset/eml:shortName", NS)
    check(short is not None and re.match(r"^[A-Za-z0-9_]+$", short.text or "") is not None,
          f"shortName sanitized to [A-Za-z0-9_] (got {short.text if short is not None else None})")
    title3 = root3.find("eml:dataset/eml:title", NS)
    check(title3 is not None and title3.text == "run 42 (T101/T102)",
          f"title keeps the original dataset_name text (got {title3.text if title3 is not None else None})")
    pkg_id = root3.attrib.get("packageId", "")
    check(pkg_id.startswith("vicarius.") and " " not in pkg_id,
          f"packageId is vicarius.<shortName>.<hex>, no spaces (got {pkg_id})")

finally:
    shutil.rmtree(d, ignore_errors=True)

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
