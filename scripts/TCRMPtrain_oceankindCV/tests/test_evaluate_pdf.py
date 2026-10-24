#!/usr/bin/env python3
"""Tests for evaluate_run.py's PDF path: the HTML assembly with the stylesheet
inlined in the head, the headless Chrome print wrapper and its success check,
the page plan that keeps every printed page number honest, and main()'s exit
code when a requested output is missing.

No GPU and no ultralytics import: metrics, context and previews are
fabricated, and the HTML assembly and the Chrome wrapper are called directly.
Cases that print a real PDF use the Chrome on this box (or the path in
TCRMP_CHROME_BINARY) and read the result back with pdfinfo, pdftotext and
pdfimages when those exist under /usr/bin.
Run: env/bin/python scripts/TCRMPtrain_oceankindCV/tests/test_evaluate_pdf.py"""
# Keep this suite off the live mask registry, expert library and clip index
# when it is run alone: scripts/tests/_isolation.py sets the isolating
# variables to temp dirs and blocks any write to the live stores.
import os as _iso_os
import sys as _iso_sys
_iso_dir = _iso_os.path.dirname(_iso_os.path.abspath(__file__))
while _iso_os.path.basename(_iso_dir) != 'scripts' and _iso_dir != _iso_os.path.dirname(_iso_dir):
    _iso_dir = _iso_os.path.dirname(_iso_dir)
_iso_sys.path.insert(0, _iso_os.path.join(_iso_dir, 'tests'))
import _isolation
_isolation.install()

import contextlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import types

import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import evaluate_run as ER
from test_evaluate_metrics import FakeConfusionMatrix, FakeMetric, FakeResults, val_out_for

SRC_PATH = os.path.join(HERE, "..", "src", "evaluate_run.py")
PDFTOTEXT = "/usr/bin/pdftotext"
PDFIMAGES = "/usr/bin/pdfimages"
FOOTER_RE = re.compile(r"p\.\s*(\d+)\s*/\s*(\d+)")
AST_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} AST")
# A stub that sleeps this long must outlive the short timeout the test sets.
STUB_SLEEP_SECONDS = 20
SHORT_TIMEOUT_SECONDS = 2

_results = []


def check(cond, msg):
    """Raise AssertionError with msg when cond is false."""
    if not cond:
        raise AssertionError(msg)


def run(name, fn):
    """Run one named case, print PASS or FAIL, and record the outcome."""
    try:
        fn()
        print(f"PASS {name}")
        _results.append((name, True))
    except Exception as exc:
        print(f"FAIL {name}: {exc}")
        traceback.print_exc()
        _results.append((name, False))


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def scratch(prefix):
    """A fresh temporary folder for one case."""
    return tempfile.mkdtemp(prefix=f"eval_pdf_{prefix}_")


def write_stub_chrome(folder, body):
    """Write an executable shell script standing in for Chrome and return its path.

    body runs with Chrome's arguments in $@; $PDF holds the --print-to-pdf target.
    """
    path = os.path.join(folder, "stub_chrome.sh")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#!/bin/sh\n")
        fh.write('PDF=""\nfor a in "$@"; do case "$a" in --print-to-pdf=*) PDF="${a#--print-to-pdf=}";; esac; done\n')
        fh.write(body + "\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def pdfinfo_pages(pdf_path):
    """Page count from pdfinfo, or None when pdfinfo is not installed."""
    if not os.path.isfile(ER.PDFINFO_BINARY):
        return None
    out = subprocess.run([ER.PDFINFO_BINARY, pdf_path], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise AssertionError(f"pdfinfo printed no Pages line for {pdf_path}: {out}")


def pdf_page_text(pdf_path, page):
    """Text of one page (1-based) via pdftotext; None when pdftotext is not installed."""
    if not os.path.isfile(PDFTOTEXT):
        return None
    return subprocess.run([PDFTOTEXT, "-f", str(page), "-l", str(page), "-layout", pdf_path, "-"],
                          capture_output=True, text=True, check=True).stdout


def pdf_image_count(pdf_path):
    """Number of embedded images per pdfimages -list; None when pdfimages is not installed."""
    if not os.path.isfile(PDFIMAGES):
        return None
    out = subprocess.run([PDFIMAGES, "-list", pdf_path], capture_output=True, text=True, check=True).stdout
    rows = [ln for ln in out.splitlines()[2:] if ln.strip()]
    return len(rows)


def class_names(n):
    """n class tokens shaped like the real dataset's (Genus-species)."""
    genera = ["Acropora", "Agaricia", "Diploria", "Madracis", "Meandrina", "Mycetophyllia",
              "Orbicella", "Porites", "Siderastrea", "Solenastrea"]
    species = ["cervicornis", "agaricites", "strigosa", "decactis", "meandrites", "lamarckiana",
               "annularis", "astreoides", "siderea", "bournoni"]
    return [f"{genera[i % 10]}-{species[(i // 10) % 10]}{'' if i < 100 else i}" for i in range(n)]


def fabricate_metrics(names, absent=(), family=ER.METRIC_FAMILY_BOX):
    """A metrics dict shaped like extract_metrics() output for the given class names."""
    per_class = {}
    for i, nm in enumerate(names):
        if nm in absent:
            continue
        ap = round(0.9 - (i % 40) * 0.02, 3)
        row = {ER.family_key(family, "precision"): 0.6, ER.family_key(family, "recall"): 0.55,
               ER.family_key(family, "map50"): min(1.0, ap + 0.1), ER.family_key(family, "map50_95"): ap}
        per_class[nm] = row
    overall = {ER.family_key(f, s): None for f in ER.METRIC_FAMILIES for s in ER.METRIC_SUFFIXES}
    overall.update({ER.family_key(family, "precision"): 0.61, ER.family_key(family, "recall"): 0.56,
                    ER.family_key(family, "map50"): 0.7, ER.family_key(family, "map50_95"): 0.5})
    return {"wall_seconds": 4.2, "weights": "/w/best.pt", "data_yaml": "/d/data.yaml",
            "class_names": {i: nm for i, nm in enumerate(names)}, "overall": overall,
            "per_class": per_class, "absent_classes": list(absent),
            "per_class_instances": {nm: (0 if nm in absent else 7) for nm in names},
            "per_class_instances_source": ER.INSTANCE_SOURCE_VALIDATOR,
            "metric_family": family, "fitness": 0.52, "speed_ms": {"inference": 1.5}}


def fabricate_previews(folder, filenames):
    """Write one small JPG per filename under folder/previews and return preview records."""
    prev_dir = os.path.join(folder, "previews")
    os.makedirs(prev_dir, exist_ok=True)
    previews = []
    for i, fn in enumerate(filenames):
        path = os.path.join(prev_dir, fn)
        Image.new("RGB", (160, 80), (226, 0, 116) if i % 2 else (40, 40, 40)).save(path, "JPEG")
        previews.append({"filename": fn, "overlay": path, "n_gt": i + 1, "n_pred": i})
    return previews


class NoPredictionModel:
    """A stand-in for a loaded ultralytics model that predicts nothing.

    render_previews needs only the ground-truth side of a preview to be drawn,
    so a model that returns no results keeps the preview path off the GPU and
    out of an ultralytics import.
    """

    def predict(self, image_path, **kwargs):
        """Return no results, the way a model that finds nothing would."""
        return []

def fake_ctx(tmp):
    """A context dict whose run and dataset folders exist but hold no plots."""
    run_dir = os.path.join(tmp, "runs", "run1")
    dataset_dir = os.path.join(tmp, "dataset")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    return {"run_dir": run_dir, "dataset_dir": dataset_dir,
            "train_args": {"model": "yolo11n.pt", "imgsz": 256, "batch": -1, "epochs": 20},
            "split_counts": {"train": 10, "valid": 2, "test": 2},
            "epochs_completed": 20, "best_epoch": 18}


def fake_args(out_dir):
    """The argparse namespace fields build_pdf reads."""
    return types.SimpleNamespace(split="test", conf=0.25, iou=0.6, imgsz=256, out_dir=out_dir,
                                 preview_count=8, pdf_export_dir="", rounds_dir="")


def make_dataset_dir(root, names):
    """A minimal split dataset the ledger writer can read: data.yaml, image folders, a pinned manifest."""
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.dump({"path": root, "train": "train/images", "val": "valid/images",
                   "test": "test/images", "names": {i: nm for i, nm in enumerate(names)}}, fh)
    for split, n in (("train", 4), ("valid", 2), ("test", 2)):
        img_dir = os.path.join(root, split, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i in range(n):
            with open(os.path.join(img_dir, f"{split}_{i}.jpeg"), "wb") as fh:
                fh.write(b"\xff\xd8\xff\xd9")
    lbl_dir = os.path.join(root, "train", "labels")
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(4):
        with open(os.path.join(lbl_dir, f"train_{i}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{i % len(names)} 0.5 0.5 0.1 0.1\n")
    with open(os.path.join(root, "split_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "holdout_mode": "transect", "pinned": {}}, fh)
    return root


def assert_pages_agree(pdf_path, html_path, expected_pages):
    """Every honesty check on a printed report: pdfinfo count, section count,
    'Page N:' titles, the footer total and the cover's table of contents."""
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    n_sections = html.count('<section class="page')
    check(n_sections == expected_pages, f"{n_sections} sections in report.html, plan says {expected_pages}")
    n_info = pdfinfo_pages(pdf_path)
    if n_info is None:
        print("  (pdfinfo not installed; page count not cross-checked)")
        return
    check(n_info == expected_pages, f"pdfinfo counts {n_info} pages, plan says {expected_pages}")
    if pdf_page_text(pdf_path, 1) is None:
        print("  (pdftotext not installed; footers not cross-checked)")
        return
    for page in range(1, n_info + 1):
        text = pdf_page_text(pdf_path, page)
        footer = FOOTER_RE.search(text)
        check(footer is not None, f"page {page} has no 'p. N / M' footer in its text: {text[-300:]!r}")
        check(int(footer.group(1)) == page and int(footer.group(2)) == n_info,
              f"page {page} footer reads {footer.group(0)!r}, expected p.{page} / {n_info}")
        if page > 1:
            check(re.search(rf"Page {page}:", text) is not None,
                  f"page {page} carries no 'Page {page}:' title; text starts {text[:200]!r}")
    cover = pdf_page_text(pdf_path, 1)
    check(re.search(rf"Glossary cheatsheet\s*p\.\s*{n_info}\b", cover) is not None,
          f"cover TOC does not put the glossary on page {n_info}: {cover!r}")


# ---------------------------------------------------------------------------
# The Chrome wrapper
# ---------------------------------------------------------------------------

def test_chrome_wrapper_writes_a_pdf_from_minimal_html():
    """Real Chrome prints a two-section document to a PDF with the magic bytes, two pages and a footer."""
    tmp = scratch("minimal")
    try:
        html = ER._assemble_report_html("Two pages", ER._build_css(), [
            '<section class="page"><h1 class="page-title">Page 1: One</h1><p>first</p></section>',
            '<section class="page last"><h1 class="page-title">Page 2: Two</h1><p>second</p></section>',
        ])
        html_path = ER._write_report_html(tmp, html)
        check(os.path.basename(html_path) == ER.REPORT_HTML_NAME, f"report written as {html_path}")
        pdf_path = os.path.join(tmp, "out.pdf")
        n_pages = ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        with open(pdf_path, "rb") as fh:
            head = fh.read(5)
        check(head == ER.PDF_MAGIC_BYTES, f"PDF does not start with {ER.PDF_MAGIC_BYTES!r}: {head!r}")
        n_info = pdfinfo_pages(pdf_path)
        if n_info is not None:
            check(n_info == 2, f"pdfinfo counts {n_info} pages for a two-section report")
            check(n_pages == 2, f"wrapper returned {n_pages} pages, pdfinfo says 2")
        text = pdf_page_text(pdf_path, 2)
        if text is not None:
            check("p.2 / 2" in re.sub(r"\s+", " ", text).replace("p. 2", "p.2"),
                  f"page 2 footer missing from {text!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_chrome_wrapper_raises_when_no_pdf_is_written():
    """A missing output folder, a missing input, a silent stub or an absent browser each raise with the path named."""
    tmp = scratch("nopdf")
    saved = os.environ.get(ER.CHROME_BINARY_ENV)
    try:
        html_path = ER._write_report_html(tmp, ER._assemble_report_html("x", ER._build_css(),
                                                                         ['<section class="page">x</section>']))
        # A missing output folder is refused at entry, before Chrome runs.
        missing_dir_pdf = os.path.join(tmp, "no_such_dir", "out.pdf")
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, missing_dir_pdf)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raised = str(exc)
        check("no_such_dir" in raised, f"missing output folder must be named in the error, got {raised!r}")

        # A missing input is refused at entry too (Chrome would print an error page as a valid PDF).
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(os.path.join(tmp, "absent.html"), os.path.join(tmp, "o.pdf"))
        except FileNotFoundError as exc:
            raised = str(exc)
        check("absent.html" in raised, f"missing input must be named in the error, got {raised!r}")

        # Chrome that exits 0 and writes nothing: the post-check must catch it.
        os.environ[ER.CHROME_BINARY_ENV] = write_stub_chrome(tmp, 'echo "stub wrote nothing" >&2; exit 0')
        pdf_path = os.path.join(tmp, "silent.pdf")
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        except RuntimeError as exc:
            raised = str(exc)
        check("silent.pdf" in raised and "stub wrote nothing" in raised,
              f"error must name the missing PDF and carry Chrome's stderr, got {raised!r}")
        check(not os.path.exists(pdf_path), "nothing may be left at the PDF path")

        # An unknown binary path is refused with the path and the variable named.
        os.environ[ER.CHROME_BINARY_ENV] = os.path.join(tmp, "not_a_browser")
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        except FileNotFoundError as exc:
            raised = str(exc)
        check("not_a_browser" in raised and ER.CHROME_BINARY_ENV in raised,
              f"missing browser must name the path and the override variable, got {raised!r}")
    finally:
        if saved is None:
            os.environ.pop(ER.CHROME_BINARY_ENV, None)
        else:
            os.environ[ER.CHROME_BINARY_ENV] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_chrome_wrapper_rejects_a_non_pdf_output():
    """An HTML file, an empty file or a non-zero exit is refused and the bad file removed."""
    tmp = scratch("notpdf")
    saved = os.environ.get(ER.CHROME_BINARY_ENV)
    try:
        html_path = ER._write_report_html(tmp, ER._assemble_report_html("x", ER._build_css(),
                                                                         ['<section class="page">x</section>']))
        os.environ[ER.CHROME_BINARY_ENV] = write_stub_chrome(tmp, 'printf "<html>error page</html>" > "$PDF"; exit 0')
        pdf_path = os.path.join(tmp, "fake.pdf")
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        except RuntimeError as exc:
            raised = str(exc)
        check("fake.pdf" in raised and "%PDF-" in raised,
              f"a non-PDF output must be refused naming the magic bytes, got {raised!r}")
        check(not os.path.exists(pdf_path), "the bad output must be removed so no reader mistakes it for a report")

        # An empty file is refused as well.
        os.environ[ER.CHROME_BINARY_ENV] = write_stub_chrome(tmp, ': > "$PDF"; exit 0')
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        except RuntimeError as exc:
            raised = str(exc)
        check("empty" in raised.lower(), f"an empty output must be called empty, got {raised!r}")

        # A non-zero exit is a failure even when a PDF-looking file exists.
        os.environ[ER.CHROME_BINARY_ENV] = write_stub_chrome(tmp, 'printf "%%PDF-1.4 junk" > "$PDF"; echo "crash" >&2; exit 21')
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, pdf_path)
        except RuntimeError as exc:
            raised = str(exc)
        check("21" in raised and "crash" in raised, f"a non-zero exit must report the code and stderr, got {raised!r}")
    finally:
        if saved is None:
            os.environ.pop(ER.CHROME_BINARY_ENV, None)
        else:
            os.environ[ER.CHROME_BINARY_ENV] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_chrome_wrapper_times_out_cleanly():
    """A browser that outlives the timeout is killed, reported with the timeout length, and its profile removed."""
    tmp = scratch("timeout")
    saved = os.environ.get(ER.CHROME_BINARY_ENV)
    real_mkdtemp = ER.tempfile.mkdtemp
    made = []

    def recording_mkdtemp(*a, **kw):
        """Wrap tempfile.mkdtemp to record every folder the wrapper creates."""
        path = real_mkdtemp(*a, **kw)
        made.append(path)
        return path

    try:
        html_path = ER._write_report_html(tmp, ER._assemble_report_html("x", ER._build_css(),
                                                                         ['<section class="page">x</section>']))
        os.environ[ER.CHROME_BINARY_ENV] = write_stub_chrome(tmp, f"sleep {STUB_SLEEP_SECONDS}; exit 0")
        ER.tempfile.mkdtemp = recording_mkdtemp
        t0 = time.time()
        raised = ""
        try:
            ER.print_html_to_pdf_with_chrome(html_path, os.path.join(tmp, "slow.pdf"),
                                             timeout_seconds=SHORT_TIMEOUT_SECONDS)
        except RuntimeError as exc:
            raised = str(exc)
        elapsed = time.time() - t0
        check("timed out" in raised.lower() and str(SHORT_TIMEOUT_SECONDS) in raised,
              f"timeout must be reported with its length, got {raised!r}")
        check(elapsed < STUB_SLEEP_SECONDS, f"wrapper waited {elapsed:.1f}s, past the {SHORT_TIMEOUT_SECONDS}s timeout")
        profiles = [p for p in made if os.path.basename(p).startswith(ER.CHROME_PROFILE_PREFIX)]
        check(len(profiles) == 1, f"expected one throwaway profile, saw {made}")
        check(not os.path.exists(profiles[0]), f"temp profile {profiles[0]} survived the timeout")
    finally:
        ER.tempfile.mkdtemp = real_mkdtemp
        if saved is None:
            os.environ.pop(ER.CHROME_BINARY_ENV, None)
        else:
            os.environ[ER.CHROME_BINARY_ENV] = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_file_uri_survives_spaces_and_hashes():
    """File URIs percent-encode spaces, hashes and non-ASCII characters and make relative paths absolute."""
    uri = ER._file_uri("/tmp/a b/c#d.jpg")
    check(uri == "file:///tmp/a%20b/c%23d.jpg", f"space and hash must be percent-encoded, got {uri}")
    check(ER._img_src("/tmp/a b/c#d.jpg") == uri, "_img_src must build the same URI")
    check(ER._img_src(None) == "" and ER._img_src("") == "", "_img_src of nothing is an empty string")
    rel = ER._file_uri("rel/path.png")
    check(rel.startswith("file:///") and rel.endswith("/rel/path.png"), f"relative paths are made absolute: {rel}")
    uni = ER._file_uri("/tmp/évaluation ∆/x.png")
    check("%C3%A9" in uni and "%E2%88%86" in uni, f"unicode must be percent-encoded as UTF-8: {uni}")


def test_css_is_inlined_in_the_head():
    """The assembled document carries the page CSS inside <head>, an escaped title and no external stylesheet."""
    html = ER._assemble_report_html('Run <x> & "y"', ER._build_css(), ['<section class="page">body</section>'])
    head, rest = html.split("</head>", 1)
    check(html.startswith("<!DOCTYPE html>"), "document must start with a doctype")
    check("<style>" in head and "</style>" in head, "the stylesheet must sit inside <head>")
    check("@page" in head and "size: letter" in head and "counter(pages)" in head,
          "the head must carry the real page CSS, not a link")
    check("<link" not in html, "no external stylesheet reference may remain")
    check('<meta charset="utf-8">' in head, "charset must be declared")
    check("Run &lt;x&gt; &amp; &quot;y&quot;" in head, f"title must be escaped, head is {head[:400]!r}")
    check('<section class="page">body</section>' in rest and rest.rstrip().endswith("</html>"),
          "sections must follow the head inside the body")
    check("print-color-adjust: exact" in head, "table header backgrounds need print-color-adjust to print")


# ---------------------------------------------------------------------------
# Page plan and page-number honesty
# ---------------------------------------------------------------------------

def test_page_plan_counts_every_section():
    """The page plan places every section from the class and preview counts and refuses bad counts."""
    plan = ER._plan_pages(n_per_class_rows=59, n_previews=8)
    m = -(-59 // ER.PER_CLASS_ROWS_PER_PAGE)
    check(plan["per_class"] == 5 and plan["per_class_table"] == 6, f"per-class pages misplaced: {plan}")
    check(plan["n_table_pages"] == m, f"{m} table pages expected for 59 rows, got {plan['n_table_pages']}")
    check(plan["confusion"] == 6 + m, f"confusion page must follow the table pages: {plan}")
    check(plan["samples"] == 7 + m and plan["n_sample_pages"] == 2, f"sample pages misplaced: {plan}")
    check(plan["glossary"] == 9 + m and plan["total"] == 9 + m, f"glossary must be last: {plan}")
    labels = [name for name, _pg in plan["toc"]]
    check(labels[0] == "Cover + top-line" and labels[-1] == "Glossary cheatsheet", f"toc order: {labels}")
    check(all(pg == plan[key] for (_n, pg), key in zip(plan["toc"], ("cover", "training", "box", "mask", "per_class",
                                                                     "per_class_table", "confusion", "samples", "glossary"))),
          f"toc page numbers must match the plan: {plan['toc']}")

    empty = ER._plan_pages(0, 0)
    check(empty["n_table_pages"] == 1 and empty["n_sample_pages"] == 0 and empty["total"] == 8,
          f"no classes and no previews still print one table page and no sample page: {empty}")
    check(not any(name.startswith("Prediction samples") for name, _pg in empty["toc"]),
          "no sample entry in the TOC when there are no previews")
    for bad in ((-1, 0), (0, -1), ("3", 0), (1.5, 0)):
        raised = False
        try:
            ER._plan_pages(*bad)
        except (TypeError, ValueError):
            raised = True
        check(raised, f"_plan_pages must refuse {bad}")


def test_per_class_pages_split_rows_and_repeat_the_header():
    """Sixty rows split into capped table pages with the header repeated and absent classes last."""
    tmp = scratch("split")
    try:
        names = class_names(60)
        metrics = fabricate_metrics(names, absent=names[-2:])
        plan = ER._plan_pages(60, 0)
        pages = ER._render_perclass(metrics, fake_args(tmp), {}, plan)
        check(len(pages) == 1 + plan["n_table_pages"], f"summary page plus {plan['n_table_pages']} table pages expected, got {len(pages)}")
        check("Page 5: Per-class breakdown" in pages[0], "summary page keeps its title")
        check(f"(page {plan['confusion']})" in pages[0], "callout must cite the computed confusion page")
        first_rows = pages[1].count("<tr>") - 1
        check(first_rows == ER.PER_CLASS_ROWS_PER_PAGE, f"first table page holds {first_rows} rows, cap is {ER.PER_CLASS_ROWS_PER_PAGE}")
        last_rows = pages[-1].count("<tr>") - 1
        check(last_rows == 60 - ER.PER_CLASS_ROWS_PER_PAGE * (plan["n_table_pages"] - 1),
              f"last table page holds {last_rows} rows")
        for i, page in enumerate(pages[1:], start=plan["per_class_table"]):
            check(page.count("<th") == 6, f"page {i} must repeat the six-column header")
            check(f"Page {i}: Per-class table" in page, f"table page numbered {i}")
        check(f"1-{ER.PER_CLASS_ROWS_PER_PAGE} of 60" in pages[1], "table page says which rows it holds")
        check(ER.ABSENT_CLASS_NOTE in pages[-1], "absent classes close the table on the last page")
        joined = "".join(pages)
        check(joined.count("<i>") == 60, f"every class, absent ones included, is named in italic: found {joined.count('<i>')} of 60")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_per_class_summary_names_the_family_and_bounds_the_chart():
    """The summary page names the run's family, describes the Instances column honestly and bounds its chart."""
    tmp = scratch("summary")
    try:
        names = class_names(55)
        det = fabricate_metrics(names)
        seg = fabricate_metrics(names, family=ER.METRIC_FAMILY_MASK)
        plan = ER._plan_pages(55, 0)
        det_page = ER._render_perclass(det, fake_args(tmp), {}, plan)[0]
        check("strict box score" in det_page and "mask score" not in det_page,
              "a detector's chart note must speak of the box score, never the mask score")
        seg_page = ER._render_perclass(seg, fake_args(tmp), {}, plan)[0]
        check("strict outline (mask) score" in seg_page, "a segmentation run's chart note names the outline score")
        check("{family}" not in det_page and "{family}" not in seg_page, "no placeholder may leak into a page")
        check("evaluation examples" in det_page and "training examples" not in det_page,
              "the callout must describe the Instances column as evaluation examples")
        raised = False
        try:
            ER._axis_block_html("per_class_bar")
        except ValueError:
            raised = True
        check(raised, "a note that names the family must refuse to render without one")
        check(ER._axis_block_html("loss_history") != "", "a note with no placeholder renders without a family")

        big = ER._plot_per_class_bar(det["per_class"], os.path.join(tmp, "big.png"), metric_key="box_map50_95")
        few = {k: v for k, v in list(det["per_class"].items())[:5]}
        small = ER._plot_per_class_bar(few, os.path.join(tmp, "small.png"), metric_key="box_map50_95")
        _wb, hb = Image.open(big).size
        _ws, hs = Image.open(small).size
        check(hb > 2 * hs, f"a 55-class chart ({hb}px) must be much taller than a 5-class chart ({hs}px)")
        check(hb <= (ER.PER_CLASS_BAR_MAX_HEIGHT_IN + 0.5) * 130, f"chart height {hb}px must stay within the page bound")
        h55, pt55 = ER._per_class_bar_geometry(55)
        h200, pt200 = ER._per_class_bar_geometry(200)
        check(h55 == ER.PER_CLASS_BAR_MAX_HEIGHT_IN and h200 == ER.PER_CLASS_BAR_MAX_HEIGHT_IN, "tall charts hit the bound")
        check(ER.PER_CLASS_BAR_LABEL_PT_MIN <= pt200 < pt55 <= ER.PER_CLASS_BAR_LABEL_PT_MAX, f"labels shrink with density: {pt55} {pt200}")
        page = ER._render_perclass(det, fake_args(tmp), {"per_class_bar": big}, plan)[0]
        check('class="fig-wrap perclass-bar"' in page, "the chart is wrapped in the bounded figure class")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_section_headings_match_page_count_at_sixty_classes():
    """A printed 60-class report has as many sheets as the plan, with agreeing titles, footers and contents."""
    tmp = scratch("sixty")
    try:
        out_dir = os.path.join(tmp, "eval")
        os.makedirs(out_dir)
        names = class_names(60)
        metrics = fabricate_metrics(names, absent=names[-3:])
        previews = fabricate_previews(out_dir, [f"frame_{i:02d}.jpg" for i in range(6)])
        plan = ER._plan_pages(60, len(previews))
        pdf_path = os.path.join(out_dir, "report.pdf")
        n_pages = ER.build_pdf(metrics, fake_ctx(tmp), previews, fake_args(out_dir), pdf_path)
        check(n_pages in (None, plan["total"]), f"build_pdf returned {n_pages}, plan total is {plan['total']}")
        html_path = os.path.join(out_dir, ER.REPORT_HTML_NAME)
        assert_pages_agree(pdf_path, html_path, plan["total"])
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
        check(AST_STAMP_RE.search(html) is not None, "the generated stamp must be an AST wall-clock time")
        check("weasy" not in html.lower(), "no engine name leaks into the report")
        check(html.count(AST_STAMP_RE.search(html).group(0)) == 2, "cover and provenance carry the same stamp")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_hostile_names_unicode_out_dir_and_odd_preview_names():
    """Hostile class names are escaped, and a unicode folder and odd preview names still print and embed."""
    tmp = scratch("hostile")
    try:
        out_dir = os.path.join(tmp, "évaluation ∆ #1")
        os.makedirs(out_dir)
        hostile = '<script>alert("x")</script>'
        quoted = 'Genus-species "quoted" & <b>bold</b>'
        names = class_names(5) + [hostile, quoted]
        metrics = fabricate_metrics(names)
        previews = fabricate_previews(out_dir, ["frame 01#a.jpg", "frame_02.jpg"])
        plan = ER._plan_pages(len(names), len(previews))
        pdf_path = os.path.join(out_dir, "report.pdf")
        ER.build_pdf(metrics, fake_ctx(tmp), previews, fake_args(out_dir), pdf_path)
        html_path = os.path.join(out_dir, ER.REPORT_HTML_NAME)
        with open(html_path, encoding="utf-8") as fh:
            html = fh.read()
        check("<script>" not in html, "a class name containing <script> must be escaped in report.html")
        check("&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html, "the escaped form must be present")
        check("<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html, "tags inside a class name are escaped")
        check("frame%2001%23a.jpg" in html, "a preview path with a space and a hash must be percent-encoded")
        assert_pages_agree(pdf_path, html_path, plan["total"])
        n_images = pdf_image_count(pdf_path)
        if n_images is not None:
            check(n_images >= len(previews), f"the PDF embeds {n_images} images; {len(previews)} previews expected at least")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_two_hundred_classes_render_within_the_timeout():
    """A 200-class report prints inside the timeout with page numbers that agree."""
    tmp = scratch("twohundred")
    try:
        out_dir = os.path.join(tmp, "eval")
        os.makedirs(out_dir)
        names = class_names(200)
        metrics = fabricate_metrics(names, absent=names[150:160])
        previews = fabricate_previews(out_dir, [f"f{i}.jpg" for i in range(9)])
        plan = ER._plan_pages(200, len(previews))
        pdf_path = os.path.join(out_dir, "report.pdf")
        t0 = time.time()
        ER.build_pdf(metrics, fake_ctx(tmp), previews, fake_args(out_dir), pdf_path)
        elapsed = time.time() - t0
        check(elapsed < ER.CHROME_PRINT_TIMEOUT_SECONDS, f"200 classes took {elapsed:.1f}s")
        assert_pages_agree(pdf_path, os.path.join(out_dir, ER.REPORT_HTML_NAME), plan["total"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# main(): exit codes and the numbers surviving a PDF failure
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def patched_pipeline(metrics, ctx, argv, chrome_binary=None):
    """Swap the GPU-bound stages for fabricated results and set argv and the Chrome override."""
    saved = {k: getattr(ER, k) for k in ("run_val", "extract_metrics", "gather_context", "render_previews")}
    saved_argv = sys.argv
    saved_env = os.environ.get(ER.CHROME_BINARY_ENV)
    ER.run_val = lambda args: {"results": None, "weights": "/w/best.pt", "data_yaml": "/d/data.yaml",
                               "wall_seconds": 1.0, "model": None, "val_dir": args.out_dir}
    ER.extract_metrics = lambda val_out: metrics
    ER.gather_context = lambda run_dir, dataset_dir: ctx
    ER.render_previews = lambda *a, **k: []
    sys.argv = argv
    if chrome_binary is not None:
        os.environ[ER.CHROME_BINARY_ENV] = chrome_binary
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(ER, k, v)
        sys.argv = saved_argv
        if saved_env is None:
            os.environ.pop(ER.CHROME_BINARY_ENV, None)
        else:
            os.environ[ER.CHROME_BINARY_ENV] = saved_env


def run_main_capturing():
    """Call main() and return (exit code, captured stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = ER.main()
    return code, buf.getvalue()


def test_main_exits_nonzero_when_pdf_fails_and_keeps_the_numbers():
    """main() returns EXIT_OUTPUT_MISSING on a failed PDF, ledger row or export copy and keeps and names the files written."""
    tmp = scratch("main")
    try:
        names = class_names(4)
        metrics = fabricate_metrics(names)
        ctx = fake_ctx(tmp)
        dataset_dir = make_dataset_dir(ctx["dataset_dir"], names)
        with open(os.path.join(ctx["run_dir"], "args.yaml"), "w", encoding="utf-8") as fh:
            fh.write("model: yolo11n.pt\n")
        stub = write_stub_chrome(tmp, "exit 0")

        # 1. PDF fails: non-zero exit, the numbers and the ledger row are kept and named.
        out1 = os.path.join(tmp, "out1")
        rounds1 = os.path.join(tmp, "rounds1")
        argv = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                "--out_dir", out1, "--rounds_dir", rounds1]
        with patched_pipeline(metrics, ctx, argv, chrome_binary=stub):
            code, out = run_main_capturing()
        check(code == ER.EXIT_OUTPUT_MISSING, f"exit code must be {ER.EXIT_OUTPUT_MISSING} when the PDF is missing, got {code}")
        for fn in ("report.md", "metrics.json"):
            check(os.path.isfile(os.path.join(out1, fn)), f"{fn} must be written before the PDF is attempted")
        check(os.path.isfile(os.path.join(rounds1, "rounds.csv")), "the ledger row must be written before the PDF")
        check(not os.path.exists(os.path.join(out1, "report.pdf")), "no PDF may exist after the failure")
        check("[eval] FAILED" in out and "report.pdf" in out, f"failure must be loud and name the PDF: {out[-600:]!r}")
        check("report.md" in out and "metrics.json" in out and "rounds.csv" in out,
              f"the exit message must list the files that were written: {out[-600:]!r}")
        with open(os.path.join(out1, "metrics.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        check(saved["metrics"]["overall"]["box_map50_95"] == 0.5, "metrics.json carries the numbers")
        check(saved["generated_at"].endswith("-04:00"), f"generated_at must carry the AST offset: {saved['generated_at']}")

        # 2. Ledger fails, PDF succeeds: still non-zero, and the message names the ledger.
        out2 = os.path.join(tmp, "out2")
        rounds2 = os.path.join(tmp, "rounds2_is_a_file")
        with open(rounds2, "w", encoding="utf-8") as fh:
            fh.write("not a folder")
        argv = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                "--out_dir", out2, "--rounds_dir", rounds2]
        with patched_pipeline(metrics, ctx, argv):
            code, out = run_main_capturing()
        check(code == ER.EXIT_OUTPUT_MISSING, f"a failed ledger write must exit {ER.EXIT_OUTPUT_MISSING}, got {code}")
        check(os.path.isfile(os.path.join(out2, "report.pdf")), "the PDF is still produced when only the ledger fails")
        check("[eval] FAILED" in out and "rounds" in out, f"ledger failure must be named: {out[-600:]!r}")

        # 3. Everything works: exit 0 and every output named.
        out3 = os.path.join(tmp, "out3")
        rounds3 = os.path.join(tmp, "rounds3")
        export = os.path.join(tmp, "export")
        argv = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                "--out_dir", out3, "--rounds_dir", rounds3, "--pdf_export_dir", export]
        with patched_pipeline(metrics, ctx, argv):
            code, out = run_main_capturing()
        check(code == ER.EXIT_OK, f"a clean run must exit {ER.EXIT_OK}, got {code}: {out[-800:]!r}")
        check("[eval] PDF written via Chrome:" in out, f"success line missing: {out[-600:]!r}")
        check(os.path.isfile(os.path.join(export, "run1_report.pdf")), "the export copy must exist")
        check("[eval] FAILED" not in out, "no failure line on a clean run")

        # 4. An export folder that cannot be created is a reported failure, not a silent one.
        out4 = os.path.join(tmp, "out4")
        bad_export = os.path.join(rounds2, "inside_a_file")
        argv = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                "--out_dir", out4, "--pdf_export_dir", bad_export]
        with patched_pipeline(metrics, ctx, argv):
            code, out = run_main_capturing()
        check(code == ER.EXIT_OUTPUT_MISSING and "inside_a_file" in out,
              f"a failed export copy must be reported with its path, got {code}: {out[-400:]!r}")

        # 5. Missing run or dataset folders are refused at entry.
        argv = ["evaluate_run.py", "--run_dir", os.path.join(tmp, "nope"), "--dataset_dir", dataset_dir,
                "--out_dir", os.path.join(tmp, "out5")]
        with patched_pipeline(metrics, ctx, argv):
            raised = ""
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ER.main()
            except FileNotFoundError as exc:
                raised = str(exc)
        check("nope" in raised, f"a missing run folder must be refused by name, got {raised!r}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Inherited reviewer items
# ---------------------------------------------------------------------------

def test_zero_confusion_matrix_is_refused_as_a_count_source():
    """An all-zero confusion matrix is refused as an instance source; a filled one still counts."""
    names = {0: "A", 1: "B"}
    zero = FakeConfusionMatrix(np.zeros((3, 3)))
    res = FakeResults(names, box=FakeMetric([0, 1]), confusion_matrix=zero)
    raised = ""
    try:
        ER.extract_metrics(val_out_for(res))
    except ValueError as exc:
        raised = str(exc)
    check("zero" in raised.lower() and "plots" in raised.lower(),
          f"an unfilled matrix must be refused with the reason, got {raised!r}")
    # A filled matrix still works as the fallback.
    filled = FakeConfusionMatrix([[2, 0, 0], [0, 1, 0], [1, 0, 0]])
    m = ER.extract_metrics(val_out_for(FakeResults(names, box=FakeMetric([0, 1]), confusion_matrix=filled)))
    check(m["per_class_instances"] == {"A": 3, "B": 1}, f"filled matrix counts: {m['per_class_instances']}")


def test_convergence_commentary_reports_a_bad_results_csv():
    """Unparseable loss columns give no commentary and a printed reason naming the file."""
    tmp = scratch("conv")
    try:
        bad = os.path.join(tmp, "results.csv")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write("epoch,train/box_loss,val/box_loss\n" + "\n".join(f"{i},{i * 0.1},notanumber" for i in range(12)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            text = ER._interpret_convergence(bad)
        check(text == "", f"unparseable losses must give no commentary, got {text!r}")
        check("results.csv" in buf.getvalue() and "notanumber" in buf.getvalue(),
              f"the skipped commentary must say what failed and where: {buf.getvalue()!r}")
        check(ER._interpret_convergence(os.path.join(tmp, "absent.csv")) == "", "a missing file gives no commentary")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bar_labels_use_the_binomial():
    """Bar-chart labels resolve each token to its full name and flag binomials for italics."""
    labels = ER._per_class_bar_labels(["Scolymia-cubensis", "sand", "SS", ""])
    check(labels == [("Scolymia cubensis", True), ("sand", False), ("Siderastrea siderea", True), ("?", False)],
          f"labels must be full names with an italic flag for binomials only, got {labels}")


def test_pdf_page_count_tolerates_a_malformed_pdfinfo():
    """A pdfinfo that prints a non-numeric, missing or failed page count yields None with a printed reason."""
    tmp = scratch("pdfinfo")
    saved = ER.PDFINFO_BINARY
    try:
        target = os.path.join(tmp, "any.pdf")
        with open(target, "wb") as fh:
            fh.write(ER.PDF_MAGIC_BYTES)

        def stub(body):
            """Write an executable script standing in for pdfinfo and point evaluate_run at it."""
            path = os.path.join(tmp, "pdfinfo.sh")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n" + body + "\n")
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
            ER.PDFINFO_BINARY = path

        for body, expected, note in (('echo "Pages:          7"', 7, "a numeric count is returned"),
                                     ('echo "Pages: many"', None, "a non-numeric count gives None"),
                                     ('echo "Title: x"', None, "a missing Pages line gives None"),
                                     ('echo "broken" >&2; exit 3', None, "a failing pdfinfo gives None")):
            stub(body)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                got = ER._pdf_page_count(target)
            check(got == expected, f"{note}: got {got!r}")
            if expected is None:
                check("pdfinfo" in buf.getvalue(), f"{note}: the reason must be printed, got {buf.getvalue()!r}")
        ER.PDFINFO_BINARY = os.path.join(tmp, "absent")
        check(ER._pdf_page_count(target) is None, "no pdfinfo at all gives None")
    finally:
        ER.PDFINFO_BINARY = saved
        shutil.rmtree(tmp, ignore_errors=True)


def test_training_history_plot_handles_a_detector_csv():
    """A detector's results.csv plots without a legend warning for the empty seg-loss panel."""
    tmp = scratch("history")
    try:
        csv_path = os.path.join(tmp, "results.csv")
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("epoch,train/box_loss,train/cls_loss,train/dfl_loss,val/box_loss,val/cls_loss,val/dfl_loss\n")
            for i in range(1, 8):
                fh.write(f"{i},{2.0 / i:.3f},{3.0 / i:.3f},{1.5 / i:.3f},{2.2 / i:.3f},{3.1 / i:.3f},{1.6 / i:.3f}\n")
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            png = ER._plot_training_history(csv_path, os.path.join(tmp, "hist.png"))
        check(png is not None and os.path.getsize(png) > 0, "a detector's results.csv still plots its three losses")
        legend_warnings = [w for w in caught if "legend" in str(w.message).lower()]
        check(not legend_warnings, f"an empty seg-loss panel must not raise a legend warning: {legend_warnings}")
        check(ER._plot_training_history(os.path.join(tmp, "absent.csv"), os.path.join(tmp, "x.png")) is None,
              "a missing results.csv gives no plot")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_no_weasyprint_remains_outside_the_docstring():
    """WeasyPrint is named exactly once, in the module docstring, and no import or dash remains."""
    with open(SRC_PATH, encoding="utf-8") as fh:
        src = fh.read()
    doc_end = src.index('"""', src.index('"""') + 3) + 3
    docstring, body = src[:doc_end], src[doc_end:]
    check("weasyprint" not in body.lower(), "WeasyPrint must not be mentioned outside the module docstring")
    check(docstring.lower().count("weasyprint") == 1, "the docstring records the engine change exactly once")
    check("2026-09-04" in docstring and "outline" in docstring.lower(),
          "the docstring must date the engine change and state that the outline sidebar is gone")
    check("import weasyprint" not in src and "from weasyprint" not in src, "no WeasyPrint import may remain")
    check("\u2014" not in src and "\u2013" not in src, "no em or en dash in the source")


# ---------------------------------------------------------------------------
# The subject noun and the image-type list, end to end through the PDF path
# ---------------------------------------------------------------------------

def test_a_printed_detector_report_says_nothing_about_corals():
    """Chrome prints a detector's report and no page of the PDF text, nor the
    running footer, tells the reader the subjects are corals."""
    tmp = scratch("fishpdf")
    try:
        ER.set_subject_noun(None)
        names = ["EPST_Dark", "PTVO_Normal", "GICI_Normal"]
        ER.set_names_full({
            "EPST_Dark": "Epinephelus striatus, Nassau grouper, dark phase",
            "PTVO_Normal": "Pterois volitans, red lionfish",
            "GICI_Normal": "Ginglymostoma cirratum, nurse shark",
        })
        metrics = fabricate_metrics(names, family=ER.METRIC_FAMILY_BOX)
        out_dir = os.path.join(tmp, "eval")
        os.makedirs(out_dir, exist_ok=True)
        args = fake_args(out_dir)
        previews = fabricate_previews(out_dir, ["a_sbs.jpg", "b_sbs.jpg"])
        pdf_path = os.path.join(out_dir, "report.pdf")
        ER.build_pdf(metrics, fake_ctx(tmp), previews, args, pdf_path)

        with open(os.path.join(out_dir, ER.REPORT_HTML_NAME), encoding="utf-8") as fh:
            html = fh.read()
        offenders = [ln for ln in html.split("</div>") if "coral" in ln.lower()][:2]
        check("coral" not in html.lower(),
              f"the printed detector report must not mention corals: {offenders}")
        check("{subject" not in html, "an unfilled subject placeholder reached report.html")
        check(" a object" not in html and " an objects" not in html,
              "an article disagrees with the neutral noun in report.html")
        check("fraction of real objects found" in html,
              "the running footer explains recall with the neutral noun")
        # Whole-document checks, because a phrase fixed in one builder and
        # left in another reaches the reader anyway: the training page said
        # the run tracks three losses and its figure caption said four.
        check("segmentation (mask) loss" not in html,
              "the printed detector report describes a segmentation loss it never had")
        check("four training losses" not in html,
              "the printed detector report counts four training losses on a three-loss run")
        check("three training losses" in html,
              "the printed detector report still names the losses it does record")
        check("mask PR curve unavailable" not in html and "mask F1 curve unavailable" not in html,
              "the printed detector report offers an outline curve it can never produce")

        if pdf_page_text(pdf_path, 1) is None:
            print("  (pdftotext not installed; the printed pages were not read back)")
            return
        n_pages = pdfinfo_pages(pdf_path) or 0
        for page in range(1, n_pages + 1):
            text = pdf_page_text(pdf_path, page) or ""
            check("coral" not in text.lower(), f"page {page} of the printed PDF mentions corals")
            check("segmentation (mask) loss" not in text,
                  f"page {page} of the printed PDF names a segmentation loss this run never had")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_names_full({})
        ER.set_subject_noun(None)


def test_previews_read_every_image_type_the_trainer_reads():
    """render_previews offers every image type the scorer opens, so a split
    held in WebP or TIFF is sampled in the report, the PDF sample pages and
    metrics.json instead of producing no samples at all."""
    tmp = scratch("previewtypes")
    try:
        dataset = os.path.join(tmp, "dataset")
        img_dir = os.path.join(dataset, "test", "images")
        lbl_dir = os.path.join(dataset, "test", "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        wanted = ("shot_a.webp", "shot_b.tif", "shot_c.bmp", "shot_d.jpg")
        for name in wanted:
            Image.new("RGB", (64, 48), (30, 60, 90)).save(os.path.join(img_dir, name))
            stem = os.path.splitext(name)[0]
            with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("0 0.5 0.5 0.2 0.2\n")
        # Not images: a label file loose in the images folder, and a folder
        # whose own name ends in an image suffix.
        with open(os.path.join(img_dir, "notes.txt"), "w", encoding="utf-8") as fh:
            fh.write("not an image\n")
        os.makedirs(os.path.join(img_dir, "spare.png"), exist_ok=True)

        out_dir = os.path.join(tmp, "eval")
        os.makedirs(out_dir, exist_ok=True)
        saved = ER.render_previews(NoPredictionModel(), dataset, "test", out_dir,
                                   count=10, conf=0.25, iou=0.6, imgsz=64)
        got = sorted(p["filename"] for p in saved)
        check(got == sorted(wanted),
              f"every readable image type must be offered as a preview, got {got}")
        for record in saved:
            check(os.path.isfile(record["overlay"]) and os.path.getsize(record["overlay"]) > 0,
                  f"preview overlay not written for {record['filename']}")
            check(record["n_gt"] == 1, f"{record['filename']}: one ground-truth box was drawn")

        # A folder holding only the less common types is no longer empty of samples.
        only_webp = os.path.join(tmp, "webp_only")
        os.makedirs(os.path.join(only_webp, "test", "images"), exist_ok=True)
        Image.new("RGB", (64, 48), (10, 10, 10)).save(
            os.path.join(only_webp, "test", "images", "one.webp"))
        saved2 = ER.render_previews(NoPredictionModel(), only_webp, "test",
                                    os.path.join(tmp, "eval2"), count=4,
                                    conf=0.25, iou=0.6, imgsz=64)
        check([p["filename"] for p in saved2] == ["one.webp"],
              f"a WebP-only split must still produce a sample, got {saved2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_main_resolves_the_subject_noun_from_the_evaluated_dataset():
    """main() reads the subject a dataset declares and writes it into the
    report, and falls back to the neutral noun when the dataset declares
    nothing, whatever the previous run in the same process left behind."""
    tmp = scratch("mainsubject")
    try:
        names = class_names(3)
        metrics = fabricate_metrics(names)
        ctx = fake_ctx(tmp)
        dataset_dir = make_dataset_dir(ctx["dataset_dir"], names)
        with open(os.path.join(ctx["run_dir"], "args.yaml"), "w", encoding="utf-8") as fh:
            fh.write("model: yolo11n.pt\n")
        data_yaml = os.path.join(dataset_dir, "data.yaml")
        with open(data_yaml, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        doc["subject"] = {"singular": "fish", "plural": "fish"}
        with open(data_yaml, "w", encoding="utf-8") as fh:
            yaml.dump(doc, fh)
        stub = write_stub_chrome(tmp, "exit 0")

        out1 = os.path.join(tmp, "declared")
        argv = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                "--out_dir", out1]
        with patched_pipeline(metrics, ctx, argv, chrome_binary=stub):
            _code, out = run_main_capturing()
        check("names one scored thing: fish" in out,
              f"main must say which noun it resolved: {out[-400:]!r}")
        check(" a object" not in out,
              f"the console line must read as a plain sentence: {out[-400:]!r}")
        with open(os.path.join(out1, "report.md"), encoding="utf-8") as fh:
            declared_md = fh.read()
        check("the fraction of the real fish the model finds" in declared_md,
              "the declared noun reaches the report main writes")
        check("object" not in declared_md, "the declared noun replaces the neutral one")
        with open(os.path.join(out1, "metrics.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        check(saved["subject_noun"] == ["fish", "fish"],
              f"metrics.json records the noun the report used: {saved.get('subject_noun')!r}")

        # A second evaluation in the same process must not inherit it.
        doc.pop("subject")
        with open(data_yaml, "w", encoding="utf-8") as fh:
            yaml.dump(doc, fh)
        out2 = os.path.join(tmp, "undeclared")
        argv2 = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                 "--out_dir", out2]
        with patched_pipeline(metrics, ctx, argv2, chrome_binary=stub):
            run_main_capturing()
        with open(os.path.join(out2, "report.md"), encoding="utf-8") as fh:
            neutral_md = fh.read()
        check("the fraction of the real objects the model finds" in neutral_md,
              "a dataset that declares nothing gets the neutral noun back")
        check("fish" not in neutral_md, "the previous run's noun must not carry over")
        with open(os.path.join(out2, "metrics.json"), encoding="utf-8") as fh:
            check(json.load(fh)["subject_noun"] == ["object", "objects"],
                  "metrics.json records the neutral noun too, so two reports can be compared")

        # The split dataset is rebuilt from scratch every time, so the noun
        # also survives in the staged source the manifest records.
        source = os.path.join(tmp, "staged_source")
        os.makedirs(source, exist_ok=True)
        with open(os.path.join(source, "data.yaml"), "w", encoding="utf-8") as fh:
            yaml.dump({"names": {i: nm for i, nm in enumerate(names)},
                       "subject": {"singular": "sponge", "plural": "sponges"}}, fh)
        with open(os.path.join(dataset_dir, ER.SPLIT_MANIFEST_NAME), "w", encoding="utf-8") as fh:
            json.dump({"schema_version": 1, "holdout_mode": "transect", "pinned": {},
                       "source": source}, fh)
        out3 = os.path.join(tmp, "from_source")
        argv3 = ["evaluate_run.py", "--run_dir", ctx["run_dir"], "--dataset_dir", dataset_dir,
                 "--out_dir", out3]
        with patched_pipeline(metrics, ctx, argv3, chrome_binary=stub):
            _code3, out_src = run_main_capturing()
        check("names one scored thing: sponge" in out_src,
              f"main says it read the noun from the staged source: {out_src[-400:]!r}")
        with open(os.path.join(out3, "report.md"), encoding="utf-8") as fh:
            source_md = fh.read()
        check("the fraction of the real sponges the model finds" in source_md,
              "the staged source's noun reaches a report built from a rebuilt dataset")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        ER.set_subject_noun(None)


if __name__ == "__main__":
    cases = [
        ("test_chrome_wrapper_writes_a_pdf_from_minimal_html", test_chrome_wrapper_writes_a_pdf_from_minimal_html),
        ("test_chrome_wrapper_raises_when_no_pdf_is_written", test_chrome_wrapper_raises_when_no_pdf_is_written),
        ("test_chrome_wrapper_rejects_a_non_pdf_output", test_chrome_wrapper_rejects_a_non_pdf_output),
        ("test_chrome_wrapper_times_out_cleanly", test_chrome_wrapper_times_out_cleanly),
        ("test_file_uri_survives_spaces_and_hashes", test_file_uri_survives_spaces_and_hashes),
        ("test_css_is_inlined_in_the_head", test_css_is_inlined_in_the_head),
        ("test_page_plan_counts_every_section", test_page_plan_counts_every_section),
        ("test_per_class_pages_split_rows_and_repeat_the_header", test_per_class_pages_split_rows_and_repeat_the_header),
        ("test_per_class_summary_names_the_family_and_bounds_the_chart", test_per_class_summary_names_the_family_and_bounds_the_chart),
        ("test_section_headings_match_page_count_at_sixty_classes", test_section_headings_match_page_count_at_sixty_classes),
        ("test_hostile_names_unicode_out_dir_and_odd_preview_names", test_hostile_names_unicode_out_dir_and_odd_preview_names),
        ("test_two_hundred_classes_render_within_the_timeout", test_two_hundred_classes_render_within_the_timeout),
        ("test_main_exits_nonzero_when_pdf_fails_and_keeps_the_numbers", test_main_exits_nonzero_when_pdf_fails_and_keeps_the_numbers),
        ("test_zero_confusion_matrix_is_refused_as_a_count_source", test_zero_confusion_matrix_is_refused_as_a_count_source),
        ("test_convergence_commentary_reports_a_bad_results_csv", test_convergence_commentary_reports_a_bad_results_csv),
        ("test_bar_labels_use_the_binomial", test_bar_labels_use_the_binomial),
        ("test_pdf_page_count_tolerates_a_malformed_pdfinfo", test_pdf_page_count_tolerates_a_malformed_pdfinfo),
        ("test_training_history_plot_handles_a_detector_csv", test_training_history_plot_handles_a_detector_csv),
        ("test_no_weasyprint_remains_outside_the_docstring", test_no_weasyprint_remains_outside_the_docstring),
        ("test_a_printed_detector_report_says_nothing_about_corals", test_a_printed_detector_report_says_nothing_about_corals),
        ("test_previews_read_every_image_type_the_trainer_reads", test_previews_read_every_image_type_the_trainer_reads),
        ("test_main_resolves_the_subject_noun_from_the_evaluated_dataset", test_main_resolves_the_subject_noun_from_the_evaluated_dataset),
    ]
    for name, fn in cases:
        run(name, fn)
    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"{len(_results) - n_fail} passed, {n_fail} failed")
    print("PASS" if n_fail == 0 else f"{n_fail} FAILED")
    sys.exit(1 if n_fail else 0)
