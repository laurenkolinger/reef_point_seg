#!/usr/bin/env python3
"""names.py: label codes and site codes resolve to full names (spec section 6)."""
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

import csv, os, sys, tempfile, shutil, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
FAILS = []
def check(c, m):
    print(("[PASS] " if c else "[FAIL] ") + m)
    if not c: FAILS.append(m)
from _matrix import names
tmp = tempfile.mkdtemp(prefix="names_test_")
try:
    lab = os.path.join(tmp, "master_codes.csv")
    open(lab, "w").write("code,category,name\nPA,Coral,Porites astreoides\nSS,Coral,Siderastrea siderea\n")
    site = os.path.join(tmp, "site_codes.csv")
    open(site, "w").write("site,site_code\nBrewers Bay,BWR\nGreat Pond,GRP\n")
    ln = names.read_label_names(lab)
    check(ln == {"PA": "Porites astreoides", "SS": "Siderastrea siderea"}, "label names read from master_codes.csv")
    sn = names.read_site_names(site)
    check(sn == {"BWR": "Brewers Bay", "GRP": "Great Pond"}, "site names read from site_codes.csv")
    check(names.read_label_names(os.path.join(tmp, "missing.csv")) == {}, "missing label file gives {}")
    check(names.read_site_names("") == {}, "blank site path gives {}")
    open(lab, "a").write("OFRA,Coral,Orbicella franksii\n")
    os.utime(lab, None)
    check("OFRA" in names.read_label_names(lab), "cache refreshes when the file changes")
    open(os.path.join(tmp, "bad.csv"), "w").write("not,a,header\nx,y,z\n")
    check(names.read_label_names(os.path.join(tmp, "bad.csv")) == {}, "a file without code/name columns gives {}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ---- Fix round 1: adversarial and edge-case tests ----

# (a) a UTF-8 BOM before the header
tmp_a = tempfile.mkdtemp(prefix="names_test_bom_")
try:
    p = os.path.join(tmp_a, "master_codes.csv")
    with open(p, "wb") as f:
        f.write(b"\xef\xbb\xbfcode,category,name\nPA,Coral,Porites astreoides\n")
    r = names.read_label_names(p)
    check(isinstance(r, dict), "(a) BOM'd header: reader returns a dict, never raises")
    check(r.get("PA") == "Porites astreoides", f"(a) BOM'd header: the good row still resolves (got {r})")
finally:
    shutil.rmtree(tmp_a, ignore_errors=True)

# (b) quoted fields with embedded commas and quotes
tmp_b = tempfile.mkdtemp(prefix="names_test_quote_")
try:
    p = os.path.join(tmp_b, "master_codes.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "category", "name"])
        w.writerow(["AL", "Coral", 'Agaricia "lamarcki", form A'])
    r = names.read_label_names(p)
    check(isinstance(r, dict), "(b) embedded comma/quote field: reader returns a dict, never raises")
    check(r.get("AL") == 'Agaricia "lamarcki", form A',
          f"(b) embedded comma/quote field resolves intact (got {r.get('AL')!r})")
finally:
    shutil.rmtree(tmp_b, ignore_errors=True)

# (c) a name with HTML and control characters (<b>x</b>, a tab, a NUL byte)
# is returned verbatim, not stripped. The oddities sit in the middle of the
# value (never at either edge) so .strip()'s own whitespace trimming cannot
# coincidentally produce a false pass.
tmp_c = tempfile.mkdtemp(prefix="names_test_ctrl_")
try:
    p = os.path.join(tmp_c, "master_codes.csv")
    weird = "<b>x</b>\tmid\x00dle<end>"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "category", "name"])
        w.writerow(["HC", "Coral", weird])
    r = names.read_label_names(p)
    check(isinstance(r, dict), "(c) HTML/control-char field: reader returns a dict, never raises")
    check(r.get("HC") == weird, f"(c) HTML + tab + NUL byte returned verbatim, not stripped (got {r.get('HC')!r})")
finally:
    shutil.rmtree(tmp_c, ignore_errors=True)

# (d) a duplicate code: the last row wins
tmp_d = tempfile.mkdtemp(prefix="names_test_dup_")
try:
    p = os.path.join(tmp_d, "master_codes.csv")
    open(p, "w").write("code,category,name\nPA,Coral,First Name\nPA,Coral,Second Name\n")
    r = names.read_label_names(p)
    check(isinstance(r, dict), "(d) duplicate code: reader returns a dict, never raises")
    check(r.get("PA") == "Second Name", f"(d) duplicate code: the last row wins, not the first (got {r.get('PA')!r})")
finally:
    shutil.rmtree(tmp_d, ignore_errors=True)

# (e) a malformed line with too few fields between good rows
tmp_e = tempfile.mkdtemp(prefix="names_test_short_")
try:
    p = os.path.join(tmp_e, "master_codes.csv")
    open(p, "w").write(
        "code,category,name\nPA,Coral,Porites astreoides\nSS,Coral\nOFRA,Coral,Orbicella franksii\n")
    r = names.read_label_names(p)
    check(isinstance(r, dict), "(e) short row between good rows: reader returns a dict, never raises")
    check(r.get("PA") == "Porites astreoides" and r.get("OFRA") == "Orbicella franksii",
          f"(e) good rows on either side of a too-short line still resolve (got {r})")
finally:
    shutil.rmtree(tmp_e, ignore_errors=True)

# (f) an empty file (0 bytes) and a header-only file both give {}
tmp_f = tempfile.mkdtemp(prefix="names_test_empty_")
try:
    p_empty = os.path.join(tmp_f, "empty.csv")
    open(p_empty, "wb").close()
    check(os.path.getsize(p_empty) == 0, "(f) fixture sanity: the empty file really is 0 bytes")
    r = names.read_label_names(p_empty)
    check(isinstance(r, dict) and r == {}, f"(f) a 0-byte file gives {{}} (got {r})")
    p_header = os.path.join(tmp_f, "header_only.csv")
    open(p_header, "w").write("code,category,name\n")
    r2 = names.read_label_names(p_header)
    check(isinstance(r2, dict) and r2 == {}, f"(f) a header-only file gives {{}} (got {r2})")
finally:
    shutil.rmtree(tmp_f, ignore_errors=True)

# (g) a large file (50,000 rows) reads in under 2 seconds and resolves the
# first and last code
tmp_g = tempfile.mkdtemp(prefix="names_test_large_")
try:
    p = os.path.join(tmp_g, "master_codes.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "category", "name"])
        for i in range(50000):
            w.writerow(["C%06d" % i, "Coral", "Species name %d" % i])
    t0 = time.time()
    r = names.read_label_names(p)
    elapsed = time.time() - t0
    check(isinstance(r, dict), "(g) 50,000-row file: reader returns a dict, never raises")
    check(elapsed < 2.0, f"(g) 50,000-row file reads in under 2s (took {elapsed:.3f}s)")
    check(r.get("C000000") == "Species name 0" and r.get("C049999") == "Species name 49999",
          f"(g) 50,000-row file resolves the first and last code "
          f"(got first={r.get('C000000')!r}, last={r.get('C049999')!r})")
finally:
    shutil.rmtree(tmp_g, ignore_errors=True)

# (h) permission denied gives {} (skipped as a no-op check if running as
# root, where chmod 000 does not actually block a read)
tmp_h = tempfile.mkdtemp(prefix="names_test_perm_")
p_h = os.path.join(tmp_h, "master_codes.csv")
open(p_h, "w").write("code,category,name\nPA,Coral,Porites astreoides\n")
os.chmod(p_h, 0o000)
try:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        print("[SKIP] (h) permission-denied check: running as root, chmod 000 is not enforced")
    else:
        r = names.read_label_names(p_h)
        check(isinstance(r, dict) and r == {}, f"(h) permission-denied file gives {{}} (got {r})")
finally:
    os.chmod(p_h, 0o644)
    shutil.rmtree(tmp_h, ignore_errors=True)

# (i) concurrent readers: 8 threads calling read_label_names on the same
# path 200 times each all get the same dict and no exception
tmp_i = tempfile.mkdtemp(prefix="names_test_concurrent_")
try:
    p = os.path.join(tmp_i, "master_codes.csv")
    open(p, "w").write("code,category,name\nPA,Coral,Porites astreoides\nSS,Coral,Siderastrea siderea\n")
    expected = {"PA": "Porites astreoides", "SS": "Siderastrea siderea"}
    errors = []
    mismatches = []
    results_lock = threading.Lock()

    def _worker():
        try:
            for _ in range(200):
                d = names.read_label_names(p)
                if d != expected:
                    with results_lock:
                        mismatches.append(d)
        except Exception as e:
            with results_lock:
                errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(errors == [], f"(i) concurrent readers: no exception across 8 threads x 200 calls (got {errors})")
    check(mismatches == [], f"(i) concurrent readers: every call returns the same dict (got {len(mismatches)} mismatches)")
finally:
    shutil.rmtree(tmp_i, ignore_errors=True)

# (j) integration: the real master_codes.csv and site_codes.csv this module
# ships resolve real codes and carry the expected minimum breadth
GR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
real_master = os.path.join(GR, "supporting_data", "master_codes.csv")
real_sites = os.path.join(GR, "supporting_data", "site_codes.csv")
real_labels = names.read_label_names(real_master)
real_sitenames = names.read_site_names(real_sites)
check(real_labels.get("PA") == "Porites astreoides",
      f"(j) real supporting_data/master_codes.csv resolves PA (got {real_labels.get('PA')!r})")
check(real_labels.get("SS") == "Siderastrea siderea",
      f"(j) real supporting_data/master_codes.csv resolves SS (got {real_labels.get('SS')!r})")
check(real_sitenames.get("BWR") == "Brewers Bay",
      f"(j) real supporting_data/site_codes.csv resolves BWR (got {real_sitenames.get('BWR')!r})")
check(len(real_labels) > 100, f"(j) real master_codes.csv carries more than 100 label names (got {len(real_labels)})")
check(len(real_sitenames) > 30, f"(j) real site_codes.csv carries more than 30 site names (got {len(real_sitenames)})")

print("%d failed" % len(FAILS))
sys.exit(1 if FAILS else 0)
