"""Unit tests for provenance.py. Run: env/bin/python .../tests/test_provenance.py"""
import os, sys, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import provenance as P

_fail = 0
def check(c, m):
    global _fail
    if not c: _fail += 1; print(f"  FAIL {m}")
    else: print(f"  PASS {m}")

def mask(species, status, source_type, review=False):
    return {'species': species, 'status': status, 'source_type': source_type, 'review': review}

# found_manual wins when any accepted mask of the species is manual.
seg = {'masks': [mask('PA','accepted','auto'), mask('PA','accepted','manual_click')]}
o = P.compute_label_outcomes(seg, ['PA'], 'step4test', 'LO')
check(o['PA']['outcome'] == 'found_manual', "manual wins over auto")
check(o['PA']['source'] == 'step4test' and o['PA']['reviewer'] == 'LO', "source+reviewer carried")
check(o['PA']['at'].endswith('-04:00'), "AST offset on timestamp")

# found_ai when only auto/exemplar accepted.
seg = {'masks': [mask('PA','accepted','auto'), mask('PA','accepted','exemplar')]}
check(P.compute_label_outcomes(seg, ['PA'], 'step5')['PA']['outcome'] == 'found_ai', "auto/exemplar -> found_ai")

# not_found when target reviewed with no accepted target mask; rejected/review ignored.
seg = {'masks': [mask('PA','rejected','auto'), mask('PA','accepted','manual_click', review=True)]}
check(P.compute_label_outcomes(seg, ['PA'], 'step5')['PA']['outcome'] == 'not_found', "rejected+review ignored -> not_found")

# non-target with no accepted mask gets no key.
check('XX' not in P.compute_label_outcomes({'masks': []}, ['PA'], 'step5'), "non-target absent gets no key")

# CSV upsert: second write for same (basename,label) replaces, other rows kept.
d = tempfile.mkdtemp()
P.write_provenance_csv(d, 'IMG_A', {'PA': {'outcome':'not_found','source':'step5','reviewer':'','at':P.ast_now()}}, 'proj1')
P.write_provenance_csv(d, 'IMG_B', {'PA': {'outcome':'found_ai','source':'step5','reviewer':'','at':P.ast_now()}}, 'proj1')
P.write_provenance_csv(d, 'IMG_A', {'PA': {'outcome':'found_manual','source':'step4test','reviewer':'LO','at':P.ast_now()}}, 'proj1')
with open(os.path.join(d, 'label_provenance.csv')) as f:
    rows = list(csv.DictReader(f))
check(len(rows) == 2, "upsert keeps one row per (basename,label)")
amap = {r['basename']: r for r in rows}
check(amap['IMG_A']['outcome'] == 'found_manual' and amap['IMG_A']['source'] == 'step4test', "IMG_A row replaced")
check(amap['IMG_B']['outcome'] == 'found_ai', "IMG_B row preserved")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
