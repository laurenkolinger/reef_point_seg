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

# accepted expert_id mask -> found_expert, no longer counted manual.
em = mask('PA','accepted','manual_click')
em['expert_id'] = {'code': 'PA', 'mode': 'EXPERT', 'labeler': 'XY'}
o = P.compute_label_outcomes({'masks': [em]}, ['PA'], 'step5')
check(o['PA']['outcome'] == 'found_expert', "expert_id mode EXPERT -> found_expert")
check(o['PA']['outcome'] != 'found_manual', "accepted expert mask not counted manual")

# expert beats a sibling manual mask of the same species.
o = P.compute_label_outcomes({'masks': [mask('PA','accepted','manual_click'), em]}, ['PA'], 'step5')
check(o['PA']['outcome'] == 'found_expert', "expert wins over manual for same code")

# review mask -> pending_expert rows labeled by distinct reviews[] codes,
# excluding the synthetic 'overlap' reviewer and blank codes.
rm = mask('REVIEW','accepted','manual_click', review=True)
rm['reviews'] = [
    {'reviewer': 'AB', 'code': 'DSTO', 'confidence': 'high', 'at': P.ast_now()},
    {'reviewer': 'overlap', 'code': 'ZZZ', 'confidence': '', 'at': P.ast_now()},
    {'reviewer': 'CD', 'code': '', 'confidence': '', 'at': P.ast_now()},
]
o = P.compute_label_outcomes({'masks': [rm]}, [], 'step5')
check(o.get('DSTO', {}).get('outcome') == 'pending_expert', "review mask -> pending_expert per tentative code")
check('ZZZ' not in o, "'overlap' reviewer code excluded from pending labels")
check('' not in o, "blank tentative code excluded from pending labels")

# review mask with no tentative codes -> literal REVIEW label.
o = P.compute_label_outcomes({'masks': [mask('REVIEW','accepted','manual_click', review=True)]}, [], 'step5')
check(o.get('REVIEW', {}).get('outcome') == 'pending_expert', "bare review mask -> REVIEW pending_expert")

# pending beats not_found for a target; loses to any found_* for the same code.
pm = mask('REVIEW','accepted','manual_click', review=True)
pm['reviews'] = [{'reviewer': 'AB', 'code': 'PA', 'confidence': '', 'at': P.ast_now()}]
check(P.compute_label_outcomes({'masks': [pm]}, ['PA'], 'step5')['PA']['outcome'] == 'pending_expert',
      "pending beats not_found for target code")
o = P.compute_label_outcomes({'masks': [mask('PA','accepted','auto'), pm]}, ['PA'], 'step5')
check(o['PA']['outcome'] == 'found_ai', "pending loses to found_ai for same code")

# rejected review mask emits nothing pending.
xm = mask('REVIEW','rejected','manual_click', review=True)
xm['reviews'] = [{'reviewer': 'EF', 'code': 'XYZ', 'confidence': '', 'at': P.ast_now()}]
o = P.compute_label_outcomes({'masks': [xm]}, [], 'step5')
check('XYZ' not in o and 'REVIEW' not in o, "rejected review mask emits no pending row")

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

# CSV header unchanged with the new outcome values.
P.write_provenance_csv(d, 'IMG_C', {
    'PA': {'outcome':'found_expert','source':'step5','reviewer':'LO','at':P.ast_now()},
    'REVIEW': {'outcome':'pending_expert','source':'step5','reviewer':'','at':P.ast_now()},
}, 'proj1')
with open(os.path.join(d, 'label_provenance.csv')) as f:
    reader = csv.DictReader(f)
    check(reader.fieldnames == P.CSV_HEADER, "CSV header unchanged for new outcomes")
    cmap = {(r['basename'], r['label']): r for r in reader}
check(cmap[('IMG_C','PA')]['outcome'] == 'found_expert', "found_expert row written")
check(cmap[('IMG_C','REVIEW')]['outcome'] == 'pending_expert', "pending_expert row written")

# model source_type counts as AI-found
seg = {"masks": [{"species": "PA", "status": "accepted", "source_type": "model"}]}
out = P.compute_label_outcomes(seg, ["PA"], "step4loop", "")
check(out["PA"]["outcome"] == "found_ai", "model source_type -> found_ai")
check(out["PA"]["source"] == "step4loop", "pane source passes through")

# an UNRECOGNIZED source_type must NOT be found_ai (the refactor makes _AI_SOURCES load-bearing;
# before the fix the found_ai else-branch would wrongly classify this as AI-found)
seg_unk = {"masks": [{"species": "PA", "status": "accepted", "source_type": "someunknown"}]}
out_unk = P.compute_label_outcomes(seg_unk, ["PA"], "step4loop", "")
check(out_unk["PA"]["outcome"] != "found_ai", "unrecognized source_type is NOT found_ai")

print("PASS" if _fail == 0 else f"{_fail} FAILED")
sys.exit(1 if _fail else 0)
