import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))            # scripts/
sys.path.insert(0, os.path.join(HERE, "..", ".."))      # scripts/.. for _reefreview pkg
from _reefreview.mask_registry import MaskRegistry, REGISTRY_FIELDS, build_registry_record

_fail = 0
def check(c, m):
    global _fail
    if not c:
        _fail += 1; print("FAIL:", m)

d = tempfile.mkdtemp()
reg = MaskRegistry(d)
r1 = reg.upsert({"uid": "SCP-20201022-T101-x10y20-abcd", "source_image": "TCRMP..._SCP_T101.jpeg",
                 "basename": "TCRMP..._SCP_T101", "species": "PA", "source_type": "manual_click",
                 "status": "accepted", "project_id": "p1"})
check(r1["species"] == "PA" and r1["created_at"] and r1["updated_at"], "insert stamps + fields")
check(set(REGISTRY_FIELDS) >= {"uid","species","status","updated_at","created_at"}, "fields defined")
# relabel: same uid, species changes, created_at preserved, updated_at is >= created_at
r2 = reg.upsert({"uid": "SCP-20201022-T101-x10y20-abcd", "species": "OFAV"})
check(r2["species"] == "OFAV", "relabel updates species")
check(r2["created_at"] == r1["created_at"], "created_at preserved on update")
check(len(reg.load()) == 1, "still one row (upsert by uid, no dup)")
check(reg.lookup("SCP-20201022-T101-x10y20-abcd")["species"] == "OFAV", "lookup reflects update")
# a second uid coexists
reg.upsert({"uid": "BID-20150101-T201-x5y5-ef01", "species": "MC", "status": "accepted"})
check(len(reg.rows()) == 2, "two masks tracked")

# build_registry_record: turn an accepted mask into a canonical record
mask = {"source_x": 10, "source_y": 20, "rle": {"counts": [1, 2], "size": [100, 200]},
        "species": "PA", "source_type": "manual_click", "status": "accepted"}
rec = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask, "p1", "Project One")
check(rec is not None, "build_registry_record returns a dict")
check(rec["uid"].startswith("SCP-20201022-T101-"), "uid derived from filename+mask")
check(rec["species"] == "PA", "species carried through")
check(rec["review"] == '', "review defaults to '' (falsy)")
check(rec["expert_mode"] == '', "expert_mode defaults to '' (falsy)")
check(rec["site"] == "SCP", "site parsed from filename")
check(rec["transect"] == 1, "transect parsed from filename")
check(rec["source_image"] == "TCRMP20201022_clip_SCP_T101.jpeg", "source_image is basename")
check(rec["basename"] == "TCRMP20201022_clip_SCP_T101", "basename strips extension")
check(rec["project_id"] == "p1" and rec["project_name"] == "Project One", "project metadata carried")
r3 = reg.upsert(rec)
check(reg.lookup(rec["uid"])["species"] == "PA", "upserting a built record works")
check(len(reg.rows()) == 3, "three masks tracked after build+upsert")

# expert_mode True when expert_id dict says EXPERT
mask_expert = dict(mask)
mask_expert["expert_id"] = {"mode": "EXPERT", "id": "abc"}
rec_expert = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask_expert, "p1", "Project One")
check(rec_expert["expert_mode"] == '1', "expert_mode '1' when expert_id.mode == EXPERT")

# unparseable filename with no source_x/source_y still yields a safe fallback uid
mask_noloc = {"species": "PA", "status": "accepted"}
rec_fallback = build_registry_record("not_a_tcrmp_filename.jpeg", mask_noloc, "p1", "Project One")
check(rec_fallback is not None, "fallback uid still safe -> record built")

# boolean CSV round-trip: 'review' must survive load->merge->write->reload as
# a correctly truthy/falsy string, not Python's "True"/"False" (both truthy
# once read back as a str, since bool("False") == True).
mask_review_true = {"source_x": 30, "source_y": 40, "species": "PA",
                     "source_type": "manual_click", "status": "accepted", "review": True}
rec_true = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask_review_true, "p1", "Project One")
reg.upsert(rec_true)

mask_review_false = {"source_x": 50, "source_y": 60, "species": "PA",
                      "source_type": "manual_click", "status": "accepted", "review": False}
rec_false = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask_review_false, "p1", "Project One")
reg.upsert(rec_false)

# a second, unrelated upsert forces a load->merge->write->reload cycle
reg.upsert({"uid": "BID-20150101-T201-x5y5-ef01", "species": "MC_updated"})

row_true = reg.lookup(rec_true["uid"])
row_false = reg.lookup(rec_false["uid"])
check(row_true.get('review') == '1', "review=True round-trips as truthy '1' string")
check(not row_false.get('review'), "review=False round-trips as falsy '' string")

# ── reviewer column (D2 / spec 4.3) ──────────────────────────────────
import csv, shutil
from _reefreview.mask_registry import (
    normalize_reviewer, is_test_residue_project_id, TEST_FIXTURE_PATTERNS)

check(REGISTRY_FIELDS[-1] == 'reviewer', "REGISTRY_FIELDS ends with 'reviewer'")
check(normalize_reviewer('  lo ') == 'LO', "normalize_reviewer strips + uppercases")
check(normalize_reviewer('abcdefghijk') == 'ABCDEFGH', "normalize_reviewer caps at 8 chars")
check(normalize_reviewer(None) == '', "normalize_reviewer(None) is blank")

# build_registry_record carries the reviewer: the mask's own value wins over
# the caller's fallback; blank stays blank.
rec_rv = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask, "p1", "Project One",
                               reviewer='ab')
check(rec_rv["reviewer"] == 'AB', f"build_registry_record reviewer fallback normalized (got {rec_rv['reviewer']!r})")
mask_own = dict(mask); mask_own["reviewer"] = "cd"
rec_own = build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask_own, "p1", "Project One",
                                reviewer='ab')
check(rec_own["reviewer"] == 'CD', "mask['reviewer'] wins over the fallback")
check(build_registry_record("TCRMP20201022_clip_SCP_T101.jpeg", mask, "p1", "Project One")["reviewer"] == '',
      "no reviewer anywhere -> '' (nothing inferred)")

# Old manifest (no reviewer column) reads back with reviewer '' and the next
# write emits the new header.
d_old = tempfile.mkdtemp()
old_fields = [f for f in REGISTRY_FIELDS if f != 'reviewer']
with open(os.path.join(d_old, 'manifest.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=old_fields); w.writeheader()
    w.writerow({'uid': 'OLD-x1y1', 'species': 'PA', 'status': 'accepted', 'project_id': 'legacy',
                'created_at': '2026-01-01 00:00:00', 'updated_at': '2026-01-01 00:00:00'})
reg_old = MaskRegistry(d_old)
loaded_old = reg_old.load()
check(loaded_old['OLD-x1y1'].get('reviewer') == '', "old manifest: missing reviewer column reads as ''")
reg_old.upsert({'uid': 'NEW-x2y2', 'species': 'MC', 'status': 'accepted', 'reviewer': 'lo'})
with open(os.path.join(d_old, 'manifest.csv'), newline='') as f:
    header = next(csv.reader(f))
check(header == REGISTRY_FIELDS, f"next write emits the new header (got {header})")
reloaded = reg_old.load()
check(reloaded['OLD-x1y1'].get('reviewer') == '' and reloaded['NEW-x2y2'].get('reviewer') == 'lo',
      "round trip keeps '' for the legacy row and the value for the new row")
# a blank reviewer on an update never erases recorded initials
reg_old.upsert({'uid': 'NEW-x2y2', 'species': 'MCAV', 'reviewer': ''})
check(reg_old.lookup('NEW-x2y2')['reviewer'] == 'lo', "blank reviewer on update keeps recorded initials")
check(reg_old.lookup('NEW-x2y2')['species'] == 'MCAV', "other fields still update alongside")
reg_old.upsert({'uid': 'NEW-x2y2', 'reviewer': 'AB'})
check(reg_old.lookup('NEW-x2y2')['reviewer'] == 'AB', "non-blank reviewer on update replaces initials")

# ── purge_test_rows (D7) ──────────────────────────────────────────────
check(len(TEST_FIXTURE_PATTERNS) == 6, "six fixture patterns declared")
for pid, exp in [('tmp', True), ('exp_neg_9mb', True), ('e2e_x', True), ('combined_smoke_1', True),
                 ('label_guard_2', True), ('expert_lock_test_3', True),
                 ('20260101_demo', False), ('tmp2', False), ('my_exp_', False), ('', False)]:
    check(is_test_residue_project_id(pid) is exp, f"is_test_residue_project_id({pid!r}) is {exp}")

d_purge = tempfile.mkdtemp()
reg_p = MaskRegistry(d_purge)
removed0, backup0 = reg_p.purge_test_rows([])
check(removed0 == 0 and backup0 == '', "purge on a missing manifest -> (0, '')")
for i, pid in enumerate(['tmp', 'exp_a', 'e2e_b', 'combined_smoke_c', 'label_guard_d',
                         'expert_lock_test_e', 'exp_live', '20260101_real']):
    reg_p.upsert({'uid': f'U{i}-x{i}y{i}', 'species': 'PA', 'status': 'accepted', 'project_id': pid})
check(len(reg_p.rows()) == 8, "8 rows before purge")
removed, backup = reg_p.purge_test_rows(['exp_live', '20260101_real'])
check(removed == 6, f"purge removed the 6 residue rows (got {removed})")
left = {r['project_id'] for r in reg_p.rows()}
check(left == {'exp_live', '20260101_real'}, f"live run_* project + real project survive (got {left})")
check(os.path.isfile(backup) and os.path.basename(backup).startswith('manifest.csv.bak-'),
      f"backup written next to the manifest (got {backup})")
import re as _re
check(bool(_re.search(r'\.bak-\d{8}-\d{6}$', backup)), "backup suffix is YYYYMMDD-HHMMSS")
with open(backup, newline='') as f:
    n_backup = sum(1 for _ in csv.DictReader(f))
check(n_backup == 8, f"backup holds the pre-purge rows (got {n_backup})")

# a retry inside the same minute with nothing left to remove writes no backup
# and leaves the pre-purge copy untouched (a double click must not clobber it)
import glob as _glob
removed_again, backup_again = reg_p.purge_test_rows(['exp_live', '20260101_real'])
check((removed_again, backup_again) == (0, ''),
      f"retry with nothing to remove -> (0, '') (got {(removed_again, backup_again)})")
with open(backup, newline='') as f:
    n_backup = sum(1 for _ in csv.DictReader(f))
check(n_backup == 8, f"first backup still holds the 8 pre-purge rows after the retry (got {n_backup})")
check(sorted(_glob.glob(reg_p.manifest_path + '.bak-*')) == [backup],
      "no second backup file after the no-op retry")

# two removing purges inside one second: the clock is frozen so both want the
# same name; the second takes -2 and each backup keeps its own pre-purge rows
import _reefreview.mask_registry as _mr
_real_datetime = _mr.datetime
# A stamp no earlier purge in this run can own, so the first frozen purge gets
# the bare name and the second the -2 suffix regardless of wall-clock timing.
_frozen = _real_datetime(2001, 2, 3, 4, 5, 6, tzinfo=_mr.AST)


class _FrozenDatetime:
    @staticmethod
    def now(tz=None):
        return _frozen


_mr.datetime = _FrozenDatetime
try:
    reg_p.upsert({'uid': 'U8-x8y8', 'species': 'PA', 'status': 'accepted', 'project_id': 'tmp'})
    r1, b1 = reg_p.purge_test_rows(['exp_live', '20260101_real'])
    reg_p.upsert({'uid': 'U9-x9y9', 'species': 'PA', 'status': 'accepted', 'project_id': 'e2e_z'})
    reg_p.upsert({'uid': 'U10-x0y0', 'species': 'PA', 'status': 'accepted', 'project_id': 'e2e_zz'})
    r2, b2 = reg_p.purge_test_rows(['exp_live', '20260101_real'])
finally:
    _mr.datetime = _real_datetime
check(r1 == 1 and r2 == 2, f"frozen-clock purges removed 1 then 2 rows (got {r1}, {r2})")
check(b1.endswith('.bak-' + _frozen.strftime('%Y%m%d-%H%M%S')), f"first backup carries the seconds stamp (got {b1})")
check(b2 == b1 + '-2', f"second backup in the same second takes the -2 suffix (got {b2})")


def _rows_in(p):
    with open(p, newline='') as f:
        return [r['project_id'] for r in csv.DictReader(f)]


check(sorted(_rows_in(b1)) == ['20260101_real', 'exp_live', 'tmp'],
      f"first backup keeps its pre-purge rows (got {_rows_in(b1)})")
check(sorted(_rows_in(b2)) == ['20260101_real', 'e2e_z', 'e2e_zz', 'exp_live'],
      f"second backup keeps its own pre-purge rows (got {_rows_in(b2)})")
check(len(_glob.glob(reg_p.manifest_path + '.bak-*')) == 3, "three backups total, none overwritten")
check({r['project_id'] for r in reg_p.rows()} == {'exp_live', '20260101_real'}, "live rows survive both purges")

# cleanup: every temp dir this test made
for _p in (d, d_old, d_purge):
    shutil.rmtree(_p, ignore_errors=True)
check(not any(os.path.exists(_p) for _p in (d, d_old, d_purge)), "temp dirs removed")

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
