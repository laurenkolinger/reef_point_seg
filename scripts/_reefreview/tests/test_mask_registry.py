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

print("PASS" if _fail == 0 else f"{_fail} FAILED"); sys.exit(1 if _fail else 0)
