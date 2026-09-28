# The runs table names training data and drops comparability from the screen

Status: accepted
Source: Lauren, 2026-09-17, on the VEGA student-ready build: "i dont really care about comparing
model runs. maybe just make a table that includes the training data subset (so we can filter by
those) if thats what you mean by not comparable? in some applications simple is better. dont
overcomplicate shit like this once the mechanics work." Plan
`docs/superpowers/plans/2026-09-16-vega-student-ready.md` decision D3, as amended by that ruling.
Last updated: 2026-09-17 (LO)

The Trained runs card on the Train step lists every run VEGA can find across the model labs and the
projects. Until 2026-09-16 it led with comparability: a state chip per row ("not comparable: split
not recorded", "not comparable: different metric family"), a summary line that counted the
incomparable rows first, a champion line that labelled one metric with another's name, run names
split by a fixed character count so they broke mid-word, and no way to tell which rows belonged to
the open project or what data any run had learned from. Lauren pasted it and wrote "wtf is this".

We decided that comparability is a mechanic, not a headline. The card keeps every mechanism that
protects a result (the family grouping, the honest disabling of an action that cannot work, the
promote confirmation that states a gate verdict, the champion's true label) and stops presenting
comparability to the person at all. In its place the card names what every run was trained on, as a
column and as a filter, because that is the question a person actually has when they look at a list
of runs.

## Considered options

- Keep the comparability chips and explain them better. Rejected by Lauren: she does not want to
  compare runs on this card, and every sentence spent explaining "split fingerprint" to a student is
  a sentence they did not need.
- Compute comparability from the data each run trained on and show that instead. Rejected: it is the
  same apparatus under a different label.
- Name the training data and let the person filter by it; leave the comparability fields in the
  payload for the code that still gates promotion. Chosen.

## Consequences

- `/api/step/6/all_runs` carries `trained_on` (a staged box dataset, a training set, or the project's
  own export) and `trained_on_kind`, read from the records that recorded them (`lab.json`,
  `project.json`, the split manifest, `provenance.json`) and never guessed from a folder name; a run
  with no record reads "not recorded". Decision record 0004 on staged datasets carrying their split
  pins is unchanged.
- The card groups runs by metric family (detectors scored in box mAP50, segmenters in outline mAP50),
  marks the open project's rows, stripes and hovers, never breaks a name mid-token, and filters by
  name, state, source and training data, remembered per browser under `vega.runs.`.
- The comparability fields stay in the payload. `runs_comparable` and the ledger's gates still refuse
  to rank two measurements that differ, and the promote route still answers 409 with every gate
  verdict until the caller confirms. None of that is rendered as a chip.
- The champion line reads the metric it shows: "box mAP50-95 0.826" where it once said "box mAP50".
- The version 2 legibility floor applies: cells 13 px, chips 12 px, nothing smaller.
