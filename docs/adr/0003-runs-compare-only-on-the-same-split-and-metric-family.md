# Runs compare only on the same split and the same metric family

Status: accepted
Source: Lauren Olinger's decision of 2026-09-04 ("unify the seed, flag old runs"), recorded in `/mnt/rip/vicarius_drive/docs/superpowers/plans/2026-09-04-vega-training-foldin.md`, section "Decisions taken with Lauren"; measurement in the same plan under "Evidence gathered before planning".
Last updated: 2026-09-04 (LO)

Measured on 2026-09-04 over the 10361 Coral Intel images: the two code paths that build a detection split seeded the shuffle differently (one on the staged dataset name, one on a user-typed lab name), so 837 of the project run's 1036 held-out test images sat inside the lab run's training set, and the Models table ranked the two mAP50 values side by side. A second mismatch sat beside it: the ledger's gate verdicts were computed from outline (mask) metrics only, so a box detector was compared against nothing and every detection round read "unpinned". We decided that a score is comparable to another only when both runs were scored on the same held-out images and both report the same metric family, and that the platform records enough to check both: every split manifest carries the seed, the fractions and a split fingerprint (SHA-256 over the sorted basename-to-split assignment, first 16 hex characters, scheme `sha256-16/netstring-utf8-sorted-v1`, written identically by the detection and the segmentation splitter); every evaluation records its metric family (`mask` or `box`) in `metrics.json` and in the rounds ledger; the champion file records its family and the score under the matching key; and a run whose fingerprint or family differs from the champion's is shown as "not comparable" with its score de-emphasised, never hidden and never ranked. Both splitters now seed on the dataset name so the same dataset splits the same way wherever it is trained.

## Considered options

- Unify the seed for future runs and leave existing runs as they are. Rejected: the two Coral Intel scores would keep sitting side by side looking comparable.
- Unify the seed and retrain the affected runs. Rejected for now: costs GPU time before the tooling exists to tell which runs need it; nothing stops Lauren retraining once the card shows the flag.
- Record the split and the family on every run and let the readers decide. Chosen.

## Consequences

- A run evaluated before 2026-09-04 has no fingerprint and no recorded family (its ledger row lacks `metrics_schema: 2`); it reads as "unknown, not comparable". Its per-class rows were also attributed by raw class id, which was wrong whenever a class was absent from the split, so those rows are not corrected retroactively: re-evaluating the run rewrites them.
- A gate verdict between two families is recorded as `incomparable`, a fourth value beside pass, fail and flag.
- The comparability check is one function, `runs_comparable` in `scripts/_modellab/labruns.py`, so the rule lives in one place; the fields it reads are named at the top of that function.
