"""The one list of mask source types that count as hand work.

Module: _reefreview.manual_sources
Purpose: every place that decides "was this mask made by a person?" (the
    annotators' label ledger, the label-coverage matrix, the composite builder
    and its source filter, the Step 5 provenance) reads this list, so the same
    accepted mask never reads manual in one file and model-made in another.
Inputs: none.
Outputs: MANUAL_SOURCE_TYPES.

`merged` is manual work. A merged mask only ever comes from an operator
merging reviewed masks by hand (POST .../merge_many, which needs a named
reviewer), even though no single box or click prompt produced its shape.

This module imports nothing, so any package can import it without a cycle.
"""

MANUAL_SOURCE_TYPES = frozenset({"manual_box", "manual_click", "merged"})
