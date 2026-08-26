"""
Configuration for TCRMPclip_addExpertIDs — the expert-ID importer.

Lightweight: no SAM3/torch. Reads the expert's returned CSV (UID -> code),
folds the IDs into the project's segmentations + the permanent library, applies
the >50%-overlap auto-relabel, removes resolved items from the review repo, and
pushes. Re-runnable as many times as expert CSVs arrive.
"""

import os

_SRC = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_SRC)                 # scripts/TCRMPclip_addExpertIDs
_SCRIPTS = os.path.dirname(_PROJECT)             # scripts/
_REPO = os.path.dirname(_SCRIPTS)                # github_repo

# Current project's step-5 export dir (segmentations live under it). The
# orchestrator sets TCRMP_EXPORT_DIR to the active project's step5 folder.
EXPORT_DIR = os.environ.get('TCRMP_EXPORT_DIR', '') or os.path.join(_PROJECT, 'output')

# Review (GitHub-Pages) repo working tree, and the permanent library.
REVIEW_DIR = os.environ.get('TCRMP_REVIEW_DIR', '') or '/mnt/tear/REVIEW_reefpointseg'
REVIEW_REPO_URL = (os.environ.get('TCRMP_REVIEW_REPO_URL', '')
                   or 'https://github.com/laurenkolinger/reefpointseg-review.git')
EXPERT_LIBRARY_DIR = os.environ.get('TCRMP_EXPERT_LIBRARY_DIR', '')   # '' -> default
REVIEW_GIT_PUSH = (os.environ.get('TCRMP_REVIEW_GIT_PUSH', '1') == '1')
REVIEW_OVERLAP_THRESH = float(os.environ.get('TCRMP_REVIEW_OVERLAP_THRESH', '') or 0.5)
MASTER_CODES_CSV = (os.environ.get('TCRMP_MASTER_CODES', '')
                    or os.path.join(_REPO, 'supporting_data', 'master_codes.csv'))

PORT = int(os.environ.get('TCRMP_ADDEXPERT_PORT', '') or 5075)
