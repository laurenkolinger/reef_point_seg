"""
Compatibility shim. The expert-ID import logic moved to the shared blueprint
package `_expertids.importer` (so the orchestrator and this standalone tool share
one implementation). This re-exports it so existing imports (`import importer`)
and tests keep working unchanged.
"""

import os
import sys

# scripts/ on path for the shared package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from _expertids.importer import *          # noqa: F401,F403
from _expertids.importer import (          # noqa: F401  (explicit re-exports)
    parse_csv_text, import_rows,
    _iter_segmentation_files, _relabel_in_segmentations, _overlap_relabel_pass,
)
