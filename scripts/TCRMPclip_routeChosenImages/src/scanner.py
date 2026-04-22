"""
Minimal scanner module — provides pts_to_raw_name for export.py compatibility.
Full scanning is handled by selected_frames.csv from TCRMPcvr_chooseImages.
"""

import re


def pts_to_raw_name(pts_filename):
    """Convert a _pts filename or _cpc.json key to a raw image filename.

    Handles:
      TCRMP20201022_clip_SCP_T101_pts.jpg -> TCRMP20201022_clip_SCP_T101.jpeg
      TCRMP20140926_clip_BWR_T101_cpc.json -> TCRMP20140926_clip_BWR_T101.jpeg
    """
    # Remove _pts or _cpc.json before extension
    name = re.sub(r'_pts\.(jpe?g|png)', r'.jpeg', pts_filename, flags=re.I)
    name = re.sub(r'_cpc\.json$', '.jpeg', name, flags=re.I)
    return name
