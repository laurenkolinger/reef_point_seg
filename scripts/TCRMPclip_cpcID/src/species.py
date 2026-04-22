"""
Species lookup - joins CPC point labels with species codes from all_points.csv.

Handles fuzzy date matching for cases where the CPC filename date
doesn't exactly match the all_points.csv date (e.g. SSJ 2018-12-07 vs 2018-12-04).
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta


class SpeciesLookup:
    """Lookup species codes by (date, site, transect, frame, label).

    Supports exact and fuzzy date matching with configurable tolerance.
    """

    def __init__(self, csv_path, date_tolerance_days=7):
        self.exact = {}
        self.by_site_year = defaultdict(list)
        self.date_tolerance = date_tolerance_days
        self._load(csv_path)

    def _load(self, csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    transect = int(float(row["transect"])) if row["transect"] else 0
                    frame = int(float(row["frame"])) if row["frame"] else 0
                except (ValueError, KeyError):
                    continue

                key = (
                    row["date"],
                    row["site"],
                    transect,
                    frame,
                    row["point_label"],
                )
                self.exact[key] = {
                    "species_code": row.get("species_code", ""),
                    "species_name": row.get("species_name", ""),
                    "category": row.get("category", ""),
                }

                year = row["date"][:4] if row["date"] else ""
                site_year = (row["site"], year)
                if row["date"] not in self.by_site_year[site_year]:
                    self.by_site_year[site_year].append(row["date"])

    def lookup(self, date, site, transect, frame, label):
        """Look up species info. Tries exact date first, then fuzzy match."""
        key = (date, site, transect, frame, label)
        result = self.exact.get(key)
        if result:
            return result, date, True

        # Fuzzy: find closest date for this site+year within tolerance
        year = date[:4]
        available = self.by_site_year.get((site, year), [])
        if not available:
            # Try site code corrections (MSR -> MRS)
            for (s, y), dates in self.by_site_year.items():
                if y == year and _codes_similar(site, s):
                    available = dates
                    site = s
                    break

        if available:
            best_date = _closest_date(date, available, self.date_tolerance)
            if best_date:
                key2 = (best_date, site, transect, frame, label)
                result = self.exact.get(key2)
                if result:
                    return result, best_date, False

        return None, None, False

    def get_available_dates(self, site, year):
        return self.by_site_year.get((site, str(year)), [])

    def __len__(self):
        return len(self.exact)


def _closest_date(target, candidates, tolerance_days):
    """Find the closest date string to target within tolerance."""
    try:
        t = datetime.strptime(target, "%Y-%m-%d")
    except ValueError:
        return None

    best = None
    best_delta = timedelta(days=tolerance_days + 1)
    for c in candidates:
        try:
            d = datetime.strptime(c, "%Y-%m-%d")
            delta = abs(d - t)
            if delta < best_delta:
                best_delta = delta
                best = c
        except ValueError:
            continue

    return best if best_delta <= timedelta(days=tolerance_days) else None


def _codes_similar(a, b):
    """Check if two site codes are likely the same (transposed letters etc.)."""
    if a == b:
        return True
    return sorted(a.upper()) == sorted(b.upper()) and len(a) == len(b)
