# TCRMPcvr_makeAllPoints

Parse all TCRMP CVR (Coral Visual Record) spreadsheets into a single unified CSV of benthic point-count observations.

## What this does

Reads 839 Excel files spanning 2001-2025 from the TCRMP monitoring program and extracts every point-count observation into a flat CSV. Each row = one point on one transect frame, identified to species.

## Input

**Directory:** `../input/TCRMP_CVR/` (or pass a custom path as arg 1)

```
TCRMP_CVR/
  TCRMP2001_CVR/
    TCRMP20010425_CVR_CBS_PR.xls
    ...
  TCRMP2002_CVR/
    ...
  ...
  TCRMP2025_CVR/
    Annual/
      TCRMP20251014_CVR_CRB_ACS.xlsx
    PBL/
      ...
```

**File naming:** `TCRMP[YYYYMMDD]_CVR_[SITE]_[OBSERVERS].xlsx`
- YYYYMMDD = survey date
- SITE = 2-5 letter site code (FLC, BWR, CBS, SRW, etc.)
- OBSERVERS = analyst initials

**Three Excel format generations:**

| Format | Years | Extension | Sheet structure |
|--------|-------|-----------|-----------------|
| Legacy | 2001-2016 | .xls/.xlsx | `01TRAN`-`06TRAN` + appendix sheets |
| Intermediate | 2017-2019 | .xlsx | `CODES`, `RAW_DATA`, `TO_MASTER`, `SUMMARY` |
| Modern | 2020-2025 | .xlsx | `ReadMe`, `Codes`, `DataValidation`, `RawData`, `ToMaster`, `Summary` |

**Data structure (modern/intermediate):**
- Row 1: Location name
- Row 3: Date of filming (Excel serial or datetime)
- Row 5: Number of data points per transect
- Row 8: Column headers (Points, Check, Notes x6 transects)
- Row 9+: Point data. Column A = frame number (sparse), Column B = point label (A-T), Columns C/F/I/L/O/R = species code per transect

**Data structure (legacy):**
- Per-transect sheets (01TRAN-06TRAN)
- Column 0 = raw species code, Column 1 = point label (A-J cycling), Column 4 = QA/QC validated code
- Column 5 = species lookup table (`"Agaricia agaricites (AA) - coral"`)

## Output

All files written to `output/` (or pass custom path as arg 2):

### `all_points_YYYYMMDD.csv`

One row per point observation. ~2.3M rows.

| Column | Type | Description |
|--------|------|-------------|
| `date` | date | Survey date (YYYY-MM-DD) |
| `year` | float | Year extracted from date |
| `site` | str | 2-5 letter site code (FLC, BWR, CBS, etc.) |
| `transect` | int | Transect number (1-6) |
| `frame` | int | Frame number within transect (1-19+) |
| `point_label` | str | Point letter (A-T, 20 per frame) |
| `species_code` | str | Species/substrate code (OFRA, DICT, SS, etc.) |
| `species_name` | str | Full species name from codes lookup |
| `category` | str | Category (Coral, Macroalgae, Gorgonian, Sponge, Non-living, etc.) |

### `master_codes.csv`

Species code lookup table aggregated from all files (~130 codes).

| Column | Type | Description |
|--------|------|-------------|
| `code` | str | Species abbreviation |
| `category` | str | Benthic category |
| `name` | str | Full scientific name |

### `parse_log.txt`

Complete log of the parse run including file counts, errors, and summary stats.

## Usage

```bash
# Default: reads from ../input/TCRMP_CVR/, writes to ./output/
python run.py

# Custom input directory
python run.py /path/to/TCRMP_CVR

# Custom input + output
python run.py /path/to/TCRMP_CVR /path/to/output
```

## Dependencies

```
pip install openpyxl xlrd pandas
```

## Notes

- 5 of 843 oldest .xls files (2001-2011) are corrupted and skipped
- Some legacy files have date conversion artifacts (years 1900/1905) from bad Excel serial numbers
- The `site` column comes from the filename; if the filename is non-standard, it falls back to parsing the location name from inside the spreadsheet
- Species codes may have changed over time due to taxonomy revisions (e.g., Montastraea -> Orbicella in 2012). Both old and new codes are preserved as-is from the source files.
- Each frame of 20 points (A-T) corresponds to one still image extracted from the video transect. Frame 1 of transect 1 = image T101, frame 2 = T102, etc.
