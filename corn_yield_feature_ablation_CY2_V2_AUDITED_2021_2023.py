# ============================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORN YIELD 2 — AUDITED FEATURE ABLATION — V2

Standalone feature-ablation workflow built directly from the audited canonical
preprocessing/model backbone.

AUDITED GUARANTEES
------------------
  • SAME-YEAR FINAL-CDL MODIS historical vegetation.
  • Calendar-complete hist_5yr / hist_3yr (no row-shift across missing years).
  • Internal-only vegetation interpolation; missing seasonal tails remain NaN.
  • Strict prior-year vegetation anomalies; current year excluded.
  • 15 compact raw PRISM features with strict key/coverage checks.
  • Training-only median imputation.
  • Expanding-year held-out validation.
  • Fixed canonical XGBoost hyperparameters.
  • Same held-out county rows for every feature variant within a year/DOY.
  • AWC/SOC/CEC duplicate-FIPS assertions (no silent drop_duplicates).
  • Canonical parity assertion: the production feature set in this ablation must
    reproduce the audited canonical matrix exactly before fitting.

FEATURE BUILD-UP
----------------
  1. Raw vegetation only
  2. Vegetation anomalies only
  3. Raw + vegetation anomalies
  4. + hist_5yr
  5. + year
  6. + 15 compact raw PRISM  <-- canonical production feature policy
  7. + 7 strict PRISM anomalies
  8. + AWC100
  9. CURRENT FULL MODEL = + 3 AWC×stress interactions
 10. Current + all raw SOC source columns
 11. Current + all raw CEC source columns
 12. Current + SOC + CEC

DROP-ONE
--------
Reproduces the prior ablation logic around CURRENT FULL MODEL:
  • minus vegetation anomalies
  • minus year
  • minus weather anomalies
  • minus weather (raw + anomalies + weather-dependent AWC interactions)
  • minus AWC (AWC100 + interactions)
  • minus hist_5yr predictor

The target remains yield anomaly and absolute yield is reconstructed as:
    predicted_yield = hist_5yr + predicted_anomaly
for every variant, matching the canonical model architecture.

VALIDATION WINDOW
-----------------
Defaults to 2021–2023. When 2024/2025 SAME-YEAR FINAL-CDL MODIS are ready,
change only VALIDATION_END_YEAR near the top from 2023 to 2025.

Local usage:
    python corn_yield_feature_ablation_CY2_V2_AUDITED_2021_2023.py ^
        --input-dir "D:\\corn_yield\\model_inputs" ^
        --work-dir "D:\\corn_yield\\yield_model_work_ablation" ^
        --output-dir "D:\\corn_yield\\feature_ablation_V2_results"

Dependencies:
    numpy pandas matplotlib scikit-learn xgboost
"""

# ============================================================
# CELL 1 — IMPORTS + COMMAND-LINE SETTINGS
# ============================================================

import argparse
import os
import glob
import zipfile
import shutil
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from xgboost import XGBRegressor

try:
    from IPython.display import display
except Exception:
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 240)



MODEL_VERSION = "CY2_FEATURE_ABLATION_V2_AUDITED_2021_2023"

# ============================================================
# VALIDATION WINDOW — CHANGE ONLY END YEAR LATER
# ============================================================
# Current interim run: 2021–2023.
# When 2024/2025 SAME-YEAR FINAL-CDL MODIS exports are ready,
# change VALIDATION_END_YEAR from 2023 to 2025 and rerun.
VALIDATION_START_YEAR = 2021
VALIDATION_END_YEAR = 2023
VALIDATION_TAG = f"{VALIDATION_START_YEAR}_{VALIDATION_END_YEAR}"

# The fixed model configuration selected by the controlled ablation.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": -1,
}



def parse_args():
    parser = argparse.ArgumentParser(
        description="Corn Yield 2 — audited V2 feature-ablation workflow"
    )

    parser.add_argument(
        "--input-dir",
        default=r"C:\Users\logan\OneDrive\BSE 508\corn_yield_model_input",
        help="Directory containing required CSVs and/or ZIP archives.",
    )

    parser.add_argument(
        "--work-dir",
        default=r"C:\Users\logan\OneDrive\BSE 508\corn_yield_model_work",
        help="Scratch directory used for extracted/staged inputs.",
    )

    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\logan\OneDrive\BSE 508\corn_yield_model_results",
        help="Directory for audited feature-ablation outputs.",
    )

    return parser.parse_args()


ARGS = parse_args()

INPUT_DIR = os.path.abspath(ARGS.input_dir)
WORK_DIR = os.path.abspath(ARGS.work_dir)
OUTPUT_DIR_BASE = os.path.abspath(ARGS.output_dir)

print("✓ imports ready")
print("Input directory :", INPUT_DIR)
print("Work directory  :", WORK_DIR)
print("Output directory:", OUTPUT_DIR_BASE)

print("Model version    :", MODEL_VERSION)
print("Workflow         : audited feature ablation with canonical parity checks")
print(f"Validation window: {VALIDATION_START_YEAR}–{VALIDATION_END_YEAR}")
print("To extend later, change only VALIDATION_END_YEAR near the top of the script.")
# ============================================================
# CELL 2 — DISCOVER + STAGE INPUT FILES
# ============================================================

if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(
        f"Input directory does not exist:\n{INPUT_DIR}"
    )

DATA_DIR = os.path.join(WORK_DIR, "data")

# Snapshot source files BEFORE recreating the work directory. This also
# prevents accidental recursive discovery when work/output live below
# the input directory.
source_csvs = []
source_zips = []

work_resolved = Path(WORK_DIR).resolve()
output_resolved = Path(OUTPUT_DIR_BASE).resolve()
input_root = Path(INPUT_DIR)


def _outside_generated_dirs(path):
    """Return True only for source files outside work/output trees."""
    resolved = path.resolve()

    for generated_root in (work_resolved, output_resolved):
        try:
            resolved.relative_to(generated_root)
            return False
        except ValueError:
            pass

    return True


# Search only file types this workflow can ingest. This avoids walking large
# raster/cache trees just to reject every non-CSV/non-ZIP file afterward.
for pattern in ("*.csv", "*.CSV"):
    for path in input_root.rglob(pattern):
        if path.is_file() and _outside_generated_dirs(path):
            source_csvs.append(path)

for pattern in ("*.zip", "*.ZIP"):
    for path in input_root.rglob(pattern):
        if path.is_file() and _outside_generated_dirs(path):
            source_zips.append(path)

source_csvs = sorted(set(source_csvs))
source_zips = sorted(set(source_zips))


print("\nSource CSV files:", len(source_csvs))
print("Source ZIP files:", len(source_zips))

if len(source_csvs) == 0 and len(source_zips) == 0:
    raise FileNotFoundError(
        "No CSV or ZIP inputs were found under:\n"
        + INPUT_DIR
    )


if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)


# ------------------------------------------------------------
# Copy direct CSV inputs while preserving their relative paths.
# ------------------------------------------------------------
for src in source_csvs:

    rel = src.relative_to(Path(INPUT_DIR))
    dst = Path(DATA_DIR) / "direct_csv" / rel

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )


# ------------------------------------------------------------
# Extract every ZIP into its own subdirectory so identically
# named files from different archives cannot overwrite each other.
# ------------------------------------------------------------
for n, src in enumerate(
    sorted(source_zips),
    start=1,
):

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        src.stem,
    )

    extract_dir = (
        Path(DATA_DIR)
        / "zip_extract"
        / f"{n:03d}_{safe_name}"
    )

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Extracting:",
        src.name,
    )

    with zipfile.ZipFile(
        src,
        "r",
    ) as z:

        z.extractall(
            extract_dir
        )


all_csvs = glob.glob(
    DATA_DIR + "/**/*.csv",
    recursive=True,
)


print(
    "\nTotal staged CSV files:",
    len(all_csvs),
)


for f in sorted(all_csvs)[:50]:

    print(
        " ",
        os.path.basename(f),
    )
# CELL 3 — CLASSIFY INPUT FILES
# ============================================================

same_year_csvs = []
final_cdl_historical_candidates = []
prism_csvs = []
yield_candidates = []


for f in all_csvs:

    name = os.path.basename(f)
    lower = name.lower()

    # ========================================================
    # SAME-YEAR HISTORICAL MODIS
    # ========================================================

    if "corn_modis_same_year" in lower:

        same_year_csvs.append(
            f
        )

        continue


    # ========================================================
    # HISTORICAL FINAL-CDL FALLBACK
    #
    # Older scenario exports such as:
    #   Corn_MODIS_2023_FINAL_CDL.csv
    # are scientifically equivalent to a same-year final-CDL
    # vegetation export for historical training.  Keep them as
    # fallback candidates, but only use a year if the canonical
    # corn_modis_same_year inputs do not already cover it.
    # ========================================================

    if re.match(
        r"^corn_modis_20\d{2}_final_cdl(?:\s*\(\d+\))?\.csv$",
        lower,
        flags=re.IGNORECASE,
    ):

        final_cdl_historical_candidates.append(
            f
        )

        continue


    # ========================================================
    # RAW PRISM WEATHER
    # ========================================================

    if (
        "prism" in lower
        or
        "cornbelt_prism_weather" in lower
    ):

        prism_csvs.append(
            f
        )

        continue


    # ========================================================
    # COUNTY YIELD
    # ========================================================

    if (
        "yield" in lower
        and
        "modis" not in lower
        and
        "prism" not in lower
        and
        "weather" not in lower
        and
        "fixed_data" not in lower
    ):

        yield_candidates.append(
            f
        )


same_year_csvs = sorted(
    same_year_csvs
)

final_cdl_historical_candidates = sorted(
    final_cdl_historical_candidates
)

prism_csvs = sorted(
    prism_csvs
)

yield_candidates = sorted(
    yield_candidates
)


# ============================================================
# DUPLICATE-DOWNLOAD PROTECTION
# ============================================================

def _normalized_download_name(path):

    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)

    # Browser duplicates:
    #   file.csv
    #   file (1).csv
    #   file (2).csv
    stem = re.sub(
        r"\s*\(\d+\)$",
        "",
        stem,
    )

    return (
        stem.lower()
        +
        ext.lower()
    )


def _dedupe_file_list(paths):

    groups = {}

    for path in paths:

        groups.setdefault(
            _normalized_download_name(path),
            [],
        ).append(path)


    chosen = []

    for copies in groups.values():

        chosen.append(
            max(
                copies,
                key=os.path.getmtime,
            )
        )


    return sorted(
        chosen
    )


same_year_csvs = _dedupe_file_list(
    same_year_csvs
)

final_cdl_historical_candidates = _dedupe_file_list(
    final_cdl_historical_candidates
)

prism_csvs = _dedupe_file_list(
    prism_csvs
)

yield_candidates = _dedupe_file_list(
    yield_candidates
)


print(
    "\n" + "=" * 75
)

print(
    "CANONICAL MODEL INPUTS"
)

print(
    "=" * 75
)

print(
    "Same-year historical MODIS:",
    len(
        same_year_csvs
    )
)

print(
    "Historical FINAL_CDL fallback candidates:",
    len(
        final_cdl_historical_candidates
    )
)

print(
    "PRISM weather files:",
    len(
        prism_csvs
    )
)

print(
    "Yield candidates:",
    len(
        yield_candidates
    )
)

print(
    "=" * 75
)


print(
    "\nSame-year files:"
)

for f in same_year_csvs:

    print(
        " ",
        os.path.basename(f)
    )


print(
    "\nFINAL_CDL fallback candidates:"
)

for f in final_cdl_historical_candidates:

    print(
        " ",
        os.path.basename(f)
    )


print(
    "\nPRISM files:"
)

for f in prism_csvs:

    print(
        " ",
        os.path.basename(f)
    )

# CELL 4 — BUILD GEE WIDE TABLE
# ============================================================

VEG_INDICES = [

    "NDVI",
    "EVI_scaled",
    "EVI2",
    "NDMI",
    "NDWI",
    "NIRv",
    "GCI"

]


def clean_fips(
    series
):

    return (
        series
        .astype(str)
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
        .str.zfill(5)
    )


def build_gee_wide(
    csv_list,
    label="GEE",
    force_year=None
):

    dfs = []


    if len(csv_list) == 0:

        print(
            f"⚠ No files for {label}"
        )

        return pd.DataFrame()


    for f in csv_list:

        print(
            "Loading:",
            os.path.basename(f)
        )


        df = pd.read_csv(
            f,
            low_memory=False
        )


        df.columns = [

            c.strip()

            for c in df.columns

        ]


        # ====================================================
        # COUNTY ID
        # ====================================================

        if "GEOID" not in df.columns:

            if "FIPS" in df.columns:

                df = df.rename(
                    columns={
                        "FIPS":
                            "GEOID"
                    }
                )

            else:

                print(
                    "Skipping — no GEOID/FIPS:",
                    os.path.basename(f)
                )

                continue


        # ====================================================
        # YEAR
        # ====================================================

        if "year" not in df.columns:

            if force_year is not None:

                df[
                    "year"
                ] = force_year

            else:

                raise KeyError(
                    f"{os.path.basename(f)} has no year."
                )


        # ====================================================
        # REQUIRED
        # ====================================================

        needed = [

            "GEOID",
            "date",
            "year"

        ] + VEG_INDICES


        missing = [

            c

            for c in needed

            if c not in df.columns

        ]


        if len(missing) > 0:

            print(
                "Skipping:",
                os.path.basename(f)
            )

            print(
                "Missing:",
                missing
            )

            continue


        df = df[
            needed
        ].copy()


        # ====================================================
        # TYPES
        # ====================================================

        df[
            "GEOID"
        ] = clean_fips(
            df[
                "GEOID"
            ]
        )


        df[
            "year"
        ] = pd.to_numeric(
            df[
                "year"
            ],
            errors="coerce"
        )


        df[
            "date"
        ] = pd.to_datetime(
            df[
                "date"
            ],
            errors="coerce"
        )


        df[
            "DOY"
        ] = df[
            "date"
        ].dt.dayofyear


        for col in VEG_INDICES:

            df[
                col
            ] = pd.to_numeric(
                df[
                    col
                ],
                errors="coerce"
            )


        # ====================================================
        # FIX MODIS NDVI SCALE IF NECESSARY
        # ====================================================

        ndvi_valid = (

            df[
                "NDVI"
            ]

            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )

            .dropna()

        )


        if len(
            ndvi_valid
        ) > 0:

            q95 = (
                ndvi_valid
                .abs()
                .quantile(
                    0.95
                )
            )


            if q95 > 2:

                print(
                    "  Scaling NDVI × 0.0001"
                )


                df[
                    "NDVI"
                ] = (

                    df[
                        "NDVI"
                    ]

                    *
                    0.0001

                )


        df.loc[

            ~df[
                "NDVI"
            ].between(
                -1,
                1
            ),

            "NDVI"

        ] = np.nan


        # ====================================================
        # VALID KEYS
        # ====================================================

        df = df[

            df[
                "year"
            ].notna()

            &

            df[
                "DOY"
            ].notna()

        ].copy()


        df[
            "year"
        ] = (
            df[
                "year"
            ]
            .astype(int)
        )


        df[
            "DOY"
        ] = (
            df[
                "DOY"
            ]
            .astype(int)
        )


        dfs.append(
            df
        )


    if len(dfs) == 0:

        raise RuntimeError(
            f"No usable {label} data."
        )


    # ========================================================
    # LONG
    # ========================================================

    gee_long = pd.concat(
        dfs,
        ignore_index=True
    )


    print(
        "\nRaw long shape:",
        gee_long.shape
    )


    duplicates = gee_long.duplicated(

        subset=[
            "GEOID",
            "year",
            "DOY"
        ],

        keep=False

    )


    print(
        "Duplicate GEOID-year-DOY rows:",
        int(
            duplicates.sum()
        )
    )


    if duplicates.any():
        duplicate_examples = (
            gee_long.loc[
                duplicates,
                ["GEOID", "year", "DOY"]
            ]
            .drop_duplicates()
            .head(20)
        )

        raise ValueError(
            f"{label}: duplicate GEOID-year-DOY observations detected. "
            "The audited workflow will NOT silently average overlapping MODIS inputs. "
            "This usually means both an annual whole-Corn-Belt export and overlapping "
            "state-level files are present, or duplicate source products were staged. "
            f"First duplicate keys:\n{duplicate_examples.to_string(index=False)}"
        )


    # ========================================================
    # WIDE
    # ========================================================

    wide_parts = []


    for col in VEG_INDICES:

        temp = gee_long.pivot_table(

            index=[
                "GEOID",
                "year"
            ],

            columns=
                "DOY",

            values=
                col,

            aggfunc=
                "mean"

        )


        temp.columns = [

            f"{col}_DOY_{int(d)}"

            for d in temp.columns

        ]


        wide_parts.append(
            temp
        )


    gee_wide = (

        pd.concat(
            wide_parts,
            axis=1
        )

        .reset_index()

        .sort_values(
            [
                "GEOID",
                "year"
            ]
        )

        .reset_index(
            drop=True
        )

    )


    print(
        f"\n{label} wide shape:",
        gee_wide.shape
    )


    print(
        "County-years:",
        gee_wide[
            [
                "GEOID",
                "year"
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


    ndvi_cols = [

        c

        for c in gee_wide.columns

        if c.startswith(
            "NDVI_DOY_"
        )

    ]


    if len(ndvi_cols) > 0:

        print(
            "NDVI range:",
            np.nanmin(
                gee_wide[
                    ndvi_cols
                ].values
            ),
            "to",
            np.nanmax(
                gee_wide[
                    ndvi_cols
                ].values
            )
        )


    return gee_wide

# ============================================================
# CELL 5 — BUILD CANONICAL HISTORICAL VEGETATION TABLE
# ============================================================

same_year_gee_wide = build_gee_wide(

    same_year_csvs,

    label=
        "SAME-YEAR FINAL CDL"

)


if same_year_gee_wide.empty:

    raise RuntimeError(
        "No usable SAME-YEAR historical vegetation data were built."
    )


# ------------------------------------------------------------
# FALL BACK TO YEAR-SPECIFIC FINAL_CDL SCENARIO EXPORTS
# ------------------------------------------------------------
#
# Example:
#   Corn_MODIS_2023_FINAL_CDL.csv
#
# If 2023 is NOT already covered by the canonical same-year files,
# this is valid historical same-year final-CDL vegetation and can
# fill that year.  If a canonical same_year export exists, it wins
# and the FINAL_CDL scenario file is left only for Cell 18.
# ------------------------------------------------------------

canonical_hist_years = set(
    pd.to_numeric(
        same_year_gee_wide["year"],
        errors="coerce",
    )
    .dropna()
    .astype(int)
    .unique()
)

fallback_wide_parts = []

for fallback_path in final_cdl_historical_candidates:

    fallback_name = os.path.basename(fallback_path)
    match = re.search(
        r"corn_modis_(20\d{2})_final_cdl",
        fallback_name,
        flags=re.IGNORECASE,
    )

    if match is None:
        continue

    fallback_year = int(match.group(1))

    if fallback_year in canonical_hist_years:
        print(
            f"Skipping FINAL_CDL historical fallback for {fallback_year}: "
            "canonical same-year vegetation already covers that year."
        )
        continue

    print(
        f"Using FINAL_CDL scenario export as historical same-year fallback for {fallback_year}: "
        f"{fallback_name}"
    )

    fallback_wide = build_gee_wide(
        [fallback_path],
        label=f"FINAL_CDL HISTORICAL FALLBACK {fallback_year}",
        force_year=fallback_year,
    )

    fallback_wide_parts.append(
        fallback_wide
    )
    canonical_hist_years.add(
        fallback_year
    )

if fallback_wide_parts:

    same_year_gee_wide = pd.concat(
        [same_year_gee_wide] + fallback_wide_parts,
        ignore_index=True,
        sort=False,
    )

    duplicate_hist = same_year_gee_wide.duplicated(
        subset=["GEOID", "year"],
        keep=False,
    )

    if duplicate_hist.any():
        examples = (
            same_year_gee_wide.loc[
                duplicate_hist,
                ["GEOID", "year"],
            ]
            .drop_duplicates()
            .head(20)
        )
        raise ValueError(
            "Historical vegetation contains duplicate county-years after FINAL_CDL fallback. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    same_year_gee_wide = (
        same_year_gee_wide
        .sort_values(["GEOID", "year"])
        .reset_index(drop=True)
    )


print(
    "\n" + "=" * 75
)

print(
    "CANONICAL HISTORICAL VEGETATION"
)

print(
    "=" * 75
)

print(
    "Same-year shape:",
    same_year_gee_wide.shape
)

# CELL 6 — LOAD + CLEAN COUNTY CORN YIELD
# ============================================================

if len(
    yield_candidates
) == 0:

    raise FileNotFoundError(
        "No yield CSV found."
    )


yield_file = max(

    yield_candidates,

    key=
        os.path.getsize

)


print(
    "Yield file:"
)

print(
    yield_file
)


yield_raw = pd.read_csv(
    yield_file,
    low_memory=False
)


yield_raw.columns = [

    c.strip()

    for c in yield_raw.columns

]


yield_df = yield_raw.copy()


yield_df = yield_df[

    (yield_df["Geo Level"] == "COUNTY")

    &

    (yield_df["Commodity"] == "CORN")

    &

    (
        yield_df["Data Item"]
        ==
        "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"
    )

    &

    (yield_df["Period"] == "YEAR")

    &

    (yield_df["Domain"] == "TOTAL")

].copy()


yield_df = yield_df[

    yield_df[
        "State ANSI"
    ].notna()

    &

    yield_df[
        "County ANSI"
    ].notna()

].copy()


state_ansi = pd.to_numeric(

    yield_df[
        "State ANSI"
    ],

    errors="coerce"

)


county_ansi = pd.to_numeric(

    yield_df[
        "County ANSI"
    ],

    errors="coerce"

)


yield_df[
    "year"
] = pd.to_numeric(

    yield_df[
        "Year"
    ],

    errors="coerce"

)


yield_df[
    "yield_bu_acre"
] = pd.to_numeric(

    yield_df[
        "Value"
    ]
    .astype(str)
    .str.replace(
        ",",
        "",
        regex=False
    ),

    errors="coerce"

)


valid = (

    state_ansi.notna()

    &

    county_ansi.notna()

    &

    yield_df[
        "year"
    ].notna()

    &

    yield_df[
        "yield_bu_acre"
    ].notna()

)


yield_df = yield_df.loc[
    valid
].copy()


yield_df[
    "FIPS"
] = (

    state_ansi.loc[
        valid
    ]

    .astype(int)

    .astype(str)

    .str.zfill(2)

    +

    county_ansi.loc[
        valid
    ]

    .astype(int)

    .astype(str)

    .str.zfill(3)

)


yield_df[
    "year"
] = (

    yield_df[
        "year"
    ]

    .astype(int)

)


yield_clean_df = (

    yield_df[
        [
            "FIPS",
            "year",
            "yield_bu_acre"
        ]
    ]

    .drop_duplicates(
        subset=[
            "FIPS",
            "year"
        ]
    )

    .sort_values(
        [
            "FIPS",
            "year"
        ]
    )

    .reset_index(
        drop=True
    )

)


print(
    "Yield shape:",
    yield_clean_df.shape
)

print(
    "Years:",
    yield_clean_df[
        "year"
    ].min(),
    "to",
    yield_clean_df[
        "year"
    ].max()
)

print(
    "Counties:",
    yield_clean_df[
        "FIPS"
    ].nunique()
)

# ============================================================
# CELL 7 — HISTORICAL YIELD FEATURES
#
# No current-year information enters hist_5yr/hist_3yr.
# No yield interpolation.
# ============================================================

# Build an explicit county × calendar-year grid before rolling. This makes
# hist_5yr mean the previous FIVE CALENDAR YEARS, not merely the previous
# five available rows. Missing annual yield remains missing; it is never
# interpolated.
_yield_min_year = int(yield_clean_df["year"].min())
_yield_max_year = int(yield_clean_df["year"].max())
_yield_fips = sorted(yield_clean_df["FIPS"].unique())

_yield_full_index = pd.MultiIndex.from_product(
    [
        _yield_fips,
        range(_yield_min_year, _yield_max_year + 1),
    ],
    names=["FIPS", "year"],
)

yield_features = (
    yield_clean_df
    .set_index(["FIPS", "year"])
    .reindex(_yield_full_index)
    .reset_index()
    .sort_values(["FIPS", "year"])
    .reset_index(drop=True)
)

yield_features["hist_5yr"] = (
    yield_features
    .groupby("FIPS")["yield_bu_acre"]
    .transform(
        lambda x: x.shift(1).rolling(5, min_periods=5).mean()
    )
)

yield_features["hist_3yr"] = (
    yield_features
    .groupby("FIPS")["yield_bu_acre"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    )
)

yield_features["yield_anomaly"] = (
    yield_features["yield_bu_acre"]
    - yield_features["hist_5yr"]
)

_missing_calendar_rows = int(yield_features["yield_bu_acre"].isna().sum())
print("Calendar-complete yield grid:", yield_features.shape)
print("Missing county-year yield cells retained as NaN:", _missing_calendar_rows)


print(
    "Rows:",
    len(
        yield_features
    )
)

print(
    "hist_5yr available:",
    yield_features[
        "hist_5yr"
    ].notna().sum()
)

# ============================================================
# CELL 8 — MERGE YIELD FEATURES INTO VEGETATION
# ============================================================

def prepare_vegetation_dataset(
    vegetation_df,
    label
):

    df = vegetation_df.copy()


    if "FIPS" not in df.columns:

        if "GEOID" in df.columns:

            df = df.rename(
                columns={
                    "GEOID":
                        "FIPS"
                }
            )

        else:

            raise KeyError(
                f"{label}: no FIPS/GEOID."
            )


    df[
        "FIPS"
    ] = clean_fips(
        df[
            "FIPS"
        ]
    )


    df[
        "year"
    ] = pd.to_numeric(
        df[
            "year"
        ],
        errors="coerce"
    )


    df = df[
        df[
            "year"
        ].notna()
    ].copy()


    df[
        "year"
    ] = (
        df[
            "year"
        ]
        .astype(int)
    )


    dup = df.duplicated(

        subset=[
            "FIPS",
            "year"
        ]

    ).sum()


    if dup > 0:

        raise ValueError(
            f"{label}: {dup} duplicate county-years."
        )


    df = df.merge(

        yield_features[
            [
                "FIPS",
                "year",
                "yield_bu_acre",
                "hist_5yr",
                "hist_3yr",
                "yield_anomaly"
            ]
        ],

        on=[
            "FIPS",
            "year"
        ],

        how="left",

        validate="one_to_one"

    )


    print(
        "\n",
        label
    )

    print(
        "Shape:",
        df.shape
    )

    print(
        "Yield rows:",
        df[
            "yield_bu_acre"
        ].notna().sum()
    )

    print(
        "hist_5yr rows:",
        df[
            "hist_5yr"
        ].notna().sum()
    )


    return df


same_year_cdl_df = prepare_vegetation_dataset(

    same_year_gee_wide,

    "SAME-YEAR FINAL CDL"

)

# ============================================================
# CELL 9 — STRICT VEGETATION ANOMALIES
#
# Current year excluded using shift(1).
# ============================================================

def add_vegetation_anomalies(
    input_df,
    label
):

    df = input_df.copy()


    df = (

        df

        .sort_values(
            [
                "FIPS",
                "year"
            ]
        )

        .reset_index(
            drop=True
        )

    )


    # Remove previous anomaly columns on rerun.

    old_anoms = [

        c

        for c in df.columns

        if c.endswith(
            "_anom"
        )

        and
        "_DOY_" in c

    ]


    if len(old_anoms) > 0:

        df = df.drop(
            columns=old_anoms
        )


    created = []


    for idx in VEG_INDICES:


        raw_cols = [

            c

            for c in df.columns

            if (
                c.startswith(
                    idx + "_DOY_"
                )

                and

                not c.endswith(
                    "_anom"
                )
            )

        ]


        raw_cols = sorted(

            raw_cols,

            key=lambda c:

                int(
                    c.split(
                        "_DOY_"
                    )[-1]
                )

        )


        for col in raw_cols:


            historical_mean = (

                df

                .groupby(
                    "FIPS"
                )[col]

                .transform(

                    lambda x:

                        x.shift(1)

                        .expanding(
                            min_periods=3
                        )

                        .mean()

                )

            )


            anomaly_col = (
                col
                +
                "_anom"
            )


            df[
                anomaly_col
            ] = (

                pd.to_numeric(
                    df[
                        col
                    ],
                    errors="coerce"
                )

                -

                historical_mean

            )


            created.append(
                anomaly_col
            )


    print(
        label,
        "| anomaly columns:",
        len(created)
    )


    return df


same_year_cdl_df = add_vegetation_anomalies(

    same_year_cdl_df,

    "SAME-YEAR FINAL CDL"

)

# ============================================================
# CELL 10 — FILTER MODEL ROWS
# ============================================================

def filter_model_rows(
    df,
    label
):

    out = df[

        df[
            "yield_bu_acre"
        ].notna()

        &

        df[
            "hist_5yr"
        ].notna()

    ].copy()


    out = out.reset_index(
        drop=True
    )


    print(
        "\n" + "=" * 75
    )

    print(
        label
    )

    print(
        "=" * 75
    )


    print(
        "Rows:",
        f"{len(out):,}"
    )

    print(
        "Counties:",
        out[
            "FIPS"
        ].nunique()
    )

    print(
        "Years:",
        sorted(
            out[
                "year"
            ].unique()
        )
    )


    return out


same_year_model_df = filter_model_rows(

    same_year_cdl_df,

    "SAME-YEAR FINAL-CDL MODEL DATA"

)


# ============================================================
# CANONICAL TRAINING TABLE
# ============================================================

model_df = same_year_model_df.copy()


print(
    "\n✓ model_df = SAME-YEAR FINAL-CDL historical training"
)

# ============================================================
# CELL 11 — COMPACT RAW PRISM WEATHER
#
# Final production weather block selected by ablation:
#
#   15 physically meaningful raw compact PRISM variables.
#
# No PRISM anomaly features.
# No AWC / AWC interactions.
# No SOC / CEC.
# No LST.
# ============================================================

import os
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. RAW PRISM VARIABLES AVAILABLE IN THE SOURCE FILES
# ============================================================

RAW_SOURCE_VARS = [

    "tmin",
    "tmean",
    "tmax",

    "ppt",

    "vpdmin",
    "vpdmean",
    "vpdmax",

    "heat30",
    "hot35days",
    "drydays"

]


# ============================================================
# 2. FINAL 15-VARIABLE COMPACT WEATHER BLOCK
# ============================================================

RAW_PRISM = [

    "recent16_tmin",
    "recent16_tmean",
    "recent16_tmax",

    "recent16_ppt",
    "recent32_ppt",
    "season_ppt",

    "recent16_vpdmean",
    "recent16_vpdmax",
    "season_vpdmean",

    "recent16_heat30",
    "season_heat30",

    "recent16_hot35days",
    "season_hot35days",

    "recent16_drydays",
    "season_drydays"

]


# The canonical production environment contains ONLY these 15 features.
BASE_ENV = list(
    RAW_PRISM
)


# ============================================================
# 3. HELPERS
# ============================================================

def infer_year_from_filename(path):

    matches = re.findall(
        r"(20\d{2})",
        os.path.basename(path)
    )

    if len(matches) == 0:

        return None

    return int(
        matches[-1]
    )


def find_existing_column(
    columns,
    candidates
):

    lookup = {

        str(c).lower():
            c

        for c in columns

    }


    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]


    return None


def discover_doys(
    columns
):

    doys = set()


    for col in columns:

        match = re.search(

            r"_DOY_(\d+)$",

            str(col),

            flags=re.IGNORECASE

        )


        if match:

            doys.add(
                int(
                    match.group(1)
                )
            )


    return sorted(
        doys
    )


def find_doy_column(
    columns,
    prefix,
    doy
):

    lookup = {

        str(c).lower():
            c

        for c in columns

    }


    candidates = [

        f"{prefix}_DOY_{doy}",

        f"{prefix}_DOY_{doy:03d}"

    ]


    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[
                candidate.lower()
            ]


    return None


# ============================================================
# 4. LOAD WIDE PRISM FILES -> LONG CHECKPOINT TABLE
# ============================================================

if len(
    prism_csvs
) == 0:

    raise FileNotFoundError(
        "No PRISM weather files found."
    )


prism_parts = []


for path in prism_csvs:


    print(
        "Loading PRISM:",
        os.path.basename(path)
    )


    wide = pd.read_csv(
        path,
        low_memory=False
    )


    wide.columns = [

        str(c).strip()

        for c in wide.columns

    ]


    # --------------------------------------------------------
    # FIPS
    # --------------------------------------------------------

    fips_col = find_existing_column(

        wide.columns,

        [
            "FIPS",
            "GEOID"
        ]

    )


    if fips_col is None:

        raise KeyError(
            "No FIPS/GEOID found in "
            +
            os.path.basename(path)
        )


    wide[
        "FIPS"
    ] = clean_fips(
        wide[
            fips_col
        ]
    )


    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    year_col = find_existing_column(

        wide.columns,

        [
            "year",
            "YEAR",
            "Year"
        ]

    )


    if year_col is not None:

        wide[
            "year"
        ] = pd.to_numeric(

            wide[
                year_col
            ],

            errors="coerce"

        )


    else:

        file_year = infer_year_from_filename(
            path
        )


        if file_year is None:

            raise ValueError(
                "Could not determine year for "
                +
                os.path.basename(path)
            )


        wide[
            "year"
        ] = file_year


    # --------------------------------------------------------
    # DOYs
    # --------------------------------------------------------

    doys = discover_doys(
        wide.columns
    )


    if len(
        doys
    ) == 0:

        raise ValueError(
            "No *_DOY_* columns found in "
            +
            os.path.basename(path)
        )


    print(
        "  DOYs:",
        doys
    )


    # --------------------------------------------------------
    # Verify the raw PRISM source variables.
    # --------------------------------------------------------

    first_doy = doys[
        0
    ]


    missing_raw = []


    for variable in RAW_SOURCE_VARS:


        found = find_doy_column(

            wide.columns,

            variable,

            first_doy

        )


        if found is None:

            missing_raw.append(
                variable
            )


    if len(
        missing_raw
    ) > 0:

        raise KeyError(
            "Missing raw PRISM variables in "
            +
            os.path.basename(path)
            +
            ":\n"
            +
            str(
                missing_raw
            )
        )


    # --------------------------------------------------------
    # Build one row per county-year-checkpoint.
    # --------------------------------------------------------

    for doy in doys:


        part = pd.DataFrame({

            "FIPS":
                wide[
                    "FIPS"
                ].values,

            "year":
                wide[
                    "year"
                ].values,

            "DOY":
                doy

        })


        for variable in RAW_SOURCE_VARS:


            source_col = find_doy_column(

                wide.columns,

                variable,

                doy

            )


            if source_col is None:

                part[
                    variable
                ] = np.nan


            else:

                part[
                    variable
                ] = pd.to_numeric(

                    wide[
                        source_col
                    ],

                    errors="coerce"

                ).values


        prism_parts.append(
            part
        )


# ============================================================
# 5. CONCATENATE YEARS + CLEAN KEYS
# ============================================================

environment_raw = pd.concat(

    prism_parts,

    ignore_index=True

)


environment_raw[
    "year"
] = pd.to_numeric(

    environment_raw[
        "year"
    ],

    errors="coerce"

)


environment_raw[
    "DOY"
] = pd.to_numeric(

    environment_raw[
        "DOY"
    ],

    errors="coerce"

)


environment_raw = environment_raw[

    environment_raw[
        "year"
    ].notna()

    &

    environment_raw[
        "DOY"
    ].notna()

].copy()


environment_raw[
    "year"
] = environment_raw[
    "year"
].astype(
    int
)


environment_raw[
    "DOY"
] = environment_raw[
    "DOY"
].astype(
    int
)


# ============================================================
# 6. HANDLE DUPLICATES DETERMINISTICALLY
# ============================================================

duplicate_count = environment_raw.duplicated(

    subset=[
        "FIPS",
        "year",
        "DOY"
    ],

    keep=False

).sum()


print(
    "\nDuplicate FIPS-year-DOY weather rows:",
    duplicate_count
)


if duplicate_count > 0:
    duplicate_weather_examples = (
        environment_raw.loc[
            environment_raw.duplicated(
                subset=["FIPS", "year", "DOY"],
                keep=False
            ),
            ["FIPS", "year", "DOY"]
        ]
        .drop_duplicates()
        .head(20)
    )

    raise ValueError(
        "Duplicate FIPS-year-DOY PRISM rows detected. The audited workflow will NOT "
        "silently average potentially different weather products. Remove overlapping "
        f"PRISM inputs. First duplicate keys:\n{duplicate_weather_examples.to_string(index=False)}"
    )


environment_raw = (

    environment_raw

    .sort_values(
        [
            "FIPS",
            "year",
            "DOY"
        ]
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 7. BUILD THE 15 COMPACT WEATHER FEATURES
# ============================================================

environment = environment_raw[
    [
        "FIPS",
        "year",
        "DOY"
    ]
].copy()


# Recent 16-day weather.
environment[
    "recent16_tmin"
] = environment_raw[
    "tmin"
]


environment[
    "recent16_tmean"
] = environment_raw[
    "tmean"
]


environment[
    "recent16_tmax"
] = environment_raw[
    "tmax"
]


environment[
    "recent16_ppt"
] = environment_raw[
    "ppt"
]


environment[
    "recent16_vpdmean"
] = environment_raw[
    "vpdmean"
]


environment[
    "recent16_vpdmax"
] = environment_raw[
    "vpdmax"
]


environment[
    "recent16_heat30"
] = environment_raw[
    "heat30"
]


environment[
    "recent16_hot35days"
] = environment_raw[
    "hot35days"
]


environment[
    "recent16_drydays"
] = environment_raw[
    "drydays"
]


# Previous 32 days of precipitation.
environment[
    "recent32_ppt"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "ppt"
    ]

    .transform(

        lambda x:

            x.rolling(
                2,
                min_periods=1
            )
            .sum()

    )

)


# Season-to-date precipitation.
environment[
    "season_ppt"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "ppt"
    ]

    .cumsum()

)


# Season-to-date mean VPD.
environment[
    "season_vpdmean"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "vpdmean"
    ]

    .transform(

        lambda x:

            x.expanding(
                min_periods=1
            )
            .mean()

    )

)


# Season-to-date heat / hot-day / dry-day stress.
environment[
    "season_heat30"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "heat30"
    ]

    .cumsum()

)


environment[
    "season_hot35days"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "hot35days"
    ]

    .cumsum()

)


environment[
    "season_drydays"
] = (

    environment_raw

    .groupby(
        [
            "FIPS",
            "year"
        ]
    )[
        "drydays"
    ]

    .cumsum()

)


# ============================================================
# 8. FINAL COMPACT-WEATHER DIAGNOSTICS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "CANONICAL COMPACT PRISM WEATHER READY"
)

print(
    "=" * 75
)


print(
    "Environment shape:",
    environment.shape
)


print(
    "Counties:",
    environment[
        "FIPS"
    ].nunique()
)


print(
    "Years:",
    environment[
        "year"
    ].min(),
    "to",
    environment[
        "year"
    ].max()
)


print(
    "DOYs:",
    sorted(
        environment[
            "DOY"
        ].unique()
    )
)


print(
    "BASE_ENV predictors:",
    len(
        BASE_ENV
    )
)


if len(
    BASE_ENV
) != 15:

    raise RuntimeError(
        "Canonical BASE_ENV must contain exactly 15 compact PRISM features."
    )


print(
    "✓ BASE_ENV = 15 compact raw PRISM predictors"
)


print(
    "\nMissing compact PRISM values:"
)


print(

    environment[
        BASE_ENV
    ]
    .isna()
    .sum()

)


display(
    environment.head(
        10
    )
)

# CELL 12 — BUILD VEGETATION FEATURES
#
# CRITICAL:
#
# 1. Remove DOYs > doy_cut
# 2. THEN interpolate season-to-date
# 3. THEN calculate derived features
#
# Future vegetation cannot enter an earlier forecast.
# ============================================================

def build_features(
    df_sub,
    doy_cut
):

    feature_list = []


    def get_doy(
        column
    ):

        value = (
            column
            .split(
                "_DOY_"
            )[-1]
            .replace(
                "_anom",
                ""
            )
        )


        return int(
            value
        )


    # ========================================================
    # SEASON-TO-DATE INTERPOLATION
    # ========================================================

    def interpolate_matrix(
        matrix,
        doys
    ):
        """Interpolate only gaps bracketed by real observations.

        np.interp normally clamps missing tails to the nearest observed value.
        That silently fabricates early/late-season vegetation. Here, leading
        and trailing gaps remain NaN and are handled later by the existing
        training-only imputation policy.
        """

        X = np.asarray(
            matrix,
            dtype=float
        ).copy()

        doys = np.asarray(
            doys,
            dtype=float
        )

        output = np.full_like(
            X,
            np.nan,
            dtype=float
        )

        for i in range(X.shape[0]):
            row = X[i]
            good = np.isfinite(row)
            n_good = int(good.sum())

            if n_good == 0:
                continue

            if n_good == 1:
                # Preserve the one real observation only. Do not invent the
                # rest of the season from a single value.
                output[i, good] = row[good]
                continue

            x_good = doys[good]
            y_good = row[good]

            # Defensive guard: DOY coordinates should already be unique after
            # pivoting, but refuse to fit/interpolate ambiguous coordinates.
            if np.unique(x_good).size < 2:
                output[i, good] = row[good]
                continue

            inside = (
                (doys >= np.min(x_good))
                &
                (doys <= np.max(x_good))
            )

            output[i, inside] = np.interp(
                doys[inside],
                x_good,
                y_good
            )

        return output


    def row_auc(
        values,
        doys
    ):

        good = np.isfinite(
            values
        )


        if good.sum() < 2:

            return np.nan


        x = doys[
            good
        ]

        y = values[
            good
        ]


        if hasattr(
            np,
            "trapezoid"
        ):

            return np.trapezoid(
                y,
                x
            )


        return np.trapz(
            y,
            x
        )


    def row_peak_doy(
        values,
        doys
    ):

        good = np.isfinite(
            values
        )


        if good.sum() == 0:

            return np.nan


        y = values[
            good
        ]

        x = doys[
            good
        ]


        return x[
            np.argmax(
                y
            )
        ]


    def row_slope(
        values,
        doys
    ):

        good = np.isfinite(
            values
        )


        if good.sum() < 2:

            return np.nan


        x_good = doys[good]
        y_good = values[good]

        if np.unique(x_good).size < 2:
            return np.nan

        try:
            return np.polyfit(
                x_good,
                y_good,
                1
            )[0]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.nan


    # ========================================================
    # LOOP INDICES
    # ========================================================

    for idx in VEG_INDICES:


        # ====================================================
        # RAW DOY COLUMNS
        # ====================================================

        raw_cols = [

            c

            for c in df_sub.columns

            if (
                c.startswith(
                    idx + "_DOY_"
                )

                and

                not c.endswith(
                    "_anom"
                )
            )

        ]


        raw_cols = [

            c

            for c in raw_cols

            if get_doy(
                c
            )
            <=
            doy_cut

        ]


        raw_cols = sorted(

            raw_cols,

            key=
                get_doy

        )


        if len(raw_cols) > 0:


            raw_doys = np.array(

                [
                    get_doy(c)
                    for c in raw_cols
                ],

                dtype=float

            )


            raw_matrix = (

                df_sub[
                    raw_cols
                ]

                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                .to_numpy(
                    dtype=float
                )

            )


            X = interpolate_matrix(

                raw_matrix,

                raw_doys

            )


            # =================================================
            # RAW DOY VALUES
            # =================================================

            feature_list.append(

                pd.DataFrame(

                    X,

                    index=
                        df_sub.index,

                    columns=
                        raw_cols

                )

            )


            # =================================================
            # SEASONAL FEATURES
            # =================================================

            feat = pd.DataFrame(
                index=
                    df_sub.index
            )


            Xdf = pd.DataFrame(

                X,

                index=
                    df_sub.index

            )


            feat[
                f"{idx}_mean"
            ] = Xdf.mean(
                axis=1
            )


            feat[
                f"{idx}_max"
            ] = Xdf.max(
                axis=1
            )


            feat[
                f"{idx}_min"
            ] = Xdf.min(
                axis=1
            )


            feat[
                f"{idx}_std"
            ] = Xdf.std(
                axis=1,
                ddof=0
            )


            feat[
                f"{idx}_last"
            ] = X[
                :,
                -1
            ]


            feat[
                f"{idx}_auc"
            ] = [

                row_auc(
                    row,
                    raw_doys
                )

                for row in X

            ]


            feat[
                f"{idx}_peak_doy"
            ] = [

                row_peak_doy(
                    row,
                    raw_doys
                )

                for row in X

            ]


            if X.shape[1] >= 3:

                feat[
                    f"{idx}_early_mean"
                ] = (

                    pd.DataFrame(

                        X[
                            :,
                            :3
                        ],

                        index=
                            df_sub.index

                    )

                    .mean(
                        axis=1
                    )

                )


            if X.shape[1] >= 2:


                delta_doy = (

                    raw_doys[
                        -1
                    ]

                    -

                    raw_doys[
                        -2
                    ]

                )


                if delta_doy > 0:

                    feat[
                        f"{idx}_slope_last"
                    ] = (

                        X[
                            :,
                            -1
                        ]

                        -

                        X[
                            :,
                            -2
                        ]

                    ) / delta_doy


            green_mask = (

                (raw_doys >= 60)

                &

                (raw_doys <= 120)

            )


            green_doys = raw_doys[
                green_mask
            ]


            green_X = X[
                :,
                green_mask
            ]


            if len(
                green_doys
            ) >= 2:

                feat[
                    f"{idx}_greenup_rate"
                ] = [

                    row_slope(
                        row,
                        green_doys
                    )

                    for row in green_X

                ]


            feature_list.append(
                feat
            )


        # ====================================================
        # ANOMALY FEATURES
        # ====================================================

        anom_cols = [

            c

            for c in df_sub.columns

            if (
                c.startswith(
                    idx + "_DOY_"
                )

                and

                c.endswith(
                    "_anom"
                )
            )

        ]


        anom_cols = [

            c

            for c in anom_cols

            if get_doy(
                c
            )
            <=
            doy_cut

        ]


        anom_cols = sorted(

            anom_cols,

            key=
                get_doy

        )


        if len(anom_cols) > 0:


            anom_doys = np.array(

                [
                    get_doy(c)
                    for c in anom_cols
                ],

                dtype=float

            )


            raw_anom = (

                df_sub[
                    anom_cols
                ]

                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                .to_numpy(
                    dtype=float
                )

            )


            X_anom = interpolate_matrix(

                raw_anom,

                anom_doys

            )


            feat_anom = pd.DataFrame(
                index=
                    df_sub.index
            )


            Adf = pd.DataFrame(

                X_anom,

                index=
                    df_sub.index

            )


            feat_anom[
                f"{idx}_anom_mean"
            ] = Adf.mean(
                axis=1
            )


            feat_anom[
                f"{idx}_anom_max"
            ] = Adf.max(
                axis=1
            )


            feat_anom[
                f"{idx}_anom_min"
            ] = Adf.min(
                axis=1
            )


            feat_anom[
                f"{idx}_anom_last"
            ] = X_anom[
                :,
                -1
            ]


            if X_anom.shape[1] >= 2:


                delta_doy = (

                    anom_doys[
                        -1
                    ]

                    -

                    anom_doys[
                        -2
                    ]

                )


                if delta_doy > 0:

                    feat_anom[
                        f"{idx}_anom_slope_last"
                    ] = (

                        X_anom[
                            :,
                            -1
                        ]

                        -

                        X_anom[
                            :,
                            -2
                        ]

                    ) / delta_doy


            feature_list.append(
                feat_anom
            )


    if len(
        feature_list
    ) == 0:

        return pd.DataFrame(
            index=
                df_sub.index
        )


    features = pd.concat(

        feature_list,

        axis=1

    )


    features = features.loc[
        :,
        ~features.columns.duplicated()
    ]


    features = features.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan

    )


    return features


print(
    "✓ build_features() ready"
)

# ============================================================
# CELL 13 — ENVIRONMENT LOOKUP
# ============================================================

environment[
    "FIPS"
] = clean_fips(
    environment[
        "FIPS"
    ]
)


env_indexed = (

    environment

    .set_index(
        [
            "FIPS",
            "year",
            "DOY"
        ]
    )

    .sort_index()

)


def get_environment(
    rows,
    doy
):
    """Return compact PRISM features and fail loudly on key misalignment."""

    keys = pd.MultiIndex.from_arrays(
        [
            rows["FIPS"].astype(str),
            rows["year"].astype(int),
            np.full(len(rows), int(doy)),
        ],
        names=["FIPS", "year", "DOY"]
    )

    X = (
        env_indexed
        .reindex(keys)[BASE_ENV]
        .copy()
    )

    X.index = rows.index

    if len(X) == 0:
        raise RuntimeError(
            f"DOY {doy}: PRISM lookup returned zero rows."
        )

    fully_missing_features = X.columns[X.isna().all(axis=0)].tolist()
    if fully_missing_features:
        raise RuntimeError(
            f"DOY {doy}: PRISM feature(s) are completely missing after key lookup: "
            f"{fully_missing_features}. Check FIPS/year/DOY alignment and source files."
        )

    fully_missing_rows = X.isna().all(axis=1)
    n_fully_missing_rows = int(fully_missing_rows.sum())
    if n_fully_missing_rows > 0:
        examples = rows.loc[fully_missing_rows, ["FIPS", "year"]].head(10)
        raise RuntimeError(
            f"DOY {doy}: {n_fully_missing_rows} requested county-year rows have NO PRISM "
            f"features. Example keys:\n{examples.to_string(index=False)}"
        )

    overall_coverage = float(X.notna().mean().mean())
    if overall_coverage < 0.99:
        warnings.warn(
            f"DOY {doy}: PRISM cell coverage is {overall_coverage:.2%}. "
            "Scattered missing values will use training-only median imputation."
        )

    return X


print(
    "✓ environment lookup ready"
)

# ============================================================
# CELL 14 — BUILD MODEL MATRICES
# ============================================================

LEAKAGE_COLS = [

    "yield_bu_acre",
    "yield_filled",
    "yield_anomaly",
    "target",
    "yield"

]


def build_model_matrices(
    train_rows,
    test_rows,
    doy
):

    # ========================================================
    # VEGETATION
    # ========================================================

    Xveg_train = build_features(
        train_rows,
        doy
    )


    Xveg_test = build_features(
        test_rows,
        doy
    )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    Xenv_train = get_environment(
        train_rows,
        doy
    )


    Xenv_test = get_environment(
        test_rows,
        doy
    )


    # ========================================================
    # CONCAT
    # ========================================================

    X_train = pd.concat(

        [
            Xveg_train,
            Xenv_train
        ],

        axis=1

    )


    X_test = pd.concat(

        [
            Xveg_test,
            Xenv_test
        ],

        axis=1

    )


    X_train = X_train.loc[
        :,
        ~X_train.columns.duplicated()
    ]


    X_test = X_test.loc[
        :,
        ~X_test.columns.duplicated()
    ]


    # ========================================================
    # REMOVE TARGET COLUMNS
    # ========================================================

    X_train = X_train.drop(
        columns=
            LEAKAGE_COLS,
        errors=
            "ignore"
    )


    X_test = X_test.drop(
        columns=
            LEAKAGE_COLS,
        errors=
            "ignore"
    )


    # ========================================================
    # EXPLICIT BASELINE + YEAR
    # ========================================================

    X_train[
        "hist_5yr"
    ] = train_rows[
        "hist_5yr"
    ].values


    X_test[
        "hist_5yr"
    ] = test_rows[
        "hist_5yr"
    ].values


    X_train[
        "year"
    ] = train_rows[
        "year"
    ].astype(float).values


    X_test[
        "year"
    ] = test_rows[
        "year"
    ].astype(float).values


    # ========================================================
    # NUMERIC ONLY
    # ========================================================

    X_train = X_train.select_dtypes(
        include=[
            np.number,
            "bool"
        ]
    )


    X_test = X_test.select_dtypes(
        include=[
            np.number,
            "bool"
        ]
    )


    # ========================================================
    # ALIGN COLUMNS
    # ========================================================

    common_cols = [

        c

        for c in X_train.columns

        if c in X_test.columns

    ]


    X_train = X_train[
        common_cols
    ]


    X_test = X_test[
        common_cols
    ]


    X_train = X_train.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    X_test = X_test.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    # ========================================================
    # REMOVE FEATURES WITH ZERO TRAIN INFORMATION
    # ========================================================

    useful = X_train.notna().any(
        axis=0
    )


    X_train = X_train.loc[
        :,
        useful
    ]


    X_test = X_test.loc[
        :,
        useful
    ]


    # ========================================================
    # TRAINING-ONLY MEDIAN IMPUTATION
    # ========================================================

    medians = X_train.median()


    X_train = X_train.fillna(
        medians
    )


    X_test = X_test.fillna(
        medians
    )


    return (
        X_train,
        X_test
    )


print(
    "✓ model matrix builder ready"
)

# ============================================================
# CELL 15 — XGBOOST MODEL
# ============================================================

def make_model(
    seed=42
):

    params = dict(
        XGB_PARAMS
    )

    params[
        "random_state"
    ] = seed


    return XGBRegressor(
        **params
    )


print(
    "✓ canonical XGBoost model ready"
)


# ============================================================
# CELL 16 — AUDITED FEATURE ABLATION
# ============================================================

DOY_LIST = [
    65, 81, 97, 113, 129, 145, 161,
    177, 193, 209, 225, 241, 257, 273,
]

REQUESTED_TEST_YEARS = list(
    range(VALIDATION_START_YEAR, VALIDATION_END_YEAR + 1)
)

ABLATION_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR_BASE,
    f"feature_ablation_V2_AUDITED_{VALIDATION_TAG}",
)
os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 105)
print("AUDITED FEATURE ABLATION")
print("=" * 105)
print("Requested test years:", REQUESTED_TEST_YEARS)
print("DOYs:", DOY_LIST)
print("Output directory:", ABLATION_OUTPUT_DIR)


# ============================================================
# 16A. VERIFY REQUESTED YEARS ARE ACTUALLY MODEL-READY
# ============================================================

available_model_years = sorted(
    int(y) for y in model_df["year"].dropna().unique()
)
missing_requested_years = [
    y for y in REQUESTED_TEST_YEARS
    if y not in available_model_years
]

print("\nModel-ready years:", available_model_years)

if missing_requested_years:
    print("\n" + "=" * 90)
    print("REQUESTED-YEAR ABLATION INPUT DIAGNOSTIC")
    print("=" * 90)
    for y in REQUESTED_TEST_YEARS:
        modis_n = int((same_year_gee_wide["year"] == y).sum()) if y in same_year_gee_wide["year"].values else 0
        yield_n = int((yield_clean_df["year"] == y).sum())
        hist_n = int(((yield_features["year"] == y) & yield_features["hist_5yr"].notna()).sum())
        final_n = int((model_df["year"] == y).sum())
        print(
            f"{y}: MODIS rows={modis_n} | yield rows={yield_n} | "
            f"hist_5yr rows={hist_n} | final model rows={final_n}"
        )
    print("=" * 90)
    raise RuntimeError(
        "Requested ablation years are missing from the final audited model table: "
        f"{missing_requested_years}."
    )


# ============================================================
# 16B. STATIC FILE DISCOVERY — AWC + SOC/CEC
# ============================================================

def _normalized_basename(path):
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return (stem + ext).lower()


def find_named_input(target_name):
    target = target_name.lower()
    matches = [
        p for p in all_csvs
        if _normalized_basename(p) == target
    ]
    if not matches:
        return None
    # Browser/ZIP duplicate protection: newest/largest copy wins deterministically.
    return max(
        matches,
        key=lambda p: (os.path.getmtime(p), os.path.getsize(p), p),
    )


AWC_PATH = find_named_input("CornBelt_SoilGrids_AWC.csv")
SOC_CEC_PATH = find_named_input("CornBelt_SoilGrids_SOC_CEC.csv")

if AWC_PATH is None:
    raise FileNotFoundError(
        "CornBelt_SoilGrids_AWC.csv was not found under --input-dir/ZIP inputs."
    )
if SOC_CEC_PATH is None:
    raise FileNotFoundError(
        "CornBelt_SoilGrids_SOC_CEC.csv was not found under --input-dir/ZIP inputs."
    )

print("\nAWC file     :", AWC_PATH)
print("SOC/CEC file :", SOC_CEC_PATH)


# ============================================================
# 16C. LOAD / AUDIT AWC100
# ============================================================

awc = pd.read_csv(AWC_PATH, low_memory=False)
awc.columns = [str(c).strip() for c in awc.columns]

if "FIPS" not in awc.columns:
    if "GEOID" in awc.columns:
        awc = awc.rename(columns={"GEOID": "FIPS"})
    else:
        raise KeyError("AWC file has no FIPS/GEOID column.")

awc["FIPS"] = clean_fips(awc["FIPS"])

awc_dup = int(awc["FIPS"].duplicated().sum())
if awc_dup:
    raise RuntimeError(
        f"AWC file contains {awc_dup} duplicate FIPS rows. "
        "Audited workflow refuses to silently drop duplicates."
    )

for col in ["AWC_0_100_mm_crop", "AWC_0_100_mm_all"]:
    if col not in awc.columns:
        raise KeyError(f"AWC file missing required column: {col}")
    awc[col] = pd.to_numeric(awc[col], errors="coerce")

awc["AWC100"] = awc["AWC_0_100_mm_crop"].fillna(
    awc["AWC_0_100_mm_all"]
)
awc_small = awc[["FIPS", "AWC100"]].copy()

if awc_small["AWC100"].notna().sum() == 0:
    raise RuntimeError("AWC100 is entirely missing after crop/all fallback.")

print(
    "AWC audit: counties=", awc_small["FIPS"].nunique(),
    "| non-missing=", int(awc_small["AWC100"].notna().sum()),
    "| range=", (float(awc_small["AWC100"].min()), float(awc_small["AWC100"].max())),
)


# ============================================================
# 16D. LOAD / AUDIT RAW SOC + CEC SOURCE COLUMNS
# ============================================================

soil = pd.read_csv(SOC_CEC_PATH, low_memory=False)
soil.columns = [str(c).strip() for c in soil.columns]

if "FIPS" not in soil.columns:
    if "GEOID" in soil.columns:
        soil = soil.rename(columns={"GEOID": "FIPS"})
    else:
        raise KeyError("SOC/CEC file has no FIPS/GEOID column.")

soil["FIPS"] = clean_fips(soil["FIPS"])
soil_dup = int(soil["FIPS"].duplicated().sum())
if soil_dup:
    raise RuntimeError(
        f"SOC/CEC file contains {soil_dup} duplicate FIPS rows. "
        "Audited workflow refuses to silently drop duplicates."
    )

# The old ablation's +SOC / +CEC blocks added the raw county soil columns.
# In the current SoilGrids export these are normally 20 SOC_* and 20 CEC_*
# columns (layer/profile × crop/all). Discover them explicitly by prefix.
SOC_SOURCE_COLS = sorted([
    c for c in soil.columns
    if str(c).upper().startswith("SOC_")
])
CEC_SOURCE_COLS = sorted([
    c for c in soil.columns
    if str(c).upper().startswith("CEC_")
])

if not SOC_SOURCE_COLS:
    raise RuntimeError("No SOC_* columns were found in CornBelt_SoilGrids_SOC_CEC.csv.")
if not CEC_SOURCE_COLS:
    raise RuntimeError("No CEC_* columns were found in CornBelt_SoilGrids_SOC_CEC.csv.")

for col in SOC_SOURCE_COLS + CEC_SOURCE_COLS:
    soil[col] = pd.to_numeric(soil[col], errors="coerce")

soil_small = soil[["FIPS"] + SOC_SOURCE_COLS + CEC_SOURCE_COLS].copy()

print("SOC source columns:", len(SOC_SOURCE_COLS))
print("CEC source columns:", len(CEC_SOURCE_COLS))
if len(SOC_SOURCE_COLS) != 20:
    print("WARNING: prior ablation had +20 SOC features; current file exposes", len(SOC_SOURCE_COLS))
if len(CEC_SOURCE_COLS) != 20:
    print("WARNING: prior ablation had +20 CEC features; current file exposes", len(CEC_SOURCE_COLS))


# ============================================================
# 16E. STRICT PRISM ANOMALIES + STATIC SOILS
# ============================================================

PRISM_ANOM_MAP = {
    "recent16_tmax": "recent16_tmax_anom",
    "recent16_ppt": "recent16_ppt_anom",
    "recent16_vpdmean": "recent16_vpdmean_anom",
    "season_ppt": "season_ppt_anom",
    "season_vpdmean": "season_vpdmean_anom",
    "season_heat30": "season_heat30_anom",
    "season_hot35days": "season_hot35days_anom",
}
PRISM_ANOM_COLS = list(PRISM_ANOM_MAP.values())
AWC_INTERACTION_COLS = [
    "AWC100_x_heat",
    "AWC100_x_vpd",
    "AWC100_x_dry",
]

# Begin from the exact audited 15-feature canonical environment.
environment_ablation = environment.copy()
environment_ablation["FIPS"] = clean_fips(environment_ablation["FIPS"])

environment_ablation = (
    environment_ablation
    .sort_values(["FIPS", "DOY", "year"])
    .reset_index(drop=True)
)

# Weather anomaly climatologies use ONLY prior observations for the same
# county/checkpoint. This is a prior-history climatology, so missing years are
# not treated as adjacent; only available prior years contribute.
for raw_col, anomaly_col in PRISM_ANOM_MAP.items():
    if raw_col not in environment_ablation.columns:
        raise KeyError(f"Raw PRISM feature missing before anomaly build: {raw_col}")

    historical_mean = (
        environment_ablation
        .groupby(["FIPS", "DOY"])[raw_col]
        .transform(
            lambda x: x.shift(1).expanding(min_periods=3).mean()
        )
    )

    environment_ablation[anomaly_col] = (
        pd.to_numeric(environment_ablation[raw_col], errors="coerce")
        - historical_mean
    )

# Static joins are audited many-to-one joins.
environment_ablation = environment_ablation.merge(
    awc_small,
    on="FIPS",
    how="left",
    validate="many_to_one",
)
environment_ablation = environment_ablation.merge(
    soil_small,
    on="FIPS",
    how="left",
    validate="many_to_one",
)

# Interactions exactly match the previous full environmental model.
environment_ablation["AWC100_x_heat"] = (
    environment_ablation["AWC100"]
    * environment_ablation["season_heat30_anom"]
)
environment_ablation["AWC100_x_vpd"] = (
    environment_ablation["AWC100"]
    * environment_ablation["season_vpdmean_anom"]
)
environment_ablation["season_ppt_deficit"] = -environment_ablation["season_ppt_anom"]
environment_ablation["AWC100_x_dry"] = (
    environment_ablation["AWC100"]
    * environment_ablation["season_ppt_deficit"]
)

ENV_ABLATION_ALL = (
    list(RAW_PRISM)
    + PRISM_ANOM_COLS
    + ["AWC100"]
    + AWC_INTERACTION_COLS
    + SOC_SOURCE_COLS
    + CEC_SOURCE_COLS
)

# Exact-key uniqueness is mandatory before lookup.
env_dup = int(
    environment_ablation.duplicated(
        subset=["FIPS", "year", "DOY"]
    ).sum()
)
if env_dup:
    raise RuntimeError(
        f"Ablation environment has {env_dup} duplicate FIPS-year-DOY rows."
    )

env_ablation_indexed = (
    environment_ablation
    .set_index(["FIPS", "year", "DOY"])
    .sort_index()
)

print("\nEnvironment ablation blocks:")
print("  raw PRISM       :", len(RAW_PRISM))
print("  PRISM anomalies :", len(PRISM_ANOM_COLS))
print("  AWC static      : 1")
print("  AWC interactions:", len(AWC_INTERACTION_COLS))
print("  SOC raw         :", len(SOC_SOURCE_COLS))
print("  CEC raw         :", len(CEC_SOURCE_COLS))


# ============================================================
# 16F. GENERIC AUDITED ENVIRONMENT LOOKUP
# ============================================================

def get_environment_block(rows, doy, columns):
    columns = list(columns)
    if not columns:
        return pd.DataFrame(index=rows.index)

    missing_schema = [c for c in columns if c not in env_ablation_indexed.columns]
    if missing_schema:
        raise KeyError(f"Requested ablation environment columns missing: {missing_schema}")

    keys = pd.MultiIndex.from_arrays(
        [
            rows["FIPS"].astype(str),
            rows["year"].astype(int),
            np.full(len(rows), int(doy)),
        ],
        names=["FIPS", "year", "DOY"],
    )

    X = env_ablation_indexed.reindex(keys)[columns].copy()
    X.index = rows.index
    X = X.replace([np.inf, -np.inf], np.nan)

    # Catch catastrophic key/schema misses before median imputation can hide them.
    fully_missing = [c for c in columns if X[c].isna().all()]
    if fully_missing:
        raise RuntimeError(
            f"DOY {doy}: requested environment features are 100% missing: {fully_missing}"
        )

    return X


# ============================================================
# 16G. VEGETATION BLOCK SPLIT — PRESERVE CANONICAL COLUMN ORDER
# ============================================================

def get_vegetation_blocks(rows, doy):
    Xveg = build_features(rows, doy)
    if Xveg.empty:
        raise RuntimeError(f"DOY {doy}: build_features() returned no vegetation features.")

    # All audited anomaly-summary features contain '_anom_' in their name.
    anom_cols = [c for c in Xveg.columns if "_anom_" in c]
    raw_cols = [c for c in Xveg.columns if c not in anom_cols]

    return (
        Xveg,
        Xveg[raw_cols].copy(),
        Xveg[anom_cols].copy(),
    )


# ============================================================
# 16H. VARIANT DEFINITIONS
# ============================================================

CANONICAL_VARIANT = "Veg + anomalies + hist + year + raw weather"
CURRENT_FULL_VARIANT = "CURRENT FULL MODEL"

BUILD_UP_VARIANTS = [
    ("Raw vegetation only", {
        "raw_veg": True,
    }),
    ("Vegetation anomalies only", {
        "veg_anom": True,
    }),
    ("Raw + vegetation anomalies", {
        "raw_veg": True, "veg_anom": True,
    }),
    ("Veg + anomalies + hist_5yr", {
        "raw_veg": True, "veg_anom": True, "hist": True,
    }),
    ("Veg + anomalies + hist + year", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
    }),
    (CANONICAL_VARIANT, {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True,
    }),
    ("Veg + anomalies + hist + year + all weather", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
    }),
    ("Veg + anomalies + hist + year + weather + AWC", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True, "awc": True,
    }),
    (CURRENT_FULL_VARIANT, {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True,
    }),
    ("Current + SOC", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True, "soc": True,
    }),
    ("Current + CEC", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True, "cec": True,
    }),
    ("Current + SOC + CEC", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True, "soc": True, "cec": True,
    }),
]

DROP_ONE_VARIANTS = [
    (CURRENT_FULL_VARIANT, {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True,
    }),
    ("Current minus VEG ANOMALIES", {
        "raw_veg": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True,
    }),
    ("Current minus YEAR", {
        "raw_veg": True, "veg_anom": True, "hist": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True,
    }),
    ("Current minus WEATHER ANOMALIES", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True,
        "awc": True, "awc_interactions": True,
    }),
    # Keep static AWC100 but remove raw/anomalous weather plus interactions
    # that are mathematically weather-dependent. This reproduces the old count logic.
    ("Current minus WEATHER", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "awc": True,
    }),
    ("Current minus AWC", {
        "raw_veg": True, "veg_anom": True, "hist": True, "year": True,
        "raw_weather": True, "weather_anom": True,
    }),
    ("Current minus hist_5yr feature", {
        "raw_veg": True, "veg_anom": True, "year": True,
        "raw_weather": True, "weather_anom": True,
        "awc": True, "awc_interactions": True,
    }),
]


# ============================================================
# 16I. MATRIX ASSEMBLY / TRAINING-ONLY IMPUTATION
# ============================================================

def assemble_variant_matrix(rows, doy, spec):
    Xveg_all, Xveg_raw, Xveg_anom = get_vegetation_blocks(rows, doy)

    pieces = []

    # Preserve the canonical build_features() order whenever both veg blocks exist.
    if spec.get("raw_veg") and spec.get("veg_anom"):
        pieces.append(Xveg_all)
    elif spec.get("raw_veg"):
        pieces.append(Xveg_raw)
    elif spec.get("veg_anom"):
        pieces.append(Xveg_anom)

    env_cols = []
    if spec.get("raw_weather"):
        env_cols += list(RAW_PRISM)
    if spec.get("weather_anom"):
        env_cols += PRISM_ANOM_COLS
    if spec.get("awc"):
        env_cols += ["AWC100"]
    if spec.get("awc_interactions"):
        env_cols += AWC_INTERACTION_COLS
    if spec.get("soc"):
        env_cols += SOC_SOURCE_COLS
    if spec.get("cec"):
        env_cols += CEC_SOURCE_COLS

    if env_cols:
        pieces.append(get_environment_block(rows, doy, env_cols))

    if spec.get("hist"):
        pieces.append(pd.DataFrame(
            {"hist_5yr": pd.to_numeric(rows["hist_5yr"], errors="coerce").values},
            index=rows.index,
        ))

    if spec.get("year"):
        pieces.append(pd.DataFrame(
            {"year": pd.to_numeric(rows["year"], errors="coerce").astype(float).values},
            index=rows.index,
        ))

    if not pieces:
        raise RuntimeError(f"DOY {doy}: variant has no requested feature blocks.")

    X = pd.concat(pieces, axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X = X.drop(columns=LEAKAGE_COLS, errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"])
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def finalize_train_test(X_train, X_test):
    # Test feature space is forced to the training feature names/order.
    X_test = X_test.reindex(columns=X_train.columns)

    useful = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, useful].copy()
    X_test = X_test.loc[:, useful].copy()

    medians = X_train.median()
    good_cols = medians.notna()
    X_train = X_train.loc[:, good_cols].copy()
    X_test = X_test.loc[:, good_cols].copy()
    medians = medians.loc[good_cols]

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise RuntimeError("NaN survived audited training-only median imputation.")

    return X_train, X_test


def build_variant_matrices(train_rows, test_rows, doy, spec):
    X_train = assemble_variant_matrix(train_rows, doy, spec)
    X_test = assemble_variant_matrix(test_rows, doy, spec)
    return finalize_train_test(X_train, X_test)


# ============================================================
# 16J. CANONICAL PARITY CHECK
# ============================================================

def assert_canonical_parity(train_rows, test_rows, doy, X_variant_train, X_variant_test):
    X_can_train, X_can_test = build_model_matrices(train_rows, test_rows, doy)

    if list(X_variant_train.columns) != list(X_can_train.columns):
        only_variant = [c for c in X_variant_train.columns if c not in X_can_train.columns]
        only_canon = [c for c in X_can_train.columns if c not in X_variant_train.columns]
        raise RuntimeError(
            f"DOY {doy}: canonical parity column mismatch. "
            f"Only ablation={only_variant[:20]} | Only canonical={only_canon[:20]}"
        )

    tr_diff = float(np.nanmax(np.abs(
        X_variant_train.to_numpy(dtype=float) - X_can_train.to_numpy(dtype=float)
    ))) if X_variant_train.size else 0.0
    te_diff = float(np.nanmax(np.abs(
        X_variant_test.to_numpy(dtype=float) - X_can_test.to_numpy(dtype=float)
    ))) if X_variant_test.size else 0.0

    if not np.allclose(
        X_variant_train.to_numpy(dtype=float),
        X_can_train.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(f"DOY {doy}: canonical parity FAILED in training matrix; max diff={tr_diff}")

    if not np.allclose(
        X_variant_test.to_numpy(dtype=float),
        X_can_test.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(f"DOY {doy}: canonical parity FAILED in test matrix; max diff={te_diff}")

    return X_can_train, X_can_test, tr_diff, te_diff


# ============================================================
# 16K. FIT / SCORE ONE VARIANT
# ============================================================

def fit_score_variant(train_rows, test_rows, doy, X_train, X_test):
    y_train = train_rows["yield_anomaly"].to_numpy(dtype=float)
    actual = test_rows["yield_bu_acre"].to_numpy(dtype=float)
    hist = test_rows["hist_5yr"].to_numpy(dtype=float)

    model = make_model(seed=42)
    model.fit(X_train, y_train)

    pred_anomaly = model.predict(X_test)
    pred_yield = hist + pred_anomaly

    return {
        "R2": r2_score(actual, pred_yield),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred_yield))),
        "MAE": mean_absolute_error(actual, pred_yield),
        "N": len(actual),
        "N_Train": len(train_rows),
        "N_Features": X_train.shape[1],
    }


# ============================================================
# 16L. RUN ALL BUILD-UP + DROP-ONE VARIANTS
# ============================================================

all_results = []
parity_rows = []
feature_count_rows = []

for doy in DOY_LIST:
    print("\n" + "#" * 105)
    print(f"DOY {doy}")
    print("#" * 105)

    for test_year in REQUESTED_TEST_YEARS:
        train_rows = model_df[
            (model_df["year"] < test_year)
            & model_df["yield_bu_acre"].notna()
            & model_df["hist_5yr"].notna()
        ].copy()

        test_rows = model_df[
            (model_df["year"] == test_year)
            & model_df["yield_bu_acre"].notna()
            & model_df["hist_5yr"].notna()
        ].copy()

        if train_rows.empty or test_rows.empty:
            raise RuntimeError(
                f"DOY {doy}, year {test_year}: empty train/test rows in audited ablation."
            )

        print(
            f"\n{test_year} | train={len(train_rows):,} | test={len(test_rows):,}"
        )

        # Cache matrices and fitted metrics for identical specs within this year/DOY.
        # CURRENT FULL appears in both build_up and drop_one, so it is fit only once.
        matrix_cache = {}
        metrics_cache = {}

        for family, variant_defs in [
            ("build_up", BUILD_UP_VARIANTS),
            ("drop_one", DROP_ONE_VARIANTS),
        ]:
            for variant_name, spec in variant_defs:
                # CURRENT FULL appears in both families but must retain a row in both.
                cache_key = tuple(sorted(k for k, v in spec.items() if v))

                if cache_key not in matrix_cache:
                    X_train, X_test = build_variant_matrices(
                        train_rows, test_rows, doy, spec
                    )
                    matrix_cache[cache_key] = (X_train, X_test)
                else:
                    X_train, X_test = matrix_cache[cache_key]

                # The canonical production variant MUST be byte-for-byte/numerically
                # equivalent to the audited canonical matrix before fitting.
                if family == "build_up" and variant_name == CANONICAL_VARIANT:
                    X_can_train, X_can_test, tr_diff, te_diff = assert_canonical_parity(
                        train_rows,
                        test_rows,
                        doy,
                        X_train,
                        X_test,
                    )
                    X_train, X_test = X_can_train, X_can_test
                    matrix_cache[cache_key] = (X_train, X_test)
                    parity_rows.append({
                        "TestYear": test_year,
                        "DOY": doy,
                        "N_Features": X_train.shape[1],
                        "MaxAbsDiff_Train": tr_diff,
                        "MaxAbsDiff_Test": te_diff,
                        "Passed": True,
                    })

                if cache_key not in metrics_cache:
                    metrics_cache[cache_key] = fit_score_variant(
                        train_rows, test_rows, doy, X_train, X_test
                    )
                metrics = dict(metrics_cache[cache_key])

                row = {
                    "Family": family,
                    "Variant": variant_name,
                    "DOY": doy,
                    "TestYear": test_year,
                    **metrics,
                }
                all_results.append(row)

                feature_count_rows.append({
                    "Family": family,
                    "Variant": variant_name,
                    "DOY": doy,
                    "TestYear": test_year,
                    "N_Features": metrics["N_Features"],
                })

                print(
                    f"  {family:8s} | {variant_name:55s} "
                    f"R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.2f} | "
                    f"p={metrics['N_Features']}"
                )


# ============================================================
# 16M. RESULTS / SUMMARY TABLES
# ============================================================

results_df = pd.DataFrame(all_results)
parity_df = pd.DataFrame(parity_rows)
feature_counts_df = pd.DataFrame(feature_count_rows)

expected_rows = (
    len(DOY_LIST)
    * len(REQUESTED_TEST_YEARS)
    * (len(BUILD_UP_VARIANTS) + len(DROP_ONE_VARIANTS))
)
if len(results_df) != expected_rows:
    raise RuntimeError(
        f"Ablation produced {len(results_df)} rows; expected {expected_rows}."
    )

if len(parity_df) != len(DOY_LIST) * len(REQUESTED_TEST_YEARS):
    raise RuntimeError("Canonical parity did not run for every year/DOY.")
if not parity_df["Passed"].all():
    raise RuntimeError("At least one canonical parity check failed.")

summary_by_doy = (
    results_df
    .groupby(["Family", "Variant", "DOY"], as_index=False)
    .agg(
        Mean_R2=("R2", "mean"),
        Median_R2=("R2", "median"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Years=("TestYear", "count"),
        Mean_N_Features=("N_Features", "mean"),
    )
)

overall_summary = (
    results_df
    .groupby(["Family", "Variant"], as_index=False)
    .agg(
        Mean_R2=("R2", "mean"),
        Median_R2=("R2", "median"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Evaluations=("R2", "count"),
        Mean_N_Features=("N_Features", "mean"),
    )
    .sort_values(["Family", "Mean_R2"], ascending=[True, False])
)

OPERATIONAL_DOYS = [161, 193, 225, 257, 273]
operational_summary = (
    results_df[results_df["DOY"].isin(OPERATIONAL_DOYS)]
    .groupby(["Family", "Variant"], as_index=False)
    .agg(
        Mean_R2=("R2", "mean"),
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Evaluations=("R2", "count"),
    )
    .sort_values(["Family", "Mean_R2"], ascending=[True, False])
)


# ============================================================
# 16N. INCREMENTAL BUILD-UP GAINS
# ============================================================

BUILD_UP_PARENT = {
    "Vegetation anomalies only": None,
    "Raw + vegetation anomalies": "Raw vegetation only",
    "Veg + anomalies + hist_5yr": "Raw + vegetation anomalies",
    "Veg + anomalies + hist + year": "Veg + anomalies + hist_5yr",
    CANONICAL_VARIANT: "Veg + anomalies + hist + year",
    "Veg + anomalies + hist + year + all weather": CANONICAL_VARIANT,
    "Veg + anomalies + hist + year + weather + AWC": "Veg + anomalies + hist + year + all weather",
    CURRENT_FULL_VARIANT: "Veg + anomalies + hist + year + weather + AWC",
    "Current + SOC": CURRENT_FULL_VARIANT,
    "Current + CEC": CURRENT_FULL_VARIANT,
    "Current + SOC + CEC": CURRENT_FULL_VARIANT,
}

build_df = results_df[results_df["Family"] == "build_up"].copy()
build_delta_rows = []
for child, parent in BUILD_UP_PARENT.items():
    if parent is None:
        continue
    child_df = build_df[build_df["Variant"] == child]
    parent_df = build_df[build_df["Variant"] == parent]
    merged = child_df.merge(
        parent_df,
        on=["DOY", "TestYear"],
        suffixes=("_child", "_parent"),
        validate="one_to_one",
    )
    for _, r in merged.iterrows():
        build_delta_rows.append({
            "Variant": child,
            "Parent": parent,
            "DOY": int(r["DOY"]),
            "TestYear": int(r["TestYear"]),
            "Delta_R2": r["R2_child"] - r["R2_parent"],
            "Delta_RMSE": r["RMSE_child"] - r["RMSE_parent"],
            "Delta_MAE": r["MAE_child"] - r["MAE_parent"],
        })

build_delta_df = pd.DataFrame(build_delta_rows)
build_delta_summary = (
    build_delta_df
    .groupby(["Variant", "Parent"], as_index=False)
    .agg(
        Mean_Delta_R2=("Delta_R2", "mean"),
        Mean_Delta_RMSE=("Delta_RMSE", "mean"),
        Mean_Delta_MAE=("Delta_MAE", "mean"),
        Wins_R2=("Delta_R2", lambda x: int((x > 0).sum())),
        Evaluations=("Delta_R2", "count"),
    )
    .sort_values("Mean_Delta_R2", ascending=False)
)


# ============================================================
# 16O. DROP-ONE LOSSES
# ============================================================

drop_df = results_df[results_df["Family"] == "drop_one"].copy()
current_drop = drop_df[drop_df["Variant"] == CURRENT_FULL_VARIANT]

drop_loss_rows = []
for variant_name, _ in DROP_ONE_VARIANTS:
    if variant_name == CURRENT_FULL_VARIANT:
        continue
    child = drop_df[drop_df["Variant"] == variant_name]
    merged = current_drop.merge(
        child,
        on=["DOY", "TestYear"],
        suffixes=("_full", "_removed"),
        validate="one_to_one",
    )
    for _, r in merged.iterrows():
        drop_loss_rows.append({
            "Variant_Removed": variant_name,
            "DOY": int(r["DOY"]),
            "TestYear": int(r["TestYear"]),
            # Positive means the removed block helped the full model.
            "R2_Loss_When_Removed": r["R2_full"] - r["R2_removed"],
            "RMSE_Increase_When_Removed": r["RMSE_removed"] - r["RMSE_full"],
            "MAE_Increase_When_Removed": r["MAE_removed"] - r["MAE_full"],
        })

drop_loss_df = pd.DataFrame(drop_loss_rows)
drop_loss_summary = (
    drop_loss_df
    .groupby("Variant_Removed", as_index=False)
    .agg(
        Mean_R2_Loss=("R2_Loss_When_Removed", "mean"),
        Mean_RMSE_Increase=("RMSE_Increase_When_Removed", "mean"),
        Mean_MAE_Increase=("MAE_Increase_When_Removed", "mean"),
        Helped_Count=("R2_Loss_When_Removed", lambda x: int((x > 0).sum())),
        Evaluations=("R2_Loss_When_Removed", "count"),
    )
    .sort_values("Mean_R2_Loss", ascending=False)
)


# ============================================================
# 16P. SAVE EVERYTHING
# ============================================================

paths = {
    "all_results": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_all_heldout_results.csv"),
    "summary_by_doy": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_summary_by_DOY.csv"),
    "overall_summary": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_overall_summary.csv"),
    "operational_summary": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_operational_summary.csv"),
    "build_delta": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_build_up_deltas.csv"),
    "build_delta_summary": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_build_up_delta_summary.csv"),
    "drop_loss": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_drop_one_losses.csv"),
    "drop_loss_summary": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_drop_one_summary.csv"),
    "parity": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_canonical_parity.csv"),
    "feature_counts": os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_feature_counts.csv"),
}

results_df.to_csv(paths["all_results"], index=False)
summary_by_doy.to_csv(paths["summary_by_doy"], index=False)
overall_summary.to_csv(paths["overall_summary"], index=False)
operational_summary.to_csv(paths["operational_summary"], index=False)
build_delta_df.to_csv(paths["build_delta"], index=False)
build_delta_summary.to_csv(paths["build_delta_summary"], index=False)
drop_loss_df.to_csv(paths["drop_loss"], index=False)
drop_loss_summary.to_csv(paths["drop_loss_summary"], index=False)
parity_df.to_csv(paths["parity"], index=False)
feature_counts_df.to_csv(paths["feature_counts"], index=False)


# ============================================================
# 16Q. PLOTS
# ============================================================

key_variants = [
    "Raw vegetation only",
    "Raw + vegetation anomalies",
    "Veg + anomalies + hist + year",
    CANONICAL_VARIANT,
    "Veg + anomalies + hist + year + all weather",
    "Veg + anomalies + hist + year + weather + AWC",
    CURRENT_FULL_VARIANT,
]

plt.figure(figsize=(13, 7))
for variant in key_variants:
    tmp = summary_by_doy[
        (summary_by_doy["Family"] == "build_up")
        & (summary_by_doy["Variant"] == variant)
    ].sort_values("DOY")
    if not tmp.empty:
        plt.plot(tmp["DOY"], tmp["Mean_R2"], marker="o", label=variant)
plt.xlabel("Day of Year")
plt.ylabel("Mean held-out-year R²")
plt.title(f"Audited Feature Build-Up — {VALIDATION_START_YEAR}–{VALIDATION_END_YEAR}")
plt.grid(alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plot1 = os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_R2_through_season.png")
plt.savefig(plot1, dpi=180)
plt.close()

plt.figure(figsize=(12, 6))
for variant in ["Current + SOC", "Current + CEC", "Current + SOC + CEC"]:
    tmp = build_delta_df[build_delta_df["Variant"] == variant]
    if tmp.empty:
        continue
    tmp = tmp.groupby("DOY", as_index=False)["Delta_R2"].mean().sort_values("DOY")
    plt.plot(tmp["DOY"], tmp["Delta_R2"], marker="o", label=variant)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Day of Year")
plt.ylabel("Mean ΔR² vs CURRENT FULL MODEL")
plt.title(f"Audited SOC / CEC Incremental Gain — {VALIDATION_START_YEAR}–{VALIDATION_END_YEAR}")
plt.grid(alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plot2 = os.path.join(ABLATION_OUTPUT_DIR, "feature_ablation_V2_AUDITED_SOC_CEC_gain.png")
plt.savefig(plot2, dpi=180)
plt.close()


# ============================================================
# 16R. FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 105)
print("AUDITED FEATURE ABLATION COMPLETE")
print("=" * 105)
print("Model version:", MODEL_VERSION)
print("Validation years:", REQUESTED_TEST_YEARS)
print("Evaluations:", len(results_df))
print("Canonical parity checks:", len(parity_df), "passed")

print("\nBUILD-UP OVERALL SUMMARY")
display(
    overall_summary[overall_summary["Family"] == "build_up"]
    .round(4)
)

print("\nDROP-ONE SUMMARY (positive R² loss = removed block helped)")
display(drop_loss_summary.round(4))

print("\nBUILD-UP INCREMENTAL GAIN SUMMARY")
display(build_delta_summary.round(4))

print("\nSaved outputs:")
for label, path in paths.items():
    print(f"  {label:22s}: {path}")
print("  plot R2               :", plot1)
print("  plot SOC/CEC gain     :", plot2)

print("\nTo extend to 2025 later: change ONLY VALIDATION_END_YEAR = 2025 near the top, then rerun.")
print("=" * 105)
