# ============================================================
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORN YIELD 2 — Yield-model workflow (Cells 1–18)

Repository-ready conversion of the first 18 cells from the Colab notebook
"new model and old experiments".

Scientific workflow retained:
  1. Load MODIS / PRISM / yield / AWC inputs.
  2. Build same-year and lagged historical vegetation tables.
  3. Build leakage-free historical yield and vegetation anomalies.
  4. Build PRISM + AWC environmental features.
  5. Build season-to-date vegetation features.
  6. Run expanding-year seasonal validation.
  7. Run the 2022/2023 ICDL deployment comparison.

Local usage:
    python corn_yield_model_cells_01_18.py ^
        --input-dir "D:\\path\\to\\input_files" ^
        --work-dir "D:\\corn_yield\\yield_model_work" ^
        --output-dir "D:\\corn_yield\\yield_model_results"

The input directory may contain CSV files directly and/or ZIP archives.
ZIP archives are extracted into the work directory. The script does not
modify the source input directory.

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Corn Yield 2 — cells 1–18 yield-model workflow"
    )

    parser.add_argument(
        "--input-dir",
        default=".",
        help="Directory containing required CSVs and/or ZIP archives.",
    )

    parser.add_argument(
        "--work-dir",
        default="./corn_yield_model_work",
        help="Scratch directory used for extracted/staged inputs.",
    )

    parser.add_argument(
        "--output-dir",
        default="./corn_yield_model_results",
        help="Directory for Cell 18 result CSVs.",
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

for path in Path(INPUT_DIR).rglob("*"):
    if not path.is_file():
        continue

    resolved = path.resolve()

    # Never re-ingest our own work or result directories.
    try:
        resolved.relative_to(work_resolved)
        continue
    except ValueError:
        pass

    try:
        resolved.relative_to(output_resolved)
        continue
    except ValueError:
        pass

    suffix = path.suffix.lower()

    if suffix == ".csv":
        source_csvs.append(path)

    elif suffix == ".zip":
        source_zips.append(path)


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

lagged_csvs = []

prism_csvs = []

yield_candidates = []

awc_candidates = []


for f in all_csvs:

    name = os.path.basename(
        f
    )

    lower = name.lower()


    # ========================================================
    # SAME-YEAR MODIS
    # ========================================================

    if (
        "corn_modis_same_year" in lower
        and
        "2008_2022" in lower
    ):

        same_year_csvs.append(
            f
        )

        continue


    # ========================================================
    # LAGGED MODIS
    # ========================================================

    if (
        "corn_modis_lagged" in lower
        or
        (
            "corn_modis" in lower
            and
            "lagged" in lower
        )
    ):

        lagged_csvs.append(
            f
        )

        continue


    # ========================================================
    # PRISM
    # ========================================================

    if (
        "prism" in lower
        or
        "weather" in lower
    ):

        prism_csvs.append(
            f
        )

        continue


    # ========================================================
    # AWC
    # ========================================================

    if (
        "awc" in lower
        and
        "prism" not in lower
    ):

        awc_candidates.append(
            f
        )

        continue


    # ========================================================
    # YIELD
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

lagged_csvs = sorted(
    lagged_csvs
)

prism_csvs = sorted(
    prism_csvs
)

yield_candidates = sorted(
    yield_candidates
)

awc_candidates = sorted(
    awc_candidates
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

    return sorted(chosen)


same_year_csvs = _dedupe_file_list(same_year_csvs)
lagged_csvs = _dedupe_file_list(lagged_csvs)
prism_csvs = _dedupe_file_list(prism_csvs)
yield_candidates = _dedupe_file_list(yield_candidates)
awc_candidates = _dedupe_file_list(awc_candidates)

# ============================================================
# 2023 MASK SCENARIOS
# ============================================================

scenario_filenames = {

    "Previous-year CDL":
        "Corn_MODIS_2023_PREVIOUS_CDL.csv",

    "Final CDL":
        "Corn_MODIS_2023_FINAL_CDL.csv",

    "Study June":
        "Corn_MODIS_2023_STUDY_JUNE.csv",

    "Study July":
        "Corn_MODIS_2023_STUDY_JULY.csv",

    "Study August":
        "Corn_MODIS_2023_STUDY_AUGUST.csv",

    "Our June":
        "Corn_MODIS_2023_OUR_JUNE.csv",

    "Our July":
        "Corn_MODIS_2023_OUR_JULY.csv",

    "Our August":
        "Corn_MODIS_2023_OUR_AUGUST.csv"

}


scenario_paths = {}


for label, expected in scenario_filenames.items():

    matches = [

        f

        for f in all_csvs

        if (
            os.path.basename(f).lower()
            ==
            expected.lower()
        )

    ]


    if len(matches) > 0:

        scenario_paths[
            label
        ] = matches[0]


# ============================================================
# REPORT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "SAME-YEAR MODIS:",
    len(same_year_csvs)
)

print(
    "LAGGED MODIS:",
    len(lagged_csvs)
)

print(
    "PRISM:",
    len(prism_csvs)
)

print(
    "AWC candidates:",
    len(awc_candidates)
)

print(
    "Yield candidates:",
    len(yield_candidates)
)

print(
    "2023 mask scenarios:",
    len(scenario_paths)
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
    "\nLagged files:"
)

for f in lagged_csvs:

    print(
        " ",
        os.path.basename(f)
    )


print(
    "\n2023 scenarios:"
)

for k, v in scenario_paths.items():

    print(
        " ",
        k,
        "->",
        os.path.basename(v)
    )

# ============================================================
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

        gee_long = (

            gee_long

            .groupby(

                [
                    "GEOID",
                    "year",
                    "DOY"
                ],

                as_index=False

            )[VEG_INDICES]

            .mean()

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
# CELL 5 — BUILD HISTORICAL VEGETATION TABLES
# ============================================================

same_year_gee_wide = build_gee_wide(

    same_year_csvs,

    label=
        "SAME-YEAR CDL"

)


lagged_gee_wide = build_gee_wide(

    lagged_csvs,

    label=
        "LAGGED CDL"

)


print(
    "\n" + "=" * 75
)

print(
    "HISTORICAL VEGETATION TABLES"
)

print(
    "=" * 75
)


print(
    "Same-year:",
    same_year_gee_wide.shape
)

print(
    "Lagged:",
    lagged_gee_wide.shape
)

# ============================================================
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

yield_features = (

    yield_clean_df

    .copy()

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


yield_features[
    "hist_5yr"
] = (

    yield_features

    .groupby(
        "FIPS"
    )[
        "yield_bu_acre"
    ]

    .transform(

        lambda x:

            x.shift(1)

            .rolling(
                5,
                min_periods=5
            )

            .mean()

    )

)


yield_features[
    "hist_3yr"
] = (

    yield_features

    .groupby(
        "FIPS"
    )[
        "yield_bu_acre"
    ]

    .transform(

        lambda x:

            x.shift(1)

            .rolling(
                3,
                min_periods=3
            )

            .mean()

    )

)


yield_features[
    "yield_anomaly"
] = (

    yield_features[
        "yield_bu_acre"
    ]

    -

    yield_features[
        "hist_5yr"
    ]

)


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

    "SAME-YEAR CDL"

)


lagged_cdl_df = prepare_vegetation_dataset(

    lagged_gee_wide,

    "LAGGED CDL"

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

    "SAME-YEAR CDL"

)


lagged_cdl_df = add_vegetation_anomalies(

    lagged_cdl_df,

    "LAGGED CDL"

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

    "SAME-YEAR MODEL DATA"

)


lagged_model_df = filter_model_rows(

    lagged_cdl_df,

    "LAGGED MODEL DATA"

)


# ============================================================
# CANONICAL TRAINING TABLE
# ============================================================

model_df = same_year_model_df.copy()


print(
    "\n✓ model_df = SAME-YEAR CDL historical training"
)

# ============================================================
# CELL 11 — PRISM + AWC ENVIRONMENT
#
# CORRECT VERSION FOR OUR RAW PRISM EXPORTS
#
# Raw PRISM prefixes actually available:
#
#   tmin
#   tmean
#   tmax
#   ppt
#   vpdmin
#   vpdmean
#   vpdmax
#   heat30
#   hot35days
#   drydays
#
# Each raw variable exists at seasonal checkpoints:
#
#   *_DOY_065
#   *_DOY_081
#   ...
#   *_DOY_273
#
# We derive the compact 15-variable environmental block:
#
#   recent16_tmin
#   recent16_tmean
#   recent16_tmax
#   recent16_ppt
#   recent32_ppt
#   season_ppt
#   recent16_vpdmean
#   recent16_vpdmax
#   season_vpdmean
#   recent16_heat30
#   season_heat30
#   recent16_hot35days
#   season_hot35days
#   recent16_drydays
#   season_drydays
#
# Then create:
#
#   7 strict historical PRISM anomalies
#   AWC100
#   AWC × heat
#   AWC × VPD
#   AWC × precipitation deficit
#
# FINAL BASE_ENV = 26 predictors
# ============================================================

import os
import re

import numpy as np
import pandas as pd


# ============================================================
# 1. RAW PRISM VARIABLES IN THE FILES
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
# 2. FINAL COMPACT MODEL VARIABLES
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


# ============================================================
# 3. PRISM ANOMALIES RETAINED IN MODEL
# ============================================================

PRISM_ANOM_MAP = {

    "recent16_tmax":
        "recent16_tmax_anom",

    "recent16_ppt":
        "recent16_ppt_anom",

    "recent16_vpdmean":
        "recent16_vpdmean_anom",

    "season_ppt":
        "season_ppt_anom",

    "season_vpdmean":
        "season_vpdmean_anom",

    "season_heat30":
        "season_heat30_anom",

    "season_hot35days":
        "season_hot35days_anom"

}


# ============================================================
# 4. HELPERS
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
# 5. LOAD WIDE PRISM FILES -> LONG CHECKPOINT TABLE
# ============================================================

if len(prism_csvs) == 0:

    raise FileNotFoundError(
        "No PRISM files found."
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

        c.strip()

        for c in wide.columns

    ]


    # ========================================================
    # FIPS
    # ========================================================

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


    # ========================================================
    # YEAR
    # ========================================================

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


    # ========================================================
    # DISCOVER CHECKPOINT DOYS
    # ========================================================

    doys = discover_doys(
        wide.columns
    )


    if len(doys) == 0:

        raise ValueError(
            "No *_DOY_* columns found in "
            +
            os.path.basename(path)
        )


    print(
        "  DOYs:",
        doys
    )


    # ========================================================
    # VERIFY RAW SOURCE PREFIXES
    # ========================================================

    first_doy = doys[0]


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


    if len(missing_raw) > 0:

        raise KeyError(
            "Missing raw PRISM variables:\n"
            +
            str(
                missing_raw
            )
        )


    # ========================================================
    # BUILD ONE LONG TABLE PER CHECKPOINT
    # ========================================================

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
# 6. CONCATENATE ALL YEARS
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
] = (
    environment_raw[
        "year"
    ]
    .astype(int)
)


environment_raw[
    "DOY"
] = (
    environment_raw[
        "DOY"
    ]
    .astype(int)
)


# ============================================================
# 7. HANDLE DUPLICATES
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
    "\nDuplicate FIPS-year-DOY rows:",
    duplicate_count
)


if duplicate_count > 0:

    environment_raw = (

        environment_raw

        .groupby(

            [
                "FIPS",
                "year",
                "DOY"
            ],

            as_index=False

        )[RAW_SOURCE_VARS]

        .mean()

    )


# ============================================================
# 8. SORT WITHIN COUNTY-YEAR SEASON
# ============================================================

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
# 9. CREATE COMPACT 15-VARIABLE PRISM TABLE
# ============================================================

environment = environment_raw[
    [
        "FIPS",
        "year",
        "DOY"
    ]
].copy()


# ============================================================
# RECENT 16-DAY VARIABLES
#
# Raw checkpoint variables represent the current 16-day
# interval.
# ============================================================

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


# ============================================================
# 10. RECENT 32-DAY PRECIPITATION
#
# current 16-day interval + previous interval
# ============================================================

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


# ============================================================
# 11. SEASONAL CUMULATIVE PRECIPITATION
# ============================================================

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


# ============================================================
# 12. SEASONAL VPD MEAN
#
# All checkpoints represent approximately equal 16-day
# windows, so expanding mean is appropriate.
# ============================================================

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


# ============================================================
# 13. SEASONAL HEAT / HOT-DAY / DRY-DAY COUNTS
# ============================================================

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
# 14. VERIFY RAW PRISM COVERAGE
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "COMPACT PRISM VARIABLE COVERAGE"
)

print(
    "=" * 75
)


for col in RAW_PRISM:

    print(

        f"{col:24s}",

        f"{environment[col].notna().sum():,}"

    )


# ============================================================
# 15. STRICT HISTORICAL PRISM ANOMALIES
#
# Baseline:
#
#   same county
#   same DOY
#   PRIOR YEARS ONLY
#
# Current year excluded with shift(1).
# ============================================================

environment = (

    environment

    .sort_values(
        [
            "FIPS",
            "DOY",
            "year"
        ]
    )

    .reset_index(
        drop=True
    )

)


for raw_col, anomaly_col in PRISM_ANOM_MAP.items():


    historical_mean = (

        environment

        .groupby(
            [
                "FIPS",
                "DOY"
            ]
        )[raw_col]

        .transform(

            lambda x:

                x.shift(1)

                .expanding(
                    min_periods=3
                )

                .mean()

        )

    )


    environment[
        anomaly_col
    ] = (

        environment[
            raw_col
        ]

        -

        historical_mean

    )


# ============================================================
# 16. LOAD AWC
# ============================================================

if len(awc_candidates) == 0:

    raise FileNotFoundError(
        "No AWC CSV found."
    )


awc_file = max(

    awc_candidates,

    key=
        os.path.getsize

)


print(
    "\nAWC file:",
    os.path.basename(
        awc_file
    )
)


awc = pd.read_csv(
    awc_file,
    low_memory=False
)


awc.columns = [

    c.strip()

    for c in awc.columns

]


awc_fips_col = find_existing_column(

    awc.columns,

    [
        "FIPS",
        "GEOID"
    ]

)


if awc_fips_col is None:

    raise KeyError(
        "AWC file has no FIPS/GEOID."
    )


awc[
    "FIPS"
] = clean_fips(
    awc[
        awc_fips_col
    ]
)


# ============================================================
# ============================================================
# 17. BUILD AWC100
#
# Primary:
#   crop-specific 0–100 cm/mm profile value
#
# Fallback:
#   all-land 0–100 value
#
# Source columns in our SoilGrids file:
#
#   AWC_0_100_mm_crop
#   AWC_0_100_mm_all
# ============================================================

crop_awc_col = "AWC_0_100_mm_crop"
all_awc_col = "AWC_0_100_mm_all"


if crop_awc_col not in awc.columns:

    raise KeyError(
        f"Missing expected crop AWC column: {crop_awc_col}"
    )


if all_awc_col not in awc.columns:

    raise KeyError(
        f"Missing expected all-land AWC column: {all_awc_col}"
    )


awc[
    crop_awc_col
] = pd.to_numeric(

    awc[
        crop_awc_col
    ],

    errors="coerce"

)


awc[
    all_awc_col
] = pd.to_numeric(

    awc[
        all_awc_col
    ],

    errors="coerce"

)


# ============================================================
# CROP VALUE FIRST, ALL-LAND VALUE AS FALLBACK
# ============================================================

awc[
    "AWC100"
] = (

    awc[
        crop_awc_col
    ]

    .fillna(
        awc[
            all_awc_col
        ]
    )

)


# ============================================================
# COUNTY-LEVEL TABLE
# ============================================================

awc_small = (

    awc[
        [
            "FIPS",
            "AWC100"
        ]
    ]

    .drop_duplicates(
        subset=[
            "FIPS"
        ]
    )

)


print(
    "\nAWC100 source:"
)

print(
    "Primary:",
    crop_awc_col
)

print(
    "Fallback:",
    all_awc_col
)


print(
    "\nAWC county rows:",
    len(
        awc_small
    )
)


print(
    "AWC100 non-missing:",
    awc_small[
        "AWC100"
    ].notna().sum()
)


print(
    "AWC100 range:",
    awc_small[
        "AWC100"
    ].min(),
    "to",
    awc_small[
        "AWC100"
    ].max()
)
# ============================================================
# 18. MERGE AWC
# ============================================================

environment = environment.merge(

    awc_small,

    on=
        "FIPS",

    how=
        "left",

    validate=
        "many_to_one"

)


# ============================================================
# 19. AWC × STRESS INTERACTIONS
# ============================================================

environment[
    "AWC100_x_heat"
] = (

    environment[
        "AWC100"
    ]

    *

    environment[
        "season_heat30_anom"
    ]

)


environment[
    "AWC100_x_vpd"
] = (

    environment[
        "AWC100"
    ]

    *

    environment[
        "season_vpdmean_anom"
    ]

)


environment[
    "season_ppt_deficit"
] = (

    -
    environment[
        "season_ppt_anom"
    ]

)


environment[
    "AWC100_x_dry"
] = (

    environment[
        "AWC100"
    ]

    *

    environment[
        "season_ppt_deficit"
    ]

)


# ============================================================
# 20. FINAL FEATURE LIST
# ============================================================

PRISM_ANOM_COLS = list(
    PRISM_ANOM_MAP.values()
)


AWC_FEATURES = [

    "AWC100",

    "AWC100_x_heat",

    "AWC100_x_vpd",

    "AWC100_x_dry"

]


BASE_ENV = (

    RAW_PRISM

    +

    PRISM_ANOM_COLS

    +

    AWC_FEATURES

)


# ============================================================
# 21. FINAL DIAGNOSTICS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "ENVIRONMENT READY"
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
) == 26:

    print(
        "✓ BASE_ENV = 26 predictors"
    )

else:

    print(
        "⚠ BASE_ENV SHOULD BE 26"
    )


print(
    "\nMissing raw PRISM:"
)


print(

    environment[
        RAW_PRISM
    ]
    .isna()
    .sum()

)


print(
    "\nMissing anomaly predictors:"
)


print(

    environment[
        PRISM_ANOM_COLS
    ]
    .isna()
    .sum()

)


print(
    "\nAWC coverage:"
)


print(

    environment[
        "AWC100"
    ]
    .notna()
    .mean()

)


display(
    environment.head(10)
)

# ============================================================
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


        for i in range(
            X.shape[0]
        ):


            row = X[
                i
            ]


            good = np.isfinite(
                row
            )


            n_good = int(
                good.sum()
            )


            if n_good == 0:

                continue


            if n_good == 1:

                output[
                    i,
                    :
                ] = row[
                    good
                ][0]


                continue


            output[
                i,
                :
            ] = np.interp(

                doys,

                doys[
                    good
                ],

                row[
                    good
                ]

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


        return np.polyfit(

            doys[
                good
            ],

            values[
                good
            ],

            1

        )[0]


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

    keys = pd.MultiIndex.from_arrays(

        [

            rows[
                "FIPS"
            ].astype(str),

            rows[
                "year"
            ].astype(int),

            np.full(
                len(rows),
                int(doy)
            )

        ],

        names=[
            "FIPS",
            "year",
            "DOY"
        ]

    )


    X = (

        env_indexed

        .reindex(
            keys
        )[
            BASE_ENV
        ]

        .copy()

    )


    X.index = rows.index


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

    return XGBRegressor(

        n_estimators=
            500,

        max_depth=
            5,

        learning_rate=
            0.04,

        subsample=
            0.8,

        colsample_bytree=
            0.8,

        objective=
            "reg:squarederror",

        tree_method=
            "hist",

        n_jobs=
            -1,

        random_state=
            seed

    )

# ============================================================
# CELL 16 — EXPANDING-YEAR SEASONAL VALIDATION
#
# Train:
#   all years < held-out year
#
# Test:
#   held-out year
#
# Historical vegetation:
#   SAME-YEAR CDL
#
# Target:
#   yield anomaly
# ============================================================

DOY_LIST = [

    65,
    81,
    97,
    113,
    129,
    145,
    161,
    177,
    193,
    209,
    225,
    241,
    257,
    273

]


available_years = sorted(
    model_df[
        "year"
    ].unique()
)


TEST_YEARS = [

    y

    for y in range(
        2018,
        2023
    )

    if y in available_years

]


print(
    "Historical validation years:",
    TEST_YEARS
)


seasonal_results = []


for doy in DOY_LIST:


    print(
        "\nDOY",
        doy
    )


    for test_year in TEST_YEARS:


        train_rows = model_df[

            model_df[
                "year"
            ]
            <
            test_year

        ].copy()


        test_rows = model_df[

            model_df[
                "year"
            ]
            ==
            test_year

        ].copy()


        if (
            len(train_rows) == 0
            or
            len(test_rows) == 0
        ):

            continue


        X_train, X_test = build_model_matrices(

            train_rows,
            test_rows,
            doy

        )


        y_train = train_rows[
            "yield_anomaly"
        ].values


        model = make_model(
            seed=42
        )


        model.fit(
            X_train,
            y_train
        )


        pred_anomaly = model.predict(
            X_test
        )


        pred_yield = (

            test_rows[
                "hist_5yr"
            ].values

            +

            pred_anomaly

        )


        actual = test_rows[
            "yield_bu_acre"
        ].values


        r2 = r2_score(
            actual,
            pred_yield
        )


        rmse = np.sqrt(

            mean_squared_error(
                actual,
                pred_yield
            )

        )


        mae = mean_absolute_error(
            actual,
            pred_yield
        )


        seasonal_results.append({

            "DOY":
                doy,

            "TestYear":
                test_year,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "N":
                len(
                    test_rows
                )

        })


        print(

            test_year,

            "| R²",
            round(
                r2,
                3
            ),

            "| RMSE",
            round(
                rmse,
                2
            )

        )


seasonal_results_df = pd.DataFrame(
    seasonal_results
)


print(
    "\n✓ historical seasonal validation complete"
)

# ============================================================
# CELL 17 — SEASONAL PERFORMANCE SUMMARY
# ============================================================

seasonal_summary = (

    seasonal_results_df

    .groupby(
        "DOY"
    )

    .agg(

        Mean_R2=(
            "R2",
            "mean"
        ),

        Median_R2=(
            "R2",
            "median"
        ),

        Mean_RMSE=(
            "RMSE",
            "mean"
        ),

        Mean_MAE=(
            "MAE",
            "mean"
        ),

        Years=(
            "TestYear",
            "count"
        )

    )

    .reset_index()

)


display(
    seasonal_summary
)


plt.figure(
    figsize=(
        11,
        6
    )
)


plt.plot(

    seasonal_summary[
        "DOY"
    ],

    seasonal_summary[
        "Mean_R2"
    ],

    marker="o"

)


plt.xlabel(
    "Day of Year"
)

plt.ylabel(
    "Mean held-out-year R²"
)

plt.title(
    "Same-Year CDL Historical Training — Seasonal Yield Performance"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(
        11,
        6
    )
)


plt.plot(

    seasonal_summary[
        "DOY"
    ],

    seasonal_summary[
        "Mean_RMSE"
    ],

    marker="o"

)


plt.xlabel(
    "Day of Year"
)

plt.ylabel(
    "Mean RMSE (bu/ac)"
)

plt.title(
    "Seasonal Yield RMSE"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.show()

# ============================================================
# CELL 18 — 2022/2023 ICDL DEPLOYMENT TEST
#
# PURPOSE
# -------
# Use the model objects created by Cells 1–17 and test:
#
#   2022:
#     Previous-year CDL
#     Our Clean June / July / August as they become available
#     Final CDL oracle
#
#   2023:
#     Previous-year CDL
#     Our Clean June / July / August
#     Published Study June / July / August if present
#     Final CDL oracle
#
# IMPORTANT
# ---------
# • 2022 model trains ONLY on years < 2022.
# • 2023 model trains ONLY on years < 2023.
# • Test-year vegetation NEVER enters its anomaly baseline.
# • XGBoost is trained ONCE per year/DOY.
# • Training feature columns + medians are frozen before testing.
# • All scenarios within each year use the same county set.
# • Incremental:
#       2022 June works now.
#       Add July/August later and rerun this same cell.
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import re
import glob
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. VERIFY CELLS 1–17
# ============================================================

required_objects = [

    "model_df",

    "same_year_cdl_df",

    "yield_features",

    "environment",

    "BASE_ENV",

    "build_features",

    "build_gee_wide",

    "clean_fips",

    "get_environment",

    "make_model"

]


missing = [

    name

    for name
    in required_objects

    if name
    not in globals()

]


if missing:

    raise RuntimeError(

        "Cell 18 is missing objects from Cells 1–17:\n\n"

        +

        "\n".join(
            missing
        )

        +

        "\n\nRun Cells 1–17 first."

    )


print(
    "✓ Cells 1–17 objects found"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TEST_YEARS_CELL18 = [

    2022,

    2023

]


# ------------------------------------------------------------
# Our operational mask timing.
#
# June mask:
#   DOY 161, 177
#
# July:
#   193, 209
#
# August:
#   225 onward
#
# Therefore, right now 2022 June gives us TWO valid
# operational checkpoints while July/August finish.
# ------------------------------------------------------------

CHECKPOINT_DOYS = {

    "June": [

        161,
        177

    ],

    "July": [

        193,
        209

    ],

    "August": [

        225,
        241,
        257,
        273

    ]

}


SCENARIO_DIR = DATA_DIR

OUTPUT_DIR_CELL18 = os.path.join(
    OUTPUT_DIR_BASE,
    "cell18_2022_2023_icdl_results",
)

os.makedirs(
    OUTPUT_DIR_CELL18,
    exist_ok=True,
)


# ============================================================
# ============================================================
# 3. SCENARIO INPUTS
# ============================================================
#
# Repository/local version:
# scenario CSVs are discovered from DATA_DIR, which was staged
# from --input-dir in Cell 2. There is no interactive Colab upload.
#
# Supported scenario names include:
#
#   Corn_MODIS_2022_PREVIOUS_CDL.csv
#   Corn_MODIS_2022_FINAL_CDL.csv
#   Corn_MODIS_2022_CLEAN_REBUILT_JUNE.csv
#   Corn_MODIS_2022_CLEAN_REBUILT_JULY.csv
#   Corn_MODIS_2022_CLEAN_REBUILT_AUGUST.csv
#   Corn_MODIS_2022_STUDY_JUNE.csv
#   Corn_MODIS_2022_STUDY_JULY.csv
#   Corn_MODIS_2022_STUDY_AUGUST.csv
#
# and the equivalent 2023 files.
# ============================================================
# 4. FIND SCENARIO FILES
#
# Search:
#
#   DATA_DIR staged from --input-dir in Cell 2
#
# Also tolerates duplicate downloads:
#
#   file.csv
#   file (1).csv
# ============================================================

SEARCH_ROOTS_CELL18 = [
    DATA_DIR,
]


def normalize_filename(
    path
):


    name = os.path.basename(
        path
    )


    stem, ext = os.path.splitext(
        name
    )


    # --------------------------------------------------------
    # Example:
    #
    # Corn_MODIS_2022_FINAL_CDL (1).csv
    #
    # becomes:
    #
    # Corn_MODIS_2022_FINAL_CDL.csv
    # --------------------------------------------------------

    stem = re.sub(

        r"\s*\(\d+\)$",

        "",

        stem

    )


    return (

        stem

        +

        ext

    ).upper()


all_scenario_csvs = []


for root in SEARCH_ROOTS_CELL18:


    if not os.path.exists(
        root
    ):

        continue


    all_scenario_csvs.extend(

        glob.glob(

            os.path.join(

                root,

                "**",

                "*.csv"

            ),

            recursive=True

        )

    )


all_scenario_csvs = sorted(

    set(
        all_scenario_csvs
    )

)


# ============================================================
# 5. PARSE SCENARIO FILENAMES
# ============================================================

def parse_scenario_file(
    path
):


    name = normalize_filename(
        path
    )


    pattern = re.compile(

        r"^CORN_MODIS_(2022|2023)_"

        r"(PREVIOUS_CDL|FINAL_CDL|"

        r"CLEAN_REBUILT_(JUNE|JULY|AUGUST)|"

        r"STUDY_(JUNE|JULY|AUGUST)|"

        r"OUR_(JUNE|JULY|AUGUST))"

        r"\.CSV$",

        re.IGNORECASE

    )


    match = pattern.match(
        name
    )


    if match is None:

        return None


    year = int(
        match.group(1)
    )


    token = (
        match.group(2)
        .upper()
    )


    # --------------------------------------------------------
    # PREVIOUS-YEAR CDL
    # --------------------------------------------------------

    if token == "PREVIOUS_CDL":


        label = (
            "Previous-year CDL"
        )


        family = (
            "Previous"
        )


        checkpoint = None


    # --------------------------------------------------------
    # FINAL CDL
    # --------------------------------------------------------

    elif token == "FINAL_CDL":


        label = (
            "Final CDL"
        )


        family = (
            "Final"
        )


        checkpoint = None


    # --------------------------------------------------------
    # CLEAN REBUILT ICDL
    # --------------------------------------------------------

    elif token.startswith(
        "CLEAN_REBUILT_"
    ):


        checkpoint = (

            token

            .replace(
                "CLEAN_REBUILT_",
                ""
            )

            .title()

        )


        label = (

            "Our Clean "

            +

            checkpoint

        )


        family = (
            "Our Clean"
        )


    # --------------------------------------------------------
    # Old OUR_* filename compatibility
    # --------------------------------------------------------

    elif token.startswith(
        "OUR_"
    ):


        checkpoint = (

            token

            .replace(
                "OUR_",
                ""
            )

            .title()

        )


        label = (

            "Our Clean "

            +

            checkpoint

        )


        family = (
            "Our Clean"
        )


    # --------------------------------------------------------
    # PUBLISHED / STUDY ICDL
    # --------------------------------------------------------

    elif token.startswith(
        "STUDY_"
    ):


        checkpoint = (

            token

            .replace(
                "STUDY_",
                ""
            )

            .title()

        )


        label = (

            "Published "

            +

            checkpoint

        )


        family = (
            "Published"
        )


    else:

        return None


    return {

        "year":
            year,

        "label":
            label,

        "family":
            family,

        "checkpoint":
            checkpoint,

        "path":
            path

    }


parsed_candidates = []


for path in all_scenario_csvs:


    parsed = parse_scenario_file(
        path
    )


    if parsed is not None:


        parsed_candidates.append(
            parsed
        )


if len(
    parsed_candidates
) == 0:


    raise FileNotFoundError(

        "No 2022/2023 scenario CSVs were detected."

    )


# ============================================================
# 6. REMOVE DUPLICATE COPIES
#
# If multiple copies exist, use newest one.
# ============================================================

scenario_info = {}


for item in parsed_candidates:


    key = (

        item[
            "year"
        ],

        item[
            "label"
        ]

    )


    if key not in scenario_info:


        scenario_info[
            key
        ] = item


    else:


        old_path = scenario_info[
            key
        ][
            "path"
        ]


        if (

            os.path.getmtime(
                item[
                    "path"
                ]
            )

            >

            os.path.getmtime(
                old_path
            )

        ):


            scenario_info[
                key
            ] = item


print()

print(
    "=" * 100
)

print(
    "SCENARIO FILES DETECTED"
)

print(
    "=" * 100
)


for key in sorted(
    scenario_info.keys()
):


    item = scenario_info[
        key
    ]


    print(

        f"{item['year']} | "

        f"{item['label']:22s} | "

        f"{os.path.basename(item['path'])}"

    )


# ============================================================
# 7. LOAD SCENARIO MODIS CSVs -> WIDE COUNTY TABLES
#
# Reuse build_gee_wide() from Cell 4.
# ============================================================

scenario_raw_cell18 = {}


for (
    test_year,
    scenario_label
), item in scenario_info.items():


    print()

    print(
        "Loading:",
        test_year,
        scenario_label
    )


    wide = build_gee_wide(

        [
            item[
                "path"
            ]
        ],

        label=
            f"{test_year} {scenario_label}",

        force_year=
            test_year

    )


    if "FIPS" not in wide.columns:


        if "GEOID" in wide.columns:


            wide = wide.rename(

                columns={

                    "GEOID":
                        "FIPS"

                }

            )


        else:


            raise KeyError(

                f"{scenario_label}: no FIPS/GEOID after conversion."

            )


    wide[
        "FIPS"
    ] = clean_fips(

        wide[
            "FIPS"
        ]

    )


    wide[
        "year"
    ] = pd.to_numeric(

        wide[
            "year"
        ],

        errors=
            "coerce"

    )


    wide = wide[

        wide[
            "year"
        ]

        ==

        test_year

    ].copy()


    wide[
        "year"
    ] = (

        wide[
            "year"
        ]

        .astype(int)

    )


    wide = (

        wide

        .drop_duplicates(

            subset=[
                "FIPS",
                "year"
            ]

        )

        .reset_index(
            drop=True
        )

    )


    scenario_raw_cell18[
        (
            test_year,
            scenario_label
        )
    ] = wide


    print(

        "  counties:",

        wide[
            "FIPS"
        ].nunique()

    )


# ============================================================
# 8. YIELD LOOKUP TABLE
# ============================================================

yield_lookup_cell18 = (

    yield_features[
        [
            "FIPS",
            "year",
            "yield_bu_acre",
            "hist_5yr",
            "hist_3yr",
            "yield_anomaly"
        ]
    ]

    .copy()

)


yield_lookup_cell18[
    "FIPS"
] = clean_fips(

    yield_lookup_cell18[
        "FIPS"
    ]

)


yield_lookup_cell18[
    "year"
] = pd.to_numeric(

    yield_lookup_cell18[
        "year"
    ],

    errors=
        "coerce"

)


yield_lookup_cell18 = yield_lookup_cell18[

    yield_lookup_cell18[
        "year"
    ].notna()

].copy()


yield_lookup_cell18[
    "year"
] = (

    yield_lookup_cell18[
        "year"
    ]

    .astype(int)

)


yield_lookup_cell18 = (

    yield_lookup_cell18

    .drop_duplicates(

        subset=[
            "FIPS",
            "year"
        ]

    )

)


# ============================================================
# 9. HISTORICAL VEGETATION SOURCE
#
# CRITICAL:
#
# For TEST YEAR 2022:
#
#   anomaly baseline uses vegetation from years < 2022
#
# For TEST YEAR 2023:
#
#   anomaly baseline uses vegetation from years < 2023
#
# Test-year vegetation never contributes to itself.
# ============================================================

same_year_hist_source = (

    same_year_cdl_df

    .copy()

)


same_year_hist_source[
    "FIPS"
] = clean_fips(

    same_year_hist_source[
        "FIPS"
    ]

)


same_year_hist_source[
    "year"
] = pd.to_numeric(

    same_year_hist_source[
        "year"
    ],

    errors=
        "coerce"

)


same_year_hist_source = same_year_hist_source[

    same_year_hist_source[
        "year"
    ].notna()

].copy()


same_year_hist_source[
    "year"
] = (

    same_year_hist_source[
        "year"
    ]

    .astype(int)

)


# ============================================================
# 10. PREPARE TEST SCENARIO
# ============================================================

def prepare_test_scenario_cell18(
    raw_test,
    test_year
):


    test = raw_test.copy()


    test[
        "FIPS"
    ] = clean_fips(

        test[
            "FIPS"
        ]

    )


    test[
        "year"
    ] = int(
        test_year
    )


    # --------------------------------------------------------
    # Add target yield + leakage-free historical yield
    # --------------------------------------------------------

    test = test.merge(

        yield_lookup_cell18,

        on=[
            "FIPS",
            "year"
        ],

        how=
            "left",

        validate=
            "one_to_one"

    )


    # --------------------------------------------------------
    # ONLY historical vegetation before target year
    # --------------------------------------------------------

    hist = same_year_hist_source[

        same_year_hist_source[
            "year"
        ]

        <

        test_year

    ].copy()


    # --------------------------------------------------------
    # Build vegetation anomalies in the same naming format
    # expected by build_features().
    # --------------------------------------------------------

    for idx in VEG_INDICES:


        raw_cols = [

            col

            for col
            in same_year_hist_source.columns

            if (

                col.startswith(
                    idx + "_DOY_"
                )

                and

                not col.endswith(
                    "_anom"
                )

            )

        ]


        for col in raw_cols:


            if col not in test.columns:


                test[
                    col
                ] = np.nan


            county_hist_mean = (

                hist

                .groupby(
                    "FIPS"
                )[
                    col
                ]

                .mean()

            )


            test[
                col
                +
                "_anom"
            ] = (

                pd.to_numeric(

                    test[
                        col
                    ],

                    errors=
                        "coerce"

                )

                -

                test[
                    "FIPS"
                ]

                .map(
                    county_hist_mean
                )

            )


    return test


# ============================================================
# 11. PREPARE ALL DISCOVERED TEST SCENARIOS
# ============================================================

scenario_test_cell18 = {}


for (
    test_year,
    scenario_label
), raw in scenario_raw_cell18.items():


    scenario_test_cell18[
        (
            test_year,
            scenario_label
        )
    ] = prepare_test_scenario_cell18(

        raw,

        test_year

    )


# ============================================================
# 12. FORCE FAIR COMMON COUNTY SET
#
# SAME counties across masks WITHIN a year.
#
# 2022 and 2023 do not need the same county count.
# ============================================================

common_fips_by_year_cell18 = {}


for test_year in TEST_YEARS_CELL18:


    year_keys = [

        key

        for key
        in scenario_test_cell18.keys()

        if key[
            0
        ]
        ==
        test_year

    ]


    if len(
        year_keys
    ) == 0:

        continue


    valid_yield_fips = set(

        yield_lookup_cell18.loc[

            (

                yield_lookup_cell18[
                    "year"
                ]

                ==

                test_year

            )

            &

            yield_lookup_cell18[
                "yield_bu_acre"
            ].notna()

            &

            yield_lookup_cell18[
                "hist_5yr"
            ].notna(),

            "FIPS"

        ]

    )


    common_fips = (

        valid_yield_fips.copy()

    )


    for key in year_keys:


        common_fips &= set(

            scenario_test_cell18[
                key
            ][
                "FIPS"
            ]

        )


    common_fips = sorted(
        common_fips
    )


    common_fips_by_year_cell18[
        test_year
    ] = common_fips


    print()

    print(

        f"{test_year} common test counties:",

        len(
            common_fips
        )

    )


    for key in year_keys:


        scenario_test_cell18[
            key
        ] = (

            scenario_test_cell18[
                key
            ][

                scenario_test_cell18[
                    key
                ][
                    "FIPS"
                ].isin(
                    common_fips
                )

            ]

            .sort_values(
                "FIPS"
            )

            .reset_index(
                drop=True
            )

        )


# ============================================================
# 13. SCENARIO HELPERS
# ============================================================

def scenario_exists_cell18(
    year,
    label
):


    return (

        year,
        label

    ) in scenario_test_cell18


def scenario_for_family_cell18(
    year,
    family,
    doy
):


    if doy in CHECKPOINT_DOYS[
        "June"
    ]:


        checkpoint = (
            "June"
        )


    elif doy in CHECKPOINT_DOYS[
        "July"
    ]:


        checkpoint = (
            "July"
        )


    elif doy in CHECKPOINT_DOYS[
        "August"
    ]:


        checkpoint = (
            "August"
        )


    else:


        return None


    if family == "Our Clean":


        label = (

            "Our Clean "

            +

            checkpoint

        )


    elif family == "Published":


        label = (

            "Published "

            +

            checkpoint

        )


    else:


        return None


    if scenario_exists_cell18(
        year,
        label
    ):


        return label


    return None


def available_doys_for_year_cell18(
    year
):


    doys = []


    for checkpoint, checkpoint_doys in CHECKPOINT_DOYS.items():


        our_label = (

            "Our Clean "

            +

            checkpoint

        )


        study_label = (

            "Published "

            +

            checkpoint

        )


        if (

            scenario_exists_cell18(
                year,
                our_label
            )

            or

            scenario_exists_cell18(
                year,
                study_label
            )

        ):


            doys.extend(
                checkpoint_doys
            )


    return sorted(

        set(
            doys
        )

    )


# ============================================================
# 14. FROZEN TRAINING FEATURE SPACE
#
# IMPORTANT IMPROVEMENT:
#
# We do NOT allow each mask to alter the feature set.
#
# For every YEAR + DOY:
#
#   build training X once
#   freeze columns
#   freeze training medians
#   train model once
#
# Every scenario is then forced into that SAME feature space.
# ============================================================

LEAKAGE_COLS_CELL18_ICDL = [

    "yield_bu_acre",

    "yield_filled",

    "yield_anomaly",

    "target",

    "yield"

]


def build_frozen_training_matrix_cell18(
    train_rows,
    doy
):


    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    Xveg = build_features(

        train_rows,

        doy

    )


    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    Xenv = get_environment(

        train_rows,

        doy

    )


    X = pd.concat(

        [
            Xveg,
            Xenv
        ],

        axis=1

    )


    X = X.loc[

        :,

        ~X.columns.duplicated()

    ]


    # --------------------------------------------------------
    # Remove targets
    # --------------------------------------------------------

    X = X.drop(

        columns=
            LEAKAGE_COLS_CELL18_ICDL,

        errors=
            "ignore"

    )


    # --------------------------------------------------------
    # Same historical yield baseline as main model
    # --------------------------------------------------------

    X[
        "hist_5yr"
    ] = train_rows[
        "hist_5yr"
    ].values


    # --------------------------------------------------------
    # Same YEAR feature as current model
    # --------------------------------------------------------

    X[
        "year"
    ] = train_rows[
        "year"
    ].astype(
        float
    ).values


    # --------------------------------------------------------
    # Numeric only
    # --------------------------------------------------------

    X = X.select_dtypes(

        include=[
            np.number,
            "bool"
        ]

    )


    X = X.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan

    )


    # --------------------------------------------------------
    # Features with training information
    # --------------------------------------------------------

    useful = X.notna().any(
        axis=0
    )


    X = X.loc[
        :,
        useful
    ].copy()


    # --------------------------------------------------------
    # TRAINING-ONLY medians
    # --------------------------------------------------------

    medians = X.median()


    # Completely unusable training columns.
    good_cols = medians.notna()


    X = X.loc[
        :,
        good_cols
    ].copy()


    medians = medians.loc[
        good_cols
    ]


    X = X.fillna(
        medians
    )


    return (

        X,

        list(
            X.columns
        ),

        medians

    )


def build_frozen_test_matrix_cell18(

    test_rows,

    doy,

    training_columns,

    training_medians

):


    Xveg = build_features(

        test_rows,

        doy

    )


    Xenv = get_environment(

        test_rows,

        doy

    )


    X = pd.concat(

        [
            Xveg,
            Xenv
        ],

        axis=1

    )


    X = X.loc[

        :,

        ~X.columns.duplicated()

    ]


    X = X.drop(

        columns=
            LEAKAGE_COLS_CELL18_ICDL,

        errors=
            "ignore"

    )


    X[
        "hist_5yr"
    ] = test_rows[
        "hist_5yr"
    ].values


    X[
        "year"
    ] = test_rows[
        "year"
    ].astype(
        float
    ).values


    X = X.select_dtypes(

        include=[
            np.number,
            "bool"
        ]

    )


    X = X.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan

    )


    # --------------------------------------------------------
    # EXACT TRAINING FEATURE SPACE
    # --------------------------------------------------------

    X = X.reindex(

        columns=
            training_columns

    )


    # --------------------------------------------------------
    # EXACT TRAINING MEDIANS
    # --------------------------------------------------------

    X = X.fillna(

        training_medians

    )


    return X


# ============================================================
# 15. RUN 2022 + 2023 ICDL TEST
# ============================================================

results_cell18_icdl = []


print()

print(
    "=" * 105
)

print(
    "RUNNING 2022 / 2023 ICDL DEPLOYMENT TEST"
)

print(
    "=" * 105
)


for test_year in TEST_YEARS_CELL18:


    if test_year not in common_fips_by_year_cell18:


        print()

        print(

            test_year,

            "skipped — no scenario files."

        )


        continue


    eval_doys = available_doys_for_year_cell18(
        test_year
    )


    if len(
        eval_doys
    ) == 0:


        print()

        print(

            test_year,

            "skipped — no in-season ICDL checkpoint found."

        )


        continue


    print()

    print(
        "#" * 105
    )

    print(
        f"TEST YEAR {test_year}"
    )

    print(

        "Operational DOYs available:",

        eval_doys

    )

    print(
        "#" * 105
    )


    # --------------------------------------------------------
    # STRICT EXPANDING-YEAR TRAINING
    # --------------------------------------------------------

    train_rows = model_df[

        (

            model_df[
                "year"
            ]

            <

            test_year

        )

        &

        model_df[
            "yield_bu_acre"
        ].notna()

        &

        model_df[
            "hist_5yr"
        ].notna()

    ].copy()


    if len(
        train_rows
    ) == 0:


        print(
            "No training rows."
        )

        continue


    print(

        "Training years:",

        int(
            train_rows[
                "year"
            ].min()
        ),

        "to",

        int(
            train_rows[
                "year"
            ].max()
        ),

        "| rows:",

        f"{len(train_rows):,}"

    )


    # ========================================================
    # EACH OPERATIONAL DOY
    # ========================================================

    for doy in eval_doys:


        print()

        print(
            f"DOY {doy}"
        )


        # ====================================================
        # TRAIN ONCE
        # ====================================================

        (

            X_train,

            training_columns,

            training_medians

        ) = build_frozen_training_matrix_cell18(

            train_rows,

            doy

        )


        y_train = (

            train_rows[
                "yield_anomaly"
            ]

            .to_numpy()

        )


        model = make_model(
            seed=42
        )


        model.fit(

            X_train,

            y_train

        )


        # ====================================================
        # ACTIVE MASKS AT THIS DOY
        # ====================================================

        active_scenarios = []


        # ----------------------------------------------------
        # Previous-year CDL
        # ----------------------------------------------------

        if scenario_exists_cell18(

            test_year,

            "Previous-year CDL"

        ):


            active_scenarios.append(

                (

                    "Previous-year CDL",

                    "Previous-year CDL"

                )

            )


        # ----------------------------------------------------
        # OUR CLEAN OPERATIONAL
        # ----------------------------------------------------

        our_active = scenario_for_family_cell18(

            test_year,

            "Our Clean",

            doy

        )


        if our_active is not None:


            active_scenarios.append(

                (

                    "Our Clean Operational",

                    our_active

                )

            )


        # ----------------------------------------------------
        # PUBLISHED OPERATIONAL
        # ----------------------------------------------------

        published_active = scenario_for_family_cell18(

            test_year,

            "Published",

            doy

        )


        if published_active is not None:


            active_scenarios.append(

                (

                    "Published Operational",

                    published_active

                )

            )


        # ----------------------------------------------------
        # Final CDL oracle
        # ----------------------------------------------------

        if scenario_exists_cell18(

            test_year,

            "Final CDL"

        ):


            active_scenarios.append(

                (

                    "Final CDL",

                    "Final CDL"

                )

            )


        # ====================================================
        # TEST ALL ACTIVE MASKS ON SAME MODEL
        # ====================================================

        for result_label, source_label in active_scenarios:


            test_rows = scenario_test_cell18[

                (
                    test_year,
                    source_label
                )

            ].copy()


            X_test = build_frozen_test_matrix_cell18(

                test_rows=
                    test_rows,

                doy=
                    doy,

                training_columns=
                    training_columns,

                training_medians=
                    training_medians

            )


            predicted_anomaly = model.predict(
                X_test
            )


            predicted_yield = (

                test_rows[
                    "hist_5yr"
                ].to_numpy()

                +

                predicted_anomaly

            )


            actual_yield = (

                test_rows[
                    "yield_bu_acre"
                ].to_numpy()

            )


            r2 = r2_score(

                actual_yield,

                predicted_yield

            )


            rmse = np.sqrt(

                mean_squared_error(

                    actual_yield,

                    predicted_yield

                )

            )


            mae = mean_absolute_error(

                actual_yield,

                predicted_yield

            )


            results_cell18_icdl.append({

                "Test_Year":
                    test_year,

                "DOY":
                    doy,

                "Scenario":
                    result_label,

                "Source_File_Scenario":
                    source_label,

                "R2":
                    r2,

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "N":
                    len(
                        actual_yield
                    ),

                "N_Train":
                    len(
                        train_rows
                    ),

                "N_Features":
                    len(
                        training_columns
                    )

            })


            print(

                f"  {result_label:24s} "

                f"R²={r2:.4f} | "

                f"RMSE={rmse:.2f} | "

                f"N={len(actual_yield)}"

            )


# ============================================================
# 16. RESULTS DATAFRAME
# ============================================================

results_cell18_icdl = pd.DataFrame(

    results_cell18_icdl

)


if results_cell18_icdl.empty:


    raise RuntimeError(

        "No Cell 18 ICDL results were produced."

    )


r2_table_cell18_icdl = (

    results_cell18_icdl

    .pivot_table(

        index=[
            "Test_Year",
            "DOY"
        ],

        columns=
            "Scenario",

        values=
            "R2",

        aggfunc=
            "first"

    )

    .reset_index()

)


# ============================================================
# 17. WITHIN-YEAR MASK VALUE
# ============================================================

if (

    "Our Clean Operational"

    in

    r2_table_cell18_icdl.columns

    and

    "Previous-year CDL"

    in

    r2_table_cell18_icdl.columns

):


    r2_table_cell18_icdl[
        "Our_gain_vs_previous"
    ] = (

        r2_table_cell18_icdl[
            "Our Clean Operational"
        ]

        -

        r2_table_cell18_icdl[
            "Previous-year CDL"
        ]

    )


if (

    "Final CDL"

    in

    r2_table_cell18_icdl.columns

    and

    "Previous-year CDL"

    in

    r2_table_cell18_icdl.columns

):


    r2_table_cell18_icdl[
        "Oracle_gain_vs_previous"
    ] = (

        r2_table_cell18_icdl[
            "Final CDL"
        ]

        -

        r2_table_cell18_icdl[
            "Previous-year CDL"
        ]

    )


if (

    "Our_gain_vs_previous"

    in

    r2_table_cell18_icdl.columns

    and

    "Oracle_gain_vs_previous"

    in

    r2_table_cell18_icdl.columns

):


    r2_table_cell18_icdl[
        "Our_oracle_gain_reproduced_pct"
    ] = np.where(

        np.abs(

            r2_table_cell18_icdl[
                "Oracle_gain_vs_previous"
            ]

        )

        >

        0.002,

        100.0

        *

        r2_table_cell18_icdl[
            "Our_gain_vs_previous"
        ]

        /

        r2_table_cell18_icdl[
            "Oracle_gain_vs_previous"
        ],

        np.nan

    )


if (

    "Published Operational"

    in

    r2_table_cell18_icdl.columns

    and

    "Previous-year CDL"

    in

    r2_table_cell18_icdl.columns

):


    r2_table_cell18_icdl[
        "Published_gain_vs_previous"
    ] = (

        r2_table_cell18_icdl[
            "Published Operational"
        ]

        -

        r2_table_cell18_icdl[
            "Previous-year CDL"
        ]

    )


print()

print(
    "=" * 105
)

print(
    "CELL 18 — ICDL R² RESULTS"
)

print(
    "=" * 105
)


display(

    r2_table_cell18_icdl

    .round(
        4
    )

)


# ============================================================
# 18. PLOT EACH TEST YEAR
# ============================================================

plot_order = [

    "Previous-year CDL",

    "Our Clean Operational",

    "Published Operational",

    "Final CDL"

]


for test_year in sorted(

    results_cell18_icdl[
        "Test_Year"
    ].unique()

):


    year_results = results_cell18_icdl[

        results_cell18_icdl[
            "Test_Year"
        ]

        ==

        test_year

    ].copy()


    plt.figure(

        figsize=(
            11,
            6
        )

    )


    for scenario in plot_order:


        temp = year_results[

            year_results[
                "Scenario"
            ]

            ==

            scenario

        ].sort_values(
            "DOY"
        )


        if len(
            temp
        ) == 0:

            continue


        plt.plot(

            temp[
                "DOY"
            ],

            temp[
                "R2"
            ],

            marker="o",

            linewidth=2,

            label=
                scenario

        )


    plt.xlabel(
        "Day of Year"
    )


    plt.ylabel(
        f"{test_year} county-level R²"
    )


    plt.title(

        f"{test_year} ICDL Deployment Test — "
        "Same-Year CDL Historical Training"

    )


    plt.xticks(

        sorted(

            year_results[
                "DOY"
            ].unique()

        )

    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.tight_layout()


    plt.show()


# ============================================================
# 19. DIRECT CLEAN ICDL GAIN VS PREVIOUS CDL
#
# This is especially useful for comparing 2022 vs 2023.
#
# Absolute R² can differ naturally between years.
#
# ΔR² vs previous-year CDL better isolates mask value.
# ============================================================

if (

    "Our_gain_vs_previous"

    in

    r2_table_cell18_icdl.columns

):


    gain_rows = r2_table_cell18_icdl[

        r2_table_cell18_icdl[
            "Our_gain_vs_previous"
        ].notna()

    ].copy()


    if len(
        gain_rows
    ) > 0:


        plt.figure(

            figsize=(
                11,
                6
            )

        )


        for test_year in sorted(

            gain_rows[
                "Test_Year"
            ].unique()

        ):


            temp = gain_rows[

                gain_rows[
                    "Test_Year"
                ]

                ==

                test_year

            ].sort_values(
                "DOY"
            )


            plt.plot(

                temp[
                    "DOY"
                ],

                temp[
                    "Our_gain_vs_previous"
                ],

                marker="o",

                linewidth=2,

                label=
                    str(
                        test_year
                    )

            )


        plt.axhline(

            0,

            linestyle="--",

            linewidth=1

        )


        plt.xlabel(
            "Day of Year"
        )


        plt.ylabel(

            "Our Clean ICDL R² − Previous-year CDL R²"

        )


        plt.title(

            "Incremental Yield Value of Our Clean ICDL"

        )


        plt.grid(
            alpha=0.25
        )


        plt.legend(
            title="Test year"
        )


        plt.tight_layout()


        plt.show()


# ============================================================
# 20. JUNE-ONLY DIRECT COMPARISON
#
# This is the table we care about RIGHT NOW while the
# 2022 July/August exports are still queued.
# ============================================================

june_comparison_cell18 = (

    r2_table_cell18_icdl[

        r2_table_cell18_icdl[
            "DOY"
        ].isin(
            CHECKPOINT_DOYS[
                "June"
            ]
        )

    ]

    .copy()

)


print()

print(
    "=" * 105
)

print(
    "JUNE COMPARISON — 2022 VS 2023"
)

print(
    "=" * 105
)


display(

    june_comparison_cell18

    .round(
        4
    )

)


# ============================================================
# 21. SAVE RESULTS
# ============================================================

results_path_cell18 = os.path.join(

    OUTPUT_DIR_CELL18,

    "cell18_2022_2023_icdl_all_results.csv"

)


table_path_cell18 = os.path.join(

    OUTPUT_DIR_CELL18,

    "cell18_2022_2023_icdl_R2_table.csv"

)


june_path_cell18 = os.path.join(

    OUTPUT_DIR_CELL18,

    "cell18_JUNE_2022_vs_2023.csv"

)


results_cell18_icdl.to_csv(

    results_path_cell18,

    index=False

)


r2_table_cell18_icdl.to_csv(

    table_path_cell18,

    index=False

)


june_comparison_cell18.to_csv(

    june_path_cell18,

    index=False

)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print()

print(
    "=" * 105
)

print(
    "CELL 18 COMPLETE"
)

print(
    "=" * 105
)


print(
    "All results:"
)

print(
    results_path_cell18
)


print(
    "\nR² table:"
)

print(
    table_path_cell18
)


print(
    "\nJune comparison:"
)

print(
    june_path_cell18
)


print(
    "\nDetected test years:",
    sorted(
        results_cell18_icdl[
            "Test_Year"
        ].unique()
    )
)


print(
    """
WHEN NEW 2022 SCENARIO CSVs FINISH:

  place them under --input-dir and rerun this script.

The deployment comparison will automatically expand when the
new July/August CLEAN_REBUILT or STUDY files are present.
"""
)
