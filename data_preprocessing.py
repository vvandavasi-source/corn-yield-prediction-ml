"""
                This code sets up a local directory, allows the user to select 
                GEE ZIP or CSV files along with a yield CSV, extracts or copies 
                them to a local 'data' folder, and lists all the CSV files found in that folder. 
                It also imports necessary libraries for data processing and modeling. 
                Make sure to install the required libraries before running the code.
"""

# If you're running this in a local Python environment, uncomment the line below to install the necessary libraries.
#!python -m pip install xgboost catboost scikit-learn matplotlib seaborn geopandas pandas numpy scipy 

# If you're running this in a Jupyter notebook in colab, uncomment the line below to install the necessary libraries.
#!pip install xgboost catboost scikit-learn matplotlib seaborn geopandas pandas numpy scipy


#import necessary libraries
import re
import os, glob, zipfile, warnings, shutil #for file handling and warnings
import tkinter as tk #gui based file selector
from tkinter import filedialog #for file selection dialog
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
import seaborn as sns #for data visualization
from numpy import trapezoid #for numerical integration (AUC calculation)
from scipy import integrate #for numerical integration (AUC calculation)
#from google.colab import files #for file upload in Colab (uncomment if using Colab)
import matplotlib.pyplot as plt #for plotting
from sklearn.model_selection import LeaveOneGroupOut #for cross-validation
from sklearn.impute import SimpleImputer #for handling missing data
from sklearn.pipeline import Pipeline #for creating machine learning pipelines
from sklearn.metrics import r2_score, mean_squared_error #for evaluating model performance
from sklearn.ensemble import RandomForestRegressor #for random forest modeling
from xgboost import XGBRegressor #for XGBoost modeling
from catboost import CatBoostRegressor #for CatBoost modeling
from sklearn.svm import SVR #for Support Vector Regression modeling
from sklearn.preprocessing import StandardScaler #for feature scaling
import geopandas as gpd #for handling geospatial data
from matplotlib.cm import ScalarMappable #for creating color maps in plots
from matplotlib.colors import Normalize #for normalizing data for color mapping in plots
import matplotlib.cm as cm #for color maps in plots
import matplotlib.colors as mcolors #for color handling in plots
import plotly.express as px #for interactive plotting
from dash import Dash, dcc, html, callback, Input, Output, State, Patch


warnings.filterwarnings("ignore")


"""
SET UP LOCAL DATA DIRECTORY AND FILE SELECTION

I do not know if this section will work on colab. It was redesigned for local Python environments. If you're using colab, you may want to use the file upload feature instead (uncomment the line above and comment out the rest of this section).
if youre using a local Python environment, this will create a 'data' folder in the same directory where your script is running. You can select your GEE ZIP or CSV files + yield CSV, and they will be extracted or copied into this 'data' folder. After processing, it will list all the CSV files found in that folder.
this is designed to be user-friendly and should work on both Windows and Mac. Just make sure to select the correct files when prompted. The code will handle the rest, including extracting ZIP files and copying CSVs to the 'data' folder. After running this section, you can check the 'data' folder to see all your processed files ready for analysis.

LTR
"""
# This creates a 'data' folder in the same folder where your script is running
DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 2. SELECT FILES. Will bring up a window to select your GEE ZIP or CSV files + yield CSV. You can select multiple files at once.
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

print("Select your GEE ZIP or CSVs + yield CSV")
file_paths = filedialog.askopenfilenames(
    title="Select Data Files",
    filetypes=[("Data files", "*.csv *.zip"), ("All files", "*.*")] #define the file types you want to select.
)

if not file_paths:
    raise ValueError("No files selected")

# -----------------------------
# CLASSIFY AND PROCESS FILES
# -----------------------------
# We iterate directly through the paths you picked on your computer
for src_path in file_paths:
    filename = os.path.basename(src_path)

    # Handle ZIP files
    if filename.lower().endswith(".zip"):
        print(f"Extracting: {filename}")
        with zipfile.ZipFile(src_path, 'r') as z:
            z.extractall(DATA_DIR)

    # Handle CSV files
    elif filename.lower().endswith(".csv"):
        dst_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(dst_path):
            # We use copy2 to keep the original file where it was 
            # and put a copy in our data folder
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")
        else:
            print(f"Skipped (already exists): {filename}")

# -----------------------------
# FIND ALL CSV FILES
# -----------------------------
# Look inside our local DATA_DIR for everything extracted/copied
all_csvs = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)

print(f"\n Total CSV files found: {len(all_csvs)}")
for f in all_csvs[:10]:
    print(" -", f)

if not all_csvs:
    print(f"\n No files found in {DATA_DIR}. Please check the folder manually.")
    exit(1)



"""
    Use this version if you are working in google colab

print("Upload your GEE ZIP or CSVs + yield CSV")
uploaded = files.upload()

if len(uploaded) == 0:
    raise ValueError(" No files uploaded")

print("Uploaded files:", list(uploaded.keys()))

"""

"""
FIND SAME-YEAR VS LAGGED-CDL FILES


"""

same_year_csvs = [
    f for f in all_csvs
    if "Corn_MODIS" in os.path.basename(f)
    and "lag" not in os.path.basename(f).lower()
    and "prev" not in os.path.basename(f).lower()
]

lagged_csvs = [
    f for f in all_csvs
    if "Corn_MODIS" in os.path.basename(f)
    and (
        "lag" in os.path.basename(f).lower()
        or "prev" in os.path.basename(f).lower()
    )
]

print("==================================================")
print("SAME-YEAR FILES FOUND:", len(same_year_csvs))
print("==================================================")
for f in same_year_csvs[:10]:
    print(os.path.basename(f))


print("LAGGED-CDL FILES FOUND:", len(lagged_csvs))

for f in lagged_csvs[:10]:
    print(os.path.basename(f))

    """
    BUILD WIDE GEE FUNCTION
    
    """
veg_cols = [
        "NDVI_scaled",
        "EVI_scaled",
        "EVI2",
        "NIRv",
        "GCI",
        "NDWI"
    ]   
def build_gee_wide(csv_list):

    dfs = []
        # ✅ UPDATED INDICES
    veg_cols = [
        "NDVI",
        "EVI_scaled",
        "EVI2",
        "NIRv",
        "GCI",
        "NDMI",
        "NDWI"
    ]

    for f in csv_list:

        print("Loading:", os.path.basename(f))

        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]


        df = df.rename(columns=rename_map)

        needed = [
            "NAME",
            "GEOID",
            "STATEFP",
            "date",
            "year",
            "NDVI",
            "EVI_scaled",
            "EVI2",
            "NIRv",
            "GCI",
            "NDMI",
            "NDWI"
        ]

        if missing := [c for c in needed if c not in df.columns]: #this is a new syntax in Python 3.8+ that allows us to check for missing columns and store them in a variable at the same time. If there are any missing columns, we print a message and skip this file.

            if missing:
                print(f"Skipping {os.path.basename(f)}")
                print("Missing columns:", missing)
                continue

        df = df[needed].copy()

        df["GEOID"] = df["GEOID"].astype(str).str.zfill(5)
        df["STATEFP"] = pd.to_numeric(df["STATEFP"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["DOY"] = df["date"].dt.dayofyear

        for col in veg_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    gee_long = pd.concat(dfs, ignore_index=True)

    print("\nRaw long shape:", gee_long.shape)

    dup_mask = gee_long.duplicated(
        subset=["GEOID", "year", "DOY"],
        keep=False
    )

    print("Duplicate GEOID-year-DOY rows:", int(dup_mask.sum()))

    if dup_mask.sum() > 0:

        print("Collapsing duplicates...")

        gee_long = (
            gee_long
            .groupby(
                ["GEOID", "STATEFP", "year", "DOY"],
                as_index=False
            )[veg_cols]
            .mean()
        )

    print("Shape after duplicate handling:", gee_long.shape)

    print("\n===== GEE LONG SAMPLE =====")
    print(gee_long.head())

    wide_parts = []

    for col in veg_cols:

        temp = gee_long.pivot_table(
            index=["GEOID", "year"],
            columns="DOY",
            values=col,
            aggfunc="mean"
        )

        temp.columns = [f"{col}_DOY_{int(c)}" for c in temp.columns]

        wide_parts.append(temp)

    gee_wide = pd.concat(wide_parts, axis=1).reset_index()

    print("\nWide shape:", gee_wide.shape)

    print(
        "Unique county-years:",
        gee_wide[["GEOID", "year"]].drop_duplicates().shape[0]
    )

    print("\n===== GEE WIDE SAMPLE =====")
    print(gee_wide.head())

    return gee_wide

"""
BUILD SAME-YEAR AND LAGGED GEE TABLES
"""

same_year_gee_wide = build_gee_wide(same_year_csvs)
lagged_gee_wide = build_gee_wide(lagged_csvs)

print("\n==================================================")
print("FINAL DATASET SHAPES")
print("==================================================")
print("Same-year GEE shape:", same_year_gee_wide.shape)
print("Lagged-CDL GEE shape:", lagged_gee_wide.shape)



print("Same-year interpolation complete")
print("Shape:", same_year_gee_wide.shape)
print("Extracted Parameters:", same_year_gee_wide.head())

"""
INTERPOLATE SAME-YEAR DATA

"""

same_year_df = same_year_gee_wide.copy()

same_year_df = same_year_df.sort_values(
    ["GEOID", "year"]
).reset_index(drop=True)

same_feature_cols = [
    c for c in same_year_df.columns
    if c not in ["GEOID", "year"]
]

same_year_df[same_feature_cols] = (
    same_year_df
    .groupby("GEOID")[same_feature_cols]
    .transform(
        lambda x: x.interpolate(
            method="linear",
            limit_direction="both"
        )
    )
)

print("Same-year interpolation complete")
print("Shape:", same_year_df.shape)
print("Extracted Parameters:", same_year_df.head())

"""
# CELL 5 — INTERPOLATE LAGGED-CDL DATA
"""

lagged_cdl_df = lagged_gee_wide.copy()

lagged_cdl_df = lagged_cdl_df.sort_values(
    ["GEOID", "year"]
).reset_index(drop=True)

lagged_feature_cols = [
    c for c in lagged_cdl_df.columns
    if c not in ["GEOID", "year"]
]

lagged_cdl_df[lagged_feature_cols] = (
    lagged_cdl_df
    .groupby("GEOID")[lagged_feature_cols]
    .transform(
        lambda x: x.interpolate(
            method="linear",
            limit_direction="both"
        )
    )
)

print("lagged-year interpolation complete")
print("Shape:", lagged_cdl_df.shape)
print("Extracted Parameters:", lagged_cdl_df.head())
"""
# FIND YIELD FILE PATH
# Run this BEFORE the yield_df cell
"""

yield_candidates = [
    f for f in all_csvs
    if "Corn_MODIS" not in os.path.basename(f)
]

print("Possible yield files found:")
for f in yield_candidates:
    print(os.path.basename(f))

YIELD_PATH = yield_candidates[0]

print("\nUsing yield file:")
print(YIELD_PATH)
print("Lagged-CDL interpolation complete")
print("Shape:", lagged_cdl_df.shape)

"""
# FIX YIELD_DF NOT FOUND
# Run this BEFORE Cell 6 merge
"""

yield_raw = pd.read_csv(YIELD_PATH, low_memory=False)
yield_raw.columns = [c.strip() for c in yield_raw.columns]

print("Raw yield columns:", yield_raw.columns.tolist())

yield_df = yield_raw.copy()

yield_df = yield_df[
    (yield_df["Geo Level"] == "COUNTY") &
    (yield_df["Commodity"] == "CORN") &
    (yield_df["Data Item"] == "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE") &
    (yield_df["Period"] == "YEAR") &
    (yield_df["Domain"] == "TOTAL")
]

yield_df = yield_df[ 
    yield_df["State ANSI"].notna() &
    yield_df["County ANSI"].notna()
]

yield_df["STATEFP"] = pd.to_numeric(
    yield_df["State ANSI"],
    errors="coerce"
).astype(int)

yield_df["COUNTYFP"] = pd.to_numeric(
    yield_df["County ANSI"],
    errors="coerce"
).astype(int)

yield_df["year"] = pd.to_numeric(
    yield_df["Year"],
    errors="coerce"
).astype(int)

yield_df["yield_bu_acre"] = pd.to_numeric(
    yield_df["Value"],
    errors="raise"
)

yield_df["GEOID"] = (
    yield_df["STATEFP"].astype(str).str.zfill(2) +
    yield_df["COUNTYFP"].astype(str).str.zfill(3)
)

yield_df = yield_df[
    ["GEOID", "year", "yield_bu_acre"]
].drop_duplicates()

print("Yield dataframe shape:", yield_df.shape)
print("Yield years:", yield_df["year"].min(), "to", yield_df["year"].max())
print("Yield counties:", yield_df["GEOID"].nunique())
print("Yield states:", yield_df["State"].nunique())

"""
# CELL 6 — MERGE YIELD INTO BOTH DATASETS
# Replace your old model_df merge cell with this
"""

same_year_df = same_year_df.merge(
    yield_df[["GEOID", "year", "yield_bu_acre", "State"]],
    on=["GEOID", "year"],
    how="inner",
    validate="one_to_one"
)

lagged_cdl_df = lagged_cdl_df.merge(
    yield_df[["GEOID", "year", "yield_bu_acre", "State"]],
    on=["GEOID", "year"],
    how="inner",
    validate="one_to_one"
)

print("Same-year merged shape:", same_year_df.shape)
print("Lagged merged shape:", lagged_cdl_df.shape)
print("Merged datasets now contain yield data. Sample rows:")
print("\nSame-year sample:")
print(same_year_df.head())
print("\nLagged-CDL sample:")
print(lagged_cdl_df.head())

def merged_df():
    return same_year_df, lagged_cdl_df

veg_cols_long = list({col for col in same_year_df.columns if col in lagged_cdl_df.columns and any(v in col for v in [
    "NDVI_scaled",
    "EVI_scaled",
    "EVI2",
    "NIRv",
    "GCI",
    "NDWI"
])})

"""
ALIGNMENT MATRIX FOR GEE AND NASS DATA
work in progress - this section will create a matrix that shows which county-years have both GEE and yield data, which have only one or the other, and which have neither. This will help us understand the coverage of our datasets and identify any gaps. The matrix will be saved as a CSV file and also visualized as a heatmap showing the percentage of counties with both datasets available for each state and year.
"""

def clean_key(text):
    #Standardizes strings for matching across different datasets.
    if pd.isna(text): return "UNKNOWN"
    return re.sub(r'[.\s-]', '', str(text)).upper()


# 2. EXTRACTION & ALIGNMENT
def generate_combined_matrix(target_path, nass_file):
    # a. Load MODIS data from the subdirectory
    modis_csvs = glob.glob(os.path.join(target_path, "Corn_MODIS*.csv"))
    all_modis_list = []
    
    for f in modis_csvs:
        df = pd.read_csv(f)
        # Extract state name from the filename as a source label
        state_label = os.path.basename(f).split('_')[-1].replace('.csv', '').upper()
        df['state_src'] = state_label
        all_modis_list.append(df)
    
    modis_all = pd.concat(all_modis_list, ignore_index=True)
    modis_all['s_key'] = modis_all['state_src'].apply(clean_key)
    modis_all['c_key'] = modis_all['NAME'].apply(clean_key)
    
    # b. Load NASS data from its location
    nass_df = pd.read_csv(os.path.join(target_path, nass_file))
    nass_df = nass_df[nass_df['Geo Level'] == 'COUNTY'].copy()
    nass_df['s_key'] = nass_df['State'].apply(clean_key)
    nass_df['c_key'] = nass_df['County'].apply(clean_key)
    
    # c. Align Presence
    m_agg = modis_all.groupby(['s_key', 'c_key', 'year']).size().reset_index(name='m_in')
    n_agg = nass_df.groupby(['s_key', 'c_key', 'Year']).size().reset_index(name='n_in')
    n_agg.rename(columns={'Year': 'year'}, inplace=True)
    
    # d. Merge and Status Labeling
    alignment = pd.merge(m_agg, n_agg, on=['s_key', 'c_key', 'year'], how='outer').fillna(0)
    alignment['status'] = np.where((alignment['m_in'] > 0) & (alignment['n_in'] > 0), 3,
                          np.where(alignment['m_in'] > 0, 2, 1))
    
    # e. Create Matrix (Index: State_County, Columns: Year)
    alignment['id'] = alignment['s_key'] + "_" + alignment['c_key']
    matrix = alignment.pivot(index='id', columns='year', values='status').fillna(0)
    
    return matrix

# Execution - Process both same-year and lagged data
final_matrix_same_year = generate_combined_matrix(same_year_gee_wide).pivot(index='id', columns='year', values='status').fillna(0).reset_index()
final_matrix_same_year.to_csv('master_combined_alignment_same_year.csv')

final_matrix_lagged = generate_combined_matrix(lagged_gee_wide).pivot(index='id', columns='year', values='status').fillna(0).reset_index()
final_matrix_lagged.to_csv('master_combined_alignment_lagged.csv')

def generate_alignment_with_heatmap(target_dir, nass_file):
    # 1. Extract MODIS files from the subdirectory
    modis_csvs = glob.glob(os.path.join(target_dir, "Corn_MODIS_lagged*.csv"))
    
    all_modis_list = []
    
    for f in modis_csvs:
        df = pd.read_csv(f)
    
    # NEW LOGIC: Correctly captures "NORTH DAKOTA" from "Corn_MODIS_2008_2024_North_Dakota.csv"
        filename = os.path.basename(f)
        state_part = filename.split('lagged')[-1].replace('.csv', '')
        state_label = state_part.replace('_', ' ').upper()
    
        df['state_src'] = state_label
        all_modis_list.append(df)
    
    modis_all = pd.concat(all_modis_list, ignore_index=True)
    modis_all['s_key'] = modis_all['state_src'].apply(clean_key)
    modis_all['c_key'] = modis_all['NAME'].apply(clean_key)
    active_state_keys = modis_all['s_key'].unique()
    # 2. Load NASS data
    nass_df = pd.read_csv(os.path.join(target_dir, nass_file))
    nass_df = nass_df[nass_df['Geo Level'] == 'COUNTY'].copy()
    nass_df['s_key'] = nass_df['State'].apply(clean_key)
    nass_df['c_key'] = nass_df['County'].apply(clean_key)
    
    nass_filtered = nass_df[nass_df['s_key'].isin(active_state_keys)].copy()
    # 3. Aggregate and Align
    m_agg = modis_all.groupby(['s_key', 'c_key', 'year']).size().reset_index(name='m_in')
    n_agg = nass_filtered.groupby(['s_key', 'c_key', 'Year']).size().reset_index(name='n_in')
    n_agg.rename(columns={'Year': 'year'}, inplace=True)
    
    alignment = pd.merge(m_agg, n_agg, on=['s_key', 'c_key', 'year'], how='outer').fillna(0)
    
    # Status Logic: 3=Both (Green), 2=MODIS only, 1=NASS only
    alignment['status'] = np.where((alignment['m_in'] > 0) & (alignment['n_in'] > 0), 3,
                          np.where(alignment['m_in'] > 0, 2, 1))
    
    # 4. Generate State-Level Heatmap Data (% of counties aligned per state/year)
    state_summary = alignment.groupby(['s_key', 'year']).agg(
        pct_aligned=('status', lambda x: (x == 3).mean() * 100)
    ).reset_index()
    
    heatmap_matrix = state_summary.pivot(index='s_key', columns='year', values='pct_aligned').fillna(0)
    
    # 5. Output: Heatmap Visualization
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".0f", cmap="RdYlGn", cbar_kws={'label': '% Alignment'})
    plt.title("Study Area Data Alignment: Percentage of Counties with Both MODIS & NASS Data")
    plt.xlabel("Year")
    plt.ylabel("State")
    plt.tight_layout()
    plt.savefig('state_alignment_heatmap.png')
    
    # 6. Output: Detailed Alignment Matrix
    alignment['spatial_id'] = alignment['s_key'] + "_" + alignment['c_key']
    final_matrix = alignment.pivot(index='spatial_id', columns='year', values='status').fillna(0)
    final_matrix.to_csv('detailed_alignment_matrix.csv')
    
    print("Files 'state_alignment_heatmap.png' and 'detailed_alignment_matrix.csv' generated.")
    return heatmap_matrix

# Execute using pre-loaded GEE dataframes
heatmap_data_same_year = generate_alignment_with_heatmap(same_year_df)
heatmap_data_lagged = generate_alignment_with_heatmap(lagged_cdl_df)
print("Alignment matrices and heatmaps generated for both same-year and lagged datasets.")
print("Same-year heatmap data (state-level alignment percentages):")
print(heatmap_data_same_year)
print("Lagged heatmap data (state-level alignment percentages):")
print(heatmap_data_lagged)
    

