================================================================================
CORN YIELD PREDICTION USING SATELLITE REMOTE SENSING & MACHINE LEARNING
Project Readme & Script Documentation
================================================================================

OVERVIEW
--------
This repository provides an end-to-end data processing, feature engineering, and 
machine learning pipeline for predicting U.S. county-level corn yields using 
MODIS satellite vegetation indices (from Google Earth Engine) and USDA NASS 
historical yield data. 

The pipeline handles automated data downloading via Earth Engine/Drive APIs, 
data alignment and pivot transformations, temporal feature engineering, out-of-time 
walk-forward cross-validation using XGBoost, and visualization of model accuracy 
and feature importances across the growing season (DOY cutoffs).


FILE SUMMARY & MODULE DESCRIPTIONS
----------------------------------

1. download_data.py
   ----------------
   Purpose: 
   Automates the extraction of MODIS satellite imagery and cropland masks from 
   Google Earth Engine (GEE) and downloads the exported CSVs locally via Google Drive API.
   
   Key Functionalities:
   - Authenticates with Google Cloud, Google Earth Engine, and Google Drive APIs 
     using OAuth 2.0 (`client_secret.json` / `token.json`).
   - Defines study area (12 Midwestern US states) and time frame (2008–2024, March–October).
   - Computes 7 key vegetation/water indices from MODIS MOD13Q1 and MOD09A1:
     * NDVI (Normalized Difference Vegetation Index)
     * EVI_scaled & EVI2 (Enhanced Vegetation Index)
     * GCI (Green Chlorophyll Index)
     * NIRv (Near-Infrared Reflectance of Vegetation)
     * NDMI (Normalized Difference Moisture Index)
     * NDWI (Normalized Difference Water Index)
   - Applies Cropland Data Layer (CDL) land-cover masks (supports options: none, 
     same_year, lagged, 5-year frequency).
   - Reduces spatial raster pixels to county-level means (`reduceRegions`) using 
     TIGER county boundaries.
   - Submits batch export tasks to Google Drive and automatically downloads 
     completed CSV files into the `./downloaded_csvs` folder.


2. data_preprocessing.py
   ---------------------
   Purpose:
   Handles raw CSV/ZIP ingestion, spatial-temporal data pivoting, missing data 
   interpolation, USDA NASS yield data cleaning, and dataset coverage alignment.
   
   Key Functionalities:
   - Interactive GUI file selector (`tkinter`) to load GEE CSV/ZIP exports and USDA 
     NASS yield CSVs into local storage.
   - `build_gee_wide(csv_list)`: Converts long-format GEE daily vegetation index 
     observations into a unified wide format indexed by county (`GEOID`), `year`, 
     and Day of Year (`DOY`). Handles duplicate entries by mean aggregation.
   - Performs linear interpolation across time (DOY) per county to fill missing 
     satellite observation dates.
   - Filters and parses USDA NASS county-level corn yield statistics (grain yield in bu/acre) 
     and formats state/county FIPS codes to 5-digit `GEOID` strings.
   - Inner-joins satellite vegetation wide datasets with ground-truth yield data.
   - Includes data alignment diagnostics (`generate_combined_matrix`, 
     `generate_alignment_with_heatmap`) that measure state-by-state county coverage 
     and output alignment heatmap charts (`state_alignment_heatmap.png`) and 
     detailed alignment matrix CSVs (`detailed_alignment_matrix.csv`).


3. feature_building.py
   --------------------
   Purpose:
   Generates historical baselines, yield/vegetation anomaly metrics, and engineered 
   temporal features aggregated up to specified growing season cutoffs (DOY).
   
   Key Functionalities:
   - Calculates historical yield baseline: 5-year rolling mean yield per county (`hist_5yr`).
   - Computes historical yield anomalies (`yield_bu_acre - hist_5yr`).
   - Computes historical vegetation index averages per county and generates seasonal 
     vegetation anomalies (`<index>_DOY_<doy>_anom`).
   - `build_features(df_sub, doy_cut)`: Aggregates vegetation index time-series up to a 
     given Day of Year (`doy_cut`), extracting:
     * Summary statistics: mean, max, std, smoothed recent values (`last`), early-season mean.
     * Seasonal Biomass Proxy: Trapezoidal Area Under Curve (`auc`) using `scipy.integrate.trapezoid`.
     * Phenology Timing: DOY of peak vegetation index value (`peak_doy`).
     * Trend / Rate of Change: Recent trajectory slope (`slope`).
     * Anomaly statistics: mean anomaly, max anomaly, last anomaly, and anomaly slope.


4. train_test_split.py
   --------------------
   Purpose:
   Executes temporal out-of-time (walk-forward) train-test splitting and model fitting 
   to evaluate predictive power under realistic forecast settings.
   
   Key Functionalities:
   - `train_test_split(...)`: Performs expanding-window cross-validation for target 
     years (2018–2024). For each test year `yr`, trains XGBoost regressor strictly on 
     historical data (`year < yr`) and evaluates out-of-sample performance on `year == yr`.
   - Incorporates historical 5-year yield trends alongside satellite-derived temporal 
     features.
   - Trains `XGBRegressor` models and computes $R^2$ accuracy metrics.
   - `feature_importance(...)`: Records feature importance weights per fold, year, and 
     DOY cutoff.


5. plot_feature_importance.py
   --------------------------
   Purpose:
   Visualizes the temporal evolution of feature importances across different DOY 
   cutoffs in the growing season.
   
   Key Functionalities:
   - `plot_feature_importance_evolution(importance_compare_df, features_to_plot)`: 
     Filters feature importances for specified key indicators (e.g., `hist_5yr`, 
     `EVI2_mean`, `NIRv_mean`, `EVI_scaled_auc`).
   - Generates line charts tracking how feature importance shifts from early season 
     (historical yield dominated) to peak growing season (satellite index dominated).
   - Saves high-resolution visualization plots (`feature_importance_<dataset>.png`).


6. model_comparison_same_prev_cdl.py
   ---------------------------------
   Purpose:
   Main orchestration script for model training, within-season progression benchmark, 
   and evaluation across sequential DOY cutoffs.
   
   Key Functionalities:
   - Iterates through Day of Year cutoffs from DOY 65 (early Spring) to DOY 273 (Autumn) 
     at 16-day increments.
   - Calls `train_test_split.py` to evaluate model predictive skill ($R^2$) at each 
     in-season prediction milestone.
   - Highlights the peak growing season (DOY 145–225) on performance plots.
   - Saves performance curve plot (`same_year_cdl_comparison.png`).
   - Triggers feature importance trajectory plotting via `plot_feature_importance.py`.


PIPELINE WORKFLOW & DATA FLOW
-----------------------------
 1. DATA ACQUISITION (`download_data.py`)
    Earth Engine -> County Reduction -> Google Drive -> Local CSVs (`./downloaded_csvs`)

 2. PREPROCESSING & ALIGNMENT (`data_preprocessing.py`)
    Raw GEE CSVs + NASS Yield CSV -> Long-to-Wide Pivot -> Interpolation -> Inner Join

 3. FEATURE ENGINEERING (`feature_building.py`)
    Historical Baselines (5yr yield) + Seasonal Index Features + Trapezoidal AUC + Anomaly Scores

 4. MODEL TRAINING & EVALUATION (`train_test_split.py` & `model_comparison_same_prev_cdl.py`)
    Expanding Window Temporal CV (2018-2024) -> XGBoost Regression -> $R^2$ Tracking by DOY

 5. VISUALIZATION & ANALYSIS (`plot_feature_importance.py`)
    $R^2$ Progression Curves + Dynamic Feature Importance Evolution Plots


PREREQUISITES & DEPENDENCIES
----------------------------
Required Python packages:
- xgboost
- catboost
- scikit-learn
- pandas
- numpy
- scipy
- geopandas
- matplotlib
- seaborn
- earthengine-api
- google-api-python-client
- google-auth-oauthlib

To install dependencies locally:
  pip install xgboost catboost scikit-learn matplotlib seaborn geopandas pandas numpy scipy earthengine-api google-api-python-client google-auth-oauthlib


USAGE
-----
1. To pull fresh remote sensing data from Earth Engine:
   python download_data.py

2. To run the full model comparison, feature engineering, and plotting pipeline:
   python model_comparison_same_prev_cdl.py

================================================================================
