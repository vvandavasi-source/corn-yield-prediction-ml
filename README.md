# In-Season County-Level Corn Yield Forecasting System

This repository provides an end-to-end machine learning framework for predicting in-season U.S. county-level corn yields (bushels per acre) across the U.S. Corn Belt for evaluation years 2021 through 2025.

The system integrates Earth Observation imagery (MODIS MOD13Q1 / MOD09A1 via Google Earth Engine), daily PRISM climate reanalysis data, SoilGrids physical/chemical soil properties, historical NASS survey yield observations, and In-Season Crop Data Layer (ICDL) land-cover masks. Predictions are generated across six seasonal Day-of-Year (DOY) checkpoints using an XGBoost gradient boosting pipeline trained strictly on expanding historical same-year Cropland Data Layer (CDL) observations.

## File Inventory & Roles

| File Name | Environment | Description / Responsibility | 
| ----- | ----- | ----- | 
| `corn_yield_FINAL_2021_2025_200trees.py` | Python 3 | Core standalone training, feature extraction, cross-validation, and scenario evaluation engine using XGBoost (200 trees, seed 42). | 
| `generate_county_yield_dashboard.py` | Python 3 | Compiles multi-seed prediction results and feature importance summaries into a single-file interactive HTML dashboard (`county_yield_dashboard.html`). | 
| `gee_modis_icdl_2022_2025_FIXED_v4 (1).py` | GEE JavaScript | Google Earth Engine Code Editor script (packaged in `.py` extension) for extracting county-aggregated MODIS vegetation index time-series masked by monthly ICDLs across 12 Corn Belt states. | 
| `gee prisms data collection.py` | GEE JavaScript | Google Earth Engine script designed to extract leakage-free PRISM climate aggregates (16-day trailing windows) at the county level for the Corn Belt. |
| `corn_yield_with_2025_MICHIGAN_ADDED.csv` | CSV Data | Primary USDA NASS annual Survey yield dataset for corn grain (measured in bu/acre), updated with final 2025 Michigan county observations. | 

## System Architecture & Data Flow

```
+-----------------------------------+        +-----------------------------------+
|     Google Earth Engine (GEE)     |        |      PRISM & SoilGrids Data       |
| MODIS MOD13Q1/MOD09A1 + ICDL Mask |        | 15 Climate Vars + AWC100 + 20 CEC |
+-----------------+-----------------+        +-----------------+-----------------+
                  |                                            |
                  v                                            v
     +------------+--------------------------------------------+------------+
     |                                                                      |
     |           Model Pipeline: corn_yield_FINAL_2021_2025_200trees.py     |
     |                                                                      |
     |  1. Causal Linear Yield Trend Baseline (Expanding Prior-Year Window)  |
     |  2. Vegetation Index Interpolation & Anomaly Derivation              |
     |  3. XGBoost Model Fitting (Same-Year CDL Training: 200 Trees)        |
     |  4. Scenario Evaluations: PREVIOUS_CDL, FINAL_CDL, CLEAN_REBUILT, STUDY|
     |                                                                      |
     +------------------------------------+---------------------------------+
                                          |
                                          v
     +------------------------------------+---------------------------------+
     |                                                                      |
     |         Dashboard Generator: generate_county_yield_dashboard.py      |
     |                                                                      |
     |  1. Aggregates Multi-Seed County Predictions & Seed Variance         |
     |  2. Extracts Gain Share Feature Importances                          |
     |  3. Renders HTML Single-Page App with Plotly Choropleth & Diagnostics  |
     |                                                                      |
     +----------------------------------------------------------------------+

```

## Technical Specifications & Features

### 1. Data Features & Predictors

* **Historical Yield Baseline (`hist_yield_trend`)**: Expanding county-specific linear yield trend calculated using only strictly prior calendar year observations ($N \ge 3$ years required per county). Eliminates temporal leakage.

* **MODIS Vegetation Indices (7 Indices)**: NDVI, EVI (scaled), EVI2, NDMI, NDWI, NIRv, and GCI sampled across 16-day observation intervals (DOY 65–273).

* **Vegetation Metrics & Anomalies**: Temporal interpolation of missing values within observed bounds, moving window averages, seasonal sums, peak DOY, slopes, and historical expanding-window vegetation index anomalies (`*_anom`).

* **PRISM Climate Predictors (15 Variables)**: 16-day and season-to-date aggregates for minimum/mean/maximum temperature, precipitation, vapor pressure deficit (VPD min/mean/max), extreme heat degree-days ($>30^\circ\text{C}$), extreme hot days ($>35^\circ\text{C}$), and dry days. Extracted cleanly without forward-looking temporal leakage.

* **Soil Physical & Chemical Properties (21 Variables)**:

  * Available Water Capacity at 0–100 cm depth (`AWC100` from SoilGrids).

  * 20 depth-wise Cation Exchange Capacity layers (`CEC_*` from SoilGrids).

### 2. Machine Learning Model Configuration

* **Algorithm**: XGBoost Regressor (`xgboost.XGBRegressor`)

* **Trees (`n_estimators`)**: 200

* **Tree Depth (`max_depth`)**: 5

* **Learning Rate (`learning_rate`)**: 0.04

* **Subsample Ratio (`subsample`)**: 0.8

* **Feature Subsample (`colsample_bytree`)**: 0.8

* **Target Variable**: County yield anomaly relative to expanding historical linear trend ($\text{Yield}_{\text{observed}} - \text{Trend}_{\text{historical}}$).

### 3. Training & Evaluation Protocols

* **Training Protocol**: Single training mechanism using historical same-year Cropland Data Layer (`SAME_YEAR_CDL`) observations. Training data strictly precedes the evaluation year ($Y_{\text{train}} < Y_{\text{test}}$).

* **Evaluation Checkpoints**: Six seasonal Day-of-Year dates:

  * **June**: DOY 193, DOY 209

  * **July**: DOY 225, DOY 241

  * **August**: DOY 257, DOY 273

* **Evaluation Scenarios (Tested on Identical Matched Counties)**:

  * `PREVIOUS_CDL`: Baseline control utilizing the prior year's CDL land cover mask.

  * `FINAL_CDL`: Benchmark hindcast using same-year post-season final CDL mask.

  * `CLEAN_REBUILT`: In-house In-Season Crop Data Layer mask (2022–2025).

  * `STUDY`: Published literature In-Season Crop Data Layer mask (2021–2025).

## Data Input Requirements & Directory Layout

Place all source CSV and ZIP files within the designated input directory (`D:\corn_yield\model_inputs` by default).

```
model_inputs/
├── corn_yield_with_2025_MICHIGAN_ADDED.csv
├── CornBelt_SoilGrids_AWC.csv
├── CornBelt_SoilGrids_SOC_CEC.csv
├── CornBelt_PRISM_Weather_2008.csv ... CornBelt_PRISM_Weather_2025.csv
├── Corn_MODIS_same_year_2012_2025.csv
├── Corn_MODIS_2021_PREVIOUS_CDL.csv ... Corn_MODIS_2025_PREVIOUS_CDL.csv
├── Corn_MODIS_YYYY_CLEAN_REBUILT_JUNE.csv (JULY / AUGUST)
└── Corn_MODIS_YYYY_STUDY_JUNE.csv (JULY / AUGUST)

```

## Step-by-Step Execution Guide

### Step 1: Extract Satellite Features (Google Earth Engine)

1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/).

2. Copy the entire contents of `gee_modis_icdl_2022_2025_FIXED_v4 (1).py` and paste into the Code Editor script panel.

3. Configure target years in `VAR YEARS = [2022, 2023, 2024, 2025];` as required.

4. Click **Run**. Verify image asset access in the console.

5. In the **Tasks** tab on the right panel, click **Run** on generated tasks to export target feature CSV files directly to your Google Drive folder (`CornYield_ICDL_COMPLETED_PERIODS_CONFIRMED`).

### Step 2: Extract Weather Features (Google Earth Engine)

1. Return to the [Google Earth Engine Code Editor](https://code.earthengine.google.com/).
2. Copy the entire contents of `gee prisms data collection.py` and paste into the Code Editor script panel.
3. Configure the `START_YEAR` and `END_YEAR` variables to set the target extraction period (e.g., 2008 to 2025).
4. Click **Run** and trigger the generated yearly export tasks from the **Tasks** tab. The output CSVs will be saved to your Google Drive folder (`CornYield_PRISM_Weather_LEAKAGE_FREE`).

### Step 3: Execute In-Season Yield Model Pipeline

Run the main Python script to validate inputs, build causal trends, fit XGBoost models, and output evaluation metrics:

```
# Standard full model fitting and evaluation run (2021-2025)
python corn_yield_FINAL_2021_2025_200trees.py \
    --input-dir "D:/corn_yield/model_inputs" \
    --trees 200 \
    --seed 42

# Check input availability and timing metadata without model training
python corn_yield_FINAL_2021_2025_200trees.py \
    --input-dir "D:/corn_yield/model_inputs" \
    --check-inputs-only

# Regenerate figures and summaries from an existing run output directory without refitting
python corn_yield_FINAL_2021_2025_200trees.py \
    --output-dir "D:/corn_yield/yield_model_results/final_2021_2025_200trees_seed42_EXAMPLE" \
    --plots-only

```

### Step 4: Build Interactive Explorer Dashboard

To generate the interactive HTML dashboard summarizing multi-seed run outputs:

```
# Ensure model outputs exist under v5_out/seeds/ (or update path references in script)
python generate_county_yield_dashboard.py

```

The script will compile the prediction records and write `v5_out/summary/county_yield_dashboard.html`. Open this HTML file in any modern web browser to interactively filter yields, prediction errors, choropleth maps, and feature importances.

## Primary Output Files & Deliverables

Model execution outputs are saved under the auto-generated directory `yield_model_results/<tag>/`:

| Deliverable Path | Description | 
| ----- | ----- | 
| `model_comparison_metrics.csv` | Summary of test evaluation metrics ($R^2$, RMSE, MAE, Anomaly $R^2$, MSE skill vs. trend) per year, DOY, and scenario mask. | 
| `county_predictions.csv` | Full county-level observed yields, expanding linear trend values, predictions, and residual errors. | 
| `icdl_vs_previous_cdl_control.csv` | Paired county-level metric improvements comparing ICDLs directly against the `PREVIOUS_CDL` control. | 
| `corrected_study_vs_our.csv` | Comparative metrics between `STUDY` (published) and `CLEAN_REBUILT` (in-house) ICDLs. | 
| `training_audit.csv` | Complete ledger of historical training years, county counts, and training window metadata per fit. | 
| `figures/` | Seasonal performance comparison plots ($R^2$, RMSE, anomaly skill) per evaluation year. | 
| `feature_importance/` | Total gain feature importances and bar chart visualizer outputs per fit checkpoint. | 
| `summary/county_yield_dashboard.html` | Standalone interactive dashboard featuring US county choropleth maps, seasonal yield curves, state MAE/RMSE charts, and feature ranking diagnostics. | 

## Software Dependencies & Environment

* **Python Environment**: Python 3.9+

* **Required Python Libraries**:

  * `numpy`

  * `pandas >= 2.0.0`

  * `matplotlib`

  * `scikit-learn`

  * `xgboost`

* **Earth Engine Environment**: Google Earth Engine account with access to `MODIS/061/MOD13Q1` and `MODIS/061/MOD09A1` collections, active ICDL asset tables, and the `OREGONSTATE/PRISM/ANd` collection.

* **Browser Requirement for Dashboard**: Modern WebGL/HTML5-compliant browser (Chrome, Edge, Firefox, Safari) with active internet access for fetching U.S. county GeoJSON boundaries via Plotly CDN.