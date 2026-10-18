# Corn-Yield Crop-Mask Timing Study

**Does more timely crop-mask information improve county-level corn-yield forecasting?**
A U.S. Corn Belt study (12 states, ~1,055 counties, test years 2021–2025) comparing
yield forecasts built on three crop masks of differing timeliness.

*BSE 509 — Team "The Kernel Kollectors" (V. Vandavasi, L. Reeder, G. Tellez-Giron)*

---

## The question

A gradient-boosted (XGBoost) model predicts each county's corn-yield **anomaly**
(actual − county historical-trend yield); the trend is added back, and forecasts are
scored against that trend baseline (Murphy 1988 skill score). We hold the model fixed
and swap only the **crop mask** used to aggregate the MODIS vegetation signal to corn
pixels, then test whether a more *timely* mask produces better in-season forecasts:

| Scenario | Mask used | Timeliness |
|---|---|---|
| **A — `PREVIOUS_CDL`** | USDA CDL from the **prior** season | Operational today |
| **B — `STUDY`** | **In-season ICDL** (CropSmart), monthly | The alternative under test |
| **Oracle — `SAME_YEAR_CDL`** | Final **same-year** CDL | Upper bound (not available in-season) |
| `CLEAN_REBUILT` *(optional)* | Team's own 30 m in-season ICDL | Secondary |

Forecasts are made at in-season checkpoints (day-of-year 193, 209, 225, 241, 257, 273 —
June through August) for each test year.

---

## Pipeline & run order

```
GEE export ─► model (50 seeds) ─► aggregate_seeds ─► run_stat_tests ─► run_oat_sweep ─► plot_analysis
   (masks)        (forecasts)      (seed stability)   (county sig.)     (sensitivity)     (figures)
```

### 0. Build the crop-mask MODIS aggregates (Google Earth Engine)
`gee_study_icdl_2021_2025.js` masks MODIS to corn using the in-season CropSmart ICDL
assets and reduces to per-county values, one CSV per year/month. Paste it into the GEE
Code Editor, set the asset IDs / `YEARS` / `PUBLISHED_CORN_CODE`, run the tasks, and
download the `Corn_MODIS_<YEAR>_STUDY_<MONTH>.csv` outputs into your input folder.
(The previous-year and same-year CDL aggregates are built the same way with the CDL
exporter.)

### 1. Run the model — all years, 50 random seeds
`corn_yield_model_CY2_V5_MASK_COMPARISON.py` is the model. It takes one `--seed`; the
50-seed ensemble is produced by calling it 50 times into per-seed folders
(`run_50_seeds.ps1`). All runs share one `--work-dir`, so the staged inputs are reused
across seeds.

```powershell
# edit the three paths at the top first
./run_50_seeds.ps1
```
→ `v5_50_seed/seeds/seed_1 … seed_50/`, each with `model_comparison_metrics.csv`
and `county_predictions.csv`.

### 2. Seed-stability analysis
`aggregate_seeds.py` — mean ± SD per scenario and a paired **t-test across seeds**
(Feng et al. 2020), i.e. *is the gap robust to training randomness?*
```powershell
python aggregate_seeds.py --seeds-dir v5_50_seed/seeds --output-dir v5_50_seed/agg_out
```

### 3. County-level significance
`run_stat_tests.py` — on one run's `county_predictions.csv`: state-clustered paired
bootstrap on the RMSE/MAE gap, Wilcoxon signed-rank, Harvey-corrected Diebold-Mariano,
skill-vs-trend, and Mincer-Zarnowitz calibration, all Holm-corrected. *Is the gap real
given the sample of counties?*
```powershell
python run_stat_tests.py --pred-file v5_50_seed/seeds/seed_1/county_predictions.csv `
  --scenario-a PREVIOUS_CDL --scenario-b STUDY --output-dir v5_50_seed/stat_out
```

### 4. Sensitivity sweep (One-At-a-Time)
`run_oat_sweep.py` — drives the model across `--trees` levels (and optionally
`--small-cohort-threshold`), writing `sensitivity_oat.csv`. Random-seed sensitivity is
already covered by the 50-seed run.
```powershell
python run_oat_sweep.py --model corn_yield_model_CY2_V5_MASK_COMPARISON.py `
  --input-dir v5_data --work-dir oat_work --out-root oat_runs `
  --sweep trees --levels -25 -10 10 25 --scenario STUDY
```

### 5. Figures
`plot_analysis.py` — one module, ~22 figures (results, seed stability, significance,
OAT sensitivity, and an AHP pairwise/decision matrix). Point it at everything:
```powershell
python plot_analysis.py `
  --seed-summary v5_50_seed/agg_out/seed_summary_mean_sd.csv `
  --paired-tests v5_50_seed/agg_out/paired_tests_PREVIOUS_CDL_vs_STUDY.csv `
  --stat-tests   v5_50_seed/stat_out/stat_tests_PREVIOUS_CDL_vs_STUDY.csv `
  --seed-metrics v5_50_seed/seeds `
  --sensitivity  oat_runs/sensitivity_oat.csv `
  --outdir final_figs
```
`python plot_analysis.py --demo --outdir figs_demo` renders the whole gallery on
synthetic data, for previewing the figures before the real runs finish.

---

## Repository contents

| File | Role |
|---|---|
| `gee_study_icdl_2021_2025.js` | GEE: build in-season (STUDY) masked MODIS county aggregates |
| `corn_yield_model_CY2_V5_MASK_COMPARISON.py` | The model (one seed per run) |
| `run_50_seeds.ps1` | Driver: run the model across seeds 1–50 |
| `aggregate_seeds.py` | Seed-stability analysis (paired t across seeds) |
| `run_stat_tests.py` | County-level significance tests |
| `run_oat_sweep.py` | OAT sensitivity sweep → `sensitivity_oat.csv` |
| `plot_analysis.py` | All figures (+ `--demo` mode) |
| `interp_before_after.py` *(optional)* | FMEA proof: interpolation fix, before/after on real data |
| `requirements.txt` | Python dependencies |

**Not tracked** (see `.gitignore`): the input data (`v5_data/`) and all run outputs
(`v5_50_seed/`, `*_out/`, `oat_runs/`, figures). These are large and/or
licensed source data — see **Data** below.

---

## Data (not in the repo)

The model reads one input folder with these exact filename conventions:

```
Corn_MODIS_<YEAR>_PREVIOUS_CDL.csv          # prior-season CDL mask  (2008–2025)
Corn_MODIS_same_year_<YEAR>.csv             # same-year/final CDL    (2008–2025)
Corn_MODIS_<YEAR>_STUDY_<MONTH>.csv         # in-season ICDL  MONTH = JUNE|JULY|AUGUST
Corn_MODIS_<YEAR>_CLEAN_REBUILT_<MONTH>.csv # optional secondary mask
CornBelt_PRISM_Weather_<YEAR>.csv           # PRISM weather          (2008–2025)
CornBelt_SoilGrids_AWC.csv                  # SoilGrids available water capacity
CornBelt_SoilGrids_SOC_CEC.csv              # SoilGrids SOC / CEC
corn_yield_with_2025_county.csv             # USDA county corn yields
```

Each MODIS file is validated on load: the filename's year must match the `year` column
**and** the observation dates inside, so mislabeled or duplicated exports are rejected
rather than silently trained on.

- **STUDY (in-season ICDL)** masks come from CropSmart (GMU CSISS). 2021 & 2025 were
  exported with `gee_study_icdl_2021_2025.js`; 2022–2024 likewise.
- **Vegetation units**: the GEE exporters emit already-scaled indices (NDVI ∈ [−1, 1],
  etc.). Do **not** re-scale on load.

---

## Environment

Python 3.10+ with:

```
numpy
pandas
scipy
scikit-learn
xgboost
matplotlib
```

```bash
pip install -r requirements.txt
```
(The original runs used a conda env named `icdl-authors`.)

---

## Two analyses, two questions

The seed-level and county-level tests answer different things, and a difference that
survives **both** is robust:

- `aggregate_seeds.py` — holds counties fixed, varies the **training seed** → robust to
  model randomness?
- `run_stat_tests.py` — holds the model fixed, resamples **counties/states** → real
  given the spatial sample?

All multi-checkpoint p-values are Holm-corrected. The model predicts anomalies, so the
honest skill yardstick is **skill vs. the county trend baseline**, not raw R².
