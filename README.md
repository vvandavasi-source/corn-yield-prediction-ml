# Timeliness of Crop-Mask Information in County-Level Corn-Yield Forecasting

Comparison of two crop-mask strategies for in-season, county-level corn-yield
prediction across the U.S. Corn Belt (12 states, test years 2022–2025):

- **Alternative A — Previous-season CDL:** mask corn pixels with the most recent
  *published* USDA Cropland Data Layer (prior year), the operational default.
- **Alternative B — In-season ICDL (CropSmart):** mask corn pixels with an
  in-season crop-type data layer generated during the current growing season.

Both feed an identical, frozen yield model (canonical compact-PRISM XGBoost);
only the crop mask changes, so any difference in skill is attributable to the
mask. The same-year *final* CDL is run through the same pipeline only as an
upper-bound reference. Evaluation uses RMSE / MAE / R² / bias and a
cluster-robust Davidson–MacKinnon (forecast-encompassing) J-test.

## Repository contents

> **Data is not included in this repository** (see *Data* below). Only code,
> the report, and figures are versioned here.

| File | Language | Purpose |
|---|---|---|
| `CY2_COMPACT_PRISM_CDL_ICDL_2022_2025_DUAL_PATCHED_v2.py` | Python | Main yield model + 2022–2025 dual-alternative deployment, J-test, and results. |
| `GEE_Corn_MODIS_FINAL_CDL_2024_2025.js` | GEE JavaScript | Earth Engine collector for same-year final-CDL-masked MODIS (2024–2025). |
| `GEE_Corn_MODIS_PREVIOUS_CDL_2024_2025.js` | GEE JavaScript | Earth Engine collector for previous-year-CDL-masked MODIS (2024–2025). |
| `GEE_SCRIPT_FOR_FINAL_CDL_COLLECTION.js` | GEE JavaScript | Original final-CDL MODIS collector (2023–2025). |
| `icdl_gee_modis_data_collection.js` | GEE JavaScript | In-season ICDL-masked MODIS collector. |
| `gee_prisms_data_collection.js` | GEE JavaScript | PRISM weather collector (2008–2025). |
| `check_scenario_inventory.py` | Python | Diagnostic: inventories scenario CSVs against the model's expected schema. |
| `make_figures.py` | Python | Reproduces the summary figures and headline table from the model's result CSVs. |
| `figures/` | — | Output figures used in the report. |
| `report/` | — | The written deliverable (DOCX / PDF). |

> **Note on Earth Engine scripts:** the `.js` files are Google Earth Engine
> Code Editor scripts (JavaScript). They run at
> [code.earthengine.google.com](https://code.earthengine.google.com), **not**
> locally. A valid Earth Engine account is required; the 2025 final-CDL export
> also requires a user-uploaded 2025 CDL image asset (set at the top of the script).

## How to run

1. **Generate the masked-MODIS scenario tables (Earth Engine).** Open each
   `.js` collector in the GEE Code Editor, set the year list / asset IDs at the
   top, run, and start the export tasks. Download the resulting CSVs.
2. **Assemble inputs locally.** Place scenario CSVs, PRISM, county yields, and
   boundaries in the input folders referenced at the top of the model script.
3. **(Optional) Sanity-check inputs:** `python check_scenario_inventory.py`.
4. **Run the model:** `python CY2_COMPACT_PRISM_CDL_ICDL_2022_2025_DUAL_PATCHED_v2.py`
   Outputs (metrics tables, J-test, figures) are written to the results folder.
5. **(Optional) Rebuild summary figures/table:**
   `python make_figures.py --results-dir <cell18 results folder>`

> **Paths are configured at the top of each script** (currently absolute Windows
> paths). Edit them to your environment before running.

## Data (not included)

The model consumes, but this repo does not redistribute:

- **MODIS** vegetation (`MOD13Q1`, `MOD09A1`) — via Google Earth Engine.
- **PRISM** weather (2008–2025) — via Google Earth Engine.
- **USDA NASS Cropland Data Layer (CDL)** — `USDA/NASS/CDL` in Earth Engine, or USDA CroplandCROS.
- **In-season ICDL (CropSmart)** — GMU CSISS, https://cloud.csiss.gmu.edu/CropSmart
- **County corn yields** — USDA NASS QuickStats.
- **County / state boundaries** — U.S. Census TIGER/Line.

## Requirements

Python 3.10+. Install with `pip install -r requirements.txt`.

## References

Key methodology reference for the in-season crop-type data layer (ICDL):
Li et al., *Automated 10-m Resolution In-season Crop-type Data Layer Mapping for
Contiguous United States*, Scientific Data 13, 750 (2026),
https://doi.org/10.1038/s41597-026-07099-1
Full reference list is in the report.

*Course project — BSE 509.*
