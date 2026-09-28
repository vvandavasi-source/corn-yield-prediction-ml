#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CORN YIELD — V5 ANNUAL CDL MODEL COMPARISON (executable Python)

Annual input filenames (CSV; imagery/target year remains YYYY):
  Corn_MODIS_YYYY_PREVIOUS_CDL.csv   -> mask year YYYY-1
  Corn_MODIS_same_year_YYYY.csv      -> mask year YYYY
  Corn_MODIS_YYYY_FINAL_CDL.csv      -> accepted same-year alias
  Corn_MODIS_YYYY_CLEAN_REBUILT_JUNE/JULY/AUGUST.csv -> our ICDLs
  Corn_MODIS_YYYY_STUDY_JUNE/JULY/AUGUST.csv         -> published ICDLs
Use one current export per scenario. Different duplicates stop the run.
Older Corn_MODIS_lagged_STATE.csv inputs are ignored; supply annual previous
exports for the historical training years as well as the test years.

Models: previous-CDL training -> previous-CDL test; same-year-CDL training ->
our/published ICDL test. Controls: same-year training -> previous-CDL test and
same-year training -> final-CDL hindsight test. Two fitted models per year/DOY.

Preserves V4: XGBoost, causal county linear yield trend, seven vegetation
indices and historical anomalies, year, 15 PRISM features, AWC100, 20 CEC.
Training histories and test counties are matched between comparisons. All
model training years precede each test year. Test medians come only from
training. June/July/August masks require July/August/September availability.
Monthly-mask historical rows may be used at later eligible forecast dates.
PRISM remains retrospective; MODIS audits check declared completed periods.
Absent metadata is flagged, not certified. Actual release dates and ICDL
training-label provenance still require independent documentation.

Required other inputs: NASS county yield CSV; annual wide PRISM CSVs;
CornBelt_SoilGrids_AWC.csv; CornBelt_SoilGrids_SOC_CEC.csv (20 CEC_* columns).
Dependencies: numpy pandas matplotlib scikit-learn scipy xgboost.
Outputs: metrics/predictions, cohort/source/temporal audits, seasonal plots,
per-model/year/DOY and average total-gain feature-importance plots + CSV.
Importance is descriptive, not causal; correlated predictors can share gain.

Example (Windows Anaconda Prompt):
python corn_yield_model_CY2_V5_MASK_COMPARISON.py --input-dir "D:/corn_yield/model_inputs" --work-dir "D:/corn_yield/yield_model_work_v5" --output-dir "D:/corn_yield/yield_model_results/v5_mask_comparison"
"""

import argparse
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import os
import glob
import zipfile
import shutil
import warnings
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from scipy import stats
try:
    from IPython.display import display
except Exception:

    def display(obj):
        print(obj)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
pd.set_option('display.max_columns', 250)
pd.set_option('display.width', 240)
MODEL_VERSION = 'CY2_V5_ANNUAL_CDL_MATCHED_MODELS'
VALIDATION_START_YEAR = 2021
VALIDATION_END_YEAR = 2025
VALIDATION_TAG = f'{VALIDATION_START_YEAR}_{VALIDATION_END_YEAR}'
XGB_PARAMS = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'reg:squarederror', 'tree_method': 'hist', 'n_jobs': -1}

def parse_args():
    parser = argparse.ArgumentParser(description='Matched previous-CDL and same-year-CDL training experiments.')
    parser.add_argument('--input-dir', default=r'D:\corn_yield\model_inputs')
    parser.add_argument('--work-dir', default=r'D:\corn_yield\yield_model_work_v5')
    parser.add_argument('--output-dir', default=r'D:\corn_yield\yield_model_results\v5_mask_comparison')
    parser.add_argument('--test-years', nargs='+', type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--doys', nargs='+', type=int, default=[193, 209, 225, 241, 257, 273])
    parser.add_argument('--trees', type=int, default=500, help='Keep 500 for production; smaller values are for smoke tests.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=-1)
    return parser.parse_args()

ARGS = parse_args()
INPUT_DIR = os.path.abspath(ARGS.input_dir)
WORK_DIR = os.path.abspath(ARGS.work_dir)
OUTPUT_DIR_BASE = os.path.abspath(ARGS.output_dir)
print('✓ imports ready')
print('Input directory :', INPUT_DIR)
print('Work directory  :', WORK_DIR)
print('Output directory:', OUTPUT_DIR_BASE)
print('Model version    :', MODEL_VERSION)
print('Feature policy   : vegetation + veg anomalies + causal county yield trend + year + 15 compact PRISM + AWC100 + 20 CEC')
print(f'Validation window: {VALIDATION_START_YEAR}–{VALIDATION_END_YEAR}')
print('Historical-yield baseline: expanding county linear trend using prior observed yields only.')
if not os.path.isdir(INPUT_DIR):
    raise FileNotFoundError(f'Input directory does not exist:\n{INPUT_DIR}')
# Avoid deleting staged data when users accidentally point work/output at their inputs.
for generated in (Path(WORK_DIR).resolve(), Path(OUTPUT_DIR_BASE).resolve()):
    root = Path(INPUT_DIR).resolve()
    if generated == root or generated in root.parents:
        raise ValueError('Work/output must not equal or contain the input directory.')
DATA_DIR = os.path.join(WORK_DIR, 'data')
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
for pattern in ('*.csv', '*.CSV'):
    for path in input_root.rglob(pattern):
        if path.is_file() and _outside_generated_dirs(path):
            source_csvs.append(path)
for pattern in ('*.zip', '*.ZIP'):
    for path in input_root.rglob(pattern):
        if path.is_file() and _outside_generated_dirs(path):
            source_zips.append(path)
source_csvs = sorted(set(source_csvs))
source_zips = sorted(set(source_zips))
print('\nSource CSV files:', len(source_csvs))
print('Source ZIP files:', len(source_zips))
if len(source_csvs) == 0 and len(source_zips) == 0:
    raise FileNotFoundError('No CSV or ZIP inputs were found under:\n' + INPUT_DIR)
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
for src in source_csvs:
    rel = src.relative_to(Path(INPUT_DIR))
    dst = Path(DATA_DIR) / 'direct_csv' / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
for n, src in enumerate(sorted(source_zips), start=1):
    safe_name = re.sub('[^A-Za-z0-9._-]+', '_', src.stem)
    extract_dir = Path(DATA_DIR) / 'zip_extract' / f'{n:03d}_{safe_name}'
    extract_dir.mkdir(parents=True, exist_ok=True)
    print('Extracting:', src.name)
    with zipfile.ZipFile(src, 'r') as z:
        z.extractall(extract_dir)
all_csvs = sorted(str(p) for p in Path(DATA_DIR).rglob('*') if p.is_file() and p.suffix.lower() == '.csv')
print('\nTotal staged CSV files:', len(all_csvs))
for f in sorted(all_csvs)[:50]:
    print(' ', os.path.basename(f))
same_year_csvs = []
final_cdl_historical_candidates = []
prism_csvs = []
yield_candidates = []
for f in all_csvs:
    name = os.path.basename(f)
    lower = name.lower()
    if 'corn_modis_same_year' in lower:
        same_year_csvs.append(f)
        continue
    if re.match('^corn_modis_20\\d{2}_final_cdl(?:\\s*\\(\\d+\\))?\\.csv$', lower, flags=re.IGNORECASE):
        final_cdl_historical_candidates.append(f)
        continue
    if 'prism' in lower or 'cornbelt_prism_weather' in lower:
        prism_csvs.append(f)
        continue
    if 'yield' in lower and 'modis' not in lower and ('prism' not in lower) and ('weather' not in lower) and ('fixed_data' not in lower):
        yield_candidates.append(f)
same_year_csvs = sorted(same_year_csvs)
final_cdl_historical_candidates = sorted(final_cdl_historical_candidates)
prism_csvs = sorted(prism_csvs)
yield_candidates = sorted(yield_candidates)

def _normalized_download_name(path):
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    stem = re.sub('\\s*\\(\\d+\\)$', '', stem)
    return stem.lower() + ext.lower()

def _dedupe_file_list(paths):
    groups = {}
    for path in paths:
        groups.setdefault(_normalized_download_name(path), []).append(path)
    chosen = []
    for copies in groups.values():
        chosen.append(max(copies, key=os.path.getmtime))
    return sorted(chosen)
same_year_csvs = _dedupe_file_list(same_year_csvs)
final_cdl_historical_candidates = _dedupe_file_list(final_cdl_historical_candidates)
prism_csvs = _dedupe_file_list(prism_csvs)
yield_candidates = _dedupe_file_list(yield_candidates)
print('\n' + '=' * 75)
print('CANONICAL MODEL INPUTS')
print('=' * 75)
print('Same-year historical MODIS:', len(same_year_csvs))
print('Historical FINAL_CDL fallback candidates:', len(final_cdl_historical_candidates))
print('PRISM weather files:', len(prism_csvs))
print('Yield candidates:', len(yield_candidates))
print('=' * 75)
print('\nSame-year files:')
for f in same_year_csvs:
    print(' ', os.path.basename(f))
print('\nFINAL_CDL fallback candidates:')
for f in final_cdl_historical_candidates:
    print(' ', os.path.basename(f))
print('\nPRISM files:')
for f in prism_csvs:
    print(' ', os.path.basename(f))
VEG_INDICES = ['NDVI', 'EVI_scaled', 'EVI2', 'NDMI', 'NDWI', 'NIRv', 'GCI']

def clean_fips(series):
    return series.astype(str).str.replace('\\.0$', '', regex=True).str.zfill(5)

def build_gee_wide(csv_list, label='GEE', force_year=None):
    dfs = []
    if len(csv_list) == 0:
        print(f'⚠ No files for {label}')
        return pd.DataFrame()
    for f in csv_list:
        print('Loading:', os.path.basename(f))
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        cropsmart_aliases = {'CS_NDVI': 'NDVI', 'CS_EVI': 'EVI_scaled', 'CS_EVI2': 'EVI2', 'CS_NDMI': 'NDMI', 'CS_NDWI': 'NDWI', 'CS_NIRv': 'NIRv', 'CS_GCI': 'GCI'}
        rename_map = {src_col: dst_col for src_col, dst_col in cropsmart_aliases.items() if src_col in df.columns and dst_col not in df.columns}
        if rename_map:
            print('  Normalizing CropSmart columns:', ', '.join((f'{src_col}->{dst_col}' for src_col, dst_col in rename_map.items())))
            df = df.rename(columns=rename_map)
        if 'GEOID' not in df.columns:
            if 'FIPS' in df.columns:
                df = df.rename(columns={'FIPS': 'GEOID'})
            else:
                print('Skipping — no GEOID/FIPS:', os.path.basename(f))
                continue
        if 'year' not in df.columns:
            if force_year is not None:
                df['year'] = force_year
            else:
                raise KeyError(f'{os.path.basename(f)} has no year.')
        needed = ['GEOID', 'date', 'year'] + VEG_INDICES
        missing = [c for c in needed if c not in df.columns]
        if len(missing) > 0:
            print('Skipping:', os.path.basename(f))
            print('Missing:', missing)
            continue
        df = df[needed].copy()
        df['GEOID'] = clean_fips(df['GEOID'])
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['DOY'] = df['date'].dt.dayofyear
        for col in VEG_INDICES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        ndvi_valid = df['NDVI'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ndvi_valid) > 0:
            q95 = ndvi_valid.abs().quantile(0.95)
            if q95 > 2:
                print('  Scaling NDVI × 0.0001')
                df['NDVI'] = df['NDVI'] * 0.0001
        df.loc[~df['NDVI'].between(-1, 1), 'NDVI'] = np.nan
        df = df[df['year'].notna() & df['DOY'].notna()].copy()
        df['year'] = df['year'].astype(int)
        df['DOY'] = df['DOY'].astype(int)
        dfs.append(df)
    if len(dfs) == 0:
        raise RuntimeError(f'No usable {label} data.')
    gee_long = pd.concat(dfs, ignore_index=True)
    print('\nRaw long shape:', gee_long.shape)
    duplicates = gee_long.duplicated(subset=['GEOID', 'year', 'DOY'], keep=False)
    print('Duplicate GEOID-year-DOY rows:', int(duplicates.sum()))
    if duplicates.any():
        duplicate_examples = gee_long.loc[duplicates, ['GEOID', 'year', 'DOY']].drop_duplicates().head(20)
        raise ValueError(f'{label}: duplicate GEOID-year-DOY observations detected. The audited workflow will NOT silently average overlapping MODIS inputs. This usually means both an annual whole-Corn-Belt export and overlapping state-level files are present, or duplicate source products were staged. First duplicate keys:\n{duplicate_examples.to_string(index=False)}')
    wide_parts = []
    for col in VEG_INDICES:
        temp = gee_long.pivot_table(index=['GEOID', 'year'], columns='DOY', values=col, aggfunc='mean')
        temp.columns = [f'{col}_DOY_{int(d)}' for d in temp.columns]
        wide_parts.append(temp)
    gee_wide = pd.concat(wide_parts, axis=1).reset_index().sort_values(['GEOID', 'year']).reset_index(drop=True)
    print(f'\n{label} wide shape:', gee_wide.shape)
    print('County-years:', gee_wide[['GEOID', 'year']].drop_duplicates().shape[0])
    ndvi_cols = [c for c in gee_wide.columns if c.startswith('NDVI_DOY_')]
    if len(ndvi_cols) > 0:
        print('NDVI range:', np.nanmin(gee_wide[ndvi_cols].values), 'to', np.nanmax(gee_wide[ndvi_cols].values))
    return gee_wide
# New annual exports are authoritative; old state-level exports are not mixed in.
# Do not move the imagery/target year back: only the mask is from year y-1.
import hashlib
import json

INPUT_AUDIT = []
PERIOD_SIGNATURES = {}
MASK_AVAILABLE = {}


def export_name(path):
    stem = Path(path).stem
    stem = re.sub(r'^Copy of\s+', '', stem, flags=re.I)
    stem = re.sub(r'\s*\(\d+\)$', '', stem)
    return stem.upper()


def parse_export(path):
    name = export_name(path)
    match = re.fullmatch(r'CORN_MODIS_SAME_YEAR_(20\d{2})', name)
    if match:
        return int(match[1]), 'FINAL_CDL', None
    match = re.fullmatch(r'CORN_MODIS_(20\d{2})_(PREVIOUS_CDL|FINAL_CDL)', name)
    if match:
        return int(match[1]), match[2], None
    match = re.fullmatch(r'CORN_MODIS_(20\d{2})_(CLEAN_REBUILT|STUDY|OUR)_(JUNE|JULY|AUGUST)', name)
    if match:
        return int(match[1]), 'CLEAN_REBUILT' if match[2] == 'OUR' else match[2], match[3]
    return None


def unique_export_files(paths):
    """Never silently choose between different exports with the same scenario key."""
    result = {}
    for path in sorted(paths):
        key = parse_export(path)
        if key is None:
            continue
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if key in result:
            old = result[key]
            if hashlib.sha256(Path(old).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Conflicting exports for {key}:\n{old}\n{path}\n'
                                 'Keep only the intended current export in the input folder.')
            print('Ignoring identical duplicate:', path)
        else:
            result[key] = path
    return result


def audit_modis(path, key):
    """Validate supplied timing metadata; missing metadata is never called verified."""
    data = pd.read_csv(path, low_memory=False)
    data.columns = data.columns.str.strip()
    year, family, month = key
    if not {'date', 'year'}.issubset(data.columns):
        raise ValueError(f'{path}: expected date and year columns from the GEE exporter.')
    dates = pd.to_datetime(data['date'], errors='coerce', utc=True).dt.tz_convert(None)
    years = pd.to_numeric(data['year'], errors='coerce')
    if dates.isna().any() or not (years.eq(year) & dates.dt.year.eq(year)).all():
        raise ValueError(f'{path}: filename, year column and observation dates disagree.')
    if 'GEOID' not in data and 'FIPS' not in data:
        raise ValueError(f'{path}: missing GEOID/FIPS county column.')
    required_veg = {'NDVI': 'CS_NDVI', 'EVI_scaled': 'CS_EVI', 'EVI2': 'CS_EVI2',
                    'NDMI': 'CS_NDMI', 'NDWI': 'CS_NDWI', 'NIRv': 'CS_NIRv', 'GCI': 'CS_GCI'}
    missing = [v for v, alias in required_veg.items() if v not in data and alias not in data]
    if missing:
        raise ValueError(f'{path}: missing vegetation columns: {missing}')
    veg_columns = [c for c in VEG_INDICES if c in data]
    veg_columns += [c for c in required_veg.values() if c in data]
    usable = data[veg_columns].apply(pd.to_numeric, errors='coerce').notna().any(axis=1)
    cutoff = dates + pd.Timedelta(days=1)
    if 'forecast_doy' in data:
        if not pd.to_numeric(data.forecast_doy, errors='coerce').eq(dates.dt.dayofyear).all():
            raise ValueError(f'{path}: forecast_doy does not match date.')
    if 'forecast_end_exclusive' in data:
        end = pd.to_datetime(data.forecast_end_exclusive, errors='coerce')
        if not end.eq(cutoff).all():
            raise ValueError(f'{path}: forecast boundary does not match date + 1 day.')
    required = {'temporal_policy', 'period_start', 'period_end_exclusive',
                'latest_observation_date', 'assumed_release_lag_days', 'complete_periods_present'}
    verified = required.issubset(data.columns)
    policy = 'UNVERIFIED'
    if verified:
        policies = data.temporal_policy.dropna().astype(str).unique()
        if len(policies) != 1 or not data.temporal_policy.notna().all():
            raise ValueError(f'{path}: mixed or missing temporal policy.')
        policy = policies[0]
        start = pd.to_datetime(data.period_start, errors='coerce')
        end = pd.to_datetime(data.period_end_exclusive, errors='coerce')
        latest = pd.to_datetime(data.latest_observation_date, errors='coerce')
        lag = pd.to_numeric(data.assumed_release_lag_days, errors='coerce')
        complete = data.complete_periods_present.astype(str).str.lower().isin(['true', '1', '1.0'])
        invalid = (start.isna() | end.isna() | latest.isna() | lag.isna() | (lag < 0)
                   | (start >= end) | (latest >= end) | (latest < start)
                   | (latest > dates) | (end > cutoff)
                   | ((end + pd.to_timedelta(lag, unit='D')) > cutoff) | ~complete)
        if (invalid & usable).any():
            raise ValueError(f'{path}: nonempty vegetation violates completed-period timing metadata.')
        # Each year/checkpoint must represent the same source period in every mask export.
        valid_meta = data.loc[usable, ['period_start', 'period_end_exclusive', 'assumed_release_lag_days']].copy()
        valid_meta['doy'] = dates.loc[usable].dt.dayofyear
        for doy, group in valid_meta.groupby('doy'):
            signatures = group.drop(columns='doy').astype(str).drop_duplicates()
            if len(signatures) != 1:
                raise ValueError(f'{path}: inconsistent source windows at DOY {doy}.')
            signature = (policy, str(signatures.iloc[0, 0]), str(signatures.iloc[0, 1]), float(signatures.iloc[0, 2]))
            lookup = (year, int(doy))
            if lookup in PERIOD_SIGNATURES and PERIOD_SIGNATURES[lookup] != signature:
                raise ValueError(f'{path}: source periods differ across masks at {lookup}. Re-export consistently.')
            PERIOD_SIGNATURES[lookup] = signature
    if 'mask_year' in data and family in ('FINAL_CDL', 'PREVIOUS_CDL'):
        expected = year - (family == 'PREVIOUS_CDL')
        if not pd.to_numeric(data.mask_year, errors='coerce').eq(expected).all():
            raise ValueError(f'{path}: mask_year must be {expected}.')
    available = None
    if 'mask_available_after' in data and family != 'FINAL_CDL':
        values = data.mask_available_after.dropna().astype(str)
        parsed = pd.to_datetime(values, errors='coerce')
        if parsed.isna().any():
            raise ValueError(f'{path}: unrecognized operational mask availability date.')
        if len(parsed):
            available = parsed.max()
    MASK_AVAILABLE[key] = available
    INPUT_AUDIT.append(dict(path=path, year=year, family=family, month=month,
                            rows=len(data), counties=data.get('GEOID', data.get('FIPS')).nunique(),
                            temporal_policy=policy, timing_metadata_checked=verified,
                            mask_available_after=str(available),
                            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))


EXPORT_FILES = unique_export_files(all_csvs)
same_year_csvs = [p for (y, f, m), p in EXPORT_FILES.items() if f == 'FINAL_CDL']
previous_csvs = [p for (y, f, m), p in EXPORT_FILES.items() if f == 'PREVIOUS_CDL']
final_cdl_historical_candidates = []
if not same_year_csvs or not previous_csvs:
    raise FileNotFoundError('Need annual Corn_MODIS_same_year_YYYY.csv and '
                            'Corn_MODIS_YYYY_PREVIOUS_CDL.csv, including historical training years.')
for key, path in EXPORT_FILES.items():
    audit_modis(path, key)
policies = {r['temporal_policy'] for r in INPUT_AUDIT if r['timing_metadata_checked']}
if len(policies) > 1:
    raise ValueError(f'Mixed MODIS temporal policies: {policies}. Re-export consistently.')
TIMING_STATUS = ('METADATA_CHECKED' if all(r['timing_metadata_checked'] for r in INPUT_AUDIT)
                 else 'MISSING_TIMING_METADATA')
if TIMING_STATUS != 'METADATA_CHECKED':
    print('WARNING: Some inputs have no complete timing metadata. Results are timing-unverified.')
pd.DataFrame(INPUT_AUDIT).to_csv(Path(OUTPUT_DIR_BASE) / 'input_modis_audit.csv', index=False)
previous_gee_wide = build_gee_wide(previous_csvs, label='PREVIOUS-YEAR CDL ANNUAL HISTORY')

same_year_gee_wide = build_gee_wide(same_year_csvs, label='SAME-YEAR FINAL CDL')
if same_year_gee_wide.empty:
    raise RuntimeError('No usable SAME-YEAR historical vegetation data were built.')
canonical_hist_years = set(pd.to_numeric(same_year_gee_wide['year'], errors='coerce').dropna().astype(int).unique())
fallback_wide_parts = []
for fallback_path in final_cdl_historical_candidates:
    fallback_name = os.path.basename(fallback_path)
    match = re.search('corn_modis_(20\\d{2})_final_cdl', fallback_name, flags=re.IGNORECASE)
    if match is None:
        continue
    fallback_year = int(match.group(1))
    if fallback_year in canonical_hist_years:
        print(f'Skipping FINAL_CDL historical fallback for {fallback_year}: canonical same-year vegetation already covers that year.')
        continue
    print(f'Using FINAL_CDL scenario export as historical same-year fallback for {fallback_year}: {fallback_name}')
    fallback_wide = build_gee_wide([fallback_path], label=f'FINAL_CDL HISTORICAL FALLBACK {fallback_year}', force_year=fallback_year)
    fallback_wide_parts.append(fallback_wide)
    canonical_hist_years.add(fallback_year)
if fallback_wide_parts:
    same_year_gee_wide = pd.concat([same_year_gee_wide] + fallback_wide_parts, ignore_index=True, sort=False)
    duplicate_hist = same_year_gee_wide.duplicated(subset=['GEOID', 'year'], keep=False)
    if duplicate_hist.any():
        examples = same_year_gee_wide.loc[duplicate_hist, ['GEOID', 'year']].drop_duplicates().head(20)
        raise ValueError(f'Historical vegetation contains duplicate county-years after FINAL_CDL fallback. Examples:\n{examples.to_string(index=False)}')
    same_year_gee_wide = same_year_gee_wide.sort_values(['GEOID', 'year']).reset_index(drop=True)
print('\n' + '=' * 75)
print('CANONICAL HISTORICAL VEGETATION')
print('=' * 75)
print('Same-year shape:', same_year_gee_wide.shape)
if len(yield_candidates) == 0:
    raise FileNotFoundError('No yield CSV found.')
yield_file = max(yield_candidates, key=os.path.getsize)
print('Yield file:')
print(yield_file)
yield_raw = pd.read_csv(yield_file, low_memory=False)
yield_raw.columns = [c.strip() for c in yield_raw.columns]
yield_df = yield_raw.copy()
yield_df = yield_df[(yield_df['Geo Level'] == 'COUNTY') & (yield_df['Commodity'] == 'CORN') & (yield_df['Data Item'] == 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE') & (yield_df['Period'] == 'YEAR') & (yield_df['Domain'] == 'TOTAL')].copy()
yield_df = yield_df[yield_df['State ANSI'].notna() & yield_df['County ANSI'].notna()].copy()
if 'County' in yield_df:
    yield_df = yield_df[~yield_df['County'].astype(str).str.contains(r'OTHER\s+COUNTIES', case=False, regex=True)].copy()
state_ansi = pd.to_numeric(yield_df['State ANSI'], errors='coerce')
county_ansi = pd.to_numeric(yield_df['County ANSI'], errors='coerce')
yield_df['year'] = pd.to_numeric(yield_df['Year'], errors='coerce')
yield_df['yield_bu_acre'] = pd.to_numeric(yield_df['Value'].astype(str).str.replace(',', '', regex=False), errors='coerce')
valid = state_ansi.notna() & county_ansi.notna() & yield_df['year'].notna() & yield_df['yield_bu_acre'].notna()
yield_df = yield_df.loc[valid].copy()
yield_df['FIPS'] = state_ansi.loc[valid].astype(int).astype(str).str.zfill(2) + county_ansi.loc[valid].astype(int).astype(str).str.zfill(3)
yield_df['year'] = yield_df['year'].astype(int)
conflicting_yields = yield_df.groupby(['FIPS', 'year'])['yield_bu_acre'].nunique()
if (conflicting_yields > 1).any():
    raise ValueError('Conflicting observed yields for the same county-year; resolve the NASS input duplicates.')
yield_clean_df = yield_df[['FIPS', 'year', 'yield_bu_acre']].drop_duplicates(subset=['FIPS', 'year']).sort_values(['FIPS', 'year']).reset_index(drop=True)
print('Yield shape:', yield_clean_df.shape)
print('Years:', yield_clean_df['year'].min(), 'to', yield_clean_df['year'].max())
print('Counties:', yield_clean_df['FIPS'].nunique())
TREND_MIN_OBS = 3
_yield_min_year = int(yield_clean_df['year'].min())
_yield_max_year = int(yield_clean_df['year'].max())
_yield_fips = sorted(yield_clean_df['FIPS'].unique())
_yield_full_index = pd.MultiIndex.from_product([_yield_fips, range(_yield_min_year, _yield_max_year + 1)], names=['FIPS', 'year'])
yield_features = yield_clean_df.set_index(['FIPS', 'year']).reindex(_yield_full_index).reset_index().sort_values(['FIPS', 'year']).reset_index(drop=True)

def _build_causal_county_yield_trend(group):
    """
    Build one leakage-safe expected-yield trend value for every calendar year.

    Each row uses only observed yields from STRICTLY EARLIER years in the same
    county.  The regression expands through time; later rows may use more
    history than earlier rows.

    Returns:
        hist_yield_trend
        hist_trend_n_obs
        hist_trend_slope
        hist_trend_intercept
        hist_trend_first_year
        hist_trend_last_year
    """
    g = group.sort_values('year').copy()
    years = pd.to_numeric(g['year'], errors='coerce').to_numpy(dtype=float)
    yields = pd.to_numeric(g['yield_bu_acre'], errors='coerce').to_numpy(dtype=float)
    trend_values = np.full(len(g), np.nan, dtype=float)
    trend_n_obs = np.zeros(len(g), dtype=int)
    trend_slopes = np.full(len(g), np.nan, dtype=float)
    trend_intercepts = np.full(len(g), np.nan, dtype=float)
    trend_first_year = np.full(len(g), np.nan, dtype=float)
    trend_last_year = np.full(len(g), np.nan, dtype=float)
    for i, target_year in enumerate(years):
        if not np.isfinite(target_year):
            continue
        prior_mask = np.isfinite(years) & np.isfinite(yields) & (years < target_year)
        x = years[prior_mask]
        y = yields[prior_mask]
        n_obs = int(len(y))
        trend_n_obs[i] = n_obs
        if n_obs < TREND_MIN_OBS:
            continue
        if np.unique(x).size < 2:
            continue
        x_center = float(np.mean(x))
        x_centered = x - x_center
        try:
            slope, centered_intercept = np.polyfit(x_centered, y, 1)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if not (np.isfinite(slope) and np.isfinite(centered_intercept)):
            continue
        trend_value = centered_intercept + slope * (target_year - x_center)
        intercept = centered_intercept - slope * x_center
        if not np.isfinite(trend_value):
            continue
        trend_values[i] = float(trend_value)
        trend_slopes[i] = float(slope)
        trend_intercepts[i] = float(intercept)
        trend_first_year[i] = float(np.min(x))
        trend_last_year[i] = float(np.max(x))
        if not trend_last_year[i] < target_year:
            raise RuntimeError(f'Historical yield trend leakage detected: source year {trend_last_year[i]} is not before target year {target_year}.')
    g['hist_yield_trend'] = trend_values
    g['hist_trend_n_obs'] = trend_n_obs
    g['hist_trend_slope'] = trend_slopes
    g['hist_trend_intercept'] = trend_intercepts
    g['hist_trend_first_year'] = trend_first_year
    g['hist_trend_last_year'] = trend_last_year
    return g
_trend_parts = []
for _fips, _group in yield_features.groupby('FIPS', sort=False, dropna=False):
    _built = _build_causal_county_yield_trend(_group.copy())
    _built['FIPS'] = str(_fips).replace('.0', '').zfill(5)
    _trend_parts.append(_built)
if len(_trend_parts) == 0:
    raise RuntimeError('Historical yield trend builder produced zero county groups.')
yield_features = pd.concat(_trend_parts, ignore_index=True).sort_values(['FIPS', 'year']).reset_index(drop=True)
yield_features['yield_anomaly'] = yield_features['yield_bu_acre'] - yield_features['hist_yield_trend']
_required_trend_cols = {'FIPS', 'year', 'yield_bu_acre', 'hist_yield_trend', 'hist_trend_n_obs', 'hist_trend_slope', 'hist_trend_intercept', 'hist_trend_first_year', 'hist_trend_last_year', 'yield_anomaly'}
_missing_trend_cols = sorted(_required_trend_cols - set(yield_features.columns))
if _missing_trend_cols:
    raise RuntimeError('Historical trend schema audit failed. Missing columns: ' + ', '.join(_missing_trend_cols))
_valid_trend_rows = yield_features[yield_features['hist_yield_trend'].notna()].copy()
if len(_valid_trend_rows) > 0:
    _leak_mask = _valid_trend_rows['hist_trend_last_year'] >= _valid_trend_rows['year']
    if _leak_mask.any():
        _examples = _valid_trend_rows.loc[_leak_mask, ['FIPS', 'year', 'hist_trend_last_year']].head(20)
        raise RuntimeError(f'Historical yield trend leakage audit failed. Examples:\n{_examples.to_string(index=False)}')
_missing_calendar_rows = int(yield_features['yield_bu_acre'].isna().sum())
print('Calendar-complete yield grid:', yield_features.shape)
print('Missing county-year yield cells retained as NaN:', _missing_calendar_rows)
print('Historical yield baseline:', 'EXPANDING COUNTY LINEAR TREND')
print('Minimum prior observed yields:', TREND_MIN_OBS)
print('Trend-baseline rows available:', int(yield_features['hist_yield_trend'].notna().sum()))
print('✓ leakage audit passed: every trend uses only years before its target year')
trend_audit_path = os.path.join(OUTPUT_DIR_BASE, f'CY2_V4_linear_trend_baseline_audit_{VALIDATION_TAG}.csv')
yield_features[['FIPS', 'year', 'yield_bu_acre', 'hist_yield_trend', 'hist_trend_n_obs', 'hist_trend_slope', 'hist_trend_first_year', 'hist_trend_last_year', 'yield_anomaly']].to_csv(trend_audit_path, index=False)
print('✓ trend baseline audit saved:', trend_audit_path)

def prepare_vegetation_dataset(vegetation_df, label):
    df = vegetation_df.copy()
    if 'FIPS' not in df.columns:
        if 'GEOID' in df.columns:
            df = df.rename(columns={'GEOID': 'FIPS'})
        else:
            raise KeyError(f'{label}: no FIPS/GEOID.')
    df['FIPS'] = clean_fips(df['FIPS'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year'].notna()].copy()
    df['year'] = df['year'].astype(int)
    dup = df.duplicated(subset=['FIPS', 'year']).sum()
    if dup > 0:
        raise ValueError(f'{label}: {dup} duplicate county-years.')
    df = df.merge(yield_features[['FIPS', 'year', 'yield_bu_acre', 'hist_yield_trend', 'hist_trend_n_obs', 'hist_trend_slope', 'hist_trend_first_year', 'hist_trend_last_year', 'yield_anomaly']], on=['FIPS', 'year'], how='left', validate='one_to_one')
    print('\n', label)
    print('Shape:', df.shape)
    print('Yield rows:', df['yield_bu_acre'].notna().sum())
    print('hist_yield_trend rows:', df['hist_yield_trend'].notna().sum())
    return df


def add_vegetation_anomalies(input_df, label):
    df = input_df.copy()
    df = df.sort_values(['FIPS', 'year']).reset_index(drop=True)
    old_anoms = [c for c in df.columns if c.endswith('_anom') and '_DOY_' in c]
    if len(old_anoms) > 0:
        df = df.drop(columns=old_anoms)
    created = []
    for idx in VEG_INDICES:
        raw_cols = [c for c in df.columns if c.startswith(idx + '_DOY_') and (not c.endswith('_anom'))]
        raw_cols = sorted(raw_cols, key=lambda c: int(c.split('_DOY_')[-1]))
        for col in raw_cols:
            historical_mean = df.groupby('FIPS')[col].transform(lambda x: x.shift(1).expanding(min_periods=3).mean())
            anomaly_col = col + '_anom'
            df[anomaly_col] = pd.to_numeric(df[col], errors='coerce') - historical_mean
            created.append(anomaly_col)
    print(label, '| anomaly columns:', len(created))
    return df


def filter_model_rows(df, label):
    out = df[df['yield_bu_acre'].notna() & df['hist_yield_trend'].notna()].copy()
    out = out.reset_index(drop=True)
    print('\n' + '=' * 75)
    print(label)
    print('=' * 75)
    print('Rows:', f'{len(out):,}')
    print('Counties:', out['FIPS'].nunique())
    print('Years:', sorted(out['year'].unique()))
    return out
RAW_SOURCE_VARS = ['tmin', 'tmean', 'tmax', 'ppt', 'vpdmin', 'vpdmean', 'vpdmax', 'heat30', 'hot35days', 'drydays']
RAW_PRISM = ['recent16_tmin', 'recent16_tmean', 'recent16_tmax', 'recent16_ppt', 'recent32_ppt', 'season_ppt', 'recent16_vpdmean', 'recent16_vpdmax', 'season_vpdmean', 'recent16_heat30', 'season_heat30', 'recent16_hot35days', 'season_hot35days', 'recent16_drydays', 'season_drydays']
BASE_ENV = list(RAW_PRISM)

def infer_year_from_filename(path):
    matches = re.findall('(20\\d{2})', os.path.basename(path))
    if len(matches) == 0:
        return None
    return int(matches[-1])

def find_existing_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None

def discover_doys(columns):
    doys = set()
    for col in columns:
        match = re.search('_DOY_(\\d+)$', str(col), flags=re.IGNORECASE)
        if match:
            doys.add(int(match.group(1)))
    return sorted(doys)

def find_doy_column(columns, prefix, doy):
    lookup = {str(c).lower(): c for c in columns}
    candidates = [f'{prefix}_DOY_{doy}', f'{prefix}_DOY_{doy:03d}']
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None
if len(prism_csvs) == 0:
    raise FileNotFoundError('No PRISM weather files found.')
prism_parts = []
for path in prism_csvs:
    print('Loading PRISM:', os.path.basename(path))
    wide = pd.read_csv(path, low_memory=False)
    wide.columns = [str(c).strip() for c in wide.columns]
    fips_col = find_existing_column(wide.columns, ['FIPS', 'GEOID'])
    if fips_col is None:
        raise KeyError('No FIPS/GEOID found in ' + os.path.basename(path))
    wide['FIPS'] = clean_fips(wide[fips_col])
    year_col = find_existing_column(wide.columns, ['year', 'YEAR', 'Year'])
    if year_col is not None:
        wide['year'] = pd.to_numeric(wide[year_col], errors='coerce')
    else:
        file_year = infer_year_from_filename(path)
        if file_year is None:
            raise ValueError('Could not determine year for ' + os.path.basename(path))
        wide['year'] = file_year
    doys = discover_doys(wide.columns)
    if len(doys) == 0:
        raise ValueError('No *_DOY_* columns found in ' + os.path.basename(path))
    print('  DOYs:', doys)
    first_doy = doys[0]
    missing_raw = []
    for variable in RAW_SOURCE_VARS:
        found = find_doy_column(wide.columns, variable, first_doy)
        if found is None:
            missing_raw.append(variable)
    if len(missing_raw) > 0:
        raise KeyError('Missing raw PRISM variables in ' + os.path.basename(path) + ':\n' + str(missing_raw))
    for doy in doys:
        part = pd.DataFrame({'FIPS': wide['FIPS'].values, 'year': wide['year'].values, 'DOY': doy})
        for variable in RAW_SOURCE_VARS:
            source_col = find_doy_column(wide.columns, variable, doy)
            if source_col is None:
                part[variable] = np.nan
            else:
                part[variable] = pd.to_numeric(wide[source_col], errors='coerce').values
        prism_parts.append(part)
environment_raw = pd.concat(prism_parts, ignore_index=True)
environment_raw['year'] = pd.to_numeric(environment_raw['year'], errors='coerce')
environment_raw['DOY'] = pd.to_numeric(environment_raw['DOY'], errors='coerce')
environment_raw = environment_raw[environment_raw['year'].notna() & environment_raw['DOY'].notna()].copy()
environment_raw['year'] = environment_raw['year'].astype(int)
environment_raw['DOY'] = environment_raw['DOY'].astype(int)
duplicate_count = environment_raw.duplicated(subset=['FIPS', 'year', 'DOY'], keep=False).sum()
print('\nDuplicate FIPS-year-DOY weather rows:', duplicate_count)
if duplicate_count > 0:
    duplicate_weather_examples = environment_raw.loc[environment_raw.duplicated(subset=['FIPS', 'year', 'DOY'], keep=False), ['FIPS', 'year', 'DOY']].drop_duplicates().head(20)
    raise ValueError(f'Duplicate FIPS-year-DOY PRISM rows detected. The audited workflow will NOT silently average potentially different weather products. Remove overlapping PRISM inputs. First duplicate keys:\n{duplicate_weather_examples.to_string(index=False)}')
environment_raw = environment_raw.sort_values(['FIPS', 'year', 'DOY']).reset_index(drop=True)
environment = environment_raw[['FIPS', 'year', 'DOY']].copy()
environment['recent16_tmin'] = environment_raw['tmin']
environment['recent16_tmean'] = environment_raw['tmean']
environment['recent16_tmax'] = environment_raw['tmax']
environment['recent16_ppt'] = environment_raw['ppt']
environment['recent16_vpdmean'] = environment_raw['vpdmean']
environment['recent16_vpdmax'] = environment_raw['vpdmax']
environment['recent16_heat30'] = environment_raw['heat30']
environment['recent16_hot35days'] = environment_raw['hot35days']
environment['recent16_drydays'] = environment_raw['drydays']
environment['recent32_ppt'] = environment_raw.groupby(['FIPS', 'year'])['ppt'].transform(lambda x: x.rolling(2, min_periods=1).sum())
environment['season_ppt'] = environment_raw.groupby(['FIPS', 'year'])['ppt'].cumsum()
environment['season_vpdmean'] = environment_raw.groupby(['FIPS', 'year'])['vpdmean'].transform(lambda x: x.expanding(min_periods=1).mean())
environment['season_heat30'] = environment_raw.groupby(['FIPS', 'year'])['heat30'].cumsum()
environment['season_hot35days'] = environment_raw.groupby(['FIPS', 'year'])['hot35days'].cumsum()
environment['season_drydays'] = environment_raw.groupby(['FIPS', 'year'])['drydays'].cumsum()

def _normalized_basename(path):
    name = os.path.basename(path)
    stem, ext = os.path.splitext(name)
    stem = re.sub('\\s*\\(\\d+\\)$', '', stem)
    return (stem + ext).lower()

def find_named_input(target_name):
    target = target_name.lower()
    matches = [p for p in all_csvs if _normalized_basename(p) == target]
    if len(matches) == 0:
        return None
    return max(matches, key=lambda p: (os.path.getmtime(p), os.path.getsize(p), p))
AWC_PATH = find_named_input('CornBelt_SoilGrids_AWC.csv')
SOC_CEC_PATH = find_named_input('CornBelt_SoilGrids_SOC_CEC.csv')
if AWC_PATH is None:
    raise FileNotFoundError('FINAL MODEL requires CornBelt_SoilGrids_AWC.csv, but it was not found under --input-dir / staged ZIP inputs.')
if SOC_CEC_PATH is None:
    raise FileNotFoundError('FINAL MODEL requires CornBelt_SoilGrids_SOC_CEC.csv for CEC, but it was not found under --input-dir / staged ZIP inputs.')
print('\nFinal-model AWC file:', AWC_PATH)
print('Final-model CEC file:', SOC_CEC_PATH)
awc = pd.read_csv(AWC_PATH, low_memory=False)
awc.columns = [str(c).strip() for c in awc.columns]
if 'FIPS' not in awc.columns:
    if 'GEOID' in awc.columns:
        awc = awc.rename(columns={'GEOID': 'FIPS'})
    else:
        raise KeyError('AWC file has no FIPS/GEOID column.')
awc['FIPS'] = clean_fips(awc['FIPS'])
awc_dup = int(awc['FIPS'].duplicated().sum())
if awc_dup > 0:
    raise RuntimeError(f'AWC file contains {awc_dup} duplicate FIPS rows. Audited final workflow refuses to silently drop duplicates.')
for col in ['AWC_0_100_mm_crop', 'AWC_0_100_mm_all']:
    if col not in awc.columns:
        raise KeyError(f'AWC file missing required column: {col}')
    awc[col] = pd.to_numeric(awc[col], errors='coerce')
awc['AWC100'] = awc['AWC_0_100_mm_crop'].fillna(awc['AWC_0_100_mm_all'])
awc_small = awc[['FIPS', 'AWC100']].copy()
if awc_small['AWC100'].notna().sum() == 0:
    raise RuntimeError('AWC100 is entirely missing after crop/all fallback.')
print('AWC100 audit | counties=', awc_small['FIPS'].nunique(), '| non-missing=', int(awc_small['AWC100'].notna().sum()), '| range=', (float(awc_small['AWC100'].min()), float(awc_small['AWC100'].max())))
soil = pd.read_csv(SOC_CEC_PATH, low_memory=False)
soil.columns = [str(c).strip() for c in soil.columns]
if 'FIPS' not in soil.columns:
    if 'GEOID' in soil.columns:
        soil = soil.rename(columns={'GEOID': 'FIPS'})
    else:
        raise KeyError('SOC/CEC file has no FIPS/GEOID column.')
soil['FIPS'] = clean_fips(soil['FIPS'])
soil_dup = int(soil['FIPS'].duplicated().sum())
if soil_dup > 0:
    raise RuntimeError(f'SOC/CEC file contains {soil_dup} duplicate FIPS rows. Audited final workflow refuses to silently drop duplicates.')
CEC_SOURCE_COLS = sorted([c for c in soil.columns if str(c).upper().startswith('CEC_')])
if len(CEC_SOURCE_COLS) == 0:
    raise RuntimeError('No CEC_* columns were found in CornBelt_SoilGrids_SOC_CEC.csv.')
if len(CEC_SOURCE_COLS) != 20:
    raise RuntimeError(f'FINAL MODEL was selected using exactly 20 raw CEC_* features, but the current file exposes {len(CEC_SOURCE_COLS)}. Detected columns: {CEC_SOURCE_COLS}')
for col in CEC_SOURCE_COLS:
    soil[col] = pd.to_numeric(soil[col], errors='coerce')
cec_small = soil[['FIPS'] + CEC_SOURCE_COLS].copy()
print('CEC audit | counties=', cec_small['FIPS'].nunique(), '| raw CEC features=', len(CEC_SOURCE_COLS))
environment = environment.merge(awc_small, on='FIPS', how='left', validate='many_to_one')
environment = environment.merge(cec_small, on='FIPS', how='left', validate='many_to_one')
static_join_duplicates = int(environment.duplicated(subset=['FIPS', 'year', 'DOY']).sum())
if static_join_duplicates > 0:
    raise RuntimeError(f'Static soil joins created duplicate FIPS-year-DOY rows: {static_join_duplicates}')
BASE_ENV = list(RAW_PRISM) + ['AWC100'] + list(CEC_SOURCE_COLS)
EXPECTED_BASE_ENV_COUNT = 36
if len(BASE_ENV) != EXPECTED_BASE_ENV_COUNT:
    raise RuntimeError(f'FINAL BASE_ENV must contain {EXPECTED_BASE_ENV_COUNT} predictors (15 PRISM + 1 AWC100 + 20 CEC); got {len(BASE_ENV)}.')
fully_missing_final_environment = [c for c in BASE_ENV if environment[c].isna().all()]
if len(fully_missing_final_environment) > 0:
    raise RuntimeError(f'Final environmental feature(s) are completely missing after the static joins: {fully_missing_final_environment}')
print('\n' + '=' * 75)
print('FINAL PRODUCTION ENVIRONMENT POLICY')
print('=' * 75)
print('Raw compact PRISM :', len(RAW_PRISM))
print('AWC100            : 1')
print('Raw CEC features  :', len(CEC_SOURCE_COLS))
print('TOTAL BASE_ENV    :', len(BASE_ENV))
print('Weather anomalies : EXCLUDED')
print('AWC interactions  : EXCLUDED')
print('SOC               : EXCLUDED')
print('LST               : EXCLUDED')
print('=' * 75)
print('\n' + '=' * 75)
print('CANONICAL COMPACT PRISM WEATHER READY')
print('=' * 75)
print('Environment shape:', environment.shape)
print('Counties:', environment['FIPS'].nunique())
print('Years:', environment['year'].min(), 'to', environment['year'].max())
print('DOYs:', sorted(environment['DOY'].unique()))
print('BASE_ENV predictors:', len(BASE_ENV))
if len(BASE_ENV) != EXPECTED_BASE_ENV_COUNT:
    raise RuntimeError('Canonical final BASE_ENV feature count changed unexpectedly.')
print('✓ BASE_ENV = 15 raw PRISM + AWC100 + 20 raw CEC predictors')
print('\nMissing compact PRISM values:')
print(environment[BASE_ENV].isna().sum())
display(environment.head(10))

def build_features(df_sub, doy_cut):
    feature_list = []

    def get_doy(column):
        value = column.split('_DOY_')[-1].replace('_anom', '')
        return int(value)

    def interpolate_matrix(matrix, doys):
        """Interpolate only gaps bracketed by real observations.

        np.interp normally clamps missing tails to the nearest observed value.
        That silently fabricates early/late-season vegetation. Here, leading
        and trailing gaps remain NaN and are handled later by the existing
        training-only imputation policy.
        """
        X = np.asarray(matrix, dtype=float).copy()
        doys = np.asarray(doys, dtype=float)
        output = np.full_like(X, np.nan, dtype=float)
        for i in range(X.shape[0]):
            row = X[i]
            good = np.isfinite(row)
            n_good = int(good.sum())
            if n_good == 0:
                continue
            if n_good == 1:
                output[i, good] = row[good]
                continue
            x_good = doys[good]
            y_good = row[good]
            if np.unique(x_good).size < 2:
                output[i, good] = row[good]
                continue
            inside = (doys >= np.min(x_good)) & (doys <= np.max(x_good))
            output[i, inside] = np.interp(doys[inside], x_good, y_good)
        return output

    def row_auc(values, doys):
        good = np.isfinite(values)
        if good.sum() < 2:
            return np.nan
        x = doys[good]
        y = values[good]
        if hasattr(np, 'trapezoid'):
            return np.trapezoid(y, x)
        return np.trapz(y, x)

    def row_peak_doy(values, doys):
        good = np.isfinite(values)
        if good.sum() == 0:
            return np.nan
        y = values[good]
        x = doys[good]
        return x[np.argmax(y)]

    def row_slope(values, doys):
        good = np.isfinite(values)
        if good.sum() < 2:
            return np.nan
        x_good = doys[good]
        y_good = values[good]
        if np.unique(x_good).size < 2:
            return np.nan
        try:
            return np.polyfit(x_good, y_good, 1)[0]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.nan
    for idx in VEG_INDICES:
        raw_cols = [c for c in df_sub.columns if c.startswith(idx + '_DOY_') and (not c.endswith('_anom'))]
        raw_cols = [c for c in raw_cols if get_doy(c) <= doy_cut]
        raw_cols = sorted(raw_cols, key=get_doy)
        if len(raw_cols) > 0:
            raw_doys = np.array([get_doy(c) for c in raw_cols], dtype=float)
            raw_matrix = df_sub[raw_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            X = interpolate_matrix(raw_matrix, raw_doys)
            feature_list.append(pd.DataFrame(X, index=df_sub.index, columns=raw_cols))
            feat = pd.DataFrame(index=df_sub.index)
            Xdf = pd.DataFrame(X, index=df_sub.index)
            feat[f'{idx}_mean'] = Xdf.mean(axis=1)
            feat[f'{idx}_max'] = Xdf.max(axis=1)
            feat[f'{idx}_min'] = Xdf.min(axis=1)
            feat[f'{idx}_std'] = Xdf.std(axis=1, ddof=0)
            feat[f'{idx}_last'] = X[:, -1]
            feat[f'{idx}_auc'] = [row_auc(row, raw_doys) for row in X]
            feat[f'{idx}_peak_doy'] = [row_peak_doy(row, raw_doys) for row in X]
            if X.shape[1] >= 3:
                feat[f'{idx}_early_mean'] = pd.DataFrame(X[:, :3], index=df_sub.index).mean(axis=1)
            if X.shape[1] >= 2:
                delta_doy = raw_doys[-1] - raw_doys[-2]
                if delta_doy > 0:
                    feat[f'{idx}_slope_last'] = (X[:, -1] - X[:, -2]) / delta_doy
            green_mask = (raw_doys >= 60) & (raw_doys <= 120)
            green_doys = raw_doys[green_mask]
            green_X = X[:, green_mask]
            if len(green_doys) >= 2:
                feat[f'{idx}_greenup_rate'] = [row_slope(row, green_doys) for row in green_X]
            feature_list.append(feat)
        anom_cols = [c for c in df_sub.columns if c.startswith(idx + '_DOY_') and c.endswith('_anom')]
        anom_cols = [c for c in anom_cols if get_doy(c) <= doy_cut]
        anom_cols = sorted(anom_cols, key=get_doy)
        if len(anom_cols) > 0:
            anom_doys = np.array([get_doy(c) for c in anom_cols], dtype=float)
            raw_anom = df_sub[anom_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            X_anom = interpolate_matrix(raw_anom, anom_doys)
            feat_anom = pd.DataFrame(index=df_sub.index)
            Adf = pd.DataFrame(X_anom, index=df_sub.index)
            feat_anom[f'{idx}_anom_mean'] = Adf.mean(axis=1)
            feat_anom[f'{idx}_anom_max'] = Adf.max(axis=1)
            feat_anom[f'{idx}_anom_min'] = Adf.min(axis=1)
            feat_anom[f'{idx}_anom_last'] = X_anom[:, -1]
            if X_anom.shape[1] >= 2:
                delta_doy = anom_doys[-1] - anom_doys[-2]
                if delta_doy > 0:
                    feat_anom[f'{idx}_anom_slope_last'] = (X_anom[:, -1] - X_anom[:, -2]) / delta_doy
            feature_list.append(feat_anom)
    if len(feature_list) == 0:
        return pd.DataFrame(index=df_sub.index)
    features = pd.concat(feature_list, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    features = features.replace([np.inf, -np.inf], np.nan)
    return features
print('✓ build_features() ready')
environment['FIPS'] = clean_fips(environment['FIPS'])
env_indexed = environment.set_index(['FIPS', 'year', 'DOY']).sort_index()

def get_environment(rows, doy):
    """Return final PRISM + AWC100 + CEC features and fail loudly on key misalignment."""
    keys = pd.MultiIndex.from_arrays([rows['FIPS'].astype(str), rows['year'].astype(int), np.full(len(rows), int(doy))], names=['FIPS', 'year', 'DOY'])
    X = env_indexed.reindex(keys)[BASE_ENV].copy()
    X.index = rows.index
    if len(X) == 0:
        raise RuntimeError(f'DOY {doy}: PRISM lookup returned zero rows.')
    fully_missing_features = X.columns[X.isna().all(axis=0)].tolist()
    if fully_missing_features:
        raise RuntimeError(f'DOY {doy}: PRISM feature(s) are completely missing after key lookup: {fully_missing_features}. Check FIPS/year/DOY alignment and source files.')
    fully_missing_rows = X.isna().all(axis=1)
    n_fully_missing_rows = int(fully_missing_rows.sum())
    if n_fully_missing_rows > 0:
        examples = rows.loc[fully_missing_rows, ['FIPS', 'year']].head(10)
        raise RuntimeError(f'DOY {doy}: {n_fully_missing_rows} requested county-year rows have NO PRISM features. Example keys:\n{examples.to_string(index=False)}')
    overall_coverage = float(X.notna().mean().mean())
    if overall_coverage < 0.99:
        warnings.warn(f'DOY {doy}: PRISM cell coverage is {overall_coverage:.2%}. Scattered missing values will use training-only median imputation.')
    return X
print('✓ environment lookup ready')
LEAKAGE_COLS = ['yield_bu_acre', 'yield_filled', 'yield_anomaly', 'target', 'yield']

def make_model(seed=42):
    params = dict(XGB_PARAMS)
    params['random_state'] = seed
    return XGBRegressor(**params)
print('✓ V5 XGBoost model ready')

# ============================================================
# MATCHED TRAINING AND TESTING EXPERIMENT
# ============================================================
DOYS = sorted(set(ARGS.doys))
TEST_YEARS = sorted(set(ARGS.test_years))
GRID = list(range(65, 274, 16))
if any(d not in [193, 209, 225, 241, 257, 273] for d in DOYS):
    raise ValueError('--doys must come from 193 209 225 241 257 273.')
if ARGS.trees < 1:
    raise ValueError('--trees must be positive.')
XGB_PARAMS['n_estimators'] = ARGS.trees
XGB_PARAMS['n_jobs'] = ARGS.jobs
OUT = Path(OUTPUT_DIR_BASE)
FIG = OUT / 'figures'
IMP = OUT / 'feature_importance'
FIG.mkdir(exist_ok=True)
IMP.mkdir(exist_ok=True)
RAW_COLS = [f'{v}_DOY_{d}' for v in VEG_INDICES for d in GRID]
KEYS = ['FIPS', 'year']


def canonical_wide(wide):
    result = wide.rename(columns={'GEOID': 'FIPS'}).copy()
    result['FIPS'] = clean_fips(result['FIPS'])
    result['year'] = pd.to_numeric(result['year'], errors='raise').astype(int)
    if result.duplicated(KEYS).any():
        raise ValueError('Duplicate vegetation county-years.')
    # Give both training masks the identical potential DOY grid, even with missing data.
    return result.reindex(columns=KEYS + RAW_COLS).sort_values(KEYS).reset_index(drop=True)


raw_histories = {'PREVIOUS_CDL': canonical_wide(previous_gee_wide),
                 'SAME_YEAR_CDL': canonical_wide(same_year_gee_wide)}
coverage = []
for label, frame in raw_histories.items():
    for year, group in frame.groupby('year'):
        coverage.append(dict(training_mask=label, year=year, county_years=len(group)))
pd.DataFrame(coverage).to_csv(OUT / 'history_coverage.csv', index=False)
shared_keys = raw_histories['PREVIOUS_CDL'][KEYS].merge(
    raw_histories['SAME_YEAR_CDL'][KEYS], on=KEYS, validate='one_to_one')
if shared_keys.empty:
    raise ValueError('No overlapping county-years between annual previous and same-year CDL histories.')
print('\nMatched history coverage:')
print(shared_keys.groupby('year').size().to_string())
matched_histories = {}
training_tables = {}
for label, frame in raw_histories.items():
    matched = shared_keys.merge(frame, on=KEYS, validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
    matched_histories[label] = matched
    with_yield = prepare_vegetation_dataset(matched, label)
    training_tables[label] = add_vegetation_anomalies(with_yield, label)
shared_keys.to_csv(OUT / 'matched_history_county_years.csv', index=False)


def prepare_test(raw, year, training_mask):
    """Test-mask observations minus PRIOR history of the TRAINING mask."""
    test = prepare_vegetation_dataset(raw, 'TEST ' + training_mask)
    if not test['year'].eq(year).all():
        raise ValueError('Test rows contain an unexpected year.')
    history = matched_histories[training_mask]
    history = history.loc[history.year < year]
    means = history.groupby('FIPS')[RAW_COLS].mean()
    counts = history.groupby('FIPS')[RAW_COLS].count()
    means = means.where(counts >= 3)  # match training expanding(min_periods=3)
    aligned = means.reindex(test.FIPS).to_numpy()
    anomalies = pd.DataFrame(test[RAW_COLS].to_numpy() - aligned,
                             columns=[c + '_anom' for c in RAW_COLS], index=test.index)
    return pd.concat([test, anomalies], axis=1)


def eligible_rows(rows, doy):
    """Require real vegetation, observed target, prior-yield baseline and weather keys."""
    columns = [c for c in RAW_COLS if int(c.rsplit('_', 1)[-1]) <= doy]
    valid = (np.isfinite(rows[columns]).any(axis=1)
             & np.isfinite(rows['yield_bu_acre']) & np.isfinite(rows['hist_yield_trend'])
             & np.isfinite(rows['yield_anomaly']))
    keys = pd.MultiIndex.from_arrays([rows.FIPS, rows.year, np.full(len(rows), doy)],
                                     names=['FIPS', 'year', 'DOY'])
    weather = env_indexed.reindex(keys)[RAW_PRISM]
    valid &= np.isfinite(weather.to_numpy()).any(axis=1)
    result = rows.loc[valid].copy()
    if not (result.hist_trend_last_year < result.year).all():
        raise ValueError('Historical yield baseline includes a current/future yield.')
    return result


def feature_matrix(rows, doy):
    veg = build_features(rows, doy)
    weather = get_environment(rows, doy)
    matrix = pd.concat([veg, weather], axis=1)
    matrix['hist_yield_trend'] = rows.hist_yield_trend.to_numpy()
    matrix['year'] = rows.year.astype(float).to_numpy()
    matrix = matrix.drop(columns=LEAKAGE_COLS, errors='ignore')
    matrix = matrix.select_dtypes(include=[np.number, 'bool']).replace([np.inf, -np.inf], np.nan)
    if matrix.columns.duplicated().any():
        raise ValueError('Duplicate model features.')
    return matrix


def month_for_checkpoint(doy):
    return 'JUNE' if doy <= 209 else ('JULY' if doy <= 241 else 'AUGUST')


def permitted_mask(year, family, month, doy):
    forecast = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=doy - 1)
    if month:
        earliest = pd.Timestamp(year, {'JUNE': 7, 'JULY': 8, 'AUGUST': 9}[month], 1)
    else:
        earliest = pd.Timestamp(year, 1, 1)
    recorded = MASK_AVAILABLE.get((year, family, month))
    if recorded is not None:
        earliest = max(earliest, recorded)
    return forecast >= earliest


SCENARIO_CACHE = {}
SKIPPED = []


def test_scenarios(year, doy):
    result = {}
    for name, source in [('PREVIOUS_CDL', 'PREVIOUS_CDL'), ('FINAL_CDL', 'SAME_YEAR_CDL')]:
        raw = raw_histories[source].loc[raw_histories[source].year == year].copy()
        if raw.empty:
            raise ValueError(f'Missing {name} export for evaluation year {year}.')
        if name == 'PREVIOUS_CDL' and not permitted_mask(year, name, None, doy):
            raise ValueError(f'Previous-year mask is not available at {year} DOY {doy}.')
        result[name] = raw
    month = month_for_checkpoint(doy)
    for family in ['CLEAN_REBUILT', 'STUDY']:
        key = (year, family, month)
        if key not in EXPORT_FILES:
            SKIPPED.append(dict(year=year, doy=doy, scenario=family, reason=f'No {month} export'))
            continue
        if not permitted_mask(year, family, month, doy):
            SKIPPED.append(dict(year=year, doy=doy, scenario=family, reason='Mask not available yet'))
            continue
        if key not in SCENARIO_CACHE:
            SCENARIO_CACHE[key] = canonical_wide(build_gee_wide([EXPORT_FILES[key]], label=str(key)))
        result[family] = SCENARIO_CACHE[key]
    return result


def feature_group(name):
    if name == 'hist_yield_trend':
        return 'Historical yield trend'
    if name == 'year':
        return 'Calendar year'
    if name in RAW_PRISM:
        return 'PRISM weather'
    if name == 'AWC100':
        return 'Soil AWC100'
    if name.upper().startswith('CEC_'):
        return 'Soil CEC'
    if '_anom_' in name or name.endswith('_anom'):
        return 'Vegetation anomalies'
    return 'Vegetation raw and seasonal'


def bar_plot(series, title, path, xlabel, top=20):
    series = series.sort_values(ascending=False).head(top).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(series) * .3 + 1)))
    ax.barh(series.index, series.values, color='#2a7893')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


IMPORTANCE_ROWS = []


def save_importance(model, columns, training_mask, year, doy):
    # TOTAL GAIN sums loss reduction across splits. Zero-use features are retained.
    scores = model.get_booster().get_score(importance_type='total_gain')
    values = pd.Series({c: float(scores.get(c, scores.get(f'f{i}', 0.0))) for i, c in enumerate(columns)})
    total = values.sum()
    share = values / total if total > 0 else values * 0
    for name in columns:
        IMPORTANCE_ROWS.append(dict(training_mask=training_mask, year=year, doy=doy,
                                    feature=name, group=feature_group(name),
                                    total_gain=values[name], gain_share=share[name]))
    prefix = f'{training_mask}_{year}_DOY{doy}'
    bar_plot(100 * share, f'{training_mask}: {year}, DOY {doy} — top training features',
             IMP / f'{prefix}_top_features.png', 'Share of total training split gain (%)')
    groups = share.groupby(share.index.map(feature_group)).sum()
    bar_plot(100 * groups, f'{training_mask}: {year}, DOY {doy} — feature groups',
             IMP / f'{prefix}_feature_groups.png', 'Share of total training split gain (%)')


METRICS = []
PREDICTIONS = []
TRAIN_AUDIT = []
COHORT_AUDIT = []
FEATURE_AUDIT = []

def write_progress():
    pd.DataFrame(METRICS).to_csv(OUT / 'model_comparison_metrics.csv', index=False)
    if PREDICTIONS:
        pd.concat(PREDICTIONS, ignore_index=True).to_csv(OUT / 'county_predictions.csv', index=False)
    pd.DataFrame(TRAIN_AUDIT).to_csv(OUT / 'training_audit.csv', index=False)
    pd.DataFrame(COHORT_AUDIT).to_csv(OUT / 'test_cohort_audit.csv', index=False)
    pd.DataFrame(FEATURE_AUDIT).to_csv(OUT / 'feature_schema_and_training_medians.csv', index=False)
    pd.DataFrame(IMPORTANCE_ROWS).to_csv(IMP / 'feature_importance.csv', index=False)
    pd.DataFrame(SKIPPED, columns=['year', 'doy', 'scenario', 'reason']).to_csv(OUT / 'skipped_scenarios.csv', index=False)


for year in TEST_YEARS:
    for doy in DOYS:
        print(f'\n===== {year} DOY {doy}: matching both training histories =====', flush=True)
        train = {label: eligible_rows(frame.loc[frame.year < year].copy(), doy)
                 for label, frame in training_tables.items()}
        common_train = train['PREVIOUS_CDL'][KEYS].merge(train['SAME_YEAR_CDL'][KEYS], on=KEYS,
                                                        validate='one_to_one').sort_values(KEYS)
        if common_train.year.nunique() < 3 or len(common_train) < 20:
            raise ValueError(f'{year} DOY {doy}: need >=3 historical training years and >=20 county-years '
                             'shared by both masks. Include all annual historical PREVIOUS_CDL exports.')
        common_train.to_csv(OUT / f'training_cohort_{year}_DOY{doy}.csv', index=False)
        train = {label: common_train.merge(frame, on=KEYS, validate='one_to_one').reset_index(drop=True)
                 for label, frame in train.items()}
        raw_tests = test_scenarios(year, doy)
        prepared = {name: eligible_rows(prepare_test(raw, year, 'SAME_YEAR_CDL'), doy)
                    for name, raw in raw_tests.items()}
        empty_optional = [name for name, rows in prepared.items() if rows.empty and name not in ('PREVIOUS_CDL', 'FINAL_CDL')]
        for name in empty_optional:
            SKIPPED.append(dict(year=year, doy=doy, scenario=name, reason='No usable county rows'))
            del prepared[name]
        common_test = prepared['PREVIOUS_CDL'][KEYS]
        for rows in prepared.values():
            common_test = common_test.merge(rows[KEYS], on=KEYS, validate='one_to_one')
        common_test = common_test.sort_values(KEYS).reset_index(drop=True)
        if len(common_test) < 2:
            raise ValueError(f'{year} DOY {doy}: fewer than two common test counties across available masks.')
        for name, rows in prepared.items():
            COHORT_AUDIT.append(dict(year=year, doy=doy, scenario=name,
                                     eligible_counties=len(rows), common_counties=len(common_test)))
        common_test.to_csv(OUT / f'test_cohort_{year}_DOY{doy}.csv', index=False)
        matrices = {label: feature_matrix(rows, doy) for label, rows in train.items()}
        # Feature availability is decided using training data only, identically for both models.
        columns = [c for c in matrices['PREVIOUS_CDL'].columns
                   if all(c in x and x[c].notna().any() for x in matrices.values())]
        if not columns:
            raise ValueError('No common usable training features.')
        for label in ['PREVIOUS_CDL', 'SAME_YEAR_CDL']:
            rows = train[label]
            assert rows.year.max() < year
            matrix = matrices[label][columns]
            medians = matrix.median()
            matrix = matrix.fillna(medians)
            if not np.isfinite(matrix.to_numpy()).all():
                raise ValueError('Nonfinite training features after training-only imputation.')
            model = make_model(ARGS.seed)
            model.fit(matrix, rows.yield_anomaly.to_numpy())
            save_importance(model, columns, label, year, doy)
            TRAIN_AUDIT.append(dict(training_mask=label, test_year=year, doy=doy, n_train=len(rows),
                                    train_min_year=int(rows.year.min()), train_max_year=int(rows.year.max()),
                                    n_train_years=rows.year.nunique(), n_features=len(columns),
                                    temporal_status=TIMING_STATUS))
            FEATURE_AUDIT.extend(dict(training_mask=label, year=year, doy=doy, feature=c,
                                      training_median=float(medians[c])) for c in columns)
            # The same-year model also tests previous-year masks as a diagnostic control.
            names = ['PREVIOUS_CDL'] if label == 'PREVIOUS_CDL' else list(prepared)
            for scenario in names:
                if label == 'PREVIOUS_CDL':
                    test = prepare_test(raw_tests[scenario], year, label)
                else:
                    test = prepared[scenario]
                test = common_test.merge(test, on=KEYS, validate='one_to_one').reset_index(drop=True)
                x_test = feature_matrix(test, doy).reindex(columns=columns).fillna(medians)
                if not np.isfinite(x_test.to_numpy()).all():
                    raise ValueError('Nonfinite test features after training-only imputation.')
                predicted = model.predict(x_test) + test.hist_yield_trend.to_numpy()
                actual = test.yield_bu_acre.to_numpy()
                result = dict(training_mask=label, testing_mask=scenario, year=year, doy=doy,
                              mask_month=month_for_checkpoint(doy) if scenario in ('CLEAN_REBUILT', 'STUDY') else '',
                              n_train=len(rows), n_test=len(test), r2=r2_score(actual, predicted),
                              rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
                              mae=mean_absolute_error(actual, predicted), temporal_status=TIMING_STATUS)
                METRICS.append(result)
                prediction = test[KEYS + ['yield_bu_acre', 'hist_yield_trend']].copy()
                prediction['predicted_yield'] = predicted
                prediction['residual'] = actual - predicted
                prediction['training_mask'] = label
                prediction['testing_mask'] = scenario
                prediction['doy'] = doy
                PREDICTIONS.append(prediction)
                print(f"{label} -> {scenario}: N={len(test)}, R2={result['r2']:.3f}, RMSE={result['rmse']:.2f}", flush=True)
        write_progress()

metrics = pd.DataFrame(METRICS)
labels = {('PREVIOUS_CDL', 'PREVIOUS_CDL'): 'Previous CDL training → previous CDL test',
          ('SAME_YEAR_CDL', 'PREVIOUS_CDL'): 'Same-year training → previous CDL test (control)',
          ('SAME_YEAR_CDL', 'CLEAN_REBUILT'): 'Same-year training → our ICDL test',
          ('SAME_YEAR_CDL', 'STUDY'): 'Same-year training → published ICDL test',
          ('SAME_YEAR_CDL', 'FINAL_CDL'): 'Same-year training → final CDL (hindsight)'}
for year, group in metrics.groupby('year'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, values in group.groupby(['training_mask', 'testing_mask']):
        values = values.sort_values('doy')
        style = '--' if key[1] == 'FINAL_CDL' else '-'
        for ax, metric, ylabel in zip(axes, ['r2', 'rmse'], ['County yield R²', 'RMSE (bu/acre)']):
            ax.plot(values.doy, values[metric], marker='o', linestyle=style, label=labels[key])
            ax.set_xlabel('Forecast day of year')
            ax.set_ylabel(ylabel)
            ax.set_xticks(DOYS)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{year}: matched training county-years and common test counties per checkpoint')
    fig.tight_layout()
    fig.savefig(FIG / f'model_comparison_{year}.png', dpi=180)
    plt.close(fig)

importance = pd.DataFrame(IMPORTANCE_ROWS)
for label, group in importance.groupby('training_mask'):
    # Every model contributes equally; unavailable features count as zero in this descriptive summary.
    pivot = group.pivot_table(index=['year', 'doy'], columns='feature', values='gain_share', fill_value=0)
    mean = pivot.mean()
    bar_plot(100 * mean, f'{label}: mean importance across evaluated model fits',
             IMP / f'{label}_mean_top_features.png', 'Mean share of total training split gain (%)')
    bar_plot(100 * mean.groupby(mean.index.map(feature_group)).sum(),
             f'{label}: mean feature-group importance', IMP / f'{label}_mean_feature_groups.png',
             'Mean share of total training split gain (%)')

# Difference isolates training-mask choice on the identical previous-CDL test inputs/cohort.
previous_scores = metrics.loc[metrics.testing_mask == 'PREVIOUS_CDL']
a = previous_scores.loc[previous_scores.training_mask == 'PREVIOUS_CDL']
b = previous_scores.loc[previous_scores.training_mask == 'SAME_YEAR_CDL']
delta = a.merge(b, on=['year', 'doy'], suffixes=('_previous_training', '_same_year_training'), validate='one_to_one')
for metric in ['r2', 'rmse', 'mae']:
    delta[f'delta_{metric}_previous_minus_same_year_training'] = (
        delta[f'{metric}_previous_training'] - delta[f'{metric}_same_year_training'])
delta.to_csv(OUT / 'training_mask_effect_on_previous_cdl.csv', index=False)
(OUT / 'run_config.json').write_text(json.dumps(dict(
    model_version=MODEL_VERSION, test_years=TEST_YEARS, doys=DOYS, xgb_parameters=XGB_PARAMS,
    seed=ARGS.seed, temporal_status=TIMING_STATUS,
    vegetation_grid=GRID, prism_features=RAW_PRISM, soil_features=['AWC100'] + CEC_SOURCE_COLS,
    protocol='Expanding historical years; paired training keys; shared feature schema; train-only medians; common test counties per DOY.',
    interpretation='Final CDL test is a hindsight benchmark, not a guaranteed numerical upper bound. '
    'PRISM is retrospective. Checked timing metadata does not verify actual historical publication dates '
    'or ICDL training-label provenance. June/July/August masks are allowed from July/August/September 1 '
    'or the later declared availability date. Importance is descriptive training total gain, not causal.'
), indent=2), encoding='utf-8')
print('\nCOMPLETE:', OUT)
print('Metrics: model_comparison_metrics.csv | Plots: figures | Importance: feature_importance')
