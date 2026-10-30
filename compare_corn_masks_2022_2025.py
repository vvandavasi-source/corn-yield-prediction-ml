#!/usr/bin/env python3
"""Compare corn maps to final annual CDL, without rerunning MODIS or yield models.

WINDOWS QUICK START (save this file in D:\\corn_yield):
  conda activate corn-yield
  cd /d "D:\\corn_yield"
  python -m pip install earthengine-api pandas numpy matplotlib
  python compare_corn_masks_2022_2025.py export --authenticate

This submits 36 Google Earth Engine Drive exports: 12 states x 2022-2024.
Tasks continue in GEE after Python exits. Download the resulting CMAP_*.csv files
from Drive's CornMask_Benchmark_2022_2024 folder to a new local folder, then:
  python compare_corn_masks_2022_2025.py summarize --csv-dir "D:\\corn_yield\\mask_benchmark_csvs"

REMAKE FIGURES FROM AN EXISTING RESULTS FOLDER (NO GEE OR NEW EXPORTS):
  python compare_corn_masks_2022_2025.py plots-only --results-dir "D:\\corn_yield\\mask_benchmark_results_20261030_013844_238021"
New figures go into a separate figures_updated_* subfolder by default.
Report colors: OUR orange, PUBLISHED purple, final CDL green, previous CDL gray.
The new corn_area_by_month_vs_final_CDL figure and companion CSV compare
June/July/August mapped corn area against final CDL on the SAME valid domain.
They do not extrapolate incomplete maps to the full 12-state region.

Optional 2025 (uses the previously configured custom CDL_2025_30m reference):
  python compare_corn_masks_2022_2025.py export --years 2025 --authenticate
  python compare_corn_masks_2022_2025.py summarize --csv-dir PATH --years 2025

No raw raster download is required. Python export needs authenticated access to
ee-gtellezgiron. Summarize is entirely local. No model fitting occurs.

Reference: final annual CDL (corn=1, all other VALID classes=non-corn).
Candidates: OUR, PUBLISHED, PREVIOUS_CDL; plus direct OUR/PUBLISHED agreement.
Final CDL is a reference/area baseline, NOT scored against itself as evidence.
Scores measure agreement with CDL, not independent ground-reference accuracy.

Two domains are ALWAYS retained:
  COMMON: final CDL AND all three candidate maps valid at a pixel. Use this for
          fair comparisons across masks. Its coverage can differ by month.
  PAIRWISE: final CDL AND that one candidate valid. Useful for coverage audits;
            scores across different domains are NOT directly comparable.
Missing data are never converted into valid non-corn predictions.

All maps use one declared EPSG:5070 30-m grid, nearest-neighbor reprojection,
and unweighted pixel-center inclusion within TIGER/2018 counties. Count x
900 m2 gives equal-area GRID area, not exact fractional polygon intersection.
This standardizes evaluation, not the information content of 10/30-m sources.
Existing assets' earlier resampling/pyramid choices cannot be reversed here.

DEFAULT ASSET PRESET: archived-published. Uses OUR assets from the confirmed
GEE exporter and the archived published 10-m sources for 2022/2023, followed
by the published 30-m sources for 2024/2025. The optional latest-extraction
preset substitutes the newer 2023 CropSmart 30-m sources.
IMPORTANT: the old yield CSV metadata records DIFFERENT 2023 OUR asset IDs
from the confirmed exporter. archived-yield-recorded retains those exact
recorded IDs, if you need to investigate that discrepancy. The default run
is therefore a mapping benchmark of the explicitly configured products,
not proof of byte-identical input maps to that old yield run. Inspect source
provenance before linking the two experiments. No substitute IDs are guessed.
These presets are different experiments; the hash prevents accidental mixing.
Every class-code/NoData assumption is explicit and exported. Check these
against the downloaded product legends before using results in a report.
In particular, published 30-m presets inherit 0=NoData from the extraction;
if YOUR source is binary 0=valid non-corn, remove 0 from that spec's nodata list
in a config JSON. No fallback asset names or implicit legend guesses are used.

Commands:
  config       Write editable JSON (no GEE login).
  export       Preflight every asset, record metadata, submit/resume tasks.
  status       Check task states in an export folder.
  summarize    Validate completeness, aggregate COUNTS, recompute metrics,
               make county/state/pooled tables, paired differences, area R2,
               coverage flags and publication figures (300-dpi PNG + PDF).
  plots-only   Rebuild report-color figures and monthly corn-area comparison
               from existing pooled/state CSVs, without contacting GEE.
  self-test    Independent synthetic checks; does not claim real-map results.

Use --help after any command for options. A new export folder is created by
default. --out-dir reuses the same tasks/config and retries only failed ones.
Supply the SAME --years / --states / --preset or --config when resuming.
No probabilities or AUC are invented from hard categorical masks. No pixel
independence assumption or spurious per-pixel significance tests are made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VERSION = 'CORN_MASK_BENCHMARK_V1'
PLOT_VERSION = 'REPORT_COLORS_MONTHLY_CORN_AREA_V2'
REPORT_COLORS = {'OUR':'#c66a15', 'PUBLISHED':'#7651a0',
                 'FINAL_CDL':'#2a7c4a', 'PREVIOUS_CDL':'#6c737a'}
ROOT = 'projects/ee-gtellezgiron/assets/'
MONTHS = ('June', 'July', 'August')
MASKS = ('OUR', 'PUBLISHED', 'PREVIOUS_CDL')
COUNTS = ('tp', 'fp', 'fn', 'tn')
STATES = {'17':'IL', '18':'IN', '19':'IA', '20':'KS', '26':'MI', '27':'MN',
          '29':'MO', '31':'NE', '38':'ND', '39':'OH', '46':'SD', '55':'WI'}
META = ['schema', 'config_hash', 'year', 'month', 'state_fips', 'county_fips',
        'county_name', 'grid_crs', 'grid_transform', 'pixel_km2', 'preset',
        'reference_asset', 'our_assets', 'published_assets', 'previous_asset',
        'our_product', 'published_product', 'reference_nodata', 'our_nodata',
        'published_nodata', 'previous_nodata', 'corn_codes', 'mosaic_order']


def now():
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def spec(ids, product, nodata=(255,), band=0):
    return {'ids': [ROOT + x for x in ids], 'band': band, 'corn_code': 1,
            'nodata': list(nodata), 'product': product}


def default_config(preset='archived-published'):
    cfg = {'schema': VERSION, 'preset': preset, 'years': [2022, 2023, 2024],
           'state_fips': sorted(STATES), 'grid_crs': 'EPSG:5070',
           'grid_transform': [30, 0, 0, 0, -30, 0],
           'county_asset': 'TIGER/2018/Counties', 'tile_scale': 4,
           'max_pixels_per_county': 2_000_000_000, 'mosaic_order':'LAST_VALID_WINS',
           'months': {}}
    for year in range(2022, 2026):
        cfg['months'][str(year)] = {}
        for month in MONTHS:
            if year == 2023:
                our = f'Our{month if month != "August" else "AUG"}2023_CornBelt_CornMask30m'
                if preset == 'archived-yield-recorded':
                    our = f'Corn_MODIS_2023_CLEAN_REBUILT_{month.upper()}'
            else:
                our = f'Our{month}{year}_CornBelt_CornMask30m_CLEAN_REBUILT'
            if year == 2022:
                pub = ([f'Inseason2022June10m_tile_0{i}_{d}'
                        for i, d in enumerate(('NW','NE','SW','SE'), 1)]
                       if month == 'June' else
                       [f'Inseason2022{"July" if month == "July" else "Aug"}10m_tile_01_NW'])
                product, nodata = 'PUBLISHED_ARCHIVED_10M', [255]
            elif year == 2023 and preset != 'latest-extraction':
                pub = (['Inseason2023June10m'] if month == 'June' else
                       [f'Inseason2023{"July" if month == "July" else "Aug"}10m_tile_01_NW'])
                product, nodata = 'PUBLISHED_ARCHIVED_10M', [255]
            elif year == 2023:
                pub = [f'CornBelt2023_{month}_CropSmart30m']
                product, nodata = 'CROPSMART_30M', [0, 255]
            else:
                pub = [f'{year}{month}CropMap30m']
                product, nodata = 'CROPSMART_30M', [0, 255]
            cfg['months'][str(year)][month] = {
                'OUR': spec([our], 'OUR_30M'),
                'PUBLISHED': spec(pub, product, nodata)}
    cfg['cdl'] = {str(y): {'ids': [f'USDA/NASS/CDL/{y}'], 'band': 'cropland',
                  'corn_code': 1, 'nodata': [0, 255], 'product':'USDA_NASS_ANNUAL_CDL'}
                  for y in range(2021, 2026)}
    cfg['cdl']['2025'] = spec(['CDL_2025_30m'], 'USER_UPLOADED_FINAL_CDL_2025_30M', [0, 255])
    return cfg


def validate_config(cfg):
    if cfg.get('schema') != VERSION:
        raise ValueError('Config schema/version mismatch.')
    if cfg['grid_crs'] != 'EPSG:5070':
        raise ValueError('Use EPSG:5070: grid area calculations require an equal-area CRS.')
    t = cfg['grid_transform']
    if len(t) != 6 or t[0] != 30 or t[4] != -30 or t[1] != 0 or t[3] != 0:
        raise ValueError('Expected north-up 30-m affine transform [30,0,x,0,-30,y].')
    if not cfg['years'] or len(set(cfg['years'])) != len(cfg['years']):
        raise ValueError('Years must be nonempty and unique.')
    if not cfg['state_fips'] or not set(cfg['state_fips']) <= set(STATES):
        raise ValueError('States must be selected from the 12 project states.')
    if len(set(cfg['state_fips'])) != len(cfg['state_fips']):
        raise ValueError('Duplicate states.')
    for year in cfg['years']:
        if year not in (2022,2023,2024,2025):
            raise ValueError('Supported years: 2022-2025.')
        for s in [cfg['cdl'][str(year)], cfg['cdl'][str(year-1)]] + [
                cfg['months'][str(year)][m][k] for m in MONTHS for k in ('OUR','PUBLISHED')]:
            if not s['ids'] or s['corn_code'] in s['nodata']:
                raise ValueError('Empty asset list or corn class also marked NoData.')
            if len(set(s['ids'])) != len(s['ids']):
                raise ValueError('Duplicate IDs within a source mosaic.')
    return cfg


def config_for(args):
    cfg = (json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
           if getattr(args, 'config', None) else default_config(args.preset))
    if getattr(args, 'years', None):
        cfg['years'] = sorted(args.years)
    if getattr(args, 'states', None):
        inverse = {v:k for k,v in STATES.items()}
        cfg['state_fips'] = sorted(inverse.get(s.upper(), s.zfill(2)) for s in args.states)
    return validate_config(cfg)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def count_bands():
    bands = ['reference_n', 'reference_corn', 'common_n', 'common_ref_corn']
    for mask in MASKS:
        bands += [f'pair_{mask}_{c}' for c in COUNTS]
        bands += [f'common_{mask}_{c}' for c in COUNTS]
    bands += [f'direct_{c}' for c in COUNTS]
    return bands


def ee_init(args):
    try:
        import ee
    except ImportError:
        raise SystemExit('Install dependencies: python -m pip install earthengine-api pandas numpy matplotlib')
    if getattr(args, 'authenticate', False):
        ee.Authenticate()
    try:
        ee.Initialize(project=args.project)
    except Exception as exc:
        raise SystemExit(f'GEE initialization failed: {exc}\nRun export with --authenticate, '
                         'and use --project if your enabled project is different.')
    return ee


def selected_specs(cfg):
    records = []
    for y in cfg['years']:
        for role, yy in [('REFERENCE', y), ('PREVIOUS_CDL', y-1)]:
            records.append((y, 'ANNUAL', role, cfg['cdl'][str(yy)]))
        for m in MONTHS:
            for role in ('OUR', 'PUBLISHED'):
                records.append((y,m,role,cfg['months'][str(y)][m][role]))
    return records


def preflight(ee, cfg, out):
    """Small metadata calls only; no map accuracy/area claims from metadata."""
    audit, cache = [], {}
    errors = []
    for year, month, role, s in selected_specs(cfg):
        for asset in s['ids']:
            key = (asset, str(s['band']))
            try:
                if key not in cache:
                    meta = ee.data.getAsset(asset)
                    if meta.get('type') != 'IMAGE':
                        raise ValueError(f'Expected IMAGE, got {meta.get("type")}')
                    im = ee.Image(asset).select([s['band']])
                    cache[key] = {'asset_type': meta.get('type'),
                                  'bands': im.bandNames().getInfo(),
                                  'projection': im.projection().getInfo()}
                audit.append({'year':year, 'month':month, 'role':role,
                              'asset':asset, 'class_encoding':s, **cache[key]})
            except Exception as exc:
                errors.append(f'{year} {month} {role}: {asset}: {exc}')
    write_json(out / 'asset_preflight.json', {'assets':audit, 'errors':errors})
    if errors:
        raise SystemExit('Preflight stopped; no tasks submitted.\n' + '\n'.join(errors) +
                         '\nCheck asset_preflight.json. Edit an explicit config if IDs differ; '
                         'the script does not guess substitutes.')
    counties = ee.FeatureCollection(cfg['county_asset']).filter(
        ee.Filter.inList('STATEFP', cfg['state_fips']))
    expected = counties.aggregate_array('GEOID').getInfo()
    write_json(out / 'expected_counties.json', sorted(expected))
    print(f'Preflight passed: {len(cache)} image/band combinations; {len(expected)} counties.')
    return expected


def class_map(ee, s, cfg):
    """Filter invalid codes BEFORE mosaic, preserve valid 0, then nearest grid."""
    images = []
    for asset in s['ids']:
        im = ee.Image(asset).select([s['band']], ['class'])
        for code in s['nodata']:
            im = im.updateMask(im.neq(code))
        # Reproject each source before mosaic: no composite's default WGS84 grid.
        im = im.reproject(crs=cfg['grid_crs'], crsTransform=cfg['grid_transform'])
        images.append(im)
    classes = images[0] if len(images) == 1 else ee.ImageCollection.fromImages(images).mosaic()
    # Extend to full plane ONLY for boolean coverage; never treat those cells as observations.
    valid = classes.mask().gt(0).unmask(0, False)
    corn = classes.eq(s['corn_code']).unmask(0, False)
    return corn, valid


def image_stack(ee, cfg, year, month):
    ref, rv = class_map(ee, cfg['cdl'][str(year)], cfg)
    candidates = {k:class_map(ee, cfg['months'][str(year)][month][k], cfg)
                  for k in ('OUR','PUBLISHED')}
    candidates['PREVIOUS_CDL'] = class_map(ee, cfg['cdl'][str(year-1)], cfg)
    common = rv
    for _, valid in candidates.values():
        common = common.And(valid)
    bands = [rv.rename('reference_n'), rv.And(ref).rename('reference_corn'),
             common.rename('common_n'), common.And(ref).rename('common_ref_corn')]

    def cells(pred, truth, domain, prefix):
        return [domain.And(pred).And(truth).rename(prefix+'_tp'),
                domain.And(pred).And(truth.Not()).rename(prefix+'_fp'),
                domain.And(pred.Not()).And(truth).rename(prefix+'_fn'),
                domain.And(pred.Not()).And(truth.Not()).rename(prefix+'_tn')]
    for role, (pred, valid) in candidates.items():
        bands += cells(pred, ref, rv.And(valid), 'pair_'+role)
        bands += cells(pred, ref, common, 'common_'+role)
    # Symmetric agreement: published map as the naming convention for truth.
    # This does not turn the published map into independent ground truth.
    bands += cells(candidates['OUR'][0], candidates['PUBLISHED'][0], common, 'direct')
    return ee.Image.cat(bands).toDouble()


def task_collection(ee, cfg, year, state, digest):
    counties = ee.FeatureCollection(cfg['county_asset']).filter(ee.Filter.eq('STATEFP',state))
    collections = []
    for month in MONTHS:
        stats = image_stack(ee,cfg,year,month).reduceRegions(
            collection=counties, reducer=ee.Reducer.sum().unweighted(),
            crs=cfg['grid_crs'], crsTransform=cfg['grid_transform'],
            tileScale=cfg['tile_scale'], maxPixelsPerRegion=cfg['max_pixels_per_county'])
        ms = cfg['months'][str(year)][month]
        final, prev = cfg['cdl'][str(year)], cfg['cdl'][str(year-1)]
        props = {'schema':VERSION, 'config_hash':digest, 'year':year, 'month':month,
                 'state_fips':state, 'grid_crs':cfg['grid_crs'],
                 'grid_transform':json.dumps(cfg['grid_transform']), 'pixel_km2':0.0009,
                 'preset':cfg['preset'], 'reference_asset':'|'.join(final['ids']),
                 'our_assets':'|'.join(ms['OUR']['ids']), 'published_assets':'|'.join(ms['PUBLISHED']['ids']),
                 'previous_asset':'|'.join(prev['ids']), 'our_product':ms['OUR']['product'],
                 'published_product':ms['PUBLISHED']['product'],
                 'reference_nodata':json.dumps(final['nodata']),
                 'our_nodata':json.dumps(ms['OUR']['nodata']),
                 'published_nodata':json.dumps(ms['PUBLISHED']['nodata']),
                 'previous_nodata':json.dumps(prev['nodata']),
                 'corn_codes':json.dumps({k:s['corn_code'] for k,s in
                     [('REFERENCE',final),('OUR',ms['OUR']),('PUBLISHED',ms['PUBLISHED']),('PREVIOUS_CDL',prev)]}),
                 'mosaic_order':cfg['mosaic_order']}

        def tidy(f):
            f = ee.Feature(f)
            return ee.Feature(None, f.toDictionary(count_bands())).set(props).set(
                {'county_fips':f.get('GEOID'), 'county_name':f.get('NAME')})
        collections.append(stats.map(tidy))
    return ee.FeatureCollection(collections).flatten()


def export_command(args):
    cfg = config_for(args)
    digest = config_hash(cfg)
    out = Path(args.out_dir or f'mask_benchmark_export_{now()}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise SystemExit('Existing folder has another configuration. Choose a new --out-dir.')
    write_json(config_path,cfg)
    ee = ee_init(args)
    preflight(ee,cfg,out)
    if args.preflight_only:
        print(f'No tasks submitted. Configuration and metadata: {out}')
        return
    drive = args.drive_folder or ('CornMask_Benchmark_' + str(min(cfg['years'])) +
             ('_' + str(max(cfg['years'])) if len(cfg['years']) > 1 else ''))
    record_path = out/'tasks.json'
    journal = json.loads(record_path.read_text()) if record_path.exists() else {}
    print('Legend assumptions: OUR valid 0=non-corn; published 10m 255=NoData; '
          'published 30m 0/255=NoData. Check asset_preflight.json and source legends.')
    for year in cfg['years']:
        for state in cfg['state_fips']:
            key = f'{year}_{state}'
            if key in journal:
                previous = ee.data.getTaskStatus(journal[key]['task_id'])[0]
                if previous.get('state') not in ('FAILED','CANCELLED'):
                    print(f'{key}: retained existing {previous.get("state")} task')
                    continue
            name = f'CMAP_{year}_{state}_{digest[:10]}'
            task = ee.batch.Export.table.toDrive(
                collection=task_collection(ee,cfg,year,state,digest),
                description=name, folder=drive, fileNamePrefix=name,
                fileFormat='CSV', selectors=META+count_bands())
            task.start()
            journal[key] = {'task_id':task.id, 'name':name, 'drive_folder':drive,
                            'config_hash':digest, 'submitted_utc':now()}
            write_json(record_path,journal)
            print(f'Submitted {name}: {task.id}',flush=True)
    print(f'\nTasks keep running on GEE. Local record: {out}\n'
          f'Download CMAP_*.csv from Drive folder {drive}, then run summarize.\n'
          'This script does not overwrite any yield-model inputs or fit a model.')


def ratio(a,b):
    return a/b if b else math.nan


def metrics(tp,fp,fn,tn):
    n, actual, pred = tp+fp+fn+tn, tp+fn, tp+fp
    recall, specificity = ratio(tp,actual),ratio(tn,tn+fp)
    return {'n':n, 'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'precision':ratio(tp,pred), 'recall':recall,
            'f1':ratio(2*tp,2*tp+fp+fn), 'iou':ratio(tp,tp+fp+fn),
            'accuracy':ratio(tp+tn,n), 'specificity':specificity,
            'balanced_accuracy':(recall+specificity)/2,
            'mcc':ratio(tp*tn-fp*fn, math.sqrt(pred*actual*(tn+fp)*(tn+fn))),
            'omission_rate':ratio(fn,actual), 'commission_rate':ratio(fp,pred),
            'corn_area_bias_pct':100*ratio(pred-actual,actual),
            'corn_area_error_km2':(pred-actual)*0.0009,
            'predicted_corn_km2':pred*0.0009, 'reference_corn_km2':actual*0.0009,
            'evaluated_area_km2':n*0.0009, 'reference_corn_fraction':ratio(actual,n)}


def validate_raw(df, expected_years=None, expected_states=None):
    import numpy as np
    missing = set(META+count_bands()) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if set(df.schema) != {VERSION} or df.config_hash.nunique() != 1:
        raise ValueError('Mixed schemas or config hashes. Keep each experiment in a separate folder.')
    df['year'] = df.year.astype(int)
    df['state_fips'] = df.state_fips.astype(str).str.zfill(2)
    df['county_fips'] = df.county_fips.astype(str).str.zfill(5)
    if df[['year','month','county_fips']].duplicated().any():
        raise ValueError('Duplicate county/year/month rows (possibly duplicate downloads). Remove duplicate CSVs.')
    if not set(df.state_fips) <= set(STATES) or not set(df.month) <= set(MONTHS):
        raise ValueError('Unknown state or month.')
    if not (df.county_fips.str[:2] == df.state_fips).all():
        raise ValueError('County/state FIPS mismatch.')
    if not df.year.isin([2022,2023,2024,2025]).all():
        raise ValueError('Unexpected year.')
    if not (df.grid_crs == 'EPSG:5070').all() or not np.allclose(df.pixel_km2.astype(float),0.0009):
        raise ValueError('Unexpected grid or pixel area.')
    for field in ['grid_transform','preset']:
        if df[field].nunique() != 1:
            raise ValueError(f'Mixed {field}.')
    constant_fields = [f for f in META if f not in ('year','month','state_fips','county_fips','county_name')]
    if (df.groupby(['year','month'])[constant_fields].nunique(dropna=False).to_numpy()>1).any():
        raise ValueError('Source/grid/encoding provenance differs within the same year/month.')
    for field in count_bands():
        df[field] = df[field].astype(float)
        x = df[field].to_numpy()
        if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x,np.rint(x),atol=1e-6,rtol=0):
            raise ValueError(f'Invalid/null/fractional count in {field}; do not fill with zero.')
    # Cross-check domain and confusion-matrix identities.
    tol = 1e-6
    if ((df.common_n > df.reference_n+tol) | (df.common_ref_corn > df.reference_corn+tol)).any():
        raise ValueError('Common domain exceeds reference domain.')
    for mask in MASKS:
        cp, pp = 'common_'+mask+'_','pair_'+mask+'_'
        if not np.allclose(df[[cp+c for c in COUNTS]].sum(axis=1),df.common_n,atol=tol,rtol=0):
            raise ValueError('Common-domain counts disagree across masks.')
        if not np.allclose(df[cp+'tp']+df[cp+'fn'],df.common_ref_corn,atol=tol,rtol=0):
            raise ValueError('Common reference corn differs by mask.')
        pn = df[[pp+c for c in COUNTS]].sum(axis=1)
        pc = df[pp+'tp']+df[pp+'fn']
        if ((pn>df.reference_n+tol)|(pn<df.common_n-tol)|
            (pc>df.reference_corn+tol)|(pc<df.common_ref_corn-tol)).any():
            raise ValueError('Pairwise coverage violates domain nesting.')
        for c in COUNTS:
            if (df[cp+c]>df[pp+c]+tol).any():
                raise ValueError('Common confusion cell exceeds corresponding pairwise cell.')
    if not np.allclose(df[['direct_'+c for c in COUNTS]].sum(axis=1),df.common_n,atol=tol,rtol=0):
        raise ValueError('Direct agreement does not use the common domain.')
    if not np.allclose(df.direct_tp+df.direct_fp,df.common_OUR_tp+df.common_OUR_fp,atol=tol,rtol=0):
        raise ValueError('Direct OUR corn total mismatch.')
    if not np.allclose(df.direct_tp+df.direct_fn,df.common_PUBLISHED_tp+df.common_PUBLISHED_fp,atol=tol,rtol=0):
        raise ValueError('Direct published corn total mismatch.')
    years = sorted(expected_years or df.year.unique().tolist())
    states = sorted(expected_states or STATES.keys())
    expected = {(y,m,s) for y in years for m in MONTHS for s in states}
    got = set(df[['year','month','state_fips']].itertuples(index=False,name=None))
    absent, extra = expected-got,got-expected
    if absent or extra:
        raise ValueError(f'Incomplete/unexpected state-year-month coverage. Missing: {sorted(absent)}; extra: {sorted(extra)}')
    for state, group in df.groupby('state_fips'):
        sets = [set(g.county_fips) for _,g in group.groupby(['year','month'])]
        if any(s != sets[0] for s in sets):
            raise ValueError(f'County inventory changes across months/years in state {state}.')
    # Require invariant final-CDL denominator across months for a county/year.
    g = df.groupby(['year','county_fips'])[['reference_n','reference_corn']].nunique()
    if (g.to_numpy()>1).any():
        raise ValueError('Final-CDL domain changes across months for the same county/year.')
    return df


def score_tables(df, level):
    import pandas as pd
    keys = ['year','month'] + (['state_fips','county_fips','county_name'] if level=='county'
                               else ['state_fips'] if level=='state' else [])
    data = df if level=='county' else df.groupby(keys,as_index=False)[count_bands()].sum()
    rows, agreements = [],[]
    for d in data.to_dict('records'):
        base = {k:d[k] for k in keys}
        base['level'] = level
        if 'state_fips' in base:
            base['state'] = STATES[base['state_fips']]
        for domain in ('COMMON','PAIRWISE'):
            for mask in MASKS:
                prefix = ('common_' if domain=='COMMON' else 'pair_')+mask+'_'
                r = {**base, 'domain':domain,'mask':mask,
                     **metrics(*(d[prefix+c] for c in COUNTS))}
                r['reference_full_area_km2'] = d['reference_n']*0.0009
                r['reference_full_corn_km2'] = d['reference_corn']*0.0009
                r['coverage_pct'] = 100*ratio(r['n'],d['reference_n'])
                r['reference_corn_coverage_pct'] = 100*ratio(r['tp']+r['fn'],d['reference_corn'])
                r['status'] = ('NO_VALID_PIXELS' if not r['n'] else
                               'NO_REFERENCE_CORN' if not r['tp']+r['fn'] else 'OK')
                rows.append(r)
        # No perfect self-score: the reference row is only an area baseline.
        rows.append({**base,'domain':'REFERENCE_FULL','mask':'FINAL_CDL_REFERENCE',
                     'n':d['reference_n'],'reference_full_area_km2':d['reference_n']*0.0009,
                     'reference_full_corn_km2':d['reference_corn']*0.0009,
                     'reference_corn_km2':d['reference_corn']*0.0009,
                     'coverage_pct':100 if d['reference_n'] else math.nan,
                     'status':'REFERENCE_ONLY_NO_SELF_ACCURACY'})
        a = {**base, 'domain':'COMMON','n':d['common_n'],
             'agreement':ratio(d['direct_tp']+d['direct_tn'],d['common_n']),
             'corn_dice':ratio(2*d['direct_tp'],2*d['direct_tp']+d['direct_fp']+d['direct_fn']),
             'corn_iou':ratio(d['direct_tp'],d['direct_tp']+d['direct_fp']+d['direct_fn']),
             'disagreement_km2':(d['direct_fp']+d['direct_fn'])*0.0009,
             'our_only_corn_km2':d['direct_fp']*0.0009,
             'published_only_corn_km2':d['direct_fn']*0.0009}
        agreements.append(a)
    return pd.DataFrame(rows),pd.DataFrame(agreements)


def paired_table(table):
    import pandas as pd
    df = table[table.domain=='COMMON']
    keys = [k for k in ['level','year','month','state_fips','state','county_fips','county_name'] if k in df]
    rows=[]
    for a,b in [('OUR','PUBLISHED'),('OUR','PREVIOUS_CDL'),('PUBLISHED','PREVIOUS_CDL')]:
        merge = df[df['mask']==a].merge(df[df['mask']==b],on=keys,suffixes=('_a','_b'),validate='one_to_one')
        for d in merge.to_dict('records'):
            r = {k:d[k] for k in keys}
            r.update({'mask_a':a,'mask_b':b,'n':d['n_a']})
            for m in ['f1','precision','recall','iou','mcc','balanced_accuracy','corn_area_bias_pct']:
                r[m+'_a_minus_b'] = d[m+'_a']-d[m+'_b']
            r['absolute_bias_pct_a_minus_b'] = abs(d['corn_area_bias_pct_a'])-abs(d['corn_area_bias_pct_b'])
            rows.append(r)
    return pd.DataFrame(rows)


def area_scores(county):
    """County-area R2 is distinct from pixel F1 and the project's yield R2."""
    import numpy as np
    import pandas as pd
    rows=[]
    for (year,month,mask),g in county[county.domain=='COMMON'].groupby(['year','month','mask']):
        # Drop undefined, zero-common-domain county rows. The count is explicit.
        g = g[g.n>0]
        y,p = g.reference_corn_km2.to_numpy(),g.predicted_corn_km2.to_numpy()
        n=len(g)
        sse=float(np.square(p-y).sum())
        sst=float(np.square(y-y.mean()).sum()) if n else 0
        rows.append({'year':year,'month':month,'mask':mask,'county_n':n,
                     'county_corn_area_r2':1-sse/sst if n>=2 and sst else math.nan,
                     'county_corn_area_rmse_km2':math.sqrt(sse/n) if n else math.nan,
                     'county_corn_area_mae_km2':float(np.abs(p-y).mean()) if n else math.nan,
                     'warning':'AREA_R2_IS_NOT_YIELD_R2_OR_PIXEL_ACCURACY'})
    return pd.DataFrame(rows)


def make_figures(pooled,states,out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'OUR':'Our ICDL','PUBLISHED':'Published ICDL','PREVIOUS_CDL':'Previous-year CDL'}
    colors=REPORT_COLORS
    d=pooled[pooled.domain=='COMMON']
    years=sorted(d.year.unique())
    fig,axes=plt.subplots(2,len(years),figsize=(4.0*len(years),7),squeeze=False)
    for j,y in enumerate(years):
        for mask in MASKS:
            g=d[(d.year==y)&(d['mask']==mask)].set_index('month').reindex(MONTHS)
            for i,metric in enumerate(['f1','corn_area_bias_pct']):
                axes[i,j].plot(MONTHS,g[metric],'-o',color=colors[mask],label=labels[mask])
        axes[0,j].set_title(str(y))
        axes[0,j].set_ylim(0,1)
        axes[1,j].axhline(0,color='black',lw=.7)
        for ax in axes[:,j]:
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Corn F1 versus final CDL')
    axes[1,0].set_ylabel('Corn area bias versus final CDL (%)')
    handles,labs=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labs,loc='upper center',ncol=3,bbox_to_anchor=(.5,.96))
    fig.suptitle('Corn-mask agreement on the common valid footprint',fontweight='bold')
    fig.text(.5,.012,'Final annual CDL is the reference. Curves aggregate confusion counts across counties; coverage is reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.90))
    for ext in ('png','pdf'):
        fig.savefig(out/f'pooled_F1_and_area_bias.{ext}',dpi=300)
    plt.close(fig)
    fig,axes=plt.subplots(1,len(years),figsize=(4.2*len(years),4.5),squeeze=False)
    for j,y in enumerate(years):
        g=d[(d.year==y)&(d['mask']=='OUR')].set_index('month').reindex(MONTHS)
        axes[0,j].plot(MONTHS,g.coverage_pct,'-o',color=colors['PREVIOUS_CDL'],label='All reference pixels')
        axes[0,j].plot(MONTHS,g.reference_corn_coverage_pct,'-s',color=colors['FINAL_CDL'],label='Reference corn pixels')
        axes[0,j].set(title=str(y),ylim=(0,101))
        axes[0,j].grid(alpha=.2)
        axes[0,j].spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Common footprint coverage (%)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Coverage of the shared evaluation footprint',fontweight='bold')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out/f'common_coverage.{ext}',dpi=300)
    plt.close(fig)
    paired=paired_table(states)
    p=paired[(paired.mask_a=='OUR')&(paired.mask_b=='PUBLISHED')].copy()
    columns=[(y,m) for y in years for m in MONTHS]
    order=sorted(p.state.unique())
    data=np.full((len(order),len(columns)),np.nan)
    for i,s in enumerate(order):
        for j,(y,m) in enumerate(columns):
            x=p[(p.state==s)&(p.year==y)&(p.month==m)]
            if len(x): data[i,j]=x.f1_a_minus_b.iloc[0]
    v=float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else .01
    v=max(v,.001)
    fig,ax=plt.subplots(figsize=(max(8,len(columns)*.8),6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap=LinearSegmentedColormap.from_list('published_neutral_our',
             [colors['PUBLISHED'],'#ffffff',colors['OUR']])
    im=ax.imshow(np.ma.masked_invalid(data),cmap=cmap,vmin=-v,vmax=v,aspect='auto')
    ax.set_xticks(range(len(columns)),[f'{y}\n{m[:3]}' for y,m in columns],fontsize=8)
    ax.set_yticks(range(len(order)),order)
    for i in range(len(order)):
        for j in range(len(columns)):
            if math.isfinite(data[i,j]):
                ax.text(j,i,f'{data[i,j]:+.3f}',ha='center',va='center',fontsize=7,
                        color='white' if abs(data[i,j])>.65*v else 'black')
    ax.set_title('State corn F1 difference: our ICDL minus published ICDL\nOrange favors our ICDL; purple favors published ICDL; common valid footprint')
    fig.colorbar(im,ax=ax,label='F1 difference')
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out/f'state_F1_difference.{ext}',dpi=300)
    plt.close(fig)
    make_corn_area_figure(pooled,out)


def monthly_corn_area_table(pooled):
    """Return comparable map areas plus an explicitly matched CDL reference.

    Full-reference area rows are deliberately excluded from the plotted
    baseline: using those with partial candidate coverage would confound
    missing coverage and classification differences.
    """
    import numpy as np
    import pandas as pd
    needed={'year','month','domain','mask','n','predicted_corn_km2',
            'reference_corn_km2','corn_area_bias_pct','coverage_pct',
            'reference_corn_coverage_pct','reference_full_corn_km2'}
    if needed-set(pooled.columns):
        raise ValueError('Monthly area comparison needs columns: '+str(sorted(needed-set(pooled.columns))))
    d=pooled.loc[pooled.domain.eq('COMMON') & pooled['mask'].isin(MASKS)].copy()
    if d.empty or d.duplicated(['year','month','mask']).any():
        raise ValueError('Missing or duplicate common-domain pooled score rows.')
    rows=[]
    years=sorted(d.year.unique())
    for year in years:
        for month in MONTHS:
            g=d.loc[d.year.eq(year)&d.month.eq(month)].set_index('mask')
            if set(g.index)!=set(MASKS):
                raise ValueError(f'Incomplete area comparison: {year} {month}.')
            for field in ['n','reference_corn_km2','coverage_pct',
                          'reference_corn_coverage_pct','reference_full_corn_km2']:
                values=g[field].to_numpy(dtype=float)
                if not np.isfinite(values).all() or not np.allclose(values,values[0],rtol=1e-10,atol=1e-8):
                    raise ValueError(f'Area reference/domain differs across masks: {year} {month} {field}.')
            base=g.loc['OUR']
            for mask in MASKS+('FINAL_CDL',):
                pred=float(g.loc[mask,'predicted_corn_km2']) if mask!='FINAL_CDL' else float(base.reference_corn_km2)
                ref=float(base.reference_corn_km2)
                if not math.isfinite(pred) or pred<0 or ref<0:
                    raise ValueError(f'Invalid mapped area: {year} {month} {mask}.')
                bias=100*ratio(pred-ref,ref)
                if mask!='FINAL_CDL' and ref>0:
                    if not math.isclose(bias,float(g.loc[mask,'corn_area_bias_pct']),rel_tol=1e-8,abs_tol=1e-7):
                        raise ValueError(f'Area bias does not reconcile: {year} {month} {mask}.')
                rows.append({'year':int(year),'month':month,'mask':mask,
                    'role':'REFERENCE_ONLY' if mask=='FINAL_CDL' else 'PREDICTED_MAP',
                    'domain':'COMMON','corn_area_km2':pred,'final_cdl_corn_km2_same_domain':ref,
                    'area_difference_km2':pred-ref,'corn_area_bias_pct':bias,
                    'area_pct_of_final_cdl':100*ratio(pred,ref),'n':base['n'],
                    'coverage_pct':base.coverage_pct,
                    'reference_corn_coverage_pct':base.reference_corn_coverage_pct,
                    'final_cdl_full_reference_corn_km2':base.reference_full_corn_km2,
                    'scope':'SHARED_VALID_FOOTPRINT_NOT_EXTRAPOLATED_TO_FULL_REGION'})
    return pd.DataFrame(rows)


def make_corn_area_figure(pooled,out):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    area=monthly_corn_area_table(pooled)
    area.to_csv(out/'corn_area_by_month_vs_final_CDL.csv',index=False)
    years=sorted(area.year.unique())
    ncols=min(2,len(years));nrows=math.ceil(len(years)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(5.9*ncols,3.7*nrows+1.0),squeeze=False)
    labels={'OUR':'Our ICDL','PUBLISHED':'Published ICDL',
            'PREVIOUS_CDL':'Previous-year CDL','FINAL_CDL':'Final CDL reference'}
    width=.19;x=np.arange(len(MONTHS))
    for ax,year in zip(axes.flat,years):
        d=area.loc[area.year.eq(year)]
        for j,mask in enumerate(MASKS+('FINAL_CDL',)):
            g=d.loc[d['mask'].eq(mask)].set_index('month').reindex(MONTHS)
            ax.bar(x+(j-1.5)*width,g.corn_area_km2,width,
                   color=REPORT_COLORS[mask],label=labels[mask])
        coverage=d.loc[d['mask'].eq('OUR')].set_index('month').reindex(MONTHS)
        ax.set_xticks(x,[f'{m}\nCorn coverage: {coverage.loc[m,"reference_corn_coverage_pct"]:.1f}%'
                        for m in MONTHS],fontsize=8)
        ax.set_title(str(year),fontweight='bold')
        ax.set_ylabel('Mapped corn area (km²)')
        max_area=float(d.corn_area_km2.max())
        decimals=2 if max_area<10 else 1 if max_area<100 else 0
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos,digits=decimals:f'{value:,.{digits}f}'))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    for ax in list(axes.flat)[len(years):]:ax.set_visible(False)
    handles,legend=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,.95),frameon=False,fontsize=9)
    fig.suptitle('June, July, and August corn maps versus final annual CDL',fontweight='bold',y=.99)
    fig.text(.5,.015,'All four bars use the same valid pixels within each year and month.\n'
             'Coverage is relative to all final-CDL corn pixels; footprints can change between months. No full-region extrapolation.',
             ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.065,1,.93),h_pad=2.0)
    for ext in ('png','pdf'):
        fig.savefig(out/f'corn_area_by_month_vs_final_CDL.{ext}',dpi=300)
    plt.close(fig)


def plots_only_command(args):
    """Read saved scores; preserve source tables and old figures by default."""
    import pandas as pd
    folder=Path(args.results_dir).resolve()
    paths=[folder/'pooled_mask_metrics.csv',folder/'state_mask_metrics.csv']
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing saved result tables: '+'; '.join(missing))
    pooled=pd.read_csv(paths[0],dtype={'state_fips':str})
    states=pd.read_csv(paths[1],dtype={'state_fips':str})
    needed={'year','month','domain','mask','f1','corn_area_bias_pct','coverage_pct',
            'reference_corn_coverage_pct'}
    for p,data in zip(paths,[pooled,states]):
        if needed-set(data.columns):
            raise ValueError(f'{p.name} is missing expected metric columns.')
        if not set(data.month)<=set(MONTHS):
            raise ValueError(f'Unexpected month in {p.name}.')
    if not {'state','state_fips'}<=set(states.columns):
        raise ValueError('State score table is missing state identifiers.')
    # Verify the common-domain area/bias identities BEFORE writing figures.
    area=monthly_corn_area_table(pooled)
    out=Path(args.out_dir).resolve() if args.out_dir else folder/f'figures_updated_{now()}'
    out.mkdir(parents=True,exist_ok=False)
    make_figures(pooled,states,out)
    write_json(out/'figure_manifest.json',{
        'plot_version':PLOT_VERSION,'analysis_schema':VERSION,'colors':REPORT_COLORS,
        'source_results_folder':str(folder),'source_files':[
            {'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'years':sorted(int(y) for y in area.year.unique()),
        'earth_engine_called':False,'classification_counts_changed':False,
        'corn_area_domain':'COMMON_SHARED_VALID_PIXELS',
        'new_figures':[p.name for p in sorted(out.glob('*.png'))]})
    print('Updated figures use OUR orange, PUBLISHED purple, FINAL_CDL green, PREVIOUS_CDL gray.')
    print(f'Saved eight figure files (four PNG + four PDF) and monthly area CSV: {out}')
    print('Read corn_area_by_month_vs_final_CDL.csv for km², signed area error, percent bias, and coverage.')
    print('Input scores and previous figure files are preserved. No GEE exports or model runs were performed.')


def summarize_command(args):
    import numpy as np
    import pandas as pd
    folder=Path(args.csv_dir).resolve()
    files=sorted(folder.rglob('CMAP_*.csv'))
    if not files:
        raise SystemExit(f'No CMAP_*.csv exports found under {folder}. Download GEE Drive exports first.')
    frames=[pd.read_csv(p,dtype={'county_fips':str,'state_fips':str}) for p in files]
    raw=pd.concat(frames,ignore_index=True)
    selected_states = None
    if args.states:
        inverse={v:k for k,v in STATES.items()}
        selected_states=sorted(inverse.get(s.upper(),s.zfill(2)) for s in args.states)
    expected_path = (Path(args.export_dir)/'expected_counties.json') if args.export_dir else None
    config = json.loads((Path(args.export_dir)/'config.json').read_text()) if args.export_dir else None
    expected_years = args.years or (config['years'] if config else [2022,2023,2024])
    selected_states = selected_states or (config['state_fips'] if config else sorted(STATES))
    raw=validate_raw(raw,expected_years,selected_states)
    if expected_path:
        expected=set(json.loads(expected_path.read_text()))
        if set(raw.county_fips)!=expected:
            raise ValueError('County inventory differs from export preflight. Check missing/truncated CSVs.')
        if config_hash(config)!=raw.config_hash.iloc[0]:
            raise ValueError('CSV configuration does not match --export-dir.')
    out=Path(args.out_dir or f'mask_benchmark_results_{now()}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    tables={}
    for level in ('county','state','pooled'):
        table,agreement=score_tables(raw,level)
        tables[level]=table
        table.to_csv(out/f'{level}_mask_metrics.csv',index=False)
        agreement.to_csv(out/f'{level}_our_vs_published_agreement.csv',index=False)
        paired_table(table).to_csv(out/f'{level}_paired_metric_differences.csv',index=False)
    area_scores(tables['county']).to_csv(out/'county_corn_area_skill.csv',index=False)
    county=tables['county']
    flags=[]
    for r in county[county['mask']!='FINAL_CDL_REFERENCE'].to_dict('records'):
        base={k:r[k] for k in ['year','month','county_fips','state','domain','mask']}
        for field in ('coverage_pct','reference_corn_coverage_pct'):
            if math.isfinite(r[field]) and r[field]<args.coverage_warning:
                flags.append({**base,'flag':field+'_BELOW_THRESHOLD','value':r[field]})
        if r['status']!='OK':
            flags.append({**base,'flag':r['status'],'value':r['n']})
        if r['n'] and r['tn']+r['fn']==0:
            flags.append({**base,'flag':'NO_VALID_NEGATIVE_PREDICTIONS_CHECK_LEGEND_OR_SELF_MASKED_SOURCE','value':r['n']})
    pd.DataFrame(flags,columns=['year','month','county_fips','state','domain','mask','flag','value']).to_csv(out/'coverage_and_encoding_flags.csv',index=False)
    # Raw confusion counts retained for independent verification.
    raw.to_csv(out/'validated_raw_counts.csv.gz',index=False,compression='gzip')
    provenance=raw[META].drop(columns=['county_fips','county_name']).drop_duplicates()
    provenance.to_csv(out/'source_provenance.csv',index=False)
    # State groups, NOT billions of correlated raster pixels, are shown in the
    # paired table; no unwarranted p-values/equivalence claims are generated.
    make_figures(tables['pooled'],tables['state'],out)
    stats={'schema':VERSION,'config_hash':raw.config_hash.iloc[0],
           'years':sorted(int(y) for y in raw.year.unique()),
           'states':sorted(raw.state_fips.unique()),'county_count':int(raw.county_fips.nunique()),
           'raw_rows':len(raw),'files':[],'preflight_county_inventory_verified':bool(expected_path),
           'warning_threshold_pct':args.coverage_warning,'warning_rows':len(flags)}
    for p in files:
        stats['files'].append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    write_json(out/'analysis_manifest.json',stats)
    text='''INTERPRETATION AND REPORTING

Primary map comparison: *_mask_metrics.csv, domain COMMON. F1, precision,
recall and IoU compare to final annual CDL on the identical footprint across
OUR/PUBLISHED/PREVIOUS_CDL within each county/year/month. State and pooled
scores SUM confusion counts first; they do not average county F1 scores.

Final CDL reference rows report full reference corn/grid area; accuracy/F1
are deliberately blank. Reference area is not harvested acreage or NASS
acreage, and CDL itself can be wrong. These are CDL agreement metrics, not
independent ground-reference validation or a reproduction of Li et al.'s F1.

COMMON can change by month/year. PAIRWISE is a coverage sensitivity analysis
using each candidate's own valid footprint. Never rank PAIRWISE scores as if
they used identical pixels. Compare coverage and reference-corn coverage.
Missing is not non-corn. Low coverage is flagged, not silently extrapolated.
If one map is valid only where corn=1 (a self-masked mask), recall/area bias
are not meaningful over the absent negative domain. Use complete categorical
source rasters, with explicit legend and valid-zero encoding, instead.

corn_area_bias_pct = 100*(predicted corn - reference corn)/reference corn on
the SAME domain. Positive=overmapping; negative=undermapping. Bias can cancel
even when spatial agreement is poor. The original full-map corn acreage is
not inferred outside valid coverage. F1=2TP/(2TP+FP+FN); IoU=TP/(TP+FP+FN).
Precision=TP/(TP+FP); recall=TP/(TP+FN). Omission=1-recall; commission=1-precision.
MCC and balanced accuracy help when non-corn dominates ordinary accuracy.
Undefined zero-denominator scores remain blank, never fabricated as 0 or 1.

county_corn_area_skill.csv compares county grid corn AREA, using common
pixels per county. Its R2 is 1-SSE/SST, not squared correlation and NOT yield
R2. Pixel classification results complement, and do not replace, the yield
forecasting validation specification R2 >= 0.80 at DOY 241.

direct OUR/PUBLISHED Dice/IoU and disagreement area quantify product
differences without treating either product as ground truth. Differences in
native resolution, existing upload pyramids, training/reference information,
registration and source coverage may all contribute; standardizing the
evaluation grid does not remove these sources of error.

No formal significance/equivalence claim is made. Adjacent pixels are not
independent replicates. Repeated previous-year CDL rows across months are a
control evaluated on changing COMMON footprints, not independent seasons.
Historical CDL also informs classifier training labels. Shared source errors
can increase agreement; this benchmark is not an independent field-label test.

Inspect source_provenance.csv and coverage_and_encoding_flags.csv before
writing conclusions. No full-state claim is justified by poor shared
coverage, especially if an asset name suggests a partial tile. The supplied
tile_01_NW names are retained exactly; their coverage is measured, not assumed.
All sources are nearest-neighbor evaluated at 30 m EPSG:5070 on the declared
grid with county pixel-center inclusion. Grid-cell area is 0.0009 km2.

Source documentation:
https://developers.google.com/earth-engine/guides/resample
https://developers.google.com/earth-engine/guides/reducers_reduce_region
https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
https://www.nature.com/articles/s41597-026-07099-1
https://doi.org/10.1109/TGRS.2024.3361895
'''
    if not expected_path:
        text+='\nCounty set is internally consistent across months/years, but completeness\nagainst the GEE county inventory was NOT verified. Rerun with --export-dir\npointing to the local export folder to establish that extra check.\n'
    (out/'README_RESULTS.txt').write_text(text,encoding='utf-8')
    display=tables['pooled']
    display=display[display.domain=='COMMON'][['year','month','mask','precision','recall','f1','iou','mcc','corn_area_bias_pct','coverage_pct','reference_corn_coverage_pct']]
    print(display.round(4).to_string(index=False))
    print(f'\nSaved tables and eight figure files: {out}\n'
          f'{len(flags)} county/domain warnings recorded; review coverage before interpreting F1.')


def self_test():
    import numpy as np
    # Eight independent hand-countable observations: TP=2,FP=1,FN=1,TN=4.
    truth=[1,1,1,0,0,0,0,0]
    pred =[1,1,0,1,0,0,0,0]
    tp=sum(a==1 and b==1 for a,b in zip(truth,pred))
    fp=sum(a==0 and b==1 for a,b in zip(truth,pred))
    fn=sum(a==1 and b==0 for a,b in zip(truth,pred))
    tn=sum(a==0 and b==0 for a,b in zip(truth,pred))
    r=metrics(tp,fp,fn,tn)
    expected={'f1':2/3,'precision':2/3,'recall':2/3,'iou':.5,'accuracy':.75,
              'corn_area_bias_pct':0.,'balanced_accuracy':(2/3+4/5)/2,
              'predicted_corn_km2':.0027,'n':8}
    for k,v in expected.items():
        assert math.isclose(r[k],v,rel_tol=1e-12), (k,r[k],v)
    assert math.isnan(metrics(0,0,0,0)['f1'])
    assert metrics(0,0,3,4)['f1']==0
    assert metrics(2,0,0,4)['f1']==1
    assert math.isnan(metrics(0,0,0,4)['recall'])
    assert metrics(0,3,0,4)['f1']==0
    assert math.isclose(metrics(2,0,1,4)['corn_area_bias_pct'],-100/3)
    # Missing cells excluded, never treated as non-corn.
    valid=np.array([1,1,0,1,1,1,1,1],dtype=bool)
    a=np.array(truth)[valid];b=np.array(pred)[valid]
    assert int(np.sum((a==1)&(b==0)))==0
    assert len(a)==7
    for preset in ['archived-published','archived-yield-recorded','latest-extraction']:
        c=validate_config(default_config(preset))
        assert config_hash(c)==config_hash(json.loads(json.dumps(c)))
    assert len(count_bands())==32
    print('PASS: synthetic confusion counts, zero denominators, signed area bias, '
          'NoData exclusion, and all config presets. Real GEE jobs not executed here.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    def options(q):
        q.add_argument('--config',help='Explicit edited JSON config, from the config command.')
        q.add_argument('--preset',choices=['archived-published','archived-yield-recorded','latest-extraction'],default='archived-published')
        q.add_argument('--years',type=int,nargs='+',help='Default 2022 2023 2024; 2025 is optional.')
        q.add_argument('--states',nargs='+',help='State abbreviations or FIPS. Default all 12.')
    q=sub.add_parser('config',help='Write editable configuration without logging into GEE.')
    options(q)
    q.add_argument('--out',default='corn_mask_benchmark_config.json')
    q=sub.add_parser('export',help='Preflight assets, then submit state-partitioned GEE exports.')
    options(q)
    q.add_argument('--project',default='ee-gtellezgiron')
    q.add_argument('--authenticate',action='store_true')
    q.add_argument('--preflight-only',action='store_true')
    q.add_argument('--drive-folder')
    q.add_argument('--out-dir',help='Local job/config folder. Reuse exact folder to resume.')
    q=sub.add_parser('status',help='Read GEE states for tasks in a local export folder.')
    q.add_argument('--export-dir',required=True)
    q.add_argument('--project',default='ee-gtellezgiron')
    q.add_argument('--authenticate',action='store_true')
    q=sub.add_parser('summarize',help='Analyze downloaded CMAP CSVs; no GEE login needed.')
    q.add_argument('--csv-dir',required=True)
    q.add_argument('--export-dir',help='Original local export folder, for exact inventory/config check.')
    q.add_argument('--years',nargs='+',type=int,help='Expected years; default 2022-2024, or export config if provided.')
    q.add_argument('--states',nargs='+',help='Expected states; default all 12. Does not filter CSV rows.')
    q.add_argument('--out-dir',help='New results folder (must not already exist).')
    q.add_argument('--coverage-warning',type=float,default=95.,help='Flag coverage below this percent.')
    q=sub.add_parser('plots-only',help='Remake report-color figures and monthly corn-area comparison from saved results; no GEE.')
    q.add_argument('--results-dir',required=True,help='Existing mask_benchmark_results folder containing pooled/state CSV tables.')
    q.add_argument('--out-dir',help='New output folder; default figures_updated_TIMESTAMP inside the results folder.')
    sub.add_parser('self-test',help='Run synthetic checks without logging into GEE.')
    args=p.parse_args()
    if args.command=='config':
        cfg=config_for(args)
        write_json(args.out,cfg)
        print(f'Editable config saved: {Path(args.out).resolve()}')
    elif args.command=='export':
        export_command(args)
    elif args.command=='status':
        ee=ee_init(args)
        journal=json.loads((Path(args.export_dir)/'tasks.json').read_text())
        for key,r in sorted(journal.items()):
            s=ee.data.getTaskStatus(r['task_id'])[0]
            print(key,s.get('state'),s.get('error_message',''))
    elif args.command=='summarize':
        if not 0<=args.coverage_warning<=100:
            raise ValueError('--coverage-warning must be between 0 and 100.')
        summarize_command(args)
    elif args.command=='plots-only':
        plots_only_command(args)
    else:
        self_test()


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,FileNotFoundError) as exc:
        raise SystemExit(f'TEST STOPPED: {exc}')
