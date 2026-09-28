// CORN YIELD 2 — our ICDLs (2022, 2023, 2025) and published ICDLs (2022, 2023).
// GEE JAVASCRIPT packaged as .py for downloading. NOT a Python executable.
// Open as text and paste into the Earth Engine Code Editor.
// Open as text and paste the ENTIRE file into https://code.earthengine.google.com/.
// READY TO PASTE: full run creates 15 model CSVs + 3 temporal audits.
// To export ONLY the newly uploaded 2025 masks, set YEARS = [2025] below.
// A 2025-only run creates 3 model CSVs + 1 temporal audit. Start tasks manually.
// V3 FIX: omit null county means; write zero counts when no valid pixels exist.
// Retains asynchronous asset checks and explicit Image casts after copyProperties.
// Model columns and corrected temporal policy match the previous exporter.
// Source docs: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1
// Source docs: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1
// Keeps the completed-MODIS-period policy used by the corrected exporter.
// Historical rows are season-to-date inputs, NOT permission to use a mask early.
// Python V4 must select June at DOY 193/209, July at 225/241, August at 257/273.
// Month-end mask availability is an experiment assumption; actual release dates
// and the information used to train each ICDL must be checked separately.

var YEARS = [2022, 2023, 2025];
// Explicit source selection prevents requesting nonexistent published 2025 assets.
var SOURCES_BY_YEAR = {
  2022: ['CLEAN_REBUILT', 'STUDY'],
  2023: ['CLEAN_REBUILT', 'STUDY'],
  2025: ['CLEAN_REBUILT']
};
var MONTHS = ['June', 'July', 'August'];
var ASSET_ROOT = 'projects/ee-gtellezgiron/assets/';
var DRIVE_FOLDER = 'CornYield_ICDL_COMPLETED_PERIODS_CONFIRMED';

// Confirmed aggregate asset IDs recovered from the original comparison script.
// Uses the aggregate masks below for the selected sources and years.
// The separate 36-state 2023 folder is a different input set.
// CSV labels remain CLEAN_REBUILT (ours) and STUDY (published) for Python V4.
var OUR_ASSETS = {
  2022: {
    June: [ASSET_ROOT + 'OurJune2022_CornBelt_CornMask30m_CLEAN_REBUILT'],
    July: [ASSET_ROOT + 'OurJuly2022_CornBelt_CornMask30m_CLEAN_REBUILT'],
    August: [ASSET_ROOT + 'OurAugust2022_CornBelt_CornMask30m_CLEAN_REBUILT']
  },
  2023: {
    June: [ASSET_ROOT + 'OurJune2023_CornBelt_CornMask30m'],
    July: [ASSET_ROOT + 'OurJuly2023_CornBelt_CornMask30m'],
    August: [ASSET_ROOT + 'OurAUG2023_CornBelt_CornMask30m']
  },
  2025: {
    June: [ASSET_ROOT + 'OurJune2025_CornBelt_CornMask30m_CLEAN_REBUILT'],
    July: [ASSET_ROOT + 'OurJuly2025_CornBelt_CornMask30m_CLEAN_REBUILT'],
    August: [ASSET_ROOT + 'OurAugust2025_CornBelt_CornMask30m_CLEAN_REBUILT']
  }
};

// Each array here lists ALL required tiles, not alternative names.
// Use July/August tile_01_NW exactly as supplied; do not invent other tiles.
var PUBLISHED_ASSETS = {
  2022: {
    June: [ASSET_ROOT + 'Inseason2022June10m_tile_01_NW',
           ASSET_ROOT + 'Inseason2022June10m_tile_02_NE',
           ASSET_ROOT + 'Inseason2022June10m_tile_03_SW',
           ASSET_ROOT + 'Inseason2022June10m_tile_04_SE'],
    July: [ASSET_ROOT + 'Inseason2022July10m_tile_01_NW'],
    August: [ASSET_ROOT + 'Inseason2022Aug10m_tile_01_NW']
  },
  2023: {
    June: [ASSET_ROOT + 'Inseason2023June10m'],
    July: [ASSET_ROOT + 'Inseason2023July10m_tile_01_NW'],
    August: [ASSET_ROOT + 'Inseason2023Aug10m_tile_01_NW']
  }
};

// Our masks: 1=corn, 0=non-corn, 255=NoData.
// Published masks: ASSUMES corn class 1; verify against the downloaded legend.
var OUR_CORN_CODE = 1;
var PUBLISHED_CORN_CODE = 1;
var NODATA_CODES = [255];

// Optional actual availability dates, keyed by output family/year/month.
// Example: {'STUDY_2022_June': '2022-07-15'}.
// Defaults: July 1 for June, August 1 for July, September 1 for August.
// These annotate exports only. If changed, update the Python deployment gates too.
var MASK_AVAILABLE_ON = {};

// Check configuration locally; validate image access asynchronously below.
function requireAssetId(id) {
  if (typeof id !== 'string' || !id.length) {
    throw new Error('Asset ID must be a nonempty string.');
  }
  return id;
}
function resolveOurAsset(candidates, label) {
  if (!candidates || candidates.length !== 1) {
    throw new Error(label + ': supply exactly one confirmed aggregate asset ID.');
  }
  return requireAssetId(candidates[0]);
}
function prepareRuns() {
  var runs = [];
  var failures = [];
  YEARS.forEach(function(year) {
    var families = SOURCES_BY_YEAR[year];
    if (!families || !families.length) {
      failures.push('No sources configured for year ' + year);
      return;
    }
    MONTHS.forEach(function(month, i) {
      var defaultAfter = year + (i === 0 ? '-07-01' : i === 1 ? '-08-01' : '-09-01');
      families.forEach(function(family) {
        var label = family + '_' + year + '_' + month;
        try {
          var ids = family === 'CLEAN_REBUILT' ?
            [resolveOurAsset(OUR_ASSETS[year][month], label)] :
            PUBLISHED_ASSETS[year][month].map(requireAssetId);
          var after = MASK_AVAILABLE_ON[label] || defaultAfter;
          if (!/^\d{4}-\d{2}-\d{2}$/.test(after) || after < defaultAfter) {
            throw new Error('Availability must be YYYY-MM-DD and no earlier than ' + defaultAfter);
          }
          runs.push({year:year, month:month, family:family, ids:ids, after:after,
            cornCode:family === 'CLEAN_REBUILT' ? OUR_CORN_CODE : PUBLISHED_CORN_CODE,
            name:'Corn_MODIS_' + year + '_' + family + '_' + month.toUpperCase()});
        } catch (error) { failures.push(label + ': ' + String(error)); }
      });
    });
  });
  if (failures.length) {
    failures.forEach(function(message) { print('ASSET / CONFIGURATION ERROR', message); });
    throw new Error('No export tasks created. Fix the asset/configuration errors above, then rerun.');
  }
  return runs;
}
function cornMaskFromRun(run) {
  var images = run.ids.map(function(id) {
    var classification = ee.Image(id).select([0], ['classification']);
    NODATA_CODES.forEach(function(value) {
      classification = classification.updateMask(classification.neq(value));
    });
    // Preserve valid non-corn pixels BEFORE mosaicking. Only NoData is masked.
    return classification;
  });
  var classes = images.length === 1 ? images[0] : ee.ImageCollection(images).mosaic();
  // GEE mosaic gives later images priority where valid; corn extraction follows.
  return classes.eq(run.cornCode).selfMask();
}

// COMPLETED-PERIOD POLICY, version CY2_COMPLETED_MODIS_V1.
// Forecast date is end of the labeled day UTC (exclusive next-midnight boundary).
// Default lag=0 guarantees observation-period causality, NOT actual publication
// availability. Set a documented lag for a sensitivity run; do not claim a fixed
// lag reproduces historical GEE ingestion or original product versions.
var MODIS_RELEASE_LAG_DAYS = 0;
var DOYS = [65,81,97,113,129,145,161,177,193,209,225,241,257,273];
var REDUCE_SCALE = 250;
var TILE_SCALE = 8;
var STATE_FIPS = ['19','17','18','31','27','29','20','46','38','39','55','26'];
var VEG = ['NDVI','EVI_scaled','EVI2','NDMI','NDWI','NIRv','GCI'];
var COUNTS = VEG.map(function(b) { return b + '_count'; });
var AUDIT_FIELDS = ['temporal_policy','forecast_doy','forecast_end_exclusive',
  'assumed_release_lag_days','period_start','period_end_exclusive',
  'latest_observation_date','mod13_period_count','mod09_period_count',
  'complete_periods_present','mask_available_after','mask_eligible_at_row_date',
  'feature_available_on','row_temporally_eligible','availability_basis'];
var COUNTIES = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.inList('STATEFP', STATE_FIPS));
var MOD13_RAW = ee.ImageCollection('MODIS/061/MOD13Q1');
var MOD09_RAW = ee.ImageCollection('MODIS/061/MOD09A1');
var ANALYSIS_CRS = ee.Image(MOD13_RAW.first()).select('NDVI').projection();
if (MODIS_RELEASE_LAG_DAYS < 0 || MODIS_RELEASE_LAG_DAYS % 1 !== 0) {
  throw new Error('MODIS_RELEASE_LAG_DAYS must be a nonnegative integer.');
}

// Periods restart on January 1; the last one may be shorter than 8/16 days.
function tagPeriod(image, days) {
  var start = ee.Date(image.get('system:time_start'));
  var nextYear = ee.Date.fromYMD(ee.Number(start.get('year')).add(1), 1, 1);
  var end = ee.Date(ee.Number(start.advance(days, 'day').millis())
    .min(nextYear.millis()));
  return image.set('period_end_ms', end.millis())
    .set('assumed_available_ms', end.advance(MODIS_RELEASE_LAG_DAYS, 'day').millis());
}
var MOD13 = MOD13_RAW.map(function(i) { return tagPeriod(i, 16); });
var MOD09 = MOD09_RAW.map(function(i) { return tagPeriod(i, 8); });

// Masked fallbacks retain bands if upstream data are missing. No zeros fabricated.
function emptyBands(names) {
  return ee.Image.constant(names.map(function() { return 0; }))
    .rename(names).updateMask(ee.Image.constant(0));
}
function forecastDate(year, doy) {
  return ee.Date.fromYMD(year, 1, 1).advance(doy - 1, 'day');
}
function buildComposite(year, doy) {
  var date = forecastDate(year, doy);
  var cutoff = date.advance(1, 'day');
  var ready = MOD13.filterDate(date.advance(-64, 'day'), cutoff)
    .filter(ee.Filter.lte('assumed_available_ms', cutoff.millis()))
    .sort('system:time_start', false).limit(1);
  var n13 = ready.size();
  var dummy = emptyBands(['NDVI','EVI','SummaryQA'])
    .set('system:time_start', date.advance(-32, 'day').millis())
    .set('period_end_ms', date.advance(-16, 'day').millis());
  var selected = ee.Image(ee.Algorithms.If(n13.gt(0), ready.first(), dummy));
  var start = ee.Date(selected.get('system:time_start'));
  var end = ee.Date(selected.get('period_end_ms'));
  // Both products cover the SAME completed period. Usually one MOD13 + two MOD09.
  var ready09 = MOD09.filterDate(start, end)
    .filter(ee.Filter.lte('period_end_ms', end.millis()))
    .filter(ee.Filter.lte('assumed_available_ms', cutoff.millis()));
  var n09 = ready09.size();
  var vi = ee.Image(prepMOD13(selected));
  var sr = ee.Image(ee.Algorithms.If(n09.gt(0), ready09.map(prepMOD09).median(),
    emptyBands(['RED','NIR','GREEN','SWIR1','EVI2','NDMI','NDWI','GCI'])));
  // Do not silently average an incomplete set of source periods.
  var expected09 = end.difference(start, 'day').divide(8).ceil();
  var complete = n13.eq(1).and(n09.eq(expected09));
  var nirv = sr.select('NIR').multiply(vi.select('NDVI')).rename('NIRv');
  var result = vi.addBands(sr.select(['EVI2','NDMI','NDWI','GCI']))
    .addBands(nirv).select(VEG).updateMask(ee.Image.constant(complete));
  return result.set({
    temporal_policy: 'CY2_COMPLETED_MODIS_V1', forecast_doy: doy,
    forecast_end_exclusive: cutoff.format('YYYY-MM-dd'),
    assumed_release_lag_days: MODIS_RELEASE_LAG_DAYS,
    period_start: ee.Algorithms.If(n13.gt(0), start.format('YYYY-MM-dd'), 'MISSING'),
    period_end_exclusive: ee.Algorithms.If(n13.gt(0), end.format('YYYY-MM-dd'), 'MISSING'),
    latest_observation_date: ee.Algorithms.If(n13.gt(0), end.advance(-1,'day').format('YYYY-MM-dd'), 'MISSING'),
    mod13_period_count: n13, mod09_period_count: n09,
    complete_periods_present: complete,
    source_available_ms: end.advance(MODIS_RELEASE_LAG_DAYS, 'day').millis()
  });
}
// Build a CSV row without passing nullable reducer results to Dictionary.set.
// Dictionary.map drops keys whose value is null; selectors retain blank CSV cells.
function countyExportFeature(f, date, year, properties) {
  f = ee.Feature(f);
  var meanNames = VEG.map(function(b) { return b + '_mean'; });
  var means = f.toDictionary().select(meanNames, true)
    .map(function(key, value) { return value; });
  // Rename only keys actually present: missing means remain missing, never zero.
  var present = means.keys();
  var renamed = present.map(function(key) {
    return ee.String(key).replace('_mean$', '');
  });
  means = means.rename(present, renamed);
  var counts = ee.Dictionary.fromLists(COUNTS, COUNTS.map(function(key) {
    var value = f.get(key);
    return ee.Algorithms.If(ee.Algorithms.IsEqual(value, null), 0, value);
  }));
  var metadata = ee.Dictionary({
    GEOID: f.get('GEOID'), date: date.format('YYYY-MM-dd'), year: year
  });
  return ee.Feature(null, metadata.combine(means).combine(counts).combine(properties));
}

function reduceCheckpoint(year, doy, cornMask, maskAfter) {
  var date = forecastDate(year, doy);
  var vegetation = buildComposite(year, doy);
  // Historical vegetation can be re-extracted with a later-available monthly mask.
  // Such early rows are history for later predictions, NOT early forecasts.
  var available = ee.Date(maskAfter);
  var eligible = date.millis().gte(available.millis());
  var complete = ee.Number(vegetation.get('complete_periods_present'));
  var featureReady = ee.Date(available.millis().max(
    ee.Number(vegetation.get('source_available_ms'))));
  var usable = eligible.and(complete).and(featureReady.millis()
    .lte(date.advance(1, 'day').millis()));
  var properties = vegetation.toDictionary(AUDIT_FIELDS.slice(0,10))
    .set('mask_available_after', maskAfter)
    .set('mask_eligible_at_row_date', eligible)
    .set('feature_available_on', ee.Algorithms.If(complete,
      featureReady.format('YYYY-MM-dd'), 'MISSING_DATA'))
    .set('row_temporally_eligible', usable)
    .set('availability_basis', 'ASSUMED_MONTH_END_AND_COMPLETED_MODIS');
  var reduced = vegetation.updateMask(cornMask).reduceRegions({
    collection: COUNTIES,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs:true}),
    crs: ANALYSIS_CRS, scale: REDUCE_SCALE, tileScale: TILE_SCALE
  });
  return reduced.map(function(f) {
    return countyExportFeature(f, date, year, properties);
  });
}
function exportMaskedYear(year, cornMask, name, maskAfter, run) {
  var rows = ee.FeatureCollection([]);
  DOYS.forEach(function(d) { rows = rows.merge(reduceCheckpoint(year,d,cornMask,maskAfter)); });
  rows = rows.map(function(f) {
    return f.set({mask_family:run.family, mask_month:run.month,
      mask_asset_ids:run.ids.join('|'), mask_corn_code:run.cornCode});
  });
  Export.table.toDrive({collection:rows, description:name+'_COMPLETED',
    folder:DRIVE_FOLDER,fileNamePrefix:name,fileFormat:'CSV',
    selectors:['GEOID','date','year'].concat(VEG,COUNTS,AUDIT_FIELDS,
      ['mask_family','mask_month','mask_asset_ids','mask_corn_code'])});
}
function exportTemporalAudit(year) {
  var rows = ee.FeatureCollection(DOYS.map(function(d) {
    var v = buildComposite(year,d);
    return ee.Feature(null,v.toDictionary(AUDIT_FIELDS.slice(0,10)).set('year',year));
  }));
  print('Check all complete_periods_present values are true:', year, rows);
  Export.table.toDrive({collection:rows,description:'TemporalAudit_MODIS_'+year,
    folder:DRIVE_FOLDER,fileNamePrefix:'TemporalAudit_MODIS_'+year,fileFormat:'CSV'});
}

// QA and index definitions retained from the attached exporter.
function prepMOD13(image) {
  var good = image.select('SummaryQA').lte(1);
  var ndvi = image.select('NDVI').multiply(0.0001).rename('NDVI');
  var evi = image.select('EVI').multiply(0.0001).rename('EVI_scaled');
  // copyProperties returns Element; restore the Image type explicitly.
  return ee.Image(ndvi.addBands(evi).updateMask(good)
    .copyProperties(image, ['system:time_start']));
}
function prepMOD09(image) {
  var qa = image.select('StateQA');
  var cloud = qa.bitwiseAnd(3);
  var good = cloud.eq(0).or(cloud.eq(3))
    .and(qa.rightShift(2).bitwiseAnd(1).eq(0))
    .and(qa.rightShift(10).bitwiseAnd(1).eq(0))
    .and(qa.rightShift(12).bitwiseAnd(1).eq(0));
  var sr = image.select(['sur_refl_b01','sur_refl_b02',
    'sur_refl_b04','sur_refl_b06']).multiply(0.0001)
    .rename(['RED','NIR','GREEN','SWIR1']);
  var red = sr.select('RED');
  var nir = sr.select('NIR');
  var green = sr.select('GREEN');
  var swir = sr.select('SWIR1');
  var valid = red.gt(0).and(red.lte(1))
    .and(nir.gt(0)).and(nir.lte(1))
    .and(green.gt(0)).and(green.lte(1))
    .and(swir.gt(0)).and(swir.lte(1));
  var evi2 = nir.subtract(red).multiply(2.5)
    .divide(nir.add(red.multiply(2.4)).add(1)).rename('EVI2');
  var ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI');
  var ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var gci = nir.divide(green).subtract(1).rename('GCI');
  // Keep mapped reflectance results typed as images as well.
  return ee.Image(sr.addBands(evi2).addBands(ndmi).addBands(ndwi).addBands(gci)
    .updateMask(good).updateMask(valid)
    .copyProperties(image, ['system:time_start']));
}

// Validate actual image access instead of relying on metadata.type spelling.
// Sequential requests also avoid flooding a project in restricted mode.
// No export tasks are created until EVERY selected image has passed.
function validateAssetsThenRun(runs, onReady) {
  var ids = [];
  var seen = {};
  runs.forEach(function(run) {
    run.ids.forEach(function(id) {
      if (!seen[id]) { seen[id] = true; ids.push(id); }
    });
  });
  function next(index) {
    if (index === ids.length) {
      print('Image access verified:', ids.length, 'assets. Creating export tasks.');
      onReady(runs);
      return;
    }
    var id = ids[index];
    function fail(message) {
      print('IMAGE ACCESS CHECK FAILED — no export tasks created.');
      print('Asset:', id);
      print('Earth Engine error:', String(message));
    }
    try {
      ee.Image(id).bandNames().evaluate(function(bands, error) {
        if (error) { fail(error); return; }
        if (!Array.isArray(bands) || bands.length === 0) {
          fail('No image bands were returned.');
          return;
        }
        print('Image OK (' + (index + 1) + '/' + ids.length + ')', id, bands);
        next(index + 1);
      });
    } catch (error) { fail(error); }
  }
  next(0);
}
function createExportTasks(runs) {
  runs.forEach(function(run) {
    print(run.name, 'Assets:', run.ids, 'Assumed available:', run.after);
    var corn = cornMaskFromRun(run);
    exportMaskedYear(run.year, corn, run.name, run.after, run);
  });
  YEARS.forEach(exportTemporalAudit);
  print(runs.length + ' model CSVs + ' + YEARS.length + ' temporal audits. Start the tasks.');
  print('CLEAN_REBUILT = our ICDL; STUDY = published ICDL, matching the V4 filename parser.');
  print('Replace matching older CSVs; do not mix completed-period and old observation windows.');
  print('Review per-band counts: county rows with no covered corn stay blank, not zero vegetation.');
  print('Different native mask resolutions and footprints remain; counts are 250 m samples, not corn area.');
  print('Historical rows before mask availability are for later season-to-date features only.');
  print('Python must enforce monthly mask availability; row flags alone do not enforce model behavior.');
}
var RUNS = prepareRuns();
print('V3: missing vegetation stays blank; missing sample counts become zero.');
print('Checking selected images. Wait for image checks before opening the Tasks tab.');
validateAssetsThenRun(RUNS, createExportTasks);
