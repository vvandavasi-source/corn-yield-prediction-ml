// CORN YIELD 2 — final/previous CDL MODIS exporter, FIXED V3.
// GEE JAVASCRIPT packaged as .py for downloading. NOT a Python executable.
// Fixes Image types after copyProperties and null county means.
// MODIS calculation, QA, indices, and county row handling match ICDL FIXED V3.
// Paste into Earth Engine Code Editor. Final same-year masks are valid for prior
// training years and for ORACLE tests; they are not operational target-year masks.
// Rebuild ALL training years, not just 2023-2025. Start with [2025] to check tasks.
var YEARS = [2025];
// Full archive: [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
var EXPORT_SAME_YEAR = true;
var EXPORT_PREVIOUS_YEAR = true; // use the same temporal rule for the comparator
var PREVIOUS_YEAR_TEST_YEARS = [2021,2022,2023,2024,2025];
var DRIVE_FOLDER = 'CornYield_CDL_COMPLETED_PERIODS';
var CUSTOM_CDL = {2025:'projects/ee-gtellezgiron/assets/CDL_2025_30m'};

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
  'complete_periods_present','mask_available_after','mask_eligible_at_row_date'];
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
  // Same-year final CDL is retrospective training/oracle input.
  // Previous-year eligibility uses a bookkeeping date, not verified publication.
  var eligible = maskAfter === null ? false : date.millis().gte(ee.Date(maskAfter).millis());
  var properties = vegetation.toDictionary(AUDIT_FIELDS.slice(0,10))
    .set('mask_available_after', maskAfter === null ? 'FINAL_CDL_ORACLE' : maskAfter)
    .set('mask_eligible_at_row_date', eligible);
  var reduced = vegetation.updateMask(cornMask).reduceRegions({
    collection: COUNTIES,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs:true}),
    crs: ANALYSIS_CRS, scale: REDUCE_SCALE, tileScale: TILE_SCALE
  });
  return reduced.map(function(f) {
    return countyExportFeature(f, date, year, properties);
  });
}
function exportMaskedYear(year, cornMask, name, maskAfter) {
  var rows = ee.FeatureCollection([]);
  DOYS.forEach(function(d) { rows = rows.merge(reduceCheckpoint(year,d,cornMask,maskAfter)); });
  Export.table.toDrive({collection:rows, description:name+'_COMPLETED',
    folder:DRIVE_FOLDER,fileNamePrefix:name,fileFormat:'CSV',
    selectors:['GEOID','date','year'].concat(VEG,COUNTS,AUDIT_FIELDS)});
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

function getCDL(year) {
  if (CUSTOM_CDL[year]) { return ee.Image(CUSTOM_CDL[year]).select([0],['cropland']); }
  var start = ee.Date.fromYMD(year,1,1);
  return ee.Image(ee.ImageCollection('USDA/NASS/CDL')
    .filterDate(start,start.advance(1,'year')).first()).select('cropland');
}
// Check both custom and catalog CDL images before creating any tasks.
// Sequential requests keep the access check small for restricted projects.
function validateCDLThenRun(onReady) {
  var required = [];
  function need(year) {
    if (required.indexOf(year) === -1) { required.push(year); }
  }
  YEARS.forEach(function(year) {
    if (EXPORT_SAME_YEAR) { need(year); }
    if (EXPORT_PREVIOUS_YEAR && PREVIOUS_YEAR_TEST_YEARS.indexOf(year) !== -1) {
      need(year - 1);
    }
  });
  function next(index) {
    if (index === required.length) {
      print('CDL image access verified. Creating export tasks.');
      onReady();
      return;
    }
    var sourceYear = required[index];
    var source = CUSTOM_CDL[sourceYear] || ('USDA/NASS/CDL, year ' + sourceYear);
    function fail(message) {
      print('CDL CHECK FAILED: no export tasks created.', source, String(message));
    }
    try {
      getCDL(sourceYear).bandNames().evaluate(function(bands, error) {
        if (error) { fail(error); return; }
        if (!Array.isArray(bands) || bands.indexOf('cropland') === -1) {
          fail('Expected a cropland band after source selection.');
          return;
        }
        print('CDL OK:', sourceYear, source, bands);
        next(index + 1);
      });
    } catch (error) { fail(error); }
  }
  next(0);
}

function createExportTasks() {
  YEARS.forEach(function(year) {
    if (EXPORT_SAME_YEAR) {
      exportMaskedYear(year,getCDL(year).eq(1).selfMask(),'Corn_MODIS_same_year_'+year,null);
    }
    if (EXPORT_PREVIOUS_YEAR && PREVIOUS_YEAR_TEST_YEARS.indexOf(year) !== -1) {
      // Jan 1 is only a bookkeeping eligibility marker, not verified CDL release.
      // The requested model checkpoints occur later; verify actual publication for live use.
      exportMaskedYear(year,getCDL(year-1).eq(1).selfMask(),
        'Corn_MODIS_'+year+'_PREVIOUS_CDL',year+'-01-01');
    }
    exportTemporalAudit(year);
  });
  print('Same-year training/oracle + previous-year comparison exports are configured.');
  print('Use only the corrected exports together. Dates and filenames remain V4-compatible.');
  print('V3: missing vegetation stays blank; missing sample counts become zero.');
  print('YEARS controls every export. Default [2025]: 2 model CSVs + 1 temporal audit.');
}
print('Checking CDL inputs. Wait for the checks before opening Tasks.');
validateCDLThenRun(createExportTasks);
