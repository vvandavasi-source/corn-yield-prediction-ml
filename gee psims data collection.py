// ============================================================
// CORN BELT COUNTY WEATHER FEATURES — PRISM
// ============================================================
//
// PURPOSE
// -------
// Generate county-level sequential PRISM weather variables
// for the Corn Yield 2 model.
//
// OUTPUT
// ------
// One CSV per year:
//
// CornBelt_PRISM_Weather_2023.csv
// CornBelt_PRISM_Weather_2024.csv
// CornBelt_PRISM_Weather_2025.csv
//
// GEE exports the full 140-variable sequential weather table.
// The Python canonical model later converts these into our
// selected 15 compact PRISM features.
//
// ============================================================


// ============================================================
// 1. SETTINGS
// ============================================================

var START_YEAR = 2023;
var END_YEAR   = 2025;


// ------------------------------------------------------------
// Existing Corn Yield 2 DOY schedule
// ------------------------------------------------------------

var DOYS = [
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
];


// ------------------------------------------------------------
// ORIGINAL 16-DAY WINDOW
//
// center DOY - 8
// through
// center DOY + 7
//
// Earth Engine filterDate end is exclusive.
// ------------------------------------------------------------

var HALF_WINDOW = 8;


// ------------------------------------------------------------
// Corn Belt state FIPS
//
// IA IL IN NE MN MO KS SD ND OH WI MI
// ------------------------------------------------------------

var STATE_FIPS = [
  '19',
  '17',
  '18',
  '31',
  '27',
  '29',
  '20',
  '46',
  '38',
  '39',
  '55',
  '26'
];


// PRISM native resolution ≈ 4.6 km.

var WEATHER_SCALE = 4638;


// Google Drive folder.

var DRIVE_FOLDER =
  'CornYield_PRISM_Weather';


// ============================================================
// 2. COUNTY BOUNDARIES
// ============================================================

var counties = ee.FeatureCollection(
    'TIGER/2018/Counties'
  )

  .filter(
    ee.Filter.inList(
      'STATEFP',
      STATE_FIPS
    )
  )

  .map(function(feature) {

    return feature.set({

      'FIPS':
        feature.get('GEOID'),

      'county_name':
        feature.get('NAME')

    });

  });


print(
  'Number of counties:',
  counties.size()
);


Map.centerObject(
  counties,
  5
);


Map.addLayer(
  counties,
  {},
  'Corn Belt Counties',
  false
);


// ============================================================
// 3. PRISM DAILY WEATHER
// ============================================================
//
// Bands:
//   ppt       mm/day
//   tmean     °C
//   tmin      °C
//   tmax      °C
//   tdmean    °C
//   vpdmin    hPa
//   vpdmax    hPa
//
// ============================================================

var prism = ee.ImageCollection(
  'OREGONSTATE/PRISM/ANd'
);


// ============================================================
// 4. HELPER — PAD DOY
// ============================================================

function padDOY(doy) {

  if (doy < 10) {
    return '00' + doy;
  }

  if (doy < 100) {
    return '0' + doy;
  }

  return String(doy);
}


// ============================================================
// 5. DAILY DERIVED WEATHER VARIABLES
// ============================================================

function addDerivedWeather(image) {


  // ----------------------------------------------------------
  // Mean VPD
  //
  // midpoint of PRISM daily VPDmin and VPDmax
  // ----------------------------------------------------------

  var vpdmean = image

    .select('vpdmin')

    .add(
      image.select('vpdmax')
    )

    .divide(2)

    .rename('vpdmean');


  // ----------------------------------------------------------
  // HEAT STRESS
  //
  // Degrees by which Tmax exceeds 30 °C.
  //
  // Example:
  // Tmax = 34
  // heat30 = 4
  // ----------------------------------------------------------

  var heat30 = image

    .select('tmax')

    .subtract(30)

    .max(0)

    .rename('heat30');


  // ----------------------------------------------------------
  // EXTREME HOT DAY
  //
  // 1 if Tmax >= 35 °C
  // ----------------------------------------------------------

  var hot35 = image

    .select('tmax')

    .gte(35)

    .rename('hot35');


  // ----------------------------------------------------------
  // DRY DAY
  //
  // 1 if precipitation < 1 mm/day
  // ----------------------------------------------------------

  var dryday = image

    .select('ppt')

    .lt(1)

    .rename('dryday');


  return image.addBands([
    vpdmean,
    heat30,
    hot35,
    dryday
  ]);

}


// ============================================================
// 6. CREATE ONE 16-DAY WEATHER IMAGE
// ============================================================

function makeWindowImage(
  year,
  doy
) {


  var doyText =
    padDOY(doy);


  // ----------------------------------------------------------
  // Center date
  // ----------------------------------------------------------

  var center = ee.Date

    .fromYMD(
      year,
      1,
      1
    )

    .advance(
      doy - 1,
      'day'
    );


  // ----------------------------------------------------------
  // ORIGINAL 16-DAY WINDOW
  //
  // Example DOY 177:
  //
  // center - 8
  // through
  // center + 7
  //
  // start inclusive
  // end exclusive
  // ----------------------------------------------------------

  var start = center.advance(
    -HALF_WINDOW,
    'day'
  );


  var end = center.advance(
    HALF_WINDOW,
    'day'
  );


  var daily = prism

    .filterDate(
      start,
      end
    )

    .map(
      addDerivedWeather
    );


  // ==========================================================
  // TEMPORAL MEANS
  // ==========================================================

  var means = daily

    .select([
      'tmean',
      'tmin',
      'tmax',
      'vpdmin',
      'vpdmean',
      'vpdmax'
    ])

    .mean()

    .rename([

      'tmean_DOY_'   + doyText,

      'tmin_DOY_'    + doyText,

      'tmax_DOY_'    + doyText,

      'vpdmin_DOY_'  + doyText,

      'vpdmean_DOY_' + doyText,

      'vpdmax_DOY_'  + doyText

    ]);


  // ==========================================================
  // TOTAL PRECIPITATION
  // ==========================================================

  var precipitation = daily

    .select('ppt')

    .sum()

    .rename(
      'ppt_DOY_' + doyText
    );


  // ==========================================================
  // CUMULATIVE HEAT STRESS
  // ==========================================================

  var heat30 = daily

    .select('heat30')

    .sum()

    .rename(
      'heat30_DOY_' + doyText
    );


  // ==========================================================
  // NUMBER OF Tmax >= 35 °C DAYS
  // ==========================================================

  var hot35days = daily

    .select('hot35')

    .sum()

    .rename(
      'hot35days_DOY_' + doyText
    );


  // ==========================================================
  // NUMBER OF PPT < 1 mm DAYS
  // ==========================================================

  var drydays = daily

    .select('dryday')

    .sum()

    .rename(
      'drydays_DOY_' + doyText
    );


  // ==========================================================
  // COMBINE
  // ==========================================================

  return means

    .addBands(
      precipitation
    )

    .addBands(
      heat30
    )

    .addBands(
      hot35days
    )

    .addBands(
      drydays
    );

}


// ============================================================
// 7. BUILD COMPLETE WEATHER IMAGE FOR ONE YEAR
// ============================================================
//
// 14 DOY windows
//
// ×
//
// 10 weather variables
//
// = 140 raw sequential PRISM features
//
// ============================================================

function makeYearWeatherImage(year) {


  var weatherImage =
    makeWindowImage(
      year,
      DOYS[0]
    );


  for (
    var i = 1;
    i < DOYS.length;
    i++
  ) {

    weatherImage =
      weatherImage.addBands(

        makeWindowImage(
          year,
          DOYS[i]
        )

      );

  }


  return weatherImage;

}


// ============================================================
// 8. COUNTY AGGREGATION
// ============================================================

function makeCountyWeather(year) {


  print(
    'Building year:',
    year
  );


  var weatherImage =
    makeYearWeatherImage(
      year
    );


  // ----------------------------------------------------------
  // Spatial county mean.
  //
  // IMPORTANT:
  //
  // precipitation was already summed TEMPORALLY.
  //
  // Therefore:
  //
  // ppt_DOY_###
  //
  // = mean county accumulated precipitation
  //   during that 16-day window.
  // ----------------------------------------------------------

  var countyWeather =
    weatherImage.reduceRegions({

      collection:
        counties,

      reducer:
        ee.Reducer.mean(),

      scale:
        WEATHER_SCALE,

      tileScale:
        4

    });


  // ----------------------------------------------------------
  // Add year and remove geometry
  // ----------------------------------------------------------

  countyWeather =
    countyWeather.map(

      function(feature) {


        var properties =
          feature

          .toDictionary()

          .set(
            'year',
            year
          );


        return ee.Feature(
          null,
          properties
        );

      }

    );


  return countyWeather;

}


// ============================================================
// 9. EXPORT ONE CSV PER YEAR
// ============================================================

for (
  var year = START_YEAR;
  year <= END_YEAR;
  year++
) {


  var output =
    makeCountyWeather(
      year
    );


  var description =

    'CornBelt_PRISM_Weather_'

    +

    year;


  Export.table.toDrive({

    collection:
      output,

    description:
      description,

    folder:
      DRIVE_FOLDER,

    fileNamePrefix:
      description,

    fileFormat:
      'CSV'

  });


  print(
    'Export task created:',
    description
  );

}


// ============================================================
// 10. VISUAL CHECK
// ============================================================
//
// If 2023 is being processed, add DOY 177 precipitation
// and Tmax layers for a quick visual sanity check.
// ============================================================

if (
  START_YEAR <= 2023 &&
  END_YEAR >= 2023
) {


  var example2023 =
    makeWindowImage(
      2023,
      177
    );


  Map.addLayer(

    example2023.select(
      'ppt_DOY_177'
    ),

    {
      min: 0,
      max: 100
    },

    '2023 PPT DOY 177',

    false

  );


  Map.addLayer(

    example2023.select(
      'tmax_DOY_177'
    ),

    {
      min: 20,
      max: 40
    },

    '2023 Tmax DOY 177',

    false

  );

}


// ============================================================
// FINISHED
// ============================================================

print(
  '=============================================='
);

print(
  'PRISM WEATHER EXPORT SETUP COMPLETE'
);

print(
  '=============================================='
);

print(
  'Years:',
  START_YEAR,
  'to',
  END_YEAR
);

print(
  'Drive folder:',
  DRIVE_FOLDER
);