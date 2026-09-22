// ============================================================================
// CORN YIELD 2
// LEAKAGE-FREE PRISM WEATHER COLLECTOR
// 2008–2025
// ============================================================================
//
// PURPOSE
// -------
// Rebuild the complete Corn Belt PRISM weather archive using
// ONLY weather available on or before each forecast DOY.
//
// OLD WINDOW:
//   DOY - 8 through DOY + 7
//   -> contained future weather
//
// NEW WINDOW:
//   DOY - 15 through DOY
//   -> exactly 16 days
//   -> causal / leakage-free
//
// OUTPUT
// ------
// One CSV per year:
//
//   CornBelt_PRISM_Weather_2008.csv
//   CornBelt_PRISM_Weather_2009.csv
//   ...
//   CornBelt_PRISM_Weather_2025.csv
//
// RAW VARIABLES PER DOY
// ---------------------
// tmin
// tmean
// tmax
// ppt
// vpdmin
// vpdmean
// vpdmax
// heat30
// hot35days
// drydays
//
// 14 DOYs × 10 variables = 140 raw sequential variables.
//
// The Python yield model later converts these into the
// selected 15 compact PRISM predictors.
//
// ============================================================================


// ============================================================================
// 1. SETTINGS
// ============================================================================

var START_YEAR = 2008;

var END_YEAR = 2025;


// Forecast checkpoints used by the yield model.

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


// ---------------------------------------------------------------------------
// TRAILING WEATHER WINDOW
//
// We want exactly:
//
//   forecast DOY - 15
//
// through
//
//   forecast DOY
//
// inclusive.
//
// Example:
//
// DOY 161:
//   146–161
//
// DOY 177:
//   162–177
//
// DOY 193:
//   178–193
//
// ---------------------------------------------------------------------------

var TRAILING_DAYS = 15;


// ---------------------------------------------------------------------------
// CORN BELT STATES
//
// IA = 19
// IL = 17
// IN = 18
// NE = 31
// MN = 27
// MO = 29
// KS = 20
// SD = 46
// ND = 38
// OH = 39
// WI = 55
// MI = 26
// ---------------------------------------------------------------------------

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


// PRISM native resolution is approximately 4.6 km.

var WEATHER_SCALE = 4638.3;


// Tile scale for county reduction.

var TILE_SCALE = 4;


// Google Drive output folder.

var DRIVE_FOLDER =

  'CornYield_PRISM_Weather_LEAKAGE_FREE';


// ============================================================================
// 2. COUNTY BOUNDARIES
// ============================================================================

var counties =

  ee.FeatureCollection(
    'TIGER/2018/Counties'
  )

  .filter(

    ee.Filter.inList(

      'STATEFP',

      STATE_FIPS

    )

  )

  .map(

    function(feature) {


      return feature.set({

        'FIPS':
          feature.get(
            'GEOID'
          ),

        'county_name':
          feature.get(
            'NAME'
          )

      });


    }

  );


print(

  'Corn Belt counties:',

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


// ============================================================================
// 3. DAILY PRISM COLLECTION
// ============================================================================
//
// Dataset:
//
//   OREGONSTATE/PRISM/ANd
//
// Native variables used:
//
//   ppt
//       Daily precipitation
//       mm/day
//
//   tmean
//       Daily mean temperature
//       degrees C
//
//   tmin
//       Daily minimum temperature
//       degrees C
//
//   tmax
//       Daily maximum temperature
//       degrees C
//
//   vpdmin
//       Minimum vapor pressure deficit
//       hPa
//
//   vpdmax
//       Maximum vapor pressure deficit
//       hPa
//
// ============================================================================

var prism =

  ee.ImageCollection(

    'OREGONSTATE/PRISM/ANd'

  );


// ============================================================================
// 4. HELPER — DOY LABEL
// ============================================================================

function padDOY(doy) {


  if (doy < 10) {

    return '00' + doy;

  }


  if (doy < 100) {

    return '0' + doy;

  }


  return String(doy);

}


// ============================================================================
// 5. ADD DAILY DERIVED VARIABLES
// ============================================================================

function addDerivedWeather(image) {


  // ========================================================================
  // MEAN VPD
  //
  // PRISM provides vpdmin and vpdmax.
  //
  // Daily approximation:
  //
  //   (vpdmin + vpdmax) / 2
  // ========================================================================

  var vpdmean =

    image

    .select(
      'vpdmin'
    )

    .add(

      image.select(
        'vpdmax'
      )

    )

    .divide(
      2
    )

    .rename(
      'vpdmean'
    );


  // ========================================================================
  // HEAT30
  //
  // Daily degrees above 30 C.
  //
  // Example:
  //
  // Tmax = 34 C
  //
  // heat30 = 4
  //
  // Tmax = 28 C
  //
  // heat30 = 0
  // ========================================================================

  var heat30 =

    image

    .select(
      'tmax'
    )

    .subtract(
      30
    )

    .max(
      0
    )

    .rename(
      'heat30'
    );


  // ========================================================================
  // HOT35
  //
  // Indicator for extreme heat:
  //
  // 1 if Tmax >= 35 C
  // 0 otherwise
  // ========================================================================

  var hot35 =

    image

    .select(
      'tmax'
    )

    .gte(
      35
    )

    .rename(
      'hot35'
    );


  // ========================================================================
  // DRY DAY
  //
  // 1 if precipitation < 1 mm/day
  // ========================================================================

  var dryday =

    image

    .select(
      'ppt'
    )

    .lt(
      1
    )

    .rename(
      'dryday'
    );


  // ========================================================================
  // RETURN
  // ========================================================================

  return image.addBands([

    vpdmean,

    heat30,

    hot35,

    dryday

  ]);

}


// ============================================================================
// 6. CREATE ONE LEAKAGE-FREE 16-DAY WINDOW
// ============================================================================

function makeWindowImage(

  year,

  doy

) {


  var doyText =

    padDOY(
      doy
    );


  // ========================================================================
  // FORECAST DATE
  // ========================================================================

  var center =

    ee.Date

    .fromYMD(

      year,

      1,

      1

    )

    .advance(

      doy - 1,

      'day'

    );


  // ========================================================================
  // CAUSAL WINDOW
  //
  // center - 15
  //
  // through
  //
  // center
  //
  // inclusive
  //
  // Earth Engine filterDate() uses:
  //
  //   start = inclusive
  //   end   = exclusive
  //
  // Therefore:
  //
  //   start = center - 15
  //   end   = center + 1
  //
  // gives exactly 16 days.
  // ========================================================================

  var start =

    center.advance(

      -TRAILING_DAYS,

      'day'

    );


  var end =

    center.advance(

      1,

      'day'

    );


  var daily =

    prism

    .filterDate(

      start,

      end

    )

    .map(

      addDerivedWeather

    );


  // ========================================================================
  // TEMPERATURE + VPD MEANS
  // ========================================================================

  var means =

    daily

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

      'tmean_DOY_' +
        doyText,

      'tmin_DOY_' +
        doyText,

      'tmax_DOY_' +
        doyText,

      'vpdmin_DOY_' +
        doyText,

      'vpdmean_DOY_' +
        doyText,

      'vpdmax_DOY_' +
        doyText

    ]);


  // ========================================================================
  // PRECIPITATION
  //
  // Total precipitation over the 16-day window.
  // ========================================================================

  var precipitation =

    daily

    .select(
      'ppt'
    )

    .sum()

    .rename(

      'ppt_DOY_' +
      doyText

    );


  // ========================================================================
  // HEAT30
  //
  // Total degrees above 30 C over the 16-day window.
  // ========================================================================

  var heat30 =

    daily

    .select(
      'heat30'
    )

    .sum()

    .rename(

      'heat30_DOY_' +
      doyText

    );


  // ========================================================================
  // HOT35 DAYS
  //
  // Number of days with Tmax >= 35 C.
  // ========================================================================

  var hot35days =

    daily

    .select(
      'hot35'
    )

    .sum()

    .rename(

      'hot35days_DOY_' +
      doyText

    );


  // ========================================================================
  // DRY DAYS
  //
  // Number of days with ppt < 1 mm.
  // ========================================================================

  var drydays =

    daily

    .select(
      'dryday'
    )

    .sum()

    .rename(

      'drydays_DOY_' +
      doyText

    );


  // ========================================================================
  // COMBINE ALL 10 VARIABLES
  // ========================================================================

  var output =

    means

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


  return output;

}


// ============================================================================
// 7. BUILD FULL YEAR WEATHER IMAGE
// ============================================================================
//
// 14 DOYs
//
// ×
//
// 10 raw variables
//
// = 140 bands
//
// ============================================================================

function makeYearWeatherImage(year) {


  // Start with first DOY.

  var weatherImage =

    makeWindowImage(

      year,

      DOYS[0]

    );


  // Add remaining DOYs.

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


// ============================================================================
// 8. COUNTY AGGREGATION
// ============================================================================

function makeCountyWeather(year) {


  print(

    'Building year:',

    year

  );


  var weatherImage =

    makeYearWeatherImage(
      year
    );


  // ========================================================================
  // SPATIAL AGGREGATION
  //
  // Temporal aggregation has already occurred above.
  //
  // Example:
  //
  // ppt_DOY_161
  //
  // already represents:
  //
  //   total precipitation from DOY 146 through 161
  //
  // reduceRegions now computes the spatial county mean of
  // that accumulated precipitation surface.
  // ========================================================================

  var countyWeather =

    weatherImage.reduceRegions({

      collection:
        counties,

      reducer:
        ee.Reducer.mean(),

      scale:
        WEATHER_SCALE,

      tileScale:
        TILE_SCALE

    });


  // ========================================================================
  // ADD YEAR + REMOVE GEOMETRY
  //
  // Removing geometry keeps exported CSV files small.
  // ========================================================================

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


// ============================================================================
// 9. EXPORT ONE CSV PER YEAR
// ============================================================================

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


// ============================================================================
// 10. TEMPORAL SANITY CHECK
// ============================================================================
//
// This prints the actual dates used for key 2025 forecast checkpoints.
//
// ============================================================================

function printWindowDates(

  year,

  doy

) {


  var center =

    ee.Date

    .fromYMD(

      year,

      1,

      1

    )

    .advance(

      doy - 1,

      'day'

    );


  var start =

    center.advance(

      -TRAILING_DAYS,

      'day'

    );


  var end =

    center.advance(

      1,

      'day'

    );


  var count =

    prism

    .filterDate(

      start,

      end

    )

    .size();


  print(

    'YEAR ' +
    year +
    ' | DOY ' +
    doy,

    'START:',

    start.format(
      'YYYY-MM-dd'
    ),

    'END:',

    center.format(
      'YYYY-MM-dd'
    ),

    'PRISM DAYS:',

    count

  );

}


// Check important operational dates.

printWindowDates(

  2025,

  161

);


printWindowDates(

  2025,

  177

);


printWindowDates(

  2025,

  193

);


printWindowDates(

  2025,

  209

);


printWindowDates(

  2025,

  225

);


// ============================================================================
// 11. VISUAL CHECK
// ============================================================================
//
// Show 2025 DOY 177 precipitation and Tmax.
//
// This does NOT affect exports.
//
// ============================================================================

if (

  START_YEAR <= 2025

  &&

  END_YEAR >= 2025

) {


  var example =

    makeWindowImage(

      2025,

      177

    );


  // ------------------------------------------------------------------------
  // Precipitation
  // ------------------------------------------------------------------------

  Map.addLayer(

    example.select(

      'ppt_DOY_177'

    ),

    {

      min:
        0,

      max:
        100

    },

    '2025 DOY 177 — trailing 16d PPT',

    false

  );


  // ------------------------------------------------------------------------
  // Tmax
  // ------------------------------------------------------------------------

  Map.addLayer(

    example.select(

      'tmax_DOY_177'

    ),

    {

      min:
        20,

      max:
        40

    },

    '2025 DOY 177 — trailing 16d Tmax',

    false

  );

}


// ============================================================================
// 12. FINAL SUMMARY
// ============================================================================

print(

  '=========================================================='

);


print(

  'LEAKAGE-FREE PRISM EXPORT SETUP COMPLETE'

);


print(

  '=========================================================='

);


print(

  'Years:',

  START_YEAR,

  'through',

  END_YEAR

);


print(

  'DOYs:',

  DOYS

);


print(

  'Window definition:',

  'DOY - 15 through DOY inclusive'

);


print(

  'Raw features per DOY:',

  10

);


print(

  'Total raw weather features:',

  DOYS.length * 10

);


print(

  'Drive folder:',

  DRIVE_FOLDER

);


print(

  'Expected files:'

);


print(

  'CornBelt_PRISM_Weather_2008.csv'

);


print(

  '...'

);


print(

  'CornBelt_PRISM_Weather_2025.csv'

);


print(

  'Open the Tasks tab and start the yearly exports.'

);