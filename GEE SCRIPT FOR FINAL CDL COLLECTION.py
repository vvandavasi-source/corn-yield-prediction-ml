// ============================================================================
// CORN YIELD 2
// SAME-YEAR FINAL-CDL MODIS EXTRACTION
// WHOLE CORN BELT — ONE EXPORT PER YEAR
// YEARS: 2023–2025
// ============================================================================
//
// OUTPUT:
//
//   Corn_MODIS_same_year_2023.csv
//   Corn_MODIS_same_year_2024.csv
//   Corn_MODIS_same_year_2025.csv
//
// ONLY 3 EXPORT TASKS.
//
// Each CSV contains ALL counties from:
//
// IA, IL, IN, NE, MN, MO,
// KS, SD, ND, OH, WI, MI
//
// FINAL CDL:
//   class 1 = corn
//
// 2023/2024:
//   USDA/NASS/CDL in Earth Engine
//
// 2025:
//   custom uploaded USDA 30 m CDL
//
// ============================================================================


// ============================================================================
// 1. SETTINGS
// ============================================================================

var YEARS = [
  2023,
  2024,
  2025
];


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
// MODIS sampling window.
//
// NOTE:
// This preserves the MODIS extraction definition we were already using.
// We can audit/change MODIS temporal alignment separately without mixing
// that methodological change into this export-organization change.
// ---------------------------------------------------------------------------

var HALF_WINDOW_DAYS = 8;


// County reduction scale.

var REDUCE_SCALE = 250;

var TILE_SCALE = 8;


// Google Drive folder.

var DRIVE_FOLDER =
  'CornYield_SameYear_FINAL_CDL';


// ============================================================================
// 2. 2025 CUSTOM CDL
// ============================================================================
//
// Change this ONLY if the asset ID in your Assets tab differs.
//

var CDL_2025_ASSET =
  'projects/ee-gtellezgiron/assets/CDL_2025_30m';


// ============================================================================
// 3. CORN BELT STATES
// ============================================================================

var STATE_FIPS = [

  '19',   // Iowa
  '17',   // Illinois
  '18',   // Indiana
  '31',   // Nebraska
  '27',   // Minnesota
  '29',   // Missouri
  '20',   // Kansas
  '46',   // South Dakota
  '38',   // North Dakota
  '39',   // Ohio
  '55',   // Wisconsin
  '26'    // Michigan

];


// ============================================================================
// 4. COUNTIES
// ============================================================================

var COUNTIES =

  ee.FeatureCollection(
    'TIGER/2018/Counties'
  )

  .filter(

    ee.Filter.inList(
      'STATEFP',
      STATE_FIPS
    )

  );


print(
  'Corn Belt counties:',
  COUNTIES.size()
);


// ============================================================================
// 5. MODIS COLLECTIONS
// ============================================================================

var MOD13 =

  ee.ImageCollection(
    'MODIS/061/MOD13Q1'
  );


var MOD09 =

  ee.ImageCollection(
    'MODIS/061/MOD09A1'
  );


var CDL_COLLECTION =

  ee.ImageCollection(
    'USDA/NASS/CDL'
  );


// ============================================================================
// 6. GET SAME-YEAR FINAL CDL
// ============================================================================

function getFinalCDL(year) {


  // -------------------------------------------------------------------------
  // 2025
  //
  // Uploaded USDA 30 m national CDL.
  // -------------------------------------------------------------------------

  if (year === 2025) {


    print(
      'Using custom 2025 CDL:',
      CDL_2025_ASSET
    );


    return ee.Image(
      CDL_2025_ASSET
    )

    .select(
      [0],
      ['cropland']
    );

  }


  // -------------------------------------------------------------------------
  // 2023 / 2024
  //
  // Official Earth Engine USDA/NASS CDL.
  // -------------------------------------------------------------------------

  var start = ee.Date.fromYMD(
    year,
    1,
    1
  );


  var end = start.advance(
    1,
    'year'
  );


  var image =

    ee.Image(

      CDL_COLLECTION

      .filterDate(
        start,
        end
      )

      .first()

    );


  return image.select(
    'cropland'
  );

}


// ============================================================================
// 7. PREP MOD13Q1
//
// NDVI
// EVI
// ============================================================================

function prepMOD13(image) {


  // -------------------------------------------------------------------------
  // SummaryQA
  //
  // 0 = good
  // 1 = marginal
  // 2 = snow / ice
  // 3 = cloudy
  // -------------------------------------------------------------------------

  var qa =

    image.select(
      'SummaryQA'
    );


  var good =

    qa.lte(1);


  // -------------------------------------------------------------------------
  // NDVI
  // -------------------------------------------------------------------------

  var ndvi =

    image

    .select(
      'NDVI'
    )

    .multiply(
      0.0001
    )

    .rename(
      'NDVI'
    );


  // -------------------------------------------------------------------------
  // EVI
  // -------------------------------------------------------------------------

  var evi =

    image

    .select(
      'EVI'
    )

    .multiply(
      0.0001
    )

    .rename(
      'EVI_scaled'
    );


  return ndvi

    .addBands(
      evi
    )

    .updateMask(
      good
    )

    .copyProperties(

      image,

      [
        'system:time_start'
      ]

    );

}


// ============================================================================
// 8. PREP MOD09A1
//
// RED
// NIR
// GREEN
// SWIR1
//
// Derived:
//
// EVI2
// NDMI
// NDWI
// GCI
// ============================================================================

function prepMOD09(image) {


  var stateQA =

    image.select(
      'StateQA'
    );


  // -------------------------------------------------------------------------
  // CLOUD STATE
  //
  // Bits 0–1
  // -------------------------------------------------------------------------

  var cloudState =

    stateQA.bitwiseAnd(
      3
    );


  var cloudOK =

    cloudState

    .eq(0)

    .or(

      cloudState.eq(3)

    );


  // -------------------------------------------------------------------------
  // CLOUD SHADOW
  //
  // Bit 2
  // -------------------------------------------------------------------------

  var cloudShadow =

    stateQA

    .rightShift(2)

    .bitwiseAnd(1);


  // -------------------------------------------------------------------------
  // INTERNAL CLOUD
  //
  // Bit 10
  // -------------------------------------------------------------------------

  var internalCloud =

    stateQA

    .rightShift(10)

    .bitwiseAnd(1);


  // -------------------------------------------------------------------------
  // SNOW / ICE
  //
  // Bit 12
  // -------------------------------------------------------------------------

  var snow =

    stateQA

    .rightShift(12)

    .bitwiseAnd(1);


  var qaMask =

    cloudOK

    .and(
      cloudShadow.eq(0)
    )

    .and(
      internalCloud.eq(0)
    )

    .and(
      snow.eq(0)
    );


  // =========================================================================
  // REFLECTANCE
  // =========================================================================

  var reflectance =

    image

    .select([

      'sur_refl_b01',   // RED

      'sur_refl_b02',   // NIR

      'sur_refl_b04',   // GREEN

      'sur_refl_b06'    // SWIR1

    ])

    .multiply(
      0.0001
    )

    .rename([

      'RED',

      'NIR',

      'GREEN',

      'SWIR1'

    ]);


  var red =

    reflectance.select(
      'RED'
    );


  var nir =

    reflectance.select(
      'NIR'
    );


  var green =

    reflectance.select(
      'GREEN'
    );


  var swir1 =

    reflectance.select(
      'SWIR1'
    );


  // -------------------------------------------------------------------------
  // REFLECTANCE VALIDITY MASK
  // -------------------------------------------------------------------------

  var reflectanceMask =

    red.gt(0)

    .and(
      red.lte(1)
    )

    .and(
      nir.gt(0)
    )

    .and(
      nir.lte(1)
    )

    .and(
      green.gt(0)
    )

    .and(
      green.lte(1)
    )

    .and(
      swir1.gt(0)
    )

    .and(
      swir1.lte(1)
    );


  // =========================================================================
  // EVI2
  //
  //       2.5 (NIR - RED)
  // ----------------------------
  //       NIR + 2.4 RED + 1
  // =========================================================================

  var evi2 =

    nir

    .subtract(
      red
    )

    .multiply(
      2.5
    )

    .divide(

      nir

      .add(

        red.multiply(
          2.4
        )

      )

      .add(1)

    )

    .rename(
      'EVI2'
    );


  // =========================================================================
  // NDMI
  //
  // NIR - SWIR1
  // -----------
  // NIR + SWIR1
  //
  // Vegetation / canopy moisture proxy.
  // =========================================================================

  var ndmi =

    nir

    .subtract(
      swir1
    )

    .divide(

      nir.add(
        swir1
      )

    )

    .rename(
      'NDMI'
    );


  // =========================================================================
  // NDWI
  //
  // GREEN - NIR
  // -----------
  // GREEN + NIR
  // =========================================================================

  var ndwi =

    green

    .subtract(
      nir
    )

    .divide(

      green.add(
        nir
      )

    )

    .rename(
      'NDWI'
    );


  // =========================================================================
  // GCI
  //
  // NIR
  // ----- - 1
  // GREEN
  // =========================================================================

  var gci =

    nir

    .divide(
      green
    )

    .subtract(
      1
    )

    .rename(
      'GCI'
    );


  // =========================================================================
  // RETURN
  // =========================================================================

  return reflectance

    .addBands(
      evi2
    )

    .addBands(
      ndmi
    )

    .addBands(
      ndwi
    )

    .addBands(
      gci
    )

    .updateMask(
      qaMask
    )

    .updateMask(
      reflectanceMask
    )

    .copyProperties(

      image,

      [
        'system:time_start'
      ]

    );

}


// ============================================================================
// 9. BUILD ONE MODIS COMPOSITE
// ============================================================================

function buildComposite(

  year,

  doy

) {


  // -------------------------------------------------------------------------
  // DOY center date
  // -------------------------------------------------------------------------

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


  // -------------------------------------------------------------------------
  // Current MODIS window definition
  // -------------------------------------------------------------------------

  var start =

    center.advance(

      -HALF_WINDOW_DAYS,

      'day'

    );


  // filterDate end is exclusive.
  //
  // HALF_WINDOW_DAYS + 1 includes center + 8.

  var end =

    center.advance(

      HALF_WINDOW_DAYS + 1,

      'day'

    );


  // =========================================================================
  // MOD13
  // =========================================================================

  var vi =

    MOD13

    .filterDate(
      start,
      end
    )

    .map(
      prepMOD13
    )

    .median();


  // =========================================================================
  // MOD09
  // =========================================================================

  var sr =

    MOD09

    .filterDate(
      start,
      end
    )

    .map(
      prepMOD09
    )

    .median();


  // =========================================================================
  // NIRv
  //
  // NIR × NDVI
  // =========================================================================

  var nirv =

    sr

    .select(
      'NIR'
    )

    .multiply(

      vi.select(
        'NDVI'
      )

    )

    .rename(
      'NIRv'
    );


  // =========================================================================
  // FINAL VEGETATION IMAGE
  // =========================================================================

  return vi

    .addBands(

      sr.select([

        'EVI2',

        'NDMI',

        'NDWI',

        'GCI'

      ])

    )

    .addBands(
      nirv
    )

    .select([

      'NDVI',

      'EVI_scaled',

      'EVI2',

      'NDMI',

      'NDWI',

      'NIRv',

      'GCI'

    ]);

}


// ============================================================================
// 10. REDUCE ONE DOY TO ALL CORN BELT COUNTIES
// ============================================================================

function reduceCheckpoint(

  year,

  doy,

  finalCDL

) {


  // -------------------------------------------------------------------------
  // Forecast / observation date
  // -------------------------------------------------------------------------

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


  // =========================================================================
  // FINAL CDL CORN MASK
  //
  // CDL class 1 = corn
  // =========================================================================

  var cornMask =

    finalCDL

    .eq(1)

    .selfMask();


  // =========================================================================
  // MASK MODIS TO CORN
  // =========================================================================

  var vegetation =

    buildComposite(

      year,

      doy

    )

    .updateMask(
      cornMask
    );


  // =========================================================================
  // COUNTY MEANS
  // =========================================================================

  var reduced =

    vegetation.reduceRegions({

      collection:
        COUNTIES,

      reducer:
        ee.Reducer.mean(),

      scale:
        REDUCE_SCALE,

      tileScale:
        TILE_SCALE

    });


  // =========================================================================
  // KEEP ONLY THE MODEL SCHEMA
  // =========================================================================

  return reduced.map(

    function(feature) {


      return ee.Feature(

        null,

        {

          GEOID:

            feature.get(
              'GEOID'
            ),


          date:

            center.format(
              'YYYY-MM-dd'
            ),


          year:

            year,


          NDVI:

            feature.get(
              'NDVI'
            ),


          EVI_scaled:

            feature.get(
              'EVI_scaled'
            ),


          EVI2:

            feature.get(
              'EVI2'
            ),


          NDMI:

            feature.get(
              'NDMI'
            ),


          NDWI:

            feature.get(
              'NDWI'
            ),


          NIRv:

            feature.get(
              'NIRv'
            ),


          GCI:

            feature.get(
              'GCI'
            )

        }

      );

    }

  );

}


// ============================================================================
// 11. BUILD ONE COMPLETE YEAR
// ============================================================================
//
// One year:
//
// ~1,055 counties
//
// ×
//
// 14 DOYs
//
// ≈ 14,770 rows
//
// ============================================================================

function buildYear(

  year,

  finalCDL

) {


  var output =

    ee.FeatureCollection(
      []
    );


  DOYS.forEach(

    function(doy) {


      output =

        output.merge(

          reduceCheckpoint(

            year,

            doy,

            finalCDL

          )

        );

    }

  );


  return output;

}


// ============================================================================
// 12. CREATE ONE EXPORT PER YEAR
// ============================================================================

YEARS.forEach(

  function(year) {


    print(
      '=============================================='
    );


    print(
      'Preparing year:',
      year
    );


    // -----------------------------------------------------------------------
    // Load final same-year CDL
    // -----------------------------------------------------------------------

    var finalCDL =

      getFinalCDL(
        year
      );


    // -----------------------------------------------------------------------
    // Build all 14 checkpoints for the entire 12-state Corn Belt
    // -----------------------------------------------------------------------

    var output =

      buildYear(

        year,

        finalCDL

      );


    // -----------------------------------------------------------------------
    // Export name
    // -----------------------------------------------------------------------

    var exportName =

      'Corn_MODIS_same_year_' +

      year;


    print(

      exportName,

      'rows:',

      output.size()

    );


    // -----------------------------------------------------------------------
    // ONE EXPORT TASK FOR THIS YEAR
    // -----------------------------------------------------------------------

    Export.table.toDrive({

      collection:
        output,

      description:
        exportName,

      folder:
        DRIVE_FOLDER,

      fileNamePrefix:
        exportName,

      fileFormat:
        'CSV',

      selectors: [

        'GEOID',

        'date',

        'year',

        'NDVI',

        'EVI_scaled',

        'EVI2',

        'NDMI',

        'NDWI',

        'NIRv',

        'GCI'

      ]

    });


  }

);


// ============================================================================
// 13. VISUAL CHECK — 2025 FINAL CDL CORN
// ============================================================================

var CDL_2025 =

  getFinalCDL(
    2025
  );


var IOWA =

  ee.FeatureCollection(
    'TIGER/2018/States'
  )

  .filter(

    ee.Filter.eq(
      'STUSPS',
      'IA'
    )

  );


Map.centerObject(

  IOWA,

  7

);


Map.addLayer(

  CDL_2025

  .eq(1)

  .selfMask(),

  {},

  '2025 FINAL CDL CORN'

);


// ============================================================================
// 14. SUMMARY
// ============================================================================

print(
  '============================================================'
);


print(
  'SAME-YEAR FINAL-CDL MODIS EXPORT SETUP COMPLETE'
);


print(
  '============================================================'
);


print(
  'Years:',
  YEARS
);


print(
  'States:',
  STATE_FIPS.length
);


print(
  'DOYs:',
  DOYS.length
);


print(
  'Exports:',
  YEARS.length
);


print(
  'Drive folder:',
  DRIVE_FOLDER
);


print(
  'Expected tasks:'
);


print(
  'Corn_MODIS_same_year_2023'
);


print(
  'Corn_MODIS_same_year_2024'
);


print(
  'Corn_MODIS_same_year_2025'
);


print(
  'Open Tasks and start the 3 exports.'
);