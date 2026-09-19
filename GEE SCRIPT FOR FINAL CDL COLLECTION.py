// ============================================================================
// CORN YIELD 2
// SAME-YEAR FINAL-CDL MODIS EXPORTS — 2023–2025
// ============================================================================
//
// OUTPUT:
//   One CSV per state containing 2023, 2024, and 2025.
//
// Example:
//   Corn_MODIS_same_year_2023_2025_Iowa.csv
//
// These can sit beside:
//   Corn_MODIS_same_year_2008_2022_Iowa.csv
//
// FINAL-CDL corn mask:
//   CDL class 1 = corn
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


var HALF_WINDOW_DAYS = 8;

var REDUCE_SCALE = 250;

var TILE_SCALE = 8;


var DRIVE_FOLDER =
  "CornYield_SameYear_2023_2025";


// ============================================================================
// 2. 2025 CDL ASSET
// ============================================================================
//
// After uploading your 2025 30 m CDL to Earth Engine,
// replace this if your actual asset name is different.
//

var CDL_2025_ASSET =
  "projects/ee-gtellezgiron/assets/CDL_2025_30m";


// ============================================================================
// 3. OPTIONAL 2022 VALIDATION
// ============================================================================
//
// I recommend turning this TRUE once.
//
// It will export a 2022 Iowa file using this new pipeline.
// We can compare that to your existing:
//
// Corn_MODIS_same_year_2008_2022_Iowa.csv
//
// If they match well, set this back to false.
//

var RUN_2022_VALIDATION = false;


// ============================================================================
// 4. CORN BELT STATES
// ============================================================================

var STATES = [

  {
    name: "Iowa",
    fips: "19"
  },

  {
    name: "Illinois",
    fips: "17"
  },

  {
    name: "Indiana",
    fips: "18"
  },

  {
    name: "Nebraska",
    fips: "31"
  },

  {
    name: "Minnesota",
    fips: "27"
  },

  {
    name: "Missouri",
    fips: "29"
  },

  {
    name: "Kansas",
    fips: "20"
  },

  {
    name: "South_Dakota",
    fips: "46"
  },

  {
    name: "North_Dakota",
    fips: "38"
  },

  {
    name: "Ohio",
    fips: "39"
  },

  {
    name: "Wisconsin",
    fips: "55"
  },

  {
    name: "Michigan",
    fips: "26"
  }

];


// ============================================================================
// 5. DATASETS
// ============================================================================

var COUNTIES =
  ee.FeatureCollection(
    "TIGER/2018/Counties"
  );


var MOD13 =
  ee.ImageCollection(
    "MODIS/061/MOD13Q1"
  );


var MOD09 =
  ee.ImageCollection(
    "MODIS/061/MOD09A1"
  );


var CDL =
  ee.ImageCollection(
    "USDA/NASS/CDL"
  );


// ============================================================================
// 6. GET SAME-YEAR FINAL CDL
// ============================================================================

function getCDL(year) {


  // ----------------------------------------------------------
  // 2025 — our uploaded USDA 30 m CDL
  // ----------------------------------------------------------

  if (year === 2025) {

    print(
      "Using custom 2025 CDL:",
      CDL_2025_ASSET
    );


    return ee.Image(
      CDL_2025_ASSET
    )
    .select(
      [0],
      ["cropland"]
    );

  }


  // ----------------------------------------------------------
  // 2023 / 2024 / validation 2022
  // ----------------------------------------------------------

  var start =
    ee.Date.fromYMD(
      year,
      1,
      1
    );


  var end =
    start.advance(
      1,
      "year"
    );


  var image =
    ee.Image(
      CDL
      .filterDate(
        start,
        end
      )
      .first()
    );


  return image.select(
    "cropland"
  );

}


// ============================================================================
// 7. MOD13Q1 — NDVI + EVI
// ============================================================================

function prepMOD13(image) {


  // SummaryQA:
  //
  // 0 = good
  // 1 = marginal
  // 2 = snow/ice
  // 3 = cloudy

  var qa =
    image.select(
      "SummaryQA"
    );


  var good =
    qa.lte(1);


  var ndvi =
    image
    .select(
      "NDVI"
    )
    .multiply(
      0.0001
    )
    .rename(
      "NDVI"
    );


  var evi =
    image
    .select(
      "EVI"
    )
    .multiply(
      0.0001
    )
    .rename(
      "EVI_scaled"
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
        "system:time_start"
      ]
    );

}


// ============================================================================
// 8. MOD09A1 — REFLECTANCE + DERIVED INDICES
// ============================================================================

function prepMOD09(image) {


  var stateQA =
    image.select(
      "StateQA"
    );


  // ----------------------------------------------------------
  // Cloud state — bits 0–1
  // ----------------------------------------------------------

  var cloudState =
    stateQA
    .bitwiseAnd(
      3
    );


  var cloudOK =
    cloudState
    .eq(0)
    .or(
      cloudState.eq(3)
    );


  // ----------------------------------------------------------
  // Cloud shadow — bit 2
  // ----------------------------------------------------------

  var cloudShadow =
    stateQA
    .rightShift(2)
    .bitwiseAnd(1);


  // ----------------------------------------------------------
  // Internal cloud — bit 10
  // ----------------------------------------------------------

  var internalCloud =
    stateQA
    .rightShift(10)
    .bitwiseAnd(1);


  // ----------------------------------------------------------
  // Snow / ice — bit 12
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // Reflectance
  // ----------------------------------------------------------

  var reflectance =
    image
    .select(
      [
        "sur_refl_b01",   // RED
        "sur_refl_b02",   // NIR
        "sur_refl_b04",   // GREEN
        "sur_refl_b06"    // SWIR1
      ]
    )
    .multiply(
      0.0001
    )
    .rename(
      [
        "RED",
        "NIR",
        "GREEN",
        "SWIR1"
      ]
    );


  var red =
    reflectance.select(
      "RED"
    );


  var nir =
    reflectance.select(
      "NIR"
    );


  var green =
    reflectance.select(
      "GREEN"
    );


  var swir1 =
    reflectance.select(
      "SWIR1"
    );


  // ----------------------------------------------------------
  // Reflectance sanity mask
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // EVI2
  //
  // 2.5 × (NIR - RED)
  // -----------------
  // NIR + 2.4 RED + 1
  // ----------------------------------------------------------

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
      .add(
        1
      )

    )
    .rename(
      "EVI2"
    );


  // ----------------------------------------------------------
  // NDMI
  //
  // NIR - SWIR1
  // -----------
  // NIR + SWIR1
  //
  // Canopy / vegetation moisture proxy.
  // ----------------------------------------------------------

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
      "NDMI"
    );


  // ----------------------------------------------------------
  // NDWI
  //
  // GREEN - NIR
  // -----------
  // GREEN + NIR
  // ----------------------------------------------------------

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
      "NDWI"
    );


  // ----------------------------------------------------------
  // GCI
  //
  // NIR / GREEN - 1
  // ----------------------------------------------------------

  var gci =
    nir
    .divide(
      green
    )
    .subtract(
      1
    )
    .rename(
      "GCI"
    );


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
        "system:time_start"
      ]
    );

}


// ============================================================================
// 9. BUILD ONE VEGETATION COMPOSITE
// ============================================================================

function buildComposite(
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
      "day"
    );


  var start =
    center.advance(
      -HALF_WINDOW_DAYS,
      "day"
    );


  // filterDate() end date is exclusive,
  // so +9 includes center + 8 days.

  var end =
    center.advance(
      HALF_WINDOW_DAYS + 1,
      "day"
    );


  // ----------------------------------------------------------
  // MOD13 vegetation indices
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // MOD09 surface reflectance indices
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // NIRv
  //
  // NIR × NDVI
  // ----------------------------------------------------------

  var nirv =
    sr
    .select(
      "NIR"
    )
    .multiply(
      vi.select(
        "NDVI"
      )
    )
    .rename(
      "NIRv"
    );


  return vi

    .addBands(

      sr.select(
        [
          "EVI2",
          "NDMI",
          "NDWI",
          "GCI"
        ]
      )

    )

    .addBands(
      nirv
    )

    .select(
      [
        "NDVI",
        "EVI_scaled",
        "EVI2",
        "NDMI",
        "NDWI",
        "NIRv",
        "GCI"
      ]
    );

}


// ============================================================================
// 10. REDUCE ONE YEAR / DOY TO COUNTY MEANS
// ============================================================================

function reduceCheckpoint(
  year,
  doy,
  counties,
  cdlImage
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
      "day"
    );


  // ----------------------------------------------------------
  // Corn mask
  //
  // CDL class 1 = corn
  // ----------------------------------------------------------

  var cornMask =
    cdlImage
    .eq(1)
    .selfMask();


  // ----------------------------------------------------------
  // Apply same-year final-CDL mask
  // ----------------------------------------------------------

  var image =
    buildComposite(
      year,
      doy
    )
    .updateMask(
      cornMask
    );


  // ----------------------------------------------------------
  // County means
  // ----------------------------------------------------------

  var reduced =
    image.reduceRegions({

      collection:
        counties,

      reducer:
        ee.Reducer.mean(),

      scale:
        REDUCE_SCALE,

      tileScale:
        TILE_SCALE

    });


  // ----------------------------------------------------------
  // Keep only model columns
  // ----------------------------------------------------------

  return reduced.map(

    function(feature) {


      return ee.Feature(

        null,

        {

          GEOID:
            feature.get(
              "GEOID"
            ),

          date:
            center.format(
              "YYYY-MM-dd"
            ),

          year:
            year,

          NDVI:
            feature.get(
              "NDVI"
            ),

          EVI_scaled:
            feature.get(
              "EVI_scaled"
            ),

          EVI2:
            feature.get(
              "EVI2"
            ),

          NDMI:
            feature.get(
              "NDMI"
            ),

          NDWI:
            feature.get(
              "NDWI"
            ),

          NIRv:
            feature.get(
              "NIRv"
            ),

          GCI:
            feature.get(
              "GCI"
            )

        }

      );

    }

  );

}


// ============================================================================
// 11. LOAD FINAL CDLs
// ============================================================================

var CDL_2023 =
  getCDL(
    2023
  );


var CDL_2024 =
  getCDL(
    2024
  );


var CDL_2025 =
  getCDL(
    2025
  );


// ============================================================================
// 12. EXPORT 2023–2025 — ONE FILE PER STATE
// ============================================================================

STATES.forEach(

  function(stateInfo) {


    print(
      "Preparing:",
      stateInfo.name
    );


    var stateCounties =
      COUNTIES.filter(

        ee.Filter.eq(
          "STATEFP",
          stateInfo.fips
        )

      );


    var output =
      ee.FeatureCollection(
        []
      );


    // --------------------------------------------------------
    // 2023
    // --------------------------------------------------------

    DOYS.forEach(

      function(doy) {


        output =
          output.merge(

            reduceCheckpoint(

              2023,

              doy,

              stateCounties,

              CDL_2023

            )

          );

      }

    );


    // --------------------------------------------------------
    // 2024
    // --------------------------------------------------------

    DOYS.forEach(

      function(doy) {


        output =
          output.merge(

            reduceCheckpoint(

              2024,

              doy,

              stateCounties,

              CDL_2024

            )

          );

      }

    );


    // --------------------------------------------------------
    // 2025
    // --------------------------------------------------------

    DOYS.forEach(

      function(doy) {


        output =
          output.merge(

            reduceCheckpoint(

              2025,

              doy,

              stateCounties,

              CDL_2025

            )

          );

      }

    );


    // --------------------------------------------------------
    // Export
    // --------------------------------------------------------

    var exportName =

      "Corn_MODIS_same_year_2023_2025_"

      +

      stateInfo.name;


    print(
      exportName,
      output.size()
    );


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
        "CSV",

      selectors: [

        "GEOID",

        "date",

        "year",

        "NDVI",

        "EVI_scaled",

        "EVI2",

        "NDMI",

        "NDWI",

        "NIRv",

        "GCI"

      ]

    });

  }

);


// ============================================================================
// 13. OPTIONAL 2022 IOWA VALIDATION EXPORT
// ============================================================================

if (
  RUN_2022_VALIDATION
) {


  print(
    "Adding 2022 Iowa validation export..."
  );


  var CDL_2022 =
    getCDL(
      2022
    );


  var iowaCounties =
    COUNTIES.filter(

      ee.Filter.eq(
        "STATEFP",
        "19"
      )

    );


  var validation =
    ee.FeatureCollection(
      []
    );


  DOYS.forEach(

    function(doy) {


      validation =
        validation.merge(

          reduceCheckpoint(

            2022,

            doy,

            iowaCounties,

            CDL_2022

          )

        );

    }

  );


  Export.table.toDrive({

    collection:
      validation,

    description:
      "VALIDATION_Corn_MODIS_same_year_2022_Iowa",

    folder:
      DRIVE_FOLDER,

    fileNamePrefix:
      "VALIDATION_Corn_MODIS_same_year_2022_Iowa",

    fileFormat:
      "CSV",

    selectors: [

      "GEOID",

      "date",

      "year",

      "NDVI",

      "EVI_scaled",

      "EVI2",

      "NDMI",

      "NDWI",

      "NIRv",

      "GCI"

    ]

  });

}


// ============================================================================
// 14. MAP CHECK — 2025 CORN
// ============================================================================

var Iowa =
  ee.FeatureCollection(
    "TIGER/2018/States"
  )
  .filter(
    ee.Filter.eq(
      "STUSPS",
      "IA"
    )
  );


Map.centerObject(
  Iowa,
  7
);


Map.addLayer(

  CDL_2025
  .eq(1)
  .selfMask(),

  {},

  "2025 Final CDL Corn"

);


// ============================================================================
// 15. FINISHED
// ============================================================================

print(
  "===================================================="
);

print(
  "SAME-YEAR FINAL-CDL EXPORT SETUP COMPLETE"
);

print(
  "===================================================="
);

print(
  "Years:",
  YEARS
);

print(
  "States:",
  STATES.length
);

print(
  "DOYs:",
  DOYS.length
);

print(
  "Drive folder:",
  DRIVE_FOLDER
);

print(
  "Go to the Tasks tab and start the exports."
);