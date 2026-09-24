// ============================================================================
// CORN YIELD 2
// MODIS VEGETATION FROM CLEAN REBUILT ICDL MASKS
// ============================================================================
//
// OUTPUT EXAMPLE:
//
// Corn_MODIS_2025_CLEAN_REBUILT_JUNE.csv
// Corn_MODIS_2025_CLEAN_REBUILT_JULY.csv
// Corn_MODIS_2025_CLEAN_REBUILT_AUGUST.csv
//
// ICDL EXPECTATION:
//
//   1   = corn
//   0   = non-corn
//   255 = NoData, if present
//
// Only pixels == 1 are used.
====================
// The Python deployment experiment uses:
//
//   DOY 193,209 -> June ICDL, after June 30
//   DOY 225,241 -> July ICDL, after July 31
//   DOY 257,273 -> August ICDL, after August 31
//
// Full-season exports are allowed; Python limits vegetation
// features to the forecast cutoff before interpolation.
// Export timestamps must also respect actual data availability.

// ============================================================================
// 1. CHANGE THESE FOR EACH YEAR
// ============================================================================

var YEAR = 2025;


// Paste the exact GEE asset IDs after uploading your clean 30 m ICDLs.

var ICDL_ASSETS = {

  June:
    "projects/ee-gtellezgiron/assets/OurJune2025_CornBelt_CornMask30m_CLEAN_REBUILT",

  July:
    "projects/ee-gtellezgiron/assets/OurJuly2025_CornBelt_CornMask30m_CLEAN_REBUILT",

  August:
    "projects/ee-gtellezgiron/assets/OurAugust2025_CornBelt_CornMask30m_CLEAN_REBUILT"

};


var DRIVE_FOLDER =
  "CornYield_SameYear_ICDL_Experiment";


// ============================================================================
// 2. MODEL DOYS
// ============================================================================

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


// ============================================================================
// 3. CORN BELT COUNTIES
// ============================================================================

var CORN_BELT_STATES = [
  "19",  // Iowa
  "17",  // Illinois
  "18",  // Indiana
  "31",  // Nebraska
  "27",  // Minnesota
  "29",  // Missouri
  "20",  // Kansas
  "46",  // South Dakota
  "38",  // North Dakota
  "39",  // Ohio
  "55",  // Wisconsin
  "26"   // Michigan
];


var COUNTIES =
  ee.FeatureCollection(
    "TIGER/2018/Counties"
  )
  .filter(
    ee.Filter.inList(
      "STATEFP",
      CORN_BELT_STATES
    )
  );


print(
  "Corn Belt counties:",
  COUNTIES.size()
);


// ============================================================================
// 4. MODIS COLLECTIONS
// ============================================================================

var MOD13 =
  ee.ImageCollection(
    "MODIS/061/MOD13Q1"
  );


var MOD09 =
  ee.ImageCollection(
    "MODIS/061/MOD09A1"
  );


// ============================================================================
// 5. MOD13Q1
//
// NDVI + EVI
// ============================================================================

function prepMOD13(image) {


  // SummaryQA
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
// 6. MOD09A1
//
// RED
// NIR
// GREEN
// SWIR1
//
// -> EVI2
// -> NDMI
// -> NDWI
// -> GCI
// ============================================================================

function prepMOD09(image) {


  var stateQA =
    image.select(
      "StateQA"
    );


  // ----------------------------------------------------------
  // CLOUD STATE — bits 0-1
  // ----------------------------------------------------------

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


  // ----------------------------------------------------------
  // CLOUD SHADOW — bit 2
  // ----------------------------------------------------------

  var cloudShadow =
    stateQA
    .rightShift(2)
    .bitwiseAnd(1);


  // ----------------------------------------------------------
  // INTERNAL CLOUD — bit 10
  // ----------------------------------------------------------

  var internalCloud =
    stateQA
    .rightShift(10)
    .bitwiseAnd(1);


  // ----------------------------------------------------------
  // SNOW / ICE — bit 12
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
  // REFLECTANCE
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
  // BASIC REFLECTANCE VALIDITY
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
  // 2.5(NIR - RED)
  // ----------------
  // NIR + 2.4RED + 1
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
      .add(1)

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
  // Canopy / vegetation moisture proxy
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
// 7. BUILD VEGETATION COMPOSITE FOR ONE DOY
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


  // End date is exclusive.
  // +9 gives center + 8 days.

  var end =
    center.advance(
      HALF_WINDOW_DAYS + 1,
      "day"
    );


  // ----------------------------------------------------------
  // MOD13 NDVI / EVI
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
  // MOD09 reflectance
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
// 8. LOAD ICDL CORN MASK
// ============================================================================

function loadICDLCornMask(
  assetId
) {


  // Select band zero regardless of uploaded band name.

  var raw =
    ee.Image(
      assetId
    )
    .select(
      [0],
      ["ICDL"]
    );


  // Keep ONLY class 1 = corn.
  //
  // This automatically excludes:
  //
  // 0   = non-corn
  // 255 = NoData

  var corn =
    raw
    .eq(1)
    .selfMask()
    .rename(
      "corn"
    );


  return corn;

}


// ============================================================================
// 9. REDUCE ONE DOY TO COUNTY MEANS
// ============================================================================

function reduceCheckpoint(
  year,
  doy,
  cornMask
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


  var vegetation =
    buildComposite(
      year,
      doy
    );


  var masked =
    vegetation.updateMask(
      cornMask
    );


  var reduced =
    masked.reduceRegions({

      collection:
        COUNTIES,

      reducer:
        ee.Reducer.mean(),

      scale:
        REDUCE_SCALE,

      tileScale:
        TILE_SCALE

    });


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
// 10. EXPORT ONE ICDL MONTH
// ============================================================================

function exportICDL(
  month,
  assetId
) {


  print(
    "Preparing:",
    YEAR,
    month
  );


  print(
    "ICDL asset:",
    assetId
  );


  var cornMask =
    loadICDLCornMask(
      assetId
    );


  var output =
    ee.FeatureCollection(
      []
    );


  DOYS.forEach(

    function(doy) {


      output =
        output.merge(

          reduceCheckpoint(

            YEAR,

            doy,

            cornMask

          )

        );

    }

  );


  var exportName =

    "Corn_MODIS_"

    +

    YEAR

    +

    "_CLEAN_REBUILT_"

    +

    month.toUpperCase();


  print(
    exportName,
    "rows:",
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


// ============================================================================
// 11. CREATE THE THREE EXPORT TASKS
// ============================================================================

exportICDL(
  "June",
  ICDL_ASSETS.June
);


exportICDL(
  "July",
  ICDL_ASSETS.July
);


exportICDL(
  "August",
  ICDL_ASSETS.August
);


// ============================================================================
// 12. VISUAL CHECK
// ============================================================================
//
// Shows August corn over Iowa.
// This is only a sanity check and does not affect exports.
//

var STATES =
  ee.FeatureCollection(
    "TIGER/2018/States"
  );


var IOWA =
  STATES.filter(

    ee.Filter.eq(
      "STUSPS",
      "IA"
    )

  );


Map.centerObject(
  IOWA,
  7
);


Map.addLayer(

  loadICDLCornMask(
    ICDL_ASSETS.August
  ),

  {},

  YEAR +
  " August Clean ICDL Corn"

);


// ============================================================================
// 13. FINISHED
// ============================================================================

print(
  "===================================================="
);

print(
  "MODIS ICDL EXPORT SETUP COMPLETE"
);

print(
  "===================================================="
);


print(
  "YEAR:",
  YEAR
);


print(
  "Drive folder:",
  DRIVE_FOLDER
);


print(
  "Expected outputs:"
);


print(
  "Corn_MODIS_" +
  YEAR +
  "_CLEAN_REBUILT_JUNE.csv"
);


print(
  "Corn_MODIS_" +
  YEAR +
  "_CLEAN_REBUILT_JULY.csv"
);


print(
  "Corn_MODIS_" +
  YEAR +
  "_CLEAN_REBUILT_AUGUST.csv"
);


print(
  "Open the Tasks tab and start all three exports."
);
