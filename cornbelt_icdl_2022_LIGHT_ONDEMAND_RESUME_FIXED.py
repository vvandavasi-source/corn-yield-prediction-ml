# ============================================================
# CORN YIELD 2 — 2022 CLEAN 30 m ICDL
# LIGHTWEIGHT / ON-DEMAND SENTINEL-2 RUNNER
# ============================================================
#
# PURPOSE
# -------
# Re-run the validated clean 30 m ICDL method for TARGET YEAR 2022
# without permanently storing the full Corn Belt Sentinel-2 archive.
#
# Scientific design intentionally mirrors the validated 2023 30 m run:
#   - 12 Corn Belt states
#   - strict historical trusted pixels
#   - target 2022 labels from CDL 2016-2021
#   - Sentinel-2 L2A, May 1 -> Aug 31, 2022
#   - cloud < 10%, nodata < 10%
#   - one best scene per MGRS tile / calendar date
#   - 30 m EPSG:5070 grid, same grid phase as validated 2023 run
#   - Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDWI
#   - chronological time-series stack
#   - balanced multiclass sampling, max 1000 / class
#   - 100-tree Random Forest, seed 42
#   - binary output: corn=1, non-corn=0, NoData=255
#   - June / July / August checkpoints
#
# LIGHTWEIGHT STORAGE MODEL
# -------------------------
#   frozen scene manifest (kept)
#      -> download ONE tile's selected scenes (temporary)
#      -> train + classify June/July/August
#      -> keep tile ICDLs + trusted labels + logs
#      -> delete temporary Sentinel scenes after success
#      -> next tile
#
# This makes storage much smaller than the full-cache 2023 route while
# keeping the classification choices the same. The frozen manifest is
# critical: once created, reruns reuse the exact same scene selection.
#
# IMPORTANT
# ---------
# This produces OUR 2022 ICDLs and benchmarks them against:
#   - previous-year CDL = 2021
#   - final truth CDL    = 2022
#
# Published 2022 ICDL comparison is intentionally left for a later step,
# because the public 2022 GeoTIFFs must first be downloaded/uploaded or
# supplied as local references.
#
# SAFE TO STOP / RERUN
# --------------------
# - frozen manifest is reused
# - completed tile masks are skipped
# - zero-date checkpoints are treated as terminal/resolved
# - insufficient-label tiles are remembered across reruns
# - trusted-label rasters are reused
# - interrupted temporary scenes are reused if valid
# - temporary scenes are removed only after tile processing succeeds
# - state mosaics are rebuilt from scratch at the end (avoids stale mosaic bug)
# ============================================================

# ============================================================
# 0. ENVIRONMENT SETTINGS — BEFORE RASTERIO
# ============================================================

import os

os.environ["GDAL_NUM_THREADS"] = "ALL_CPUS"
os.environ["GDAL_CACHEMAX"] = "4096"
os.environ["GDAL_HTTP_MULTIRANGE"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_TCP_KEEPALIVE"] = "YES"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
os.environ["CPL_VSIL_CURL_CACHE_SIZE"] = str(256 * 1024 * 1024)

# ============================================================
# 1. IMPORTS
# ============================================================

import io
import gc
import re
import time
import shutil
import zipfile
import traceback
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio

from pystac_client import Client
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from sklearn.ensemble import RandomForestClassifier

from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.features import geometry_mask
from rasterio.merge import merge
from affine import Affine

warnings.filterwarnings("ignore")

# ============================================================
# 2. SETTINGS
# ============================================================

ROOT_DIR = r"D:\corn_yield"
YEAR = 2022
PROJECT_ID = "ee-gtellezgiron"

START_DATE = "2022-05-01"
END_DATE = "2022-08-31"

CHECKPOINTS = OrderedDict(
    [
        ("June", "2022-06-30"),
        ("July", "2022-07-31"),
        ("August", "2022-08-31"),
    ]
)

# Paper + leakage-free target-year logic:
# 2022 training labels use only CDL 2016-2021.
HISTORICAL_YEARS = list(range(2016, 2022))
PREVIOUS_YEAR = 2021
FINAL_YEAR = 2022

STATE_FIPS = {
    "IA": "19",
    "IL": "17",
    "IN": "18",
    "NE": "31",
    "MN": "27",
    "MO": "29",
    "KS": "20",
    "SD": "46",
    "ND": "38",
    "OH": "39",
    "WI": "55",
    "MI": "26",
}

# Reuse a validated 30 m EPSG:5070 raster ONLY as a grid anchor.
# The year of the template has no role in classification.
GRID_TEMPLATE_CANDIDATES = [
    os.path.join(
        ROOT_DIR,
        "Our_ICDL_2023_Iowa",
        "statewide",
        "OurAugust2023_Iowa_CornMask30m.tif",
    ),
    os.path.join(
        ROOT_DIR,
        "Our_ICDL_2023_CornBelt",
        "states_clean_rebuilt_2023",
        "August",
        "IA",
        "OurAugust2023_IA_CornMask30m_CLEAN_REBUILT.tif",
    ),
    os.path.join(
        ROOT_DIR,
        "Our_ICDL_2023_CornBelt",
        "states",
        "IA",
        "OurAugust2023_IA_CornMask30m.tif",
    ),
]

# Keep the small manifest/inventory in sentinel_cache/2022 so our existing
# downstream clean-rebuild tooling can use the same convention. Large scene
# rasters are NOT kept here.
MANIFEST_ROOT = os.path.join(ROOT_DIR, "sentinel_cache", str(YEAR))
SCENE_MANIFEST = os.path.join(MANIFEST_ROOT, "cornbelt_2022_scene_manifest.csv")
TILE_INVENTORY = os.path.join(MANIFEST_ROOT, "cornbelt_2022_tile_inventory.csv")

OUT_ROOT = os.path.join(ROOT_DIR, "Our_ICDL_2022_CornBelt")
TILE_OUT = os.path.join(OUT_ROOT, "tiles")
MODEL_DIR = os.path.join(OUT_ROOT, "models")
TRUSTED_CACHE = os.path.join(OUT_ROOT, "trusted_label_cache")
STATE_CLEAN_ROOT = os.path.join(OUT_ROOT, "states_clean_rebuilt_2022")
REFERENCE_CACHE = os.path.join(OUT_ROOT, "reference_cache")
TABLE_DIR = os.path.join(OUT_ROOT, "tables")

# Only one tile's Sentinel scenes live here at a time.
TEMP_ROOT = os.path.join(ROOT_DIR, "temp_icdl_2022_light")

TIGER_FILE = os.path.join(ROOT_DIR, "tl_2023_us_state.zip")

for folder in [
    MANIFEST_ROOT,
    OUT_ROOT,
    TILE_OUT,
    MODEL_DIR,
    TRUSTED_CACHE,
    STATE_CLEAN_ROOT,
    REFERENCE_CACHE,
    TABLE_DIR,
    TEMP_ROOT,
]:
    os.makedirs(folder, exist_ok=True)

# Scene selection: same as validated 2023 cache.
MAX_CLOUD_PCT = 10.0
MAX_NODATA_PCT = 10.0
MIN_TILE_OVERLAP_KM2 = 1.0

RAW_ASSETS = {
    "Blue": "blue",
    "Green": "green",
    "Red": "red",
    "NIR": "nir",
    "SWIR1": "swir16",
    "SWIR2": "swir22",
}
REQUIRED_ASSETS = list(RAW_ASSETS.values()) + ["scl"]
BAND_NAMES = list(RAW_ASSETS.keys()) + ["SCL"]
VARIABLE_ORDER = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI", "NDWI"]

# Classifier settings: same clean 30 m design.
N_TREES = 100
MAX_SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42
MIN_TRAIN_POINTS = 50
MIN_TRAIN_CLASSES = 2
MIN_CORN_POINTS = 5
MIN_DATES_PER_CHECKPOINT = 1

# I/O / processing.
CACHE_BLOCK_SIZE = 1024
BLOCK_SIZE = 1536
EE_CHUNK_SIZE = 3072
METRIC_BLOCK_SIZE = 2048
MAX_SCENE_ATTEMPTS = 3
MAX_EE_ATTEMPTS = 5

CACHE_NODATA = -32768
OUTPUT_NODATA = 255
REF_NODATA = 255
BAD_SCL = {0, 1, 3, 8, 9, 10, 11}

# Lightweight behavior.
CLEAN_TEMP_AFTER_SUCCESS = True
KEEP_MODELS = True
RUN_STATE_BENCHMARK = True

STATUS_CSV = os.path.join(TABLE_DIR, "CornBelt_2022_generation_status.csv")
STATE_SUMMARY_CSV = os.path.join(TABLE_DIR, "CornBelt_2022_clean_state_mask_summary.csv")
STATE_BENCHMARK_CSV = os.path.join(TABLE_DIR, "CornBelt_2022_state_benchmark.csv")
BELT_BENCHMARK_CSV = os.path.join(TABLE_DIR, "CornBelt_2022_pooled_benchmark.csv")
BENCHMARK_FAILURES_CSV = os.path.join(TABLE_DIR, "CornBelt_2022_benchmark_failures.csv")

# ============================================================
# 3. BASIC HELPERS
# ============================================================


def header(text):
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def valid_raster(path, count=None, dtype=None, width=None, height=None, transform=None, crs=None):
    if not path or not os.path.exists(path):
        return False
    try:
        with rasterio.open(path) as src:
            ok = src.width > 0 and src.height > 0
            if count is not None:
                ok = ok and src.count == count
            if dtype is not None:
                ok = ok and src.dtypes[0] == dtype
            if width is not None:
                ok = ok and src.width == int(width)
            if height is not None:
                ok = ok and src.height == int(height)
            if transform is not None:
                ok = ok and src.transform.almost_equals(transform)
            if crs is not None:
                ok = ok and src.crs == crs
            return bool(ok)
    except Exception:
        return False


def save_status(rows):
    pd.DataFrame(rows).to_csv(STATUS_CSV, index=False)


def response_to_array(content):
    if len(content) >= 2 and content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise RuntimeError("GEE zip contained no TIFF.")
            tif_bytes = z.read(tif_names[0])
            with MemoryFile(tif_bytes) as mem:
                with mem.open() as src:
                    return src.read(1)
    with MemoryFile(content) as mem:
        with mem.open() as src:
            return src.read(1)


def empty_counts():
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def add_counts(target, source):
    for key in ["TP", "FP", "FN", "TN"]:
        target[key] += int(source[key])


def update_counts(counts, pred, truth):
    counts["TP"] += int(np.count_nonzero(pred & truth))
    counts["FP"] += int(np.count_nonzero(pred & ~truth))
    counts["FN"] += int(np.count_nonzero(~pred & truth))
    counts["TN"] += int(np.count_nonzero(~pred & ~truth))


def counts_to_metrics(counts, pixel_area_km2):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    accuracy = (tp + tn) / n if n else np.nan
    predicted = tp + fp
    actual = tp + fn
    bias = (predicted - actual) / actual * 100 if actual else np.nan
    return {
        **counts,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Accuracy": accuracy,
        "Predicted_corn_km2": predicted * pixel_area_km2,
        "Actual_corn_km2": actual * pixel_area_km2,
        "Corn_area_bias_pct": bias,
        "N": n,
    }


def transform_geom(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shapely_transform(transformer.transform, geom)


def get_mgrs_tile(item):
    props = item.properties
    grid_code = props.get("grid:code")
    if grid_code:
        text = str(grid_code)
        if text.upper().startswith("MGRS-"):
            return text.split("-", 1)[1]
        return text
    direct = props.get("s2:mgrs_tile")
    if direct:
        return str(direct)
    zone = props.get("mgrs:utm_zone")
    band = props.get("mgrs:latitude_band")
    square = props.get("mgrs:grid_square")
    if zone is not None and band is not None and square is not None:
        return f"{int(zone):02d}{band}{square}"
    match = re.search(r"\d{2}[A-Z]{3}", item.id)
    return match.group(0) if match else None


def get_radiometry(asset):
    raster_bands = asset.extra_fields.get("raster:bands", [])
    meta = raster_bands[0] if raster_bands else {}
    scale = float(meta.get("scale", 0.0001))
    offset = float(meta.get("offset", 0.0))
    return scale, offset


def full_tile_geometry_from_href(href):
    from shapely.geometry import box
    with rasterio.open(href) as src:
        native = box(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        return shapely_transform(transformer.transform, native)


def snap_geometry_bounds_to_grid(geometry, anchor_transform):
    xmin, ymin, xmax, ymax = geometry.bounds
    inv = ~anchor_transform
    c1, r1 = inv * (xmin, ymin)
    c2, r2 = inv * (xmax, ymax)
    col_min = int(np.floor(min(c1, c2)))
    col_max = int(np.ceil(max(c1, c2)))
    row_min = int(np.floor(min(r1, r2)))
    row_max = int(np.ceil(max(r1, r2)))
    win = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    transform = window_transform(win, anchor_transform)
    return int(win.width), int(win.height), transform

# ============================================================
# 4. TARGET GRID + STATE BOUNDARIES
# ============================================================

header("TARGET 30 m GRID")

GRID_TEMPLATE = next(
    (p for p in GRID_TEMPLATE_CANDIDATES if valid_raster(p, count=1, dtype="uint8")),
    None,
)

if GRID_TEMPLATE is None:
    raise FileNotFoundError(
        "Could not find a validated 30 m grid-template raster.\n"
        "Expected one of:\n  " + "\n  ".join(GRID_TEMPLATE_CANDIDATES)
    )

with rasterio.open(GRID_TEMPLATE) as src:
    TARGET_CRS = src.crs
    GRID_ANCHOR = src.transform

print("Grid template:", GRID_TEMPLATE)
print("Target CRS:", TARGET_CRS)
print("Grid anchor:", GRID_ANCHOR)

header("12-STATE CORN BELT")

if not os.path.exists(TIGER_FILE):
    url = "https://www2.census.gov/geo/tiger/TIGER2023/STATE/tl_2023_us_state.zip"
    print("Downloading TIGER state boundaries...")
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    with open(TIGER_FILE, "wb") as f:
        f.write(r.content)

states = gpd.read_file(TIGER_FILE)
selected = states[states["STATEFP"].astype(str).isin(STATE_FIPS.values())].copy()
fips_to_abbr = {v: k for k, v in STATE_FIPS.items()}
selected["abbr"] = selected["STATEFP"].astype(str).map(fips_to_abbr)
selected = selected[selected["abbr"].notna()].copy()
selected_4326 = selected.to_crs("EPSG:4326")
selected_target = selected.to_crs(TARGET_CRS)

cornbelt_union_4326 = selected_4326.geometry.union_all()
cornbelt_union_target = selected_target.geometry.union_all()
state_geom = {row["abbr"]: row.geometry for _, row in selected_target.iterrows()}
state_area_km2 = {abbr: geom.area / 1e6 for abbr, geom in state_geom.items()}

print("States:", ", ".join(STATE_FIPS.keys()))
print("Corn Belt area:", f"{cornbelt_union_target.area / 1e6:,.0f} km²")

# ============================================================
# 5. FROZEN 2022 SENTINEL SCENE MANIFEST
# ============================================================


def build_manifest_if_needed():
    if os.path.exists(SCENE_MANIFEST) and os.path.exists(TILE_INVENTORY):
        print("✓ Reusing frozen 2022 scene manifest:")
        print(" ", SCENE_MANIFEST)
        print("✓ Reusing tile inventory:")
        print(" ", TILE_INVENTORY)
        return

    header("BUILDING FROZEN 2022 SENTINEL-2 MANIFEST")
    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=list(cornbelt_union_4326.bounds),
        datetime=f"{START_DATE}/{END_DATE}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
    )
    all_items = list(search.items())
    print("Scenes returned by bbox/cloud query:", len(all_items))

    scene_rows = []
    for item in all_items:
        tile = get_mgrs_tile(item)
        if not tile:
            continue
        if not all(key in item.assets for key in REQUIRED_ASSETS):
            continue

        cloud = item.properties.get("eo:cloud_cover")
        nodata = item.properties.get("s2:nodata_pixel_percentage")
        if cloud is None or nodata is None:
            continue
        cloud = float(cloud)
        nodata = float(nodata)
        if cloud >= MAX_CLOUD_PCT or nodata >= MAX_NODATA_PCT:
            continue

        try:
            geom = shape(item.geometry) if item.geometry else full_tile_geometry_from_href(item.assets["blue"].href)
        except Exception:
            continue
        if geom.is_empty or not geom.intersects(cornbelt_union_4326):
            continue

        dt = pd.Timestamp(item.datetime)
        row = {
            "tile": tile,
            "date": str(dt.date()),
            "datetime": dt.isoformat(),
            "cloud_pct": cloud,
            "nodata_pct": nodata,
            "item_id": item.id,
        }

        for band_name, asset_key in RAW_ASSETS.items():
            asset = item.assets[asset_key]
            scale, offset = get_radiometry(asset)
            row[f"{band_name}_href"] = asset.href
            row[f"{band_name}_scale"] = scale
            row[f"{band_name}_offset"] = offset
        row["SCL_href"] = item.assets["scl"].href
        scene_rows.append(row)

    scene_df = pd.DataFrame(scene_rows)
    if scene_df.empty:
        raise RuntimeError("No qualifying Sentinel-2 scenes were found for 2022.")

    scene_df = (
        scene_df.sort_values(["tile", "date", "nodata_pct", "cloud_pct"])
        .drop_duplicates(subset=["tile", "date"], keep="first")
        .sort_values(["tile", "date"])
        .reset_index(drop=True)
    )

    tile_rows = []
    keep_tiles = []

    for tile, group in scene_df.groupby("tile", sort=True):
        first_href = group.iloc[0]["Blue_href"]
        tile_geom_4326 = full_tile_geometry_from_href(first_href)
        tile_geom_target = transform_geom(tile_geom_4326, "EPSG:4326", TARGET_CRS)
        roi_target = tile_geom_target.intersection(cornbelt_union_target)
        if roi_target.is_empty:
            continue

        overlap_km2 = roi_target.area / 1e6
        if overlap_km2 < MIN_TILE_OVERLAP_KM2:
            continue

        width, height, transform = snap_geometry_bounds_to_grid(roi_target, GRID_ANCHOR)
        overlapping_states = []
        for _, st in selected_target.iterrows():
            overlap = tile_geom_target.intersection(st.geometry)
            if not overlap.is_empty and overlap.area / 1e6 >= MIN_TILE_OVERLAP_KM2:
                overlapping_states.append(st["abbr"])

        keep_tiles.append(tile)
        tile_rows.append(
            {
                "tile": tile,
                "states": ";".join(sorted(overlapping_states)),
                "overlap_km2": overlap_km2,
                "width": width,
                "height": height,
                "selected_dates": len(group),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "transform_a": transform.a,
                "transform_b": transform.b,
                "transform_c": transform.c,
                "transform_d": transform.d,
                "transform_e": transform.e,
                "transform_f": transform.f,
            }
        )

    scene_df = scene_df[scene_df["tile"].isin(keep_tiles)].copy().reset_index(drop=True)
    tile_df = pd.DataFrame(tile_rows).sort_values("overlap_km2", ascending=False)

    scene_df.to_csv(SCENE_MANIFEST, index=False)
    tile_df.to_csv(TILE_INVENTORY, index=False)

    print("Selected tile-date scenes:", len(scene_df))
    print("Corn Belt MGRS tiles:", len(tile_df))
    print("Frozen manifest:", SCENE_MANIFEST)
    print("Tile inventory:", TILE_INVENTORY)


build_manifest_if_needed()
scene_manifest = pd.read_csv(SCENE_MANIFEST)
tile_inventory = pd.read_csv(TILE_INVENTORY)
scene_manifest["date_obj"] = pd.to_datetime(scene_manifest["date"]).dt.date
scene_manifest = scene_manifest.sort_values(["tile", "date_obj"]).reset_index(drop=True)

# Rebuild deterministic tile metadata from frozen inventory + source bounds.
tile_info = {}
tile_states = {}
for _, row in tile_inventory.iterrows():
    tile = str(row["tile"])
    group = scene_manifest[scene_manifest["tile"].astype(str) == tile].sort_values("date_obj")
    if group.empty:
        continue
    first_href = group.iloc[0]["Blue_href"]
    tile_geom_4326 = full_tile_geometry_from_href(first_href)
    tile_geom_target = transform_geom(tile_geom_4326, "EPSG:4326", TARGET_CRS)
    roi_target = tile_geom_target.intersection(cornbelt_union_target)
    tr = Affine(
        float(row["transform_a"]),
        float(row["transform_b"]),
        float(row["transform_c"]),
        float(row["transform_d"]),
        float(row["transform_e"]),
        float(row["transform_f"]),
    )
    tile_info[tile] = {
        "roi_target": roi_target,
        "width": int(row["width"]),
        "height": int(row["height"]),
        "transform": tr,
        "overlap_km2": float(row["overlap_km2"]),
        "states": [x for x in str(row["states"]).split(";") if x],
    }
    tile_states[tile] = tile_info[tile]["states"]

all_tiles = tile_inventory.sort_values("overlap_km2", ascending=False)["tile"].astype(str).tolist()
print("\nFrozen 2022 scenes:", len(scene_manifest))
print("Frozen MGRS tiles:", len(all_tiles))

# ============================================================
# 6. EARTH ENGINE — STRICT 2022 TRUSTED LABELS
# ============================================================

header("EARTH ENGINE — STRICT 2022 TRUSTED LABELS")

try:
    ee.Initialize(project=PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

print("✓ Earth Engine initialized:", PROJECT_ID)


def cdl_image(year):
    return (
        ee.ImageCollection("USDA/NASS/CDL")
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("cropland")
        .toInt16()
    )


def build_trusted_target_image():
    imgs = [cdl_image(y) for y in HISTORICAL_YEARS]

    # Stable land-cover/crop class across all six historical years.
    stable = ee.Image(1)
    for img in imgs[1:]:
        stable = stable.And(img.eq(imgs[0]))

    # Strict corn-soy alternation only.
    cornsoy = ee.Image(1)
    for img in imgs:
        cornsoy = cornsoy.And(img.eq(1).Or(img.eq(5)))

    alternating = cornsoy
    for i in range(1, len(imgs)):
        alternating = alternating.And(imgs[i].neq(imgs[i - 1]))

    last = imgs[-1]  # 2021
    next_rotation = last.eq(1).multiply(5).add(last.eq(5).multiply(1))

    trusted = ee.Image(0)
    trusted = trusted.where(stable, last)
    trusted = trusted.where(alternating, next_rotation)
    return trusted.rename("trusted_class").toInt16()


TRUSTED_EE = build_trusted_target_image()
CDL_PREVIOUS_EE = cdl_image(PREVIOUS_YEAR)
CDL_FINAL_EE = cdl_image(FINAL_YEAR)

# ============================================================
# 7. EXACT-GRID EE DOWNLOAD
# ============================================================


def download_ee_exact(
    ee_image,
    template_path,
    output_path,
    local_geometry,
    out_dtype,
    out_nodata,
    mode,
    label,
):
    if valid_raster(output_path, count=1, dtype=out_dtype):
        with rasterio.open(template_path) as t, rasterio.open(output_path) as o:
            if (
                t.width == o.width
                and t.height == o.height
                and t.transform.almost_equals(o.transform)
                and t.crs == o.crs
            ):
                return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".partial.tif"
    if os.path.exists(temp_path):
        os.remove(temp_path)

    with rasterio.open(template_path) as template:
        width = template.width
        height = template.height
        transform = template.transform
        crs = template.crs

    if mode == "trusted":
        image = ee_image.unmask(0).toInt16()
    elif mode == "corn":
        image = ee_image.eq(1).unmask(REF_NODATA).toUint8()
    else:
        raise ValueError("Unknown mode")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": out_dtype,
        "crs": crs,
        "transform": transform,
        "nodata": out_nodata,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    row_blocks = int(np.ceil(height / EE_CHUNK_SIZE))
    col_blocks = int(np.ceil(width / EE_CHUNK_SIZE))
    total = row_blocks * col_blocks
    counter = 0

    with rasterio.open(temp_path, "w", **profile) as dst:
        for row_off in range(0, height, EE_CHUNK_SIZE):
            for col_off in range(0, width, EE_CHUNK_SIZE):
                counter += 1
                h = min(EE_CHUNK_SIZE, height - row_off)
                w = min(EE_CHUNK_SIZE, width - col_off)
                win = Window(col_off, row_off, w, h)
                chunk_transform = window_transform(win, transform)
                print(f"\r      {label}: chunk {counter}/{total}", end="", flush=True)

                params = {
                    "crs": crs.to_string(),
                    "crs_transform": [
                        chunk_transform.a,
                        chunk_transform.b,
                        chunk_transform.c,
                        chunk_transform.d,
                        chunk_transform.e,
                        chunk_transform.f,
                    ],
                    "dimensions": f"{w}x{h}",
                    "format": "GEO_TIFF",
                }

                arr = None
                last_error = None
                for attempt in range(1, MAX_EE_ATTEMPTS + 1):
                    try:
                        url = image.getDownloadURL(params)
                        response = requests.get(url, timeout=300)
                        response.raise_for_status()
                        arr = response_to_array(response.content)
                        if arr.shape != (h, w):
                            raise RuntimeError(f"Unexpected shape {arr.shape}; expected {(h, w)}")
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < MAX_EE_ATTEMPTS:
                            wait = 5 * attempt
                            print(f"\n        EE attempt {attempt} failed: {exc}")
                            print(f"        retrying in {wait} sec...")
                            time.sleep(wait)

                if last_error is not None:
                    raise RuntimeError(f"EE download failed for {label}") from last_error

                inside = geometry_mask(
                    [mapping(local_geometry)],
                    out_shape=(h, w),
                    transform=chunk_transform,
                    invert=True,
                )

                if mode == "trusted":
                    out = arr.astype(np.int16, copy=False)
                    out[~inside] = 0
                else:
                    out = arr.astype(np.uint8, copy=False)
                    out[~inside] = REF_NODATA

                dst.write(out, 1, window=win)

    print()
    os.replace(temp_path, output_path)
    return output_path

# ============================================================
# 8. TEMPORARY SENTINEL TILE CACHE
# ============================================================


def valid_temp_scene(path, info):
    return valid_raster(
        path,
        count=7,
        dtype="int16",
        width=info["width"],
        height=info["height"],
        transform=info["transform"],
        crs=TARGET_CRS,
    )


def cache_scene_from_manifest(row, tile, output_path):
    info = tile_info[tile]
    width = info["width"]
    height = info["height"]
    transform = info["transform"]
    roi_target = info["roi_target"]

    if valid_temp_scene(output_path, info):
        return "SKIPPED"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".partial.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 7,
        "dtype": "int16",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": CACHE_NODATA,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 4,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    for attempt in range(1, MAX_SCENE_ATTEMPTS + 1):
        if os.path.exists(temp_path):
            os.remove(temp_path)
        sources = []
        vrts = []
        try:
            reflectance_sources = []
            for band_name in RAW_ASSETS:
                href = row[f"{band_name}_href"]
                scale = float(row[f"{band_name}_scale"])
                offset = float(row[f"{band_name}_offset"])
                src = rasterio.open(href)
                sources.append(src)
                vrt = WarpedVRT(
                    src,
                    crs=TARGET_CRS,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                    nodata=src.nodata,
                )
                vrts.append(vrt)
                reflectance_sources.append((band_name, vrt, scale, offset))

            scl_src = rasterio.open(row["SCL_href"])
            sources.append(scl_src)
            scl_vrt = WarpedVRT(
                scl_src,
                crs=TARGET_CRS,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
                nodata=scl_src.nodata,
            )
            vrts.append(scl_vrt)

            row_blocks = int(np.ceil(height / CACHE_BLOCK_SIZE))
            col_blocks = int(np.ceil(width / CACHE_BLOCK_SIZE))
            total_blocks = row_blocks * col_blocks
            counter = 0

            with rasterio.open(temp_path, "w", **profile) as dst:
                for i, name in enumerate(BAND_NAMES, start=1):
                    dst.set_band_description(i, name)

                dst.update_tags(
                    item_id=str(row["item_id"]),
                    mgrs_tile=tile,
                    acquisition_date=str(row["date"]),
                    source="Element84 Earth Search sentinel-2-l2a",
                    storage="temporary on-demand 2022 light runner",
                    reflectance_encoding="bands 1-6 = surface reflectance * 10000",
                    scl_encoding="band 7 = Sentinel-2 Scene Classification Layer",
                )

                for row_off in range(0, height, CACHE_BLOCK_SIZE):
                    for col_off in range(0, width, CACHE_BLOCK_SIZE):
                        counter += 1
                        h = min(CACHE_BLOCK_SIZE, height - row_off)
                        w = min(CACHE_BLOCK_SIZE, width - col_off)
                        win = Window(col_off, row_off, w, h)
                        block_transform = window_transform(win, transform)
                        inside = geometry_mask(
                            [mapping(roi_target)],
                            out_shape=(h, w),
                            transform=block_transform,
                            invert=True,
                        )

                        print(
                            f"\r      {tile} {str(row['date'])}: block {counter}/{total_blocks}",
                            end="",
                            flush=True,
                        )

                        for band_index, (_, vrt, scale, offset) in enumerate(reflectance_sources, start=1):
                            raw = vrt.read(1, window=win, masked=True).astype(np.float32)
                            reflectance = raw.filled(np.nan) * scale + offset
                            valid = inside & np.isfinite(reflectance)
                            out = np.full((h, w), CACHE_NODATA, dtype=np.int16)
                            if valid.any():
                                encoded = np.rint(reflectance[valid] * 10000.0)
                                encoded = np.clip(encoded, -32767, 32767).astype(np.int16)
                                out[valid] = encoded
                            dst.write(out, band_index, window=win)

                        scl = scl_vrt.read(1, window=win, masked=True)
                        scl_mask = np.ma.getmaskarray(scl)
                        scl_values = scl.astype(np.int16).filled(CACHE_NODATA)
                        scl_valid = inside & (~scl_mask)
                        scl_out = np.full((h, w), CACHE_NODATA, dtype=np.int16)
                        scl_out[scl_valid] = scl_values[scl_valid]
                        dst.write(scl_out, 7, window=win)
                print()

            for vrt in reversed(vrts):
                vrt.close()
            for src in reversed(sources):
                src.close()

            os.replace(temp_path, output_path)
            if not valid_temp_scene(output_path, info):
                raise RuntimeError("Temporary scene validation failed after write.")
            return "CACHED"

        except Exception as exc:
            print(f"\n      attempt {attempt}/{MAX_SCENE_ATTEMPTS} failed: {exc}")
            traceback.print_exc(limit=1)
            for vrt in reversed(vrts):
                try:
                    vrt.close()
                except Exception:
                    pass
            for src in reversed(sources):
                try:
                    src.close()
                except Exception:
                    pass
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            if attempt < MAX_SCENE_ATTEMPTS:
                wait = 10 * attempt
                print(f"      retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise


def ensure_tile_temp_scenes(tile, group):
    tile_temp = os.path.join(TEMP_ROOT, tile)
    os.makedirs(tile_temp, exist_ok=True)
    paths = []
    for _, row in group.iterrows():
        date_text = pd.Timestamp(row["date"]).strftime("%Y%m%d")
        out = os.path.join(tile_temp, f"{date_text}.tif")
        print(
            f"    Sentinel scene {date_text} | cloud={float(row['cloud_pct']):.2f}% "
            f"nodata={float(row['nodata_pct']):.2f}%"
        )
        result = cache_scene_from_manifest(row, tile, out)
        print("      ✓", result)
        paths.append(out)
    return paths

# ============================================================
# 9. TRAINING / CLASSIFICATION — SAME 30 m METHOD
# ============================================================


def get_balanced_training_pixels(trusted_path):
    with rasterio.open(trusted_path) as src:
        labels = src.read(1)
    valid = labels > 0
    classes = np.unique(labels[valid])
    rng = np.random.default_rng(RANDOM_SEED)
    selected_flat = []
    class_counts = {}
    for cls in classes:
        idx = np.flatnonzero(labels.ravel() == cls)
        class_counts[int(cls)] = int(len(idx))
        if len(idx) > MAX_SAMPLES_PER_CLASS:
            idx = rng.choice(idx, size=MAX_SAMPLES_PER_CLASS, replace=False)
        selected_flat.extend(idx.tolist())
    selected_flat = np.asarray(selected_flat, dtype=np.int64)
    rows, cols = np.unravel_index(selected_flat, labels.shape)
    y = labels.ravel()[selected_flat].astype(np.int16)
    return rows, cols, y, class_counts


def build_training_cube(cache_paths, rows, cols):
    n_points = len(rows)
    n_dates = len(cache_paths)
    cube = np.full((n_points, n_dates, len(VARIABLE_ORDER)), np.nan, dtype=np.float32)
    for d, path in enumerate(cache_paths):
        date_text = os.path.splitext(os.path.basename(path))[0]
        print(f"      training temp scene {d + 1}/{n_dates}: {date_text}")
        with rasterio.open(path) as src:
            data = src.read()
        sample = data[:, rows, cols]
        refl_raw = sample[:6].astype(np.float32)
        scl = sample[6].astype(np.int16)
        good = scl != CACHE_NODATA
        good &= ~np.isin(scl, list(BAD_SCL))
        good &= np.all(refl_raw != CACHE_NODATA, axis=0)
        refl = refl_raw / 10000.0
        refl[:, ~good] = np.nan
        blue, green, red, nir, swir1, swir2 = refl
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndwi = (green - nir) / (green + nir + 1e-8)
        features = [blue, green, red, nir, swir1, swir2, ndvi, ndwi]
        for v, values in enumerate(features):
            cube[:, d, v] = values
        del data, sample, refl_raw, refl
        gc.collect()
    return cube


def prepare_checkpoint_matrix(full_cube, date_indices):
    subset = full_cube[:, date_indices, :].copy()
    n_vars = subset.shape[2]
    point_median = np.nanmedian(subset, axis=1)
    global_median = np.nanmedian(subset.reshape(-1, n_vars), axis=0)
    global_median = np.where(np.isfinite(global_median), global_median, 0.0).astype(np.float32)
    for v in range(n_vars):
        fill = point_median[:, v].copy()
        fill[~np.isfinite(fill)] = global_median[v]
        for d in range(subset.shape[1]):
            missing = ~np.isfinite(subset[:, d, v])
            subset[missing, d, v] = fill[missing]
    return subset.reshape(subset.shape[0], -1).astype(np.float32), global_median


def train_models(full_cube, labels, dates, tile):
    models = {}
    medians = {}
    checkpoint_indices = {}
    stats = []
    item_dates = [pd.Timestamp(d).date() for d in dates]

    for checkpoint, cutoff_text in CHECKPOINTS.items():
        cutoff = pd.Timestamp(cutoff_text).date()
        indices = [i for i, d in enumerate(item_dates) if d <= cutoff]
        checkpoint_indices[checkpoint] = indices
        print(f"\n    {checkpoint}: {len(indices)} dates through {cutoff}")
        if len(indices) < MIN_DATES_PER_CHECKPOINT:
            stats.append(
                {"Tile": tile, "Checkpoint": checkpoint, "Status": "INSUFFICIENT_DATES", "Dates": len(indices)}
            )
            continue

        X, median = prepare_checkpoint_matrix(full_cube, indices)
        rf = RandomForestClassifier(
            n_estimators=N_TREES,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        rf.fit(X, labels)
        models[checkpoint] = rf
        medians[checkpoint] = median
        status = "LOW_DATE_COUNT" if len(indices) == 1 else "TRAINED"
        stats.append(
            {
                "Tile": tile,
                "Checkpoint": checkpoint,
                "Status": status,
                "Dates": len(indices),
                "Features": X.shape[1],
                "Training_points": len(labels),
                "Classes": len(np.unique(labels)),
            }
        )
        print("      ✓ RF trained")
        print("      Features:", X.shape[1])
        print("      Training points:", f"{len(labels):,}")
    return models, medians, checkpoint_indices, stats


def read_cached_block(open_sources, window):
    h = int(window.height)
    w = int(window.width)
    cube = np.full((len(open_sources), len(VARIABLE_ORDER), h, w), np.nan, dtype=np.float32)
    for d, src in enumerate(open_sources):
        data = src.read(window=window)
        refl_raw = data[:6].astype(np.float32)
        scl = data[6].astype(np.int16)
        good = scl != CACHE_NODATA
        good &= ~np.isin(scl, list(BAD_SCL))
        good &= np.all(refl_raw != CACHE_NODATA, axis=0)
        refl = refl_raw / 10000.0
        refl[:, ~good] = np.nan
        blue, green, red, nir, swir1, swir2 = refl
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndwi = (green - nir) / (green + nir + 1e-8)
        features = [blue, green, red, nir, swir1, swir2, ndvi, ndwi]
        for v, values in enumerate(features):
            cube[d, v] = values
    return cube


def prepare_block_features(full_cube, date_indices, global_median):
    subset = full_cube[date_indices, :, :, :].copy()
    has_data = np.any(np.isfinite(subset), axis=(0, 1))
    temporal_median = np.nanmedian(subset, axis=0)
    for v in range(subset.shape[1]):
        fill = temporal_median[v]
        fill = np.where(np.isfinite(fill), fill, global_median[v])
        for d in range(subset.shape[0]):
            missing = ~np.isfinite(subset[d, v])
            subset[d, v][missing] = fill[missing]
    X = (
        subset.transpose(2, 3, 0, 1)
        .reshape(subset.shape[2] * subset.shape[3], -1)
        .astype(np.float32)
    )
    return X, has_data


def classify_tile(tile, cache_paths, models, medians, checkpoint_indices):
    outputs = {}
    missing = []
    for checkpoint in CHECKPOINTS:
        folder = os.path.join(TILE_OUT, checkpoint)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{tile}_Our{checkpoint}2022_CornMask30m.tif")
        outputs[checkpoint] = path
        if checkpoint in models and not valid_raster(path, count=1, dtype="uint8"):
            missing.append(checkpoint)

    if not missing:
        print("    ✓ all trainable checkpoint tile masks already exist")
        return outputs

    sources = [rasterio.open(p) for p in cache_paths]
    template = sources[0]
    width, height = template.width, template.height
    profile = template.profile.copy()
    profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": OUTPUT_NODATA,
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
    )

    writers = {}
    temp_paths = {}
    try:
        for checkpoint in missing:
            tmp = os.path.join(TEMP_ROOT, f"{tile}_{checkpoint}.partial.tif")
            if os.path.exists(tmp):
                os.remove(tmp)
            temp_paths[checkpoint] = tmp
            writers[checkpoint] = rasterio.open(tmp, "w", **profile)

        total_blocks = int(np.ceil(height / BLOCK_SIZE)) * int(np.ceil(width / BLOCK_SIZE))
        counter = 0
        for row_off in range(0, height, BLOCK_SIZE):
            for col_off in range(0, width, BLOCK_SIZE):
                counter += 1
                h = min(BLOCK_SIZE, height - row_off)
                w = min(BLOCK_SIZE, width - col_off)
                win = Window(col_off, row_off, w, h)
                print(f"\r    {tile}: block {counter}/{total_blocks}", end="", flush=True)
                full_cube = read_cached_block(sources, win)
                spatial_valid = np.any(np.isfinite(full_cube), axis=(0, 1)).reshape(-1)

                for checkpoint in missing:
                    X, has_data = prepare_block_features(
                        full_cube,
                        checkpoint_indices[checkpoint],
                        medians[checkpoint],
                    )
                    valid = spatial_valid & has_data.reshape(-1)
                    out = np.full(h * w, OUTPUT_NODATA, dtype=np.uint8)
                    if valid.any():
                        predicted_classes = models[checkpoint].predict(X[valid])
                        out[valid] = (predicted_classes == 1).astype(np.uint8)
                    writers[checkpoint].write(out.reshape(h, w), 1, window=win)
                    del X, has_data, out
                del full_cube
                gc.collect()
        print()
    finally:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        for src in sources:
            try:
                src.close()
            except Exception:
                pass

    for checkpoint in missing:
        os.replace(temp_paths[checkpoint], outputs[checkpoint])
        print(f"    ✓ {checkpoint} tile mask saved")
    return outputs

# ============================================================
# 10. TILE PRODUCTION — ON DEMAND
# ============================================================

header("2022 LIGHTWEIGHT ON-DEMAND TILE PRODUCTION")

# ------------------------------------------------------------------
# ROBUST RESUME LEDGER
# ------------------------------------------------------------------
# Preserve terminal statuses from earlier runs.  The previous version
# reset status_rows = [] on every launch, so tiles that were legitimately
# unbuildable (for example, insufficient trusted corn labels) were
# forgotten and retried on every restart.
if os.path.exists(STATUS_CSV):
    try:
        status_rows = pd.read_csv(STATUS_CSV).to_dict("records")
        print(f"✓ Loaded existing resume ledger: {len(status_rows):,} rows")
    except Exception as exc:
        print("WARNING: could not read existing resume ledger:", exc)
        status_rows = []
else:
    status_rows = []


def latest_checkpoint_status(tile, checkpoint):
    """Return the most recent persisted status for one tile/checkpoint."""
    for row in reversed(status_rows):
        if (
            str(row.get("Tile", "")) == str(tile)
            and str(row.get("Checkpoint", "")) == str(checkpoint)
        ):
            value = row.get("Status", "")
            if pd.isna(value):
                return ""
            return str(value)
    return ""


for tile_number, tile in enumerate(all_tiles, start=1):
    tile_start = time.perf_counter()
    header(f"TILE {tile_number}/{len(all_tiles)} — {tile}")

    expected = {
        checkpoint: os.path.join(TILE_OUT, checkpoint, f"{tile}_Our{checkpoint}2022_CornMask30m.tif")
        for checkpoint in CHECKPOINTS
    }

    # Read the frozen manifest locally BEFORE downloading anything.
    # This lets resume logic recognize checkpoints that can never have
    # a mask because there were zero eligible scenes by that date.
    group = (
        scene_manifest[
            scene_manifest["tile"].astype(str) == str(tile)
        ]
        .copy()
        .sort_values("date_obj")
    )

    manifest_dates = [
        pd.Timestamp(d).date()
        for d in group["date_obj"].tolist()
    ]

    resolved = {}
    resolved_reason = {}

    # If this tile was already proven to have insufficient trusted labels,
    # that is a terminal methodological outcome, not work that should be
    # repeated after every restart.
    prior_label_skip = any(
        latest_checkpoint_status(tile, checkpoint)
        == "SKIPPED_INSUFFICIENT_LABELS"
        for checkpoint in CHECKPOINTS
    )

    for checkpoint, cutoff_text in CHECKPOINTS.items():
        output_ok = valid_raster(
            expected[checkpoint],
            count=1,
            dtype="uint8",
        )

        cutoff = pd.Timestamp(cutoff_text).date()
        has_dates = any(d <= cutoff for d in manifest_dates)

        if output_ok:
            resolved[checkpoint] = True
            resolved_reason[checkpoint] = "existing output"

        elif prior_label_skip:
            resolved[checkpoint] = True
            resolved_reason[checkpoint] = "previously skipped: insufficient trusted labels"

        elif not has_dates:
            resolved[checkpoint] = True
            resolved_reason[checkpoint] = "no eligible scenes by checkpoint"

            # Persist this terminal checkpoint state if it is not already
            # in the ledger.
            if latest_checkpoint_status(tile, checkpoint) != "INSUFFICIENT_DATES":
                status_rows.append(
                    {
                        "Tile": tile,
                        "Checkpoint": checkpoint,
                        "Status": "INSUFFICIENT_DATES",
                        "Dates": 0,
                        "Output": expected[checkpoint],
                    }
                )

        else:
            resolved[checkpoint] = False
            resolved_reason[checkpoint] = "needs processing"

    if all(resolved.values()):
        print("✓ Tile already resolved — no Sentinel download needed.")
        for checkpoint in CHECKPOINTS:
            print(
                f"    {checkpoint}: {resolved_reason[checkpoint]}"
            )
        save_status(status_rows)
        continue

    try:
        # group was already read above from the frozen manifest.
        if group.empty:
            raise RuntimeError("Frozen manifest contains no scenes for tile.")

        dates = group["date_obj"].tolist()
        print("States:", ", ".join(tile_states.get(tile, [])))
        print("Selected May-Aug dates:", len(group))

        print("\n    Downloading/reusing this tile's TEMPORARY Sentinel scenes...")
        cache_paths = ensure_tile_temp_scenes(tile, group)

        template_path = cache_paths[0]
        trusted_path = os.path.join(TRUSTED_CACHE, f"{tile}_TrustedLabels_2022.tif")
        print("\nTrusted-label cache:", trusted_path)
        download_ee_exact(
            TRUSTED_EE,
            template_path,
            trusted_path,
            cornbelt_union_target,
            "int16",
            0,
            "trusted",
            f"{tile} trusted labels",
        )

        rows, cols, labels, class_counts = get_balanced_training_pixels(trusted_path)
        n_points = len(labels)
        n_classes = len(np.unique(labels))
        n_corn = int(np.count_nonzero(labels == 1))

        print("Selected training points:", f"{n_points:,}")
        print("Selected classes:", n_classes)
        print("Selected corn points:", n_corn)
        print("Raw trusted class counts:", class_counts)

        if n_points < MIN_TRAIN_POINTS or n_classes < MIN_TRAIN_CLASSES or n_corn < MIN_CORN_POINTS:
            print("⚠ SKIPPED — insufficient trusted labels.")
            for checkpoint in CHECKPOINTS:
                status_rows.append(
                    {
                        "Tile": tile,
                        "Checkpoint": checkpoint,
                        "Status": "SKIPPED_INSUFFICIENT_LABELS",
                        "Training_points": n_points,
                        "Classes": n_classes,
                        "Corn_points": n_corn,
                    }
                )
            save_status(status_rows)
            if CLEAN_TEMP_AFTER_SUCCESS:
                shutil.rmtree(os.path.join(TEMP_ROOT, tile), ignore_errors=True)
            continue

        print("\n    Building training feature cube from TEMPORARY local scenes...")
        training_cube = build_training_cube(cache_paths, rows, cols)

        models, medians, checkpoint_indices, training_stats = train_models(
            training_cube, labels, dates, tile
        )

        if KEEP_MODELS:
            model_tile_dir = os.path.join(MODEL_DIR, tile)
            os.makedirs(model_tile_dir, exist_ok=True)
            for checkpoint, model in models.items():
                joblib.dump(
                    {
                        "model": model,
                        "global_variable_median": medians[checkpoint],
                        "variables": VARIABLE_ORDER,
                        "dates": [str(dates[i]) for i in checkpoint_indices[checkpoint]],
                        "target_year": YEAR,
                        "historical_cdl_years": HISTORICAL_YEARS,
                        "scene_manifest": SCENE_MANIFEST,
                    },
                    os.path.join(model_tile_dir, f"{tile}_{checkpoint}_RF_2022.joblib"),
                )

        print("\n    Classifying from TEMPORARY local scenes...")
        outputs = classify_tile(tile, cache_paths, models, medians, checkpoint_indices)

        stats_lookup = {row["Checkpoint"]: row for row in training_stats}
        for checkpoint in CHECKPOINTS:
            info = stats_lookup.get(checkpoint, {})
            output = outputs.get(checkpoint)
            if output and valid_raster(output, count=1, dtype="uint8"):
                status = info.get("Status", "COMPLETE")
                if status == "TRAINED":
                    status = "COMPLETE"
                elif status == "LOW_DATE_COUNT":
                    status = "COMPLETE_LOW_DATE_COUNT"
            else:
                status = info.get("Status", "FAILED")

            status_rows.append(
                {
                    "Tile": tile,
                    "Checkpoint": checkpoint,
                    "Status": status,
                    "Dates": info.get("Dates", np.nan),
                    "Features": info.get("Features", np.nan),
                    "Training_points": n_points,
                    "Classes": n_classes,
                    "Corn_points": n_corn,
                    "Output": output,
                }
            )
        save_status(status_rows)

        del training_cube, models, medians
        gc.collect()

        if CLEAN_TEMP_AFTER_SUCCESS:
            tile_temp = os.path.join(TEMP_ROOT, tile)
            shutil.rmtree(tile_temp, ignore_errors=True)
            print("    ✓ temporary Sentinel scenes deleted for", tile)

        print(f"\n✓ {tile} finished in {(time.perf_counter() - tile_start) / 60:.1f} min")

    except KeyboardInterrupt:
        print("\nStopped by user. Temporary scenes for this tile are retained for resume.")
        raise
    except Exception as exc:
        print("\n!!! TILE FAILED — continuing !!!")
        print(exc)
        traceback.print_exc(limit=2)
        for checkpoint in CHECKPOINTS:
            status_rows.append(
                {
                    "Tile": tile,
                    "Checkpoint": checkpoint,
                    "Status": "FAILED_TILE_EXCEPTION",
                    "Error": str(exc)[:1000],
                }
            )
        save_status(status_rows)
        print("Temporary scenes retained for retry:", os.path.join(TEMP_ROOT, tile))
        gc.collect()

# ============================================================
# 11. CLEAN STATE MOSAICS — ALWAYS REBUILD FROM VALID TILES
# ============================================================


def state_tile_paths(state_abbr, checkpoint):
    paths = []
    for tile in all_tiles:
        if state_abbr not in tile_states.get(tile, []):
            continue
        path = os.path.join(TILE_OUT, checkpoint, f"{tile}_Our{checkpoint}2022_CornMask30m.tif")
        if valid_raster(path, count=1, dtype="uint8"):
            paths.append(path)
    return paths


def snapped_bounds_for_geometry(geometry, anchor_transform):
    xmin, ymin, xmax, ymax = geometry.bounds
    inv = ~anchor_transform
    c1, r1 = inv * (xmin, ymin)
    c2, r2 = inv * (xmax, ymax)
    col_min = int(np.floor(min(c1, c2)))
    col_max = int(np.ceil(max(c1, c2)))
    row_min = int(np.floor(min(r1, r2)))
    row_max = int(np.ceil(max(r1, r2)))
    win = Window(col_min, row_min, col_max - col_min, row_max - row_min)
    snapped_transform = window_transform(win, anchor_transform)
    left = snapped_transform.c
    top = snapped_transform.f
    right = left + win.width * snapped_transform.a
    bottom = top + win.height * snapped_transform.e
    return (left, bottom, right, top), snapped_transform


def build_clean_state_mosaic(state_abbr, checkpoint):
    out_dir = os.path.join(STATE_CLEAN_ROOT, checkpoint, state_abbr)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"Our{checkpoint}2022_{state_abbr}_CornMask30m_CLEAN_REBUILT.tif",
    )

    paths = state_tile_paths(state_abbr, checkpoint)
    if not paths:
        return None

    # Critical difference from the stale 2023 bug: ALWAYS rebuild from tile files.
    if os.path.exists(out_path):
        os.remove(out_path)

    sources = [rasterio.open(p) for p in paths]
    try:
        geom = state_geom[state_abbr]
        snapped_bounds, expected_transform = snapped_bounds_for_geometry(geom, sources[0].transform)
        mosaic_arr, mosaic_transform = merge(
            sources,
            bounds=snapped_bounds,
            nodata=OUTPUT_NODATA,
            method="first",
        )
        if not np.allclose(tuple(mosaic_transform)[:6], tuple(expected_transform)[:6], atol=1e-9):
            raise RuntimeError(
                f"State mosaic grid drift for {state_abbr} {checkpoint}: "
                f"{mosaic_transform} != {expected_transform}"
            )

        arr = mosaic_arr[0].astype(np.uint8, copy=False)
        inside = geometry_mask(
            [mapping(geom)],
            out_shape=arr.shape,
            transform=mosaic_transform,
            invert=True,
        )
        arr[~inside] = OUTPUT_NODATA

        profile = sources[0].profile.copy()
        profile.update(
            {
                "height": arr.shape[0],
                "width": arr.shape[1],
                "transform": mosaic_transform,
                "count": 1,
                "dtype": "uint8",
                "nodata": OUTPUT_NODATA,
                "compress": "DEFLATE",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "IF_SAFER",
            }
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
            try:
                dst.build_overviews([2, 4, 8, 16], Resampling.mode)
                dst.update_tags(ns="rio_overview", resampling="mode")
            except Exception:
                pass
        return out_path
    finally:
        for src in sources:
            src.close()


header("CLEAN REBUILD — 2022 STATE MOSAICS")
state_summary_rows = []
state_mask_paths = defaultdict(dict)

for state_abbr in STATE_FIPS:
    print("\n" + "-" * 100)
    print(state_abbr)
    for checkpoint in CHECKPOINTS:
        path = build_clean_state_mosaic(state_abbr, checkpoint)
        state_mask_paths[state_abbr][checkpoint] = path
        if path is None:
            print(f"  {checkpoint}: NO OUTPUT")
            continue

        with rasterio.open(path) as src:
            arr = src.read(1)
            pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6
        valid = np.isin(arr, [0, 1])
        corn = arr == 1
        valid_area = valid.sum() * pixel_area_km2
        corn_area = corn.sum() * pixel_area_km2
        coverage = valid_area / state_area_km2[state_abbr] * 100

        state_summary_rows.append(
            {
                "State": state_abbr,
                "Checkpoint": checkpoint,
                "Valid_area_km2": valid_area,
                "State_coverage_pct": coverage,
                "Predicted_corn_km2": corn_area,
                "Path": path,
            }
        )
        print(f"  {checkpoint}: coverage={coverage:.3f}% | corn={corn_area:,.1f} km²")

pd.DataFrame(state_summary_rows).to_csv(STATE_SUMMARY_CSV, index=False)

# ============================================================
# 12. 2022 BENCHMARK: PREVIOUS 2021 vs OUR 2022 vs FINAL 2022
# ============================================================


def prepare_state_reference(state_abbr, reference_name, ee_image, template_path):
    out_dir = os.path.join(REFERENCE_CACHE, state_abbr)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{reference_name}_Corn30m.tif")
    return download_ee_exact(
        ee_image,
        template_path,
        out_path,
        state_geom[state_abbr],
        "uint8",
        REF_NODATA,
        "corn",
        f"{state_abbr} {reference_name}",
    )


def benchmark_state_checkpoint(state_abbr, checkpoint, refs):
    our_path = state_mask_paths[state_abbr][checkpoint]
    if not our_path or not valid_raster(our_path, count=1, dtype="uint8"):
        raise RuntimeError("Our state mask is missing.")

    truth_path = refs["CDL_2022"]
    previous_path = refs["CDL_2021"]

    counts = {
        "Previous-year CDL": empty_counts(),
        "Our ICDL": empty_counts(),
    }
    common_n = 0

    with rasterio.open(our_path) as ours, rasterio.open(truth_path) as truth, rasterio.open(previous_path) as previous:
        if not (
            ours.width == truth.width == previous.width
            and ours.height == truth.height == previous.height
            and ours.transform.almost_equals(truth.transform)
            and ours.transform.almost_equals(previous.transform)
            and ours.crs == truth.crs == previous.crs
        ):
            raise RuntimeError("Reference alignment mismatch.")

        pixel_area_km2 = abs(ours.transform.a * ours.transform.e) / 1e6
        total_blocks = int(np.ceil(ours.height / METRIC_BLOCK_SIZE)) * int(
            np.ceil(ours.width / METRIC_BLOCK_SIZE)
        )
        counter = 0
        for row_off in range(0, ours.height, METRIC_BLOCK_SIZE):
            for col_off in range(0, ours.width, METRIC_BLOCK_SIZE):
                counter += 1
                h = min(METRIC_BLOCK_SIZE, ours.height - row_off)
                w = min(METRIC_BLOCK_SIZE, ours.width - col_off)
                win = Window(col_off, row_off, w, h)
                print(
                    f"\r      {state_abbr} {checkpoint}: metric block {counter}/{total_blocks}",
                    end="",
                    flush=True,
                )
                a = ours.read(1, window=win)
                t = truth.read(1, window=win)
                p = previous.read(1, window=win)
                common = np.isin(a, [0, 1]) & np.isin(t, [0, 1]) & np.isin(p, [0, 1])
                if not common.any():
                    continue
                truth_bool = t[common] == 1
                prev_bool = p[common] == 1
                our_bool = a[common] == 1
                update_counts(counts["Previous-year CDL"], prev_bool, truth_bool)
                update_counts(counts["Our ICDL"], our_bool, truth_bool)
                common_n += int(common.sum())
        print()

    rows = []
    for mask_name, c in counts.items():
        metrics = counts_to_metrics(c, pixel_area_km2)
        metrics.update(
            {
                "State": state_abbr,
                "Checkpoint": checkpoint,
                "Mask": mask_name,
                "Common_coverage_pct": common_n * pixel_area_km2 / state_area_km2[state_abbr] * 100,
            }
        )
        rows.append(metrics)
    return rows, counts


if RUN_STATE_BENCHMARK:
    header("2022 STATE + CORN BELT BENCHMARK")
    benchmark_rows = []
    failures = []
    pooled_counts = {
        checkpoint: {
            "Previous-year CDL": empty_counts(),
            "Our ICDL": empty_counts(),
        }
        for checkpoint in CHECKPOINTS
    }

    for state_abbr in STATE_FIPS:
        print("\n" + "=" * 100)
        print("BENCHMARK STATE:", state_abbr)
        print("=" * 100)

        template_path = state_mask_paths[state_abbr].get("August")
        if not template_path or not valid_raster(template_path, count=1, dtype="uint8"):
            print("No August state mask — skipping benchmark.")
            continue

        try:
            refs = {
                "CDL_2021": prepare_state_reference(state_abbr, "CDL_2021", CDL_PREVIOUS_EE, template_path),
                "CDL_2022": prepare_state_reference(state_abbr, "CDL_2022", CDL_FINAL_EE, template_path),
            }
            for checkpoint in CHECKPOINTS:
                rows, counts = benchmark_state_checkpoint(state_abbr, checkpoint, refs)
                benchmark_rows.extend(rows)
                for mask_name in pooled_counts[checkpoint]:
                    add_counts(pooled_counts[checkpoint][mask_name], counts[mask_name])
            pd.DataFrame(benchmark_rows).to_csv(STATE_BENCHMARK_CSV, index=False)

        except Exception as exc:
            print("BENCHMARK FAILURE:", exc)
            traceback.print_exc(limit=2)
            failures.append({"State": state_abbr, "Error": str(exc)[:1000]})
            pd.DataFrame(failures).to_csv(BENCHMARK_FAILURES_CSV, index=False)

    state_benchmark_df = pd.DataFrame(benchmark_rows)
    state_benchmark_df.to_csv(STATE_BENCHMARK_CSV, index=False)

    pooled_rows = []
    pooled_pixel_area_km2 = 0.0009
    for checkpoint in CHECKPOINTS:
        for mask_name, counts in pooled_counts[checkpoint].items():
            metrics = counts_to_metrics(counts, pooled_pixel_area_km2)
            metrics.update({"Checkpoint": checkpoint, "Mask": mask_name})
            pooled_rows.append(metrics)

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(BELT_BENCHMARK_CSV, index=False)

    header("POOLED 12-STATE 2022 BENCHMARK")
    if not pooled_df.empty:
        print(
            pooled_df[
                [
                    "Checkpoint",
                    "Mask",
                    "Precision",
                    "Recall",
                    "F1",
                    "IoU",
                    "Accuracy",
                    "Predicted_corn_km2",
                    "Actual_corn_km2",
                    "Corn_area_bias_pct",
                    "N",
                ]
            ].to_string(
                index=False,
                formatters={
                    "Precision": lambda x: f"{x:.4f}",
                    "Recall": lambda x: f"{x:.4f}",
                    "F1": lambda x: f"{x:.4f}",
                    "IoU": lambda x: f"{x:.4f}",
                    "Accuracy": lambda x: f"{x:.4f}",
                    "Predicted_corn_km2": lambda x: f"{x:,.1f}",
                    "Actual_corn_km2": lambda x: f"{x:,.1f}",
                    "Corn_area_bias_pct": lambda x: f"{x:+.2f}%",
                },
            )
        )

# ============================================================
# 13. DONE
# ============================================================

header("2022 LIGHTWEIGHT ICDL RUN COMPLETE")
print("Frozen scene manifest:")
print(SCENE_MANIFEST)
print("\nTile inventory:")
print(TILE_INVENTORY)
print("\nProduction root:")
print(OUT_ROOT)
print("\nClean state mosaics:")
print(STATE_CLEAN_ROOT)
print("\nState summary:")
print(STATE_SUMMARY_CSV)
print("\nGeneration status:")
print(STATUS_CSV)
if RUN_STATE_BENCHMARK:
    print("\nState benchmark:")
    print(STATE_BENCHMARK_CSV)
    print("\nPooled benchmark:")
    print(BELT_BENCHMARK_CSV)
print("\nTemporary Sentinel scenes are deleted tile-by-tile after success.")
print("2022 CORN BELT LIGHTWEIGHT WORKFLOW COMPLETE ✓")
