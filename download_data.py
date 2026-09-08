import os
import io
import time
import ee
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Configuration
project_id = 'bse509'
CLIENT_SECRET_FILE = "C:\\Users\\logan\\Downloads\\client_secret_422768806391-nckbupemlp8brug50t61g6i1vko9emdf.apps.googleusercontent.com.json"
TOKEN_FILE = 'token.json'
LOCAL_DOWNLOAD_DIR = './downloaded_csvs'

os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# 1. Set OAuth Scopes (including Google Drive)
SCOPES = [
    'https://www.googleapis.com/auth/earthengine',
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive'  # Grants access to your Drive
]

creds = None
if os.path.exists(TOKEN_FILE):
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = flow.run_local_server(port=0)
    
    with open(TOKEN_FILE, 'w') as token:
        token.write(creds.to_json())

# 2. Initialize Earth Engine and Drive API with User Credentials
ee.Initialize(credentials=creds, project=project_id)
drive_service = build('drive', 'v3', credentials=creds)

#def auth_and_init():
"""Authenticate and initialize the Earth Engine API."""
try:
    ee.Initialize(credentials=gee_credentials)
    print("Earth Engine initialized successfully.")
except ee.EEException as e:
    print("Authentication required. Please authenticate with Earth Engine.")
    ee.Authenticate()
    ee.Initialize(credentials=gee_credentials)
    print("Earth Engine initialized successfully after authentication.")
# Initialize Earth Engine (Uncomment below if you haven't initialized yet)
# auth_and_init()

# 1. SETTINGS

global start_year, end_year, start_month, end_month
start_year = 2008
end_year = 2024
start_month = 3
end_month = 10

# 2008 has no lagged CDL (no 2007 CDL exists)

state_names = [
    "North Dakota", "South Dakota", "Nebraska", "Kansas",
    "Minnesota", "Iowa", "Missouri", "Wisconsin",
    "Illinois", "Michigan", "Indiana", "Ohio"
]
CORN_CLASS = 1


# MASK SETTINGS 

MASK_TYPE = "lagged"
FIVE_YEAR_THRESHOLD = 0.6


# 2. LOAD COUNTIES & STATES [cite: 4, 5]

counties = ee.FeatureCollection("TIGER/2018/Counties")
states = ee.FeatureCollection("TIGER/2018/States")

selected_states = states.filter(ee.Filter.inList('NAME', state_names))
state_fp_list = selected_states.aggregate_array('STATEFP')
selected_counties = counties.filter(ee.Filter.inList('STATEFP', state_fp_list))

# ======================================================
# 3. VEGETATION INDEX FUNCTION (NDVI + NDMI + NDWI) [cite: 6, 15]
# ======================================================
def add_indices(image):
    img = ee.Image(image)

    red = img.select('sur_refl_b01').multiply(0.0001)  # [cite: 7]
    nir = img.select('sur_refl_b02').multiply(0.0001)  # [cite: 7]
    green = img.select('sur_refl_b04').multiply(0.0001)  # [cite: 7]
    swir = img.select('sur_refl_b06').multiply(0.0001)  # [cite: 8]

    # NDVI (NEW — correct) [cite: 8]
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    
    # EVI (from MODIS) [cite: 9]
    evi = img.select('EVI').multiply(0.0001).rename('EVI_scaled')
    
    # EVI2 [cite: 10]
    evi2 = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(2.4)).add(1)).rename('EVI2')
    
    # GCI [cite: 11]
    gci = nir.divide(green).subtract(1).rename('GCI')
    
    # NIRv [cite: 12]
    nirv = ndvi.multiply(nir).rename('NIRv')
    
    # NDMI (moisture — your original NDWI) [cite: 13]
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI')
    
    # TRUE NDWI (water index) [cite: 14]
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    return img.addBands([ndvi, evi, evi2, nirv, gci, ndmi, ndwi]) \
              .select(['NDVI', 'EVI_scaled', 'EVI2', 'NIRv', 'GCI', 'NDMI', 'NDWI'])  # [cite: 15]

# ======================================================
# 4. MASK FUNCTION [cite: 16, 24]
# ======================================================
def get_corn_mask(year):
    year = ee.Number(year)  # [cite: 16]

    if MASK_TYPE == "none":
        return ee.Image.constant(1).rename('corn_mask')  # [cite: 17]

    if MASK_TYPE == "same_year":
        same_year_cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first() \
            .select('cropland')  # [cite: 18]
        return same_year_cdl.eq(CORN_CLASS).rename('corn_mask')  # [cite: 19]

    if MASK_TYPE == "lagged":
        lag_year = year.subtract(1)  # [cite: 19]
        lagged_cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(lag_year, lag_year, 'year')) \
            .first()  # [cite: 20]
            
        return ee.Image(ee.Algorithms.If(
            lagged_cdl,
            ee.Image(lagged_cdl).select('cropland').eq(CORN_CLASS),
            ee.Image.constant(1)
        ))  # [cite: 21]

    # five-year mask [cite: 22]
    cdl_years = ee.List.sequence(year.subtract(5), year.subtract(1))  # [cite: 22]

    def map_cdl(y):
        cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(y, y, 'year')) \
            .first() \
            .select('cropland')
        return cdl.eq(CORN_CLASS)

    cdl_stack = ee.ImageCollection.fromImages(cdl_years.map(map_cdl))  # [cite: 23]
    corn_freq_5yr = cdl_stack.mean()  # [cite: 24]
    return corn_freq_5yr.gte(FIVE_YEAR_THRESHOLD)  # [cite: 24]

# ======================================================
# 5. COLLECT DATA [cite: 24, 31]
# ======================================================
all_years_stats = ee.List([])  # [cite: 24]

for year in range(start_year, end_year + 1):  # [cite: 25]
    print(f'Processing year: {year}')

    start_date = ee.Date.fromYMD(year, start_month, 1)  # [cite: 25]
    end_date = ee.Date.fromYMD(year, end_month, 31)  # [cite: 26]

    corn_mask = ee.Image(get_corn_mask(year))  # [cite: 26]

    mod13 = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start_date, end_date) \
        .map(lambda img: img.updateMask(img.select('SummaryQA').lte(1)))  # [cite: 27]

    mod09 = ee.ImageCollection("MODIS/061/MOD09A1") \
        .filterDate(start_date, end_date)  # [cite: 28]

    joined = ee.Join.inner().apply(
        primary=mod13,
        secondary=mod09,
        condition=ee.Filter.equals(
            leftField='system:time_start',
            rightField='system:time_start'
        )
    )  # [cite: 28]

    processed_col = ee.ImageCollection(
        joined.map(lambda f: ee.Image(f.get('primary')).addBands(ee.Image(f.get('secondary'))).updateMask(corn_mask))).map(add_indices) 

    def reduce_and_set(image):
        date_string = image.date().format('YYYY-MM-dd')
        return image.reduceRegions(
            collection=selected_counties,
            reducer=ee.Reducer.mean(),
            scale=250,
            tileScale=8
        ).map(lambda f: f.set({
            'date': date_string,
            'year': year,
            'mask_type': MASK_TYPE
        }))

    year_stats = processed_col.map(reduce_and_set).flatten()  # [cite: 30]
    all_years_stats = all_years_stats.add(year_stats)  # [cite: 31]
    
print(f'flattening all years stats, total years processed: {end_year - start_year + 1}')
final_collection = ee.FeatureCollection(all_years_stats).flatten()  # [cite: 31]

# In local Python, limit(10).getInfo() will actually evaluate and print the data preview.
print('Preview:', final_collection.limit(10).getInfo())

# ======================================================
# 6. EXPORT [cite: 32, 33]
# ======================================================
for state_name in state_names:  # [cite: 32]
    
    state_fp = selected_states \
        .filter(ee.Filter.eq('NAME', state_name)) \
        .first() \
        .get('STATEFP')  # [cite: 32]

    state_data = final_collection.filter(ee.Filter.eq('STATEFP', state_fp))  # [cite: 32]
    
    # Format state string to avoid issues in Drive filenames
    safe_state_name = state_name.replace(" ", "_")
    task_desc = f'Corn_MODIS_{MASK_TYPE}_{safe_state_name}'

    task = ee.batch.Export.table.toDrive(
        collection=state_data,
        description=task_desc,
        folder='EarthEngineExportsL',
        fileNamePrefix=task_desc,
        fileFormat='CSV',
        selectors=[
            'NAME', 'GEOID', 'STATEFP', 'date', 'year', 'mask_type',  # [cite: 32, 33]
            'NDVI', 'EVI_scaled', 'EVI2', 'NIRv', 'GCI', 'NDMI', 'NDWI'  # [cite: 33]
        ]
    )
    
    # Start the task queue in Earth Engine
    task.start()
    print(f'Started Drive export task: {task_desc}')

    print("Waiting for Earth Engine export to finish...")
    while task.status()['state'] in ['READY', 'RUNNING']:
        print(f"Task state: {task.status()['state']}...")
        time.sleep(30)

    if task.status()['state'] != 'COMPLETED':
        raise Exception(f"Export failed: {task.status().get('error_message', 'Unknown error')}")
"""
    print("Export completed! Connecting to Google Drive...")
    # 3. Search for the file in Drive
    file_name = f"{task_desc}.csv"
    results = drive_service.files().list(
        q=f"name='{file_name}' and trashed=false",
        spaces='drive',
        fields="files(id, name)"
    ).execute()

    items = results.get('files', [])

    if not items:
        print(f"No files found matching: {file_name}")
    else:
        # 4. Download the file
        file_id = items[0]['id']
        print(f"Downloading {file_name} (ID: {file_id})")
        
        request = drive_service.files().get_media(fileId=file_id)
        
        with io.FileIO(file_name, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%.")
                
        print(f"Successfully downloaded {file_name} to your local machine.")
        """
        
print("All GEE files have been exported and sent to drive.")