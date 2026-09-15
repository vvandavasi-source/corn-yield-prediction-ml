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
PROJECT_ID = 'ee-lreeder'
CLIENT_SECRET_FILE = 'client_secret.json'  # Place your downloaded OAuth client secret file here
TOKEN_FILE = 'token.json'
LOCAL_DOWNLOAD_DIR = './downloaded_csvs'

os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

# 1. OAuth Scopes for User Auth (Earth Engine + Google Drive)
SCOPES = [
    'https://www.googleapis.com/auth/earthengine',
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive'
]

creds = None
if os.path.exists(TOKEN_FILE):
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        auth_flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        creds = auth_flow.run_local_server(port=0)
    
    with open(TOKEN_FILE, 'w') as token:
        token.write(creds.to_json())

# 2. Initialize Earth Engine and Drive API with User Credentials
ee.Initialize(credentials=creds, project=PROJECT_ID)
drive_service = build('drive', 'v3', credentials=creds)

# 3. Settings
start_year = 2008
end_year = 2024
start_month = 3
end_month = 10

state_names = [
    "North Dakota", "South Dakota", "Nebraska", "Kansas",
    "Minnesota", "Iowa", "Missouri", "Wisconsin",
    "Illinois", "Michigan", "Indiana", "Ohio"
]
CORN_CLASS = 1
MASK_TYPE = "same_year"  # Options: "none", "same_year", "lagged", "five_year"
FIVE_YEAR_THRESHOLD = 0.6

# 4. Load Boundaries
counties = ee.FeatureCollection("TIGER/2018/Counties")
states = ee.FeatureCollection("TIGER/2018/States")

selected_states = states.filter(ee.Filter.inList('NAME', state_names))
state_fp_list = selected_states.aggregate_array('STATEFP')
selected_counties = counties.filter(ee.Filter.inList('STATEFP', state_fp_list))

# 5. Vegetation Index Function
def add_indices(image):
    img = ee.Image(image)
    red = img.select('sur_refl_b01').multiply(0.0001)
    nir = img.select('sur_refl_b02').multiply(0.0001)
    green = img.select('sur_refl_b04').multiply(0.0001)
    swir = img.select('sur_refl_b06').multiply(0.0001)

    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    evi = img.select('EVI').multiply(0.0001).rename('EVI_scaled')
    evi2 = nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(2.4)).add(1)).rename('EVI2')
    gci = nir.divide(green).subtract(1).rename('GCI')
    nirv = ndvi.multiply(nir).rename('NIRv')
    ndmi = nir.subtract(swir).divide(nir.add(swir)).rename('NDMI')
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')

    return img.addBands([ndvi, evi, evi2, nirv, gci, ndmi, ndwi]) \
              .select(['NDVI', 'EVI_scaled', 'EVI2', 'NIRv', 'GCI', 'NDMI', 'NDWI'])

# 6. Mask Function
def get_corn_mask(year):
    year = ee.Number(year)

    if MASK_TYPE == "none":
        return ee.Image.constant(1).rename('corn_mask')

    if MASK_TYPE == "same_year":
        same_year_cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(year, year, 'year')) \
            .first() \
            .select('cropland')
        return same_year_cdl.eq(CORN_CLASS).rename('corn_mask')

    if MASK_TYPE == "lagged":
        lag_year = year.subtract(1)
        lagged_cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(lag_year, lag_year, 'year')) \
            .first()
            
        return ee.Image(ee.Algorithms.If(
            lagged_cdl,
            ee.Image(lagged_cdl).select('cropland').eq(CORN_CLASS),
            ee.Image.constant(1)
        ))

    cdl_years = ee.List.sequence(year.subtract(5), year.subtract(1))

    def map_cdl(y):
        cdl = ee.ImageCollection("USDA/NASS/CDL") \
            .filter(ee.Filter.calendarRange(y, y, 'year')) \
            .first() \
            .select('cropland')
        return cdl.eq(CORN_CLASS)

    cdl_stack = ee.ImageCollection.fromImages(cdl_years.map(map_cdl))
    corn_freq_5yr = cdl_stack.mean()
    return corn_freq_5yr.gte(FIVE_YEAR_THRESHOLD)

# 7. Data Collection & Export Loop (Per State)
for state_name in state_names:
    print(f"\n--- Submitting task for: {state_name} ---")
    
    # Filter boundary ONLY for the current state to keep the GEE graph small
    state_feature = selected_states.filter(ee.Filter.eq('NAME', state_name)).first()
    state_fp = state_feature.get('STATEFP')
    state_counties = selected_counties.filter(ee.Filter.eq('STATEFP', state_fp))

    state_years_stats = ee.List([])

    for year in range(start_year, end_year + 1):
        start_date = ee.Date.fromYMD(year, start_month, 1)
        end_date = ee.Date.fromYMD(year, end_month, 31)

        corn_mask = ee.Image(get_corn_mask(year)) #change to icdl file for prediction year

        mod13 = ee.ImageCollection("MODIS/061/MOD13Q1") \
            .filterDate(start_date, end_date) \
            .map(lambda img: img.updateMask(img.select('SummaryQA').lte(1)))

        mod09 = ee.ImageCollection("MODIS/061/MOD09A1") \
            .filterDate(start_date, end_date)

        joined = ee.Join.inner().apply(
            primary=mod13,
            secondary=mod09,
            condition=ee.Filter.equals(
                leftField='system:time_start',
                rightField='system:time_start'
            )
        )

        processed_col = ee.ImageCollection(
            joined.map(lambda f: ee.Image(f.get('primary')).addBands(ee.Image(f.get('secondary'))).updateMask(corn_mask))
        ).map(add_indices) 

        def reduce_and_set(image):
            date_string = image.date().format('YYYY-MM-dd')
            return image.reduceRegions(
                collection=state_counties,  # Only reduces counties in this state
                reducer=ee.Reducer.mean(),
                scale=250,
                tileScale=8
            ).map(lambda f: f.set({
                'date': date_string,
                'year': year,
                'mask_type': MASK_TYPE
            }))

        year_stats = processed_col.map(reduce_and_set).flatten()
        state_years_stats = state_years_stats.add(year_stats)

    final_state_collection = ee.FeatureCollection(state_years_stats).flatten()

    safe_state_name = state_name.replace(" ", "_")
    task_desc = f'Corn_MODIS_{MASK_TYPE}_{safe_state_name}'

    task = ee.batch.Export.table.toDrive(
        collection=final_state_collection,
        description=task_desc,
        folder='EarthEngineExportsL',
        fileNamePrefix=task_desc,
        fileFormat='CSV',
        selectors=[
            'NAME', 'GEOID', 'STATEFP', 'date', 'year', 'mask_type',
            'NDVI', 'EVI_scaled', 'EVI2', 'NIRv', 'GCI', 'NDMI', 'NDWI'
        ]
    )
    
    task.start()
    print(f'Started Drive export task: {task_desc}')
    
    timer = 0
    while task.status()['state'] in ['READY', 'RUNNING']:
        print(f"Task state: {task_desc} - {task.status()['state']} elapsed time: {timer} minutes")
        timer += 0.5
        time.sleep(30)

    if task.status()['state'] != 'COMPLETED':
        print(f"Export failed for {state_name}: {task.status().get('error_message', 'Unknown error')}")
        continue

    # Download from Google Drive to local machine
    file_name = f"{task_desc}.csv"
    results = drive_service.files().list(
        q=f"name='{file_name}' and trashed=false",
        spaces='drive',
        fields="files(id, name)"
    ).execute()

    items = results.get('files', [])

    if not items:
        print(f"No files found matching: {file_name}. Please check your Google Drive for the exported file.")
    else:
        file_id = items[0]['id']
        local_filepath = os.path.join(LOCAL_DOWNLOAD_DIR, file_name)
        print(f"Downloading {file_name} to {local_filepath}")
        
        request = drive_service.files().get_media(fileId=file_id)
        with io.FileIO(local_filepath, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download progress: {int(status.progress() * 100)}%.")
                    
        print(f"Successfully downloaded {file_name}")

print("All GEE files processed and downloaded.")
