import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import os
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

SHAPEFILE_PATH = "../district_boundary_data/survey_of_india/DISTRICT_BOUNDARY.shp" 
STATE_COLUMN = "STATE_UT"
DISTRICT_COLUMN = "DISTRICT"

TARGET_STATES = [
    # "PUNJAB",
    # "HARYANA",
    # "UTTAR PRADESH",
    # "MADHYA PRADESH",
    # "RAJASTHAN",
    'GUJARAT',
    'BIHAR',
]

OUTPUT_DIR = "downloaded_data_wcs"
OUTPUT_FILENAME = "district_soil_data_guj_bih.csv"

SOIL_FEATURES = {
    "Soil Organic Carbon": "soc",
    "Soil pH": "phh2o",
    "Bulk Density": "bdod",
    "Sand Content": "sand",
    "Silt Content": "silt",
    "Clay Content": "clay",
    "Cation Exchange Capacity": "cec",
    "Total Nitrogen": "nitrogen"
}

DEPTH_INTERVALS = [
    "0-5cm",
    "5-15cm",
    "15-30cm",
    "30-60cm",
    "60-100cm",
    "100-200cm"
]

MAX_WORKERS_FEATURES = 2  # Number of parallel downloads for soil features per district
MAX_WORKERS_DISTRICTS = 1  # Number of parallel districts to process
REQUEST_DELAY = 0.5  # Delay between requests in seconds to be respectful to the server
MAX_RETRIES = 3  # Number of retries for failed requests
CONNECTION_TIMEOUT = 60  # Connection timeout in seconds


def create_robust_session():
    """
    Create a requests session with retry logic and proper timeouts.
    """
    session = requests.Session()
    
    # Define retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set timeouts
    session.timeout = CONNECTION_TIMEOUT
    
    return session


def download_soil_feature(args):
    """
    Download a single soil feature for a district using the direct WCS API.
    Returns a dictionary with the result or error information.
    """
    (district_name, state_name, district_geom, feature_name, 
     service_id, depth, minx, miny, maxx, maxy, 
     output_dir) = args
    
    # Add a small delay to be respectful to the server
    time.sleep(REQUEST_DELAY)
    
    session = create_robust_session()
    
    # Define a temporary file path for the downloaded raster
    temp_raster_path = os.path.join(output_dir, f"{service_id}_{depth}_{district_name.replace(' ', '_')}.tif")
    
    # Construct the WCS API request URL
    # Resolution is set to ~250m in degrees
    base_url = "https://maps.isric.org/mapserv"
    coverage_id = f"{service_id}_{depth}_mean"
    params = {
        'map': f'/map/{service_id}.map',
        'service': 'WCS',
        'version': '1.0.0',
        'request': 'GetCoverage',
        'coverage': coverage_id,
        'crs': 'EPSG:4326',
        'bbox': f'{minx},{miny},{maxx},{maxy}',
        'width': max(int(round((maxx - minx) / 0.002083333333333)), 2),
        'height': max(int(round((maxy - miny) / 0.002083333333333)), 2),
        'format': 'GEOTIFF_INT16'
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            # Fetch the data using the session
            response = session.get(base_url, params=params, timeout=CONNECTION_TIMEOUT)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Save the raster content to a temporary file
            with open(temp_raster_path, 'wb') as f:
                f.write(response.content)

            # Perform zonal statistics to get the mean value within the polygon
            stats = zonal_stats(
                district_geom,
                temp_raster_path,
                stats="mean",
                geojson_out=False
            )
            
            mean_val = stats[0].get('mean') if stats and stats[0] else None
            mean_value = mean_val if mean_val is not None else 'NaN'

            # Clean up the temporary raster file
            if os.path.exists(temp_raster_path):
                os.remove(temp_raster_path)

            return {
                "state": state_name,
                "district": district_name,
                "feature_name": feature_name,
                "depth": depth,
                "value": mean_value,
                "success": True,
                "attempts": attempt + 1
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                # Wait before retrying (exponential backoff)
                wait_time = (2 ** attempt) * REQUEST_DELAY
                time.sleep(wait_time)
                continue
            else:
                # Final attempt failed
                return {
                    "state": state_name,
                    "district": district_name,
                    "feature_name": feature_name,
                    "depth": depth,
                    "value": 'NaN',
                    "success": False,
                    "error": str(e),
                    "attempts": attempt + 1
                }


def main():
    print("Starting soil data extraction process...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # --- Step 1: Load and Filter Shapefile ---
    try:
        print(f"Loading shapefile from: {SHAPEFILE_PATH}")
        india_gdf = gpd.read_file(SHAPEFILE_PATH)
    except Exception as e:
        print(f"Error: Could not read the shapefile. Please check the path: {SHAPEFILE_PATH}")
        print(f"Details: {e}")
        return

    # Ensure the shapefile is in WGS84 (EPSG:4326)
    if india_gdf.crs.to_epsg()!= 4326:
        print("Reprojecting shapefile to EPSG:4326 (WGS84)...")
        india_gdf = india_gdf.to_crs(epsg=4326)

    print(f"Filtering for districts in the target states: {', '.join(TARGET_STATES)}")
    roi_gdf = india_gdf[india_gdf[STATE_COLUMN].isin(TARGET_STATES)].copy()
    
    if roi_gdf.empty:
        print("Error: No districts found for the specified states. Please check the STATE_COLUMN name and TARGET_STATES list.")
        return
        
    print(f"Found {len(roi_gdf)} districts in the region of interest.")

    # --- Step 2: Prepare tasks for parallel processing ---
    print("Preparing download tasks...")
    all_tasks = []
    
    for _, district_row in roi_gdf.iterrows():
        district_name = district_row[DISTRICT_COLUMN]
        state_name = district_row[STATE_COLUMN]
        district_geom = district_row.geometry

        # Get bounding box for the WCS request
        minx, miny, maxx, maxy = district_geom.bounds    
        
        # Create tasks for all soil features and depths for this district
        for feature_name, service_id in SOIL_FEATURES.items():
            for depth in DEPTH_INTERVALS:
                task_args = (
                    district_name, state_name, district_geom, feature_name,
                    service_id, depth, minx, miny, maxx, maxy, 
                    OUTPUT_DIR
                )
                all_tasks.append(task_args)

    print(f"Created {len(all_tasks)} download tasks for {len(roi_gdf)} districts")
    print(f"Using {MAX_WORKERS_FEATURES} parallel workers for downloads")
    print(f"Retry settings: {MAX_RETRIES} retries, {CONNECTION_TIMEOUT}s timeout, {REQUEST_DELAY}s delay between requests")

    # --- Step 3: Execute downloads in parallel ---
    all_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FEATURES) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(download_soil_feature, task): task for task in all_tasks}
        
        # Process completed tasks with progress bar
        for future in tqdm(as_completed(future_to_task), total=len(all_tasks), desc="Downloading soil data"):
            result = future.result()
            all_results.append(result)
            
            # Print errors as they occur
            if not result.get('success', True):
                attempts = result.get('attempts', 1)
                print(f"ERROR: {result['feature_name']} at {result['depth']} for {result['district']} (after {attempts} attempts): {result.get('error', 'Unknown error')}")

    elapsed_time = time.time() - start_time
    print(f"\nDownload completed in {elapsed_time:.2f} seconds")
    print(f"Average time per task: {elapsed_time/len(all_tasks):.2f} seconds")

    # --- Step 4: Consolidate and Save Final Output ---
    print("\nConsolidating all results...")
    results_df = pd.DataFrame(all_results)

    # Create the final 'district_id' column
    results_df['district_id'] = results_df['state'] + "_" + results_df['district']
    
    # Select and reorder columns to match the desired output format
    final_df = results_df[['district_id', 'feature_name', 'depth', 'value']]

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    final_df.to_csv(output_path, index=False)

    # Calculate success statistics
    successful_downloads = len([r for r in all_results if r.get('success', True)])
    failed_downloads = len(all_results) - successful_downloads
    
    print(f"\nProcess complete. Data saved to: {output_path}")
    print(f"Total records created: {len(final_df)}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    if failed_downloads > 0:
        print(f"Success rate: {(successful_downloads/len(all_results)*100):.1f}%")


if __name__ == "__main__":
    main()