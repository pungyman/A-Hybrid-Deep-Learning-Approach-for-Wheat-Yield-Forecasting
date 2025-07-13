import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import Point
from io import StringIO
import time
import os
from datetime import datetime

def read_shapefile(filepath):
    """Read shapefile and return GeoDataFrame"""
    try:
        gdf = gpd.read_file(filepath)
        print(f"Successfully loaded shapefile with {len(gdf)} records")
        print(f"Columns: {list(gdf.columns)}")
        return gdf
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        return None

def filter_districts_by_states(gdf, target_states):
    """Filter GeoDataFrame for specific states"""
    try:
        # Filter for the target states
        filtered_gdf = gdf[gdf['STATE_UT'].isin(target_states)]
        
        if len(filtered_gdf) == 0:
            print(f"No records found for target states: {target_states}")
            print(f"Available states: {sorted(gdf['STATE_UT'].unique())}")
            return None
        
        print(f"Found {len(filtered_gdf)} districts in target states")
        for state in target_states:
            state_districts = filtered_gdf[filtered_gdf['STATE_UT'] == state]
            print(f"  {state}: {len(state_districts)} districts")
        
        return filtered_gdf
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None

def get_centroid_coordinates(gdf):
    """Get centroid coordinates from geometry and transform to WGS84"""
    try:
        # Get the geometry and calculate centroid
        geometry = gdf.geometry.iloc[0]
        centroid = geometry.centroid
        
        # Create a GeoDataFrame with the centroid point
        centroid_point = Point(centroid.x, centroid.y)
        centroid_gdf = gpd.GeoDataFrame([1], geometry=[centroid_point], crs=gdf.crs)
        
        # Transform to WGS84 (EPSG:4326) - geographic coordinates
        centroid_wgs84 = centroid_gdf.to_crs(epsg=4326)
        
        # Extract longitude and latitude
        longitude = centroid_wgs84.geometry.iloc[0].x
        latitude = centroid_wgs84.geometry.iloc[0].y
        
        return longitude, latitude
    except Exception as e:
        print(f"Error calculating centroid: {e}")
        return None, None

def fetch_nasa_power_data(longitude, latitude, start_date, end_date, parameters):
    """Fetch weather data from NASA POWER API in CSV format"""
    try:
        # NASA POWER API endpoint
        base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Parameters for the API request
        params = {
            'parameters': ','.join(parameters),
            'community': 'AG',  # Agricultural community
            'longitude': longitude,
            'latitude': latitude,
            'start': start_date,
            'end': end_date,
            'format': 'CSV'
        }
        
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Return CSV text directly
        csv_data = response.text
        return csv_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NASA POWER data: {e}")
        return None

def process_nasa_csv_data(csv_data):
    """Process NASA POWER CSV data into pandas DataFrame"""
    try:
        # Split the response into lines
        lines = csv_data.strip().split('\n')
        
        # Find the start of actual CSV data (after -END HEADER-)
        csv_start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == '-END HEADER-':
                csv_start_idx = i + 1
                break
        
        # Extract only the CSV portion (skip empty lines)
        csv_lines = [line for line in lines[csv_start_idx:] if line.strip()]
        clean_csv_data = '\n'.join(csv_lines)
        
        # Read the cleaned CSV data
        df = pd.read_csv(StringIO(clean_csv_data))
        
        # Convert YEAR,DOY to proper date
        if 'YEAR' in df.columns and 'DOY' in df.columns:
            # Create date from YEAR and DOY (Day of Year)
            df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['DOY'].astype(str), format='%Y-%j')
            # Remove original YEAR and DOY columns
            df = df.drop(['YEAR', 'DOY'], axis=1)
            # Reorder columns to put Date first
            cols = ['Date'] + [col for col in df.columns if col != 'Date']
            df = df[cols]
        
        return df
        
    except Exception as e:
        print(f"Error processing NASA CSV data: {e}")
        return None

def fetch_weather_for_district(gdf_row, start_date, end_date, parameters, original_gdf):
    """Fetch weather data for a single district"""
    try:
        state_name = gdf_row['STATE_UT']
        district_name = gdf_row['DISTRICT']
        
        # Get the row index to extract from original GeoDataFrame
        row_idx = gdf_row.name
        
        # Get centroid coordinates from the original GeoDataFrame
        longitude, latitude = get_centroid_coordinates(original_gdf.iloc[[row_idx]])
        if longitude is None or latitude is None:
            return None
        
        # Fetch NASA POWER data
        csv_data = fetch_nasa_power_data(longitude, latitude, start_date, end_date, parameters)
        if csv_data is None:
            return None
        
        # Process CSV data
        df = process_nasa_csv_data(csv_data)
        if df is None:
            return None
        
        # Add state and district columns
        df['State'] = state_name
        df['District'] = district_name
        
        # Reorder columns to put State and District first
        cols = ['State', 'District', 'Date'] + [col for col in df.columns if col not in ['State', 'District', 'Date']]
        df = df[cols]
        
        return df
        
    except Exception as e:
        print(f"Error processing district {district_name}: {e}")
        return None

def main():
    # Configuration
    SHAPEFILE_PATH = 'DISTRICT_BOUNDARY.shp'
    TARGET_STATES = ['UTTAR PRADESH', 'MADHYA PRADESH', 'HARYANA', 'PUNJAB', 'RAJASTHAN']
    START_DATE = '20200101'  # Start of 2020 (shorter period for testing)
    END_DATE = '20201231'    # End of 2020
    
    NASA_PARAMETERS = [
        "T2M_MAX",          # Maximum Temperature at 2 Meters
        "T2M_MIN",          # Minimum Temperature at 2 Meters
        "T2M",              # Mean Temperature at 2 Meters
        "PRECTOTCORR",      # Precipitation Corrected
        "RH2M",             # Relative Humidity at 2 Meters
        "ALLSKY_SFC_SW_DWN",# All Sky Insolation Incident on a Horizontal Surface
        "WS10M",            # Wind Speed at 10 Meters
        "QV2M",             # Specific Humidity at 2 Meters
        "PS"                # Surface Pressure
    ]
    
    print("TESTING: NASA POWER Weather Data Fetcher for Multiple Districts")
    print("=" * 70)
    print(f"Target States: {', '.join(TARGET_STATES)}")
    print(f"Date Range: {START_DATE} to {END_DATE} (TESTING WITH 1 YEAR)")
    print(f"Parameters: {', '.join(NASA_PARAMETERS)}")
    print("=" * 70)
    
    # Step 1: Read shapefile
    print("\n1. Reading shapefile...")
    gdf = read_shapefile(SHAPEFILE_PATH)
    if gdf is None:
        return
    
    # Step 2: Filter for target states
    print(f"\n2. Filtering for target states...")
    filtered_gdf = filter_districts_by_states(gdf, TARGET_STATES)
    if filtered_gdf is None:
        return
    
    # Step 3: Process only first 3 districts for testing
    print(f"\n3. Processing first 3 districts (TESTING)...")
    all_data = []
    total_districts = 3  # Only test with 3 districts
    successful_districts = 0
    failed_districts = 0
    
    start_time = time.time()
    
    for idx, (_, district_row) in enumerate(filtered_gdf.head(3).iterrows(), 1):
        state_name = district_row['STATE_UT']
        district_name = district_row['DISTRICT']
        
        print(f"\n[{idx}/{total_districts}] Processing {district_name}, {state_name}...")
        
        # Fetch weather data for this district
        district_data = fetch_weather_for_district(district_row, START_DATE, END_DATE, NASA_PARAMETERS, filtered_gdf)
        
        if district_data is not None:
            all_data.append(district_data)
            successful_districts += 1
            print(f"  ✅ Success: {len(district_data)} records")
        else:
            failed_districts += 1
            print(f"  ❌ Failed")
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Step 4: Combine all data
    print(f"\n4. Combining data...")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined data: {len(combined_df)} total records")
        print(f"Date range: {combined_df['Date'].min().strftime('%Y-%m-%d')} to {combined_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Step 5: Save data to CSV
        print(f"\n5. Saving data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"TEST_NASA_POWER_3_Districts_{START_DATE}_{END_DATE}_{timestamp}.csv"
        combined_df.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
        
        # Final summary
        print(f"\n6. Final Summary:")
        print(f"   Total districts processed: {total_districts}")
        print(f"   Successful: {successful_districts}")
        print(f"   Failed: {failed_districts}")
        print(f"   Success rate: {successful_districts/total_districts*100:.1f}%")
        print(f"   Total records: {len(combined_df)}")
        print(f"   Output file: {output_filename}")
        
        # Show sample data
        print(f"\nSample data (first 3 rows):")
        print(combined_df.head(3))
        
        # Show unique states and districts
        print(f"\nUnique states: {combined_df['State'].unique()}")
        print(f"Unique districts: {combined_df['District'].unique()}")
        
    else:
        print("No data was successfully fetched!")
    
    print("\n" + "=" * 70)
    print("TEST completed!")

if __name__ == "__main__":
    main() 