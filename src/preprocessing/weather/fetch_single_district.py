import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import Point
from io import StringIO

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

def filter_district(gdf, state_name, district_name):
    """Filter GeoDataFrame for specific state and district"""
    try:
        # Filter for the specific state and district
        filtered_gdf = gdf[
            (gdf['STATE_UT'].str.upper() == state_name.upper()) & 
            (gdf['DISTRICT'].str.upper() == district_name.upper())
        ]
        
        if len(filtered_gdf) == 0:
            print(f"No records found for {district_name} in {state_name}")
            print(f"Available states: {gdf['STATE_UT'].unique()}")
            print(f"Available districts in {state_name}: {gdf[gdf['STATE_UT'].str.upper() == state_name.upper()]['DISTRICT'].unique()}")
            return None
        
        print(f"Found {len(filtered_gdf)} record(s) for {district_name}, {state_name}")
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
        
        # Create a GeoSeries with the centroid point
        centroid_point = Point(centroid.x, centroid.y)
        centroid_gdf = gpd.GeoDataFrame([1], geometry=[centroid_point], crs=gdf.crs)
        
        # Transform to WGS84 (EPSG:4326) - geographic coordinates
        centroid_wgs84 = centroid_gdf.to_crs(epsg=4326)
        
        # Extract longitude and latitude
        longitude = centroid_wgs84.geometry.iloc[0].x
        latitude = centroid_wgs84.geometry.iloc[0].y
        
        print(f"Original centroid coordinates: Longitude={centroid.x:.6f}, Latitude={centroid.y:.6f}")
        print(f"Transformed to WGS84: Longitude={longitude:.6f}, Latitude={latitude:.6f}")
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
        
        print(f"Fetching NASA POWER data for coordinates ({latitude:.6f}, {longitude:.6f})")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Parameters: {', '.join(parameters)}")
        
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Return CSV text directly
        csv_data = response.text
        print("Successfully fetched NASA POWER data in CSV format")
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
        
        print(f"Processed CSV data, found {len(lines)} total lines, CSV data starts at line {csv_start_idx + 1}")
        print(f"Final DataFrame has {len(df)} records")
        print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        
        return df
        
    except Exception as e:
        print(f"Error processing NASA CSV data: {e}")
        return None

def save_data_to_csv(df, filename):
    """Save DataFrame to CSV file"""
    try:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def main():
    # Configuration
    SHAPEFILE_PATH = 'DISTRICT_BOUNDARY.shp'
    STATE_NAME = 'UTTAR PRADESH'
    DISTRICT_NAME = 'AGRA'
    START_DATE = '20000101'  # Start of 2000
    END_DATE = '20241231'    # End of 2024
    
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
    
    print("Starting NASA POWER Weather Data Fetcher (CSV Format)")
    print("=" * 60)
    
    # Step 1: Read shapefile
    print("\n1. Reading shapefile...")
    gdf = read_shapefile(SHAPEFILE_PATH)
    if gdf is None:
        return
    
    # Step 2: Filter for Agra district in Uttar Pradesh
    print(f"\n2. Filtering for {DISTRICT_NAME} district in {STATE_NAME}...")
    filtered_gdf = filter_district(gdf, STATE_NAME, DISTRICT_NAME)
    if filtered_gdf is None:
        return
    
    # Step 3: Get centroid coordinates
    print(f"\n3. Calculating centroid coordinates...")
    longitude, latitude = get_centroid_coordinates(filtered_gdf)
    if longitude is None or latitude is None:
        return
    
    # Step 4: Fetch NASA POWER data
    print(f"\n4. Fetching NASA POWER data...")
    csv_data = fetch_nasa_power_data(longitude, latitude, START_DATE, END_DATE, NASA_PARAMETERS)
    if csv_data is None:
        return
    
    # Step 5: Process CSV data into DataFrame
    print(f"\n5. Processing CSV data...")
    df = process_nasa_csv_data(csv_data)
    if df is None:
        return
    
    # Step 6: Save data to CSV
    print(f"\n6. Saving data...")
    output_filename = f"NASA_POWER_Data_{DISTRICT_NAME}_{STATE_NAME}_{START_DATE}_{END_DATE}.csv"
    save_data_to_csv(df, output_filename)
    
    # Display basic statistics
    print(f"\n7. Data Summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   Parameters: {[col for col in df.columns if col != 'Date']}")
    
    # Display first few rows
    print(f"\nFirst 5 rows of data:")
    print(df.head())
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()