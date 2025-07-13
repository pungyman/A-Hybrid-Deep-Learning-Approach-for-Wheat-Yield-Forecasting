import pandas as pd
import geopandas as gpd
import requests
from io import StringIO

# Test with a single district to see the API response format
SHAPEFILE_NAME = 'DISTRICT_BOUNDARY.shp'
NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Load shapefile and get first district from Uttar Pradesh
india_gdf = gpd.read_file(SHAPEFILE_NAME)
india_gdf = india_gdf.rename(columns={'STATE_UT': 'STATE'})
up_districts = india_gdf[india_gdf['STATE'] == 'UTTAR PRADESH'].to_crs("EPSG:4326")
test_district = up_districts.iloc[0]

print(f"Testing with district: {test_district['DISTRICT']}")

# Get centroid
centroid = test_district.geometry.centroid
lat, lon = round(centroid.y, 4), round(centroid.x, 4)
print(f"Centroid: {lat}, {lon}")

# Make API request
params = {
    "parameters": "T2M_MAX",
    "community": "AG",
    "format": "CSV",
    "start": "20230101",
    "end": "20241231",
    "latitude": str(lat),
    "longitude": str(lon),
}

print("Making API request...")
response = requests.get(NASA_POWER_API_URL, params=params, timeout=120)
print(f"Status code: {response.status_code}")

if response.status_code == 200:
    content = response.text
    print("\n--- RAW API RESPONSE (first 2000 characters) ---")
    print(content[:2000])
    print("--- END RAW RESPONSE ---\n")
    
    # Analyze the response structure
    lines = content.splitlines()
    print(f"Total lines in response: {len(lines)}")
    
    print("\n--- FIRST 20 LINES ---")
    for i, line in enumerate(lines[:20]):
        print(f"Line {i:2d}: {repr(line)}")
    
    # Find potential header lines
    print("\n--- LOOKING FOR HEADER LINES ---")
    for i, line in enumerate(lines):
        if 'LAT' in line or 'YEAR' in line or line.count(',') > 3:
            print(f"Line {i:2d}: {line}")
        if i > 50:  # Don't check too many lines
            break
    
    # Try to parse with pandas
    print("\n--- TRYING TO PARSE WITH PANDAS ---")
    try:
        # Find the actual data start
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("LAT,LON,YEAR") or (line.count(',') >= 4 and not line.startswith('#') and not line.startswith('-')):
                data_start = i
                break
        
        if data_start is not None:
            print(f"Found data starting at line: {data_start}")
            print(f"Header line: {lines[data_start]}")
            
            # Try to read CSV
            csv_data = '\n'.join(lines[data_start:])
            df = pd.read_csv(StringIO(csv_data))
            print(f"Successfully parsed! Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df.head())
        else:
            print("Could not find data start line")
            
    except Exception as e:
        print(f"Parsing error: {e}")
        
else:
    print(f"API request failed with status {response.status_code}")
    print(response.text)