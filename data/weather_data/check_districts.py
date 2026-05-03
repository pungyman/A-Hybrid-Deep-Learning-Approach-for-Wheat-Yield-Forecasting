import geopandas as gpd

# Read the shapefile
gdf = gpd.read_file('DISTRICT_BOUNDARY.shp')

# Target states
target_states = ['UTTAR PRADESH', 'MADHYA PRADESH', 'HARYANA', 'PUNJAB', 'RAJASTHAN']

print("Available districts in target states:")
print("=" * 50)

total_districts = 0
for state in target_states:
    districts = gdf[gdf['STATE_UT'] == state]['DISTRICT'].dropna().unique()
    print(f"{state}: {len(districts)} districts")
    total_districts += len(districts)
    
    # Show first few districts
    if len(districts) > 0:
        print(f"  Sample: {sorted(districts)[:3]}")
    print()

print(f"Total districts to process: {total_districts}") 