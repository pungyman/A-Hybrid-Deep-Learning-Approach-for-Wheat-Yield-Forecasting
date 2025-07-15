import geopandas as gpd
import matplotlib.pyplot as plt

# Read the shapefile
district_boundary = gpd.read_file('../shapefiles/DISTRICT_BOUNDARY.shp')

# Display basic information about the shapefile
print("Basic Information about the Shapefile:")
print("-" * 40)
print(f"Number of features: {len(district_boundary)}")
print("\nColumns in the dataset:")
print(district_boundary.columns.tolist())
print("\nCoordinate Reference System (CRS):")
print(district_boundary.crs)

# Display the first few rows of attribute data
print("\nFirst few rows of attribute data:")
print(district_boundary.head())

# Create a simple plot of the boundary
plt.figure(figsize=(12, 8))
district_boundary.plot()
plt.title('District Boundary')
plt.axis('equal')
plt.savefig('district_boundary_plot.png')
plt.close() 