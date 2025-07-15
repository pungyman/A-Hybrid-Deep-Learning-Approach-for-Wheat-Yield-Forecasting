import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

class NDVIEVIExtractor:
    def __init__(self, shapefile_path, output_dir="output", dataset_type="MOD13Q1"):
        """
        Initialize the NDVI/EVI extractor
        
        Args:
            shapefile_path (str): Path to the district boundary shapefile
            output_dir (str): Directory to save output files
            dataset_type (str): Dataset type - either "MOD09A1" or "MOD13Q1"
                - MOD09A1: Manual computation with Red, NIR, Blue bands
                - MOD13Q1: Precomputed NDVI and EVI values
        """
        self.shapefile_path = shapefile_path
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.districts_gdf = None
        
        # Dataset configuration
        self.dataset_config = {
            "MOD09A1": {
                "collection": "MODIS/061/MOD09A1",
                "scale": 500,  # 500m resolution
                "description": "Daily Surface Reflectance (Manual NDVI/EVI computation)"
            },
            "MOD13Q1": {
                "collection": "MODIS/061/MOD13Q1", 
                "scale": 250,  # 250m resolution
                "description": "16-day Vegetation Indices (Precomputed NDVI/EVI)"
            }
        }
        
        # Validate dataset type
        if dataset_type not in self.dataset_config:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Must be one of: {list(self.dataset_config.keys())}")
        
        print(f"Dataset selected: {dataset_type}")
        print(f"Description: {self.dataset_config[dataset_type]['description']}")
        print(f"Resolution: {self.dataset_config[dataset_type]['scale']}m")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize Google Earth Engine
        self._initialize_ee()
        
    def _initialize_ee(self):
        """Initialize Google Earth Engine"""
        try:
            ee.Initialize(project='gee-wheat-yield')
            print("Google Earth Engine initialized successfully!")
        except Exception as e:
            print(f"Error initializing Google Earth Engine: {e}")
            print("Please run 'earthengine authenticate' first")
            raise
    
    def load_districts(self, states_list):
        """
        Load and filter districts by state names
        
        Args:
            states_list (list): List of state names to filter
        """
        print(f"Loading districts from {self.shapefile_path}")
        self.districts_gdf = gpd.read_file(self.shapefile_path)
        
        # Filter by states
        self.districts_gdf = self.districts_gdf[
            self.districts_gdf['STATE_UT'].isin(states_list)
        ].copy()
        
        print(f"Found {len(self.districts_gdf)} districts in {', '.join(states_list)}")
        print("Districts:", self.districts_gdf['DISTRICT'].tolist())
        
        return self.districts_gdf
    
    def calculate_ndvi_evi(self, image):
        """
        Calculate NDVI and EVI based on dataset type
        
        Args:
            image: Earth Engine image
        """
        if self.dataset_type == "MOD09A1":
            return self._calculate_ndvi_evi_mod09a1(image)
        elif self.dataset_type == "MOD13Q1":
            return self._extract_ndvi_evi_mod13q1(image)
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
    
    def _calculate_ndvi_evi_mod09a1(self, image):
        """
        Calculate NDVI and EVI from MOD09A1 surface reflectance (Manual computation)
        
        Formulas:
        - NDVI = (NIR - Red) / (NIR + Red)
        - EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
        
        Args:
            image: Earth Engine image
        """
        # Get surface reflectance bands for MOD09A1
        red = image.select('sur_refl_b01')    # Red band (620-670nm)
        nir = image.select('sur_refl_b02')    # NIR band (841-876nm)  
        blue = image.select('sur_refl_b03')   # Blue band (459-479nm)
        
        # Scale factor for MOD09A1 (reflectance values are scaled by 10000)
        scale_factor = 0.0001
        red = red.multiply(scale_factor)
        nir = nir.multiply(scale_factor)
        blue = blue.multiply(scale_factor)
        
        # Calculate NDVI: (NIR - Red) / (NIR + Red)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        
        # Calculate EVI: 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
        evi = nir.subtract(red).divide(
            nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
        ).multiply(2.5).rename('EVI')
        
        # Add date information
        date = image.date()
        
        return image.addBands([ndvi, evi]).set({
            'date': date,
            'year': date.get('year'),
            'month': date.get('month'),
            'day': date.get('day')
        })
    
    def _extract_ndvi_evi_mod13q1(self, image):
        """
        Extract precomputed NDVI and EVI from MOD13Q1 vegetation indices
        
        Args:
            image: Earth Engine image
        """
        # Get precomputed NDVI and EVI bands (raw values - scaling applied in Python)
        ndvi = image.select('NDVI').rename('NDVI')
        evi = image.select('EVI').rename('EVI')
        
        # Add date information
        date = image.date()
        
        return image.addBands([ndvi, evi]).set({
            'date': date,
            'year': date.get('year'),
            'month': date.get('month'),
            'day': date.get('day')
        })
    
    def extract_time_series(self, start_date='2000-02-18', end_date='2024-12-31'):
        """
        Original sequential extraction method (most reliable but slowest)
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        print(f"Extracting NDVI/EVI data from {start_date} to {end_date}")
        print(f"Using dataset: {self.dataset_type}")
        
        # Load collection based on dataset type
        collection = ee.ImageCollection(self.dataset_config[self.dataset_type]['collection']) \
            .filterDate(start_date, end_date) \
            .map(self.calculate_ndvi_evi)
        
        # Get collection info
        collection_size = collection.size()
        print(f"Found {collection_size.getInfo()} images in the collection")
        
        # Initialize results list
        all_results = []
        
        # Process each district
        for idx, district_row in self.districts_gdf.iterrows():
            district_name = district_row['DISTRICT']
            state_name = district_row['STATE_UT']
            
            print(f"Processing district: {district_name}, {state_name}")
            
            # Convert district geometry to Earth Engine geometry
            district_geom = self._geopandas_to_ee_geometry(district_row.geometry)
            
            # Extract time series for this district
            district_results = self._extract_district_time_series(
                collection, district_geom, district_name, state_name
            )
            
            all_results.extend(district_results)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(all_results)
        return df
    
    def extract_time_series_optimized(self, start_date='2000-02-18', end_date='2024-12-31', 
                                     batch_size=50, max_workers=4):
        """
        Optimized extraction that processes multiple districts in parallel
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            batch_size (int): Number of time periods to process in each batch
            max_workers (int): Number of parallel workers (districts per batch)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        print(f"Extracting NDVI/EVI data from {start_date} to {end_date}")
        print(f"Using dataset: {self.dataset_type}")
        print(f"Using optimized batch processing: {batch_size} periods, {max_workers} workers")
        
        # Load collection based on dataset type
        collection = ee.ImageCollection(self.dataset_config[self.dataset_type]['collection']) \
            .filterDate(start_date, end_date) \
            .map(self.calculate_ndvi_evi)
        
        # Get collection info
        collection_size = collection.size()
        print(f"Found {collection_size.getInfo()} images in the collection")
        
        # Convert all district geometries upfront
        print("Converting district geometries...")
        district_data = []
        for idx, district_row in self.districts_gdf.iterrows():
            district_data.append({
                'name': district_row['DISTRICT'],
                'state': district_row['STATE_UT'],
                'geometry': self._geopandas_to_ee_geometry(district_row.geometry)
            })
        
        # Process all districts for the entire time series in one batch
        print(f"Processing {len(district_data)} districts in parallel...")
        
        all_results = []
        results_lock = threading.Lock()
        
        def process_district(district_info):
            """Process one district for the entire time series"""
            district_name = district_info['name']
            state_name = district_info['state']
            district_geom = district_info['geometry']
            
            try:
                print(f"Processing district: {district_name}")
                
                # Extract time series for this district
                district_results = self._extract_district_time_series_batch(
                    collection, district_geom, district_name, state_name
                )
                
                with results_lock:
                    all_results.extend(district_results)
                    
                print(f"✓ Completed {district_name}: {len(district_results)} records")
                return len(district_results)
                
            except Exception as e:
                print(f"✗ Error processing {district_name}: {str(e)}")
                return 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all district processing tasks
            future_to_district = {
                executor.submit(process_district, district_info): district_info['name']
                for district_info in district_data
            }
            
            # Process completed tasks
            completed_count = 0
            total_records = 0
            
            for future in as_completed(future_to_district):
                district_name = future_to_district[future]
                try:
                    record_count = future.result()
                    total_records += record_count
                    completed_count += 1
                    
                    progress = (completed_count / len(district_data)) * 100
                    print(f"Progress: {completed_count}/{len(district_data)} districts ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"District {district_name} generated an exception: {e}")
        
        print(f"Total records extracted: {total_records}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        return df
    
    def _extract_district_time_series_batch(self, collection, district_geom, district_name, state_name):
        """Extract time series for a single district using batch processing"""
        
        # Create a feature collection with all districts for batch processing
        def extract_values(image):
            # Calculate mean NDVI and EVI for the district
            reduction = image.select(['NDVI', 'EVI']).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district_geom,
                scale=self.dataset_config[self.dataset_type]['scale'],
                maxPixels=1e9,
                bestEffort=True,  # Allow partial coverage
                tileScale=2  # Reduce memory usage
            )
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'year': image.get('year'),
                'month': image.get('month'),
                'day': image.get('day'),
                'ndvi': reduction.get('NDVI'),
                'evi': reduction.get('EVI'),
                'district': district_name,
                'state': state_name
            })
        
        # Map over the collection
        time_series = collection.map(extract_values)
        
        # Convert to list and get info (this is the bottleneck)
        try:
            time_series_list = time_series.getInfo()
        except Exception as e:
            print(f"Error getting time series for {district_name}: {e}")
            return []
        
        # Extract results
        results = []
        for feature in time_series_list['features']:
            props = feature['properties']
            if props['ndvi'] is not None and props['evi'] is not None:
                # Apply scaling for MOD13Q1 values (raw values need to be scaled by 0.0001)
                ndvi_value = props['ndvi']
                evi_value = props['evi']
                
                if self.dataset_type == "MOD13Q1":
                    # MOD13Q1 values are stored as integers scaled by 10000
                    ndvi_value = ndvi_value * 0.0001
                    evi_value = evi_value * 0.0001
                
                results.append({
                    'date': props['date'],
                    'year': props['year'],
                    'month': props['month'],
                    'day': props['day'],
                    'district': props['district'],
                    'state': props['state'],
                    'ndvi': ndvi_value,
                    'evi': evi_value
                })
        
        return results
    
    def extract_time_series_chunked(self, start_date='2000-02-18', end_date='2024-12-31', 
                                  chunk_years=2, save_intermediate=True):
        """
        Extract data in yearly chunks to avoid memory issues and provide progress checkpoints
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            chunk_years (int): Number of years to process in each chunk
            save_intermediate (bool): Save intermediate results
        """
        from datetime import datetime, timedelta
        
        print(f"Extracting NDVI/EVI data from {start_date} to {end_date}")
        print(f"Using chunked processing: {chunk_years}-year chunks")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_results = []
        chunk_num = 1
        
        current_start = start_dt
        while current_start < end_dt:
            # Calculate chunk end date
            chunk_end = datetime(current_start.year + chunk_years, 1, 1) - timedelta(days=1)
            if chunk_end > end_dt:
                chunk_end = end_dt
            
            chunk_start_str = current_start.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')
            
            print(f"\n{'='*60}")
            print(f"Processing Chunk {chunk_num}: {chunk_start_str} to {chunk_end_str}")
            print(f"{'='*60}")
            
            # Extract data for this chunk
            chunk_results = self.extract_time_series_optimized(
                start_date=chunk_start_str,
                end_date=chunk_end_str,
                max_workers=2  # Reduce workers for chunked processing
            )
            
            if not chunk_results.empty:
                all_results.append(chunk_results)
                
                # Save intermediate results
                if save_intermediate:
                    chunk_filename = f"chunk_{chunk_num}_{current_start.year}_{chunk_end.year}"
                    self.save_results(chunk_results, chunk_filename)
                    
                print(f"✓ Chunk {chunk_num} completed: {len(chunk_results)} records")
            else:
                print(f"✗ Chunk {chunk_num} returned no data")
            
            # Move to next chunk
            current_start = datetime(chunk_end.year + 1, 1, 1)
            chunk_num += 1
            
            # Small delay between chunks
            time.sleep(1)
        
        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            print(f"\n{'='*60}")
            print(f"ALL CHUNKS COMPLETED!")
            print(f"Total records: {len(final_df)}")
            print(f"{'='*60}")
            return final_df
        else:
            return pd.DataFrame()
    
    def _geopandas_to_ee_geometry(self, geom):
        """Convert GeoPandas geometry to Earth Engine geometry"""
        # Convert to WGS84 if not already
        if self.districts_gdf.crs != 'EPSG:4326':
            geom = gpd.GeoSeries([geom], crs=self.districts_gdf.crs).to_crs('EPSG:4326').iloc[0]
        
        if geom.geom_type == 'Polygon':
            # Get coordinates and convert 3D to 2D if necessary
            coords = list(geom.exterior.coords)
            if len(coords[0]) == 3:  # 3D coordinates
                coords_2d = [(x, y) for x, y, z in coords]
            else:  # 2D coordinates
                coords_2d = coords
            coords = [coords_2d]
            
        elif geom.geom_type == 'MultiPolygon':
            coords = []
            for polygon in geom.geoms:
                poly_coords = list(polygon.exterior.coords)
                if len(poly_coords[0]) == 3:  # 3D coordinates
                    coords_2d = [(x, y) for x, y, z in poly_coords]
                else:  # 2D coordinates
                    coords_2d = poly_coords
                coords.append(coords_2d)
        else:
            raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
        
        return ee.Geometry.Polygon(coords)
    
    def _extract_district_time_series(self, collection, district_geom, district_name, state_name):
        """Extract time series for a single district"""
        # Get the time series as a list
        def extract_values(image):
            # Calculate mean NDVI and EVI for the district
            ndvi_mean = image.select('NDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district_geom,
                scale=self.dataset_config[self.dataset_type]['scale'],
                maxPixels=1e9
            ).get('NDVI')
            
            evi_mean = image.select('EVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district_geom,
                scale=self.dataset_config[self.dataset_type]['scale'],
                maxPixels=1e9
            ).get('EVI')
            
            return ee.Feature(None, {
                'date': image.date().format('YYYY-MM-dd'),
                'year': image.get('year'),
                'month': image.get('month'),
                'day': image.get('day'),
                'ndvi': ndvi_mean,
                'evi': evi_mean,
                'district': district_name,
                'state': state_name
            })
        
        # Map over the collection
        time_series = collection.map(extract_values)
        
        # Convert to list and get info
        time_series_list = time_series.getInfo()
        
        # Extract results
        results = []
        for feature in time_series_list['features']:
            props = feature['properties']
            if props['ndvi'] is not None and props['evi'] is not None:
                # Apply scaling for MOD13Q1 values (raw values need to be scaled by 0.0001)
                ndvi_value = props['ndvi']
                evi_value = props['evi']
                
                if self.dataset_type == "MOD13Q1":
                    # MOD13Q1 values are stored as integers scaled by 10000
                    ndvi_value = ndvi_value * 0.0001
                    evi_value = evi_value * 0.0001
                
                results.append({
                    'date': props['date'],
                    'year': props['year'],
                    'month': props['month'],
                    'day': props['day'],
                    'district': props['district'],
                    'state': props['state'],
                    'ndvi': ndvi_value,
                    'evi': evi_value
                })
        
        return results
    
    def save_results(self, df, filename_prefix="ndvi_evi_data"):
        """Save results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"Results saved to: {filepath}")
        
        # Also save a summary
        summary_filename = f"{filename_prefix}_summary_{timestamp}.csv"
        summary_filepath = os.path.join(self.output_dir, summary_filename)
        
        summary = df.groupby(['state', 'district']).agg({
            'ndvi': ['count', 'mean', 'std', 'min', 'max'],
            'evi': ['count', 'mean', 'std', 'min', 'max']
        }).round(4)
        
        summary.to_csv(summary_filepath)
        print(f"Summary saved to: {summary_filepath}")
        
        return filepath, summary_filepath

def main():
    """Main function to run the extraction"""
    # Configuration
    SHAPEFILE_PATH = "shapefiles/DISTRICT_BOUNDARY.shp"
    STATES_LIST = ["PUNJAB", "HARYANA"]  # Easily extendable - just add more states
    START_DATE = "2000-02-18"  # Dataset start date
    END_DATE = "2024-12-31"    # Complete 2024 data
    
    # DATASET SELECTION - Choose one of the following:
    # Option 1: MOD09A1 - Manual computation with Red, NIR, Blue bands
    # Option 2: MOD13Q1 - Precomputed NDVI and EVI values
    DATASET_TYPE = "MOD13Q1"  # Change to "MOD13Q1" for precomputed values
    
    # Initialize extractor
    extractor = NDVIEVIExtractor(SHAPEFILE_PATH, dataset_type=DATASET_TYPE)
    
    # Load districts for specified states
    districts_df = extractor.load_districts(STATES_LIST)
    
    # Choose extraction method:
    # Option 1: Chunked processing (recommended for full dataset) - no intermediate files
    results_df = extractor.extract_time_series_chunked(START_DATE, END_DATE, chunk_years=2, save_intermediate=False)
    
    # Option 2: Optimized parallel processing (faster but may hit memory limits)
    # results_df = extractor.extract_time_series_optimized(START_DATE, END_DATE, max_workers=4)
    
    # Option 3: Original sequential processing (slowest but most reliable)
    # results_df = extractor.extract_time_series(START_DATE, END_DATE)
    
    # Save results
    if not results_df.empty:
        main_file, summary_file = extractor.save_results(results_df)
        
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE!")
        print("="*50)
        print(f"Total records: {len(results_df)}")
        print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
        print(f"Districts: {results_df['district'].nunique()}")
        print(f"Main data file: {main_file}")
        print(f"Summary file: {summary_file}")
        
        # Display sample data
        print("\nSample data:")
        print(results_df.head())
        
    else:
        print("No data extracted. Please check your parameters and try again.")

if __name__ == "__main__":
    main() 