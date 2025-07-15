# NDVI/EVI Extraction from Google Earth Engine

This project extracts NDVI (Normalized Difference Vegetation Index) and EVI (Enhanced Vegetation Index) data for Indian districts using Google Earth Engine and the MOD09Q1 dataset.

## Features

- **Automated extraction** of NDVI and EVI time series data
- **District-level analysis** using official boundary shapefiles
- **8-day temporal resolution** using MODIS MOD09Q1 surface reflectance data
- **Extensible state selection** - easily add more states
- **CSV output** with comprehensive statistics
- **Time period**: 2000-2024 (based on data availability)

## Prerequisites

1. **Google Earth Engine Account**: Sign up at [earthengine.google.com](https://earthengine.google.com)
2. **Python 3.7+** installed on your system
3. **District boundary shapefile** (included in this project)

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
python setup_and_run.py
```

This will:
- Install all required packages
- Authenticate with Google Earth Engine
- Check for required files
- Run the extraction

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Authenticate with Google Earth Engine**:
   ```bash
   earthengine authenticate
   ```

3. **Run the extraction**:
   ```bash
   python extract_ndvi_evi.py
   ```

## Configuration

### Adding More States

To extract data for additional states, modify the `STATES_LIST` in `extract_ndvi_evi.py`:

```python
STATES_LIST = ["Punjab", "Haryana", "Uttar Pradesh"]  # Add more states here
```

### Changing Date Range

Modify the date range in `extract_ndvi_evi.py`:

```python
START_DATE = "2010-01-01"  # Change start date
END_DATE = "2023-12-31"    # Change end date
```

## Output Files

The script generates two CSV files in the `output/` directory:

1. **Main data file**: `ndvi_evi_data_YYYYMMDD_HHMMSS.csv`
   - Contains all extracted data points
   - Columns: date, year, month, day, district, state, ndvi, evi

2. **Summary file**: `ndvi_evi_data_summary_YYYYMMDD_HHMMSS.csv`
   - Statistical summary by district
   - Includes count, mean, std, min, max for both NDVI and EVI

## Data Description

### NDVI (Normalized Difference Vegetation Index)
- **Range**: -1 to +1
- **Interpretation**: 
  - Values close to +1 indicate dense vegetation
  - Values close to 0 indicate bare soil or rock
  - Negative values indicate water bodies

### EVI (Enhanced Vegetation Index)
- **Range**: -1 to +1
- **Interpretation**:
  - Similar to NDVI but with improved sensitivity in high biomass regions
  - Less affected by atmospheric conditions and canopy background

### Dataset Details
- **Source**: MODIS MOD09Q1 Surface Reflectance (8-day composite)
- **Spatial Resolution**: 250 meters
- **Temporal Resolution**: 8 days
- **Bands Used**: 
  - Band 1 (Red): 620-670 nm
  - Band 2 (NIR): 841-876 nm
  - Band 3 (Blue): 459-479 nm

## File Structure

```
remote-sensing/
├── extract_ndvi_evi.py      # Main extraction script
├── setup_and_run.py         # Automated setup script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── DATASET_USAGE.md         # Available datasets' information
├── metadata/
   ├── read_shapefile.py     # Reading and extracting information from the shapefiles
```

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   ```
   Error initializing Google Earth Engine
   ```
   **Solution**: Run `earthengine authenticate` and follow the browser authentication process.

2. **Shapefile Not Found**:
   ```
   Shapefile not found: shapefiles/DISTRICT_BOUNDARY.shp
   ```
   **Solution**: Ensure all shapefile components (.shp, .dbf, .prj, etc.) are in the `shapefiles/` directory.

3. **Memory/Timeout Issues**:
   - Large date ranges may cause timeouts
   - Reduce the date range or process fewer districts at once

4. **No Data Returned**:
   - Check if the state name exactly matches the shapefile
   - Verify the date range has available data

### Getting Help

1. Check the [Google Earth Engine documentation](https://developers.google.com/earth-engine)
2. Verify your Earth Engine account has the necessary permissions
3. Ensure your internet connection is stable for large downloads

## Technical Details

### Calculation Formulas for MOD09A1 Dataset

**NDVI**:
```
NDVI = (NIR - Red) / (NIR + Red)
```

**EVI**:
```
EVI = 2.5 * (NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1)
```

### Processing Steps

1. Load district boundaries from shapefile
2. Filter districts by specified states
3. Load MOD09Q1 surface reflectance collection
4. Calculate NDVI and EVI for each image
5. Extract time series for each district using zonal statistics
6. Export results to CSV format

## Performance Notes

- **Processing Time**: Depends on date range and number of districts
- **Rate Limiting**: Small delays added between districts to avoid API limits
- **Memory Usage**: Optimized for large datasets with batch processing

## License

This project is open source. Please ensure you comply with Google Earth Engine's terms of service when using this code.

## Contributing

Feel free to submit issues and enhancement requests! 