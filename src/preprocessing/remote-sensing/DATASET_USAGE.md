# NDVI/EVI Dataset Usage Guide

This project now supports two different MODIS datasets for NDVI and EVI extraction. Choose the one that best fits your needs.

## Dataset Options

### 1. MOD09A1 - Manual Computation
- **Product**: MODIS/Terra Surface Reflectance 8-Day Global 500m
- **Resolution**: 500m
- **Temporal**: 8-day composites
- **Computation**: Manual calculation using the exact formulas:
  - NDVI = (NIR - Red) / (NIR + Red)
  - EVI = 2.5 × (NIR - Red) / (NIR + 6 × Red - 7.5 × Blue + 1)
- **Bands Used**: Red (620-670nm), NIR (841-876nm), Blue (459-479nm)
- **Advantages**: 
  - More spectral bands available
  - Full control over computation
  - Better for research requiring custom calculations
- **Disadvantages**: 
  - Requires more processing time
  - Larger data volumes

### 2. MOD13Q1 - Precomputed Values
- **Product**: MODIS/Terra Vegetation Indices 16-Day Global 250m
- **Resolution**: 250m (better spatial resolution)
- **Temporal**: 16-day composites
- **Computation**: Uses precomputed NDVI and EVI values
- **Advantages**:
  - Faster processing (no computation needed)
  - Higher spatial resolution (250m vs 500m)
  - Quality-controlled by NASA
  - Smaller data volumes
- **Disadvantages**:
  - Less frequent (16-day vs 8-day)
  - No access to individual spectral bands
  - Less flexibility for custom computations

## How to Choose

| Use Case | Recommended Dataset | Reason |
|----------|-------------------|---------|
| General vegetation monitoring | MOD13Q1 | Faster, higher resolution, quality-controlled |
| Research requiring exact formulas | MOD09A1 | Full control over calculations |
| Custom index development | MOD09A1 | Access to individual spectral bands |
| Operational monitoring | MOD13Q1 | Faster processing, reliable results |
| Time series analysis | MOD13Q1 | Higher spatial resolution, quality flags |

## Usage Instructions

### 1. Quick Test (Recommended First Step)
```bash
python test_datasets.py
```
This will test both datasets with minimal data extraction to verify they work correctly.

### 2. Demo Extraction
```bash
python demo_extract.py
```
Edit the `DATASET_TYPE` variable in the script to choose between "MOD09A1" or "MOD13Q1".

### 3. Full Extraction
```bash
python extract_ndvi_evi.py
```
Edit the `DATASET_TYPE` variable in the script to choose your preferred dataset.

## Configuration

In any of the main scripts (`extract_ndvi_evi.py`, `demo_extract.py`, `test_datasets.py`), simply change:

```python
# For manual computation with full spectral bands
DATASET_TYPE = "MOD09A1"

# For precomputed values with higher resolution
DATASET_TYPE = "MOD13Q1"
```

## Technical Details

### MOD09A1 Technical Specifications
- **Collection**: MODIS/061/MOD09A1
- **Spatial Resolution**: 500m
- **Temporal Resolution**: 8-day composite
- **Spectral Bands**: 
  - Band 1 (Red): 620-670 nm
  - Band 2 (NIR): 841-876 nm
  - Band 3 (Blue): 459-479 nm
- **Scale Factor**: 0.0001
- **Date Range**: 2000-02-24 to present

### MOD13Q1 Technical Specifications
- **Collection**: MODIS/061/MOD13Q1
- **Spatial Resolution**: 250m
- **Temporal Resolution**: 16-day composite
- **Indices**: 
  - NDVI: Normalized Difference Vegetation Index
  - EVI: Enhanced Vegetation Index
- **Scale Factor**: 0.0001
- **Date Range**: 2000-02-18 to present

## Mathematical Formulas

### MOD09A1 (Manual Computation)
```
NDVI = (NIR - Red) / (NIR + Red)
EVI = 2.5 × (NIR - Red) / (NIR + 6 × Red - 7.5 × Blue + 1)
```

### MOD13Q1 (Precomputed)
Values are already computed by NASA using optimized algorithms and quality control.

## Performance Comparison

| Aspect | MOD09A1 | MOD13Q1 |
|--------|---------|---------|
| Processing Speed | Slower | Faster |
| Data Volume | Larger | Smaller |
| Spatial Resolution | 500m | 250m |
| Temporal Resolution | 8-day | 16-day |
| Computation Control | Full | Limited |
| Quality Control | Manual | Automated |

## Example Output

Both datasets produce the same output format:
```csv
date,year,month,day,district,state,ndvi,evi
2023-06-01,2023,6,1,AMBALA,HARYANA,0.6234,0.4567
2023-06-01,2023,6,1,BHIWANI,HARYANA,0.5891,0.4123
...
```

## Troubleshooting

### Common Issues

1. **No data extracted**: Check date ranges and ensure Google Earth Engine authentication
2. **Slow processing**: Try MOD13Q1 for faster results
3. **Memory errors**: Use chunked processing or reduce date range
4. **Authentication errors**: Run `earthengine authenticate`

### Getting Help

1. Run the test script first: `python test_datasets.py`
2. Check the logs for specific error messages
3. Verify your Google Earth Engine project is set up correctly
4. Ensure you have the required dependencies installed

## Best Practices

1. **Start with testing**: Always run `test_datasets.py` first
2. **Use demos**: Try `demo_extract.py` before full extraction
3. **Choose based on needs**: MOD13Q1 for speed, MOD09A1 for research
4. **Monitor resources**: Both datasets require stable internet and adequate disk space
5. **Save intermediate results**: Use chunked processing for large extractions

## Data Citations

- MOD09A1: Vermote, E. (2015). MOD09A1 MODIS/Terra Surface Reflectance 8-Day L3 Global 500m SIN Grid V006. NASA EOSDIS Land Processes DAAC.
- MOD13Q1: Didan, K. (2015). MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN Grid V006. NASA EOSDIS Land Processes DAAC. 