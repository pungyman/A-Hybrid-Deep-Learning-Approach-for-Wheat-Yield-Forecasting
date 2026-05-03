"""
Soil data wrangling script.

This script reads the district_soil_data.csv file and transforms it into a format
where each district has a 2D matrix of soil features (rows) by depths (columns).

The output dataframe has two columns:
- DISTRICT_ID: renamed from district_id
- soil_features: 2D array where rows are features (sorted alphabetically) 
  and columns are depths (sorted in ascending order)
"""

import pandas as pd
import numpy as np


def wrangle_soil_data(input_file='downloaded_data/district_soil_data.csv'):
    """
    Wrangle soil data into the desired format.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: DataFrame with DISTRICT_ID and soil_features columns
    """
    
    # Read the data
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original data shape: {df.shape}")
    print(f"Unique districts: {df['district_id'].nunique()}")
    print(f"Unique features: {df['feature_name'].nunique()}")
    print(f"Unique depths: {df['depth'].nunique()}")
    
    # Define the correct depth order (ascending)
    depth_order = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
    
    # Get unique features and sort them alphabetically
    features = sorted(df['feature_name'].unique())
    
    print(f"\nFeatures (sorted alphabetically): {features}")
    print(f"Depths (sorted ascending): {depth_order}")
    
    # Create the result list
    result_data = []
    
    # Process each district
    for district_id in df['district_id'].unique():
        district_data = df[df['district_id'] == district_id]
        
        # Create a matrix to store the features x depths data
        # Rows = features, Columns = depths
        feature_matrix = np.full((len(features), len(depth_order)), np.nan)
        
        # Fill the matrix
        for _, row in district_data.iterrows():
            feature_idx = features.index(row['feature_name'])
            depth_idx = depth_order.index(row['depth'])
            feature_matrix[feature_idx, depth_idx] = row['value']
        
        # Convert to list of lists for JSON serialization
        soil_features = feature_matrix.tolist()
        
        result_data.append({
            'DISTRICT_ID': district_id,
            'soil_features': soil_features
        })
    
    # Create the final dataframe
    result_df = pd.DataFrame(result_data)
    
    print(f"\nResult dataframe shape: {result_df.shape}")
    print(f"Result columns: {result_df.columns.tolist()}")
    
    # Show sample of the first district's soil features matrix
    if len(result_df) > 0:
        print(f"\nSample soil features matrix for {result_df.iloc[0]['DISTRICT_ID']}:")
        sample_matrix = np.array(result_df.iloc[0]['soil_features'])
        print(f"Matrix shape: {sample_matrix.shape}")
        print(f"Features: {features}")
        print(f"Depths: {depth_order}")
        print("Matrix values:")
        print(sample_matrix)
    
    return result_df


def save_wrangled_data(df, output_file='downloaded_data/district_soil_data_wrangled.csv'):
    """
    Save the wrangled data to CSV.
    
    Args:
        df (pd.DataFrame): The wrangled dataframe
        output_file (str): Output file path
    """
    df.to_csv(output_file, index=False)
    print(f"\nWrangled data saved to: {output_file}")


if __name__ == "__main__":
    INPUT_FILE = 'downloaded_data_wcs/district_soil_data_7_states.csv'
    OUTPUT_FILE = 'downloaded_data_wcs/district_soil_data_7_states_wrangled.csv'

    # Wrangle the data
    wrangled_df = wrangle_soil_data(INPUT_FILE)
    
    # Save the result
    save_wrangled_data(wrangled_df, OUTPUT_FILE)
    
    print("\nWrangling completed successfully!")
