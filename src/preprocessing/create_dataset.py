import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import json

DATASET_DESC = "7 states; refined at 50% thresh; with soil features"

YIELD_DATA_PATH = '../../data/yield_data/yield_data_wrangled_7_states_refined_50.csv'
MONTHLY_FEATURES_PATH = '../../data/processed/monthly_features_7_states.csv'
SOIL_FEATURES_PATH = '../../data/soilgrids/downloaded_data_wcs/district_soil_data_7_states_wrangled.csv'

OUTPUT_DIR = '../../data/datasets/7_states/2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

soil_features_available = SOIL_FEATURES_PATH is not None

# Define the number of years for each split
TRAIN_YEARS_COUNT = 18
VAL_YEARS_COUNT = 3
# The remaining years will be allocated to the test set

yield_data = pd.read_csv(YIELD_DATA_PATH)
monthly_features = pd.read_csv(MONTHLY_FEATURES_PATH)

if soil_features_available:
    soil_data = pd.read_csv(SOIL_FEATURES_PATH)

##### 1. Process monthly features
#   - drop features for the month of May
#   - create sowing_year column
#   - sort by YEAR, then MONTH

monthly_features = monthly_features[monthly_features['MONTH'] != 5].copy()

monthly_features['sowing_year'] = monthly_features['YEAR'].copy()
monthly_features.loc[monthly_features['MONTH'] <= 4, 'sowing_year'] -= 1

monthly_features = monthly_features.sort_values(by=['YEAR', 'MONTH'], ignore_index=True)
feature_cols = [col for col in monthly_features.columns if col not in ['DISTRICT_ID', 'YEAR', 'MONTH', 'sowing_year']]

# Wrangle to a DataFrame with columns: DISTRICT_ID, sowing_year, feature_sequence
feature_sequences = monthly_features.groupby(['DISTRICT_ID', 'sowing_year'])[feature_cols].apply(lambda x: x.values.tolist())
feature_sequences = feature_sequences.reset_index().rename(columns={0: 'feature_sequence'})

##### 2. Merge with yield data

# Label as many of the data points as possible
labeled_feature_sequences = feature_sequences.merge(yield_data[['DISTRICT_ID', 'sowing_year', 'Yield']], on=['DISTRICT_ID', 'sowing_year'], how='left')

# Drop unlabeled data points
labeled_feature_sequences = labeled_feature_sequences.dropna()

##### 3. Add lagged yield feature (past_yield column)

# Add lagged yield: yield from previous sowing_year for the same DISTRICT_ID
labeled_feature_sequences = labeled_feature_sequences.sort_values(by=['DISTRICT_ID', 'sowing_year'], ignore_index=True)
labeled_feature_sequences['past_yield'] = labeled_feature_sequences.groupby('DISTRICT_ID')['Yield'].shift(1)

# Drop rows where lagged yield is missing (i.e., first year for each district)
labeled_feature_sequences = labeled_feature_sequences.dropna(subset=['past_yield']).reset_index(drop=True)

##### 4. Carry out train test split

# Find the unique, sorted years in the dataset
all_years = sorted(labeled_feature_sequences['sowing_year'].unique())
n_years = len(all_years)

print(f"\nData available for {n_years} years: from {all_years[0]} to {all_years[-1]}")

if (TRAIN_YEARS_COUNT + VAL_YEARS_COUNT) >= n_years:
    raise ValueError(
        f"The sum of training ({TRAIN_YEARS_COUNT}) and validation ({VAL_YEARS_COUNT}) "
        f"years must be less than the total number of years ({n_years}). "
        f"Please adjust the split."
    )

train_years = all_years[:TRAIN_YEARS_COUNT]
val_years = all_years[TRAIN_YEARS_COUNT : TRAIN_YEARS_COUNT + VAL_YEARS_COUNT]
test_years = all_years[TRAIN_YEARS_COUNT + VAL_YEARS_COUNT :]

print("Splitting data chronologically:")
print(f"  Training:   {len(train_years)} years ({train_years[0]}-{train_years[-1]})")
print(f"  Validation: {len(val_years)} years ({val_years[0]}-{val_years[-1]})")
print(f"  Test:       {len(test_years)} years ({test_years[0]}-{test_years[-1]})")

# Create dataframes based on the year splits
train_df = labeled_feature_sequences[labeled_feature_sequences['sowing_year'].isin(train_years)]
val_df = labeled_feature_sequences[labeled_feature_sequences['sowing_year'].isin(val_years)]
test_df = labeled_feature_sequences[labeled_feature_sequences['sowing_year'].isin(test_years)]

##### 5. Apply Z-score normalization to the features, and save the scaler objects

print("\nApplying Z-score normalization...")

# Reshape training data to be 2D [samples * timesteps, features] for fitting the scaler
train_sequences = np.array(train_df['feature_sequence'].tolist(), dtype=np.float32)
# train_sequences is a 3d array of shape samples, timesteps, features
if train_sequences.size == 0:
    raise ValueError("Training data is empty. Cannot fit the scaler.")
_, _, num_features = train_sequences.shape
train_sequences_reshaped = train_sequences.reshape(-1, num_features)
# train_sequences_reshaped has shape samples * timesteps, features

# Create and fit the scaler ONLY on the training data
scaler = StandardScaler()
scaler.fit(train_sequences_reshaped)
print("  Scaler fitted on training data.")

def apply_scaling(df, scaler):
    """Applies a fitted scaler to the 'feature_sequence' column of a dataframe."""
    if df.empty:
        return df
    df_scaled = df.copy()
    sequences = np.array(df_scaled['feature_sequence'].tolist(), dtype=np.float32)
    num_samples, seq_len, num_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, num_features)
    # sequences_reshaped has shape samples * timesteps, features
    scaled_sequences_reshaped = scaler.transform(sequences_reshaped)
    # reshape back to 3d array
    scaled_sequences = scaled_sequences_reshaped.reshape(num_samples, seq_len, num_features)
    # update the feature_sequence column with scaled features
    df_scaled['feature_sequence'] = [seq.tolist() for seq in scaled_sequences]
    return df_scaled

# Apply the fitted scaler to train, validation, and test
train_df = apply_scaling(train_df, scaler)
val_df = apply_scaling(val_df, scaler)
test_df = apply_scaling(test_df, scaler)

# Save the fitted scaler object for later use (during inference)
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"  Scaler object saved to '{scaler_path}'")

# Z-score normalize the lagged yield (past_yield) column using a separate scaler
past_yield_scaler = StandardScaler()
train_df['past_yield'] = past_yield_scaler.fit_transform(train_df[['past_yield']])
val_df['past_yield'] = past_yield_scaler.transform(val_df[['past_yield']])
test_df['past_yield'] = past_yield_scaler.transform(test_df[['past_yield']])

# Save the lagged-yield scaler
past_yield_scaler_path = os.path.join(OUTPUT_DIR, 'past_yield_scaler.joblib')
joblib.dump(past_yield_scaler, past_yield_scaler_path)
print(f"  Lagged yield scaler object saved to '{past_yield_scaler_path}'")

##### 6. Process and normalize soil features (if available)

if soil_features_available:
    print("\nProcessing soil features...")

    # Parse soil features from string format to numpy arrays
    def parse_soil_features(soil_features_str):
        """Parse soil features from string representation to numpy array."""
        import ast
        return np.array(ast.literal_eval(soil_features_str), dtype=np.float32)


    # Apply parsing to soil data
    soil_data['soil_features_parsed'] = soil_data['soil_features'].apply(parse_soil_features)

    # Merge soil features with the main dataset
    print("  Merging soil features with main dataset...")
    train_df = train_df.merge(soil_data[['DISTRICT_ID', 'soil_features_parsed']], on='DISTRICT_ID', how='left')
    val_df = val_df.merge(soil_data[['DISTRICT_ID', 'soil_features_parsed']], on='DISTRICT_ID', how='left')
    test_df = test_df.merge(soil_data[['DISTRICT_ID', 'soil_features_parsed']], on='DISTRICT_ID', how='left')

    # Check for missing soil data
    train_missing = train_df['soil_features_parsed'].isna().sum()
    val_missing = val_df['soil_features_parsed'].isna().sum()
    test_missing = test_df['soil_features_parsed'].isna().sum()
    print(f"  Missing soil data - Train: {train_missing}, Val: {val_missing}, Test: {test_missing}")

    # Apply z-score normalization to soil features
    print("  Applying z-score normalization to soil features...")

    # Get unique districts from training data
    train_districts = set(train_df['DISTRICT_ID'].unique())

    # Filter soil data to only include districts that appear in training set
    train_soil_data = soil_data[soil_data['DISTRICT_ID'].isin(train_districts)]

    # Get unique soil features from training districts only
    train_soil_features = np.stack(train_soil_data['soil_features_parsed'].dropna())
    if len(train_soil_features) == 0:
        raise ValueError("len(train_soil_features) == 0")

    # train_soil_features shape: (n_training_districts, 8_features, 6_depths)
    # We want to normalize each of the 8 features across training districts and depths

    # Reshape to (n_training_districts * 6_depths, 8_features) for feature-wise normalization
    train_soil_reshaped = train_soil_features.reshape(-1, train_soil_features.shape[1])

    # Create and fit the soil features scaler
    soil_scaler = StandardScaler()
    soil_scaler.fit(train_soil_reshaped)

    def apply_soil_scaling(df, scaler):
        """Apply soil features scaler to the dataframe."""
        if df.empty:
            return df
        df_scaled = df.copy()
        
        # Process each row
        scaled_soil_features = []
        for soil_features in df_scaled['soil_features_parsed']:
            # soil_features shape: (8_features, 6_depths)
            # Reshape to (6_depths, 8_features) for scaling
            soil_reshaped = soil_features.T
            # Apply scaling
            scaled_soil = scaler.transform(soil_reshaped)
            # Reshape back
            scaled_soil_features.append(scaled_soil.T.tolist())
        
        df_scaled['soil_features'] = scaled_soil_features
        return df_scaled

    # Apply soil scaling to all datasets
    train_df = apply_soil_scaling(train_df, soil_scaler)
    val_df = apply_soil_scaling(val_df, soil_scaler)
    test_df = apply_soil_scaling(test_df, soil_scaler)

    # Remove the temporary parsed column
    train_df = train_df.drop('soil_features_parsed', axis=1)
    val_df = val_df.drop('soil_features_parsed', axis=1)
    test_df = test_df.drop('soil_features_parsed', axis=1)

    # Save the soil features scaler
    soil_scaler_path = os.path.join(OUTPUT_DIR, 'soil_scaler.joblib')
    joblib.dump(soil_scaler, soil_scaler_path)
    print(f"  Soil features scaler saved to '{soil_scaler_path}'")
else:
    print("Soil feature processing skipped since soil features path is not provided")

##### 7. Write the datasets with metadata

feature_names = feature_cols
feature_names += ['past_yield']  # lagged yield feature
if soil_features_available:
    feature_names += ['soil features']

metadata = {
    'desc.': DATASET_DESC,
    'feature_names': feature_names,
    'split_details': {
        'train_years': [int(y) for y in train_years],
        'val_years': [int(y) for y in val_years],
        'test_years': [int(y) for y in test_years]
    },
}

metadata_path = os.path.join(OUTPUT_DIR, 'metadata.json')

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)
    
print(f"  Metadata saved to '{metadata_path}'")

# Write the csv files
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print(f"\nDatasets saved successfully to '{OUTPUT_DIR}'.")
print(f"  Training set shape:   {train_df.shape}")
print(f"  Validation set shape: {val_df.shape}")
print(f"  Test set shape:       {test_df.shape}")