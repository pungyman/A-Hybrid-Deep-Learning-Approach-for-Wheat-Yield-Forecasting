import pandas as pd

WEATHER_FEATURES_FILEPATH = '../../data/weather_data/weather_features_7_states.csv'
REMOTE_SENSING_FEATURES_FILEPATH = '../../data/remote_sensing_data/daily_ndvi_evi_data_7_states.csv'

weather = pd.read_csv(WEATHER_FEATURES_FILEPATH)
remote_sensing = pd.read_csv(REMOTE_SENSING_FEATURES_FILEPATH)

weather = weather.rename(columns={'Date': 'DATE'})
remote_sensing = remote_sensing.rename(columns={'date': 'DATE'})

# Merge weather and remote_sensing on DISTRICT_ID, DATE
daily_X = pd.merge(weather, remote_sensing, on=['DISTRICT_ID', 'DATE'], how='inner')

# Convert DATE to datetime if not already
daily_X['DATE'] = pd.to_datetime(daily_X['DATE'])

# Extract YEAR and MONTH
daily_X['YEAR'] = daily_X['DATE'].dt.year
daily_X['MONTH'] = daily_X['DATE'].dt.month

# Drop DATE column
daily_X = daily_X.drop(columns=['DATE'])

# Group by DISTRICT_ID, YEAR, MONTH and aggregate by mean
monthly_X = daily_X.groupby(['DISTRICT_ID', 'YEAR', 'MONTH'], as_index=False).mean()

# Reorder columns: DISTRICT_ID, YEAR, MONTH, then the rest
cols = ['DISTRICT_ID', 'YEAR', 'MONTH'] + [col for col in monthly_X.columns if col not in ['DISTRICT_ID', 'YEAR', 'MONTH']]
monthly_X = monthly_X[cols]

# Subset to only include rows from June 2000 to May 2024, inclusive
mask = (
    ((monthly_X['YEAR'] > 2000) | ((monthly_X['YEAR'] == 2000) & (monthly_X['MONTH'] >= 6))) &
    ((monthly_X['YEAR'] < 2024) | ((monthly_X['YEAR'] == 2024) & (monthly_X['MONTH'] <= 5)))
)
monthly_X = monthly_X[mask].reset_index(drop=True)

monthly_X.to_csv('../../data/processed/monthly_features_7_states.csv', index=False)