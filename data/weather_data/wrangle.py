import pandas as pd

weather_data = pd.read_csv('NASA_POWER_All_Districts_20000101_20241231_20251007_200434.csv')

weather_data['DISTRICT_ID'] = weather_data['State'] + '_' + weather_data['District']

# Drop the 'State' and 'District' columns
weather_data = weather_data.drop(['State', 'District'], axis=1)

# Reorder columns: 'DISTRICT_ID', 'Date', then the rest
cols = weather_data.columns.tolist()
# Find all columns except 'DISTRICT_ID' and 'Date'
other_cols = [col for col in cols if col not in ['DISTRICT_ID', 'Date']]
# New order
new_order = ['DISTRICT_ID', 'Date'] + other_cols
weather_data = weather_data[new_order]

weather_data.to_csv('weather_data_guj_bih.csv', index=False)