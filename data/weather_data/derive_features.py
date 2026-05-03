import pandas as pd
import numpy as np

def calculate_svp(temp_c):
    """Calculates saturation vapor pressure (SVP) in kPa."""
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))

def derive_weather_features(input_path, output_path):
    """
    Reads weather data, calculates vapor pressure deficit (VPD) features,
    and saves the new data to a CSV file.
    """
    df = pd.read_csv(input_path)

    # Assuming temperature is in Celsius and humidity in percentage
    # T2M_MAX: Maximum Temperature at 2 Meters (C)
    # T2M_MIN: Minimum Temperature at 2 Meters (C)
    # RH2M: Relative Humidity at 2 Meters (%)
    
    t_max_col = 'T2M_MAX'
    t_min_col = 'T2M_MIN'
    rh_col = 'RH2M'

    # Calculate SVP for max and min temperatures
    svp_max = calculate_svp(df[t_max_col])
    svp_min = calculate_svp(df[t_min_col])

    # Calculate actual vapor pressure (AVP)
    # Using average SVP for AVP calculation based on a single RH value
    svp_avg = (svp_max + svp_min) / 2
    avp = svp_avg * (df[rh_col] / 100)

    # Calculate VPD for max and min temperatures
    df['max_vpd'] = svp_max - avp
    df['min_vpd'] = svp_min - avp

    df.to_csv(output_path, index=False)
    print(f"Weather features saved to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "weather_data_guj_bih.csv"
    OUTPUT_FILE = "weather_features_guj_bih.csv"
    derive_weather_features(INPUT_FILE, OUTPUT_FILE)
