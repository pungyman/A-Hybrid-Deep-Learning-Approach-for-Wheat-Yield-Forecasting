import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def detect_statistical_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers using statistical methods.
    
    Args:
        df: DataFrame
        column: Column name to analyze
        method: 'iqr' (Interquartile Range) or 'zscore'
        threshold: Multiplier for IQR or Z-score threshold
    
    Returns:
        dict: Outlier statistics
    """
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers = data[z_scores > threshold]
    
    return {
        'total_values': len(data),
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(data)) * 100,
        'outliers': outliers,
        'min_outlier': outliers.min() if len(outliers) > 0 else None,
        'max_outlier': outliers.max() if len(outliers) > 0 else None,
        'method': method,
        'threshold': threshold
    }

def analyze_weather_outliers(csv_file_path):
    """
    Perform comprehensive outlier analysis on weather data.
    
    Args:
        csv_file_path (str): Path to the NASA POWER CSV file
    
    Returns:
        dict: Outlier analysis results
    """
    print("Reading CSV file for outlier analysis...")
    df = pd.read_csv(csv_file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    weather_columns = [col for col in df.columns if col not in ['State', 'District', 'Date']]
    
    print(f"Analyzing outliers for {len(weather_columns)} weather parameters...")
    
    results = {
        'physical_constraints': {},
        'statistical_outliers': {},
        'seasonal_analysis': {},
        'geographic_analysis': {}
    }
    
    # 1. Physical Constraints (Obvious Outliers)
    print("\n1. Checking physical constraints...")
    for col in weather_columns:
        if col in ['T2M_MAX', 'T2M_MIN', 'T2M']:
            # Temperature constraints
            outliers = df[(df[col] > 60) | (df[col] < -50)][col]
            results['physical_constraints'][col] = {
                'constraint': 'Temperature > 60°C or < -50°C',
                'outlier_count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
        elif col == 'PRECTOTCORR':
            # Precipitation constraints
            outliers = df[df[col] < 0][col]
            results['physical_constraints'][col] = {
                'constraint': 'Negative precipitation',
                'outlier_count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
        elif col == 'RH2M':
            # Relative humidity constraints
            outliers = df[(df[col] < 0) | (df[col] > 100)][col]
            results['physical_constraints'][col] = {
                'constraint': 'RH < 0% or > 100%',
                'outlier_count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100
            }
    
    # 2. Statistical Outliers
    print("2. Detecting statistical outliers...")
    for col in weather_columns:
        print(f"  Analyzing {col}...")
        
        # IQR method
        iqr_results = detect_statistical_outliers(df, col, method='iqr', threshold=1.5)
        
        # Z-score method
        zscore_results = detect_statistical_outliers(df, col, method='zscore', threshold=3)
        
        results['statistical_outliers'][col] = {
            'iqr': iqr_results,
            'zscore': zscore_results
        }
    
    # 3. Seasonal Analysis
    print("3. Analyzing seasonal patterns...")
    df['Month'] = df['Date'].dt.month
    df['Season'] = df['Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    for col in weather_columns:
        seasonal_stats = df.groupby('Season')[col].agg(['mean', 'std', 'min', 'max']).round(2)
        results['seasonal_analysis'][col] = seasonal_stats.to_dict()
    
    # 4. Geographic Analysis
    print("4. Analyzing geographic patterns...")
    for col in weather_columns:
        state_stats = df.groupby('State')[col].agg(['mean', 'std', 'min', 'max']).round(2)
        results['geographic_analysis'][col] = state_stats.to_dict()
    
    return results

def print_outlier_analysis(results):
    """Print comprehensive outlier analysis results."""
    print("\n" + "="*80)
    print("COMPREHENSIVE OUTLIER ANALYSIS")
    print("="*80)
    
    # Physical Constraints
    print("\n1. PHYSICAL CONSTRAINT VIOLATIONS:")
    print("-" * 50)
    for col, info in results['physical_constraints'].items():
        if info['outlier_count'] > 0:
            print(f"  {col}: {info['outlier_count']:,} violations ({info['percentage']:.3f}%)")
            print(f"    Constraint: {info['constraint']}")
        else:
            print(f"  {col}: No violations detected")
    
    # Statistical Outliers
    print("\n2. STATISTICAL OUTLIERS:")
    print("-" * 50)
    for col, methods in results['statistical_outliers'].items():
        print(f"\n  {col}:")
        
        iqr_info = methods['iqr']
        zscore_info = methods['zscore']
        
        print(f"    IQR method (1.5x): {iqr_info['outlier_count']:,} outliers ({iqr_info['outlier_percentage']:.2f}%)")
        print(f"    Z-score method (3σ): {zscore_info['outlier_count']:,} outliers ({zscore_info['outlier_percentage']:.2f}%)")
        
        if iqr_info['outlier_count'] > 0:
            print(f"    IQR outlier range: {iqr_info['min_outlier']:.2f} to {iqr_info['max_outlier']:.2f}")
        if zscore_info['outlier_count'] > 0:
            print(f"    Z-score outlier range: {zscore_info['min_outlier']:.2f} to {zscore_info['max_outlier']:.2f}")
    
    # Seasonal Analysis
    print("\n3. SEASONAL PATTERNS (Summary):")
    print("-" * 50)
    for col, seasonal_data in results['seasonal_analysis'].items():
        print(f"\n  {col}:")
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            if season in seasonal_data['mean']:
                mean_val = seasonal_data['mean'][season]
                std_val = seasonal_data['std'][season]
                min_val = seasonal_data['min'][season]
                max_val = seasonal_data['max'][season]
                print(f"    {season}: Mean={mean_val:.2f}, Std={std_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]")
    
    # Geographic Analysis
    print("\n4. GEOGRAPHIC PATTERNS (Summary):")
    print("-" * 50)
    for col, state_data in results['geographic_analysis'].items():
        print(f"\n  {col}:")
        for state in ['HARYANA', 'MADHYA PRADESH', 'PUNJAB', 'RAJASTHAN', 'UTTAR PRADESH']:
            if state in state_data['mean']:
                mean_val = state_data['mean'][state]
                std_val = state_data['std'][state]
                min_val = state_data['min'][state]
                max_val = state_data['max'][state]
                print(f"    {state}: Mean={mean_val:.2f}, Std={std_val:.2f}, Range=[{min_val:.2f}, {max_val:.2f}]")

def save_outlier_report(results, output_file):
    """Save outlier analysis to a file."""
    with open(output_file, 'w') as f:
        f.write("COMPREHENSIVE OUTLIER ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Physical constraints
        f.write("1. PHYSICAL CONSTRAINT VIOLATIONS:\n")
        f.write("-" * 30 + "\n")
        for col, info in results['physical_constraints'].items():
            f.write(f"{col}: {info['outlier_count']} violations ({info['percentage']:.3f}%)\n")
            f.write(f"  Constraint: {info['constraint']}\n\n")
        
        # Statistical outliers
        f.write("2. STATISTICAL OUTLIERS:\n")
        f.write("-" * 30 + "\n")
        for col, methods in results['statistical_outliers'].items():
            iqr_info = methods['iqr']
            zscore_info = methods['zscore']
            f.write(f"{col}:\n")
            f.write(f"  IQR outliers: {iqr_info['outlier_count']} ({iqr_info['outlier_percentage']:.2f}%)\n")
            f.write(f"  Z-score outliers: {zscore_info['outlier_count']} ({zscore_info['outlier_percentage']:.2f}%)\n\n")

def main():
    """Main function to run the outlier analysis."""
    csv_file = "NASA_POWER_All_Districts_20000101_20241231_20250706_123558.csv"
    
    try:
        print("Starting comprehensive outlier analysis...")
        results = analyze_weather_outliers(csv_file)
        print_outlier_analysis(results)
        
        # Save results
        output_file = "outlier_analysis_report.txt"
        save_outlier_report(results, output_file)
        print(f"\nOutlier analysis report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during outlier analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 