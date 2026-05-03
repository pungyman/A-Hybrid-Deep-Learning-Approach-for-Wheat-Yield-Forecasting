import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def analyze_weather_data(csv_file_path):
    """
    Perform detailed analysis of the NASA POWER weather data.
    
    Args:
        csv_file_path (str): Path to the NASA POWER CSV file
    
    Returns:
        dict: Analysis results
    """
    print("Reading CSV file for detailed analysis...")
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get unique districts
    districts = df[['State', 'District']].drop_duplicates()
    print(f"\nTotal unique districts: {len(districts)}")
    
    # Overall date range analysis
    overall_min_date = df['Date'].min()
    overall_max_date = df['Date'].max()
    print(f"\nOverall date range: {overall_min_date.date()} to {overall_max_date.date()}")
    
    # Analysis results
    analysis = {
        'overall_stats': {
            'total_rows': len(df),
            'total_districts': len(districts),
            'date_range': (overall_min_date, overall_max_date),
            'weather_columns': [col for col in df.columns if col not in ['State', 'District', 'Date']]
        },
        'district_details': [],
        'state_summary': {},
        'data_quality': {}
    }
    
    # Analyze each district
    print("\nAnalyzing each district...")
    
    for idx, (_, row) in enumerate(districts.iterrows()):
        state = row['State']
        district = row['District']
        
        # Filter data for this district
        district_data = df[(df['State'] == state) & (df['District'] == district)]
        
        # Get date range for this district
        min_date = district_data['Date'].min()
        max_date = district_data['Date'].max()
        actual_days = len(district_data)
        
        # Calculate expected days (2000-01-01 to 2024-12-31)
        start_date = pd.to_datetime('2000-01-01')
        end_date = pd.to_datetime('2024-12-31')
        expected_days = (end_date - start_date).days + 1
        
        # Check for missing values in weather columns
        weather_columns = [col for col in df.columns if col not in ['State', 'District', 'Date']]
        missing_values = district_data[weather_columns].isnull().sum().to_dict()
        
        district_info = {
            'state': state,
            'district': district,
            'min_date': min_date,
            'max_date': max_date,
            'actual_days': actual_days,
            'expected_days': expected_days,
            'missing_days': max(0, expected_days - actual_days),
            'missing_values': missing_values,
            'is_complete': actual_days >= expected_days and min_date <= start_date and max_date >= end_date
        }
        
        analysis['district_details'].append(district_info)
        
        # Update state summary
        if state not in analysis['state_summary']:
            analysis['state_summary'][state] = {
                'districts': 0,
                'complete_districts': 0,
                'total_days': 0,
                'missing_days': 0
            }
        
        analysis['state_summary'][state]['districts'] += 1
        analysis['state_summary'][state]['total_days'] += actual_days
        analysis['state_summary'][state]['missing_days'] += district_info['missing_days']
        
        if district_info['is_complete']:
            analysis['state_summary'][state]['complete_districts'] += 1
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(districts)} districts...")
    
    # Data quality analysis
    print("\nPerforming data quality analysis...")
    
    # Check for missing values in the entire dataset
    missing_data = df.isnull().sum()
    analysis['data_quality']['missing_values'] = missing_data.to_dict()
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    analysis['data_quality']['duplicate_records'] = duplicates
    
    # Check for outliers in weather data (basic check)
    weather_columns = [col for col in df.columns if col not in ['State', 'District', 'Date']]
    outlier_summary = {}
    
    for col in weather_columns:
        if col in ['T2M_MAX', 'T2M_MIN', 'T2M']:  # Temperature columns
            # Check for unrealistic temperature values (e.g., > 60°C or < -50°C)
            outliers = df[(df[col] > 60) | (df[col] < -50)][col].count()
            outlier_summary[col] = outliers
        elif col == 'PRECTOTCORR':  # Precipitation
            # Check for negative precipitation
            outliers = df[df[col] < 0][col].count()
            outlier_summary[col] = outliers
        elif col == 'RH2M':  # Relative humidity
            # Check for values outside 0-100 range
            outliers = df[(df[col] < 0) | (df[col] > 100)][col].count()
            outlier_summary[col] = outliers
    
    analysis['data_quality']['outliers'] = outlier_summary
    
    return analysis

def print_detailed_analysis(analysis):
    """Print detailed analysis results."""
    print("\n" + "="*80)
    print("DETAILED WEATHER DATA ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"Total rows: {analysis['overall_stats']['total_rows']:,}")
    print(f"Total districts: {analysis['overall_stats']['total_districts']}")
    print(f"Date range: {analysis['overall_stats']['date_range'][0].date()} to {analysis['overall_stats']['date_range'][1].date()}")
    print(f"Weather parameters: {', '.join(analysis['overall_stats']['weather_columns'])}")
    
    # State summary
    print(f"\nSTATE SUMMARY:")
    print("-" * 80)
    print(f"{'State':<20} {'Districts':<12} {'Complete':<12} {'Total Days':<12} {'Missing Days':<12}")
    print("-" * 80)
    
    for state, stats in analysis['state_summary'].items():
        complete_pct = (stats['complete_districts'] / stats['districts']) * 100
        print(f"{state:<20} {stats['districts']:<12} {stats['complete_districts']:<12} "
              f"{stats['total_days']:<12} {stats['missing_days']:<12}")
    
    # Data quality
    print(f"\nDATA QUALITY ANALYSIS:")
    print("-" * 80)
    
    print(f"Missing values per column:")
    for col, missing_count in analysis['data_quality']['missing_values'].items():
        if missing_count > 0:
            percentage = (missing_count / analysis['overall_stats']['total_rows']) * 100
            print(f"  {col}: {missing_count:,} ({percentage:.2f}%)")
        else:
            print(f"  {col}: No missing values")
    
    print(f"\nDuplicate records: {analysis['data_quality']['duplicate_records']}")
    
    print(f"\nPotential outliers:")
    for col, outlier_count in analysis['data_quality']['outliers'].items():
        if outlier_count > 0:
            print(f"  {col}: {outlier_count:,} potential outliers")
        else:
            print(f"  {col}: No obvious outliers detected")
    
    # District completeness summary
    complete_districts = [d for d in analysis['district_details'] if d['is_complete']]
    incomplete_districts = [d for d in analysis['district_details'] if not d['is_complete']]
    
    print(f"\nDISTRICT COMPLETENESS:")
    print(f"Complete districts: {len(complete_districts)} ({len(complete_districts)/len(analysis['district_details'])*100:.1f}%)")
    print(f"Incomplete districts: {len(incomplete_districts)} ({len(incomplete_districts)/len(analysis['district_details'])*100:.1f}%)")
    
    if incomplete_districts:
        print(f"\nINCOMPLETE DISTRICTS:")
        for district in incomplete_districts[:10]:  # Show first 10
            print(f"  {district['state']}, {district['district']}: {district['missing_days']} missing days")
        if len(incomplete_districts) > 10:
            print(f"  ... and {len(incomplete_districts) - 10} more")

def save_analysis_results(analysis, output_file):
    """Save analysis results to a file."""
    with open(output_file, 'w') as f:
        f.write("DETAILED WEATHER DATA ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Overall stats
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total rows: {analysis['overall_stats']['total_rows']:,}\n")
        f.write(f"Total districts: {analysis['overall_stats']['total_districts']}\n")
        f.write(f"Date range: {analysis['overall_stats']['date_range'][0].date()} to {analysis['overall_stats']['date_range'][1].date()}\n")
        f.write(f"Weather parameters: {', '.join(analysis['overall_stats']['weather_columns'])}\n\n")
        
        # State summary
        f.write("STATE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        for state, stats in analysis['state_summary'].items():
            complete_pct = (stats['complete_districts'] / stats['districts']) * 100
            f.write(f"{state}: {stats['districts']} districts, {stats['complete_districts']} complete ({complete_pct:.1f}%)\n")
        f.write("\n")
        
        # Data quality
        f.write("DATA QUALITY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Duplicate records: {analysis['data_quality']['duplicate_records']}\n")
        f.write("Missing values:\n")
        for col, missing_count in analysis['data_quality']['missing_values'].items():
            if missing_count > 0:
                f.write(f"  {col}: {missing_count:,}\n")
        f.write("\n")
        
        # District details
        f.write("DISTRICT DETAILS:\n")
        f.write("-" * 30 + "\n")
        for district in analysis['district_details']:
            status = "COMPLETE" if district['is_complete'] else "INCOMPLETE"
            f.write(f"{district['state']}, {district['district']}: {status} "
                   f"({district['actual_days']}/{district['expected_days']} days)\n")

def main():
    """Main function to run the detailed analysis."""
    csv_file = "NASA_POWER_All_Districts_20000101_20241231_20250706_123558.csv"
    
    try:
        print("Starting detailed weather data analysis...")
        analysis = analyze_weather_data(csv_file)
        print_detailed_analysis(analysis)
        
        # Save results
        output_file = "detailed_weather_analysis_report.txt"
        save_analysis_results(analysis, output_file)
        print(f"\nDetailed analysis report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 