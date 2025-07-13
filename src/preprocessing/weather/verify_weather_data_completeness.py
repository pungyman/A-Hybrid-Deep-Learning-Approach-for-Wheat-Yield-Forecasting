import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def verify_weather_data_completeness(csv_file_path):
    """
    Verify that each district has complete weather data from 2000 to 2024.
    
    Args:
        csv_file_path (str): Path to the NASA POWER CSV file
    
    Returns:
        dict: Summary of verification results
    """
    print("Reading CSV file...")
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get unique districts
    districts = df[['State', 'District']].drop_duplicates()
    print(f"\nTotal unique districts: {len(districts)}")
    
    # Expected date range
    start_date = pd.to_datetime('2000-01-01')
    end_date = pd.to_datetime('2024-12-31')
    expected_days = (end_date - start_date).days + 1
    print(f"Expected days per district: {expected_days} (from {start_date.date()} to {end_date.date()})")
    
    # Results storage
    results = {
        'complete_districts': [],
        'incomplete_districts': [],
        'missing_data_summary': {}
    }
    
    print("\nVerifying data completeness for each district...")
    
    for idx, (_, row) in enumerate(districts.iterrows()):
        state = row['State']
        district = row['District']
        
        # Filter data for this district
        district_data = df[(df['State'] == state) & (df['District'] == district)]
        
        # Get date range for this district
        min_date = district_data['Date'].min()
        max_date = district_data['Date'].max()
        actual_days = len(district_data)
        
        # Check if data covers the full period
        covers_full_period = (min_date <= start_date) and (max_date >= end_date)
        
        # Check if we have the expected number of days
        has_expected_days = actual_days >= expected_days
        
        district_info = {
            'state': state,
            'district': district,
            'min_date': min_date,
            'max_date': max_date,
            'actual_days': actual_days,
            'expected_days': expected_days,
            'covers_full_period': covers_full_period,
            'has_expected_days': has_expected_days,
            'missing_days': max(0, expected_days - actual_days)
        }
        
        if covers_full_period and has_expected_days:
            results['complete_districts'].append(district_info)
        else:
            results['incomplete_districts'].append(district_info)
            
            # Find missing dates
            if not covers_full_period:
                expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                actual_dates = set(district_data['Date'].dt.date)
                missing_dates = [d.date() for d in expected_dates if d.date() not in actual_dates]
                
                if len(missing_dates) > 0:
                    district_key = f"{state}_{district}"
                    results['missing_data_summary'][district_key] = {
                        'missing_dates': missing_dates,
                        'missing_count': len(missing_dates)
                    }
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(districts)} districts...")
    
    return results

def print_verification_results(results):
    """Print detailed verification results."""
    print("\n" + "="*80)
    print("WEATHER DATA COMPLETENESS VERIFICATION RESULTS")
    print("="*80)
    
    total_districts = len(results['complete_districts']) + len(results['incomplete_districts'])
    complete_count = len(results['complete_districts'])
    incomplete_count = len(results['incomplete_districts'])
    
    print(f"\nSUMMARY:")
    print(f"Total districts: {total_districts}")
    print(f"Complete districts: {complete_count} ({complete_count/total_districts*100:.1f}%)")
    print(f"Incomplete districts: {incomplete_count} ({incomplete_count/total_districts*100:.1f}%)")
    
    if incomplete_count > 0:
        print(f"\nINCOMPLETE DISTRICTS ({incomplete_count}):")
        print("-" * 80)
        
        for district_info in results['incomplete_districts']:
            print(f"State: {district_info['state']}")
            print(f"District: {district_info['district']}")
            print(f"Date range: {district_info['min_date'].date()} to {district_info['max_date'].date()}")
            print(f"Actual days: {district_info['actual_days']}")
            print(f"Expected days: {district_info['expected_days']}")
            print(f"Missing days: {district_info['missing_days']}")
            print(f"Covers full period: {district_info['covers_full_period']}")
            print(f"Has expected days: {district_info['has_expected_days']}")
            print("-" * 40)
    
    if results['missing_data_summary']:
        print(f"\nDETAILED MISSING DATA SUMMARY:")
        print("-" * 80)
        
        for district_key, missing_info in results['missing_data_summary'].items():
            print(f"\n{district_key}:")
            print(f"  Missing dates: {missing_info['missing_count']}")
            if missing_info['missing_count'] <= 10:
                print(f"  Missing dates: {missing_info['missing_dates']}")
            else:
                print(f"  First 5 missing dates: {missing_info['missing_dates'][:5]}")
                print(f"  Last 5 missing dates: {missing_info['missing_dates'][-5:]}")

def main():
    """Main function to run the verification."""
    csv_file = "NASA_POWER_All_Districts_20000101_20241231_20250706_123558.csv"
    
    try:
        print("Starting weather data completeness verification...")
        results = verify_weather_data_completeness(csv_file)
        print_verification_results(results)
        
        # Save results to a summary file
        summary_file = "weather_data_verification_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("WEATHER DATA COMPLETENESS VERIFICATION SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            total_districts = len(results['complete_districts']) + len(results['incomplete_districts'])
            complete_count = len(results['complete_districts'])
            incomplete_count = len(results['incomplete_districts'])
            
            f.write(f"Total districts: {total_districts}\n")
            f.write(f"Complete districts: {complete_count} ({complete_count/total_districts*100:.1f}%)\n")
            f.write(f"Incomplete districts: {incomplete_count} ({incomplete_count/total_districts*100:.1f}%)\n\n")
            
            if incomplete_count > 0:
                f.write("INCOMPLETE DISTRICTS:\n")
                f.write("-" * 30 + "\n")
                for district_info in results['incomplete_districts']:
                    f.write(f"{district_info['state']}, {district_info['district']}: "
                           f"{district_info['missing_days']} missing days\n")
        
        print(f"\nVerification summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        raise

if __name__ == "__main__":
    main() 