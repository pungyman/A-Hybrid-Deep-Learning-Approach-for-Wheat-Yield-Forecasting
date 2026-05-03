import os
import pandas as pd

def extract_years(columns):
    """Extract all unique year substrings from column names."""
    years = set()
    for col in columns:
        if '-' in col and any(metric in col for metric in ['Area', 'Production', 'Yield']):
            try:
                year = col.split('-')[-2] + '-' + col.split('-')[-1]
                years.add(year)
            except:
                continue
    return sorted(years)

def reshape_wide_to_long(df, filename):
    """Reshape one CSV from wide to long format."""
    id_vars = ['State', 'District', 'Crop', 'Season']
    value_vars = [col for col in df.columns if col not in id_vars]
    years = set()
    long_rows = []

    for col in value_vars:
        for metric in ['Area', 'Production', 'Yield']:
            if col.startswith(metric):
                year = col.replace(metric + '-', '')
                years.add(year)
                break

    years = sorted(years)
    print(f"  Found years in {filename}: {years}")

    for year in years:
        try:
            temp_df = df[id_vars].copy()
            temp_df['Year'] = year
            temp_df['Area'] = df.get(f'Area-{year}', pd.NA)
            temp_df['Production'] = df.get(f'Production-{year}', pd.NA)
            temp_df['Yield'] = df.get(f'Yield-{year}', pd.NA)
            long_rows.append(temp_df)
        except Exception as e:
            print(f"  ⚠️ Error processing year {year} in {filename}: {e}")

    return pd.concat(long_rows, ignore_index=True)

def merge_all_csvs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv') and f != 'merged_output.csv']
    print(f"🔍 Found {len(csv_files)} CSV files to merge...")

    all_data = []

    for file in csv_files:
        try:
            print(f"\n📄 Processing: {file}")
            df = pd.read_csv(file)
            reshaped_df = reshape_wide_to_long(df, file)
            all_data.append(reshaped_df)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

    if not all_data:
        print("⚠️ No valid data found. Exiting.")
        return

    merged_df = pd.concat(all_data, ignore_index=True)

    # Sort by State, then by Year within each state
    merged_df = merged_df.sort_values(by=['State', 'Year'])

    output_path = os.path.join(current_dir, 'merged_output.csv')
    merged_df.to_csv(output_path, index=False)
    print(f"\n✅ Merged CSV written to: {output_path}")
    print(f"📊 Total rows in merged file: {len(merged_df)}")

if __name__ == '__main__':
    merge_all_csvs()
