"""
Generate a publication-ready comparison table of hybrid DL architectures for crop yield prediction.
Populate the LIT_TABLE list with entries from the literature; the script outputs CSV and LaTeX.
Columns: Study, Region, Crop, Architecture, Data Modalities, Spatial Scale, Best R2/RMSE, Gap vs This Work.
"""
import os
import argparse
import pandas as pd
from pathlib import Path

# Populate this list from literature. Example structure; add rows as needed.
LIT_TABLE = [
    {
        'Study': 'Khaki et al. (2020)',
        'Region': 'US Corn Belt',
        'Crop': 'Corn, Soybean',
        'Architecture': 'CNN-RNN',
        'Data Modalities': 'Weather, RS, soil',
        'Spatial Scale': 'County',
        'Best R2 or RMSE': 'R2 0.77',
        'Gap vs This Work': 'No attention; no India',
    },
    {
        'Study': 'Fan et al. (2022)',
        'Region': 'US',
        'Crop': 'Corn, Soybean',
        'Architecture': 'GNN-RNN',
        'Data Modalities': 'Weather, RS',
        'Spatial Scale': 'County',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'No soil branch; no India',
    },
    {
        'Study': 'Wolanin et al. (2020)',
        'Region': 'Indian Wheat Belt',
        'Crop': 'Wheat',
        'Architecture': 'CNN (1D)',
        'Data Modalities': 'RS, weather',
        'Spatial Scale': 'District',
        'Best R2 or RMSE': 'R2 0.77',
        'Gap vs This Work': 'No soil; no temporal attention; smaller scale',
    },
    {
        'Study': 'Bali and Singla (2021)',
        'Region': 'Punjab, India',
        'Crop': 'Wheat',
        'Architecture': 'LSTM',
        'Data Modalities': 'Weather, RS',
        'Spatial Scale': 'District',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'No soil; no hybrid CNN-RNN; single state',
    },
    {
        'Study': 'Saini et al. (2023)',
        'Region': 'India',
        'Crop': 'Sugarcane',
        'Architecture': 'CNN-BiLSTM',
        'Data Modalities': 'Weather, RS',
        'Spatial Scale': 'District',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'Different crop; no soil; no attention',
    },
    {
        'Study': 'Saravanan and Bhagavathiappan (2024)',
        'Region': 'India',
        'Crop': 'Multiple',
        'Architecture': 'Hybrid DL',
        'Data Modalities': 'Weather, RS',
        'Spatial Scale': 'District',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'No dedicated soil branch; no attention',
    },
    {
        'Study': 'Joshi et al. (2025)',
        'Region': 'India',
        'Crop': 'Wheat',
        'Architecture': 'BiLSTM (explainable)',
        'Data Modalities': 'Weather, RS',
        'Spatial Scale': 'District',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'No soil CNN; no multi-modal fusion',
    },
    {
        'Study': 'Jiang et al. (2020)',
        'Region': 'US Corn Belt',
        'Crop': 'Corn',
        'Architecture': 'DL (CNN+LSTM)',
        'Data Modalities': 'Weather, RS, soil',
        'Spatial Scale': 'County',
        'Best R2 or RMSE': 'R2 0.87',
        'Gap vs This Work': 'US only; different scale',
    },
    {
        'Study': 'Ye et al. (2024)',
        'Region': 'China',
        'Crop': 'Winter wheat',
        'Architecture': 'ASTGNN',
        'Data Modalities': 'Multi-source',
        'Spatial Scale': 'Regional',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'China; no India wheat belt',
    },
    {
        'Study': 'Nejad et al. (2023)',
        'Region': '—',
        'Crop': 'Crop yield',
        'Architecture': '3D-CNN, Attention ConvLSTM',
        'Data Modalities': 'Multispectral',
        'Spatial Scale': 'Field',
        'Best R2 or RMSE': '—',
        'Gap vs This Work': 'No soil; no India; different scale',
    },
    {
        'Study': 'This work',
        'Region': 'India (7 states)',
        'Crop': 'Winter wheat',
        'Architecture': 'CNN-BiLSTM-Attention',
        'Data Modalities': 'Weather, RS, soil, lagged yield',
        'Spatial Scale': 'District (275)',
        'Best R2 or RMSE': 'R2 0.81, MAE 265 kg/ha',
        'Gap vs This Work': '—',
    },
]


def generate_table(output_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(LIT_TABLE)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'literature_comparison.csv'), index=False)
    with open(os.path.join(output_dir, 'literature_comparison.tex'), 'w') as f:
        # Use longtable-friendly format if many rows
        f.write(df.to_latex(index=False, escape=False, longtable=len(df) > 20))
    print(f'Saved literature_comparison.csv and .tex to {output_dir}')
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate literature comparison table.')
    parser.add_argument('--output-dir', default='.', help='Output directory for CSV and LaTeX')
    args = parser.parse_args()
    generate_table(args.output_dir)


if __name__ == '__main__':
    main()
