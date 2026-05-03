"""
Soil CNN ablation analysis: compare full model (CNN-BiLSTM-Attn) vs no-soil (BiLSTM-Attn) per state.
Generates per-state RMSE difference and figure showing which states benefit most from the soil branch.
Expects two results CSVs with same columns (district_id, sowing_year, actual_yield, predicted_yield, residual, absolute_error).
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from .plot_style import apply_style
except ImportError:
    try:
        from plot_style import apply_style
    except ImportError:
        def apply_style(): pass
apply_style()


def extract_state(district_id: str) -> str:
    if pd.isna(district_id):
        return ''
    parts = str(district_id).split('_', 1)
    return parts[0] if len(parts) > 1 else str(district_id)


def run_ablation_analysis(
    full_results_path: str,
    no_soil_results_path: str,
    output_dir: str,
) -> None:
    """
    full_results_path: CSV from CNN-BiLSTM-Attn (with soil).
    no_soil_results_path: CSV from BiLSTM-Attn (no soil).
    """
    full = pd.read_csv(full_results_path)
    no_soil = pd.read_csv(no_soil_results_path)
    for col in ['district_id', 'sowing_year', 'actual_yield', 'predicted_yield', 'absolute_error']:
        if col not in full.columns or col not in no_soil.columns:
            raise ValueError(f'Both CSVs must have column: {col}')
    # Merge on district_id + sowing_year so we compare same samples
    full = full.copy()
    full['state'] = full['district_id'].apply(extract_state)
    no_soil = no_soil.rename(columns={'absolute_error': 'absolute_error_no_soil', 'predicted_yield': 'predicted_yield_no_soil'})
    merged = full.merge(
        no_soil[['district_id', 'sowing_year', 'absolute_error_no_soil', 'predicted_yield_no_soil']],
        on=['district_id', 'sowing_year'],
        how='inner',
    )
    if len(merged) == 0:
        raise ValueError('No matching district_id + sowing_year between the two result files. Check they are from the same split.')
    # Per-state: RMSE with soil vs without; improvement = RMSE_no_soil - RMSE_full (positive = soil helps)
    rows = []
    for state in merged['state'].dropna().unique():
        if state == '':
            continue
        sub = merged[merged['state'] == state]
        rmse_full = np.sqrt((sub['absolute_error'] ** 2).mean())
        rmse_no_soil = np.sqrt((sub['absolute_error_no_soil'] ** 2).mean())
        improvement = rmse_no_soil - rmse_full  # positive means soil branch helps
        rows.append({
            'state': state,
            'n': len(sub),
            'RMSE_full_model': rmse_full,
            'RMSE_no_soil': rmse_no_soil,
            'RMSE_improvement_with_soil': improvement,
        })
    table = pd.DataFrame(rows).sort_values('RMSE_improvement_with_soil', ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, 'ablation_per_state_rmse.csv'), index=False)
    with open(os.path.join(output_dir, 'ablation_per_state_rmse.tex'), 'w') as f:
        f.write(table.to_latex(index=False, float_format='%.2f'))
    print('Per-state RMSE: full model vs no-soil (positive improvement = soil helps):')
    print(table.to_string())
    # Bar chart: RMSE improvement by state
    if len(table):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(table))
        ax.barh(x, table['RMSE_improvement_with_soil'], color=['C2' if v >= 0 else 'C3' for v in table['RMSE_improvement_with_soil']], alpha=0.8)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(table['state'])
        ax.set_xlabel('RMSE improvement with soil branch (kg/ha)')
        ax.set_title('Per-state benefit of soil CNN branch (positive = soil reduces error)')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_soil_benefit_by_state.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved ablation_soil_benefit_by_state.png to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Compare full model vs no-soil ablation per state.')
    parser.add_argument('full_results', help='Results CSV from full model (CNN-BiLSTM-Attn)')
    parser.add_argument('no_soil_results', help='Results CSV from BiLSTM-Attn (no soil)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: dir of full_results)')
    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.full_results).parent)
    run_ablation_analysis(args.full_results, args.no_soil_results, output_dir)


if __name__ == '__main__':
    main()
