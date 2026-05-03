"""
Extreme year analysis: evaluate model performance in anomalous vs normal years.
Expects a results CSV with columns: district_id, sowing_year, actual_yield, predicted_yield,
residual, absolute_error; optional column 'split' (train/val/test).

To generate a CSV covering all splits, run from src/rnn:
  python run_inference_all_splits.py --config training_config.yaml --model-dir <path_to_saved_models>
Then pass the path to all_splits_results.csv to this script.
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


# Known extreme events: sowing_year -> short label
EXTREME_YEARS_MANUAL = {
    2021: 'March 2022 heat wave',
    2014: '2015 unseasonal hailstorms',
}


def extract_state(district_id: str) -> str:
    """Extract state from district_id (format STATE_DISTRICT, e.g. PUNJAB_AMRITSAR)."""
    if pd.isna(district_id):
        return ''
    parts = str(district_id).split('_', 1)
    return parts[0] if len(parts) > 1 else str(district_id)


def flag_extreme_years_programmatic(df: pd.DataFrame, yield_col: str = 'actual_yield', n_std: float = 1.0) -> pd.Series:
    """
    Flag years where national mean yield deviates from linear trend by more than n_std.
    Returns a boolean Series indexable by sowing_year (True = extreme).
    """
    national = df.groupby('sowing_year')[yield_col].agg(['mean', 'count']).reset_index()
    if len(national) < 5:
        return pd.Series(dtype=bool)
    x = national['sowing_year'].values.astype(float)
    y = national['mean'].values
    coef = np.polyfit(x, y, 1)
    trend = np.polyval(coef, x)
    resid = y - trend
    thresh = np.nanstd(resid) * n_std
    extreme_years = set(national.loc[np.abs(resid) > thresh, 'sowing_year'].tolist())
    return df['sowing_year'].isin(extreme_years)


def run_analysis(results_path: str, output_dir: str, use_programmatic: bool = True, n_std: float = 1.0) -> None:
    df = pd.read_csv(results_path)
    for col in ['district_id', 'sowing_year', 'actual_yield', 'predicted_yield', 'absolute_error']:
        if col not in df.columns:
            raise ValueError(f'Results CSV must contain column: {col}')
    df['state'] = df['district_id'].apply(extract_state)

    # Define extreme years: manual + optional programmatic
    extreme_sowing_years = set(EXTREME_YEARS_MANUAL.keys())
    if use_programmatic:
        is_extreme = flag_extreme_years_programmatic(df, n_std=n_std)
        programmatic_years = set(df.loc[is_extreme, 'sowing_year'].unique())
        extreme_sowing_years = extreme_sowing_years | programmatic_years
    df['year_type'] = df['sowing_year'].map(lambda y: 'extreme' if y in extreme_sowing_years else 'normal')

    os.makedirs(output_dir, exist_ok=True)

    # Metrics overall: normal vs extreme
    def metrics_subset(s):
        return {
            'MAE': s['absolute_error'].mean(),
            'RMSE': np.sqrt((s['absolute_error'] ** 2).mean()),
            'n': len(s),
        }
    normal = df[df['year_type'] == 'normal']
    extreme = df[df['year_type'] == 'extreme']
    table_overall = pd.DataFrame({
        'Normal years': metrics_subset(normal),
        'Extreme years': metrics_subset(extreme),
    }).T
    table_overall.to_csv(os.path.join(output_dir, 'extreme_year_metrics_overall.csv'))
    print('Overall metrics (normal vs extreme):')
    print(table_overall)

    # Per-state metrics for normal vs extreme
    rows = []
    for state in df['state'].dropna().unique():
        if state == '':
            continue
        sub = df[df['state'] == state]
        for yt in ['normal', 'extreme']:
            s = sub[sub['year_type'] == yt]
            if len(s) == 0:
                continue
            rows.append({
                'state': state,
                'year_type': yt,
                'MAE': s['absolute_error'].mean(),
                'RMSE': np.sqrt((s['absolute_error'] ** 2).mean()),
                'n': len(s),
            })
    table_state = pd.DataFrame(rows)
    if len(table_state):
        table_state.to_csv(os.path.join(output_dir, 'extreme_year_metrics_by_state.csv'), index=False)
        pivot_mae = table_state.pivot(index='state', columns='year_type', values='MAE').reset_index()
        pivot_mae.to_csv(os.path.join(output_dir, 'extreme_year_MAE_by_state.csv'), index=False)

    # By split if available
    if 'split' in df.columns:
        rows_split = []
        for split in df['split'].unique():
            for yt in ['normal', 'extreme']:
                s = df[(df['split'] == split) & (df['year_type'] == yt)]
                if len(s) == 0:
                    continue
                rows_split.append({
                    'split': split,
                    'year_type': yt,
                    'MAE': s['absolute_error'].mean(),
                    'RMSE': np.sqrt((s['absolute_error'] ** 2).mean()),
                    'n': len(s),
                })
        table_split = pd.DataFrame(rows_split)
        table_split.to_csv(os.path.join(output_dir, 'extreme_year_metrics_by_split.csv'), index=False)

    # Time series: national mean actual vs predicted by year
    by_year = df.groupby('sowing_year').agg(
        actual_mean=('actual_yield', 'mean'),
        predicted_mean=('predicted_yield', 'mean'),
        n=('district_id', 'count'),
    ).reset_index()
    by_year['extreme'] = by_year['sowing_year'].isin(extreme_sowing_years)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(by_year['sowing_year'], by_year['actual_mean'], 'o-', label='Actual (national mean)', color='C0')
    ax.plot(by_year['sowing_year'], by_year['predicted_mean'], 's-', label='Predicted (national mean)', color='C1')
    for _, row in by_year[by_year['extreme']].iterrows():
        ax.axvline(row['sowing_year'], color='red', alpha=0.4, linestyle='--')
    ax.set_xlabel('Sowing year')
    ax.set_ylabel('Yield (kg/ha)')
    ax.set_title('National mean yield: actual vs predicted (red vertical = extreme year)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_year_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved time series to {output_dir}/extreme_year_timeseries.png')

    # Bar chart: MAE in normal vs extreme by state (if we have enough states)
    if len(table_state) >= 2 and 'extreme' in table_state['year_type'].values:
        wide = table_state.pivot(index='state', columns='year_type', values='MAE')
        if 'extreme' in wide.columns and 'normal' in wide.columns:
            wide = wide.dropna(how='all')
            if len(wide):
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(wide))
                w = 0.35
                ax.bar(x - w/2, wide['normal'], w, label='Normal years', color='C0')
                ax.bar(x + w/2, wide['extreme'], w, label='Extreme years', color='C1')
                ax.set_xticks(x)
                ax.set_xticklabels(wide.index, rotation=45, ha='right')
                ax.set_ylabel('MAE (kg/ha)')
                ax.set_title('MAE by state: normal vs extreme years')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'extreme_year_MAE_by_state.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f'Saved MAE by state to {output_dir}/extreme_year_MAE_by_state.png')

    # LaTeX table (overall)
    latex_path = os.path.join(output_dir, 'extreme_year_metrics_overall.tex')
    with open(latex_path, 'w') as f:
        f.write(table_overall.to_latex(float_format='%.2f'))
    print(f'Saved LaTeX table to {latex_path}')


def main():
    parser = argparse.ArgumentParser(description='Extreme year analysis from results CSV.')
    parser.add_argument('results_csv', help='Path to results CSV (e.g. all_splits_results.csv or *_test_results.csv)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same dir as CSV)')
    parser.add_argument('--no-programmatic', action='store_true', help='Do not flag extra extreme years by deviation from trend')
    parser.add_argument('--n-std', type=float, default=1.0, help='Number of std for programmatic extreme years')
    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.results_csv).parent)
    run_analysis(
        args.results_csv,
        output_dir,
        use_programmatic=not args.no_programmatic,
        n_std=args.n_std,
    )


if __name__ == '__main__':
    main()
