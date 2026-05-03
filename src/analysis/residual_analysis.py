"""
Residual analysis by state and limitations quantification.
- Items #2, #4, #10: state-wise residuals, regression-to-mean metrics, yield-class distribution.

Expects results CSV with: district_id, sowing_year, actual_yield, predicted_yield, residual, absolute_error.
For yield-class distribution and lagged-yield analysis, optionally pass paths to train/val/test CSVs
and a results CSV that includes lagged yield (past_yield) if available.
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
    """Extract state from district_id (format STATE_DISTRICT, e.g. PUNJAB_AMRITSAR)."""
    if pd.isna(district_id):
        return ''
    parts = str(district_id).split('_', 1)
    return parts[0] if len(parts) > 1 else str(district_id)


def load_results(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path)
    for col in ['district_id', 'actual_yield', 'predicted_yield']:
        if col not in df.columns:
            raise ValueError(f'Results CSV must contain column: {col}')
    # Recompute residual using standard convention: actual - predicted
    # (CSV may have predicted - actual, so we recompute to ensure consistency)
    df['residual'] = df['actual_yield'] - df['predicted_yield']
    df['state'] = df['district_id'].apply(extract_state)
    if 'absolute_error' not in df.columns:
        df['absolute_error'] = np.abs(df['residual'])
    return df


def per_state_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-state: mean residual, RMSE, MAE, mean actual yield, std residual."""
    rows = []
    for state in df['state'].dropna().unique():
        if state == '':
            continue
        sub = df[df['state'] == state]
        if len(sub) < 2:
            continue
        rows.append({
            'state': state,
            'n': len(sub),
            'mean_actual_yield': sub['actual_yield'].mean(),
            'mean_predicted_yield': sub['predicted_yield'].mean(),
            'mean_residual': sub['residual'].mean(),
            'std_residual': sub['residual'].std(),
            'RMSE': np.sqrt((sub['residual'] ** 2).mean()),
            'MAE': sub['absolute_error'].mean(),
        })
    return pd.DataFrame(rows).sort_values('mean_actual_yield', ascending=False)


def plot_residuals_by_state(df: pd.DataFrame, output_dir: str) -> None:
    """Box plot of residuals grouped by state."""
    df_plot = df[df['state'] != ''].copy()
    if df_plot.empty:
        return
    states_ordered = df_plot.groupby('state')['actual_yield'].mean().sort_values(ascending=False).index.tolist()
    df_plot['state'] = pd.Categorical(df_plot['state'], categories=states_ordered, ordered=True)
    df_plot = df_plot.sort_values('state')
    fig, ax = plt.subplots(figsize=(10, 6))
    df_plot.boxplot(column='residual', by='state', ax=ax)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('State')
    ax.set_ylabel('Residual (actual - predicted, kg/ha)')
    ax.set_title('Residuals by state')
    plt.suptitle('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_by_state_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved residuals_by_state_boxplot.png to {output_dir}')


def plot_actual_vs_residual(df: pd.DataFrame, output_dir: str) -> None:
    """Scatter: actual yield vs residual, points colored by state (visualizes regression-to-mean)."""
    df_plot = df[df['state'] != ''].copy()
    if df_plot.empty:
        return
    states = df_plot['state'].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(states), 1)))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, state in enumerate(states):
        sub = df_plot[df_plot['state'] == state]
        ax.scatter(sub['actual_yield'], sub['residual'], alpha=0.6, label=state, color=colors[i % len(colors)], s=20)
    ax.axhline(0, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel('Actual yield (kg/ha)')
    ax.set_ylabel('Residual (actual - predicted, kg/ha)')
    ax.set_title('Residual vs actual yield by state (regression-to-mean: high yield underpredicted, low overpredicted)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_yield_vs_residual_by_state.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved actual_yield_vs_residual_by_state.png to {output_dir}')


def run_residual_analysis(results_path: str, output_dir: str) -> pd.DataFrame:
    """Load results, compute per-state metrics, generate figures and tables. Returns state metrics DataFrame."""
    df = load_results(results_path)
    os.makedirs(output_dir, exist_ok=True)
    state_metrics = per_state_metrics(df)
    state_metrics.to_csv(os.path.join(output_dir, 'per_state_metrics.csv'), index=False)
    # LaTeX
    with open(os.path.join(output_dir, 'per_state_metrics.tex'), 'w') as f:
        f.write(state_metrics.to_latex(index=False, float_format='%.2f'))
    plot_residuals_by_state(df, output_dir)
    plot_actual_vs_residual(df, output_dir)
    # Punjab vs Rajasthan summary
    for state in ['PUNJAB', 'RAJASTHAN']:
        row = state_metrics[state_metrics['state'] == state]
        if not row.empty:
            print(f'{state}: mean residual = {row["mean_residual"].values[0]:.1f} kg/ha, RMSE = {row["RMSE"].values[0]:.1f}')
    return state_metrics


# ---------- Limitations (#4): prediction compression, yield bins, lagged yield ----------

def prediction_compression_ratio(df: pd.DataFrame, group_col: str = 'state') -> pd.DataFrame:
    """Var(predicted) / Var(actual) per group. Values < 1 indicate regression-to-mean."""
    rows = []
    for name, sub in df.groupby(group_col):
        if len(sub) < 3:
            continue
        var_act = sub['actual_yield'].var()
        var_pred = sub['predicted_yield'].var()
        ratio = var_pred / var_act if var_act > 0 else np.nan
        rows.append({group_col: name, 'var_actual': var_act, 'var_predicted': var_pred, 'compression_ratio': ratio, 'n': len(sub)})
    return pd.DataFrame(rows)


def yield_bin_metrics(df: pd.DataFrame, bins: list = [0, 2000, 3500, 10000], labels: list = ['low', 'medium', 'high']) -> pd.DataFrame:
    """MAE and sample count per yield bin (by actual_yield)."""
    if len(bins) != len(labels) + 1:
        labels = [f'bin_{i}' for i in range(len(bins) - 1)]
    df = df.copy()
    df['yield_bin'] = pd.cut(df['actual_yield'], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for b in labels:
        sub = df[df['yield_bin'] == b]
        if len(sub) == 0:
            continue
        rows.append({
            'yield_bin': b,
            'n': len(sub),
            'MAE': sub['absolute_error'].mean(),
            'RMSE': np.sqrt((sub['residual'].pow(2).mean())),
            'mean_actual': sub['actual_yield'].mean(),
            'mean_predicted': sub['predicted_yield'].mean(),
        })
    return pd.DataFrame(rows)


def run_limitations_analysis(df: pd.DataFrame, output_dir: str, past_yield_series: pd.Series = None) -> None:
    """
    Add limitations-related outputs: compression ratio, yield-bin metrics, optional lagged-yield correlation.
    past_yield_series: lagged yield values; if provided, index must align with df for correlation with predicted_yield.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Compression ratio by state
    comp = prediction_compression_ratio(df)
    comp.to_csv(os.path.join(output_dir, 'prediction_compression_ratio_by_state.csv'), index=False)
    print('Prediction compression ratio (Var(pred)/Var(act)) by state:')
    print(comp.to_string())
    # Yield-bin metrics
    bin_metrics = yield_bin_metrics(df)
    bin_metrics.to_csv(os.path.join(output_dir, 'yield_bin_metrics.csv'), index=False)
    # Figure: predicted vs actual range by state (whisker)
    state_order = df.groupby('state')['actual_yield'].mean().sort_values(ascending=False).index
    state_order = [s for s in state_order if s != '']
    if state_order:
        fig, ax = plt.subplots(figsize=(10, 5))
        data_act = [df[df['state'] == s]['actual_yield'].values for s in state_order]
        data_pred = [df[df['state'] == s]['predicted_yield'].values for s in state_order]
        x = np.arange(len(state_order))
        w = 0.35
        bp1 = ax.boxplot(data_act, positions=x - w/2, widths=w*0.8, patch_artist=True, labels=['']*len(state_order))
        bp2 = ax.boxplot(data_pred, positions=x + w/2, widths=w*0.8, patch_artist=True, labels=['']*len(state_order))
        for b in bp1['boxes']:
            b.set_facecolor('C0')
            b.set_alpha(0.7)
        for b in bp2['boxes']:
            b.set_facecolor('C1')
            b.set_alpha(0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(state_order, rotation=45, ha='right')
        ax.set_ylabel('Yield (kg/ha)')
        ax.set_title('Actual vs predicted yield range by state (compression in extremes)')
        ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Actual', 'Predicted'])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'yield_range_by_state.png'), dpi=300, bbox_inches='tight')
        plt.close()
    if past_yield_series is not None and len(past_yield_series) == len(df):
        corr_pred_lag = np.corrcoef(df['predicted_yield'], past_yield_series)[0, 1]
        corr_act_lag = np.corrcoef(df['actual_yield'], past_yield_series)[0, 1]
        corr_act_pred = np.corrcoef(df['actual_yield'], df['predicted_yield'])[0, 1]
        with open(os.path.join(output_dir, 'lagged_yield_correlations.txt'), 'w') as f:
            f.write(f'Correlation(predicted, lagged_yield) = {corr_pred_lag:.4f}\n')
            f.write(f'Correlation(actual, lagged_yield)     = {corr_act_lag:.4f}\n')
            f.write(f'Correlation(actual, predicted)        = {corr_act_pred:.4f}\n')


# ---------- Yield-class distribution (#10) ----------

def yield_class_distribution_table(train_path: str, val_path: str, test_path: str, output_dir: str,
                                   bins: list = [0, 2000, 3500, 10000], labels: list = ['low', 'medium', 'high']) -> pd.DataFrame:
    """
    Count samples in each yield bin for train/val/test. CSVs must have column 'Yield'.
    """
    if len(bins) != len(labels) + 1:
        labels = [f'bin_{i}' for i in range(len(bins) - 1)]
    rows = []
    for split_name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not os.path.exists(path):
            print(f'Skipping {split_name}: file not found {path}')
            continue
        d = pd.read_csv(path)
        if 'Yield' not in d.columns:
            print(f'Skipping {split_name}: no Yield column')
            continue
        d['bin'] = pd.cut(d['Yield'], bins=bins, labels=labels, include_lowest=True)
        total = len(d)
        for b in labels:
            n = (d['bin'] == b).sum()
            pct = 100 * n / total if total else 0
            rows.append({'split': split_name, 'yield_bin': b, 'count': n, 'pct': pct})
    table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(os.path.join(output_dir, 'yield_class_distribution.csv'), index=False)
    # Pivot for readability
    pivot = table.pivot_table(index='yield_bin', columns='split', values='count', aggfunc='sum').fillna(0).astype(int)
    pivot.to_csv(os.path.join(output_dir, 'yield_class_distribution_pivot.csv'))
    with open(os.path.join(output_dir, 'yield_class_distribution.tex'), 'w') as f:
        f.write(pivot.to_latex())
    return table


def main():
    parser = argparse.ArgumentParser(description='Residual analysis by state and limitations.')
    parser.add_argument('results_csv', help='Path to results CSV from evaluation')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same dir as CSV)')
    parser.add_argument('--limitations', action='store_true', help='Also run limitations analysis (compression, yield bins)')
    parser.add_argument('--yield-dist', action='store_true', help='Generate yield-class distribution table (requires --train/--val/--test)')
    parser.add_argument('--train', default='', help='Path to train.csv for yield-class distribution')
    parser.add_argument('--val', default='', help='Path to val.csv for yield-class distribution')
    parser.add_argument('--test', default='', help='Path to test.csv for yield-class distribution')
    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.results_csv).parent)
    df = load_results(args.results_csv)
    run_residual_analysis(args.results_csv, output_dir)
    if args.limitations:
        run_limitations_analysis(df, output_dir)
    if args.yield_dist and args.train and args.val and args.test:
        yield_class_distribution_table(args.train, args.val, args.test, output_dir)


if __name__ == '__main__':
    main()
