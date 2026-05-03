"""
Feature correlation matrix across the full dataset (action_items.md §3.a, reviewer R1-1).

Computes the Pearson correlation matrix across the 13 temporal features (collapsed to
their growing-season means) plus lagged yield, over all district-year samples in
train + val + test (2001-2023). Renders an annotated heatmap and saves the matrix
to disk for reproducibility.

Note: Pearson correlation is invariant under per-feature affine transforms, so the
values computed here on the StandardScaler-normalised inputs are numerically identical
to those on the raw inputs.

Run:
    python -m src.analysis.feature_correlation                # from repo root
    python feature_correlation.py                              # from src/analysis/
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .plot_style import apply_style
except ImportError:
    from plot_style import apply_style

apply_style()

# Canonical 13 temporal features; order must match data/datasets/final/metadata.json
FEATURE_NAMES = [
    'T2M_MAX', 'T2M_MIN', 'T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN',
    'WS10M', 'QV2M', 'PS', 'max_vpd', 'min_vpd', 'NDVI', 'EVI',
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / 'data' / 'datasets' / 'final'
DEFAULT_TRAIN_CSV = DATASET_DIR / 'train.csv'
DEFAULT_VAL_CSV = DATASET_DIR / 'val.csv'
DEFAULT_TEST_CSV = DATASET_DIR / 'test.csv'
DEFAULT_OUTPUT_PNG = REPO_ROOT / 'paper_revision' / 'figures' / 'feature_correlation.png'


def build_feature_matrix(csv_paths: list[Path]) -> pd.DataFrame:
    """Concatenate the given CSVs, take growing-season mean per row, append lagged yield."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    sequences = np.array(
        df['feature_sequence'].apply(ast.literal_eval).tolist(),
        dtype=np.float32,
    )  # shape (N, 11, 13)

    if sequences.ndim != 3 or sequences.shape[2] != len(FEATURE_NAMES):
        raise ValueError(
            f'Unexpected feature_sequence shape {sequences.shape}; '
            f'expected (N, 11, {len(FEATURE_NAMES)}).'
        )

    season_mean = sequences.mean(axis=1)  # (N, 13)
    feat = pd.DataFrame(season_mean, columns=FEATURE_NAMES)
    feat['lagged_yield'] = df['past_yield'].to_numpy()
    return feat


def plot_heatmap(corr: pd.DataFrame, out_png: Path) -> None:
    """Render annotated correlation heatmap and save PNG + PDF variants."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr.values, vmin=-1.0, vmax=1.0, cmap='RdBu_r', aspect='equal')

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)

    for i in range(n):
        for j in range(n):
            r = corr.iat[i, j]
            # Use white text on saturated cells for readability
            color = 'white' if abs(r) > 0.6 else 'black'
            ax.text(j, i, f'{r:.2f}', ha='center', va='center', fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson r')

    ax.set_title('Feature correlation matrix (all district-years, growing-season means)')
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix('.pdf'))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--train-csv',
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help='Path to training CSV with feature_sequence and past_yield columns.',
    )
    parser.add_argument(
        '--val-csv',
        type=Path,
        default=DEFAULT_VAL_CSV,
        help='Path to validation CSV; same schema as training CSV.',
    )
    parser.add_argument(
        '--test-csv',
        type=Path,
        default=DEFAULT_TEST_CSV,
        help='Path to test CSV; same schema as training CSV.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_PNG,
        help='Output PNG path; PDF and CSV are written alongside.',
    )
    args = parser.parse_args()

    csv_paths = [args.train_csv, args.val_csv, args.test_csv]
    for p in csv_paths:
        print(f'Loading {p}')
    feat = build_feature_matrix(csv_paths)
    print(f'Feature matrix shape: {feat.shape}')

    corr = feat.corr(method='pearson')

    if corr.isna().sum().sum() != 0:
        raise RuntimeError('Correlation matrix contains NaN values.')
    if not np.allclose(corr.values, corr.values.T):
        raise RuntimeError('Correlation matrix is not symmetric.')
    if not np.allclose(np.diag(corr.values), 1.0):
        raise RuntimeError('Correlation matrix diagonal is not unity.')

    out_png = args.output
    out_csv = out_png.with_suffix('.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out_csv, float_format='%.6f')
    print(f'Wrote correlation matrix CSV: {out_csv}')

    plot_heatmap(corr, out_png)
    print(f'Wrote heatmap: {out_png}')
    print(f'Wrote heatmap: {out_png.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
