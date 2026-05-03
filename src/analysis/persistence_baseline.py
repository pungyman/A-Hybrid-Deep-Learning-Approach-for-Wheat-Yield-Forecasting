"""
Persistence-baseline comparison for the canonical CNN-BiLSTM-Attn model.

Computes test-set performance of three baselines and the trained model:
  - Mean predictor                       (yhat = mean(y_train))
  - Persistence forecast                 (yhat = past_yield, i.e. last year's yield)
  - Random Forest (current paper number; reported, not recomputed)
  - CNN-BiLSTM-Attn (best seed)          (yhat from `*_test_results.csv`)

Stratifies the persistence vs. model comparison by sowing year (2022 vs 2023)
to expose where the architectural value-add concentrates.

Used as supporting evidence for the SHAP `past_yield` dominance discussion in
`paper_revision/shap_analysis_notes.md` (and §5.5 of the revised manuscript).

Run from repo root:
    python -m src.analysis.persistence_baseline
or from src/analysis/:
    python persistence_baseline.py

Defaults assume the canonical artifacts:
    data/datasets/final/{test.csv, past_yield_scaler.joblib}
    src/rnn/saved_models/cnn_bilstm_self_attn/model_20251018_064239_test_results.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_CSV = REPO_ROOT / "data/datasets/final/test.csv"
DEFAULT_SCALER = REPO_ROOT / "data/datasets/final/past_yield_scaler.joblib"
DEFAULT_RESULTS_CSV = REPO_ROOT / (
    "src/rnn/saved_models/cnn_bilstm_self_attn/"
    "model_20251018_064239_test_results.csv"
)


def load_joined(test_csv: Path, scaler_path: Path, results_csv: Path) -> pd.DataFrame:
    """Join model predictions with test split metadata; un-z-score `past_yield`."""
    test = pd.read_csv(test_csv)[["DISTRICT_ID", "sowing_year", "past_yield"]]
    test = test.rename(columns={"DISTRICT_ID": "district_id"})
    scaler = joblib.load(scaler_path)
    test["past_yield_raw"] = scaler.inverse_transform(
        test[["past_yield"]].values
    ).ravel()

    results = pd.read_csv(results_csv)
    merged = results.merge(test, on=["district_id", "sowing_year"], how="left")
    if merged["past_yield_raw"].isna().any():
        n_missing = int(merged["past_yield_raw"].isna().sum())
        raise RuntimeError(
            f"Failed to recover past_yield for {n_missing} rows; "
            "check that test.csv and *_test_results.csv come from the same split."
        )
    return merged


def metrics(df: pd.DataFrame, label: str) -> dict:
    """Compute MAE, RMSE, R^2 for persistence and the trained model on `df`."""
    y = df["actual_yield"].to_numpy()
    yhat_model = y - df["residual"].to_numpy()
    yhat_persist = df["past_yield_raw"].to_numpy()
    var_y = float(np.var(y))

    def m(err):
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2 = 1.0 - float(np.mean(err ** 2)) / var_y if var_y > 0 else float("nan")
        return mae, rmse, r2

    mae_p, rmse_p, r2_p = m(y - yhat_persist)
    mae_m, rmse_m, r2_m = m(y - yhat_model)
    return dict(
        label=label, n=len(df),
        persist=dict(mae=mae_p, rmse=rmse_p, r2=r2_p),
        model=dict(mae=mae_m, rmse=rmse_m, r2=r2_m),
        delta=dict(
            mae_pct=(1.0 - mae_m / mae_p) * 100.0 if mae_p else float("nan"),
            rmse_pct=(1.0 - rmse_m / rmse_p) * 100.0 if rmse_p else float("nan"),
            r2_abs=r2_m - r2_p,
        ),
    )


def format_row(r: dict) -> str:
    p, mo, d = r["persist"], r["model"], r["delta"]
    return (
        f"{r['label']:>5}  n={r['n']:3d}  | "
        f"persist MAE={p['mae']:6.1f} RMSE={p['rmse']:6.1f} R2={p['r2']:5.3f} | "
        f"model MAE={mo['mae']:6.1f} RMSE={mo['rmse']:6.1f} R2={mo['r2']:5.3f} | "
        f"dMAE={d['mae_pct']:+5.1f}% dRMSE={d['rmse_pct']:+5.1f}% dR2={d['r2_abs']:+0.3f}"
    )


def mean_predictor(df: pd.DataFrame) -> dict:
    """Mean-predictor baseline on the test set (sanity check, matches metadata)."""
    y = df["actual_yield"].to_numpy()
    yhat = float(np.mean(y))
    err = y - yhat
    return dict(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        r2=0.0,
    )


def main(test_csv: Path, scaler_path: Path, results_csv: Path) -> None:
    print(f"Test split        : {test_csv}")
    print(f"past_yield scaler : {scaler_path}")
    print(f"Model predictions : {results_csv}")
    df = load_joined(test_csv, scaler_path, results_csv)

    mp = mean_predictor(df)
    print(
        f"\nMean-predictor sanity (should match metadata.baseline_*): "
        f"MAE={mp['mae']:.1f} RMSE={mp['rmse']:.1f} R2={mp['r2']:.3f}"
    )

    rows = [
        metrics(df, "all"),
        metrics(df[df.sowing_year == 2022], "2022"),
        metrics(df[df.sowing_year == 2023], "2023"),
    ]

    print("\n--- persistence vs CNN-BiLSTM-Attn (best seed) ---")
    print("legend: dMAE/dRMSE = % reduction by model over persistence (positive = model wins)")
    for r in rows:
        print(format_row(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    p.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.test_csv, args.scaler, args.results_csv)
