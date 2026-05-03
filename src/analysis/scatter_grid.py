"""
Per-baseline predicted-vs-actual scatter grid (R2-2, Appendix E).

Renders a 3×3 grid of scatter plots, one per ML/DL baseline reported in
Table~\\ref{tab:performance-comparison}. Each subplot:
  * scatters predicted vs. actual yield from the best-seed
    ``model_*_test_results.csv`` (the only seed kept on disk per the protocol);
  * overlays the 1:1 identity line;
  * annotates **both** the best-seed MAE / R^2 (computed from the points actually
    plotted) and the 5-seed mean ± std MAE / R^2 (the values used for ranking
    in Table 1) — so high-variance baselines can no longer hide behind a lucky
    seed in this figure;
  * shares X/Y axes for visual comparability across panels.

The proposed CNN-BiLSTM-Attn occupies the top-left tile and is highlighted.
Persistence is intentionally excluded — it is a non-ML reference handled in §5.5.
The Random Forest tile uses the canonical grid-search joblib at
``src/rf/saved_models/rf_model_flat.joblib`` (the same artifact whose
five-seed-aggregated metrics are reported in Table 1) re-predicted on
``data/datasets/rf_flat/test_rf_flat.csv``; per-seed std is read from the
companion ``test_metrics_flat_summary.json``.

Output:
    paper_revision/figures/scatter_grid.png   (and .pdf)

Run from repo root:
    python -m src.analysis.scatter_grid
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from .plot_style import apply_style, DPI
    from .build_baseline_table import (
        BASELINE_SPECS, SAVED_MODELS, latest_metadata, REPO_ROOT,
    )
except ImportError:  # script-style invocation
    from plot_style import apply_style, DPI  # type: ignore[no-redef]
    from build_baseline_table import (  # type: ignore[no-redef]
        BASELINE_SPECS, SAVED_MODELS, latest_metadata, REPO_ROOT,
    )


DEFAULT_OUTPUT = REPO_ROOT / "paper_revision" / "figures" / "scatter_grid.png"

# Random Forest artifacts: a single best-seed joblib + a 5-seed aggregated JSON.
RF_DIR = REPO_ROOT / "src" / "rf" / "saved_models"
RF_JOBLIB = RF_DIR / "rf_model_flat.joblib"
RF_SUMMARY_JSON = RF_DIR / "test_metrics_flat_summary.json"
RF_PREDICTIONS_CSV = RF_DIR / "rf_model_flat_test_results.csv"
RF_TEST_CSV = REPO_ROOT / "data" / "datasets" / "rf_flat" / "test_rf_flat.csv"


@dataclass(frozen=True)
class PanelData:
    display: str
    is_ours: bool
    actual: np.ndarray
    predicted: np.ndarray
    # Best-seed (computed from the on-disk predictions).
    mae_best: float
    r2_best: float
    # 5-seed mean ± std (from metadata YAML / aggregated JSON). Optional because
    # a model may have been trained for a single seed only.
    mae_mean: Optional[float]
    mae_std: Optional[float]
    r2_mean: Optional[float]
    r2_std: Optional[float]


def _best_seed_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    err = predicted - actual
    var_y = float(np.var(actual))
    mae = float(np.mean(np.abs(err)))
    r2 = 1.0 - float(np.mean(err ** 2)) / var_y if var_y > 0 else float("nan")
    return mae, r2


def load_baseline_panel(spec) -> PanelData:
    """Best-seed CSV + 5-seed aggregates from the metadata YAML."""
    meta_path = latest_metadata(SAVED_MODELS / spec.run_dir)
    results_path = meta_path.with_name(
        meta_path.name.replace("_metadata.yaml", "_test_results.csv")
    )
    if not results_path.exists():
        raise FileNotFoundError(f"Missing test results CSV: {results_path}")
    df = pd.read_csv(results_path)
    actual = df["actual_yield"].to_numpy(dtype=float)
    predicted = df["predicted_yield"].to_numpy(dtype=float)
    mae_best, r2_best = _best_seed_metrics(actual, predicted)

    with meta_path.open() as fh:
        meta = yaml.safe_load(fh)
    mean = meta.get("mean_test_metrics") or {}
    std = meta.get("std_test_metrics") or {}
    return PanelData(
        display=spec.display, is_ours=spec.is_ours,
        actual=actual, predicted=predicted,
        mae_best=mae_best, r2_best=r2_best,
        mae_mean=float(mean["mae"]) if "mae" in mean else None,
        mae_std=float(std["mae"]) if "mae" in std else None,
        r2_mean=float(mean["r2_score"]) if "r2_score" in mean else None,
        r2_std=float(std["r2_score"]) if "r2_score" in std else None,
    )


def _build_rf_predictions_csv(
    joblib_path: Path, test_csv: Path, out_csv: Path,
) -> pd.DataFrame:
    """Re-predict the saved RF on the flat test split; cache as CSV with the same
    schema as DL/GBM ``model_*_test_results.csv`` files (district_id, sowing_year,
    actual_yield, predicted_yield, residual, absolute_error)."""
    test = pd.read_csv(test_csv)
    feature_cols = [c for c in test.columns if c not in ("yield_value", "district", "year")]
    with warnings.catch_warnings():
        # The .joblib was pickled with sklearn 1.7 and we are on 1.8; the version
        # warning is noisy and not actionable here (RF inference is robust).
        warnings.simplefilter("ignore")
        model = joblib.load(joblib_path)
    if model.n_features_in_ != len(feature_cols):
        raise RuntimeError(
            f"RF feature mismatch: model expects {model.n_features_in_} features, "
            f"test CSV provides {len(feature_cols)}."
        )
    # Pass the DataFrame (not .to_numpy()) so sklearn matches by feature name
    # — the joblib was fit with feature names, and stripping them triggers a
    # noisy UserWarning even though predictions are correct.
    predicted = model.predict(test[feature_cols])
    actual = test["yield_value"].to_numpy(dtype=float)
    residual = predicted - actual
    out = pd.DataFrame({
        "district_id": test["district"],
        "sowing_year": test["year"],
        "actual_yield": actual,
        "predicted_yield": predicted,
        "residual": residual,
        "absolute_error": np.abs(residual),
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


def load_rf_panel() -> PanelData:
    """Random Forest panel: predictions from the saved joblib, std from JSON."""
    if not RF_JOBLIB.exists():
        raise FileNotFoundError(f"Missing RF joblib: {RF_JOBLIB}")
    if not RF_SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Missing RF summary JSON: {RF_SUMMARY_JSON}")

    df = _build_rf_predictions_csv(RF_JOBLIB, RF_TEST_CSV, RF_PREDICTIONS_CSV)
    actual = df["actual_yield"].to_numpy(dtype=float)
    predicted = df["predicted_yield"].to_numpy(dtype=float)
    mae_best, r2_best = _best_seed_metrics(actual, predicted)

    with RF_SUMMARY_JSON.open() as fh:
        summary = json.load(fh)
    mean = summary.get("mean", {})
    std = summary.get("std_dev", {})
    return PanelData(
        display="Random Forest", is_ours=False,
        actual=actual, predicted=predicted,
        mae_best=mae_best, r2_best=r2_best,
        mae_mean=float(mean.get("mae")) if "mae" in mean else None,
        mae_std=float(std.get("mae")) if "mae" in std else None,
        r2_mean=float(mean.get("r2_score")) if "r2_score" in mean else None,
        r2_std=float(std.get("r2_score")) if "r2_score" in std else None,
    )


def panel_order(panels: list[PanelData]) -> list[PanelData]:
    """Ours first, then remaining baselines ascending by 5-seed mean MAE
    (matches Table 1 ranking). Panels missing ``mae_mean`` fall to the end."""
    ours = [p for p in panels if p.is_ours]
    rest = [p for p in panels if not p.is_ours]
    rest.sort(key=lambda p: (p.mae_mean is None, p.mae_mean if p.mae_mean is not None else 0.0))
    return ours + rest


def _annotation_text(p: PanelData) -> str:
    """Two-line annotation: best-seed (matches the dots) + 5-seed mean ± std."""
    line1 = f"seed: MAE {p.mae_best:.0f}, $R^2$ {p.r2_best:.3f}"
    if p.mae_mean is None or p.mae_std is None:
        return line1
    mae_mean_std = f"{p.mae_mean:.0f} $\\pm$ {p.mae_std:.0f}"
    if p.r2_mean is not None and p.r2_std is not None:
        r2_mean_std = f"{p.r2_mean:.3f} $\\pm$ {p.r2_std:.3f}"
    else:
        r2_mean_std = "—"
    line2 = f"5-seed: {mae_mean_std}, {r2_mean_std}"
    return f"{line1}\n{line2}"


def render_grid(panels: list[PanelData], out_png: Path, ncols: int = 3) -> None:
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 4.2 * nrows),
        sharex=True, sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    all_actual = np.concatenate([p.actual for p in panels])
    all_pred = np.concatenate([p.predicted for p in panels])
    lo = float(min(all_actual.min(), all_pred.min()))
    hi = float(max(all_actual.max(), all_pred.max()))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    for idx, ax in enumerate(axes_flat):
        if idx >= len(panels):
            ax.set_visible(False)
            continue
        p = panels[idx]
        ax.scatter(p.actual, p.predicted, s=10, alpha=0.55,
                   edgecolors="none",
                   color="C1" if p.is_ours else "C0")
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04, 0.96,
            _annotation_text(p),
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", alpha=0.85),
        )
        title_weight = "bold" if p.is_ours else "normal"
        ax.set_title(p.display, fontsize=11, fontweight=title_weight)
        if idx % ncols == 0:
            ax.set_ylabel("Predicted yield (kg/ha)")
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Actual yield (kg/ha)")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main(out_png: Path) -> None:
    apply_style()
    panels = [load_baseline_panel(spec) for spec in BASELINE_SPECS]
    panels.append(load_rf_panel())
    panels = panel_order(panels)

    print("Panels (in display order):")
    for p in panels:
        mean_str = (
            f"{p.mae_mean:6.1f} ± {p.mae_std:5.1f}"
            if p.mae_mean is not None and p.mae_std is not None else "      —      "
        )
        print(
            f"  {p.display:>34s}  n={len(p.actual):4d}  "
            f"best MAE={p.mae_best:6.1f}  R2={p.r2_best:6.3f}  "
            f"|  5-seed MAE {mean_str}"
        )

    render_grid(panels, out_png)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_png.with_suffix('.pdf')}")
    print(f"Cached RF predictions: {RF_PREDICTIONS_CSV}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PNG path (PDF written alongside).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output)
