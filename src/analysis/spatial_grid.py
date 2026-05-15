"""
Per-baseline spatial residual map grid (R2-2, Appendix C.2).

Renders a 3x3 grid of choropleth maps, one per ML/DL baseline reported in
Table~\\ref{tab:performance-comparison}. Each subplot shows the 2023 sowing-year
residual (``actual - predicted``, kg/ha) per district on a shared symmetric
``coolwarm`` colour scale; red marks underprediction, blue overprediction.

The proposed CNN-BiLSTM-Attn occupies the top-left tile and is highlighted.
The Random Forest tile uses the cached predictions CSV produced by
``scatter_grid.py`` (``src/rf/saved_models/rf_model_flat_test_results.csv``).
A single horizontal colourbar at the bottom of the figure makes the spatial
distribution of bias directly comparable across panels.

District boundaries are drawn in black; state boundaries (dissolved on
``STATE_UT``) are overlaid in white per R2-1.

Output:
    paper_revision/figures/spatial_residual_grid.png   (and .pdf)

Run from repo root:
    python -m src.analysis.spatial_grid
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from .plot_style import apply_style, DPI
    from .build_baseline_table import (
        BASELINE_SPECS, SAVED_MODELS, latest_metadata, REPO_ROOT,
    )
    from .scatter_grid import RF_PREDICTIONS_CSV
except ImportError:  # script-style invocation
    from plot_style import apply_style, DPI  # type: ignore[no-redef]
    from build_baseline_table import (  # type: ignore[no-redef]
        BASELINE_SPECS, SAVED_MODELS, latest_metadata, REPO_ROOT,
    )
    from scatter_grid import RF_PREDICTIONS_CSV  # type: ignore[no-redef]


DEFAULT_OUTPUT = REPO_ROOT / "paper_revision" / "figures" / "spatial_residual_grid.png"
SHAPEFILE = (
    REPO_ROOT / "data" / "district_boundary_data" / "survey_of_india"
    / "DISTRICT_BOUNDARY.shp"
)
TARGET_YEAR = 2023


@dataclass(frozen=True)
class SpatialPanel:
    display: str
    is_ours: bool
    # Per-district 2023 predictions joined onto the GeoDataFrame; one row per
    # district that had a 2023 ground truth (missing districts stay NaN and
    # render as light grey via missing_kwds).
    gdf: gpd.GeoDataFrame
    mae_best: float
    rmse_best: float
    # Five-seed mean MAE — used only for ordering (matches Table 1 ranking).
    mae_mean: Optional[float]


def _load_districts() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHAPEFILE)
    gdf["district_id"] = (
        gdf["STATE_UT"].astype(str) + "_" + gdf["DISTRICT"].astype(str)
    )
    return gdf


def _best_seed_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    err = predicted - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def _load_predictions_csv(csv_path: Path) -> pd.DataFrame:
    """Read a per-baseline predictions CSV and filter to the target sowing year.

    All baselines (DL/GBM/RF) share the same on-disk schema:
        district_id, sowing_year, actual_yield, predicted_yield, residual,
        absolute_error
    """
    df = pd.read_csv(csv_path)
    df = df[df["sowing_year"] == TARGET_YEAR].copy()
    # Recompute residual as ``actual - predicted`` (the on-disk ``residual``
    # column is stored as ``predicted - actual``); coolwarm + this convention
    # gives red = underprediction, blue = overprediction.
    df["spatial_residual"] = df["actual_yield"] - df["predicted_yield"]
    return df


def _join_to_districts(
    base_districts: gpd.GeoDataFrame, preds: pd.DataFrame,
) -> gpd.GeoDataFrame:
    keep = ["district_id", "actual_yield", "predicted_yield", "spatial_residual"]
    return base_districts.merge(preds[keep], on="district_id", how="left")


def load_baseline_panel(
    spec, base_districts: gpd.GeoDataFrame,
) -> SpatialPanel:
    meta_path = latest_metadata(SAVED_MODELS / spec.run_dir)
    results_path = meta_path.with_name(
        meta_path.name.replace("_metadata.yaml", "_test_results.csv")
    )
    if not results_path.exists():
        raise FileNotFoundError(f"Missing test results CSV: {results_path}")

    preds = _load_predictions_csv(results_path)
    mae_best, rmse_best = _best_seed_metrics(
        preds["actual_yield"].to_numpy(dtype=float),
        preds["predicted_yield"].to_numpy(dtype=float),
    )

    with meta_path.open() as fh:
        meta = yaml.safe_load(fh)
    mean = meta.get("mean_test_metrics") or {}
    mae_mean = float(mean["mae"]) if "mae" in mean else None

    return SpatialPanel(
        display=spec.display, is_ours=spec.is_ours,
        gdf=_join_to_districts(base_districts, preds),
        mae_best=mae_best, rmse_best=rmse_best,
        mae_mean=mae_mean,
    )


def load_rf_panel(base_districts: gpd.GeoDataFrame) -> SpatialPanel:
    if not RF_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Missing cached RF predictions CSV: {RF_PREDICTIONS_CSV}.\n"
            "Run `python -m src.analysis.scatter_grid` first to materialise it."
        )
    preds = _load_predictions_csv(RF_PREDICTIONS_CSV)
    mae_best, rmse_best = _best_seed_metrics(
        preds["actual_yield"].to_numpy(dtype=float),
        preds["predicted_yield"].to_numpy(dtype=float),
    )
    # Original-paper RF uses a single grid-search seed; mae_mean (used only
    # for ordering) falls back to the best-seed value so the RF panel still
    # finds a well-defined slot in the sort.
    return SpatialPanel(
        display="Random Forest", is_ours=False,
        gdf=_join_to_districts(base_districts, preds),
        mae_best=mae_best, rmse_best=rmse_best,
        mae_mean=mae_best,
    )


def panel_order(panels: list[SpatialPanel]) -> list[SpatialPanel]:
    """Ours first, then remaining baselines ascending by 5-seed mean MAE
    (matches Table 1 ranking and the scatter-grid panel order)."""
    ours = [p for p in panels if p.is_ours]
    rest = [p for p in panels if not p.is_ours]
    rest.sort(
        key=lambda p: (
            p.mae_mean is None,
            p.mae_mean if p.mae_mean is not None else 0.0,
        )
    )
    return ours + rest


def _annotation_text(p: SpatialPanel) -> str:
    return f"MAE {p.mae_best:.0f}\nRMSE {p.rmse_best:.0f}"


def _state_boundaries(base_districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return base_districts.dissolve(by="STATE_UT", as_index=False)[["STATE_UT", "geometry"]]


def render_grid(
    panels: list[SpatialPanel],
    state_boundaries: gpd.GeoDataFrame,
    out_png: Path,
    ncols: int = 3,
) -> None:
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.6 * ncols, 4.4 * nrows),
    )
    axes_flat = np.atleast_1d(axes).ravel()

    # Symmetric colour scale: max |residual| across every panel's 2023 subset.
    all_residuals = np.concatenate([
        p.gdf["spatial_residual"].dropna().to_numpy(dtype=float) for p in panels
    ])
    vmax = float(np.nanmax(np.abs(all_residuals)))
    vmin = -vmax

    for idx, ax in enumerate(axes_flat):
        if idx >= len(panels):
            ax.set_visible(False)
            continue
        p = panels[idx]
        # ``rasterized=True`` keeps the PDF small (~2-3 MB instead of ~375 MB)
        # without sacrificing the vector axes / annotations / colourbar.
        p.gdf.plot(
            column="spatial_residual",
            cmap="coolwarm",
            vmin=vmin, vmax=vmax,
            linewidth=0.4, edgecolor="black",
            ax=ax,
            missing_kwds={"color": "lightgrey"},
            rasterized=True,
        )
        # White state boundaries overlay (R2-1).
        state_boundaries.boundary.plot(
            ax=ax, color="white", linewidth=0.8, rasterized=True,
        )
        ax.set_axis_off()
        ax.text(
            0.02, 0.02,
            _annotation_text(p),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey", alpha=0.85),
        )
        title_weight = "bold" if p.is_ours else "normal"
        ax.set_title(p.display, fontsize=11, fontweight=title_weight)

    # Reserve room at the bottom for one shared horizontal colourbar.
    fig.subplots_adjust(bottom=0.10, top=0.96, left=0.02, right=0.98,
                        wspace=0.05, hspace=0.18)
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar_ax = fig.add_axes([0.20, 0.045, 0.60, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Residual (Actual $-$ Predicted) yield (kg/ha)\n"
        "red: underprediction $\\;|\\;$ blue: overprediction",
        fontsize=10,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main(out_png: Path) -> None:
    apply_style()
    base_districts = _load_districts()
    state_boundaries = _state_boundaries(base_districts)

    panels = [load_baseline_panel(spec, base_districts) for spec in BASELINE_SPECS]
    panels.append(load_rf_panel(base_districts))
    panels = panel_order(panels)

    print(f"Spatial residual panels for sowing_year = {TARGET_YEAR}:")
    for p in panels:
        n_present = int(p.gdf["spatial_residual"].notna().sum())
        mean_str = (
            f"{p.mae_mean:6.1f}"
            if p.mae_mean is not None else "    —  "
        )
        print(
            f"  {p.display:>34s}  n={n_present:4d}  "
            f"MAE_best={p.mae_best:6.1f}  RMSE_best={p.rmse_best:6.1f}  "
            f"|  5-seed MAE {mean_str}"
        )

    render_grid(panels, state_boundaries, out_png)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_png.with_suffix('.pdf')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output PNG path (PDF written alongside).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output)
