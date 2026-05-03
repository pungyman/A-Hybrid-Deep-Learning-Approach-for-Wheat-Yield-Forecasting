"""
Canonical Table 1 generator — single source of truth for the revised manuscript.

Reads every `src/rnn/saved_models/<baseline>/model_*_metadata.yaml` enumerated in
``BASELINE_SPECS`` and emits three artifacts in ``paper_revision/figures/``:

  1. ``table1_baseline_comparison.{tex,csv}`` — primary benchmark table for §5.1.
     Rows are ML/DL models only (no persistence), sorted ascending by mean MAE
     across five seeds; the ours row is bolded. The Random Forest entry is
     hard-coded with the original-paper grid-search numbers per user directive
     (``rf_without_soil`` is excluded).

  2. ``table_persistence_comparison.{tex,csv}`` — three-row §5.5 table:
     persistence (`ŷ = past_yield`), CNN-BiLSTM-Attn 5-seed mean, and
     CNN-BiLSTM-Attn best seed.

  3. ``table_persistence_per_year.{tex,csv}`` — per-year (2022, 2023, all)
     persistence-vs-model breakdown for §5.5; uses the canonical best-seed
     predictions from ``cnn_bilstm_self_attn`` and is labeled as such.

Per-baseline numbers come from ``mean_test_metrics`` / ``std_test_metrics``;
parameter counts come from ``total_parameters`` (DL) or are left blank (GBM, RF).

Run from repo root:
    python -m src.analysis.build_baseline_table
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    from .persistence_baseline import (
        DEFAULT_RESULTS_CSV as PERSISTENCE_RESULTS_CSV,
        DEFAULT_SCALER as PERSISTENCE_SCALER,
        DEFAULT_TEST_CSV as PERSISTENCE_TEST_CSV,
        load_joined,
        metrics as persistence_metrics,
    )
except ImportError:  # script-style invocation
    from persistence_baseline import (  # type: ignore[no-redef]
        DEFAULT_RESULTS_CSV as PERSISTENCE_RESULTS_CSV,
        DEFAULT_SCALER as PERSISTENCE_SCALER,
        DEFAULT_TEST_CSV as PERSISTENCE_TEST_CSV,
        load_joined,
        metrics as persistence_metrics,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
SAVED_MODELS = REPO_ROOT / "src" / "rnn" / "saved_models"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_revision" / "figures"


@dataclass(frozen=True)
class BaselineSpec:
    """Display name + on-disk run-dir for a baseline shown in Table 1."""
    display: str
    run_dir: str
    is_ours: bool = False


# Order here is irrelevant; the table is sorted by MAE downstream. The
# ``rf_without_soil`` directory exists but is excluded per user directive
# (Random Forest stays at the original-paper grid-search numbers below).
BASELINE_SPECS: list[BaselineSpec] = [
    BaselineSpec("CNN-BiLSTM-Attn (ours)", "cnn_bilstm_self_attn", is_ours=True),
    BaselineSpec("LSTM (flat soil)", "lstm_flatsoil"),
    BaselineSpec("BiLSTM (flat soil)", "bilstm_flatsoil"),
    BaselineSpec("Transformer encoder", "transformer_with_soil"),
    BaselineSpec("1D-CNN", "cnn1d_with_soil"),
    BaselineSpec("Parallel CNN-LSTM-Attn (TPCLA)", "parallel_with_soil"),
    BaselineSpec("XGBoost", "xgb_with_soil"),
    BaselineSpec("LightGBM", "lgbm_with_soil"),
]

# Original-paper Random Forest with-soil grid-search numbers; retained verbatim
# per user directive. Single-seed point estimates — std deliberately blank.
RF_ORIGINAL = {
    "display": "Random Forest",
    "n_params": None,
    "mae_mean": 486.432,
    "mae_std": None,
    "rmse_mean": 601.260,
    "rmse_std": None,
    "r2_mean": 0.458,
    "r2_std": None,
    "is_ours": False,
}


def latest_metadata(run_dir: Path) -> Path:
    """Return the lexicographically latest ``model_*_metadata.yaml`` in ``run_dir``."""
    candidates = sorted(run_dir.glob("model_*_metadata.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No model_*_metadata.yaml under {run_dir}")
    return candidates[-1]


def read_baseline_row(spec: BaselineSpec) -> dict:
    """Pull mean / std test metrics + param count from a baseline metadata YAML."""
    meta_path = latest_metadata(SAVED_MODELS / spec.run_dir)
    with meta_path.open() as fh:
        meta = yaml.safe_load(fh)
    mean = meta["mean_test_metrics"]
    std = meta["std_test_metrics"]
    return {
        "display": spec.display,
        "n_params": meta.get("total_parameters"),  # absent for GBMs
        "mae_mean": float(mean["mae"]),
        "mae_std": float(std["mae"]),
        "rmse_mean": float(mean["rmse"]),
        "rmse_std": float(std["rmse"]),
        "r2_mean": float(mean["r2_score"]),
        "r2_std": float(std["r2_score"]),
        "is_ours": spec.is_ours,
        "metadata_path": str(meta_path.relative_to(REPO_ROOT)),
    }


def _is_missing(x) -> bool:
    """True for None or NaN floats (which is how missing cells survive a DataFrame round-trip)."""
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except TypeError:
        return False


def fmt_metric(mean: float, std: Optional[float], decimals: int) -> str:
    """Render ``mean ± std`` with consistent decimal places; missing std → mean only."""
    fmt = f"{{:.{decimals}f}}"
    if _is_missing(std):
        return fmt.format(mean)
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"


def fmt_params(n: Optional[int]) -> str:
    """Format parameter counts as comma-separated integers; missing → em-dash."""
    if _is_missing(n):
        return "---"
    return f"{int(n):,}"


def build_table1_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Produce a sorted DataFrame matching the Table 1 column schema."""
    df = pd.DataFrame(rows)
    rf_row = df[df["display"] == RF_ORIGINAL["display"]]
    other_rows = df[df["display"] != RF_ORIGINAL["display"]].sort_values(
        by="mae_mean", ascending=True, kind="stable",
    )
    df = pd.concat([other_rows, rf_row], ignore_index=True)
    return df


def write_table1_csv(df: pd.DataFrame, out_csv: Path) -> None:
    """Plain-CSV dump of the same rows shown in the LaTeX table."""
    cols = [
        "display", "n_params",
        "mae_mean", "mae_std",
        "rmse_mean", "rmse_std",
        "r2_mean", "r2_std",
        "is_ours",
    ]
    if "metadata_path" in df.columns:
        cols.append("metadata_path")
    df = df[[c for c in cols if c in df.columns]].rename(columns={
        "display": "model",
        "n_params": "parameters",
    })
    df.to_csv(out_csv, index=False)


def write_table1_tex(df: pd.DataFrame, out_tex: Path) -> None:
    """Emit a ready-to-include ``tabular*`` block — same style as tab:performance-comparison."""
    header = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Performance comparison on the held-out test set (2022--2023) under "
        "identical feature access (temporal + soil + past\\_yield) and an identical "
        "five-seed protocol. Cells report mean $\\pm$ std across five seeds; best "
        "values per column in bold. Row order is best-to-worst on MAE.}\n"
        "  \\label{tab:performance-comparison}\n"
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrlll@{}}\n"
        "    \\toprule\n"
        "    \\textbf{Model} & \\textbf{\\#Params} & \\textbf{MAE (kg/ha)} & "
        "\\textbf{RMSE (kg/ha)} & $\\boldsymbol{R^{2}}$ \\\\\n"
        "    \\midrule\n"
    )

    body_lines: list[str] = []
    for _, row in df.iterrows():
        mae = fmt_metric(row["mae_mean"], row.get("mae_std"), 2)
        rmse = fmt_metric(row["rmse_mean"], row.get("rmse_std"), 2)
        r2 = fmt_metric(row["r2_mean"], row.get("r2_std"), 3)
        params = fmt_params(row.get("n_params"))
        name = row["display"]
        if row.get("is_ours"):
            name = f"\\textbf{{{name}}}"
            mae = f"\\textbf{{{mae}}}"
            rmse = f"\\textbf{{{rmse}}}"
            r2 = f"\\textbf{{{r2}}}"
        body_lines.append(f"    {name} & {params} & {mae} & {rmse} & {r2} \\\\")

    footer = (
        "\n    \\bottomrule\n"
        "  \\end{tabular*}\n"
        "\\end{table}\n"
    )

    out_tex.write_text(header + "\n".join(body_lines) + footer)


def build_persistence_comparison(ours_meta_path: Path) -> tuple[dict, dict, dict]:
    """Return persistence, ours-mean, ours-best metric dicts for the §5.5 table."""
    df = load_joined(PERSISTENCE_TEST_CSV, PERSISTENCE_SCALER, ours_meta_path.parent / (
        ours_meta_path.name.replace("_metadata.yaml", "_test_results.csv")
    ))
    overall = persistence_metrics(df, "all")
    persistence = {
        "label": "Persistence ($\\hat{y} = \\text{past\\_yield}$)",
        "mae": overall["persist"]["mae"],
        "rmse": overall["persist"]["rmse"],
        "r2": overall["persist"]["r2"],
    }
    with ours_meta_path.open() as fh:
        meta = yaml.safe_load(fh)
    ours_mean = {
        "label": "CNN-BiLSTM-Attn (5-seed mean)",
        "mae": float(meta["mean_test_metrics"]["mae"]),
        "rmse": float(meta["mean_test_metrics"]["rmse"]),
        "r2": float(meta["mean_test_metrics"]["r2_score"]),
    }
    ours_best = {
        "label": "CNN-BiLSTM-Attn (best seed)",
        "mae": float(meta["test_metrics"]["mae"]),
        "rmse": float(meta["test_metrics"]["rmse"]),
        "r2": float(meta["test_metrics"]["r2_score"]),
    }
    return persistence, ours_mean, ours_best


def write_persistence_comparison(
    persistence: dict, ours_mean: dict, ours_best: dict,
    out_tex: Path, out_csv: Path,
) -> None:
    rows = [persistence, ours_mean, ours_best]
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Persistence baseline ($\\hat{y} = \\text{past\\_yield}$) versus the "
        "proposed CNN-BiLSTM-Attn on the held-out test set (2022--2023). Persistence "
        "essentially ties the five-seed mean, motivating the per-year decomposition in "
        "Table~\\ref{tab:persistence-per-year} and the SHAP analysis that follows.}",
        "  \\label{tab:persistence-comparison}",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrr@{}}",
        "    \\toprule",
        "    \\textbf{Predictor} & \\textbf{MAE (kg/ha)} & \\textbf{RMSE (kg/ha)} & "
        "$\\boldsymbol{R^{2}}$ \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            f"    {row['label']} & {row['mae']:.2f} & {row['rmse']:.2f} & {row['r2']:.3f} \\\\"
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines))


def build_per_year_table(ours_meta_path: Path) -> pd.DataFrame:
    """Compute persistence vs. CNN-BiLSTM-Attn (best seed) for {all, 2022, 2023}."""
    results_csv = ours_meta_path.parent / ours_meta_path.name.replace(
        "_metadata.yaml", "_test_results.csv"
    )
    df = load_joined(PERSISTENCE_TEST_CSV, PERSISTENCE_SCALER, results_csv)
    rows = []
    for label, sub in (
        ("All", df),
        ("2022", df[df.sowing_year == 2022]),
        ("2023", df[df.sowing_year == 2023]),
    ):
        m = persistence_metrics(sub, label)
        rows.append({
            "Period": label,
            "n": int(m["n"]),
            "Persistence MAE": m["persist"]["mae"],
            "Persistence RMSE": m["persist"]["rmse"],
            "Persistence R2": m["persist"]["r2"],
            "Model MAE": m["model"]["mae"],
            "Model RMSE": m["model"]["rmse"],
            "Model R2": m["model"]["r2"],
            "dMAE_pct": m["delta"]["mae_pct"],
            "dRMSE_pct": m["delta"]["rmse_pct"],
            "dR2_abs": m["delta"]["r2_abs"],
        })
    return pd.DataFrame(rows)


def write_per_year_table(df: pd.DataFrame, out_tex: Path, out_csv: Path) -> None:
    df.to_csv(out_csv, index=False, float_format="%.3f")

    def signed(x: float, decimals: int = 1) -> str:
        # %+ guarantees an explicit sign for negative deltas (model loses).
        return f"{x:+.{decimals}f}"

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Per-year persistence vs.\\ CNN-BiLSTM-Attn (best seed) on the "
        "test set. $\\Delta$ columns report the percentage MAE/RMSE reduction "
        "(positive: model improves over persistence) and the absolute change in "
        "$R^{2}$. The architectural value-add concentrates in the heat-stressed "
        "2023 season.}",
        "  \\label{tab:persistence-per-year}",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrrr@{}}",
        "    \\toprule",
        "    & & \\multicolumn{3}{c}{\\textbf{Persistence}} & "
        "\\multicolumn{3}{c}{\\textbf{CNN-BiLSTM-Attn}} & "
        "\\multicolumn{3}{c}{\\textbf{$\\Delta$ (model $-$ persistence)}} \\\\",
        "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
        "    \\textbf{Period} & \\textbf{$n$} & \\textbf{MAE} & \\textbf{RMSE} & "
        "$\\boldsymbol{R^{2}}$ & \\textbf{MAE} & \\textbf{RMSE} & "
        "$\\boldsymbol{R^{2}}$ & \\textbf{MAE\\%} & \\textbf{RMSE\\%} & "
        "$\\boldsymbol{\\Delta R^{2}}$ \\\\",
        "    \\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            "    {Period} & {n} & {pmae:.1f} & {prmse:.1f} & {pr2:.3f} & "
            "{mmae:.1f} & {mrmse:.1f} & {mr2:.3f} & "
            "{dmae} & {drmse} & {dr2} \\\\".format(
                Period=row["Period"], n=int(row["n"]),
                pmae=row["Persistence MAE"], prmse=row["Persistence RMSE"],
                pr2=row["Persistence R2"],
                mmae=row["Model MAE"], mrmse=row["Model RMSE"], mr2=row["Model R2"],
                dmae=signed(row["dMAE_pct"], 1),
                drmse=signed(row["dRMSE_pct"], 1),
                dr2=signed(row["dR2_abs"], 3),
            )
        )
    lines += [
        "    \\bottomrule",
        "  \\end{tabular*}",
        "\\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines))


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 1 -----------------------------------------------------------
    rows = [read_baseline_row(s) for s in BASELINE_SPECS]
    rows.append(RF_ORIGINAL.copy())
    df = build_table1_dataframe(rows)

    print("Baseline ranking (by mean MAE, ascending):")
    print(df[["display", "n_params", "mae_mean", "rmse_mean", "r2_mean"]].to_string(index=False))

    write_table1_csv(df, output_dir / "table1_baseline_comparison.csv")
    write_table1_tex(df, output_dir / "table1_baseline_comparison.tex")
    print(f"\nWrote {output_dir / 'table1_baseline_comparison.tex'}")
    print(f"Wrote {output_dir / 'table1_baseline_comparison.csv'}")

    # --- §5.5 persistence comparison + per-year ----------------------------
    ours_meta = latest_metadata(SAVED_MODELS / "cnn_bilstm_self_attn")
    persistence, ours_mean, ours_best = build_persistence_comparison(ours_meta)
    write_persistence_comparison(
        persistence, ours_mean, ours_best,
        out_tex=output_dir / "table_persistence_comparison.tex",
        out_csv=output_dir / "table_persistence_comparison.csv",
    )
    print(f"Wrote {output_dir / 'table_persistence_comparison.tex'}")

    per_year = build_per_year_table(ours_meta)
    print("\nPer-year persistence vs. model:")
    print(per_year.to_string(index=False))
    write_per_year_table(
        per_year,
        out_tex=output_dir / "table_persistence_per_year.tex",
        out_csv=output_dir / "table_persistence_per_year.csv",
    )
    print(f"Wrote {output_dir / 'table_persistence_per_year.tex'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for the .tex / .csv outputs.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.output_dir)
