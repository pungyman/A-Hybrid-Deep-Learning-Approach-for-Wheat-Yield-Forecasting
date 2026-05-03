"""
Train-set yield-distribution histogram (R1-3, Appendix D).

Renders a histogram of ``Yield`` (raw kg/ha) from the training split, with the
low/medium/high yield-bin cut-offs used in Table~\\ref{tab:yield-bins}
overlaid as vertical reference lines. Intended to back the rebuttal sentence
that the regression target is approximately unimodal with a mild right skew —
no special class-imbalance handling required.

Output:
    paper_revision/figures/yield_histogram_train.png   (and .pdf)

Run from repo root:
    python -m src.analysis.yield_histogram
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .plot_style import apply_style, DPI
except ImportError:
    from plot_style import apply_style, DPI  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "datasets" / "final" / "train.csv"
DEFAULT_OUTPUT = REPO_ROOT / "paper_revision" / "figures" / "yield_histogram_train.png"

# Same cut-offs used by tab:yield-bins in sections/05-results.tex.
BIN_EDGES = (2000.0, 3500.0)


def main(train_csv: Path, out_png: Path) -> None:
    apply_style()
    df = pd.read_csv(train_csv, usecols=["Yield"])
    y = df["Yield"].to_numpy(dtype=float)
    n = len(y)
    # Boundary convention matches the existing tab:yield-bins counts in
    # sections/05-results.tex: low includes y == 2000, high is strictly > 3500.
    low = int((y <= BIN_EDGES[0]).sum())
    high = int((y > BIN_EDGES[1]).sum())
    med = n - low - high

    print(f"Train yield: n={n}  mean={y.mean():.1f}  std={y.std():.1f}  "
          f"min={y.min():.0f}  max={y.max():.0f}")
    print(f"  low (<{BIN_EDGES[0]:.0f}):    {low} ({low / n:.1%})")
    print(f"  medium ({BIN_EDGES[0]:.0f}-{BIN_EDGES[1]:.0f}): {med} ({med / n:.1%})")
    print(f"  high (>{BIN_EDGES[1]:.0f}):   {high} ({high / n:.1%})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y, bins=40, color="C0", edgecolor="white", alpha=0.85)
    for edge in BIN_EDGES:
        ax.axvline(edge, color="black", linestyle="--", linewidth=1)
    ymax = ax.get_ylim()[1]
    label_y = 0.92 * ymax
    ax.text(BIN_EDGES[0] / 2, label_y, f"low\nn={low}",
            ha="center", va="top", fontsize=10)
    ax.text(np.mean(BIN_EDGES), label_y, f"medium\nn={med}",
            ha="center", va="top", fontsize=10)
    ax.text((BIN_EDGES[1] + y.max()) / 2, label_y, f"high\nn={high}",
            ha="center", va="top", fontsize=10)

    ax.set_xlabel("Yield (kg/ha)")
    ax.set_ylabel("District-year count")
    ax.set_title(f"Training-set yield distribution (n = {n})")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_png.with_suffix('.pdf')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.train_csv, args.output)
