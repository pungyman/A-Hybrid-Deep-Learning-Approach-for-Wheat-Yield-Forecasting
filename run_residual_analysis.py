"""
Run residual analysis (by state), limitations analysis, and yield-class distribution.
Uses the test results CSV and train/val/test splits paths below.
Run from project root: python run_residual_analysis.py
"""
import os
import sys

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Paths
RESULTS_CSV = os.path.join(
    ROOT,
    "src", "rnn", "saved_models", "cnn_bilstm_self_attn",
    "model_20251018_064239_test_results.csv",
)
OUTPUT_DIR = os.path.join(ROOT, "analysis_output", "residual_analysis")
TRAIN_CSV = os.path.join(ROOT, "data", "datasets", "final", "train.csv")
VAL_CSV = os.path.join(ROOT, "data", "datasets", "final", "val.csv")
TEST_CSV = os.path.join(ROOT, "data", "datasets", "final", "test.csv")

if __name__ == "__main__":
    if not os.path.exists(RESULTS_CSV):
        print(f"Results CSV not found: {RESULTS_CSV}")
        sys.exit(1)

    from analysis.residual_analysis import (
        load_results,
        run_residual_analysis,
        run_limitations_analysis,
        yield_class_distribution_table,
    )

    print("Running residual analysis by state...")
    run_residual_analysis(RESULTS_CSV, OUTPUT_DIR)

    df = load_results(RESULTS_CSV)
    print("Running limitations analysis...")
    run_limitations_analysis(df, OUTPUT_DIR)

    print("Building yield-class distribution table...")
    yield_class_distribution_table(TRAIN_CSV, VAL_CSV, TEST_CSV, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
