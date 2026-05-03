"""
Regenerate evaluation plots from existing results CSV files with consistent styling.

This script loads results CSV files (containing actual_yield and predicted_yield columns)
and regenerates the evaluation plots using the updated consistent plotting style.

Usage:
    # Regenerate plots from a specific CSV file
    python regenerate_evaluation_plots.py <results_csv_path> [output_plot_path]
    
    # Find and regenerate all evaluation plots in a directory
    python regenerate_evaluation_plots.py --find-all [--directory <dir>]
    
Examples:
    python regenerate_evaluation_plots.py results/test_results.csv
    python regenerate_evaluation_plots.py --find-all
    python regenerate_evaluation_plots.py --find-all --directory src/rnn/saved_models
"""
import os
import sys
import argparse
import glob
import pandas as pd
import numpy as np

# Add src to path to import training_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rnn'))
from training_utils import create_evaluation_plots


def find_results_csvs(search_dir: str = None) -> list:
    """
    Find all results CSV files in the project.
    
    Args:
        search_dir: Directory to search in. If None, searches common locations.
    
    Returns:
        List of paths to results CSV files.
    """
    if search_dir is None:
        # Common locations based on the codebase structure
        search_dirs = [
            os.path.join('src', 'rnn', 'saved_models'),
            'results',
            'output',
        ]
        csv_files = []
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                csv_files.extend(glob.glob(os.path.join(dir_path, '**', '*_results.csv'), recursive=True))
    else:
        csv_files = glob.glob(os.path.join(search_dir, '**', '*_results.csv'), recursive=True)
    
    return sorted(csv_files)


def regenerate_plots_from_csv(csv_path: str, output_path: str = None) -> None:
    """
    Load results CSV and regenerate evaluation plots with consistent styling.
    
    Args:
        csv_path: Path to results CSV file (must contain 'actual_yield' and 'predicted_yield' columns)
        output_path: Optional output path for the plot. If not provided, uses CSV basename + '_evaluation_plots.png'
    """
    # Load CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['actual_yield', 'predicted_yield']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Extract arrays
    y_true = df['actual_yield'].values
    y_pred = df['predicted_yield'].values
    
    # Determine output path
    if output_path is None:
        csv_dir = os.path.dirname(csv_path)
        csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(csv_dir, f"{csv_basename}_evaluation_plots.png")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate plots with consistent styling
    print(f"Loading results from: {csv_path}")
    print(f"Found {len(df)} samples")
    print(f"Regenerating evaluation plots with consistent styling...")
    create_evaluation_plots(y_true, y_pred, output_path)
    print(f"Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate evaluation plots from results CSV with consistent styling"
    )
    parser.add_argument(
        'csv_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to results CSV file (must contain actual_yield and predicted_yield columns)'
    )
    parser.add_argument(
        'output_path',
        type=str,
        nargs='?',
        default=None,
        help='Optional output path for the plot. Default: <csv_basename>_evaluation_plots.png'
    )
    parser.add_argument(
        '--find-all',
        action='store_true',
        help='Find and regenerate all evaluation plots in common directories'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default=None,
        help='Directory to search for results CSV files (used with --find-all)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.find_all:
            # Find all results CSV files
            csv_files = find_results_csvs(args.directory)
            if not csv_files:
                print("No results CSV files found.")
                sys.exit(1)
            
            print(f"Found {len(csv_files)} results CSV file(s):")
            for csv_file in csv_files:
                print(f"  - {csv_file}")
            
            print("\nRegenerating evaluation plots...")
            for csv_file in csv_files:
                try:
                    regenerate_plots_from_csv(csv_file)
                    print()
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}", file=sys.stderr)
                    continue
            
            print("Done!")
        elif args.csv_path:
            regenerate_plots_from_csv(args.csv_path, args.output_path)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
