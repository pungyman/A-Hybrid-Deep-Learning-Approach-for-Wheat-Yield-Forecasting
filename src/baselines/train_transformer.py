"""
Training driver for the Transformer encoder baseline (Wang 2024 CompAG-inspired).

Usage:
    python train_transformer.py --config config/transformer_with_soil.yaml
    python train_transformer.py --config config/transformer_without_soil.yaml

Follows the shared protocol from paper_revision/action_items.md §2.0 / §2.2:
- L2 (MSE) loss only for training and early stopping
- AdamW optimizer (Transformers benefit from decoupled weight decay)
- 5-seed evaluation, best seed checkpoint kept
- Artifacts: .pt model, metadata YAML, test results CSV, evaluation plots
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

_project_src = os.path.join(os.path.dirname(__file__), '..')
_rnn_dir = os.path.join(_project_src, 'rnn')
for _p in (_project_src, _rnn_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rnn.dataset import YieldPredictionDataset
from rnn.training_utils import (
    load_config,
    create_dataloaders,
    train_model,
    evaluate_model,
    set_random_seed,
)
from transformer_model import TransformerEncoder

try:
    from analysis.plot_style import apply_style
    apply_style()
except ImportError:
    pass


def convert_numpy_to_native(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data


def setup_transformer_model(config, input_features, flat_soil_dim, device):
    """Instantiate TransformerEncoder, loss, and optimizer from config."""
    mc = config['model']
    tc = config['training']

    model = TransformerEncoder(
        input_features=input_features,
        d_model=mc['d_model'],
        nhead=mc['nhead'],
        num_layers=mc['num_layers'],
        dim_feedforward=mc['dim_feedforward'],
        dropout=mc['dropout'],
        fc_hidden_dims=mc['fc_hidden_dims'],
        fc_dropout_prob=mc['fc_dropout_prob'],
        use_soil_features=mc['use_soil_features'],
        flat_soil_dim=flat_soil_dim,
    )
    model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc['learning_rate'],
        weight_decay=tc['weight_decay'],
    )
    return model, criterion, optimizer


def main(config_path: str):
    try:
        print("Loading configuration...")
        config = load_config(config_path)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        # --- datasets ---
        print("Loading datasets...")
        base_path = config['data']['base_path']
        soil_on = config['model']['use_soil_features']

        train_path = os.path.join(base_path, config['data']['train_file'])
        val_path = os.path.join(base_path, config['data']['val_file'])
        test_path = os.path.join(base_path, config['data']['test_file'])

        train_dataset = YieldPredictionDataset(csv_file=train_path, soil_features_available=soil_on)
        val_dataset = YieldPredictionDataset(csv_file=val_path, soil_features_available=soil_on)
        test_dataset = YieldPredictionDataset(csv_file=test_path, inference=True, soil_features_available=soil_on)

        if len(train_dataset) == 0:
            raise ValueError("Train dataset is empty")

        sample = train_dataset[0]
        input_features = sample[0].shape[-1]  # F=13

        flat_soil_dim = None
        if soil_on:
            soil_tensor = sample[2]
            flat_soil_dim = soil_tensor.shape[0] * soil_tensor.shape[1]  # 8*6 = 48

        # --- multi-seed loop ---
        repro = config['reproducibility']
        num_runs = repro['num_runs']
        base_seed = repro['seed']

        if num_runs > 1:
            print(f"Starting {num_runs} training runs with different seeds.")
            np.random.seed(base_seed)
            seeds = [int(s) for s in np.random.randint(0, 10000, size=num_runs)]
        else:
            seeds = [base_seed]

        all_test_metrics = []
        best_overall_test_r2 = -float('inf')
        best_model_info = {}
        saved_model_paths = []

        for run_idx, seed in enumerate(seeds):
            print(f"\n--- Starting Run {run_idx + 1}/{num_runs} with seed {seed} ---")
            set_random_seed(seed, repro['deterministic'])

            train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)

            print("Setting up model...")
            model, criterion, optimizer = setup_transformer_model(
                config, input_features, flat_soil_dim, device
            )

            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model has {total_params:,} parameters")

            train_losses, val_losses, model_save_path = train_model(
                model, criterion, optimizer, train_loader, val_loader, config, device
            )
            saved_model_paths.append(model_save_path)

            model.load_state_dict(torch.load(model_save_path, map_location=device))

            output_dir = os.path.dirname(model_save_path)
            model_name = os.path.splitext(os.path.basename(model_save_path))[0]

            test_metrics = evaluate_model(model, test_dataset, device, output_dir, model_name)
            all_test_metrics.append(test_metrics)

            current_test_r2 = test_metrics['r2_score']
            if current_test_r2 > best_overall_test_r2:
                best_overall_test_r2 = current_test_r2
                best_model_info = {
                    'config': config,
                    'best_val_loss': min(val_losses),
                    'final_epoch': len(train_losses),
                    'total_parameters': total_params,
                    'random_seed': seed,
                    'test_metrics': test_metrics,
                    'model_save_path': model_save_path,
                }

        # --- post-processing ---
        if num_runs > 1:
            print("\n--- All runs completed. Processing results. ---")

            best_model_path = best_model_info['model_save_path']
            for path in saved_model_paths:
                if path != best_model_path:
                    try:
                        os.remove(path)
                        output_dir = os.path.dirname(path)
                        model_name = os.path.splitext(os.path.basename(path))[0]
                        for ext in ['_training_history.png', '_evaluation_plots.png', '_test_results.csv']:
                            plot_path = os.path.join(output_dir, f"{model_name}{ext}")
                            if os.path.exists(plot_path):
                                os.remove(plot_path)
                    except OSError as e:
                        print(f"Error removing file {path}: {e}")

            metrics_df = pd.DataFrame(all_test_metrics)
            mean_metrics = metrics_df.mean().to_dict()
            std_metrics = metrics_df.std().to_dict()

            best_model_info['mean_test_metrics'] = mean_metrics
            best_model_info['std_test_metrics'] = std_metrics
            best_model_info['all_test_metrics'] = all_test_metrics
            best_model_info['seeds'] = seeds

        # --- metadata ---
        output_dir = os.path.dirname(best_model_info['model_save_path'])
        model_name = os.path.splitext(os.path.basename(best_model_info['model_save_path']))[0]

        metadata = {
            'config': best_model_info['config'],
            'best_val_loss': best_model_info['best_val_loss'],
            'final_epoch': best_model_info['final_epoch'],
            'total_parameters': best_model_info['total_parameters'],
            'random_seed': best_model_info['random_seed'],
            'test_metrics': best_model_info['test_metrics'],
        }
        if num_runs > 1:
            metadata['mean_test_metrics'] = best_model_info['mean_test_metrics']
            metadata['std_test_metrics'] = best_model_info['std_test_metrics']
            metadata['seeds'] = best_model_info['seeds']

        metadata = convert_numpy_to_native(metadata)

        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, sort_keys=False)

        print(f"Metadata for the best model saved to: {metadata_path}")
        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer encoder baseline for yield prediction')
    parser.add_argument('--config', default='config/transformer_with_soil.yaml', help='Config file path')
    args = parser.parse_args()
    main(args.config)
