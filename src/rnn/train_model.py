import os
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
from dataset import YieldPredictionDataset
from training_utils import (
    load_config, 
    create_dataloaders,
    setup_model_and_training, 
    train_model, 
    save_training_plots,
    evaluate_model,
    set_random_seed
)


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
    else:
        return data


def main(config_path: str = 'training_config.yaml'):
    try:
        # 1. Load configuration
        print("Loading configuration...")
        config = load_config(config_path)
        
        # 2. Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        # 3. Load datasets
        print("Loading datasets...")
        base_path = config['data']['base_path']
        soil_features_available = config['model']['use_soil_features']
        
        train_path = os.path.join(base_path, config['data']['train_file'])
        val_path = os.path.join(base_path, config['data']['val_file'])
        test_path = os.path.join(base_path, config['data']['test_file'])
        
        train_dataset = YieldPredictionDataset(csv_file=train_path, soil_features_available=soil_features_available)
        val_dataset = YieldPredictionDataset(csv_file=val_path, soil_features_available=soil_features_available)
        test_dataset = YieldPredictionDataset(csv_file=test_path, inference=True, soil_features_available=soil_features_available)

        if len(train_dataset) == 0:
            raise ValueError("Train dataset is empty")

        # Get input dimension and soil channels from a sample
        sample = train_dataset[0]
        sample_sequence = sample[0]
        input_dim = sample_sequence.shape[-1]
        
        if soil_features_available:
            soil_tensor = sample[2]
            soil_mode = config['model'].get('soil_mode', 'cnn')
            if soil_mode == 'flat':
                config['model']['flat_soil_dim'] = soil_tensor.shape[0] * soil_tensor.shape[1]
            else:
                config['model']['soil_in_channels'] = soil_tensor.shape[0]

        # Handle multiple runs
        reproducibility_config = config['reproducibility']
        num_runs = reproducibility_config['num_runs']
        base_seed = reproducibility_config['seed']

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
            
            # Set random seed for reproducibility for this run
            set_random_seed(seed, reproducibility_config['deterministic'])
            
            # Create dataloaders with the new seed
            train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)
            
            # Setup model and training
            print("Setting up model...")
            model, criterion, optimizer = setup_model_and_training(config, input_dim, device)
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model has {total_params:,} parameters")
            
            # Train model
            train_losses, val_losses, model_save_path = train_model(
                model, criterion, optimizer, train_loader, val_loader, config, device
            )
            saved_model_paths.append(model_save_path)
            
            # Load best model from this run for evaluation
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

        # After all runs, process results
        if num_runs > 1:
            print("\n--- All runs completed. Processing results. ---")
            
            # Clean up non-best models
            best_model_path = best_model_info['model_save_path']
            for path in saved_model_paths:
                if path != best_model_path:
                    try:
                        os.remove(path)
                        # Attempt to remove other files from that run, assuming they are in the same directory
                        output_dir = os.path.dirname(path)
                        model_name = os.path.splitext(os.path.basename(path))[0]
                        for ext in ['_training_history.png', '_evaluation_plots.png', '_test_results.csv']:
                            plot_path = os.path.join(output_dir, f"{model_name}{ext}")
                            if os.path.exists(plot_path):
                                os.remove(plot_path)
                    except OSError as e:
                        print(f"Error removing file {path}: {e}")

            # Calculate mean and std dev of test metrics
            metrics_df = pd.DataFrame(all_test_metrics)
            mean_metrics = metrics_df.mean().to_dict()
            std_metrics = metrics_df.std().to_dict()
            
            best_model_info['mean_test_metrics'] = mean_metrics
            best_model_info['std_test_metrics'] = std_metrics
            best_model_info['all_test_metrics'] = all_test_metrics
            best_model_info['seeds'] = seeds

        # Save plots and metadata for the best model
        output_dir = os.path.dirname(best_model_info['model_save_path'])
        model_name = os.path.splitext(os.path.basename(best_model_info['model_save_path']))[0]

        # Save training plots
        # training_plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
        # save_training_plots(best_model_info['train_losses'], best_model_info['val_losses'], training_plot_path)

        # 8. Save metadata
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

        # Convert numpy types to native python types for clean yaml output
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
    parser = argparse.ArgumentParser(description='Train model for yield prediction')
    parser.add_argument('--config', default='training_config.yaml', help='Config file path')
    args = parser.parse_args()
    main(args.config)