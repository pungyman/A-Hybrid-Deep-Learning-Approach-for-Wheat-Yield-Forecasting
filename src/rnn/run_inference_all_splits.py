"""
Run trained model inference on train, validation, and test sets and save results.
Used to generate a single CSV with a 'split' column for extreme-year and other analyses.
Run from project root or from src/rnn with appropriate paths.

Usage:
  python -m src.rnn.run_inference_all_splits --config src/rnn/training_config.yaml --model-dir saved_models/cnn_bilstm_self_attn
  # Or from src/rnn:
  python run_inference_all_splits.py --config training_config.yaml --model-dir ../saved_models/cnn_bilstm_self_attn
"""
import os
import sys
import argparse
import yaml
import torch
import pandas as pd

# Allow running from project root or from src/rnn
if __name__ == '__main__' and os.path.basename(os.getcwd()) == 'rnn':
    sys.path.insert(0, os.getcwd())
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset import YieldPredictionDataset
from models import CNN_RNN
from training_utils import load_config, setup_model_and_training, run_inference_on_dataset


def main():
    parser = argparse.ArgumentParser(description='Run inference on train, val, and test sets.')
    parser.add_argument('--config', default='training_config.yaml', help='Path to training config YAML')
    parser.add_argument('--model-dir', required=True, help='Directory containing the .pt model and optionally _metadata.yaml')
    parser.add_argument('--output-csv', default=None, help='Path for merged all_splits_results.csv (default: <model_dir>/all_splits_results.csv)')
    args = parser.parse_args()

    config_path = args.config
    model_dir = os.path.abspath(args.model_dir)
    if args.output_csv:
        merged_path = os.path.abspath(args.output_csv)
    else:
        merged_path = os.path.join(model_dir, 'all_splits_results.csv')

    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    base_path = os.path.normpath(os.path.join(config_dir, config['data']['base_path']))
    soil_available = config['model']['use_soil_features']

    train_path = os.path.join(base_path, config['data']['train_file'])
    val_path = os.path.join(base_path, config['data']['val_file'])
    test_path = os.path.join(base_path, config['data']['test_file'])

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Dataset not found: {p}')

    # Load datasets with inference=True so we get district_id and sowing_year
    train_ds = YieldPredictionDataset(csv_file=train_path, inference=True, soil_features_available=soil_available)
    val_ds = YieldPredictionDataset(csv_file=val_path, inference=True, soil_features_available=soil_available)
    test_ds = YieldPredictionDataset(csv_file=test_path, inference=True, soil_features_available=soil_available)

    sample = train_ds[0]
    # With inference=True: (district_id, sowing_year, sequence, lagged yield feature (past_yield), [soil], yield)
    input_dim = sample[2].shape[-1]
    if soil_available:
        soil_tensor = train_ds.soil_features[0]
        config['model']['soil_in_channels'] = soil_tensor.shape[0]

    model, _, _ = setup_model_and_training(config, input_dim, device)
    # Find .pt in model_dir
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError(f'No .pt file found in {model_dir}')
    model_path = os.path.join(model_dir, pt_files[0])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dfs = []
    for split_name, ds in [('train', train_ds), ('val', val_ds), ('test', test_ds)]:
        out_path = os.path.join(model_dir, f'{split_name}_results.csv')
        run_inference_on_dataset(model, ds, device, out_path, save_plots=False)
        df = pd.read_csv(out_path)
        df['split'] = split_name
        dfs.append(df)
        print(f'Saved {split_name} results to {out_path}')

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(merged_path, index=False)
    print(f'Merged results saved to {merged_path}')


if __name__ == '__main__':
    main()
