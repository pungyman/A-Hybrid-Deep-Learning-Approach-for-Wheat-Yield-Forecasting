"""
Generate full architecture specification table from a trained model (metadata YAML + .pt).
Outputs CSV and LaTeX with Layer, Type, Input Dim, Output Dim, Kernel Size, Activation, Dropout, Notes.
Run from project root: python -m src.analysis.architecture_table --metadata path/to/model_metadata.yaml
"""
import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from pathlib import Path

# Allow importing from src/rnn when run as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'rnn'))
from models import CNN_RNN


def get_layer_spec(name: str, module: torch.nn.Module) -> dict:
    """Extract spec from a single module."""
    row = {'Layer': name, 'Type': type(module).__name__, 'Input Dim': '', 'Output Dim': '', 'Kernel Size': '', 'Stride': '', 'Activation': '', 'Dropout': '', 'Notes': ''}
    if isinstance(module, torch.nn.Linear):
        row['Input Dim'] = module.in_features
        row['Output Dim'] = module.out_features
    elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU)):
        row['Input Dim'] = module.input_size
        row['Output Dim'] = module.hidden_size * (2 if module.bidirectional else 1)
        row['Notes'] = f'layers={module.num_layers}, bidirectional={getattr(module, "bidirectional", False)}'
    elif isinstance(module, torch.nn.Conv1d):
        row['Input Dim'] = module.in_channels
        row['Output Dim'] = module.out_channels
        row['Kernel Size'] = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
        row['Stride'] = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    elif isinstance(module, torch.nn.BatchNorm1d):
        row['Input Dim'] = module.num_features
        row['Output Dim'] = module.num_features
    elif isinstance(module, torch.nn.LayerNorm):
        row['Input Dim'] = module.normalized_shape[0]
        row['Output Dim'] = module.normalized_shape[0]
    elif isinstance(module, torch.nn.Dropout):
        row['Dropout'] = module.p
    elif isinstance(module, torch.nn.ReLU):
        row['Activation'] = 'ReLU'
    elif isinstance(module, torch.nn.AdaptiveMaxPool1d):
        row['Output Dim'] = module.output_size if isinstance(module.output_size, int) else module.output_size[0]
    elif isinstance(module, torch.nn.Module) and hasattr(module, '__class__'):
        # Attention etc.: try to get dimensions from submodules or leave Notes
        if 'attention' in name.lower() or 'W_q' in name or 'W_k' in name or 'W_v' in name:
            for child_name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    row['Notes'] = f'{child_name}: {child.in_features}->{child.out_features}'
                    break
    return row


def build_model_from_metadata(metadata_path: str):
    """Load config from metadata, build model (no weights needed for spec)."""
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    config = metadata['config']
    model_dir = os.path.dirname(metadata_path)
    base_path = config['data']['base_path']
    if not os.path.isabs(base_path):
        base_path = os.path.normpath(os.path.join(model_dir, base_path))
    test_path = os.path.join(base_path, config['data']['test_file'])
    if not os.path.exists(test_path):
        # Fallback: use config only; input_dim from paper (13 temporal features)
        input_dim = 13
        soil_in_channels = 8 if config['model']['use_soil_features'] else None
    else:
        import ast
        df = pd.read_csv(test_path, nrows=1)
        seq = ast.literal_eval(df['feature_sequence'].iloc[0])
        input_dim = len(seq[0])
        soil_in_channels = None
        if config['model']['use_soil_features'] and 'soil_features' in df.columns:
            soil = ast.literal_eval(df['soil_features'].iloc[0])
            import numpy as np
            soil_in_channels = np.array(soil).shape[0]
    mp = config['model']
    mp['soil_in_channels'] = mp.get('soil_in_channels') or soil_in_channels
    model = CNN_RNN(
        input_dim=input_dim,
        rnn_hidden_dim=mp['rnn_hidden_dim'],
        rnn_n_layers=mp['rnn_n_layers'],
        rnn_dropout_prob=mp['rnn_dropout_prob'],
        fc_dropout_prob=mp['fc_dropout_prob'],
        rnn_type=mp['rnn_type'],
        fc_hidden_dims=mp.get('fc_hidden_dims'),
        use_temporal_attention=mp['use_temporal_attention'],
        attention_type=mp.get('attention_type', 'bahdanau'),
        attention_dim=mp.get('attention_dim'),
        attention_use_bias=mp.get('attention_use_bias', True),
        use_layer_norm=mp.get('use_layer_norm', False),
        bidirectional=mp.get('bidirectional', False),
        soil_in_channels=mp.get('soil_in_channels'),
        soil_out_channels=mp.get('soil_out_channels'),
        soil_kernel_size=mp.get('soil_kernel_size'),
        soil_n_conv_blocks=mp.get('soil_n_conv_blocks', 1),
    )
    return model, config


def run_architecture_table(metadata_path: str, output_dir: str) -> None:
    model, config = build_model_from_metadata(metadata_path)
    rows = []
    for name, module in model.named_modules():
        if name == '':
            continue
        row = get_layer_spec(name, module)
        if row['Type'] != 'CNN_RNN' and (row['Input Dim'] or row['Output Dim'] or row['Type'] in ('Linear', 'LSTM', 'GRU', 'Conv1d', 'Dropout', 'ReLU')):
            rows.append(row)
    # Add total parameters
    n_params = sum(p.numel() for p in model.parameters())
    rows.append({'Layer': 'Total', 'Type': '—', 'Input Dim': '', 'Output Dim': n_params, 'Kernel Size': '', 'Stride': '', 'Activation': '', 'Dropout': '', 'Notes': 'trainable parameters'})
    table = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, 'architecture_spec.csv')
    out_tex = os.path.join(output_dir, 'architecture_spec.tex')
    table.to_csv(out_csv, index=False)
    with open(out_tex, 'w') as f:
        f.write(table.to_latex(index=False))
    print(f'Saved {out_csv} and {out_tex}')
    print(table.to_string())


def main():
    parser = argparse.ArgumentParser(description='Generate architecture specification table.')
    parser.add_argument('--metadata', required=True, help='Path to model metadata YAML (e.g. .../model_metadata.yaml)')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as metadata)')
    args = parser.parse_args()
    output_dir = args.output_dir or str(Path(args.metadata).parent)
    run_architecture_table(args.metadata, output_dir)


if __name__ == '__main__':
    main()
