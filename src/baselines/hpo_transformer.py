"""
Optuna HPO for the Transformer encoder baseline.

Usage:
    python hpo_transformer.py --config config/transformer_with_soil.yaml
    python hpo_transformer.py --config config/transformer_without_soil.yaml

Search space (from action_items.md §2.2):
    d_model:         {32, 48, 64, 96}
    nhead:           {2, 4, 8}  (constrained: d_model % nhead == 0)
    num_layers:      1–3
    dim_feedforward: {64, 96, 128, 192, 256}
    dropout:         0.2–0.5
    n_fc_layers:     2–4
    fc_layer_size:   32–256, step 16
    fc_dropout_prob: 0.1–0.5
    learning_rate:   1e-5 – 5e-3  (loguniform)
    weight_decay:    1e-6 – 1e-3  (loguniform)
    batch_size:      {16, 32, 64}

Loss: MSE only (L2).  Early stopping on validation MSE, patience 15.
500 trials, SQLite storage.
"""

import os
import sys
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import optuna

_project_src = os.path.join(os.path.dirname(__file__), '..')
_rnn_dir = os.path.join(_project_src, 'rnn')
for _p in (_project_src, _rnn_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rnn.dataset import YieldPredictionDataset
from rnn.training_utils import EarlyStopping, train_epoch, validate_epoch
from transformer_model import TransformerEncoder


def objective(trial, config):
    use_pruning = config.get('optimization', {}).get('pruning', True)
    use_soil = config['model']['use_soil_features']

    # -- 1. Sample hyperparameters --
    d_model = trial.suggest_categorical('d_model', [32, 48, 64, 96])
    valid_nheads = [h for h in [2, 4, 8] if d_model % h == 0]
    nhead = trial.suggest_categorical('nhead', valid_nheads)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [64, 96, 128, 192, 256])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)

    n_fc_layers = trial.suggest_int('n_fc_layers', 2, 4)
    fc_hidden_dims = [
        trial.suggest_int(f'fc_layer_size_{i}', 32, 256, step=16)
        for i in range(n_fc_layers)
    ]
    fc_dropout_prob = trial.suggest_float('fc_dropout_prob', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # -- 2. Load data and create dataloaders --
    base_path = config['data']['base_path']
    train_path = os.path.join(base_path, config['data']['train_file'])
    val_path = os.path.join(base_path, config['data']['val_file'])

    train_dataset = YieldPredictionDataset(csv_file=train_path, soil_features_available=use_soil)
    val_dataset = YieldPredictionDataset(csv_file=val_path, soil_features_available=use_soil)

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty")

    sample = train_dataset[0]
    input_features = sample[0].shape[-1]

    flat_soil_dim = None
    if use_soil:
        soil_tensor = sample[2]
        flat_soil_dim = soil_tensor.shape[0] * soil_tensor.shape[1]

    dl_cfg = config['dataloader']
    prefetch = dl_cfg['prefetch_factor']
    if prefetch == 'None':
        prefetch = None
    common = {
        'batch_size': batch_size,
        'num_workers': dl_cfg['num_workers'],
        'pin_memory': dl_cfg['pin_memory'],
        'persistent_workers': dl_cfg['persistent_workers'],
        'prefetch_factor': prefetch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=dl_cfg['drop_last_train'], **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)

    # -- 3. Build model, optimizer, loss --
    model = TransformerEncoder(
        input_features=input_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        fc_hidden_dims=fc_hidden_dims,
        fc_dropout_prob=fc_dropout_prob,
        use_soil_features=use_soil,
        flat_soil_dim=flat_soil_dim,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # -- 4. Training loop --
    n_epochs = config['training']['n_epochs']
    patience = config['training']['early_stopping_patience']
    early_stopping = EarlyStopping(patience=patience, verbose=False, hpo=True)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        trial.report(val_loss, epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break

        if use_pruning and trial.should_prune():
            raise optuna.TrialPruned()

    return early_stopping.val_loss_min


def run_hpo(config):
    hpo_dir = config['optimization']['hpo_dir']
    os.makedirs(hpo_dir, exist_ok=True)

    soil_suffix = 'with_soil' if config['model']['use_soil_features'] else 'without_soil'
    study_name = f"{config['optimization']['study_name']}_{soil_suffix}"
    storage_path = f"sqlite:///{os.path.join(hpo_dir, f'{study_name}.db')}"

    config_yaml_path = os.path.join(hpo_dir, f'{study_name}_config.yaml')
    with open(config_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        direction='minimize',
        load_if_exists=True,
    )

    study.optimize(
        partial(objective, config=config),
        n_trials=config['optimization']['n_trials'],
        timeout=config['optimization'].get('timeout_hours', 8) * 3600,
    )

    print("Best trial:")
    best = study.best_trial
    print(f"  Value: {best.value}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    best_params_path = os.path.join(hpo_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w') as f:
        yaml.dump(best.params, f, default_flow_style=False, sort_keys=False)
    print(f"Best hyperparameters saved to {best_params_path}")


def main():
    parser = argparse.ArgumentParser(description='HPO for Transformer encoder baseline')
    parser.add_argument('--config', default='config/transformer_with_soil.yaml', help='Config file path')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Number of Optuna trials (overrides config value)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config.setdefault('optimization', {})
    config['optimization'].setdefault('study_name', 'transformer_baseline')
    config['optimization'].setdefault('hpo_dir', '../../experiments/hpo/transformer')
    config['optimization'].setdefault('n_trials', 500)
    config['optimization'].setdefault('timeout_hours', 8)

    if args.n_trials is not None:
        config['optimization']['n_trials'] = args.n_trials

    run_hpo(config)


if __name__ == '__main__':
    main()
