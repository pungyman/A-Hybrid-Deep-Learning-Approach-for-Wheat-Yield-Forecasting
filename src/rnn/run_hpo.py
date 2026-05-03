import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import optuna
from functools import partial
from dataset import YieldPredictionDataset
from models import CNN_RNN
from training_utils import EarlyStopping, train_epoch, validate_epoch


def objective(trial, config):
    use_pruning = config['optimization'].get('pruning', True)
    
    # -- 1. Sample hyperparameters --
    hparam_search_space = config['optimization']['hparam_search_space']
    
    hyperparameters = {
        'use_soil_features': trial.suggest_categorical('use_soil_features', hparam_search_space['use_soil_features']['choices']),
        'rnn_type': trial.suggest_categorical('rnn_type', hparam_search_space['rnn_type']['choices']),
        'bidirectional': trial.suggest_categorical('bidirectional', hparam_search_space['bidirectional']['choices']),
        'rnn_n_layers': trial.suggest_int('rnn_n_layers', hparam_search_space['rnn_n_layers']['low'], hparam_search_space['rnn_n_layers']['high']),
        'rnn_hidden_dim': trial.suggest_int('rnn_hidden_dim', hparam_search_space['rnn_hidden_dim']['low'], hparam_search_space['rnn_hidden_dim']['high'], step=hparam_search_space['rnn_hidden_dim']['step']),
        'rnn_dropout_prob': trial.suggest_float('rnn_dropout_prob', hparam_search_space['rnn_dropout_prob']['low'], hparam_search_space['rnn_dropout_prob']['high']),
        'fc_dropout_prob': trial.suggest_float('fc_dropout_prob', hparam_search_space['fc_dropout_prob']['low'], hparam_search_space['fc_dropout_prob']['high']),
        'learning_rate': trial.suggest_float('learning_rate', float(hparam_search_space['learning_rate']['low']), float(hparam_search_space['learning_rate']['high']), log=True),
        'weight_decay': trial.suggest_float('weight_decay', float(hparam_search_space['weight_decay']['low']), float(hparam_search_space['weight_decay']['high']), log=True),
        'batch_size': trial.suggest_categorical('batch_size', hparam_search_space['batch_size']['choices']),
        'n_fc_layers': trial.suggest_int('n_fc_layers', hparam_search_space['n_fc_layers']['low'], hparam_search_space['n_fc_layers']['high']),
        'use_temporal_attention': trial.suggest_categorical('use_temporal_attention', hparam_search_space['use_temporal_attention']['choices']),
        'use_layer_norm': trial.suggest_categorical('use_layer_norm', hparam_search_space['use_layer_norm']['choices']),
        'huber_delta': trial.suggest_float('huber_delta', hparam_search_space['huber_delta']['low'], hparam_search_space['huber_delta']['high']),
    }
    
    # Sample FC layer sizes
    fc_hidden_dims = []
    for i in range(hyperparameters['n_fc_layers']):
        size = trial.suggest_int(f'fc_layer_size_{i}', hparam_search_space['fc_layer_size']['low'], hparam_search_space['fc_layer_size']['high'], step=hparam_search_space['fc_layer_size']['step'])
        fc_hidden_dims.append(size)
    
    if hyperparameters['use_temporal_attention'] is True:
        hyperparameters['attention_type'] = trial.suggest_categorical('attention_type', hparam_search_space['attention_type']['choices'])
        hyperparameters['attention_dim'] = trial.suggest_int('attention_dim', hparam_search_space['attention_dim']['low'], hparam_search_space['attention_dim']['high'], step=hparam_search_space['attention_dim']['step'])
        hyperparameters['attention_use_bias'] = trial.suggest_categorical('attention_use_bias', hparam_search_space['attention_use_bias']['choices'])
    else:
        hyperparameters['attention_type'] = 'bahdanau'  # default
        hyperparameters['attention_dim'] = None
        hyperparameters['attention_use_bias'] = False

    if hyperparameters['use_soil_features'] is True:
        hyperparameters['soil_out_channels'] = trial.suggest_int('soil_out_channels', hparam_search_space['soil_out_channels']['low'], hparam_search_space['soil_out_channels']['high'], step=hparam_search_space['soil_out_channels']['step'])
        hyperparameters['soil_kernel_size'] = trial.suggest_categorical('soil_kernel_size', hparam_search_space['soil_kernel_size']['choices'])
        hyperparameters['soil_n_conv_blocks'] = trial.suggest_int('soil_n_conv_blocks', hparam_search_space['soil_n_conv_blocks']['low'], hparam_search_space['soil_n_conv_blocks']['high'])
    else:
        hyperparameters['soil_out_channels'] = None
        hyperparameters['soil_kernel_size'] = None
        hyperparameters['soil_n_conv_blocks'] = 1  # default
    
    # -- 2. Load data and create dataloaders --
    base_path = config['data']['base_path']
    train_path = os.path.join(base_path, config['data']['train_file'])
    val_path = os.path.join(base_path, config['data']['val_file'])
    
    use_soil_features = hyperparameters['use_soil_features']
    train_dataset = YieldPredictionDataset(csv_file=train_path, soil_features_available=use_soil_features)
    val_dataset = YieldPredictionDataset(csv_file=val_path, soil_features_available=use_soil_features)

    if len(train_dataset) == 0:
        raise ValueError("Train dataset is empty")
        
    sample = train_dataset[0]
    sample_sequence = sample[0]
    input_dim = sample_sequence.shape[-1]
    
    soil_in_channels = None
    if use_soil_features:
        soil_tensor = sample[2]
        soil_in_channels = soil_tensor.shape[0]

    dataloader_config = config['dataloader']
    prefetch_factor = dataloader_config['prefetch_factor']
    if prefetch_factor == 'None':
        prefetch_factor = None
    common_args = {
        'batch_size': hyperparameters['batch_size'],
        'num_workers': dataloader_config['num_workers'],
        'pin_memory': dataloader_config['pin_memory'],
        'persistent_workers': dataloader_config['persistent_workers'],
        'prefetch_factor': prefetch_factor,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=dataloader_config['drop_last_train'], **common_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_args)

    # -- 3. Initialize model, optimizer, and loss function --
    model = CNN_RNN(
        input_dim=input_dim,
        rnn_hidden_dim=hyperparameters['rnn_hidden_dim'],
        rnn_n_layers=hyperparameters['rnn_n_layers'],
        rnn_dropout_prob=hyperparameters['rnn_dropout_prob'],
        fc_dropout_prob=hyperparameters['fc_dropout_prob'],
        rnn_type=hyperparameters['rnn_type'],
        fc_hidden_dims=fc_hidden_dims,
        use_temporal_attention=hyperparameters['use_temporal_attention'],
        attention_type=hyperparameters['attention_type'],
        attention_dim=hyperparameters['attention_dim'],
        attention_use_bias=hyperparameters['attention_use_bias'],
        use_layer_norm=hyperparameters['use_layer_norm'],
        bidirectional=hyperparameters['bidirectional'],
        soil_in_channels=soil_in_channels,
        soil_out_channels=hyperparameters['soil_out_channels'],
        soil_kernel_size=hyperparameters['soil_kernel_size'],
        soil_n_conv_blocks=hyperparameters['soil_n_conv_blocks']
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparameters['learning_rate'], 
        weight_decay=hyperparameters['weight_decay']
    )

    criterion = nn.HuberLoss(delta=hyperparameters['huber_delta'])

    validation_loss_type = config['training'].get('validation_loss', 'l1')
    if validation_loss_type == 'l1':
        val_criterion = nn.L1Loss()
    elif validation_loss_type == 'mse':
        val_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported validation loss: {validation_loss_type}")
    
    # -- 4. Training loop --
    n_epochs = config['training']['n_epochs']
    patience = config['training']['early_stopping_patience']
    
    early_stopping = EarlyStopping(patience=patience, verbose=False, hpo=True)

    for epoch in range(1, n_epochs + 1):
        # Train epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate epoch
        val_loss = validate_epoch(model, val_loader, val_criterion, device)
        
        trial.report(val_loss, epoch)

        early_stopping(val_loss, model)
        if early_stopping.early_stop is True:
            break

        if use_pruning and trial.should_prune():
            raise optuna.TrialPruned()
            
    return early_stopping.val_loss_min


def run_hpo(config):
    hpo_dir = config['optimization']['hpo_dir']
    os.makedirs(hpo_dir, exist_ok=True)
    
    study_name = config['optimization']['study_name']
    storage_path = f"sqlite:///{os.path.join(hpo_dir, f'{study_name}.db')}"
    
    # Save the config YAML file alongside the database for reproducibility
    config_yaml_path = os.path.join(hpo_dir, f'{study_name}_config.yaml')
    with open(config_yaml_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    study = optuna.create_study(study_name=study_name, storage=storage_path, direction='minimize', load_if_exists=True)
    
    objective_with_config = partial(objective, config=config)
    
    study.optimize(
        objective_with_config, 
        n_trials=config['optimization']['n_trials'], 
        timeout=config['optimization']['timeout_hours'] * 3600
    )
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best hyperparameters
    best_params_path = os.path.join(hpo_dir, f'{study_name}_best_params.yaml')
    with open(best_params_path, 'w') as file:
        yaml.dump(trial.params, file, default_flow_style=False, sort_keys=False)
    print(f"Best hyperparameters saved to {best_params_path}")


if __name__ == '__main__':
    with open('hparam_search_space.yaml', 'r') as file:
        config = yaml.safe_load(file)
    run_hpo(config)