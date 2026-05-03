import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import optuna
from functools import partial

# Add the parent directory (src/rnn) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import YieldPredictionDataset
from models import CNN_RNN
from training_utils import EarlyStopping, train_epoch, validate_epoch


def objective(trial, config):
    use_pruning = config['optimization']['pruning']
    
    # -- 1. Load base model configuration and weights --
    metadata_path = config['model_to_finetune']['path_to_metadata']
    with open(metadata_path, 'r') as f:
        model_metadata = yaml.safe_load(f)
    
    model_config = model_metadata['config']['model']
    
    # Construct model path from metadata path
    model_dir = os.path.dirname(metadata_path)
    model_filename = os.path.basename(metadata_path).replace('_metadata.yaml', '.pt')
    model_path = os.path.join(model_dir, model_filename)

    # -- 2. Sample hyperparameters for fine-tuning --
    hparam_search_space = config['optimization']['hparam_search_space']
    
    hyperparameters = {
        'rnn_dropout_prob': trial.suggest_float('rnn_dropout_prob', hparam_search_space['rnn_dropout_prob']['low'], hparam_search_space['rnn_dropout_prob']['high']),
        'fc_dropout_prob': trial.suggest_float('fc_dropout_prob', hparam_search_space['fc_dropout_prob']['low'], hparam_search_space['fc_dropout_prob']['high']),
        'learning_rate': trial.suggest_float('learning_rate', float(hparam_search_space['learning_rate']['low']), float(hparam_search_space['learning_rate']['high']), log=True),
        'weight_decay': trial.suggest_float('weight_decay', float(hparam_search_space['weight_decay']['low']), float(hparam_search_space['weight_decay']['high']), log=True),
        'batch_size': trial.suggest_categorical('batch_size', hparam_search_space['batch_size']['choices']),
        'huber_delta': trial.suggest_float('huber_delta', hparam_search_space['huber_delta']['low'], hparam_search_space['huber_delta']['high']),
    }

    # -- 3. Load data and create dataloaders --
    base_path = config['data']['base_path']
    train_path = os.path.join(base_path, config['data']['train_file'])
    val_path = os.path.join(base_path, config['data']['val_file'])
    
    state_for_finetuning = config['data']['state_for_finetuning']
    use_soil_features = model_config['use_soil_features']

    train_dataset = YieldPredictionDataset(
        csv_file=train_path, 
        soil_features_available=use_soil_features, 
        ft=True, 
        state=state_for_finetuning
    )
    val_dataset = YieldPredictionDataset(
        csv_file=val_path, 
        soil_features_available=use_soil_features, 
        ft=True, 
        state=state_for_finetuning
    )

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty for state: {state_for_finetuning}")
        
    sample = train_dataset[0]
    sample_sequence = sample[0]
    input_dim = sample_sequence.shape[-1]
    
    soil_in_channels = None
    if use_soil_features is True:
        soil_tensor = sample[2]
        soil_in_channels = soil_tensor.shape[0]

    dataloader_config = config['dataloader']
    prefetch_factor = dataloader_config['prefetch_factor']
    if prefetch_factor == 'None' or prefetch_factor is None:
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

    # -- 4. Initialize model, optimizer, and loss function --
    model = CNN_RNN(
        input_dim=input_dim,
        rnn_hidden_dim=model_config['rnn_hidden_dim'],
        rnn_n_layers=model_config['rnn_n_layers'],
        rnn_dropout_prob=hyperparameters['rnn_dropout_prob'],
        fc_dropout_prob=hyperparameters['fc_dropout_prob'],
        rnn_type=model_config['rnn_type'],
        fc_hidden_dims=model_config['fc_hidden_dims'],
        use_temporal_attention=model_config['use_temporal_attention'],
        attention_type=model_config['attention_type'],
        attention_dim=model_config['attention_dim'],
        attention_use_bias=model_config['attention_use_bias'],
        use_layer_norm=model_config['use_layer_norm'],
        soil_in_channels=soil_in_channels,
        soil_out_channels=model_config['soil_out_channels'],
        soil_kernel_size=model_config['soil_kernel_size'],
        soil_n_conv_blocks=model_config['soil_n_conv_blocks']
    )
    
    # Load pretrained weights
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hyperparameters['learning_rate'], 
        weight_decay=hyperparameters['weight_decay']
    )

    criterion = nn.HuberLoss(delta=hyperparameters['huber_delta'])

    validation_loss_type = config['training']['validation_loss']
    if validation_loss_type == 'l1':
        val_criterion = nn.L1Loss()
    elif validation_loss_type == 'mse':
        val_criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported validation loss: {validation_loss_type}")
    
    # -- 5. Training loop --
    n_epochs = config['training']['n_epochs']
    patience = config['training']['early_stopping_patience']
    
    early_stopping = EarlyStopping(patience=patience, verbose=False, hpo=True)

    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
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
    
    # Save the config YAML file
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

    # Save the results of the study
    results = {
        'model_finetuned': config['model_to_finetune']['path_to_metadata'],
        'state_finetuned_on': config['data']['state_for_finetuning'],
        'best_hyperparameters': trial.params,
        'best_validation_loss': trial.value
    }

    results_path = os.path.join(hpo_dir, f'{study_name}_results.yaml')
    with open(results_path, 'w') as file:
        yaml.dump(results, file, default_flow_style=False, sort_keys=False)
    print(f"HPO results saved to {results_path}")


if __name__ == '__main__':
    with open('hpo_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    run_hpo(config)
