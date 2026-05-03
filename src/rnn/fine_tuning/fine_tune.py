import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from datetime import datetime

# Add the parent directory (src/rnn) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import YieldPredictionDataset
from models import CNN_RNN
from training_utils import (
    create_dataloaders,
    set_random_seed,
    save_training_plots,
    evaluate_model,
    EarlyStopping,
    train_epoch,
    validate_epoch
)

def main(config_path: str):
    try:
        # 1. Load configuration
        print("Loading configuration...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 2. Set random seed for reproducibility
        reproducibility_config = config['reproducibility']
        set_random_seed(reproducibility_config['seed'], reproducibility_config['deterministic'])
        
        # 3. Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 4. Load pre-trained model configuration and locate weights
        print("Loading pre-trained model configuration...")
        metadata_path = config['model_to_finetune']['path_to_metadata']
        with open(metadata_path, 'r') as f:
            model_metadata = yaml.safe_load(f)
        
        model_config = model_metadata['config']['model']
        model_dir = os.path.dirname(metadata_path)
        model_filename = os.path.basename(metadata_path).replace('_metadata.yaml', '.pt')
        model_path = os.path.join(model_dir, model_filename)
        print(f"Found pre-trained model weights at: {model_path}")

        # 5. Load datasets for the specific state
        print(f"Loading datasets for state: {config['data']['state_for_finetuning']}...")
        base_path = config['data']['base_path']
        soil_features_available = model_config['use_soil_features']
        state_for_finetuning = config['data']['state_for_finetuning']
        
        train_path = os.path.join(base_path, config['data']['train_file'])
        val_path = os.path.join(base_path, config['data']['val_file'])
        
        train_dataset = YieldPredictionDataset(
            csv_file=train_path, 
            soil_features_available=soil_features_available, 
            ft=True, 
            state=state_for_finetuning
        )
        val_dataset = YieldPredictionDataset(
            csv_file=val_path, 
            soil_features_available=soil_features_available, 
            ft=True, 
            state=state_for_finetuning
        )
        
        if len(train_dataset) == 0:
            raise ValueError(f"Train dataset is empty for state: {state_for_finetuning}")
        
        # 6. Get input dimensions from data
        sample = train_dataset[0]
        sample_sequence = sample[0]
        input_dim = sample_sequence.shape[-1]
        soil_in_channels = None
        if soil_features_available:
            soil_tensor = sample[2]
            soil_in_channels = soil_tensor.shape[0]

        # 7. Create dataloaders
        train_loader, val_loader = create_dataloaders(config, train_dataset, val_dataset)

        # 8. Setup model
        print("Setting up model for fine-tuning...")
        model = CNN_RNN(
            input_dim=input_dim,
            rnn_hidden_dim=model_config['rnn_hidden_dim'],
            rnn_n_layers=model_config['rnn_n_layers'],
            rnn_dropout_prob=config['hyperparameters']['rnn_dropout_prob'], # Use HPO value
            fc_dropout_prob=config['hyperparameters']['fc_dropout_prob'],   # Use HPO value
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
        
        # Load pre-trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")

        # 9. Setup optimizer and loss functions
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['hyperparameters']['learning_rate'], 
            weight_decay=config['hyperparameters']['weight_decay']
        )
        
        training_loss_type = config['training']['loss']
        if training_loss_type == 'huber':
            criterion = nn.HuberLoss(delta=config['hyperparameters']['huber_delta'])
        elif training_loss_type == 'l1':
            criterion = nn.L1Loss()
        elif training_loss_type == 'l2':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported training loss: {training_loss_type}")

        validation_loss_type = config['training'].get('validation_loss')

        if validation_loss_type is None:
            val_criterion = criterion
            print("Validation loss not specified. Using training loss for validation.")
        elif validation_loss_type == 'l1':
            val_criterion = nn.L1Loss()
        elif validation_loss_type == 'l2':
            val_criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported validation loss: {validation_loss_type}")

        # 10. Fine-tuning loop
        print("Starting fine-tuning...")
        n_epochs = config['training']['n_epochs']
        patience = config['training']['early_stopping_patience']
        
        output_dir = config['output']['model_save_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"finetuned_{config['data']['state_for_finetuning']}_{timestamp}"
        model_save_path = os.path.join(output_dir, f"{model_name}.pt")

        early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path)
        
        train_losses = []
        val_losses = []

        for epoch in range(1, n_epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validate_epoch(model, val_loader, val_criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
        # 11. Evaluate model
        print("Evaluating fine-tuned model...")
        test_path = os.path.join(base_path, config['data']['test_file'])
        test_dataset = YieldPredictionDataset(
            csv_file=test_path, 
            inference=True, 
            soil_features_available=soil_features_available,
            ft=True,
            state=state_for_finetuning
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        
        test_metrics = evaluate_model(model, test_dataset, device, output_dir, model_name)
        
        # Save training plots
        training_plot_path = os.path.join(output_dir, f"{model_name}_training_history.png")
        save_training_plots(train_losses, val_losses, training_plot_path)
        
        # 12. Save metadata
        print("Saving metadata...")
        metadata = {
            'fine_tuning_config': config,
            'base_model_metadata_path': metadata_path,
            'best_val_loss': min(val_losses) if val_losses else None,
            'final_epoch': len(train_losses),
            'total_parameters': total_params,
            'test_metrics': test_metrics,
        }
        metadata_path = os.path.join(output_dir, f"{model_name}_metadata.yaml")
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, sort_keys=False)
        
        print(f"Metadata saved to: {metadata_path}")
        print("Fine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune model for yield prediction on a specific state')
    parser.add_argument('--config', default='fine_tuning_config.yaml', help='Config file path')
    args = parser.parse_args()
    main(args.config)
