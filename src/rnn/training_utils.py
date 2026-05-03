import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
# Apply shared plot style when analysis package is available
try:
    _src = os.path.join(os.path.dirname(__file__), '..')
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from analysis.plot_style import apply_style, FIG_SIZE_SINGLE, FIG_SIZE_DOUBLE, DPI
    apply_style()
except ImportError:
    # Fallback values if plot_style is not available
    FIG_SIZE_SINGLE = (8, 6)
    FIG_SIZE_DOUBLE = (12, 8)
    DPI = 300
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from models import CNN_RNN


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt', hpo=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            hpo (bool): If True, checkpoints will not be saved.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.hpo = hpo

        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):

        if val_loss < self.val_loss_min - self.delta:
            self.counter = 0    
            
            if self.hpo is False:
                torch.save(model.state_dict(), self.path)
         
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')

            self.val_loss_min = val_loss

        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True       

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')       


def set_random_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables full determinism (slower but 100% reproducible).
                      If False, uses faster non-deterministic algorithms (still reproducible for most cases).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        # For fully deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed} (fully deterministic mode)")
    else:
        # Faster mode - still reproducible for most practical purposes
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"Random seed set to: {seed} (fast mode - mostly reproducible)")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Basic validation
    required_sections = ['data', 'model', 'training', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config


def setup_model_and_training(config: Dict[str, Any], input_dim: int, device: torch.device):
    """Setup model, criterion, and optimizer."""
    model_config = config['model']
    
    model_args = {
        'input_dim': input_dim,
        'rnn_hidden_dim': model_config['rnn_hidden_dim'],
        'rnn_n_layers': model_config['rnn_n_layers'],
        'rnn_dropout_prob': model_config['rnn_dropout_prob'],
        'fc_dropout_prob': model_config['fc_dropout_prob'],
        'rnn_type': model_config['rnn_type'],
        'bidirectional': model_config.get('bidirectional', False),
        'fc_hidden_dims': model_config['fc_hidden_dims'],
        'use_temporal_attention': model_config['use_temporal_attention'],
        'attention_type': model_config['attention_type'],
        'attention_dim': model_config['attention_dim'],
        'attention_use_bias': model_config['attention_use_bias'],
        'use_layer_norm': model_config['use_layer_norm']
    }
    
    if model_config['use_soil_features']:
        soil_mode = model_config.get('soil_mode', 'cnn')
        if soil_mode == 'flat':
            model_args['flat_soil_dim'] = model_config['flat_soil_dim']
        else:
            model_args.update({
                'soil_in_channels': model_config['soil_in_channels'],
                'soil_out_channels': model_config['soil_out_channels'],
                'soil_kernel_size': model_config['soil_kernel_size'],
                'soil_n_conv_blocks': model_config['soil_n_conv_blocks']
            })

    model = CNN_RNN(**model_args)
    model.to(device)
    
    # Loss function
    loss_type = config['training']['loss']
    if loss_type == 'l2':
        criterion = nn.MSELoss()
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
    elif loss_type == 'huber':
        huber_delta = config['training']['huber_delta']
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Optimizer
    training_config = config['training']
    weight_decay = training_config['weight_decay']
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=weight_decay
    )
    
    return model, criterion, optimizer


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for batch in train_loader:
        # Batch may have 3 or 4 tensors depending on soil features
        if len(batch) == 3:
            sequences, past_yields, yields = batch
            soil_features = None
            sequences = sequences.to(device)
            past_yields = past_yields.to(device)
            yields = yields.to(device)
        elif len(batch) == 4:
            sequences, past_yields, soil_features, yields = batch
            sequences = sequences.to(device)
            past_yields = past_yields.to(device)
            soil_features = soil_features.to(device)
            yields = yields.to(device)
        else:
            raise ValueError("Unexpected batch size: expected 3 or 4 tensors, got {}".format(len(batch)))
        
        optimizer.zero_grad()

        if soil_features is not None:
            outputs = model(sequences, past_yields, soil_features)
        else:
            outputs = model(sequences, past_yields)

        loss = criterion(outputs, yields)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
        num_samples += sequences.size(0)
    
    return total_loss / num_samples


def validate_epoch(model, val_loader, criterion, device):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                sequences, past_yields, yields = batch
                soil_features = None
            elif len(batch) == 4:
                sequences, past_yields, soil_features, yields = batch
            else:
                raise ValueError("Unexpected batch size: expected 3 or 4 tensors, got {}".format(len(batch)))
            sequences = sequences.to(device)
            past_yields = past_yields.to(device)
            yields = yields.to(device)
            if soil_features is not None:
                soil_features = soil_features.to(device)
                outputs = model(sequences, past_yields, soil_features)
            else:
                outputs = model(sequences, past_yields)
            loss = criterion(outputs, yields)
            
            total_loss += loss.item() * sequences.size(0)
            num_samples += sequences.size(0)
    
    return total_loss / num_samples


def train_model(model, criterion, optimizer, train_loader, val_loader, config: Dict[str, Any], device: torch.device):
    """Train the model with early stopping."""
    n_epochs = config['training']['n_epochs']
    patience = config['training']['early_stopping_patience']
    
    # Create timestamped model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = config['output']['model_save_dir']
    model_filename = f"model_{timestamp}.pt"
    model_save_path = os.path.join(model_save_dir, model_filename)
    
    # Create directory if it doesn't exist
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_save_path, hpo=False)
    
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {n_epochs} epochs")
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Log progress
        print(f"Epoch {epoch:3d}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    
    print(f"Training completed. Best validation loss: {early_stopping.val_loss_min:.6f}")
    return train_losses, val_losses, model_save_path


def run_inference_on_dataset(model, dataset, device: torch.device, results_path: str, save_plots: bool = False, plot_path: str = None):
    """
    Run model inference on any dataset (train, val, or test) and save results to CSV.
    The dataset must have been created with inference=True so it has district_ids and sowing_years.
    Returns metrics dict, or None if dataset is empty.
    """
    if len(dataset) == 0:
        print("Dataset is empty. Skipping inference.")
        return None
    if not getattr(dataset, 'inference', False):
        raise ValueError("Dataset must be created with inference=True to have district_ids and sowing_years.")

    sequences = dataset.sequences.to(device)
    past_yields = dataset.past_yields.to(device)
    actual_yields = dataset.yields.to(device)
    district_ids = dataset.district_ids
    sowing_years = dataset.sowing_years

    model.eval()
    with torch.no_grad():
        if dataset.soil_features_available:
            soil_features = dataset.soil_features.to(device)
            predicted_yields = model(sequences, past_yields, soil_features).cpu().numpy()
        else:
            predicted_yields = model(sequences, past_yields).cpu().numpy()

    actual_yields_np = actual_yields.cpu().numpy()
    metrics = calculate_metrics(actual_yields_np, predicted_yields)
    save_results(actual_yields_np, predicted_yields, district_ids, sowing_years, metrics, results_path)
    if save_plots and plot_path:
        create_evaluation_plots(actual_yields_np, predicted_yields, plot_path)
    return metrics


def evaluate_model(model, test_dataset, device: torch.device, output_dir: str, model_name: str):
    """Evaluate the trained model on test data."""
    if len(test_dataset) == 0:
        print("Test dataset is empty. Skipping evaluation.")
        return None
    
    print("Evaluating model on test data...")
    
    results_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
    plot_path = os.path.join(output_dir, f"{model_name}_evaluation_plots.png")
    metrics = run_inference_on_dataset(
        model, test_dataset, device, results_path, save_plots=True, plot_path=plot_path
    )
    return metrics


class _WorkerInitFn:
    """Picklable worker init callable for DataLoader multiprocessing."""
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)


def create_dataloaders(config: dict, train_dataset, val_dataset):
    """Create training and validation dataloaders from config."""
    from torch.utils.data import DataLoader
    
    dataloader_config = config['dataloader']
    prefetch_factor = dataloader_config['prefetch_factor']
    if prefetch_factor == 'None':
        prefetch_factor = None
    
    num_workers = dataloader_config['num_workers']
    
    common_args = {
        'batch_size': dataloader_config['batch_size'],
        'num_workers': num_workers,
        'pin_memory': dataloader_config['pin_memory'],
        'persistent_workers': dataloader_config['persistent_workers'] if num_workers > 0 else False,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'worker_init_fn': _WorkerInitFn(config['reproducibility']['seed']) if num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=dataloader_config['drop_last_train'],
        **common_args
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args
    )
    
    return train_loader, val_loader


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mse)
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Baseline metrics (using mean)
    mean_yield = np.mean(y_true)
    baseline_pred = np.full_like(y_true, mean_yield)
    baseline_mse = mean_squared_error(y_true, baseline_pred)
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = sqrt(baseline_mse)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'correlation': float(correlation),
        'baseline_mse': float(baseline_mse),
        'baseline_mae': float(baseline_mae),
        'baseline_rmse': float(baseline_rmse)
    }


def create_evaluation_plots(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Create comprehensive evaluation plots."""
    plt.figure(figsize=FIG_SIZE_DOUBLE)
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Predicted vs Actual Yield')
    plt.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # Residual plot
    plt.subplot(2, 2, 2)
    residuals = y_pred - y_true
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Yield')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Residual distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Yield distribution comparison
    plt.subplot(2, 2, 4)
    plt.hist(y_true, bins=20, alpha=0.7, label='Actual', edgecolor='black')
    plt.hist(y_pred, bins=20, alpha=0.7, label='Predicted', edgecolor='black')
    plt.xlabel('Yield')
    plt.ylabel('Frequency')
    plt.title('Yield Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Evaluation plots saved to: {save_path}")


def save_results(y_true: np.ndarray, y_pred: np.ndarray, 
                district_ids: list, sowing_years: list, 
                metrics: Dict[str, float], save_path: str) -> None:
    """Save evaluation results to CSV."""
    results_df = pd.DataFrame({
        'district_id': district_ids,
        'sowing_year': sowing_years,
        'actual_yield': y_true,
        'predicted_yield': y_pred,
        'residual': y_pred - y_true,
        'absolute_error': np.abs(y_pred - y_true)
    })
    
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")
    
    # Print metrics summary
    print(f"\nModel Performance:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2_score']:.4f}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    
    print(f"\nBaseline Performance:")
    print(f"Baseline RMSE: {metrics['baseline_rmse']:.4f}")
    print(f"Baseline MAE: {metrics['baseline_mae']:.4f}")
    
    improvement = ((metrics['baseline_rmse'] - metrics['rmse']) / metrics['baseline_rmse']) * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")


def save_training_plots(train_losses: List[float], val_losses: List[float], save_path: str) -> None:
    """Save training history plot."""
    plt.figure(figsize=FIG_SIZE_SINGLE)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to: {save_path}")
