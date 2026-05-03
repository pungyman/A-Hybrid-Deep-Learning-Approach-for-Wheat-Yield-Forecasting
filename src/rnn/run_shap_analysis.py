import os
import sys
import yaml
import argparse
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
# Apply shared plot style when analysis package is available
try:
    _src = os.path.join(os.path.dirname(__file__), '..')
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from analysis.plot_style import apply_style
    apply_style()
except ImportError:
    pass

from dataset import YieldPredictionDataset
from models import CNN_RNN
from training_utils import set_random_seed


# Canonical filenames for paper figures, mapped from the suffix of the
# timestamped artifact name produced by this script.
PAPER_FIGURE_NAMES = {
    'feature_importance': 'shap_feature_importance.png',
    'month_importance': 'shap_month_importance.png',
    'depth_importance': 'shap_depth_importance.png',
    'feature_month_heatmap': 'shap_feature_month_heatmap.png',
}


def save_figure(fig, output_dir: Path, plot_filename_base: str, suffix: str,
                paper_figures_dir: Path = None, dpi: int = 300):
    """Save the current figure under the timestamped name and, when
    paper_figures_dir is provided, also under its canonical paper-figure name."""
    timestamped_path = output_dir / f"{plot_filename_base}_{suffix}.png"
    fig.savefig(timestamped_path, dpi=dpi)
    print(f"  saved {timestamped_path}")
    if paper_figures_dir is not None and suffix in PAPER_FIGURE_NAMES:
        canonical_path = paper_figures_dir / PAPER_FIGURE_NAMES[suffix]
        fig.savefig(canonical_path, dpi=dpi)
        print(f"  saved {canonical_path}")


def main(config_path: str):
    """
    Main function to run SHAP analysis on a trained model.
    """
    # 1. Load Configuration
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reproducibility: SHAP background/test sampling and any torch ops
    # depend on RNG state, so seed before any DataLoader is constructed.
    set_random_seed(42, deterministic=True)

    # Create output directory if it doesn't exist
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional canonical paper-figure directory
    paper_figures_dir = None
    if config.get('paper_figures_dir'):
        paper_figures_dir = Path(config['paper_figures_dir'])
        paper_figures_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load Model
    print("Loading model...")
    with open(config['model_metadata_path'], 'r') as f:
        metadata = yaml.safe_load(f)
    
    model_config = metadata['config']['model']
    use_soil_features = model_config.get('use_soil_features', False)
    
    # Infer model weights path
    model_dir = os.path.dirname(config['model_metadata_path'])
    model_filename = os.path.basename(config['model_metadata_path']).replace('_metadata.yaml', '.pt')
    model_weights_path = os.path.join(model_dir, model_filename)
    model_name = model_filename.replace('.pt', '')

    # Instantiate model
    # The input_dim will be derived from the dataset later. For now, use from config.
    input_dim = len(config['temporal_feature_names'])
    
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
    
    if use_soil_features:
        model_args.update({
            'soil_in_channels': model_config['soil_in_channels'],
            'soil_out_channels': model_config['soil_out_channels'],
            'soil_kernel_size': model_config['soil_kernel_size'],
            'soil_n_conv_blocks': model_config['soil_n_conv_blocks']
        })

    model = CNN_RNN(**model_args)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_weights_path}")

    # 4. Load Data
    print("Loading datasets...")
    dataset_dir = config['dataset_dir']
    train_path = os.path.join(dataset_dir, 'train.csv')
    test_path = os.path.join(dataset_dir, 'test.csv')

    # Assuming soil features are not used
    train_dataset = YieldPredictionDataset(csv_file=train_path, soil_features_available=use_soil_features)
    test_dataset = YieldPredictionDataset(csv_file=test_path, soil_features_available=use_soil_features)

    # 5. Prepare Data for SHAP
    print("Preparing data for SHAP analysis...")
    
    # Background data from training set using DataLoader
    background_loader = DataLoader(train_dataset, batch_size=config['n_background_samples'], shuffle=True)
    background_data = next(iter(background_loader))
    background_sequences = background_data[0].to(device)
    background_past_yields = background_data[1].to(device)
    background_soil_features = background_data[2].to(device) if use_soil_features else None

    # Test data for explanation using DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config['n_test_samples'], shuffle=True)
    test_data = next(iter(test_loader))
    test_sequences = test_data[0].to(device)
    test_past_yields = test_data[1].to(device)
    test_soil_features = test_data[2].to(device) if use_soil_features else None

    # 6. Run SHAP GradientExplainer
    print("Running SHAP GradientExplainer...")
    
    explainer_inputs = [background_sequences, background_past_yields]
    test_inputs = [test_sequences, test_past_yields]
    
    if use_soil_features:
        explainer_inputs.append(background_soil_features)
        test_inputs.append(test_soil_features)

    # Wrap the model to ensure the output is 2D
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, *args, **kwargs):
            output = self.model(*args, **kwargs)
            if output.dim() == 1:
                return output.unsqueeze(1)
            return output

    wrapped_model = ModelWrapper(model)

    explainer = shap.GradientExplainer(wrapped_model, explainer_inputs)
    shap_values = explainer.shap_values(test_inputs)

    # The model wrapper added an extra dimension to the output, so we squeeze it from the SHAP values
    shap_values = [np.squeeze(v, axis=-1) for v in shap_values]

    # shap_values[0] will contain the SHAP values for the sequences input
    # shap_values[1] will contain the SHAP values for the lagged yield (past_yield) input
    # shap_values[2] will contain SHAP values for soil features if they are used

    # shape of shap_values[0]: num_samples, num_months, num_temporal_features
    # shape of shap_values[1]: num_samples,
    # shape of shap_values[2]: num_samples, num_soil_features, num_depths

    # Each individual SHAP value tells you how much that specific feature value, for that specific data point, 
    # contributed to pushing the model's output away from the average prediction (which was learned from the background data). 
    # A positive SHAP value means the feature pushed the final prediction higher, and a negative value means it pushed it lower.

    # 7. Analyze and Plot SHAP values
    print("Generating and saving plots...")
    
    # Create a base filename for the plots
    plot_filename_base = f"{model_name}_bg_{config['n_background_samples']}_test_{config['n_test_samples']}"
    
    # Feature Importance
    temporal_shap_values = shap_values[0]
    static_shap_values = shap_values[1]

    # Mean absolute SHAP for temporal features
    mean_abs_shap_temporal = np.abs(temporal_shap_values).mean(axis=(0, 1))
    
    # Mean absolute SHAP for static feature
    mean_abs_shap_static = np.abs(static_shap_values).mean()

    feature_names = config['temporal_feature_names'] + [config['static_feature_name']]
    feature_importances = np.concatenate([mean_abs_shap_temporal, [mean_abs_shap_static]])

    if use_soil_features:
        soil_shap_values = shap_values[2]
        # Average SHAP values across the sample and depth dimension
        mean_abs_shap_soil = np.abs(soil_shap_values).mean(axis=(0, 2))
        feature_names += config['soil_feature_names']
        feature_importances = np.concatenate([feature_importances, mean_abs_shap_soil])
    
    # Sort features by importance
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_importances = feature_importances[sorted_indices]

    # Display label fix: present the lagged-yield input under its paper name
    # rather than the internal `past_yield` key.
    display_feature_names = [
        'lagged_yield' if name == config['static_feature_name'] else name
        for name in sorted_feature_names
    ]

    # Grouped aggregate totals shown as an inset, to contextualize the
    # apparent dominance of the scalar lagged-yield input over the
    # per-cell averages of the multidimensional temporal/soil blocks.
    # Each group total is the sum of mean |SHAP| across its constituent
    # cells (months x features for temporal, properties x depths for soil).
    lagged_yield_total = float(mean_abs_shap_static)
    temporal_total = float(np.abs(temporal_shap_values).mean(axis=0).sum())
    group_labels = ['Lagged yield\n(1 cell)',
                    f'All temporal\n({temporal_shap_values.shape[1]}\u00d7'
                    f'{temporal_shap_values.shape[2]} cells)']
    group_totals = [lagged_yield_total, temporal_total]
    if use_soil_features:
        soil_total = float(np.abs(soil_shap_values).mean(axis=0).sum())
        group_labels.append(
            f'All soil\n({soil_shap_values.shape[1]}\u00d7'
            f'{soil_shap_values.shape[2]} cells)'
        )
        group_totals.append(soil_total)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(display_feature_names, sorted_importances)
    ax.set_xlabel('Mean Absolute SHAP Value')
    ax.set_title('Feature Importance')
    ax.invert_yaxis()

    # Inset (bottom-right) with grouped totals on a comparable scale.
    inset_ax = ax.inset_axes([0.55, 0.05, 0.42, 0.35])
    inset_colors = ['#4c72b0', '#dd8452', '#55a868'][:len(group_totals)]
    inset_ax.bar(range(len(group_totals)), group_totals, color=inset_colors)
    inset_ax.set_xticks(range(len(group_totals)))
    inset_ax.set_xticklabels(group_labels, fontsize=8)
    inset_ax.set_ylabel('Sum of mean |SHAP|', fontsize=8)
    inset_ax.set_title('Grouped totals', fontsize=9)
    inset_ax.tick_params(axis='y', labelsize=8)
    for i, total in enumerate(group_totals):
        inset_ax.text(i, total, f'{total:.0f}', ha='center', va='bottom',
                      fontsize=8)
    inset_ax.margins(y=0.15)

    plt.tight_layout()
    save_figure(fig, output_dir, plot_filename_base, 'feature_importance', paper_figures_dir)
    plt.close(fig)

    # Month Importance
    # Sum absolute SHAP values across features for each month, then average over samples
    month_importances = np.abs(temporal_shap_values).sum(axis=2).mean(axis=0)
    months = ['June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    
    if len(month_importances) != len(months):
        print(f"Warning: Number of time steps ({len(month_importances)}) does not match number of months ({len(months)}). Using numerical labels for months.")
        months = range(len(month_importances))

    fig = plt.figure(figsize=(10, 6))
    plt.bar(months, month_importances)
    plt.xlabel('Month')
    plt.ylabel('Mean Absolute SHAP Value Sum')
    plt.title('Month Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure(fig, output_dir, plot_filename_base, 'month_importance', paper_figures_dir)
    plt.close(fig)

    # Feature x Month Heatmap (R2-4): mean |SHAP| over samples,
    # leaving a (months, features) grid that exposes which RS / weather
    # indices dominate at which point in the growing season.
    feature_month_importance = np.abs(temporal_shap_values).mean(axis=0)  # (T=11, F=13)
    features_axis = config['temporal_feature_names']

    if feature_month_importance.shape != (len(months), len(features_axis)):
        print(
            f"Warning: feature-month SHAP grid shape {feature_month_importance.shape} "
            f"does not match ({len(months)}, {len(features_axis)}); skipping heatmap."
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(feature_month_importance, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(features_axis)))
        ax.set_xticklabels(features_axis, rotation=45, ha='right')
        ax.set_yticks(range(len(months)))
        ax.set_yticklabels(months)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Month')
        ax.set_title('Mean |SHAP| by feature and month')
        fig.colorbar(im, ax=ax, label='Mean |SHAP|')
        plt.tight_layout()
        save_figure(fig, output_dir, plot_filename_base, 'feature_month_heatmap', paper_figures_dir)
        plt.close(fig)

    # Depth Importance (if soil features are used)
    if use_soil_features:
        # Sum absolute SHAP values across features for each depth, then average over samples
        depth_importances = np.abs(soil_shap_values).sum(axis=1).mean(axis=0)
        depths = config['soil_depth_names']

        if len(depth_importances) != len(depths):
            print(f"Warning: Number of depth steps ({len(depth_importances)}) does not match number of depths ({len(depths)}). Using numerical labels for depths.")
            depths = range(len(depth_importances))

        fig = plt.figure(figsize=(10, 6))
        plt.bar(depths, depth_importances)
        plt.xlabel('Depth')
        plt.ylabel('Mean Absolute SHAP Value Sum')
        plt.title('Soil Depth Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_figure(fig, output_dir, plot_filename_base, 'depth_importance', paper_figures_dir)
        plt.close(fig)

    print("SHAP analysis completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SHAP analysis for yield prediction model.')
    parser.add_argument('--config', default='shap_config.yaml', help='Path to the SHAP config file.')
    args = parser.parse_args()
    main(args.config)
