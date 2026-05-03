import os
import sys
import yaml
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# Apply shared plot style when analysis package is available
try:
    _src = os.path.join(os.path.dirname(__file__), '..')
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from analysis.plot_style import apply_style, FIG_SIZE_DOUBLE, DPI
    apply_style()
    _use_plot_style = True
except ImportError:
    FIG_SIZE_DOUBLE = (12, 8)
    DPI = 300
    _use_plot_style = False
from torch.utils.data import DataLoader

from dataset import YieldPredictionDataset
from models import CNN_RNN

def plot_attention(metadata_path):
    """
    Loads a trained model from a metadata file, evaluates attention weights on the test set,
    and saves a plot of the aggregated attention weights.
    """
    # 1. Load metadata and config
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    config = metadata['config']

    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Load test dataset
    base_path = config['data']['base_path']
    test_path = os.path.join(base_path, config['data']['test_file'])
    soil_features_available = config['model']['use_soil_features']
    test_dataset = YieldPredictionDataset(csv_file=test_path, soil_features_available=soil_features_available)
    test_loader = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False)

    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")

    # 4. Re-create model from config
    sample = test_dataset[0]
    input_dim = sample[0].shape[-1]
    
    model_params = config['model']
    if soil_features_available:
        soil_tensor = sample[2]
        model_params['soil_in_channels'] = soil_tensor.shape[0]

    model = CNN_RNN(
        input_dim=input_dim,
        rnn_hidden_dim=model_params['rnn_hidden_dim'],
        rnn_n_layers=model_params['rnn_n_layers'],
        rnn_dropout_prob=model_params['rnn_dropout_prob'],
        fc_dropout_prob=model_params['fc_dropout_prob'],
        rnn_type=model_params['rnn_type'],
        fc_hidden_dims=model_params.get('fc_hidden_dims'),
        use_temporal_attention=model_params['use_temporal_attention'],
        attention_type=model_params['attention_type'],
        attention_dim=model_params.get('attention_dim'),
        attention_use_bias=model_params['attention_use_bias'],
        use_layer_norm=model_params['use_layer_norm'],
        bidirectional=model_params['bidirectional'],
        soil_in_channels=model_params.get('soil_in_channels'),
        soil_out_channels=model_params.get('soil_out_channels'),
        soil_kernel_size=model_params.get('soil_kernel_size'),
        soil_n_conv_blocks=model_params.get('soil_n_conv_blocks', 1)
    )

    # 5. Load model weights
    model_dir = os.path.dirname(metadata_path)
    model_basename = os.path.basename(metadata_path).replace('_metadata.yaml', '')
    model_path = os.path.join(model_dir, f"{model_basename}.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 6. Get attention weights
    all_attention_weights = []
    with torch.no_grad():
        for batch in test_loader:
            if soil_features_available:
                sequences, past_yields, soil_features, _ = batch
                sequences, past_yields, soil_features = sequences.to(device), past_yields.to(device), soil_features.to(device)
                _, attention_weights = model(sequences, past_yields, soil_features, return_attention=True)
            else:
                sequences, past_yields, _ = batch
                sequences, past_yields = sequences.to(device), past_yields.to(device)
                _, attention_weights = model(sequences, past_yields, return_attention=True)
            
            if attention_weights is not None:
                all_attention_weights.append(attention_weights.cpu().numpy())

    if not all_attention_weights:
        print("No attention weights to plot. The model might not be using attention.")
        return

    # 7. Aggregate and plot attention weights
    aggregated_weights = np.concatenate(all_attention_weights, axis=0)
    mean_weights = np.mean(aggregated_weights, axis=0)
    
    seq_len = test_dataset[0][0].shape[0]

    month_labels = ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

    # Indian wheat phenological stages (June–April): map month indices to shaded regions
    # Jun-Sep 0-3: Monsoon/Fallow; Oct-Nov 4-5: Sowing/Establishment; Dec-Jan 6-7: Tillering; Feb 8: Heading; Mar-Apr 9-10: Grain fill/Harvest
    phenology_regions = [
        (0, 4, 'Monsoon / Land prep', 'lightblue'),
        (4, 6, 'Sowing / Establishment', 'lightgreen'),
        (6, 8, 'Tillering / Vegetative', 'lightyellow'),
        (8, 9, 'Heading / Flowering', 'navajowhite'),
        (9, 11, 'Grain fill / Harvest', 'mistyrose'),
    ]

    # Use shared figure size and style (no seaborn theme so apply_style fonts/sizes stick)
    fig, ax = plt.subplots(figsize=FIG_SIZE_DOUBLE)
    # Draw phenological background bands first (behind bars)
    for i_start, i_end, label, color in phenology_regions:
        if i_end <= seq_len:
            ax.axvspan(i_start - 0.5, i_end - 0.5, facecolor=color, alpha=0.4, zorder=0)
    # Bar plot on top; use first color from style palette for consistency
    x_pos = np.arange(seq_len)
    bar_color = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['#6086B6'])[0]
    ax.bar(x_pos, mean_weights, zorder=1, color=bar_color, edgecolor='gray', alpha=0.85)
    ax.set_title("Aggregated Attention Weights over Test Set (Indian wheat phenology)", pad=20)
    ax.set_ylabel("Average Attention Weight")
    if seq_len == len(month_labels):
        ax.set_xticks(x_pos)
        ax.set_xticklabels(month_labels, ha='right')
    else:
        print(f"Warning: Sequence length ({seq_len}) does not match the number of month labels ({len(month_labels)}). Using numeric labels for x-axis.")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    # Legend for phenological stages
    legend_handles = [Patch(facecolor=c, alpha=0.4, label=l) for (_, _, l, c) in phenology_regions]
    ax.legend(handles=legend_handles, loc='upper right')
    plt.tight_layout()

    plot_path = os.path.join(model_dir, f"{model_basename}_attention_plot.png")
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    print(f"Attention plot saved to {plot_path}")
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot attention weights for a trained model.')
    parser.add_argument(
        '--metadata', 
        type=str, 
        required=True, 
        help='Path to the model metadata yaml file.'
    )
    args = parser.parse_args()
    plot_attention(args.metadata)
