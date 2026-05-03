import os
import sys

import torch
import torch.nn as nn

_rnn_dir = os.path.join(os.path.dirname(__file__), '..', 'rnn')
if _rnn_dir not in sys.path:
    sys.path.insert(0, _rnn_dir)

from models import TemporalAttention


class ParallelCNNLSTMAttn(nn.Module):
    """
    Parallel CNN-LSTM-Attention baseline (Song et al. 2025 TPCLA-inspired).

    CNN and LSTM process the *same* temporal stream in parallel.  Bahdanau
    attention is applied to the LSTM hidden states, and both branch outputs
    are fused before an MLP head.

    Expected input shapes from YieldPredictionDataset:
        sequences:     (B, T=11, F=13)
        past_yield:    (B,) or (B, 1)
        soil_features: (B, 8, 6) when use_soil_features=True, else None
    """

    def __init__(self, input_features, conv1_channels, conv2_channels,
                 lstm_hidden, lstm_layers, lstm_dropout,
                 attention_dim, fc_hidden_dims, fc_dropout_prob,
                 use_soil_features=False, flat_soil_dim=None):
        """
        Args:
            input_features:    Number of temporal features per timestep (F=13).
            conv1_channels:    Output channels of the first conv block.
            conv2_channels:    Output channels of the second conv block.
            lstm_hidden:       LSTM hidden-state size.
            lstm_layers:       Number of stacked LSTM layers.
            lstm_dropout:      Dropout between LSTM layers (ignored when lstm_layers==1).
            attention_dim:     Projection dimension for Bahdanau attention.
            fc_hidden_dims:    List of hidden-layer sizes for the MLP head.
            fc_dropout_prob:   Dropout probability in the MLP head.
            use_soil_features: Whether to concatenate flattened soil at the head.
            flat_soil_dim:     Dimensionality of the flattened soil vector (e.g. 48).
        """
        super().__init__()
        self.use_soil_features = use_soil_features

        # --- CNN branch (temporal) ---
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, conv1_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(conv1_channels),
            nn.ReLU(),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(conv2_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # --- LSTM branch (temporal) ---
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # --- Bahdanau attention over LSTM hidden states ---
        self.attention = TemporalAttention(lstm_hidden, attention_dim)

        # --- MLP head ---
        head_input_dim = conv2_channels + lstm_hidden + 1  # CNN + attention ctx + past_yield
        if use_soil_features and flat_soil_dim is not None:
            head_input_dim += flat_soil_dim

        layers = []
        in_dim = head_input_dim
        for h_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(fc_dropout_prob))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, sequences, past_yield, soil_features=None, return_attention=False):
        """
        Args:
            sequences:        (B, T, F) temporal features.
            past_yield:       (B,) or (B, 1) lagged yield scalar.
            soil_features:    (B, C, D) soil tensor or None.
            return_attention: If True, also return Bahdanau attention weights.
        Returns:
            Tensor of shape (B,).
            If return_attention is True, returns (output, attention_weights).
        """
        # CNN branch: (B, T, F) -> (B, F, T) for Conv1d
        cnn_out = self.conv(sequences.transpose(1, 2)).squeeze(-1)  # (B, conv2_channels)

        # LSTM branch: (B, T, F) -> all hidden states
        lstm_out, _ = self.lstm(sequences)  # (B, T, lstm_hidden)
        context, attn_weights = self.attention(lstm_out)  # (B, lstm_hidden), (B, T)

        if past_yield.dim() == 1:
            past_yield = past_yield.unsqueeze(1)

        parts = [cnn_out, context, past_yield]
        if self.use_soil_features and soil_features is not None:
            parts.append(soil_features.flatten(start_dim=1))

        combined = torch.cat(parts, dim=1)
        out = self.fc(combined).squeeze(1)

        if return_attention:
            return out, attn_weights
        return out
