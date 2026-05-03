import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D-CNN baseline for crop yield prediction (Wolanin 2020 / Nevavuori 2019 style).

    Applies Conv1d along the temporal (month) axis of the input sequence, then
    pools and feeds through an MLP head that optionally incorporates flattened
    soil features and a lagged yield scalar.

    Expected input shapes from YieldPredictionDataset:
        sequences:     (B, T=11, F=13)   -- transposed internally to (B, F, T)
        past_yield:    (B,) or (B, 1)
        soil_features: (B, 8, 6) when use_soil_features=True, else None
    """

    def __init__(self, input_features, conv1_channels, conv2_channels,
                 fc_hidden_dims, fc_dropout_prob, use_soil_features=False,
                 flat_soil_dim=None):
        """
        Args:
            input_features: Number of temporal features (channel dim for Conv1d).
            conv1_channels: Output channels of the first conv block.
            conv2_channels: Output channels of the second conv block.
            fc_hidden_dims: List of hidden-layer sizes for the MLP head.
            fc_dropout_prob: Dropout probability in the MLP head.
            use_soil_features: Whether to concatenate flattened soil at the head.
            flat_soil_dim: Dimensionality of the flattened soil vector (e.g. 48).
        """
        super().__init__()
        self.use_soil_features = use_soil_features

        # --- Conv backbone ---
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, conv1_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(conv1_channels),
            nn.ReLU(),
            nn.Conv1d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(conv2_channels),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # --- MLP head ---
        head_input_dim = conv2_channels + 1  # +1 for past_yield
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
            sequences:      (B, T, F) temporal features.
            past_yield:     (B,) or (B, 1) lagged yield scalar.
            soil_features:  (B, C, D) soil tensor or None.
            return_attention: compatibility flag (CNN has no attention).
        Returns:
            Tensor of shape (B,).  If return_attention is True, returns (output, None).
        """
        # (B, T=11, F=13) -> (B, F=13, T=11)
        x = sequences.transpose(1, 2)
        x = self.conv(x)           # (B, conv2_channels, 1)
        x = x.squeeze(-1)          # (B, conv2_channels)

        if past_yield.dim() == 1:
            past_yield = past_yield.unsqueeze(1)

        parts = [x, past_yield]
        if self.use_soil_features and soil_features is not None:
            parts.append(soil_features.flatten(start_dim=1))

        combined = torch.cat(parts, dim=1)
        out = self.fc(combined).squeeze(1)

        if return_attention:
            return out, None
        return out
