import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Pure Transformer encoder baseline for crop yield prediction
    (Wang 2024 CompAG / SLTF-inspired, simplified to vanilla encoder).

    Applies a learned input projection, adds learnable positional embeddings,
    passes through a stack of Transformer encoder layers, mean-pools across
    months, and feeds the result through an MLP head that optionally
    incorporates flattened soil features and a lagged yield scalar.

    Expected input shapes from YieldPredictionDataset:
        sequences:     (B, T=11, F=13)
        past_yield:    (B,) or (B, 1)
        soil_features: (B, 8, 6) when use_soil_features=True, else None
    """

    def __init__(self, input_features, d_model, nhead, num_layers,
                 dim_feedforward, dropout, fc_hidden_dims, fc_dropout_prob,
                 use_soil_features=False, flat_soil_dim=None, seq_len=11):
        super().__init__()
        self.use_soil_features = use_soil_features

        self.input_proj = nn.Linear(input_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- MLP head ---
        head_input_dim = d_model + 1  # +1 for past_yield
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
            return_attention: compatibility flag (returns None for weights).
        Returns:
            Tensor of shape (B,).  If return_attention is True, returns (output, None).
        """
        x = self.input_proj(sequences)       # (B, T, d_model)
        x = x + self.pos_embed               # broadcast over batch

        x = self.encoder(x)                  # (B, T, d_model)
        x = x.mean(dim=1)                    # (B, d_model)  mean-pool over months

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
