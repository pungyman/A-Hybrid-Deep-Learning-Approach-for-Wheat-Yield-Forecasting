import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN(nn.Module):
    def __init__(self, input_dim, rnn_hidden_dim, rnn_n_layers, rnn_dropout_prob, fc_dropout_prob, 
                 rnn_type='gru', fc_hidden_dims=None, use_temporal_attention=False, attention_type='bahdanau',
                 attention_dim=None, attention_use_bias=True, use_layer_norm=False,
                 bidirectional=False, soil_in_channels=None, soil_out_channels=None, soil_kernel_size=None, 
                 soil_n_conv_blocks=1, flat_soil_dim=None):
        """
        Args:
            input_dim (int): Number of temporal features in the input.
            rnn_hidden_dim (int): Number of features in the RNN hidden state `h`.
            rnn_n_layers (int): Number of GRU/LSTM layers.
            rnn_dropout_prob (float): Dropout probability for RNN layers.
            fc_dropout_prob (float): Dropout probability for fully connected layers.
            rnn_type (str): Type of RNN to use ('gru' or 'lstm').
            fc_hidden_dims (list of int, optional): A list of hidden layer sizes for the fully-connected part of the network.
            use_temporal_attention (bool): Whether to use temporal attention mechanism. Defaults to False.
            attention_type (str): Type of attention to use ('bahdanau' or 'self'). Defaults to 'bahdanau'.
            attention_dim (int, optional): Dimension of the attention space. If None, defaults to rnn_hidden_dim.
            attention_use_bias (bool): Whether to use bias in the attention computation. Defaults to True.
            use_layer_norm (bool): Whether to use layer normalization after RNN and attention layers. Defaults to False.
            bidirectional (bool): Whether to use a bidirectional RNN. Defaults to False.
            soil_in_channels (int): Number of input soil features (channels).
            soil_out_channels (int): Number of output channels from the CNN.
            soil_kernel_size (int): Kernel size for the 1D convolution layers.
            soil_n_conv_blocks (int): Number of conv blocks (1 or 2). Defaults to 1.
            flat_soil_dim (int, optional): When set, soil features are flattened to this
                dimensionality and concatenated directly at the MLP head instead of
                being processed by a SoilCNN. Mutually exclusive with the soil CNN
                parameters.
        """
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_type = rnn_type.lower()
        self.use_temporal_attention = use_temporal_attention
        self.attention_type = attention_type
        self.use_layer_norm = use_layer_norm
        self.bidirectional = bidirectional

        # The effective output dimension of the RNN, which doubles if bidirectional
        self.rnn_output_dim = self.rnn_hidden_dim * 2 if self.bidirectional else self.rnn_hidden_dim

        if self.rnn_type not in ['gru', 'lstm']:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'gru' or 'lstm'.")

        if self.use_temporal_attention and attention_type not in ['bahdanau', 'self']:
            raise ValueError(f"Unsupported attention type: {attention_type}. Choose 'bahdanau' or 'self'.")

        # RNN branch
        RNN = nn.GRU if self.rnn_type == 'gru' else nn.LSTM
        self.rnn = RNN(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=rnn_dropout_prob if rnn_n_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Add temporal attention mechanism if requested
        if self.use_temporal_attention:
            if attention_type == 'bahdanau':
                self.attention = TemporalAttention(self.rnn_output_dim, attention_dim, attention_use_bias)
            elif attention_type == 'self':
                self.attention = SelfAttention(self.rnn_output_dim, attention_dim, attention_use_bias)
        
        # Add layer normalization layers if requested
        if self.use_layer_norm:
            self.rnn_norm = nn.LayerNorm(self.rnn_output_dim)
            if self.use_temporal_attention:
                self.attention_norm = nn.LayerNorm(self.rnn_output_dim)
        
        # Soil branch: CNN-based or flat concatenation
        self.flat_soil = False
        if soil_in_channels is not None and soil_out_channels is not None and soil_kernel_size is not None:
            self.soil_cnn = SoilCNN(soil_in_channels, soil_out_channels, soil_kernel_size, soil_n_conv_blocks)
            self.use_soil_features = True
        elif flat_soil_dim is not None:
            self.use_soil_features = True
            self.flat_soil = True
            self.flat_soil_dim = flat_soil_dim
        else:
            self.use_soil_features = False
        
        # Fully connected layers
        if fc_hidden_dims is None:
            fc_hidden_dims = []

        fc_layers = []
        fc_input_dim = self.rnn_output_dim + 1  # rnn output + lagged yield feature
        if self.use_soil_features:
            if self.flat_soil:
                fc_input_dim += flat_soil_dim
            else:
                fc_input_dim += soil_out_channels
        
        for dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(fc_dropout_prob))
            fc_input_dim = dim
        
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, past_yield, soil_features=None, return_attention=False):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            past_yield (torch.Tensor): Lagged yield feature. Tensor of shape (batch_size,) or (batch_size, 1).
            soil_features (torch.Tensor, optional): Soil features tensor of shape (batch_size, features, depth).
            return_attention (bool): Whether to return attention weights.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
            or
            (torch.Tensor, torch.Tensor): Output tensor and attention weights if return_attention is True.
        """
        attention_weights = None

        # RNN branch
        if self.use_temporal_attention:
            # Get all hidden states from RNN for attention
            if self.rnn_type == 'lstm':
                rnn_outputs, (h_n, c_n) = self.rnn(x)
            else:  # 'gru'
                rnn_outputs, h_n = self.rnn(x)
            
            # Apply layer normalization to RNN outputs if enabled
            if self.use_layer_norm:
                rnn_outputs = self.rnn_norm(rnn_outputs)
            
            # rnn_outputs: (batch_size, seq_length, rnn_hidden_dim)
            if self.attention_type == 'bahdanau':
                # Apply temporal attention to get context vector
                context_vector, attention_weights = self.attention(rnn_outputs)
                rnn_out = context_vector  # (batch_size, rnn_hidden_dim)
            elif self.attention_type == 'self':
                # Apply self-attention to enhance the last hidden state
                rnn_out, attention_weights = self.attention(rnn_outputs)  # (batch_size, rnn_hidden_dim)
            
            # Apply layer normalization to attention output if enabled
            if self.use_layer_norm:
                rnn_out = self.attention_norm(rnn_out)
        else:
            # Pass the last hidden state
            if self.rnn_type == 'lstm':
                _, (h_n, _) = self.rnn(x)
            else: # 'gru'
                _, h_n = self.rnn(x)

            if self.bidirectional:
                # h_n is of shape (rnn_n_layers * 2, batch_size, rnn_hidden_dim)
                # Concatenate the last forward and backward hidden states from the last layer
                rnn_out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            else:
                # h_n is of shape (rnn_n_layers, batch_size, rnn_hidden_dim)
                rnn_out = h_n[-1]  # (batch_size, rnn_hidden_dim)
            
            # Apply layer normalization to RNN output if enabled
            if self.use_layer_norm:
                rnn_out = self.rnn_norm(rnn_out)
        
        # Soil branch
        if self.use_soil_features and soil_features is not None:
            if self.flat_soil:
                soil_out = soil_features.flatten(start_dim=1)  # (batch_size, flat_soil_dim)
            else:
                soil_out = self.soil_cnn(soil_features)  # (batch_size, soil_out_channels)
        else:
            soil_out = None
        
        # Ensure past_yield is (batch_size, 1)
        if past_yield.dim() == 1:
            past_yield = past_yield.unsqueeze(1)
        
        # Concatenate features
        features_to_concat = [rnn_out, past_yield]  # (batch_size, rnn_hidden_dim + 1) lagged yield feature
        if soil_out is not None:
            features_to_concat.append(soil_out)  # add soil features
        
        combined = torch.cat(features_to_concat, dim=1)  # (batch_size, rnn_hidden_dim + 1 + soil_out_channels)
        out = self.fc(combined)
        out = out.squeeze(1)

        if return_attention:
            return out, attention_weights
        
        return out


class RNNModel(nn.Module):
    """
    An RNN-based model for sequence data, supporting GRU or LSTM cells.
    """
    def __init__(self, input_dim, rnn_hidden_dim, rnn_n_layers, dropout_prob, rnn_type='gru', fc_hidden_dims=None):
        """
        Args:
            input_dim (int): Number of features in the input.
            rnn_hidden_dim (int): Number of features in the RNN hidden state `h`.
            rnn_n_layers (int): Number of recurrent layers.
            dropout_prob (float): Dropout probability.
            rnn_type (str): Type of RNN to use ('gru' or 'lstm').
            fc_hidden_dims (list of int, optional): A list of hidden layer sizes for the fully connected part. Defaults to None.
        """
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type not in ['gru', 'lstm']:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'gru' or 'lstm'.")

        RNN = nn.GRU if self.rnn_type == 'gru' else nn.LSTM
        self.rnn = RNN(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=dropout_prob if rnn_n_layers > 1 else 0
        )
        
        if fc_hidden_dims is None:
            fc_hidden_dims = []

        fc_layers = []
        fc_input_dim = rnn_hidden_dim
        for dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_prob))
            fc_input_dim = dim
        
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        if self.rnn_type == 'lstm':
            _, (h_n, _) = self.rnn(x)
        else: # 'gru'
            _, h_n = self.rnn(x)
        
        # h_n is of shape (rnn_n_layers, batch_size, rnn_hidden_dim)
        # We take the hidden state of the last layer
        out = self.fc(h_n[-1])
        
        return out.squeeze(1)


class RNNModel_with_past_yield(nn.Module):
    """
    An RNN-based model for sequence data, supporting GRU or LSTM layers, with direct lagged yield feature input to the fully-connected layers.
    Optionally supports temporal attention mechanism for better sequence modeling.
    """
    def __init__(self, input_dim, rnn_hidden_dim, rnn_n_layers, dropout_prob, rnn_type='gru', 
                 fc_hidden_dims=None, use_temporal_attention=False, attention_type='bahdanau',
                 attention_dim=None, attention_use_bias=True, use_layer_norm=False):
        """
        Args:
            input_dim (int): Number of temporal features in the input.
            rnn_hidden_dim (int): Number of features in the RNN hidden state `h`.
            rnn_n_layers (int): Number of GRU/LSTM layers.
            dropout_prob (float): Dropout probability.
            rnn_type (str): Type of RNN to use ('gru' or 'lstm').
            fc_hidden_dims (list of int, optional): A list of hidden layer sizes for the fully-connected part of the network.
                there will always be at least one fc layer (the output layer, that will project to size 1)
                in case None is passed, there will only be the layer which will project from rnn_hidden_dim + 1 -> 1
                if you pass say [64, 32], then there will be two fc layers before the output layer
            use_temporal_attention (bool): Whether to use temporal attention mechanism. Defaults to False.
            attention_type (str): Type of attention to use ('bahdanau' or 'self'). Defaults to 'bahdanau'.
            attention_dim (int, optional): Dimension of the attention space. If None, defaults to rnn_hidden_dim.
            attention_use_bias (bool): Whether to use bias in the attention computation. Defaults to True.
            use_layer_norm (bool): Whether to use layer normalization after RNN and attention layers. Defaults to False.
        """
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_n_layers = rnn_n_layers
        self.rnn_type = rnn_type.lower()
        self.use_temporal_attention = use_temporal_attention
        self.attention_type = attention_type
        self.use_layer_norm = use_layer_norm

        if self.rnn_type not in ['gru', 'lstm']:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'gru' or 'lstm'.")

        if self.use_temporal_attention and attention_type not in ['bahdanau', 'self']:
            raise ValueError(f"Unsupported attention type: {attention_type}. Choose 'bahdanau' or 'self'.")

        RNN = nn.GRU if self.rnn_type == 'gru' else nn.LSTM
        self.rnn = RNN(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=dropout_prob if rnn_n_layers > 1 else 0
        )
        
        # Add temporal attention mechanism if requested
        if self.use_temporal_attention:
            if attention_type == 'bahdanau':
                self.attention = TemporalAttention(rnn_hidden_dim, attention_dim, attention_use_bias)
            elif attention_type == 'self':
                self.attention = SelfAttention(rnn_hidden_dim, attention_dim, attention_use_bias)
        
        # Add layer normalization layers if requested
        if self.use_layer_norm:
            self.rnn_norm = nn.LayerNorm(rnn_hidden_dim)
            if self.use_temporal_attention:
                self.attention_norm = nn.LayerNorm(rnn_hidden_dim)
        
        if fc_hidden_dims is None:
            fc_hidden_dims = []

        fc_layers = []
        # The input to the first FC layer is rnn_hidden_dim + 1 (lagged yield feature)
        fc_input_dim = rnn_hidden_dim + 1
        for dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_prob))
            fc_input_dim = dim
        
        fc_layers.append(nn.Linear(fc_input_dim, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x, past_yield):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).
            past_yield (torch.Tensor): Lagged yield feature. Tensor of shape (batch_size,) or (batch_size, 1).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        if self.use_temporal_attention:
            # Get all hidden states from RNN for attention
            if self.rnn_type == 'lstm':
                rnn_outputs, (h_n, c_n) = self.rnn(x)
            else:  # 'gru'
                rnn_outputs, h_n = self.rnn(x)
            
            # Apply layer normalization to RNN outputs if enabled
            if self.use_layer_norm:
                rnn_outputs = self.rnn_norm(rnn_outputs)
            
            # rnn_outputs: (batch_size, seq_length, rnn_hidden_dim)
            if self.attention_type == 'bahdanau':
                # Apply temporal attention to get context vector
                context_vector, attention_weights = self.attention(rnn_outputs)
                rnn_out = context_vector  # (batch_size, rnn_hidden_dim)
            elif self.attention_type == 'self':
                # Apply self-attention to enhance the last hidden state
                rnn_out, attention_weights = self.attention(rnn_outputs)  # (batch_size, rnn_hidden_dim)
            
            # Apply layer normalization to attention output if enabled
            if self.use_layer_norm:
                rnn_out = self.attention_norm(rnn_out)
        else:
            # Pass the last hidden state
            if self.rnn_type == 'lstm':
                _, (h_n, _) = self.rnn(x)
            else: # 'gru'
                _, h_n = self.rnn(x)
            # h_n is of shape (rnn_n_layers, batch_size, rnn_hidden_dim)
            rnn_out = h_n[-1]  # (batch_size, rnn_hidden_dim)
            
            # Apply layer normalization to RNN output if enabled
            if self.use_layer_norm:
                rnn_out = self.rnn_norm(rnn_out)
        
        # Ensure past_yield is (batch_size, 1)
        if past_yield.dim() == 1:
            past_yield = past_yield.unsqueeze(1)
        # Concatenate along feature dimension
        combined = torch.cat([rnn_out, past_yield], dim=1)  # (batch_size, rnn_hidden_dim + 1) with lagged yield
        out = self.fc(combined)
        return out.squeeze(1)


class TemporalAttention(nn.Module):
    """
    Bahdanau-style additive attention mechanism for temporal sequences.
    
    This module computes attention weights over a sequence of hidden states
    and returns a context vector as a weighted sum of the hidden states.
    """
    def __init__(self, hidden_dim, attention_dim=None, use_bias=True):
        """
        Args:
            hidden_dim (int): Dimension of the hidden states from RNN
            attention_dim (int, optional): Dimension of the attention space. 
                                         If None, defaults to hidden_dim.
            use_bias (bool): Whether to use bias in the attention computation.
                           Defaults to True.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        self.use_bias = use_bias
        
        # Bahdanau-style additive attention
        # W1: hidden_dim -> attention_dim
        self.W1 = nn.Linear(hidden_dim, self.attention_dim, bias=use_bias)
        # W2: attention_dim -> 1 (for computing attention scores)
        self.W2 = nn.Linear(self.attention_dim, 1, bias=use_bias)
        
    def forward(self, hidden_states):
        """
        Compute attention weights and context vector.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from RNN of shape 
                                        (batch_size, seq_length, hidden_dim)
        
        Returns:
            context_vector (torch.Tensor): Weighted sum of hidden states, 
                                         shape (batch_size, hidden_dim)
            attention_weights (torch.Tensor): Attention weights, 
                                            shape (batch_size, seq_length)
        """
        # Compute attention energies
        # hidden_states: (batch_size, seq_length, hidden_dim)
        # W1(hidden_states): (batch_size, seq_length, attention_dim)
        energies = self.W2(torch.tanh(self.W1(hidden_states)))  # (batch_size, seq_length, 1)
        energies = energies.squeeze(-1)  # (batch_size, seq_length)
        
        # Compute attention weights using softmax
        attention_weights = F.softmax(energies, dim=1)  # (batch_size, seq_length)
        
        # Compute context vector as weighted sum
        # attention_weights: (batch_size, seq_length, 1) after unsqueeze
        # hidden_states: (batch_size, seq_length, hidden_dim)
        context_vector = torch.sum(
            attention_weights.unsqueeze(-1) * hidden_states, 
            dim=1
        )  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights


class SelfAttention(nn.Module):
    """
    Self-attention mechanism where the last hidden state generates a query
    and all hidden states generate keys and values.
    
    This implements scaled dot-product attention similar to Transformers,
    but specifically designed for RNN sequences where the last state is special.
    """
    def __init__(self, hidden_dim, attention_dim=None, use_bias=True):
        """
        Args:
            hidden_dim (int): Dimension of the hidden states from RNN
            attention_dim (int, optional): Dimension of the attention space. 
                                         If None, defaults to hidden_dim.
            use_bias (bool): Whether to use bias in the linear transformations.
                           Defaults to True.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim
        self.use_bias = use_bias
        
        # Linear transformations for Query, Key, and Value
        self.W_q = nn.Linear(hidden_dim, self.attention_dim, bias=use_bias)  # Query from last state
        self.W_k = nn.Linear(hidden_dim, self.attention_dim, bias=use_bias)  # Keys from all states
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)          # Values from all states
        
        # Scaling factor for dot-product attention
        self.scale = self.attention_dim ** -0.5
        
    def forward(self, hidden_states):
        """
        Compute self-attention where last state queries all states.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from RNN of shape 
                                        (batch_size, seq_length, hidden_dim)
        
        Returns:
            enhanced_last_state (torch.Tensor): Last hidden state enhanced with 
                                              attended context, shape (batch_size, hidden_dim)
            attention_weights (torch.Tensor): Attention weights, 
                                            shape (batch_size, seq_length)
        """
        # Get the last hidden state for query
        last_state = hidden_states[:, -1, :]  # (batch_size, hidden_dim)
        
        # Generate Query from last state
        query = self.W_q(last_state)  # (batch_size, attention_dim)
        
        # Generate Keys and Values from all states
        keys = self.W_k(hidden_states)    # (batch_size, seq_length, attention_dim)
        values = self.W_v(hidden_states)  # (batch_size, seq_length, hidden_dim)
        
        # Compute attention scores: query @ keys^T
        # query: (batch_size, attention_dim) -> (batch_size, 1, attention_dim)
        # keys: (batch_size, seq_length, attention_dim)
        # scores: (batch_size, 1, seq_length)
        scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1)) * self.scale
        scores = scores.squeeze(1)  # (batch_size, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (batch_size, seq_length)
        
        # Compute weighted sum of values
        # attention_weights: (batch_size, seq_length, 1) after unsqueeze
        # values: (batch_size, seq_length, hidden_dim)
        # attended_context: (batch_size, hidden_dim)
        attended_context = torch.sum(
            attention_weights.unsqueeze(-1) * values, 
            dim=1
        )
        
        # Add attended context to the last hidden state
        enhanced_last_state = last_state + attended_context
        
        return enhanced_last_state, attention_weights


class SoilCNN(nn.Module):
    """
    CNN branch for processing soil features.
    Processes soil features tensor of shape (batch_size, features, depth) through
    convolutional layers followed by max pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, n_conv_blocks=1):
        """
        Args:
            in_channels (int): Number of input soil features (channels).
            out_channels (int): Number of output channels from the CNN.
            kernel_size (int): Kernel size for the 1D convolution layers.
            n_conv_blocks (int): Number of conv blocks (1 or 2). Defaults to 1.
        """
        super().__init__()
        self.n_conv_blocks = n_conv_blocks
        
        if n_conv_blocks not in [1, 2]:
            raise ValueError(f"n_conv_blocks must be 1 or 2, got {n_conv_blocks}")
        
        # Build conv blocks
        conv_layers = []
        
        if n_conv_blocks == 1:
            # Single conv block
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
        else:  # n_conv_blocks == 2
            # First conv block
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
            # Second conv block
            conv_layers.extend([
                nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
        
        # Add max pooling
        conv_layers.append(nn.AdaptiveMaxPool1d(1))
        
        self.cnn = nn.Sequential(*conv_layers)
    
    def forward(self, x):
        """
        Forward pass of the CNN branch.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features, depth).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels).
        """
        # x: (batch_size, features, depth)
        cnn_out = self.cnn(x)  # (batch_size, out_channels, 1)
        return cnn_out.squeeze(-1)  # (batch_size, out_channels)
