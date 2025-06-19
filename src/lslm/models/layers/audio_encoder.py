"""Audio encoder implementation for LSLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class AudioConvBlock(nn.Module):
    """Convolutional block for audio processing."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Apply convolution
        x = self.conv(x)
        
        # Transpose for layer norm (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Transpose back (batch_size, channels, seq_len)
        return x.transpose(1, 2)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for audio sequences."""
    
    def __init__(self, d_model: int, max_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class AudioEncoder(nn.Module):
    """High-quality audio encoder with convolutional layers and attention."""
    
    def __init__(
        self,
        input_dim: int = 80,  # Mel bins
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_layers: int = 3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Convolutional layers for local feature extraction
        self.conv_layers = nn.ModuleList([
            AudioConvBlock(
                in_channels=hidden_dim if i > 0 else hidden_dim,
                out_channels=hidden_dim,
                dropout=dropout
            ) for i in range(conv_layers)
        ])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer layers for long-range dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: Mel spectrograms of shape (batch_size, mel_bins, time_steps)
            attention_mask: Optional mask of shape (batch_size, time_steps)
            
        Returns:
            Tuple of (encoded_features, attention_mask)
            encoded_features: (batch_size, time_steps, hidden_dim)
            attention_mask: (batch_size, time_steps)
        """
        batch_size, mel_bins, time_steps = audio_features.shape
        
        # Transpose to (batch_size, time_steps, mel_bins)
        x = audio_features.transpose(1, 2)
        
        # Project to hidden dimension
        x = self.input_projection(x)  # (batch_size, time_steps, hidden_dim)
        
        # Apply convolutional layers
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, time_steps)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.transpose(1, 2)  # (batch_size, time_steps, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, time_steps, 
                device=x.device, dtype=torch.bool
            )
        
        # Apply transformer layers
        # Convert attention mask to the format expected by transformer
        # (True means attend, False means don't attend)
        transformer_mask = ~attention_mask  # Invert mask
        
        x = self.transformer(x, src_key_padding_mask=transformer_mask)
        
        # Output normalization
        x = self.output_norm(x)
        
        return x, attention_mask
        
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.hidden_dim 