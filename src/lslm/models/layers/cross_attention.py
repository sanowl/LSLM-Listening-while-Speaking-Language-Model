"""Advanced cross-attention mechanisms for LSLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for audio-text fusion."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_states: Query tensor (batch_size, seq_len_q, hidden_size)
            key_states: Key tensor (batch_size, seq_len_k, hidden_size)
            value_states: Value tensor (batch_size, seq_len_v, hidden_size)
            attention_mask: Mask tensor (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            Tuple of (context_layer, attention_probs)
        """
        # Project to Q, K, V
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        
        # Transpose for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for heads
            attention_mask = attention_mask.unsqueeze(1).expand(
                -1, self.num_attention_heads, -1, -1
            )
            attention_scores = attention_scores + attention_mask
            
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # Transpose back and reshape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Final projection and residual connection
        context_layer = self.dense(context_layer)
        context_layer = self.layer_norm(context_layer + query_states)
        
        return context_layer, attention_probs


class CrossAttentionLayer(nn.Module):
    """Complete cross-attention layer with feed-forward network."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        # Cross-attention
        self.cross_attention = MultiHeadCrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_states: Query tensor (typically text features)
            key_states: Key tensor (typically audio features)
            value_states: Value tensor (typically audio features)
            attention_mask: Attention mask
            
        Returns:
            Tuple of (output_states, attention_probs)
        """
        # Cross-attention
        attention_output, attention_probs = self.cross_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.gelu(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm(layer_output + attention_output)
        
        return layer_output, attention_probs


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention for audio-text interaction."""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        attention_probs_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        
        # Text-to-Audio attention
        self.text_to_audio = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        
        # Audio-to-Text attention
        self.audio_to_text = CrossAttentionLayer(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        
    def forward(
        self,
        text_states: torch.Tensor,
        audio_states: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_states: Text features (batch_size, text_len, hidden_size)
            audio_states: Audio features (batch_size, audio_len, hidden_size)
            text_attention_mask: Text attention mask
            audio_attention_mask: Audio attention mask
            
        Returns:
            Tuple of (enhanced_text, enhanced_audio, text_attn_probs, audio_attn_probs)
        """
        # Text attending to Audio
        enhanced_text, text_attn_probs = self.text_to_audio(
            query_states=text_states,
            key_states=audio_states,
            value_states=audio_states,
            attention_mask=audio_attention_mask
        )
        
        # Audio attending to Text
        enhanced_audio, audio_attn_probs = self.audio_to_text(
            query_states=audio_states,
            key_states=text_states,
            value_states=text_states,
            attention_mask=text_attention_mask
        )
        
        return enhanced_text, enhanced_audio, text_attn_probs, audio_attn_probs 