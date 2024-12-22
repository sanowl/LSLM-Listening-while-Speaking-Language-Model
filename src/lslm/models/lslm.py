"""LSLM model implementation."""

import torch
import torch.nn as nn
from transformers import PreTrainedModel

class LSLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Audio encoder
        self.audio_encoder = nn.ModuleDict({
            'conv_layers': nn.Sequential(
                nn.Conv1d(config.mel_bins, config.hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob)
            ),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size)
        })
        
        # Text encoder
        self.text_encoder = nn.ModuleDict({
            'embeddings': nn.Embedding(config.vocab_size, config.hidden_size),
            'position_embeddings': nn.Embedding(config.max_position_embeddings, config.hidden_size),
            'layer_norm': nn.LayerNorm(config.hidden_size),
            'dropout': nn.Dropout(config.hidden_dropout_prob)
        })
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                config.hidden_size,
                config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob
            ) for _ in range(config.num_hidden_layers)
        ])
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
    def forward(self, audio_features, text_input_ids, attention_mask=None):
        # Process audio input
        audio_encoded = self.audio_encoder['conv_layers'](audio_features)
        audio_pos = torch.arange(audio_encoded.size(1), device=audio_encoded.device)
        audio_pos_emb = self.audio_encoder['position_embeddings'](audio_pos)
        audio_encoded = audio_encoded + audio_pos_emb
        
        # Process text input
        text_emb = self.text_encoder['embeddings'](text_input_ids)
        text_pos = torch.arange(text_emb.size(1), device=text_emb.device)
        text_pos_emb = self.text_encoder['position_embeddings'](text_pos)
        text_encoded = text_emb + text_pos_emb
        text_encoded = self.text_encoder['layer_norm'](text_encoded)
        text_encoded = self.text_encoder['dropout'](text_encoded)
        
        # Cross attention
        hidden_states = text_encoded
        for layer in self.cross_attention:
            hidden_states = layer(
                hidden_states,
                audio_encoded,
                audio_encoded,
                attn_mask=attention_mask
            )[0]
        
        # Output prediction
        logits = self.output(hidden_states)
        
        return logits

    def generate(self, audio_features, max_length=50, temperature=1.0):
        """Generate text from audio features."""
        device = next(self.parameters()).device
        batch_size = audio_features.size(0)
        
        # Start with empty sequence
        current_ids = torch.full((batch_size, 1), self.config.bos_token_id, device=device)
        
        for _ in range(max_length):
            # Get predictions
            logits = self.forward(audio_features, current_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for end of sequence
            if (next_token == self.config.eos_token_id).all():
                break
                
        return current_ids 