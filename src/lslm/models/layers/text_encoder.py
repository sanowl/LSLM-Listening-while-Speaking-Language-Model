"""Text encoder implementation for LSLM."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import BertModel, BertConfig


class TextEncoder(nn.Module):
    """High-quality text encoder based on BERT with custom modifications."""
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        use_pretrained: bool = True,
        pretrained_model_name: str = "bert-base-uncased"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Use pretrained BERT
            self.bert = BertModel.from_pretrained(
                pretrained_model_name,
                add_pooling_layer=False
            )
            # Ensure hidden size matches
            if self.bert.config.hidden_size != hidden_size:
                self.projection = nn.Linear(
                    self.bert.config.hidden_size, 
                    hidden_size
                )
            else:
                self.projection = nn.Identity()
        else:
            # Create BERT from scratch
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                max_position_embeddings=max_position_embeddings,
                layer_norm_eps=layer_norm_eps,
                pad_token_id=pad_token_id
            )
            self.bert = BertModel(config, add_pooling_layer=False)
            self.projection = nn.Identity()
            
        # Additional normalization layer
        self.output_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type IDs of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (encoded_features, attention_mask)
            encoded_features: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.bert.config.pad_token_id).long()
            
        # Encode text
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Apply projection if needed
        hidden_states = self.projection(hidden_states)
        
        # Apply output normalization
        hidden_states = self.output_norm(hidden_states)
        
        return hidden_states, attention_mask.bool()
        
    def get_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        return self.hidden_size
        
    def freeze_backbone(self) -> None:
        """Freeze the BERT backbone parameters."""
        if self.use_pretrained:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def unfreeze_backbone(self) -> None:
        """Unfreeze the BERT backbone parameters.""" 
        if self.use_pretrained:
            for param in self.bert.parameters():
                param.requires_grad = True 