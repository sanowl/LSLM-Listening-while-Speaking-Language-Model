"""Default configuration for LSLM model."""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model architecture
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: int = 0.1
    attention_probs_dropout_prob: int = 0.1
    max_position_embeddings: int = 512

@dataclass
class TrainingConfig:
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 5e-5
    max_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
@dataclass
class DataConfig:
    # Data processing
    max_seq_length: int = 512
    audio_sample_rate: int = 16000
    mel_bins: int = 80
    hop_length: int = 160
    
@dataclass
class LSLMConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig() 