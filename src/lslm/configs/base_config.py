"""Base configuration classes for LSLM with validation."""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Core architecture
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 30522
    
    # Dropout and regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Position and embeddings
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    
    # Special tokens
    bos_token_id: int = 101  # [CLS]
    eos_token_id: int = 102  # [SEP]
    pad_token_id: int = 0    # [PAD]
    unk_token_id: int = 100  # [UNK]
    
    # LSLM specific
    fusion_strategy: str = "cross_attention"  # early, middle, late, cross_attention
    audio_hidden_size: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.audio_hidden_size is None:
            self.audio_hidden_size = self.hidden_size
            
        # Validation
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive" 
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size must be divisible by num_attention_heads"
        assert self.fusion_strategy in ["early", "middle", "late", "cross_attention"], \
            f"Invalid fusion_strategy: {self.fusion_strategy}"


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # Text processing
    max_seq_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # Audio processing  
    audio_sample_rate: int = 16000
    mel_bins: int = 80
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 512
    
    # Data augmentation
    time_masking: bool = True
    freq_masking: bool = True
    noise_prob: float = 0.1
    speed_perturbation: bool = True
    
    # Dataset
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    def __post_init__(self):
        """Validate data configuration."""
        assert self.max_seq_length > 0, "max_seq_length must be positive"
        assert self.audio_sample_rate > 0, "audio_sample_rate must be positive"
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Data splits must sum to 1.0"
        assert self.padding in ["max_length", "longest", "do_not_pad"], \
            f"Invalid padding strategy: {self.padding}"


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    
    # Training loop
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    
    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Mixed precision and optimization
    fp16: bool = False
    bf16: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_train_epochs > 0, "num_train_epochs must be positive"
        assert self.per_device_train_batch_size > 0, "batch_size must be positive"
        assert self.lr_scheduler_type in ["linear", "cosine", "constant", "polynomial"], \
            f"Invalid lr_scheduler_type: {self.lr_scheduler_type}"


@dataclass  
class LSLMConfig:
    """Complete LSLM configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig) 
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment tracking
    experiment_name: str = "lslm_experiment"
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def __post_init__(self):
        """Validate complete configuration."""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
            
        # Set device automatically
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu" 