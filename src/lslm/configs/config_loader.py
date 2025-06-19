"""Configuration loading and validation utilities."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union
import logging

from .base_config import LSLMConfig, ModelConfig, TrainingConfig, DataConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> LSLMConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        LSLMConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    # Load configuration data
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    # Parse configuration sections
    model_config = ModelConfig(**config_data.get('model', {}))
    data_config = DataConfig(**config_data.get('data', {}))
    training_config = TrainingConfig(**config_data.get('training', {}))
    
    # Create main config
    main_config_data = {k: v for k, v in config_data.items() 
                       if k not in ['model', 'data', 'training']}
    
    config = LSLMConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        **main_config_data
    )
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: LSLMConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        'model': config.model.__dict__,
        'data': config.data.__dict__,
        'training': config.training.__dict__,
        'experiment_name': config.experiment_name,
        'output_dir': config.output_dir,
        'logging_dir': config.logging_dir,
        'run_name': config.run_name,
        'device': config.device
    }
    
    # Save as YAML
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {save_path}")


def validate_config(config: LSLMConfig) -> bool:
    """
    Validate configuration for consistency and correctness.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        # Check model configuration
        if config.model.hidden_size <= 0:
            raise ValueError("Model hidden_size must be positive")
            
        if config.model.hidden_size % config.model.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
            
        # Check data configuration
        if config.data.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
            
        if config.data.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive")
            
        # Check training configuration
        if config.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        if config.training.per_device_train_batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        # Check compatibility
        effective_batch_size = (
            config.training.per_device_train_batch_size * 
            config.training.gradient_accumulation_steps
        )
        
        if effective_batch_size > 1024:
            logger.warning(f"Large effective batch size: {effective_batch_size}")
            
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def merge_configs(base_config: LSLMConfig, override_config: Dict[str, Any]) -> LSLMConfig:
    """
    Merge override configuration into base configuration.
    
    Args:
        base_config: Base configuration
        override_config: Override values
        
    Returns:
        Merged configuration
    """
    # Convert base config to dict
    base_dict = {
        'model': base_config.model.__dict__.copy(),
        'data': base_config.data.__dict__.copy(),
        'training': base_config.training.__dict__.copy(),
        'experiment_name': base_config.experiment_name,
        'output_dir': base_config.output_dir,
        'logging_dir': base_config.logging_dir,
        'run_name': base_config.run_name,
        'device': base_config.device
    }
    
    # Apply overrides
    for key, value in override_config.items():
        if key in ['model', 'data', 'training'] and isinstance(value, dict):
            base_dict[key].update(value)
        else:
            base_dict[key] = value
    
    # Create new config
    model_config = ModelConfig(**base_dict['model'])
    data_config = DataConfig(**base_dict['data'])
    training_config = TrainingConfig(**base_dict['training'])
    
    main_config_data = {k: v for k, v in base_dict.items() 
                       if k not in ['model', 'data', 'training']}
    
    return LSLMConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        **main_config_data
    ) 