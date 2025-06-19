"""Tests for configuration system."""

import pytest
import tempfile
import yaml
from pathlib import Path

from lslm.configs import (
    LSLMConfig, ModelConfig, TrainingConfig, DataConfig,
    load_config, save_config, validate_config, merge_configs
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_attention_heads == 12
        assert config.vocab_size == 30522
        
    def test_validation(self):
        """Test configuration validation."""
        # Valid config should pass
        config = ModelConfig(hidden_size=768, num_attention_heads=12)
        # Should not raise
        
        # Invalid config should fail
        with pytest.raises(AssertionError):
            ModelConfig(hidden_size=0)
            
        with pytest.raises(AssertionError):
            ModelConfig(hidden_size=100, num_attention_heads=7)  # Not divisible
            
    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            hidden_size=512,
            num_attention_heads=8,
            vocab_size=50000
        )
        assert config.hidden_size == 512
        assert config.num_attention_heads == 8
        assert config.vocab_size == 50000


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_config(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.max_seq_length == 512
        assert config.audio_sample_rate == 16000
        assert config.mel_bins == 80
        
    def test_split_validation(self):
        """Test data split validation."""
        # Valid splits
        config = DataConfig(train_split=0.8, val_split=0.1, test_split=0.1)
        # Should not raise
        
        # Invalid splits
        with pytest.raises(AssertionError):
            DataConfig(train_split=0.5, val_split=0.3, test_split=0.3)  # Sum > 1
            
    def test_positive_values(self):
        """Test positive value validation."""
        with pytest.raises(AssertionError):
            DataConfig(max_seq_length=0)
            
        with pytest.raises(AssertionError):
            DataConfig(audio_sample_rate=-1000)


class TestTrainingConfig:
    """Test TrainingConfig class."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.learning_rate == 5e-5
        assert config.num_train_epochs == 10
        assert config.per_device_train_batch_size == 8
        
    def test_validation(self):
        """Test training configuration validation."""
        # Valid config
        config = TrainingConfig(learning_rate=1e-4, num_train_epochs=5)
        # Should not raise
        
        # Invalid configs
        with pytest.raises(AssertionError):
            TrainingConfig(learning_rate=0)
            
        with pytest.raises(AssertionError):
            TrainingConfig(num_train_epochs=0)
            
        with pytest.raises(AssertionError):
            TrainingConfig(per_device_train_batch_size=0)


class TestLSLMConfig:
    """Test complete LSLM configuration."""
    
    def test_default_config(self):
        """Test default LSLM configuration."""
        config = LSLMConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert config.experiment_name == "lslm_experiment"
        
    def test_device_auto_detection(self):
        """Test automatic device detection."""
        config = LSLMConfig(device="auto")
        assert config.device in ["cuda", "mps", "cpu"]
        
    def test_custom_config(self):
        """Test custom configuration creation."""
        model_config = ModelConfig(hidden_size=512)
        data_config = DataConfig(max_seq_length=256)
        training_config = TrainingConfig(learning_rate=1e-4)
        
        config = LSLMConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            experiment_name="test_experiment"
        )
        
        assert config.model.hidden_size == 512
        assert config.data.max_seq_length == 256
        assert config.training.learning_rate == 1e-4
        assert config.experiment_name == "test_experiment"


class TestConfigIO:
    """Test configuration I/O operations."""
    
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        config = LSLMConfig(experiment_name="test_save_load")
        
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = Path(f.name)
            
        try:
            # Save config
            save_config(config, temp_path)
            assert temp_path.exists()
            
            # Load config
            loaded_config = load_config(temp_path)
            assert loaded_config.experiment_name == "test_save_load"
            assert loaded_config.model.hidden_size == config.model.hidden_size
            
        finally:
            temp_path.unlink(missing_ok=True)
            
    def test_load_yaml_config(self):
        """Test loading from YAML file."""
        config_data = {
            'model': {
                'hidden_size': 512,
                'num_attention_heads': 8
            },
            'training': {
                'learning_rate': 1e-4,
                'num_train_epochs': 5
            },
            'experiment_name': 'yaml_test'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
            
        try:
            config = load_config(temp_path)
            assert config.model.hidden_size == 512
            assert config.model.num_attention_heads == 8
            assert config.training.learning_rate == 1e-4
            assert config.experiment_name == 'yaml_test'
            
        finally:
            temp_path.unlink(missing_ok=True)
            
    def test_invalid_file_format(self):
        """Test loading invalid file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)
            
        try:
            with pytest.raises(ValueError):
                load_config(temp_path)
                
        finally:
            temp_path.unlink(missing_ok=True)


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = LSLMConfig()
        assert validate_config(config) is True
        
    def test_validate_invalid_config(self):
        """Test validation of invalid configuration."""
        config = LSLMConfig()
        config.model.hidden_size = 0  # Invalid
        
        with pytest.raises(ValueError):
            validate_config(config)


class TestConfigMerging:
    """Test configuration merging."""
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = LSLMConfig()
        override_config = {
            'model': {'hidden_size': 1024},
            'training': {'learning_rate': 1e-3},
            'experiment_name': 'merged_experiment'
        }
        
        merged_config = merge_configs(base_config, override_config)
        
        assert merged_config.model.hidden_size == 1024
        assert merged_config.training.learning_rate == 1e-3
        assert merged_config.experiment_name == 'merged_experiment'
        # Other values should remain from base
        assert merged_config.model.num_attention_heads == base_config.model.num_attention_heads 