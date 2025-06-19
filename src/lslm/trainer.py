"""trainer for LSLM with state-of-the-art techniques."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import wandb
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from .models import LSLMModel
from .configs import LSLMConfig
from .utils.logging import TrainingLogger, get_logger
from .utils.metrics import LSLMMetrics, RealTimeMetrics
from .data.augmentation import AdvancedDataAugmentation


@dataclass
class TrainingConfig:
    """training configuration."""
    # Basic training settings
    num_epochs: int = 100
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Learning rate and optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # linear, cosine, polynomial
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    
    # Mixed precision training
    use_amp: bool = True
    amp_loss_scale: str = "dynamic"
    
    # Regularization
    label_smoothing: float = 0.1
    dropout_prob: float = 0.1
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # steps, epoch, no
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4
    
    # Distributed training
    local_rank: int = -1
    world_size: int = 1
    
    # Advanced techniques
    use_gradient_checkpointing: bool = False
    use_data_augmentation: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Logging and monitoring
    logging_steps: int = 50
    log_level: str = "INFO"
    use_wandb: bool = True
    wandb_project: str = "lslm"
    
    # Experiment settings
    experiment_name: str = "lslm_experiment"
    output_dir: str = "./outputs"
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        threshold: float = 1e-4,
        mode: str = "min"
    ):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "min" and score < self.best_score - self.threshold:
            self.best_score = score
            self.counter = 0
        elif self.mode == "max" and score > self.best_score + self.threshold:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
                
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup.clear()


class LSLMTrainer:
    """Advanced trainer for LSLM with cutting-edge techniques."""
    
    def __init__(
        self,
        model: LSLMModel,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        model_config: Optional[LSLMConfig] = None
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.model_config = model_config
        
        # Setup device and distributed training
        self._setup_device_and_distributed()
        
        # Setup model
        self._setup_model()
        
        # Setup optimization
        self._setup_optimization()
        
        # Setup monitoring and logging
        self._setup_monitoring()
        
        # Setup advanced techniques
        self._setup_advanced_techniques()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
    def _setup_device_and_distributed(self):
        """Setup device and distributed training."""
        if self.config.local_rank != -1:
            # Distributed training
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(backend='nccl')
            self.device = torch.device(f'cuda:{self.config.local_rank}')
            self.is_distributed = True
            self.is_main_process = self.config.local_rank == 0
        else:
            # Single GPU or CPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.is_distributed = False
            self.is_main_process = True
            
    def _setup_model(self):
        """Setup model for training."""
        self.model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Wrap with DDP for distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            )
            
    def _setup_optimization(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total steps
        num_training_steps = len(self.train_dataloader) * self.config.num_epochs
        num_training_steps //= self.config.gradient_accumulation_steps
        
        # Scheduler
        if self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            self.scheduler = None
            
        # Mixed precision scaler
        if self.config.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
    def _setup_monitoring(self):
        """Setup logging and monitoring."""
        # Logger
        self.logger = TrainingLogger(
            self.config.experiment_name,
            log_level=self.config.log_level,
            log_dir=str(self.config.output_dir / "logs")
        )
        
        # Metrics
        self.metrics_calculator = LSLMMetrics(tokenizer=self.tokenizer)
        self.real_time_metrics = RealTimeMetrics()
        
        # Weights & Biases
        if self.config.use_wandb and self.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
            
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            threshold=self.config.early_stopping_threshold
        )
        
    def _setup_advanced_techniques(self):
        """Setup advanced training techniques."""
        # Exponential Moving Average
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(
                self.model,
                decay=self.config.ema_decay
            )
        else:
            self.ema = None
            
        # Data augmentation
        if self.config.use_data_augmentation:
            self.augmentation = AdvancedDataAugmentation(
                audio_config={"sample_rate": 16000},
                text_config={"vocab_size": 30522},
                mixed_config={"mixup_alpha": 0.2}
            )
        else:
            self.augmentation = None
            
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training", config=self.config.__dict__)
        
        # Log model info
        self.logger.log_model_info(self.model)
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Evaluation
            if self.eval_dataloader and self._should_evaluate():
                eval_metrics = self._evaluate()
                
                # Early stopping check
                if self.early_stopping(eval_metrics.get('loss', float('inf'))):
                    self.logger.info("Early stopping triggered")
                    break
                    
                # Save best model
                if eval_metrics.get('loss', float('inf')) < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self._save_checkpoint("best_model")
                    
            # Save periodic checkpoint
            if self._should_save():
                self._save_checkpoint(f"checkpoint_epoch_{epoch}")
                
        # Save final model
        self._save_checkpoint("final_model")
        
        # Save training history
        self.logger.save_training_history()
        
        # Cleanup
        if self.config.use_wandb and self.is_main_process:
            wandb.finish()
            
        self.logger.info("Training completed")
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        num_steps = 0
        
        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Data augmentation
            if self.augmentation:
                batch = self.augmentation(**batch)
                
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps
                    
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Optimization step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                    
                # EMA update
                if self.ema:
                    self.ema.update(self.model)
                    
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_step(loss.item())
                    
            epoch_loss += loss.item()
            num_steps += 1
            
        return {"loss": epoch_loss / num_steps}
        
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        # Use EMA parameters if available
        if self.ema:
            self.ema.apply_shadow(self.model)
            
        eval_loss = 0.0
        num_steps = 0
        
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                eval_loss += loss.item()
                num_steps += 1
                
                # Collect metrics
                if hasattr(outputs, 'predictions'):
                    self.metrics_calculator.add_batch(
                        predictions=outputs.predictions,
                        references=batch.get('labels'),
                        loss=loss.item()
                    )
                    
        # Restore original parameters
        if self.ema:
            self.ema.restore(self.model)
            
        # Compute metrics
        eval_metrics = self.metrics_calculator.compute_all_metrics()
        eval_metrics['loss'] = eval_loss / num_steps
        
        # Log evaluation results
        self.logger.log_validation_results(self.epoch, eval_metrics)
        
        if self.config.use_wandb and self.is_main_process:
            wandb.log({f"eval_{k}": v for k, v in eval_metrics.items()})
            
        return eval_metrics
        
    def _log_training_step(self, loss: float):
        """Log training step metrics."""
        lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
        
        metrics = {
            "loss": loss,
            "learning_rate": lr,
            "epoch": self.epoch,
            "global_step": self.global_step
        }
        
        self.logger.log_training_metrics(
            self.global_step,
            self.epoch,
            metrics
        )
        
        if self.config.use_wandb and self.is_main_process:
            wandb.log(metrics)
            
    def _should_evaluate(self) -> bool:
        """Check if should evaluate."""
        if self.config.eval_strategy == "steps":
            return self.global_step % self.config.eval_steps == 0
        elif self.config.eval_strategy == "epoch":
            return True
        else:
            return False
            
    def _should_save(self) -> bool:
        """Check if should save checkpoint."""
        return self.global_step % self.config.save_steps == 0
        
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        if not self.is_main_process:
            return
            
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state dict
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
            
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config,
            "model_config": self.model_config
        }
        
        # Save EMA state
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.shadow
            
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
        
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints to save space."""
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > self.config.save_total_limit:
            # Sort by modification time
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")


def main():
    """Example training script."""
    # This would be called from a separate training script
    pass


if __name__ == "__main__":
    main() 