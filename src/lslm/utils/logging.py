"""logging and monitoring system for LSLM."""

import logging
import logging.handlers
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import psutil
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    

class LSLMLogger:
    """logger for LSLM with structured logging and monitoring."""
    
    def __init__(
        self,
        name: str = "lslm",
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        structured_logging: bool = True,
        max_log_size_mb: int = 100,
        backup_count: int = 5
    ):
        self.name = name
        self.structured_logging = structured_logging
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        if log_to_console:
            self._setup_console_handler()
            
        if log_to_file:
            self._setup_file_handler(max_log_size_mb, backup_count)
            
        # Metrics tracking
        self.metrics = []
        self.start_time = time.time()
        
        # Performance monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
    def _setup_file_handler(self, max_size_mb: int, backup_count: int):
        """Setup rotating file handler."""
        log_file = self.log_dir / f"{self.name}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log(logging.INFO, message, kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log(logging.DEBUG, message, kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log(logging.WARNING, message, kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        self._log(logging.ERROR, message, kwargs)
        
    def critical(self, message: str, **kwargs):
        """Log critical message with optional structured data."""
        self._log(logging.CRITICAL, message, kwargs)
        
    def _log(self, level: int, message: str, extra_data: Dict[str, Any]):
        """Internal logging method."""
        if self.structured_logging and extra_data:
            # Create structured log record
            extra_data["message"] = message
            extra_data["timestamp"] = datetime.utcnow().isoformat()
            self.logger.log(level, json.dumps(extra_data))
        else:
            self.logger.log(level, message)
            
    def log_training_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        **metrics
    ):
        """Log training step information."""
        self.info(
            f"Training step {step}",
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            **metrics
        )
        
    def log_validation_results(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Log validation results."""
        self.info(
            f"Validation epoch {epoch}",
            epoch=epoch,
            validation_metrics=metrics
        )
        
    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(
            "Model information",
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=total_params * 4 / (1024 * 1024)  # Assuming float32
        )
        
    def log_system_metrics(self):
        """Log current system metrics."""
        metrics = self._collect_system_metrics()
        self.metrics.append(metrics)
        
        self.info(
            "System metrics",
            **asdict(metrics)
        )
        
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3)
        )
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                metrics.gpu_memory_used_gb = gpu_memory["allocated_bytes.all.current"] / (1024**3)
                metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                metrics.gpu_utilization = torch.cuda.utilization()
            except Exception:
                pass  # GPU metrics not available
                
        return metrics
        
    def start_monitoring(self, interval: float = 30.0):
        """Start background system monitoring."""
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                self.log_system_metrics()
                time.sleep(interval)
                
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        
        self.info("Started system monitoring", interval=interval)
        
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        self.info("Stopped system monitoring")
        
    def save_metrics(self, filepath: Optional[str] = None):
        """Save collected metrics to file."""
        if not filepath:
            filepath = self.log_dir / f"{self.name}_metrics.json"
            
        metrics_data = {
            "start_time": self.start_time,
            "end_time": time.time(),
            "total_duration": time.time() - self.start_time,
            "metrics": [asdict(m) for m in self.metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        self.info(f"Saved metrics to {filepath}")
        
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time operations."""
        start_time = time.time()
        self.debug(f"Starting {operation_name}")
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.info(
                f"Completed {operation_name}",
                operation=operation_name,
                duration_seconds=duration
            )


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # Try to parse message as JSON
            log_data = json.loads(record.getMessage())
        except (json.JSONDecodeError, ValueError):
            # Fallback to standard format
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "filename": record.filename,
                "lineno": record.lineno
            }
            
        return json.dumps(log_data)


class TrainingLogger(LSLMLogger):
    """Specialized logger for training workflows."""
    
    def __init__(self, experiment_name: str, **kwargs):
        super().__init__(name=f"lslm_training_{experiment_name}", **kwargs)
        self.experiment_name = experiment_name
        self.training_metrics = []
        self.validation_metrics = []
        
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        self.info(
            "Training hyperparameters",
            experiment=self.experiment_name,
            hyperparameters=config
        )
        
    def log_training_metrics(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Log training metrics."""
        metrics_with_meta = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics
        }
        
        self.training_metrics.append(metrics_with_meta)
        
        self.info(
            f"Training metrics - Step {step}",
            **metrics_with_meta
        )
        
    def log_validation_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ):
        """Log validation metrics."""
        metrics_with_meta = {
            "epoch": epoch,
            "timestamp": time.time(),
            **metrics
        }
        
        self.validation_metrics.append(metrics_with_meta)
        
        self.info(
            f"Validation metrics - Epoch {epoch}",
            **metrics_with_meta
        )
        
    def save_training_history(self):
        """Save training history to files."""
        # Save training metrics
        train_file = self.log_dir / f"{self.experiment_name}_training_metrics.json"
        with open(train_file, 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
            
        # Save validation metrics
        val_file = self.log_dir / f"{self.experiment_name}_validation_metrics.json"
        with open(val_file, 'w') as f:
            json.dump(self.validation_metrics, f, indent=2)
            
        self.info("Saved training history")


# Global logger instance
_global_logger = None


def get_logger(name: str = "lslm", **kwargs) -> LSLMLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = LSLMLogger(name=name, **kwargs)
        
    return _global_logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> LSLMLogger:
    """Setup logging for LSLM."""
    if experiment_name:
        logger = TrainingLogger(experiment_name, log_level=log_level, log_dir=log_dir)
    else:
        logger = LSLMLogger(log_level=log_level, log_dir=log_dir)
        
    # Start system monitoring for training
    if experiment_name:
        logger.start_monitoring()
        
    return logger 