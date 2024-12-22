"""Training script for LSLM model."""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lslm.configs.default_config import LSLMConfig
from lslm.data.dataset import LSLMDataset
from lslm.models.lslm import LSLMModel
from lslm.utils.training import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSLM model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    config = LSLMConfig()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = LSLMModel(config.model)
    
    # Setup data
    train_dataset = LSLMDataset(args.data_dir, config.data, split="train")
    val_dataset = LSLMDataset(args.data_dir, config.data, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main() 