#!/usr/bin/env python3
"""Evaluation script for LSLM model."""

import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from lslm.models.lslm import LSLMModel
from lslm.data.dataset import LSLMDataset
from lslm.configs.default_config import LSLMConfig
from lslm.utils.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSLM model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to test data")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save results")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()

def load_model(checkpoint_path):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]
    model = LSLMModel(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config

def evaluate(model, data_loader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            audio_features = batch["audio_features"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            
            # Generate predictions
            predictions = model.generate(audio_features)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(text_input_ids.cpu().numpy())
    
    return all_predictions, all_targets

def main():
    args = parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path)
    model.to(device)
    
    # Load data
    test_dataset = LSLMDataset(data_dir, config.data, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LSLMDataset.collate_fn
    )
    
    # Evaluate
    predictions, targets = evaluate(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    
    # Save results
    results = {
        "metrics": metrics,
        "config": vars(config),
        "checkpoint": str(checkpoint_path)
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 