"""evaluation metrics for LSLM."""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import editdistance
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LSLMMetrics:
    """Comprehensive metrics calculator for LSLM evaluation."""
    
    def __init__(
        self,
        tokenizer=None,
        rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL'],
        bleu_order: int = 4
    ):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        self.bleu_scorer = BLEU(effective_order=True)
        self.rouge_types = rouge_types
        self.bleu_order = bleu_order
        
        # Metrics storage
        self.reset()
        
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.references = []
        self.attention_weights = []
        self.losses = []
        self.perplexities = []
        
    def add_batch(
        self,
        predictions: torch.Tensor,
        references: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """Add a batch of predictions and references."""
        # Convert to lists
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy().tolist()
        if isinstance(references, torch.Tensor):
            references = references.cpu().numpy().tolist()
            
        self.predictions.extend(predictions)
        self.references.extend(references)
        
        if attention_weights is not None:
            self.attention_weights.append(attention_weights.cpu().numpy())
            
        if loss is not None:
            self.losses.append(loss)
            self.perplexities.append(np.exp(loss))
            
    def compute_text_metrics(self) -> Dict[str, float]:
        """Compute text generation metrics."""
        if not self.predictions or not self.references:
            return {}
            
        metrics = {}
        
        # Convert token IDs to text if tokenizer is available
        if self.tokenizer is not None:
            pred_texts = [
                self.tokenizer.decode(pred, skip_special_tokens=True) 
                for pred in self.predictions
            ]
            ref_texts = [
                self.tokenizer.decode(ref, skip_special_tokens=True)
                for ref in self.references
            ]
        else:
            pred_texts = self.predictions
            ref_texts = self.references
            
        # ROUGE scores
        rouge_scores = defaultdict(list)
        for pred, ref in zip(pred_texts, ref_texts):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                rouge_scores[f"{rouge_type}_f1"].append(scores[rouge_type].fmeasure)
                rouge_scores[f"{rouge_type}_precision"].append(scores[rouge_type].precision)
                rouge_scores[f"{rouge_type}_recall"].append(scores[rouge_type].recall)
                
        # Average ROUGE scores
        for key, values in rouge_scores.items():
            metrics[key] = np.mean(values)
            
        # BLEU score
        try:
            bleu_score = self.bleu_scorer.corpus_score(pred_texts, [ref_texts])
            metrics['bleu'] = bleu_score.score
        except Exception as e:
            logger.warning(f"Could not compute BLEU score: {e}")
            metrics['bleu'] = 0.0
            
        # Edit distance (character level)
        edit_distances = []
        for pred, ref in zip(pred_texts, ref_texts):
            edit_distances.append(editdistance.eval(pred, ref))
        metrics['edit_distance'] = np.mean(edit_distances)
        metrics['normalized_edit_distance'] = np.mean([
            ed / max(len(ref), 1) for ed, ref in zip(edit_distances, ref_texts)
        ])
        
        return metrics
        
    def compute_perplexity_metrics(self) -> Dict[str, float]:
        """Compute perplexity-based metrics."""
        if not self.losses:
            return {}
            
        return {
            'loss': np.mean(self.losses),
            'perplexity': np.mean(self.perplexities),
            'loss_std': np.std(self.losses),
            'perplexity_std': np.std(self.perplexities)
        }
        
    def compute_attention_metrics(self) -> Dict[str, float]:
        """Compute attention-based metrics."""
        if not self.attention_weights:
            return {}
            
        metrics = {}
        
        # Concatenate all attention weights
        all_attention = np.concatenate(self.attention_weights, axis=0)
        
        # Attention entropy (measure of focus vs. diffusion)
        attention_entropy = []
        for batch in all_attention:
            for head in batch:
                for seq in head:
                    # Compute entropy for each attention distribution
                    entropy = -np.sum(seq * np.log(seq + 1e-8))
                    attention_entropy.append(entropy)
                    
        metrics['attention_entropy_mean'] = np.mean(attention_entropy)
        metrics['attention_entropy_std'] = np.std(attention_entropy)
        
        # Attention sparsity (measure of how concentrated attention is)
        attention_sparsity = []
        for batch in all_attention:
            for head in batch:
                for seq in head:
                    # Gini coefficient as sparsity measure
                    sorted_attention = np.sort(seq)
                    n = len(sorted_attention)
                    index = np.arange(1, n + 1)
                    gini = 2 * np.sum(index * sorted_attention) / (n * np.sum(sorted_attention)) - (n + 1) / n
                    attention_sparsity.append(gini)
                    
        metrics['attention_sparsity_mean'] = np.mean(attention_sparsity)
        metrics['attention_sparsity_std'] = np.std(attention_sparsity)
        
        return metrics
        
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics."""
        all_metrics = {}
        
        # Text metrics
        text_metrics = self.compute_text_metrics()
        all_metrics.update(text_metrics)
        
        # Perplexity metrics
        perplexity_metrics = self.compute_perplexity_metrics()
        all_metrics.update(perplexity_metrics)
        
        # Attention metrics
        attention_metrics = self.compute_attention_metrics()
        all_metrics.update(attention_metrics)
        
        return all_metrics
        
    def plot_attention_heatmap(
        self, 
        attention_weights: torch.Tensor,
        input_tokens: List[str],
        output_tokens: List[str],
        save_path: Optional[str] = None
    ):
        """Plot attention heatmap visualization."""
        # Average over heads
        attention = attention_weights.mean(dim=0).cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            attention,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap='Blues',
            cbar=True
        )
        plt.title('Cross-Modal Attention Heatmap')
        plt.xlabel('Input Tokens (Audio/Text)')
        plt.ylabel('Output Tokens')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_metrics_over_time(
        self,
        metrics_history: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Plot metrics evolution over training."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metric_names = list(metrics_history.keys())[:4]  # Plot first 4 metrics
        
        for i, metric_name in enumerate(metric_names):
            if i < len(axes):
                axes[i].plot(metrics_history[metric_name])
                axes[i].set_title(f'{metric_name.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch/Step')
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class RealTimeMetrics:
    """Real-time metrics for streaming evaluation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_buffer = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            self.metrics_buffer[key].append(value)
            # Keep only recent values
            if len(self.metrics_buffer[key]) > self.window_size:
                self.metrics_buffer[key].pop(0)
                
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current moving average metrics."""
        current_metrics = {}
        for key, values in self.metrics_buffer.items():
            if values:
                current_metrics[f"{key}_avg"] = np.mean(values)
                current_metrics[f"{key}_std"] = np.std(values)
        return current_metrics
        
    def get_trend(self, metric_name: str, window: int = 10) -> str:
        """Get trend direction for a metric."""
        if metric_name not in self.metrics_buffer:
            return "unknown"
            
        values = self.metrics_buffer[metric_name]
        if len(values) < window:
            return "insufficient_data"
            
        recent = np.mean(values[-window:])
        previous = np.mean(values[-2*window:-window])
        
        if recent > previous * 1.05:
            return "improving"
        elif recent < previous * 0.95:
            return "degrading"
        else:
            return "stable"


def compute_metrics(predictions: List, targets: List, tokenizer=None) -> Dict[str, float]:
    """Convenience function to compute all metrics."""
    metrics_calculator = LSLMMetrics(tokenizer=tokenizer)
    
    for pred, target in zip(predictions, targets):
        metrics_calculator.add_batch([pred], [target])
        
    return metrics_calculator.compute_all_metrics()


def compare_models(
    model_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> Dict[str, str]:
    """Compare multiple models and determine the best for each metric."""
    best_models = {}
    
    # Get all metrics
    all_metrics = set()
    for results in model_results.values():
        all_metrics.update(results.keys())
        
    # Find best model for each metric
    for metric in all_metrics:
        best_score = float('-inf')
        best_model = None
        
        for model_name, results in model_results.items():
            if metric in results:
                score = results[metric]
                # For loss-based metrics, lower is better
                if 'loss' in metric or 'distance' in metric:
                    score = -score
                    
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    
        if best_model:
            best_models[metric] = best_model
            
    # Create comparison visualization
    if save_path:
        metrics_df = []
        for model_name, results in model_results.items():
            for metric, value in results.items():
                metrics_df.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': value
                })
                        
    return best_models 