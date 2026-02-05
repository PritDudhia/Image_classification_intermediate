"""
Evaluation metrics for classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict, Optional


class MetricTracker:
    """
    Track and compute metrics during training.
    
    Intermediate concepts:
    - Top-k accuracy
    - Precision, Recall, F1
    - Confusion matrix
    """
    
    def __init__(self, num_classes: int):
        """
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            loss: Loss value for this batch
        """
        # Convert to numpy and store
        preds = predictions.detach().cpu().argmax(dim=1).numpy()
        targs = targets.detach().cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targs)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Precision, Recall, F1 (weighted average)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            predictions,
            average='weighted',
            zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(
                targets,
                predictions,
                average=None,
                zero_division=0
            )
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }
        
        # Add per-class metrics
        for i in range(min(len(precision_per_class), self.num_classes)):
            metrics[f'precision_class_{i}'] = precision_per_class[i]
            metrics[f'recall_class_{i}'] = recall_per_class[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return confusion_matrix(targets, predictions, labels=range(self.num_classes))
    
    def get_classification_report(self) -> str:
        """
        Get detailed classification report.
        
        Returns:
            Classification report string
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets,
            predictions,
            labels=range(self.num_classes),
            zero_division=0
        )


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.
    
    Intermediate concept: Top-k metrics
    
    Checks if the correct class is in the top-k predictions.
    Useful for large number of classes.
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred_top_k = predictions.topk(k, dim=1, largest=True, sorted=True)
        pred_top_k = pred_top_k.t()
        correct = pred_top_k.eq(targets.view(1, -1).expand_as(pred_top_k))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        return (correct_k / batch_size).item()


def calibration_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Advanced concept: Model calibration
    
    Measures how well predicted probabilities match actual outcomes.
    Lower is better (perfectly calibrated = 0).
    
    Args:
        predictions: Model predictions [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        n_bins: Number of bins for calibration
    
    Returns:
        Expected Calibration Error
    """
    with torch.no_grad():
        # Get probabilities and predicted classes
        probs = torch.softmax(predictions, dim=1)
        confidences, pred_classes = torch.max(probs, dim=1)
        
        # Check if predictions are correct
        correct = pred_classes.eq(targets)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1, device=predictions.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) * (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                # Accuracy in this bin
                accuracy_in_bin = correct[in_bin].float().mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # Add to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Simple utility for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
