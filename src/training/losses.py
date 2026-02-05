"""
Custom loss functions for advanced training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Intermediate concept: Regularization via label smoothing
    
    Instead of hard targets [0, 1, 0], use soft targets [ε/K, 1-ε+ε/K, ε/K]
    where ε is smoothing factor and K is number of classes.
    
    Benefits:
    - Prevents overconfidence
    - Better generalization
    - Reduces overfitting
    
    Paper: "Rethinking the Inception Architecture" (Szegedy et al., 2016)
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (typically 0.1)
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch_size, num_classes]
            target: Ground truth class indices [batch_size]
        
        Returns:
            Loss value
        """
        num_classes = pred.size(-1)
        
        # Convert to log probabilities
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # KL divergence loss
        loss = torch.sum(-true_dist * log_probs, dim=-1).mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Advanced concept: Handling imbalanced datasets
    
    Down-weights easy examples and focuses on hard negatives.
    Useful when you have severe class imbalance.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (typically 2.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [batch_size, num_classes]
            target: Ground truth class indices [batch_size]
        
        Returns:
            Loss value
        """
        # Get probabilities
        probs = F.softmax(pred, dim=-1)
        
        # Get probability of true class
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Cross-entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            focal_loss = alpha_weight * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss.
    
    Advanced concept: Model compression and knowledge transfer
    
    Train a student model to mimic a teacher model.
    Combines:
    1. Hard loss: Student vs ground truth
    2. Soft loss: Student vs teacher (with temperature)
    
    Paper: "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,
        base_criterion: nn.Module = nn.CrossEntropyLoss()
    ):
        """
        Args:
            temperature: Softmax temperature for distillation
            alpha: Balance between hard and soft loss (0-1)
            base_criterion: Loss for hard targets
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.base_criterion = base_criterion
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            targets: Ground truth labels
        
        Returns:
            Combined loss
        """
        # Hard loss (student vs ground truth)
        hard_loss = self.base_criterion(student_logits, targets)
        
        # Soft loss (student vs teacher with temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        soft_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss


def get_criterion(config: dict) -> nn.Module:
    """
    Factory function to get loss criterion based on config.
    
    Args:
        config: Loss configuration dictionary
    
    Returns:
        Loss criterion
    """
    loss_name = config.get('name', 'cross_entropy').lower()
    
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    
    elif loss_name == 'label_smoothing':
        smoothing = config.get('label_smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    elif loss_name == 'focal_loss':
        gamma = config.get('gamma', 2.0)
        return FocalLoss(gamma=gamma)
    
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


# Example usage and theory
"""
When to use each loss:

1. Cross-Entropy:
   - Balanced datasets
   - Standard classification
   - Fast and stable

2. Label Smoothing:
   - Prevent overconfidence
   - Better calibration
   - Slight accuracy improvement
   - Use smoothing=0.1 as default

3. Focal Loss:
   - Severe class imbalance
   - Hard negative mining
   - Object detection tasks
   - Adjust gamma (higher = more focus on hard examples)

4. Distillation Loss:
   - Model compression
   - Knowledge transfer
   - Ensemble distillation
   - temperature=3-5 typically works well
"""
