"""
Training callbacks for advanced features.
"""

import torch
import numpy as np
from typing import Optional, Callable
from pathlib import Path


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Intermediate concept: Regularization via early stopping
    
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (whether lower or higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.is_better = self._get_comparator()
    
    def _get_comparator(self) -> Callable:
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return lambda current, best: current < (best - self.min_delta)
        else:
            return lambda current, best: current > (best + self.min_delta)
    
    def __call__(self, metric: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            metric: Current epoch metric value
        
        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.is_better(metric, self.best_score):
            # Improvement
            self.best_score = metric
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️  Early stopping triggered (patience={self.patience})")
                return True
        
        return False


class ModelCheckpoint:
    """
    Save model checkpoints during training.
    
    Intermediate concept: Model versioning and saving
    
    Saves best models based on monitored metric.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Number of best models to keep
            save_last: Whether to save the last model
            verbose: Print save messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose
        
        self.best_k_models = []
        self.is_better = self._get_comparator()
    
    def _get_comparator(self) -> Callable:
        """Get comparison function based on mode."""
        if self.mode == 'min':
            return lambda current, best: current < best
        else:
            return lambda current, best: current > best
    
    def __call__(
        self,
        model: torch.nn.Module,
        epoch: int,
        metrics: dict
    ):
        """
        Save checkpoint if model improved.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            print(f"Warning: Metric '{self.monitor}' not found in metrics")
            return
        
        # Check if this is a top-k model
        should_save = False
        
        if len(self.best_k_models) < self.save_top_k:
            should_save = True
        else:
            worst_best = self.best_k_models[-1][1]
            if self.is_better(current_score, worst_best):
                should_save = True
                # Remove worst model
                old_path = self.best_k_models.pop()[0]
                if old_path.exists():
                    old_path.unlink()
        
        if should_save:
            # Save model
            filename = f"epoch_{epoch}_{self.monitor}_{current_score:.4f}.pth"
            filepath = self.checkpoint_dir / filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, filepath)
            
            # Update best models list
            self.best_k_models.append((filepath, current_score))
            self.best_k_models.sort(
                key=lambda x: x[1],
                reverse=(self.mode == 'max')
            )
            
            if self.verbose:
                print(f"✓ Saved checkpoint: {filename}")
        
        # Save last model
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }, last_path)


class LearningRateScheduler:
    """
    Custom learning rate scheduling.
    
    Advanced concept: Learning rate schedules
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'cosine',
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-5,
        **kwargs
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            schedule_type: Type of schedule (cosine, step, exponential)
            warmup_epochs: Number of warmup epochs
            warmup_lr: Learning rate during warmup
            **kwargs: Additional schedule-specific parameters
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.current_epoch = 0
        
        # Get base learning rate
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Store kwargs for scheduler
        self.kwargs = kwargs
    
    def step(self, epoch: Optional[int] = None):
        """
        Update learning rate for current epoch.
        
        Args:
            epoch: Current epoch (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            lr = self._warmup_lr()
        else:
            # Normal schedule
            lr = self._scheduled_lr()
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        
        return lr
    
    def _warmup_lr(self) -> float:
        """Calculate learning rate during warmup."""
        # Linear warmup
        alpha = self.current_epoch / self.warmup_epochs
        return self.warmup_lr + (self.base_lr - self.warmup_lr) * alpha
    
    def _scheduled_lr(self) -> float:
        """Calculate learning rate for current epoch."""
        epoch = self.current_epoch - self.warmup_epochs
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            T_max = self.kwargs.get('T_max', 100)
            eta_min = self.kwargs.get('eta_min', 0)
            return eta_min + (self.base_lr - eta_min) * (
                1 + np.cos(np.pi * epoch / T_max)
            ) / 2
        
        elif self.schedule_type == 'step':
            # Step decay
            step_size = self.kwargs.get('step_size', 30)
            gamma = self.kwargs.get('gamma', 0.1)
            return self.base_lr * (gamma ** (epoch // step_size))
        
        elif self.schedule_type == 'exponential':
            # Exponential decay
            gamma = self.kwargs.get('gamma', 0.95)
            return self.base_lr * (gamma ** epoch)
        
        else:
            return self.base_lr


class GradualUnfreezing:
    """
    Gradually unfreeze model layers during training.
    
    Advanced concept: Progressive fine-tuning
    
    Starts with frozen backbone, gradually unfreezes layers.
    Useful for transfer learning.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        unfreeze_epochs: list
    ):
        """
        Args:
            model: Model to unfreeze
            unfreeze_epochs: List of epochs at which to unfreeze layers
        """
        self.model = model
        self.unfreeze_epochs = sorted(unfreeze_epochs)
        self.current_step = 0
    
    def step(self, epoch: int):
        """
        Check if should unfreeze layers at this epoch.
        
        Args:
            epoch: Current epoch
        """
        if (self.current_step < len(self.unfreeze_epochs) and 
            epoch >= self.unfreeze_epochs[self.current_step]):
            
            # Unfreeze next layer group
            if hasattr(self.model, 'unfreeze_layer'):
                self.model.unfreeze_layer(self.current_step)
                print(f"✓ Unfroze layer group {self.current_step} at epoch {epoch}")
            
            self.current_step += 1
