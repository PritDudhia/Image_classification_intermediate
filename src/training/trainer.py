"""
Main trainer class with modern training techniques.

Advanced concepts implemented:
- Mixed precision training (AMP)
- Gradient accumulation
- Gradient clipping
- CutMix/MixUp augmentations
- Learning rate scheduling with warmup
- Experiment tracking (W&B)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from typing import Optional, Dict
from pathlib import Path

from .metrics import MetricTracker, AverageMeter, top_k_accuracy
from .callbacks import EarlyStopping, ModelCheckpoint
from ..data.augmentations import CutMix, MixUp, mixup_criterion


class Trainer:
    """
    Advanced trainer with modern deep learning techniques.
    
    Key features:
    - Automatic Mixed Precision (AMP) for faster training
    - Gradient accumulation for larger effective batch sizes
    - CutMix/MixUp data augmentation
    - Comprehensive metric tracking
    - W&B integration for experiment tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.device = device
        
        # Training settings
        self.epochs = self.config.get('epochs', 100)
        self.use_amp = self.config.get('use_amp', True)
        self.grad_clip = self.config.get('grad_clip', 1.0)
        self.grad_accumulation = self.config.get('grad_accumulation_steps', 1)
        self.log_interval = self.config.get('log_interval', 10)
        
        # Advanced augmentations
        aug_config = self.config.get('augmentation', {}).get('train', {})
        
        # CutMix
        cutmix_cfg = aug_config.get('cutmix', {})
        self.cutmix = CutMix(
            alpha=cutmix_cfg.get('alpha', 1.0),
            prob=cutmix_cfg.get('prob', 0.5)
        ) if cutmix_cfg.get('enabled', False) else None
        
        # MixUp
        mixup_cfg = aug_config.get('mixup', {})
        self.mixup = MixUp(
            alpha=mixup_cfg.get('alpha', 0.2),
            prob=mixup_cfg.get('prob', 0.5)
        ) if mixup_cfg.get('enabled', False) else None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Callbacks
        early_stop_cfg = self.config.get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_cfg.get('patience', 15),
            min_delta=early_stop_cfg.get('min_delta', 0.001),
            mode='max'  # Monitor validation accuracy
        ) if early_stop_cfg.get('enabled', True) else None
        
        checkpoint_cfg = self.config.get('checkpoint', {})
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.config.get('checkpoint_dir', './checkpoints'),
            monitor=checkpoint_cfg.get('monitor', 'val_acc'),
            mode=checkpoint_cfg.get('mode', 'max'),
            save_top_k=checkpoint_cfg.get('save_top_k', 3),
            save_last=checkpoint_cfg.get('save_last', True)
        )
        
        # Metrics
        num_classes = self.config.get('num_classes', 10)
        self.train_metrics = MetricTracker(num_classes)
        self.val_metrics = MetricTracker(num_classes)
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        print(f"\n{'='*60}")
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Gradient Accumulation: {self.grad_accumulation}x")
        print(f"  CutMix: {'✓' if self.cutmix else '✗'}")
        print(f"  MixUp: {'✓' if self.mixup else '✗'}")
        print(f"{'='*60}\n")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            batch_size = images.size(0)
            
            # Apply CutMix or MixUp
            mixed = False
            if self.cutmix is not None:
                images, targets_a, targets_b, lam = self.cutmix(images, targets)
                mixed = True
            elif self.mixup is not None:
                images, targets_a, targets_b, lam = self.mixup(images, targets)
                mixed = True
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                
                # Compute loss
                if mixed:
                    loss = mixup_criterion(
                        self.criterion, outputs, targets_a, targets_b, lam
                    )
                else:
                    loss = self.criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accumulation
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accumulation == 0:
                # Gradient clipping
                if self.grad_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Metrics
            with torch.no_grad():
                if mixed:
                    # For mixed inputs, use original targets for accuracy
                    targets_for_acc = targets_a if lam > 0.5 else targets_b
                else:
                    targets_for_acc = targets
                
                acc = (outputs.argmax(1) == targets_for_acc).float().mean()
                
                loss_meter.update(loss.item() * self.grad_accumulation, batch_size)
                acc_meter.update(acc.item(), batch_size)
                
                # Update metrics tracker (use original targets)
                if not mixed:
                    self.train_metrics.update(outputs, targets, loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to W&B
            if wandb.run is not None and batch_idx % self.log_interval == 0:
                wandb.log({
                    'train/batch_loss': loss.item() * self.grad_accumulation,
                    'train/batch_acc': acc.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Compute epoch metrics
        metrics = {
            'train_loss': loss_meter.avg,
            'train_acc': acc_meter.avg
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        top5_acc_meter = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc='Validation')
        
        for images, targets in pbar:
            images, targets = images.to(self.device), targets.to(self.device)
            batch_size = images.size(0)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Metrics
            acc = (outputs.argmax(1) == targets).float().mean()
            
            # Top-5 accuracy (if applicable)
            if outputs.size(1) >= 5:
                top5_acc = top_k_accuracy(outputs, targets, k=5)
                top5_acc_meter.update(top5_acc, batch_size)
            
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc.item(), batch_size)
            
            # Update metrics tracker
            self.val_metrics.update(outputs, targets, loss.item())
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
        
        # Compute detailed metrics
        detailed_metrics = self.val_metrics.compute()
        
        metrics = {
            'val_loss': loss_meter.avg,
            'val_acc': acc_meter.avg,
            'val_precision': detailed_metrics['precision'],
            'val_recall': detailed_metrics['recall'],
            'val_f1': detailed_metrics['f1']
        }
        
        if outputs.size(1) >= 5:
            metrics['val_top5_acc'] = top5_acc_meter.avg
        
        return metrics
    
    def fit(self):
        """Main training loop."""
        print(f"\n🚀 Starting training for {self.epochs} epochs...\n")
        
        for epoch in range(1, self.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics['epoch'] = epoch
            all_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_acc'].append(train_metrics['train_acc'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_acc'].append(val_metrics['val_acc'])
            self.history['lr'].append(all_metrics['lr'])
            
            # Print epoch summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['val_acc']:.4f}")
            print(f"  LR:         {all_metrics['lr']:.6f}")
            print(f"{'='*60}\n")
            
            # Log to W&B
            if wandb.run is not None:
                wandb.log(all_metrics)
            
            # Checkpointing
            self.checkpoint_callback(self.model, epoch, all_metrics)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics['val_acc']):
                    print("Early stopping triggered!")
                    break
        
        print("\n✅ Training completed!")
        
        # Final summary
        best_epoch = self.history['val_acc'].index(max(self.history['val_acc'])) + 1
        print(f"\n{'='*60}")
        print("Best Results:")
        print(f"  Epoch: {best_epoch}")
        print(f"  Val Acc: {max(self.history['val_acc']):.4f}")
        print(f"  Val Loss: {self.history['val_loss'][best_epoch-1]:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
