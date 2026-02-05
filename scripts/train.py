"""
Main training script with Hydra configuration management.

Run examples:
    python scripts/train.py
    python scripts/train.py model=vit_base
    python scripts/train.py model=efficientnet_b3 data.batch_size=128
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.models import create_model
from src.training import Trainer, get_criterion


def setup_optimizer(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from config.
    
    Intermediate concept: Different optimizers and their use cases
    """
    opt_config = config.training.optimizer
    name = opt_config.name.lower()
    
    if name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_config.lr,
            betas=opt_config.get('betas', [0.9, 0.999]),
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    
    elif name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_config.lr,
            betas=opt_config.get('betas', [0.9, 0.999]),
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    
    elif name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_config.lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0.0),
            nesterov=opt_config.get('nesterov', True)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DictConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler from config.
    
    Advanced concept: Different LR scheduling strategies
    """
    sched_config = config.training.scheduler
    name = sched_config.name.lower()
    
    if name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_config.get('T_max', config.training.epochs),
            eta_min=sched_config.get('eta_min', 1e-6)
        )
    
    elif name == 'onecycle':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sched_config.get('max_lr', 0.01),
            epochs=config.training.epochs,
            steps_per_epoch=sched_config.get('steps_per_epoch', 100),
            pct_start=sched_config.get('pct_start', 0.3)
        )
    
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 30),
            gamma=sched_config.get('gamma', 0.1)
        )
    
    elif name == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_config.get('gamma', 0.1),
            patience=sched_config.get('patience', 10)
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration
    """
    print("\n" + "="*60)
    print("🚀 Advanced Image Classification Training")
    print("="*60 + "\n")
    
    # Print config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")
    
    # Set seed for reproducibility
    if cfg.experiment.get('seed'):
        torch.manual_seed(cfg.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.experiment.seed)
        print(f"✓ Set random seed: {cfg.experiment.seed}\n")
    
    # Device
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    print(f"✓ Using device: {device}\n")
    
    # Initialize W&B
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get('entity'),
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name,
            tags=cfg.wandb.get('tags', [])
        )
        print("✓ Weights & Biases initialized\n")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(OmegaConf.to_container(cfg.data))
    print()
    
    # Create model
    print("Creating model...")
    model = create_model(OmegaConf.to_container(cfg.model))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model: {cfg.model.architecture}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()
    
    # Create optimizer
    print("Setting up optimizer and scheduler...")
    optimizer = setup_optimizer(model, cfg)
    scheduler = setup_scheduler(optimizer, cfg)
    
    print(f"✓ Optimizer: {cfg.training.optimizer.name}")
    print(f"✓ Scheduler: {cfg.training.scheduler.name}")
    print()
    
    # Create loss criterion
    criterion = get_criterion(OmegaConf.to_container(cfg.training.loss))
    print(f"✓ Loss function: {cfg.training.loss.name}\n")
    
    # Prepare training config
    training_config = OmegaConf.to_container(cfg.training)
    training_config['num_classes'] = cfg.model.num_classes
    training_config['checkpoint_dir'] = cfg.paths.checkpoint_dir
    training_config['augmentation'] = cfg.data.augmentation
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
        device=device
    )
    
    # Train
    history = trainer.fit()
    
    # Save final model
    final_model_path = Path(cfg.paths.checkpoint_dir) / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': OmegaConf.to_container(cfg),
        'history': history
    }, final_model_path)
    print(f"\n✓ Saved final model to {final_model_path}")
    
    # Finish W&B
    if cfg.wandb.enabled:
        wandb.finish()
    
    print("\n✅ Training complete!\n")


if __name__ == "__main__":
    main()
