"""
Model evaluation script.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_dataloaders
from src.models import create_model
from src.training import MetricTracker


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int
) -> dict:
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics_tracker = MetricTracker(num_classes)
    
    all_predictions = []
    all_targets = []
    
    print("\nEvaluating...")
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Track metrics
        metrics_tracker.update(outputs, targets)
        
        # Store for confusion matrix
        preds = outputs.argmax(1).cpu().numpy()
        targs = targets.cpu().numpy()
        all_predictions.extend(preds)
        all_targets.extend(targs)
    
    # Compute metrics
    metrics = metrics_tracker.compute()
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Classification report
    report = classification_report(all_targets, all_predictions)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'report': report,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    print("\n" + "="*60)
    print("📊 Model Evaluation")
    print("="*60 + "\n")
    
    # Device
    device = cfg.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Device: {device}\n")
    
    # Load checkpoint
    checkpoint_path = input("Enter checkpoint path (or 'best' for best model): ")
    
    if checkpoint_path == 'best':
        checkpoint_dir = Path(cfg.paths.checkpoint_dir)
        checkpoints = list(checkpoint_dir.glob("epoch_*.pth"))
        if not checkpoints:
            print("❌ No checkpoints found!")
            return
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_model(OmegaConf.to_container(cfg.model))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Model loaded\n")
    
    # Create dataloaders
    _, val_loader, test_loader = create_dataloaders(OmegaConf.to_container(cfg.data))
    
    # Evaluate on validation set
    print("="*60)
    print("Validation Set Results:")
    print("="*60)
    val_results = evaluate_model(model, val_loader, device, cfg.model.num_classes)
    
    print(f"\nAccuracy:  {val_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {val_results['metrics']['precision']:.4f}")
    print(f"Recall:    {val_results['metrics']['recall']:.4f}")
    print(f"F1 Score:  {val_results['metrics']['f1']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(val_results['report'])
    
    # Save confusion matrix
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = output_dir / "confusion_matrix_val.png"
    plot_confusion_matrix(val_results['confusion_matrix'], str(cm_path))
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Results:")
    print("="*60)
    test_results = evaluate_model(model, test_loader, device, cfg.model.num_classes)
    
    print(f"\nAccuracy:  {test_results['metrics']['accuracy']:.4f}")
    print(f"Precision: {test_results['metrics']['precision']:.4f}")
    print(f"Recall:    {test_results['metrics']['recall']:.4f}")
    print(f"F1 Score:  {test_results['metrics']['f1']:.4f}")
    
    cm_path = output_dir / "confusion_matrix_test.png"
    plot_confusion_matrix(test_results['confusion_matrix'], str(cm_path))
    
    print("\n✅ Evaluation complete!\n")


if __name__ == "__main__":
    main()
