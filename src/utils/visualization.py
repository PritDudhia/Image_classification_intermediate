"""
Visualization utilities for model interpretation.

Advanced concepts:
- GradCAM for visualizing what the model focuses on
- Attention map visualization for Vision Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Advanced concept: Model interpretability
    
    Visualizes which regions of an image the model focuses on for predictions.
    
    Paper: "Grad-CAM: Visual Explanations from Deep Networks" (Selvaraju et al., 2017)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The model to visualize
            target_layer: The layer to visualize (typically last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input image tensor [1, C, H, W]
            target_class: Target class index (None = predicted class)
        
        Returns:
            Heatmap as numpy array [H, W]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        input_image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay visualization of Grad-CAM.
        
        Args:
            input_image: Preprocessed input tensor
            original_image: Original image as numpy array [H, W, 3]
            target_class: Target class
            alpha: Overlay transparency
        
        Returns:
            Overlay image as numpy array
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to original image size
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (alpha * heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay


def visualize_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    num_images: int = 16,
    device: str = 'cuda'
) -> None:
    """
    Visualize model predictions on a batch of images.
    
    Args:
        model: Model to use
        images: Batch of images
        labels: True labels
        class_names: List of class names
        num_images: Number of images to show
        device: Device
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(images.to(device))
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(1).cpu()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(num_images, len(images))):
        ax = axes[i]
        
        # Denormalize image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Title with prediction
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i, preds[i]].item()
        
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(
            f"True: {true_label}\nPred: {pred_label} ({confidence:.2f})",
            color=color,
            fontsize=8
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✓ Saved plot to {save_path}")
    else:
        plt.show()
