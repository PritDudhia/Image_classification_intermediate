# Advanced Topics Guide

## 1. Transfer Learning Strategies

### Feature Extraction

Freeze the entire backbone, only train classifier:

```yaml
# configs/model/resnet50_frozen.yaml
freeze_backbone: true
```

**When to use**: Small datasets (<10k images)

### Fine-tuning

Train all layers with lower learning rate:

```yaml
freeze_backbone: false
freeze_layers: 0
```

**When to use**: Medium-large datasets (>50k images)

### Gradual Unfreezing

Start frozen, unfreeze progressively:

```python
# In training script
from src.training.callbacks import GradualUnfreezing

unfreezer = GradualUnfreezing(
    model=model,
    unfreeze_epochs=[10, 20, 30]  # Unfreeze at these epochs
)
```

**When to use**: Domain shift between ImageNet and your data

## 2. Advanced Augmentations

### Understanding CutMix

Cuts a region from one image and pastes it on another:

```yaml
data:
  augmentation:
    train:
      cutmix:
        enabled: true
        alpha: 1.0      # Controls mix ratio distribution
        prob: 0.5       # Apply to 50% of batches
```

**Alpha values**:
- `alpha=1.0`: Uniform distribution (standard)
- `alpha=0.5`: More extreme mixes
- `alpha=2.0`: More conservative mixes

### Understanding MixUp

Linear interpolation between images:

```yaml
mixup:
  enabled: true
  alpha: 0.2    # Smaller = more conservative
  prob: 0.5
```

**Best practices**:
- Use CutMix OR MixUp, not both
- CutMix better for localization tasks
- MixUp better for classification

### RandAugment

Automatically searches for good augmentations:

```yaml
randaugment:
  enabled: true
  num_ops: 2       # Number of operations per image
  magnitude: 9     # Strength (0-10)
```

## 3. Learning Rate Strategies

### Cosine Annealing

Gradually decreases LR following a cosine curve:

```yaml
training:
  scheduler:
    name: "cosine"
    T_max: 100      # Total epochs
    eta_min: 1.0e-6 # Minimum LR
```

**Benefits**: Smooth convergence, no need to tune

### OneCycleLR

Super-convergence: peak LR then decay:

```yaml
training:
  scheduler:
    name: "onecycle"
    max_lr: 0.01
    pct_start: 0.3  # 30% warmup
```

**Benefits**: Faster training, often better results

### Warmup

Linear increase from low to target LR:

```yaml
training:
  scheduler:
    warmup_epochs: 5
    warmup_lr: 1.0e-5
```

**When to use**: Large batch sizes, Vision Transformers

## 4. Handling Class Imbalance

### Class Weights

```python
from src.data.dataloader import get_class_weights

# In training script
weights = get_class_weights(train_dataset, num_classes)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### Focal Loss

Focuses on hard examples:

```yaml
training:
  loss:
    name: "focal_loss"
    gamma: 2.0  # Higher = more focus on hard examples
```

### Balanced Sampling

```python
from src.data.dataset import BalancedBatchSampler

sampler = BalancedBatchSampler(
    labels=train_labels,
    batch_size=64
)
train_loader = DataLoader(dataset, batch_sampler=sampler)
```

## 5. Model Interpretability

### Grad-CAM Visualization

```python
from src.utils.visualization import GradCAM

# Get last conv layer (ResNet example)
target_layer = model.backbone.layer4[-1]

gradcam = GradCAM(model, target_layer)
cam = gradcam.generate_cam(input_image)
overlay = gradcam.visualize(input_image, original_image)
```

### Attention Maps (ViT)

```python
# For Vision Transformers
attention_maps = model.get_attention_maps(input_image, layer_idx=-1)
```

## 6. Experiment Tracking

### Weights & Biases Setup

```python
# Already integrated in train.py
# Just enable in config:
wandb:
  enabled: true
  project: "my-project"
  entity: "my-username"
```

**Features**:
- Real-time metric plots
- Hyperparameter comparison
- Model artifact versioning
- Attention map visualization

### Custom Logging

```python
import wandb

# Log custom metrics
wandb.log({
    "custom_metric": value,
    "learning_rate": lr
})

# Log images
wandb.log({"predictions": wandb.Image(image)})
```

## 7. Multi-GPU Training

### DataParallel (Simple)

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### DistributedDataParallel (Better)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
```

## 8. Gradient Accumulation

Simulate larger batch sizes:

```yaml
training:
  batch_size: 32
  grad_accumulation_steps: 4  # Effective batch = 32*4 = 128
```

**When to use**: GPU memory limitations

## 9. Knowledge Distillation

Train a smaller student model from a larger teacher:

```python
from src.training.losses import DistillationLoss

criterion = DistillationLoss(
    temperature=3.0,
    alpha=0.5  # Balance between hard and soft loss
)

# In training loop
student_logits = student_model(images)
with torch.no_grad():
    teacher_logits = teacher_model(images)

loss = criterion(student_logits, teacher_logits, targets)
```

## 10. Test-Time Augmentation (TTA)

Improves inference accuracy:

```python
def predict_with_tta(model, image, num_augmentations=5):
    predictions = []
    
    for _ in range(num_augmentations):
        # Apply random augmentations
        aug_image = augment(image)
        
        with torch.no_grad():
            pred = model(aug_image)
        predictions.append(pred)
    
    # Average predictions
    return torch.stack(predictions).mean(0)
```

## 11. Model Ensembling

Combine multiple models:

```python
def ensemble_predict(models, image):
    predictions = []
    
    for model in models:
        with torch.no_grad():
            pred = model(image)
        predictions.append(F.softmax(pred, dim=1))
    
    # Average probabilities
    return torch.stack(predictions).mean(0)
```

## 12. Custom Dataset Integration

```python
# Option 1: Using ImageFolder structure
# data/custom/
#   ├── train/
#   │   ├── class1/
#   │   └── class2/
#   └── val/
#       ├── class1/
#       └── class2/

# In config:
data:
  dataset: "CUSTOM"
  data_dir: "./data/custom"
  num_classes: 2  # Update this

# Option 2: Custom Dataset class
from src.data.dataset import CustomImageDataset

dataset = CustomImageDataset(
    root="./data/custom/train",
    transform=transform
)
```

## Further Reading

- **Vision Transformers**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **EfficientNet**: [Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- **CutMix**: [Regularization Strategy](https://arxiv.org/abs/1905.04899)
- **Label Smoothing**: [Rethinking Inception](https://arxiv.org/abs/1512.00567)
- **Mixed Precision**: [NVIDIA AMP](https://pytorch.org/docs/stable/amp.html)
