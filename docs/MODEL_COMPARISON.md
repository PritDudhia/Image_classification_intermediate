# Model Architecture Comparison

## Overview

This guide compares the three main architectures implemented in this project: ResNet, Vision Transformer (ViT), and EfficientNet.

## Architecture Comparison Table

| Architecture | Parameters | Speed | Accuracy | Best For | Complexity |
|-------------|------------|-------|----------|----------|------------|
| ResNet50 | 25M | Fast | Good | General purpose | Medium |
| ViT-Base | 86M | Slow | Excellent | Large datasets | High |
| EfficientNet-B3 | 12M | Medium | Excellent | Resource-constrained | Medium |

## ResNet (Residual Networks)

### Key Concepts

- **Residual Connections**: Skip connections that help gradients flow
- **Bottleneck Blocks**: Reduces parameters while maintaining capacity
- **Batch Normalization**: Stabilizes training

### Architecture

```
Input (224x224x3)
    ↓
Conv1 (7x7, stride=2)
    ↓
MaxPool (3x3, stride=2)
    ↓
Layer1 (3 blocks)
    ↓
Layer2 (4 blocks, stride=2)
    ↓
Layer3 (6 blocks, stride=2)
    ↓
Layer4 (3 blocks, stride=2)
    ↓
Global Average Pool
    ↓
Fully Connected (num_classes)
```

### When to Use

✅ **Use ResNet when**:
- You have small to medium datasets
- You need fast inference
- You want proven, stable architecture
- You're starting with image classification

❌ **Don't use when**:
- You have huge datasets (>1M images) → Use ViT
- You need the smallest possible model → Use EfficientNet

### Code Example

```python
from src.models import create_resnet

model = create_resnet({
    'architecture': 'resnet50',
    'num_classes': 10,
    'pretrained': True,
    'freeze_layers': 2  # Freeze first 2 layer groups
})
```

## Vision Transformer (ViT)

### Key Concepts

- **Patch Embeddings**: Images split into patches
- **Self-Attention**: Learns relationships between all patches
- **Positional Encodings**: Adds spatial information
- **No Convolutions**: Pure transformer architecture

### Architecture

```
Input (224x224x3)
    ↓
Patch Embedding (16x16 patches)
    → 196 patches of 768-dim
    ↓
Add [CLS] token + Positional Encoding
    ↓
Transformer Encoder (12 layers)
    Each layer:
    - Multi-Head Attention (12 heads)
    - Layer Norm
    - MLP (3072 hidden units)
    - Layer Norm
    ↓
Extract [CLS] token
    ↓
MLP Head (num_classes)
```

### When to Use

✅ **Use ViT when**:
- You have large datasets (>100k images)
- You can use ImageNet pretraining
- You want state-of-the-art accuracy
- You have good GPU resources

❌ **Don't use when**:
- Small datasets without pretraining
- Limited compute (slow on CPU)
- Need fast inference

### Code Example

```python
from src.models import create_vit

model = create_vit({
    'architecture': 'vit_base_patch16_224',
    'num_classes': 10,
    'pretrained': True,
    'dropout': 0.1,
    'attention_dropout': 0.1
})
```

### Unique Features

**Attention Maps**: Visualize what the model focuses on

```python
attention_maps = model.get_attention_maps(image, layer_idx=-1)
# Returns: [batch, num_heads, num_patches, num_patches]
```

## EfficientNet

### Key Concepts

- **Compound Scaling**: Balances depth, width, resolution
- **MBConv Blocks**: Mobile inverted bottleneck convolutions
- **Squeeze-and-Excitation**: Channel attention mechanism
- **Neural Architecture Search**: Discovered via AutoML

### Architecture

```
Input (resolution varies by variant)
    ↓
Stem Conv (3x3)
    ↓
MBConv Blocks (7 stages)
    Each block:
    - Expansion (1x1 conv)
    - Depthwise Conv (3x3 or 5x5)
    - Squeeze-Excitation
    - Projection (1x1 conv)
    - Residual connection
    ↓
Head Conv (1x1)
    ↓
Global Average Pool
    ↓
Dropout
    ↓
Fully Connected (num_classes)
```

### Scaling Strategy

| Model | Resolution | Depth Coef | Width Coef | Params |
|-------|-----------|------------|------------|--------|
| B0 | 224 | 1.0 | 1.0 | 5.3M |
| B1 | 240 | 1.1 | 1.0 | 7.8M |
| B2 | 260 | 1.2 | 1.1 | 9.2M |
| B3 | 300 | 1.4 | 1.2 | 12M |
| B4 | 380 | 1.8 | 1.4 | 19M |

### When to Use

✅ **Use EfficientNet when**:
- You want the best accuracy/parameter ratio
- You need efficient inference
- You're deploying to mobile/edge devices
- You want to balance accuracy and speed

❌ **Don't use when**:
- You need the absolute simplest architecture
- You're learning fundamentals (ResNet is clearer)

### Code Example

```python
from src.models import create_efficientnet

model = create_efficientnet({
    'architecture': 'efficientnet_b3',
    'num_classes': 10,
    'pretrained': True,
    'dropout': 0.3
})
```

## Performance Comparison

### CIFAR-10 Benchmark (100 epochs)

| Model | Top-1 Acc | Train Time | Params | GPU Memory |
|-------|-----------|------------|--------|------------|
| ResNet50 | 94.5% | 2h | 25M | 4GB |
| ViT-Base | 95.2% | 5h | 86M | 8GB |
| EfficientNet-B3 | 95.8% | 3h | 12M | 6GB |

*Note: Results vary based on hyperparameters and augmentations*

### ImageNet-1k (Transfer Learning)

| Model | Top-1 Acc | Top-5 Acc | FLOPs |
|-------|-----------|-----------|-------|
| ResNet50 | 76.1% | 92.9% | 4.1G |
| ViT-Base/16 | 84.5% | 97.2% | 17.6G |
| EfficientNet-B3 | 81.6% | 95.7% | 1.8G |

## Choosing the Right Model

### Decision Tree

```
Are you learning?
├─ Yes → Start with ResNet50
└─ No → Continue

Have >100k images?
├─ Yes → ViT-Base
└─ No → Continue

Need fast inference?
├─ Yes → EfficientNet-B0 or B1
└─ No → Continue

Have good GPU?
├─ Yes → ViT-Base or EfficientNet-B4
└─ No → ResNet50 or EfficientNet-B2

Deploying to mobile?
└─ EfficientNet-Lite or B0
```

### Recommendations by Use Case

**Academic Research / Learning**
- Start: ResNet50
- Advanced: ViT-Base

**Production / Real-world Applications**
- Balanced: EfficientNet-B3
- High accuracy priority: ViT-Base
- Speed priority: EfficientNet-B1

**Edge Deployment**
- EfficientNet-B0 or EfficientNet-Lite

**Kaggle Competitions**
- Single model: ViT-Large
- Ensemble: Mix of all three

## Implementation Details

### Transfer Learning Performance

```python
# Frozen backbone (fastest, decent accuracy)
freeze_backbone: true
# Training time: 30 min
# Expected: 92-93% on CIFAR-10

# Freeze early layers (good balance)
freeze_layers: 2
# Training time: 1.5h
# Expected: 94-95% on CIFAR-10

# Full fine-tuning (best accuracy)
freeze_layers: 0
# Training time: 2-3h
# Expected: 95-96% on CIFAR-10
```

### Recommended Batch Sizes

| Model | GPU 8GB | GPU 16GB | GPU 24GB |
|-------|---------|----------|----------|
| ResNet50 | 64 | 128 | 256 |
| ViT-Base | 32 | 64 | 128 |
| EfficientNet-B3 | 48 | 96 | 192 |

## Papers and Resources

**ResNet**
- Paper: [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- Year: 2015

**Vision Transformer**
- Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Year: 2020

**EfficientNet**
- Paper: [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)
- Year: 2019
