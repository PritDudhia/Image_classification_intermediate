# 📚 Intermediate Concepts Reference

Quick reference guide for all intermediate/advanced concepts in this project.

## Table of Contents

1. [Transfer Learning](#transfer-learning)
2. [Data Augmentation](#data-augmentation)
3. [Model Architectures](#model-architectures)
4. [Training Techniques](#training-techniques)
5. [Optimization](#optimization)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Model Interpretability](#model-interpretability)
8. [Deployment](#deployment)

---

## Transfer Learning

### Concept
Using pretrained weights from ImageNet to boost performance on new tasks.

### Why It Works
- Pretrained models learned general visual features
- Early layers detect edges, textures
- Later layers detect shapes, patterns
- Only need to adapt final layers for new task

### Implementation

```python
# Frozen backbone (feature extraction)
model = ResNetClassifier(
    pretrained=True,
    freeze_backbone=True  # Freeze all except classifier
)

# Fine-tuning with frozen early layers
model = ResNetClassifier(
    pretrained=True,
    freeze_layers=2  # Freeze first 2 layer groups
)

# Full fine-tuning
model = ResNetClassifier(
    pretrained=True,
    freeze_backbone=False  # Train all layers
)
```

### When to Use Each
- **Feature Extraction**: Small dataset (<10k), similar to ImageNet
- **Partial Freezing**: Medium dataset (10k-100k)
- **Full Fine-tuning**: Large dataset (>100k), or very different from ImageNet

---

## Data Augmentation

### 1. CutMix

**Concept**: Cut a patch from one image and paste on another, mix labels proportionally.

```python
cutmix = CutMix(alpha=1.0, prob=0.5)
images, labels_a, labels_b, lam = cutmix(images, labels)
loss = lam * criterion(pred, labels_a) + (1-lam) * criterion(pred, labels_b)
```

**Benefits**:
- Improves localization
- Better generalization
- Reduces overfitting

**Parameters**:
- `alpha`: Controls mixing ratio (1.0 = standard)
- `prob`: How often to apply (0.5 = 50% of batches)

### 2. MixUp

**Concept**: Linear interpolation of images and labels.

```python
mixup = MixUp(alpha=0.2, prob=0.5)
images = lam * images + (1-lam) * images_shuffled
labels = lam * labels + (1-lam) * labels_shuffled
```

**Benefits**:
- Smoother decision boundaries
- Better calibration
- Regularization effect

### 3. RandAugment

**Concept**: Automatically finds good augmentation policies.

```yaml
randaugment:
  num_ops: 2    # How many operations
  magnitude: 9  # Strength (0-10)
```

**Operations**: Rotation, translation, color, contrast, sharpness, etc.

---

## Model Architectures

### ResNet (2015)

**Key Innovation**: Skip connections solve vanishing gradients

```python
# Residual block
def forward(x):
    identity = x
    out = conv1(x)
    out = conv2(out)
    out += identity  # Skip connection!
    return out
```

**Variants**: ResNet18, 34, 50, 101, 152

**Best For**: General purpose, learning

### Vision Transformer (2020)

**Key Innovation**: Self-attention for images, no convolutions

**Process**:
1. Split image into 16x16 patches
2. Linearly embed each patch
3. Add positional encodings
4. Pass through transformer encoder
5. Use [CLS] token for classification

**Best For**: Large datasets, state-of-the-art accuracy

### EfficientNet (2019)

**Key Innovation**: Compound scaling (depth + width + resolution)

**Scaling Formula**:
```
depth = α^φ
width = β^φ
resolution = γ^φ

constraint: α · β² · γ² ≈ 2
```

**Best For**: Efficiency, deployment, mobile

---

## Training Techniques

### 1. Mixed Precision Training

**Concept**: Use FP16 for faster training, FP32 for numerical stability.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Training loop
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2-3x speedup on modern GPUs
- 50% less memory
- Same accuracy

### 2. Gradient Accumulation

**Concept**: Accumulate gradients over multiple batches.

```python
for i, (images, labels) in enumerate(dataloader):
    loss = criterion(model(images), labels)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Use Case**: Simulate larger batch sizes with limited GPU memory.

**Effective Batch**: `batch_size × accumulation_steps`

### 3. Gradient Clipping

**Concept**: Clip gradients to prevent exploding gradients.

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

**When**: Training RNNs, unstable training

### 4. Label Smoothing

**Concept**: Soft targets instead of hard labels.

Instead of `[0, 1, 0]`, use `[ε/K, 1-ε+ε/K, ε/K]` where ε=0.1, K=num_classes.

```python
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
```

**Benefits**:
- Prevents overconfidence
- Better calibration
- Slight accuracy boost

---

## Optimization

### Optimizers

| Optimizer | When to Use | Learning Rate |
|-----------|-------------|---------------|
| **SGD** | From scratch training, best final accuracy | 0.01 - 0.1 |
| **Adam** | Quick convergence, adaptive LR | 0.0001 - 0.001 |
| **AdamW** | Adam + weight decay (better generalization) | 0.0001 - 0.001 |

### Learning Rate Schedules

#### Cosine Annealing
```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```
- Smooth decay following cosine curve
- No hyperparameters to tune
- Works well in most cases

#### OneCycleLR
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=epochs,
    steps_per_epoch=len(dataloader)
)
```
- Super-convergence: warm up then decay
- Faster training
- Often better results

#### Warmup
```python
# Linear warmup for first 5 epochs
for epoch in range(warmup_epochs):
    lr = base_lr * (epoch / warmup_epochs)
```
- Essential for Vision Transformers
- Helps with large batch sizes

---

## Evaluation Metrics

### Classification Metrics

```python
# Accuracy
accuracy = (predictions == targets).mean()

# Precision (positive predictive value)
precision = TP / (TP + FP)

# Recall (sensitivity)
recall = TP / (TP + FN)

# F1 Score (harmonic mean)
f1 = 2 * (precision * recall) / (precision + recall)
```

### Top-K Accuracy

```python
def top_k_accuracy(predictions, targets, k=5):
    _, top_k_preds = predictions.topk(k, dim=1)
    correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
    return correct.any(dim=1).float().mean()
```

**When**: Many classes (ImageNet has 1000)

### Calibration Error

Measures if predicted probabilities match actual accuracy.

```python
# Expected Calibration Error
ece = sum(|confidence - accuracy| * proportion)
```

**Lower is better**: 0 = perfectly calibrated

---

## Model Interpretability

### Grad-CAM

**Concept**: Visualize which regions model focuses on.

```python
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate_cam(image)
overlay = gradcam.visualize(image, original_image)
```

**How it Works**:
1. Forward pass
2. Backward to get gradients
3. Weight activation maps by gradients
4. Generate heatmap

### Attention Maps (ViT)

```python
attention = model.get_attention_maps(image, layer_idx=-1)
# Shape: [batch, heads, patches, patches]
```

Shows which image patches attend to which.

---

## Deployment

### ONNX Export

**Why**: Cross-platform deployment, optimization.

```python
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)
```

**Benefits**:
- Run on different frameworks
- Hardware optimization
- Reduced model size

### Quantization

**Concept**: Reduce precision (FP32 → INT8).

```python
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

**Benefits**:
- 4x smaller model
- 2-4x faster inference
- Minimal accuracy loss (~1%)

---

## Common Hyperparameter Ranges

| Hyperparameter | Typical Range | Start With |
|----------------|---------------|------------|
| Learning Rate | 1e-5 to 1e-1 | 1e-3 |
| Batch Size | 16 to 512 | 64 |
| Weight Decay | 0 to 1e-2 | 1e-4 |
| Dropout | 0 to 0.5 | 0.3 |
| Label Smoothing | 0 to 0.2 | 0.1 |
| Grad Clip | 0 to 5.0 | 1.0 |

---

## Best Practices Checklist

### Data
- [ ] Normalize inputs (ImageNet stats for transfer learning)
- [ ] Use data augmentation
- [ ] Verify class balance
- [ ] Check for data leakage

### Training
- [ ] Use mixed precision on GPU
- [ ] Implement early stopping
- [ ] Save best model checkpoint
- [ ] Track experiments (W&B)
- [ ] Monitor both train and val metrics

### Hyperparameters
- [ ] Try different learning rates (1e-5 to 1e-2)
- [ ] Use learning rate scheduler
- [ ] Start with pretrained weights
- [ ] Tune batch size for GPU

### Evaluation
- [ ] Use separate test set
- [ ] Compute multiple metrics (precision, recall, F1)
- [ ] Check confusion matrix
- [ ] Visualize predictions

### Deployment
- [ ] Export to ONNX
- [ ] Benchmark inference speed
- [ ] Test on real data
- [ ] Monitor production performance

---

## Further Reading

- **Papers**: See `docs/LEARNING_ROADMAP.md`
- **Tutorials**: Fast.ai, Stanford CS231n
- **Community**: PyTorch Forums, r/MachineLearning
