# Getting Started Guide

## Installation

### 1. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Setup Weights & Biases

For experiment tracking:

```bash
wandb login
```

## Quick Start

### Training Your First Model

Train a ResNet50 on CIFAR-10:

```bash
python scripts/train.py
```

This will:
- Download CIFAR-10 automatically
- Train ResNet50 with transfer learning
- Save checkpoints to `./checkpoints`
- Log metrics to W&B (if enabled)

### Training Different Models

```bash
# Vision Transformer
python scripts/train.py model=vit_base

# EfficientNet-B3
python scripts/train.py model=efficientnet_b3
```

### Customize Training

```bash
# Larger batch size
python scripts/train.py data.batch_size=128

# More epochs
python scripts/train.py training.epochs=200

# Different learning rate
python scripts/train.py training.optimizer.lr=0.0001

# Disable mixed precision (if GPU issues)
python scripts/train.py training.use_amp=false
```

### Multiple Configurations

```bash
# Change model AND batch size
python scripts/train.py model=vit_base data.batch_size=64 training.epochs=50
```

## Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py
# Enter checkpoint path when prompted, or type 'best' for latest
```

## Export for Deployment

Export to ONNX format:

```bash
python scripts/export_onnx.py
# Enter checkpoint path when prompted
```

## Understanding the Configuration System

This project uses Hydra for configuration management. All configs are in `configs/`:

```
configs/
├── config.yaml          # Main config
├── model/
│   ├── resnet50.yaml
│   ├── vit_base.yaml
│   └── efficientnet_b3.yaml
├── data/
│   └── cifar10.yaml
└── training/
    └── default.yaml
```

### Override Examples

```bash
# Override any config value
python scripts/train.py model.dropout=0.5

# Disable early stopping
python scripts/train.py training.early_stopping.enabled=false

# Change augmentation settings
python scripts/train.py data.augmentation.train.cutmix.enabled=false
```

## Common Issues

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/train.py data.batch_size=32

# Use gradient accumulation instead
python scripts/train.py data.batch_size=32 training.grad_accumulation_steps=2
```

### Slow Training on CPU

Vision Transformers are slow on CPU. Use ResNet instead:

```bash
python scripts/train.py model=resnet50 device=cpu
```

## Next Steps

1. **Explore Notebooks**: Check `notebooks/` for tutorials
2. **Experiment with Augmentations**: Try different combinations
3. **Try Custom Datasets**: Add your own data
4. **Compare Models**: Train multiple architectures and compare in W&B
5. **Advanced Techniques**: Implement knowledge distillation, ensemble methods

## Key Intermediate Concepts to Learn

1. **Transfer Learning**
   - File: `src/models/resnet.py`
   - Try: Freeze different numbers of layers

2. **Mixed Precision Training**
   - File: `src/training/trainer.py`
   - Benefit: 2-3x speedup on modern GPUs

3. **Advanced Augmentations**
   - File: `src/data/augmentations.py`
   - Try: Adjust CutMix/MixUp probabilities

4. **Learning Rate Scheduling**
   - File: `scripts/train.py`
   - Try: Different schedulers (cosine, onecycle, step)

5. **Label Smoothing**
   - File: `src/training/losses.py`
   - Benefit: Better calibration, slight accuracy boost

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [timm Documentation](https://huggingface.co/docs/timm/)
- [Albumentations](https://albumentations.ai/)
- [Weights & Biases](https://wandb.ai/)
