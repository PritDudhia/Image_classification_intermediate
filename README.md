# Advanced Image Classification Project

An intermediate-to-advanced level image classification project implementing modern deep learning techniques and MLOps best practices.

## 🎯 Learning Objectives

This project covers intermediate/advanced concepts:

### Deep Learning Concepts
- **Transfer Learning** - Fine-tuning pre-trained models (ResNet, EfficientNet, Vision Transformers)
- **Mixed Precision Training** - Using FP16 for faster training and reduced memory
- **Learning Rate Scheduling** - Cosine annealing, OneCycleLR, warmup strategies
- **Advanced Augmentations** - CutMix, MixUp, RandAugment, AutoAugment
- **Attention Mechanisms** - Vision Transformers (ViT) and self-attention
- **Model Ensembling** - Combining multiple models for better performance
- **Gradient Accumulation** - Training with larger effective batch sizes
- **Label Smoothing** - Regularization technique for better generalization

### MLOps & Best Practices
- **Experiment Tracking** - Weights & Biases integration
- **Model Checkpointing** - Saving best models with versioning
- **Configuration Management** - Hydra for managing experiments
- **Data Versioning** - DVC concepts for reproducibility
- **Model Registry** - Organizing and versioning trained models
- **TensorBoard** - Monitoring training metrics
- **ONNX Export** - Model deployment optimization
- **Distributed Training** - Multi-GPU training basics

### Latest AI Trends (2026)
- **Vision Transformers (ViT)** - Attention-based architectures
- **Knowledge Distillation** - Compressing large models
- **Self-Supervised Learning** - SimCLR, DINO approaches
- **Neural Architecture Search (NAS)** - EfficientNet family
- **Explainability** - GradCAM, attention visualization

## 📁 Project Structure

```
image_classification/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/                 # Model configs
│   ├── data/                  # Dataset configs
│   └── experiment/            # Experiment presets
├── src/
│   ├── data/
│   │   ├── dataset.py         # Custom dataset classes
│   │   ├── augmentations.py   # Advanced augmentation pipeline
│   │   └── dataloader.py      # DataLoader setup
│   ├── models/
│   │   ├── resnet.py          # ResNet variants
│   │   ├── efficientnet.py    # EfficientNet models
│   │   ├── vit.py             # Vision Transformer
│   │   └── custom_models.py   # Custom architectures
│   ├── training/
│   │   ├── trainer.py         # Training loop with mixed precision
│   │   ├── losses.py          # Custom loss functions
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── callbacks.py       # Training callbacks
│   ├── utils/
│   │   ├── config.py          # Config utilities
│   │   ├── visualization.py   # GradCAM, attention maps
│   │   └── checkpoint.py      # Model checkpointing
│   └── inference/
│       ├── predictor.py       # Inference pipeline
│       └── ensemble.py        # Model ensembling
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_augmentations.ipynb # Visualize augmentations
│   └── 03_model_analysis.ipynb # Model interpretation
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── export_onnx.py         # Model export
├── tests/                     # Unit tests
├── requirements.txt
└── setup.py
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training with ResNet50
python scripts/train.py model=resnet50

# Train Vision Transformer
python scripts/train.py model=vit_base experiment=vit_finetune

# Multi-GPU training
python scripts/train.py trainer.gpus=2 training.batch_size=64
```

### Experiment Tracking

The project uses Weights & Biases for experiment tracking:
- View training metrics in real-time
- Compare different model architectures
- Track hyperparameter performance
- Visualize model predictions and attention maps

## 🧪 Intermediate Concepts Demonstrated

### 1. Mixed Precision Training
Uses PyTorch's AMP (Automatic Mixed Precision) for faster training:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### 2. Advanced Data Augmentation
Implements modern augmentation strategies:
- **RandAugment**: Automatically searches for augmentation policies
- **CutMix**: Mixes patches from different images
- **MixUp**: Blends images and labels
- **AutoAugment**: Learned augmentation strategies

### 3. Learning Rate Schedules
- Cosine Annealing with Warm Restarts
- OneCycleLR for super-convergence
- Linear warmup strategies

### 4. Vision Transformers
Implements attention-based architectures that represent the latest trend in computer vision.

### 5. Model Interpretability
- GradCAM for visualizing what the model focuses on
- Attention map visualization for ViT
- Feature map analysis

## 📊 Datasets Supported

- CIFAR-10/100 (built-in PyTorch)
- ImageNet (requires download)
- Custom datasets (via config)
- Kaggle competitions datasets

## 🎓 Learning Path

1. **Start**: Train ResNet50 with transfer learning
2. **Intermediate**: Experiment with advanced augmentations
3. **Advanced**: Implement Vision Transformer from scratch
4. **Expert**: Add knowledge distillation and ensemble methods

## 📖 Key Libraries

- **PyTorch** - Deep learning framework
- **timm** - State-of-the-art computer vision models
- **albumentations** - Advanced image augmentations
- **wandb** - Experiment tracking
- **hydra** - Configuration management
- **torch.onnx** - Model export for deployment

## 🔥 Advanced Features to Explore

- [ ] Implement knowledge distillation
- [ ] Add self-supervised pre-training
- [ ] Implement neural architecture search
- [ ] Add test-time augmentation
- [ ] Implement gradual unfreezing
- [ ] Add adversarial training
- [ ] Implement semi-supervised learning

## 📝 License

MIT License
