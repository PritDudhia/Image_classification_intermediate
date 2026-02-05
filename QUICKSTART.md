# 🚀 Quick Start - 5 Minutes to Your First Model

## Prerequisites

- Python 3.8+
- (Optional) NVIDIA GPU with CUDA

## Step 1: Installation (2 minutes)

```bash
# Clone or download this repository
cd d:\project\image_classification

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Step 2: Train Your First Model (2 minutes to start)

```bash
python scripts/train.py
```

That's it! The script will:
- ✅ Download CIFAR-10 dataset automatically
- ✅ Train ResNet50 with transfer learning
- ✅ Save checkpoints to `./checkpoints`
- ✅ Show progress with a nice progress bar

**Expected output:**
```
🚀 Advanced Image Classification Training
============================================================

Configuration:
  model: resnet50
  dataset: CIFAR10
  batch_size: 64
  epochs: 100
...

✓ Dataloaders created:
  Train: 704 batches (45000 samples)
  Val:   79 batches (5000 samples)
  Test:  157 batches (10000 samples)

✓ Model: resnet50
  Total parameters: 25,557,032
  Trainable parameters: 25,557,032

🚀 Starting training for 100 epochs...

Epoch 1/100: 100%|██████████| 704/704 [01:23<00:00]
Val: 100%|██████████| 79/79 [00:05<00:00]

============================================================
Epoch 1 Summary:
  Train Loss: 1.2345
  Train Acc:  0.7234
  Val Loss:   0.9876
  Val Acc:    0.7856
  LR:         0.001000
============================================================
```

## Step 3: Monitor Training (1 minute)

Watch the terminal output to see:
- Training progress bar
- Loss and accuracy after each epoch
- Checkpoints being saved

**To stop training**: Press `Ctrl+C` (your best model is already saved!)

## What Just Happened?

You trained a deep learning model that:
- Uses ResNet50 architecture (25M parameters)
- Leverages ImageNet pretrained weights (transfer learning)
- Applies advanced data augmentations (CutMix, color jitter)
- Uses mixed precision training for speed
- Automatically saves the best model

**Expected results after 100 epochs:**
- Validation Accuracy: ~94-95%
- Training time: ~2-3 hours on GPU, 8-12 hours on CPU

## Next Steps

### Try Different Models

```bash
# Vision Transformer (latest trend, 2021-2026)
python scripts/train.py model=vit_base

# EfficientNet (best accuracy/size ratio)
python scripts/train.py model=efficientnet_b3
```

### Customize Training

```bash
# Smaller batch for less memory
python scripts/train.py data.batch_size=32

# Fewer epochs for quick experiment
python scripts/train.py training.epochs=10

# Different learning rate
python scripts/train.py training.optimizer.lr=0.0001
```

### Evaluate Your Model

```bash
python scripts/evaluate.py
# When prompted, enter checkpoint path or type 'best'
```

### Visualize Results

Check the saved files:
- `./checkpoints/` - Your trained models
- `./logs/` - TensorBoard logs (run: `tensorboard --logdir logs`)
- `./outputs/` - Confusion matrices and visualizations

## Common First-Run Issues

**"CUDA out of memory"**
```bash
python scripts/train.py data.batch_size=32
```

**"Too slow on CPU"**
```bash
# Use smaller model
python scripts/train.py model=resnet50 training.epochs=10
```

**"Want to see W&B tracking"**
```bash
wandb login  # Enter your API key
python scripts/train.py wandb.enabled=true
```

## Learning Path

Now that you have a working model, dive deeper:

1. **Week 1-2**: Read [Getting Started](docs/GETTING_STARTED.md)
2. **Week 3-4**: Follow [Learning Roadmap](docs/LEARNING_ROADMAP.md)
3. **Week 5+**: Explore [Advanced Topics](docs/ADVANCED_TOPICS.md)

## Architecture Comparison

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| ResNet50 | 94.5% | ⚡⚡⚡ | Learning, general use |
| ViT-Base | 95.2% | ⚡ | Large datasets, SOTA |
| EfficientNet-B3 | 95.8% | ⚡⚡ | Production, efficiency |

See [Model Comparison](docs/MODEL_COMPARISON.md) for details.

## Intermediate Concepts You'll Learn

✅ **Transfer Learning** - Use pretrained weights  
✅ **Data Augmentation** - CutMix, MixUp, RandAugment  
✅ **Mixed Precision** - Train 2x faster  
✅ **Learning Rate Scheduling** - Cosine, OneCycle  
✅ **Vision Transformers** - Latest architecture trend  
✅ **Model Interpretability** - GradCAM visualization  
✅ **Experiment Tracking** - W&B integration  
✅ **Model Export** - ONNX for deployment  

## Need Help?

- **Documentation**: Check `docs/` folder
- **Troubleshooting**: [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Examples**: See `notebooks/` (coming soon)

---

**Ready to become an ML expert? Let's go! 🎓**
