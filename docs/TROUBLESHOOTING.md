# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: Torch installation fails

```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solution**:
```bash
# Install PyTorch first with CUDA support (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

Or for CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Training Issues

#### Issue: CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions**:

1. **Reduce batch size**:
```bash
python scripts/train.py data.batch_size=32
```

2. **Use gradient accumulation instead**:
```bash
python scripts/train.py data.batch_size=32 training.grad_accumulation_steps=2
```

3. **Disable mixed precision** (if causing issues):
```bash
python scripts/train.py training.use_amp=false
```

4. **Use smaller model**:
```bash
# Instead of ViT, use ResNet
python scripts/train.py model=resnet50
```

#### Issue: Training is very slow

**Solutions**:

1. **Check if using GPU**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

2. **Reduce data loading bottleneck**:
```yaml
# In config
data:
  num_workers: 0  # Try reducing if causing issues
  pin_memory: false
```

3. **Use smaller image size**:
```bash
python scripts/train.py data.image_size=128
```

#### Issue: Loss becomes NaN

```
Epoch 5: loss = nan
```

**Solutions**:

1. **Reduce learning rate**:
```bash
python scripts/train.py training.optimizer.lr=0.0001
```

2. **Enable gradient clipping**:
```bash
python scripts/train.py training.grad_clip=1.0
```

3. **Check for data issues**:
```python
# Verify data normalization
print(images.min(), images.max())
# Should be roughly [-2, 2] after normalization
```

4. **Disable mixed precision**:
```bash
python scripts/train.py training.use_amp=false
```

#### Issue: Model not improving

**Checklist**:

- [ ] Learning rate too high/low
- [ ] Forgot to unfreeze layers
- [ ] Data augmentation too aggressive
- [ ] Wrong optimizer
- [ ] Labels corrupted

**Solutions**:

1. **Start with baseline (no augmentation)**
2. **Verify data**:
```python
# Check few samples
for images, labels in train_loader:
    print(labels)
    break
```

3. **Try different LR**:
```bash
# Lower
python scripts/train.py training.optimizer.lr=0.00001

# Higher
python scripts/train.py training.optimizer.lr=0.001
```

---

### Data Issues

#### Issue: Dataset download fails

```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**Solution**:

For Windows:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

Or download manually and place in `./data/`

#### Issue: Custom dataset not loading

**Solutions**:

1. **Check folder structure**:
```
data/custom/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       ├── img1.jpg
│       └── img2.jpg
└── val/
    └── ...
```

2. **Verify image formats**:
```python
from PIL import Image
img = Image.open('path/to/image.jpg')
print(img.size, img.mode)
```

3. **Check config**:
```yaml
data:
  dataset: "CUSTOM"  # Must be uppercase
  data_dir: "./data/custom"
  num_classes: 2  # Update this!
```

---

### Configuration Issues

#### Issue: Hydra config override not working

```bash
# This doesn't work:
python scripts/train.py --model.dropout=0.5
```

**Solution**:
```bash
# Correct syntax (no dashes):
python scripts/train.py model.dropout=0.5
```

#### Issue: Can't find config file

```
ConfigFileNotFound: Could not find 'config'
```

**Solution**:

Make sure you're running from project root:
```bash
cd d:\project\image_classification
python scripts/train.py
```

---

### W&B Issues

#### Issue: W&B not logging

**Solutions**:

1. **Login first**:
```bash
wandb login
```

2. **Enable in config**:
```yaml
wandb:
  enabled: true
```

3. **Check internet connection**

#### Issue: W&B asking for login every time

**Solution**:
```bash
wandb login --relogin
# Enter your API key
```

---

### Model Export Issues

#### Issue: ONNX export fails

```
RuntimeError: Exporting the operator xxx to ONNX opset version 14 is not supported
```

**Solutions**:

1. **Try different opset**:
```python
# In export_onnx.py
opset_version=11  # Instead of 14
```

2. **Simplify model**:
Some custom operations may not be supported

#### Issue: ONNX model gives different results

**Solution**:

Check tolerance and verify:
```python
import onnxruntime
import numpy as np

# PyTorch
torch_out = model(input)

# ONNX
sess = onnxruntime.InferenceSession("model.onnx")
onnx_out = sess.run(None, {"input": input.numpy()})[0]

# Compare
np.testing.assert_allclose(
    torch_out.detach().numpy(),
    onnx_out,
    rtol=1e-03,
    atol=1e-05
)
```

---

### Performance Issues

#### Issue: Low GPU utilization

**Solutions**:

1. **Increase batch size**
2. **Increase num_workers**:
```yaml
data:
  num_workers: 8  # Adjust based on CPU cores
```

3. **Use pin_memory**:
```yaml
data:
  pin_memory: true
```

4. **Check for data bottleneck**:
```python
import time
start = time.time()
for batch in train_loader:
    pass
print(f"Data loading: {time.time() - start:.2f}s")
```

#### Issue: Model too large for deployment

**Solutions**:

1. **Use smaller variant**:
```bash
# Instead of B3, use B0
python scripts/train.py model=efficientnet_b0
```

2. **Knowledge distillation** (see Advanced Topics)

3. **Quantization**:
```python
import torch.quantization
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

### Debugging Tips

#### Enable debug mode

```bash
# Get more verbose output
python scripts/train.py --debug
```

#### Check gradient flow

```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.abs().mean())
```

#### Visualize data

```python
import matplotlib.pyplot as plt

for images, labels in train_loader:
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = images[i].permute(1, 2, 0)
        plt.imshow(img)
        plt.title(f"Label: {labels[i]}")
    plt.show()
    break
```

---

## Still Having Issues?

1. **Check GitHub Issues**: See if others faced same problem
2. **Enable verbose logging**: Add print statements
3. **Isolate the problem**: Test components individually
4. **Ask for help**: PyTorch forums, Stack Overflow

### Useful Commands

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Check package versions
pip list | grep torch

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `CUDA out of memory` | Batch too large | Reduce batch_size |
| `RuntimeError: Expected all tensors to be on the same device` | Data/model on different devices | Check `.to(device)` |
| `ValueError: num_samples should be a positive integer` | Empty dataset | Check data paths |
| `KeyError: 'model_state_dict'` | Wrong checkpoint format | Check saved checkpoint |
| `AttributeError: 'NoneType' object has no attribute 'shape'` | Data not loaded | Check dataset |
