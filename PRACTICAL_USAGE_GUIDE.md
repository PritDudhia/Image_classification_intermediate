# 🎯 Practical Usage Guide: Using Your Trained Model

## What Your Model Can Do RIGHT NOW

Your downloaded `best_model.pth` from Colab can classify these **10 objects**:

| Class | Examples |
|-------|----------|
| ✈️ Airplane | Commercial planes, fighter jets |
| 🚗 Automobile | Cars, sedans, SUVs |
| 🐦 Bird | Any bird species |
| 🐱 Cat | Any cat breed |
| 🦌 Deer | Deer, elk |
| 🐕 Dog | Any dog breed |
| 🐸 Frog | Frogs, toads |
| 🐴 Horse | Horses, ponies |
| 🚢 Ship | Boats, cruise ships |
| 🚚 Truck | Pickup trucks, delivery trucks |

---

## Option 1: Use Model As-Is (CIFAR-10 Classes)

### Quick Test

```bash
# 1. Download best_model.pth from Colab to project root
# 2. Get test images (download from Google Images)
# 3. Run prediction

python scripts/use_trained_model.py
```

### Example Code

```python
from scripts.use_trained_model import load_model, predict_single_image

model = load_model('best_model.pth')
predict_single_image(model, 'my_cat.jpg')
# Output: 🎯 Prediction: CAT (Confidence: 95.3%)
```

### Practical Applications

**1. Animal Classifier App**
- Upload pet photos
- Identify: cat, dog, bird, horse, frog, deer
- Show confidence score

**2. Vehicle Detector**
- Classify: airplane, automobile, truck, ship
- Good for transportation categorization

**3. Image Organizer**
- Scan photo folder
- Auto-organize by class
- Works for these 10 categories

---

## Option 2: Fine-tune on YOUR Custom Images (RECOMMENDED!)

This is the **POWERFUL** option - train on YOUR data!

### Example: Flower Classifier

**Step 1: Collect Images** (50-100 per class minimum)
```
flower_dataset/
    roses/
        rose_001.jpg
        rose_002.jpg
        ... (100 images)
    tulips/
        tulip_001.jpg
        ... (100 images)
    sunflowers/
        sunflower_001.jpg
        ... (100 images)
```

**Step 2: Fine-tune**
```python
from scripts.finetune_custom_dataset import finetune_model

finetune_model(
    checkpoint_path='best_model.pth',  # Your Colab model
    data_dir='flower_dataset',
    num_classes=3,  # roses, tulips, sunflowers
    epochs=20
)
```

**Step 3: Use Your Custom Model**
```python
model = load_model('custom_model.pth')
predict_single_image(model, 'unknown_flower.jpg')
# Output: 🎯 Prediction: SUNFLOWER (Confidence: 92.1%)
```

### More Custom Dataset Ideas

| Project | Classes | Use Case |
|---------|---------|----------|
| **Product Classifier** | Shoes, Shirts, Bags | E-commerce categorization |
| **Food Recognizer** | Pizza, Burger, Sushi | Restaurant menu analysis |
| **Plant Disease Detector** | Healthy, Rust, Blight | Agriculture monitoring |
| **Document Classifier** | Invoice, Receipt, Contract | Office automation |
| **Skin Lesion Classifier** | Benign, Melanoma, Nevus | Medical screening |
| **Garbage Classifier** | Plastic, Paper, Metal | Recycling automation |

---

## Option 3: Build Real-World Application

### Web App (Streamlit)

```python
# app.py
import streamlit as st
from PIL import Image
from scripts.use_trained_model import load_model, preprocess_image
import torch

st.title("🐕 Pet Classifier")
st.write("Upload image of: cat, dog, bird, horse, frog, deer")

model = load_model('best_model.pth')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict
    input_tensor, _ = preprocess_image(uploaded_file)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    st.success(f"Prediction: {CLASSES[predicted]} ({confidence*100:.1f}%)")
```

**Run**: `streamlit run app.py`

### API Server (Flask)

```python
# api.py
from flask import Flask, request, jsonify
from scripts.use_trained_model import load_model, preprocess_image
import torch

app = Flask(__name__)
model = load_model('best_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save('temp.jpg')
    
    input_tensor, _ = preprocess_image('temp.jpg')
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return jsonify({
        'class': CLASSES[predicted.item()],
        'confidence': float(confidence.item())
    })

if __name__ == '__main__':
    app.run(debug=True)
```

**Test**: `curl -X POST -F "image=@cat.jpg" http://localhost:5000/predict`

---

## Decision Guide

### Use Model As-Is If:
- ✅ You need to classify the 10 CIFAR-10 classes
- ✅ Quick prototype/demo
- ✅ Learning how models work

### Fine-tune on Custom Data If:
- ✅ You need different classes (flowers, products, etc.)
- ✅ Building real product
- ✅ Need high accuracy on specific domain
- ✅ Have 50+ images per class

---

## Performance Expectations

| Scenario | Accuracy | Notes |
|----------|----------|-------|
| CIFAR-10 test set | ~94% | Original training data |
| Real photos (10 classes) | 70-85% | CIFAR is 32x32, real photos are better quality |
| Custom fine-tuned (100 imgs/class) | 85-95% | Depends on data quality |
| Custom fine-tuned (500 imgs/class) | 90-98% | Professional level |

---

## Next Steps Checklist

- [ ] Download `best_model.pth` from Colab
- [ ] Test on sample images (dog, cat, car photos)
- [ ] Decide: Use as-is OR fine-tune custom?
- [ ] If custom: Collect 100+ images per class
- [ ] Fine-tune on custom dataset
- [ ] Build web app or API
- [ ] Deploy (Hugging Face Spaces, Streamlit Cloud)
- [ ] Share your project!

---

## FAQs

**Q: Can I classify objects not in CIFAR-10?**
A: Yes! Fine-tune on your custom dataset (Option 2)

**Q: How many images do I need for fine-tuning?**
A: Minimum 50 per class, better with 100-500

**Q: Will my model work on real photos?**
A: Yes, but accuracy drops slightly (70-85% vs 94% on test set)

**Q: How to improve accuracy?**
A: 1) More training data, 2) Better quality images, 3) More epochs, 4) Data augmentation

**Q: Can I use multiple models together?**
A: Yes! Train ResNet, ViT, EfficientNet and average predictions (ensemble)

**Q: How fast is inference?**
A: ~50ms on GPU, ~200ms on CPU per image

---

Ready to build something? Start with Option 1 (test as-is) then move to Option 2 (custom data)! 🚀
