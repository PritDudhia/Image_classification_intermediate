"""
How to Use Your Downloaded Model from Colab
This shows practical examples of using the trained model.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.simple_model import create_model

# CIFAR-10 Classes
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(checkpoint_path='best_model.pth'):
    """Load your trained model from Colab."""
    # Create model architecture
    model = create_model('resnet50', num_classes=10, pretrained=False)
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded!")
    print(f"   Trained to epoch: {checkpoint['epoch']}")
    print(f"   Validation accuracy: {checkpoint['accuracy']:.2f}%\n")
    
    return model

def preprocess_image(image_path):
    """Preprocess image for model."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def predict_single_image(model, image_path):
    """Predict class for a single image."""
    input_tensor, original_image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_pct = confidence.item() * 100
    
    print(f"📸 Image: {os.path.basename(image_path)}")
    print(f"🎯 Prediction: {predicted_class.upper()}")
    print(f"💯 Confidence: {confidence_pct:.1f}%")
    
    # Show top 3 predictions
    top3_prob, top3_idx = probabilities[0].topk(3)
    print(f"\nTop 3 predictions:")
    for i, (prob, idx) in enumerate(zip(top3_prob, top3_idx), 1):
        print(f"  {i}. {CLASSES[idx]:12s} - {prob*100:.1f}%")
    
    return predicted_class, confidence_pct

def predict_folder(model, folder_path):
    """Classify all images in a folder."""
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            input_tensor, _ = preprocess_image(image_path)
            
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = probabilities.max(1)
            
            results.append({
                'filename': filename,
                'class': CLASSES[predicted.item()],
                'confidence': confidence.item() * 100
            })
    
    # Print summary
    print(f"\n📊 Processed {len(results)} images\n")
    for r in results:
        print(f"{r['filename']:30s} → {r['class']:12s} ({r['confidence']:.1f}%)")
    
    return results


if __name__ == "__main__":
    print("🚀 Using Your Trained Model\n")
    print("=" * 60)
    
    # Example 1: Load model
    model = load_model('best_model.pth')
    
    # Example 2: Predict single image
    print("\n" + "=" * 60)
    print("Example: Classify single image")
    print("=" * 60 + "\n")
    
    # You need to provide an image path
    # predict_single_image(model, 'test_images/cat.jpg')
    
    # Example 3: Classify folder of images
    # predict_folder(model, 'test_images/')
    
    print("\n" + "=" * 60)
    print("💡 How to Use:")
    print("=" * 60)
    print("""
    1. Download best_model.pth from Colab
    2. Put test images in test_images/ folder
    3. Run: python scripts/use_trained_model.py
    
    Your model can classify these 10 classes:
    ✈️  airplane, 🚗 automobile, 🐦 bird, 🐱 cat, 🦌 deer
    🐕 dog, 🐸 frog, 🐴 horse, 🚢 ship, 🚚 truck
    """)
