"""
Quick Test Script - Test your downloaded model on sample images
Run this FIRST after downloading model from Colab
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# CIFAR-10 Classes
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_model(checkpoint_path='best_model.pth'):
    """Load model from Colab."""
    try:
        import timm
        model = timm.create_model('resnet50', pretrained=False, num_classes=10)
    except:
        # Fallback to torchvision
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("=" * 60)
    print("✅ MODEL LOADED SUCCESSFULLY!")
    print("=" * 60)
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Validation Accuracy: {checkpoint['accuracy']:.2f}%")
    print("=" * 60 + "\n")
    
    return model

def preprocess_image(image_path):
    """Preprocess image for model."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def test_single_image(model, image_path):
    """Test on single image."""
    input_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = CLASSES[predicted.item()]
    confidence_pct = confidence.item() * 100
    
    # Determine if good prediction
    status = "✓" if confidence_pct > 50 else "⚠️"
    
    filename = os.path.basename(image_path)
    print(f"{filename:30s} → {predicted_class.upper():12s} ({confidence_pct:5.1f}%) {status}")
    
    return predicted_class, confidence_pct

def main():
    print("\n" + "=" * 60)
    print("🧪 QUICK MODEL TEST")
    print("=" * 60 + "\n")
    
    # Check if model exists
    if not os.path.exists('best_model.pth'):
        print("❌ ERROR: best_model.pth not found!")
        print("\n📋 Steps to fix:")
        print("   1. Download best_model.pth from Colab")
        print("   2. Save to: d:\\project\\image_classification\\best_model.pth")
        print("   3. Run this script again\n")
        print("In Colab, run this code:")
        print("   from google.colab import files")
        print("   files.download('best_model.pth')")
        print("\n" + "=" * 60 + "\n")
        return
    
    # Load model
    model = load_model('best_model.pth')
    
    # Check for test images
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"📁 Creating {test_dir}/ folder...\n")
        os.makedirs(test_dir, exist_ok=True)
        print("=" * 60)
        print("⚠️  NO TEST IMAGES FOUND")
        print("=" * 60)
        print("\n📋 Next steps:")
        print(f"   1. Download sample images (cat, dog, car, bird, airplane)")
        print(f"   2. Save to: {test_dir}/")
        print(f"   3. Run this script again")
        print("\n💡 Where to get images:")
        print("   - Google Images (search: 'cat', 'dog', etc.)")
        print("   - Unsplash.com (free stock photos)")
        print("   - Your own photos!")
        print("\n🎯 Model can classify:")
        print("   " + ", ".join(CLASSES))
        print("\n" + "=" * 60 + "\n")
        return
    
    # Find test images
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("=" * 60)
        print("⚠️  NO IMAGES IN test_images/ FOLDER")
        print("=" * 60)
        print("\n📋 Add some images:")
        print(f"   1. Download photos (cat, dog, car, etc.)")
        print(f"   2. Save to: {test_dir}/")
        print(f"   3. Run: python scripts/quick_test.py")
        print("\n" + "=" * 60 + "\n")
        return
    
    # Test all images
    print(f"📸 Testing {len(image_files)} images...\n")
    print("-" * 60)
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        pred_class, confidence = test_single_image(model, img_path)
        results.append({
            'file': img_file,
            'class': pred_class,
            'confidence': confidence
        })
    
    print("-" * 60)
    
    # Summary
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    high_conf = sum(1 for r in results if r['confidence'] > 70)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"   Images tested: {len(results)}")
    print(f"   Average confidence: {avg_confidence:.1f}%")
    print(f"   High confidence (>70%): {high_conf}/{len(results)}")
    print("=" * 60 + "\n")
    
    if avg_confidence > 70:
        print("🎉 EXCELLENT! Model working well!")
        print("\n✅ Next step: Build web app")
        print("   Run: streamlit run app_streamlit.py")
    elif avg_confidence > 50:
        print("✓ GOOD! Model working reasonably.")
        print("\n💡 Tips to improve:")
        print("   - Use higher quality images")
        print("   - Ensure images match CIFAR-10 classes")
        print("\n✅ Next step: Build web app anyway!")
        print("   Run: streamlit run app_streamlit.py")
    else:
        print("⚠️  LOW CONFIDENCE - Check your images")
        print("\n💡 Possible issues:")
        print("   - Images don't match CIFAR-10 classes")
        print("   - Images too complex/abstract")
        print("   - Try simpler, clearer images")
        print("\n🎯 Model works best with:")
        print("   " + ", ".join(CLASSES))
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
