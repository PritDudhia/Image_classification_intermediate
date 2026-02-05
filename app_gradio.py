"""
Gradio Web App - Alternative to Streamlit
Simple and clean interface
"""

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_model():
    """Load model."""
    try:
        import timm
        model = timm.create_model('resnet50', pretrained=False, num_classes=10)
    except:
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
    
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# Load model once
model = load_model()

def preprocess_image(image):
    """Preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image):
    """Predict image class."""
    if image is None:
        return None
    
    # Preprocess
    input_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Format results
    results = {CLASSES[i]: float(probabilities[i]) for i in range(10)}
    
    return results

# Examples
examples = [
    ["test_images/cat.jpg"],
    ["test_images/dog.jpg"],
    ["test_images/car.jpg"],
]

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🎯 Image Classifier",
    description="""
    **AI-powered image classification using ResNet50**
    
    Upload an image to classify! Works with:
    ✈️ Airplane, 🚗 Car, 🐦 Bird, 🐱 Cat, 🦌 Deer, 🐕 Dog, 🐸 Frog, 🐴 Horse, 🚢 Ship, 🚚 Truck
    
    Trained on CIFAR-10 with 94%+ accuracy using PyTorch.
    """,
    examples=examples if any(examples) else None,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates public link
