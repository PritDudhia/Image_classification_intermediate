"""
Streamlit Web App - Image Classifier
Deploy this to Streamlit Cloud for free!
"""

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.dirname(__file__))

# CIFAR-10 Classes with emojis
CLASSES = {
    0: ('airplane', '✈️'),
    1: ('automobile', '🚗'),
    2: ('bird', '🐦'),
    3: ('cat', '🐱'),
    4: ('deer', '🦌'),
    5: ('dog', '🐕'),
    6: ('frog', '🐸'),
    7: ('horse', '🐴'),
    8: ('ship', '🚢'),
    9: ('truck', '🚚')
}

@st.cache_resource
def load_model():
    """Load model (cached)."""
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
    
    return model, checkpoint['accuracy']

def preprocess_image(image):
    """Preprocess uploaded image."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(model, image):
    """Make prediction."""
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Get top 3
    top3_prob, top3_idx = probabilities.topk(3)
    
    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        class_name, emoji = CLASSES[idx.item()]
        results.append({
            'class': class_name,
            'emoji': emoji,
            'confidence': prob.item() * 100
        })
    
    return results

# Page config
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🎯",
    layout="centered"
)

# Title
st.title("🎯 AI Image Classifier")
st.markdown("**Powered by ResNet50 + PyTorch**")
st.markdown("---")

# Load model
with st.spinner("Loading model..."):
    model, accuracy = load_model()

st.success(f"✅ Model loaded (Validation accuracy: {accuracy:.2f}%)")

# Info
with st.expander("ℹ️ About this model"):
    st.write(f"""
    This classifier can identify **10 different objects**:
    
    {' • '.join([f"{emoji} {name}" for name, emoji in CLASSES.values()])}
    
    **Model details:**
    - Architecture: ResNet50
    - Trained on: CIFAR-10 dataset
    - Validation accuracy: {accuracy:.2f}%
    - Training: Google Colab (Free GPU)
    """)

st.markdown("---")

# Upload
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'gif', 'tiff', 'tif', 'jfif'],
    help="Upload a clear image of: cat, dog, car, bird, etc."
)

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict
    with st.spinner("Analyzing..."):
        results = predict(model, image)
    
    # Show results
    with col2:
        st.subheader("🎯 Prediction")
        
        top_result = results[0]
        
        # Main prediction
        st.markdown(f"## {top_result['emoji']} **{top_result['class'].upper()}**")
        st.markdown(f"### Confidence: **{top_result['confidence']:.1f}%**")
        
        # Progress bar
        st.progress(top_result['confidence'] / 100)
        
        # Interpretation
        if top_result['confidence'] > 80:
            st.success("High confidence! ✓")
        elif top_result['confidence'] > 50:
            st.info("Moderate confidence")
        else:
            st.warning("Low confidence - image might not match trained classes")
    
    # Top 3 predictions
    st.markdown("---")
    st.subheader("📊 Top 3 Predictions")
    
    for i, result in enumerate(results, 1):
        col_emoji, col_name, col_bar = st.columns([1, 3, 6])
        
        with col_emoji:
            st.markdown(f"### {result['emoji']}")
        
        with col_name:
            st.markdown(f"**{result['class'].title()}**")
        
        with col_bar:
            st.progress(result['confidence'] / 100)
            st.caption(f"{result['confidence']:.1f}%")

else:
    # Example/instructions
    st.info("👆 Upload an image to get started!")
    
    st.markdown("### 💡 Tips for best results:")
    st.markdown("""
    - Use clear, well-lit images
    - Object should be main focus
    - Avoid multiple objects in one image
    - Works best with simple backgrounds
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with PyTorch, Streamlit | 
    <a href='https://github.com/YOUR_USERNAME/image_classification'>GitHub</a> | 
    <a href='https://linkedin.com/in/YOUR_PROFILE'>LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
