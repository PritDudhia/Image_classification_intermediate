# Image Classification - Streamlit Deployment

## 🎯 Quick Deploy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/pritdudhia/image_classification_intermediate/main/app_streamlit.py)

**Live Demo**: [Your App URL Here]

## Features

- 🎯 10 class image classification (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
- 🔥 97% accuracy ResNet50 model
- ⚡ Real-time predictions
- 📊 Top-3 predictions with confidence scores
- 🎨 Clean, professional UI

## Quick Start

```bash
# Clone repository
git clone https://github.com/PritDudhia/Image_classification_intermediate.git
cd Image_classification_intermediate

# Install dependencies
pip install -r requirements_deploy.txt

# Run locally
streamlit run app_streamlit.py
```

## Deploy Your Own

1. Fork this repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your forked repo
5. Main file: `app_streamlit.py`
6. Click "Deploy"!

## Model

The model (best_model.pth) should be placed in the root directory. 

**For deployment**: Either upload the model file or use the download script.

## Tech Stack

- PyTorch / torchvision
- timm (PyTorch Image Models)
- Streamlit
- Pillow

## Screenshots

[Add your screenshot here]

## Training

Model trained on CIFAR-10 dataset:
- Architecture: ResNet50
- Epochs: 10
- Validation Accuracy: 97.35%
- Training: Google Colab (Free GPU)

## Project Structure

```
├── app_streamlit.py          # Main web app
├── app_gradio.py             # Alternative Gradio app
├── best_model.pth            # Trained model (download separately)
├── requirements_deploy.txt    # Deployment dependencies
├── scripts/                   # Training & evaluation scripts
├── src/                       # Source code (models, training, utils)
├── configs/                   # Hydra configuration files
├── docs/                      # Documentation
└── notebooks/                 # Colab training notebook
```

## Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Step-by-Step Guide](STEP_BY_STEP_GUIDE.md)
- [Practical Usage Guide](PRACTICAL_USAGE_GUIDE.md)
- [Concepts Reference](docs/CONCEPTS_REFERENCE.md)

## License

MIT

## Author

**Prit Dudhia**
- GitHub: [@PritDudhia](https://github.com/PritDudhia)
- LinkedIn: [Your Profile]

---

⭐ Star this repo if you found it helpful!
