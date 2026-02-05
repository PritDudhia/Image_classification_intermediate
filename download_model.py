"""
Script to download model for Streamlit Cloud deployment
Run this if model file is not in repository
"""

import os
import urllib.request
from pathlib import Path

MODEL_URL = "YOUR_MODEL_URL_HERE"  # Upload to Google Drive/Dropbox and add public link
MODEL_PATH = "best_model.pth"

def download_model():
    """Download model if not present."""
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists at {MODEL_PATH}")
        return True
    
    print(f"Downloading model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"✓ Model downloaded to {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    download_model()
