"""
Fine-tune Your Downloaded Model on Custom Images
Use this to train on YOUR OWN images (flowers, products, etc.)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.simple_model import create_model

class CustomImageDataset(Dataset):
    """
    Organize your images like this:
    
    my_dataset/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all image paths
        self.images = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append({
                            'path': os.path.join(class_dir, img_name),
                            'label': self.class_to_idx[class_name]
                        })
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        image = Image.open(img_info['path']).convert('RGB')
        label = img_info['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def finetune_model(
    checkpoint_path='best_model.pth',
    data_dir='my_custom_dataset',
    num_classes=5,
    epochs=10,
    batch_size=32,
    lr=0.0001
):
    """
    Fine-tune on your custom dataset.
    
    Args:
        checkpoint_path: Path to downloaded model from Colab
        data_dir: Folder with your images organized by class
        num_classes: How many classes in YOUR dataset
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate (small for fine-tuning)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Training on: {device}\n")
    
    # Load pretrained model
    model = create_model('resnet50', num_classes=10, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded pretrained model (accuracy: {checkpoint['accuracy']:.2f}%)")
    
    # Replace final layer for YOUR classes
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, 'classifier'):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    model = model.to(device)
    print(f"✅ Adapted model for {num_classes} classes\n")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(data_dir, transform=transform)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"📊 Dataset: {len(dataset)} images")
    print(f"   Classes: {dataset.classes}")
    print(f"   Train: {train_size}, Val: {val_size}\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Train
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/len(pbar):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Acc: {100.*train_correct/train_total:.2f}% | Val Acc: {val_acc:.2f}%')
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'classes': dataset.classes
            }, 'custom_model.pth')
            print(f'✓ Saved best model (acc: {val_acc:.2f}%)')
    
    print(f'\n🎉 Fine-tuning complete! Best accuracy: {best_acc:.2f}%')
    print(f'💾 Model saved as: custom_model.pth')


if __name__ == "__main__":
    print("🎯 Fine-tune on Your Custom Dataset\n")
    print("=" * 60)
    
    print("""
    📁 Organize your images like this:
    
    my_custom_dataset/
        flowers/
            rose1.jpg
            rose2.jpg
        cars/
            car1.jpg
            car2.jpg
        buildings/
            building1.jpg
            building2.jpg
    
    Then run:
    python scripts/finetune_custom_dataset.py
    """)
    
    # Example: Fine-tune on your dataset
    # finetune_model(
    #     checkpoint_path='best_model.pth',
    #     data_dir='my_custom_dataset',
    #     num_classes=3,  # flowers, cars, buildings
    #     epochs=20,
    #     lr=0.0001
    # )
