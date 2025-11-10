# ----------------------------
# Fix OpenMP duplicate library error
# Must be at the very top, before any imports
# ----------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----------------------------
# Imports
# ----------------------------
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import requests
from io import BytesIO

MODEL_PATH = "garbage_cnn.pth"  # file to save/load model

# ----------------------------
# 1. Dataset class
# ----------------------------
class GarbageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ----------------------------
# 2. Load dataset
# ----------------------------
pixel_data = np.load("X_images.npy")
labels = np.load("y_labels.npy")

categories = sorted(np.unique(labels))
y = np.array([categories.index(l) for l in labels])

# Shuffle all data
indices = np.arange(len(pixel_data))
np.random.shuffle(indices)
pixel_data, y = pixel_data[indices], y[indices]

# Use all images for training
X_train = pixel_data
y_train = y

train_dataset = GarbageDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ----------------------------
# 3. Define CNN
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 124 * 124, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(categories)).to(device)

# ----------------------------
# 4. Check if model file exists
# ----------------------------
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Loaded model from file, skipping training.")
else:
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels_batch in train_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# ----------------------------
# 5. Loop for user input URLs
# ----------------------------
while True:
    url = input("\nEnter image URL (or type 'exit' to quit): ").strip()
    if url.lower() == "exit" or url == "":
        break

    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("L")
        img = img.resize((128, 128))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = torch.argmax(output, dim=1).item()

        print(f"Predicted class: {categories[pred_class]}")

    except Exception as e:
        print(f"Error loading image: {e}")
