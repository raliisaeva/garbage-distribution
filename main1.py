import os
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

# Convert labels to indices
categories = sorted(np.unique(labels))
y = np.array([categories.index(l) for l in labels])

# Shuffle data
indices = np.arange(len(pixel_data))
np.random.shuffle(indices)
pixel_data, y = pixel_data[indices], y[indices]

# Split train/test
split = int(0.8 * len(pixel_data))
X_train, X_test = pixel_data[:split], pixel_data[split:]
y_train, y_test = y[:split], y[split:]

# ----------------------------
# 3. Create datasets and loaders (manual conversion)
# ----------------------------
train_dataset = GarbageDataset(X_train, y_train)
test_dataset = GarbageDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 4. Define 2-layer CNN
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)       # 128x128 -> 126x126
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)      # 126x126 -> 124x124
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
# 5. Loss and optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 6. Training loop
# ----------------------------
epochs = 5  # start small for testing

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

# ----------------------------
# 7. Evaluate accuracy
# ----------------------------
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels_batch in loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    return correct / total

train_acc = evaluate(train_loader)
test_acc = evaluate(test_loader)
print(f"Training Accuracy: {train_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")






# ~~~~~~~~~~~~ NEW CODE ~~~~~~~~~~~~~~~~~~~~~~

# --- Load image directly from URL ---
url = "https://sfmcd.org/wp-content/uploads/2021/07/Cardboard-hero.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("L")  # grayscale
img = img.resize((128, 128))

# Display the image
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

# --- Convert to tensor and normalize ---
img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
img_tensor = img_tensor.to(device)

# --- Predict with the model ---
model.eval()
with torch.no_grad():
    output = model(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()

print(f"Predicted class: {categories[pred_class]}")