#Had to import this because of problems with torch and numpy libraries. Got a little help from ChatGPT for debugging this
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
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

#Beginning prints
st.write("Garbage Classification")
st.write("Hello and welcome to my project! In it I'll be analyzing and training a model on a dataset about classification of garbage.")
st.write("This dataset is from Kaggle, https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download.")
st.write("In it there are many images of garbage and we'll train the model to be able to distribute which of them is: glass, metal, paper, plastic, or other trash.")
st.write("I've analyzed the dataset and I've come to the conslusion that the data is pretty well distributed between the different classes of garbage and therefore the dataset is fair to work with.")
st.write("In this project, you can enter the url of an image from the internet and the model will predict its class. The accuracy of the model is around 30% (checked in the other files that can be found in the GitHub repository).")
st.write("If you want to get a 'prediction' for your personal image, please upload it to a website like https://postimages.org/ and then paste the url to this application.")

#Save the model
MODEL_PATH = "garbage_cnn.pth"

#Create a dataset to make the model learn efficientlly
class GarbageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Used to generate the files "X_image.npy" and "y_labels.npy". Since the files have already been created and the data inside of them doesn't change, I've commented out the code in order to speed up the program and just the already computed data

# df = pd.read_csv("Garbage_Dataset_Classification/metadata.csv")
# folder = "Garbage_Dataset_Classification/images/"
# pixel_data = []
# labels = []
# #Convert the images to gray-scale pixels
# for label in df['label'].unique():
#     df_label = df[df['label'] == label]
#     n = len(df_label)
#     df_sample = df_label.sample(n = n, random_state = 42)

#     for _, row in df_sample.iterrows():
#         image_path = os.path.join(folder, row['label'], row['filename'])
#         if os.path.exists(image_path):
#             img = Image.open(image_path).convert('L').resize((128, 128))
#             pixel_data.append(np.array(img))
#             labels.append(row['label'])

# pixel_data = np.array(pixel_data)
# labels = np.array(labels)
# st.write(f"Loaded {len(pixel_data)} images")

# np.save("X_images.npy", pixel_data)
# np.save("y_labels.npy", labels)

#Load data from files
pixel_data = np.load("X_images.npy")
labels = np.load("y_labels.npy")

st.write("Loaded data from files X_images and y_labels")

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
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

#Create CNN class that will be the model
class CNN_model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3)
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
model = CNN_model(num_classes=len(categories)).to(device)

#Load model if it already exists, otherwise generate it
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location = device))
    model.eval()
    st.write("CNN model is loaded from file. Continue to program")

else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
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
	    #Prints to show the user that the program is processing, not just not loading or frozen
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    #Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    st.write(f"Model saved to {MODEL_PATH}")

#Prompt the user to enter an url
while True:
	url = ""
	with st.form(key = "url_form"):
		url = st.text_input("Enter image URL")
		st.form_submit_button("Submit")
	
	if url.lower() == "exit":
		break
	try:
		st.write("Current URL: ", url)
		response = requests.get(url)
		img = Image.open(BytesIO(response.content)).convert("L")
		img = img.resize((128, 128))
        #plt.imshow(img, cmap = 'gray')
        #plt.axis('off')
        #plt.show()

		img_tensor = torch.tensor(np.array(img), dtype = torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
		img_tensor = img_tensor.to(device)

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            pred_class = torch.argmax(output, dim = 1).item()

        st.write(f"Predicted class: {categories[pred_class]}")

    except Exception as e:
        st.write(f"Error loading image: {e}")
