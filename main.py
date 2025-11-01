import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#Show text before data
print("Hello and welcome to my project! In it I'll be analyzing and training a model on a dataset about classification of garbage.")
print("This dataset is from Kaggle, https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download")
print("In it there are many images of garbage and we'll train the model to be able to distribute which of them is: cardboard, glass, metal, paper, plastic, or other trash.")
print("Note that the dataset is pretty large so it may take some time for things to load.")

#Load dataset
df = pd.read_csv("Garbage_Dataset_Classification/metadata.csv")
folder = "Garbage_Dataset_Classification/images/"
pixel_data = []
labels = []


#Convert the images to gray-scale pixels
for _, row in df.iterrows():
    image_path = os.path.join(folder, row['label'], row['filename'])
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize((64, 64))
            pixel_data.append(np.array(img))
            labels.append(row['label'])
        except Exception as e:
            print(f"Failed to load {image_path}: {e}")

pixel_data = np.array(pixel_data)
labels = np.array(labels)
print(f"Loaded {len(pixel_data)} images.")


#Output an example image from all the categories
input("Before we jump in the project, press enter to see how an image from each garbage category looks like. I've pixelized the pictures and I've also made them gray-scale. To continue after that, just close the opened windows from the cross in the top right corner.")
categories = np.unique(labels)
for category in categories:
    index = np.where(labels == category)[0][0]
    plt.imshow(pixel_data[index], cmap = 'gray')
    plt.title(f"Label: {category}")
    plt.axis('off')
    plt.show()

#Training and testing data
X = pixel_data.reshape(len(pixel_data), -1) / 255.0
categories = sorted(np.unique(labels))
y = np.array([categories.index(l) for l in labels])

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

num_classes = len(categories)
Y_train = np.eye(num_classes)[y_train]
