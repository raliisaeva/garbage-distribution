'''


import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

print("Hello and welcome to my project! In it I'll be analyzing and training a model on a dataset about classification of garbage.")
print("This dataset is from Kaggle, https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification?resource=download")
print("In it there are many images of garbage and we'll train the model to be able to distribute which of them is: glass, metal, paper, plastic, or other trash.")
#Load data and save the images to pixels
df = pd.read_csv("Garbage_Dataset_Classification/metadata.csv")
folder = "Garbage_Dataset_Classification/images/"
pixel_data = []
labels = []

#Convert the images to gray-scale pixels
for i, row in df.iterrows():
    image_path = os.path.join(folder, row['label'], row['filename'])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize((128, 128))
        img_array = np.array(img)
        pixel_data.append(img_array)
        labels.append(row['label'])

pixel_data = np.array(pixel_data)
labels = np.array(labels)

#Output an example image from all the categories
input("Before we jump in the project, press enter to see how an image from each garbage category looks like. I've pixelized the pictures and I've also made them gray-scale. To continue after that, just close the opened windows from the cross in the top right corner.")
categories = np.unique(labels)
for category in categories:
    index = np.where(labels == category)[0][0]
    plt.imshow(np.squeeze(pixel_data[index]), cmap = 'gray')
    plt.title(f"Label: {category}")
    plt.axis('off')
    plt.show()

#Make an observation about the distribution of data in the dataset.
print("Before we start anything, we'll first analyze our dataset. Here's a glance of it:")
print(df.head())

print("In order to do that, we first have to decode all of our images so that we're able to look at them as pixels, and not as just a picture that the computer can't really analyze without any data.")
print("Now we'll have to check if the data is balanced or not.")
input("Press enter to see a histogram with information about the distribution of the data between different garbage categories. Later, to continue, close the histogram with the cross in the top right corner.")

category_counts = df['label'].value_counts()
category_counts.plot(kind = "bar", color = "skyblue", edgecolor = "black")
plt.title("Number of Images per Category")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation = 45)
plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
plt.show()

print("From that we can say that the data is pretty well distributed and therefore the dataset will be fair to work with.")
print("The dataset is from Kaggle and it was created this year. Its creater is said to be a student in a pretty prestigious and well-known school in Italy. In the same time, from the creator's account we can see that they are pretty active and decently good at machine learning, so we can say that the dataset is pretty authentic.")

labelEncoding = LabelEncoder()

X_train, X_tmp, y_train, y_tmp = train_test_split(pixel_data, labels, test_size = 0.3, stratify = labels, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size = 0.5, stratify = y_tmp, random_state = 42)

print(X_train.shape, X_val.shape, X_test.shape)



'''