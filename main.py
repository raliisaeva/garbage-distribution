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
# print(f"Loaded {len(pixel_data)} images")

# np.save("X_images.npy", pixel_data)
# np.save("y_labels.npy", labels)

pixel_data = np.load("X_images.npy")
labels = np.load("y_labels.npy")

#Output an example image from all the categories
input("Before we jump in the project, press enter to see how a random image from each garbage category looks like. I've pixelized the pictures and I've also made them gray-scale. To continue after that, just close the opened windows from the cross in the top right corner.")
categories = np.unique(labels)
for category in categories:
    indices = np.where(labels == category)[0]
    index = np.random.choice(indices)
    plt.imshow(pixel_data[index], cmap = 'gray')
    plt.title(f"Label: {category}")
    plt.axis('off')
    plt.show()


#Divide the dataset into training and testing data
X = pixel_data.reshape(len(pixel_data), -1) / 255.0
categories = sorted(np.unique(labels))
y = np.array([categories.index(l) for l in labels])

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Use one-hot encoding on the labels
num_classes = len(categories)
Y_train = np.eye(num_classes)[y_train]


weight_matrix = np.random.randn(X_train.shape[1], num_classes) * 0.01
bias_vector = np.zeros((1, num_classes))

#Calculate probability function
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

# logits = np.random.randn(5, num_classes)  # 5 random samples
# probabilities = softmax(logits)
# print("Softmax output:\n", probabilities)
# print("Sum of probabilities for each sample:", np.sum(probabilities, axis=1))
