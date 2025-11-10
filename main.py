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
# input("Before we jump in the project, press enter to see how a random image from each garbage category looks like. I've pixelized the pictures and I've also made them gray-scale. To continue after that, just close the opened windows from the cross in the top right corner.")
# categories = np.unique(labels)
# for category in categories:
#     indices = np.where(labels == category)[0]
#     index = np.random.choice(indices)
#     plt.imshow(pixel_data[index], cmap = 'gray')
#     plt.title(f"Label: {category}")
#     plt.axis('off')
#     plt.show()


#Divide the dataset into training and testing data
X = pixel_data.reshape(len(pixel_data), -1) / 255.0
categories = sorted(np.unique(labels))
y = np.array([categories.index(l) for l in labels])

indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

split = int(0.01 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

#Use one-hot encoding on the labels
num_classes = len(categories)
Y_train = np.eye(num_classes)[y_train]

W = np.random.randn(X_train.shape[1], num_classes) * 0.01
b = np.zeros((1, num_classes))

# #Calculate probability function
# def softmax(z):
#     exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
#     return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

# #Linear classifier
# lr = 0.1
# epochs = 100

# for epoch in range(epochs):
#     # Forward
#     z = X_train.dot(W) + b
#     preds = softmax(z)
#     loss = -np.mean(np.sum(Y_train * np.log(preds + 1e-8), axis=1))
    
#     # Backward
#     grad_z = preds - Y_train
#     dW = X_train.T.dot(grad_z) / len(X_train)
#     db = np.mean(grad_z, axis=0, keepdims=True)
    
#     # Update weights
#     W -= lr * dW
#     b -= lr * db
    
#     if (epoch + 1) % 10 == 0:
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# z_test = X_test.dot(W) + b
# preds_test = np.argmax(softmax(z_test), axis=1)
# accuracy = np.mean(preds_test == y_test)
# print(f"Test accuracy: {accuracy*100:.2f}%")

# --- CNN hyperparameters ---
filter_size = 3
num_filters1 = 8
num_filters2 = 16

# --- Initialize CNN weights and biases ---
# Layer 1: single-channel input
W1 = np.random.randn(num_filters1, 1, filter_size, filter_size) * 0.01
b1 = np.zeros((num_filters1, 1))

# Layer 2: multi-channel input from conv1
W2 = np.random.randn(num_filters2, num_filters1, filter_size, filter_size) * 0.01
b2 = np.zeros((num_filters2, 1))

# Fully connected layer (after flattening conv2 output)
conv_output_size = 124  # 128 -> conv1(3x3)->126 -> conv2(3x3)->124
flatten_size = num_filters2 * conv_output_size * conv_output_size
W_fc = np.random.randn(flatten_size, num_classes) * 0.01
b_fc = np.zeros((1, num_classes))

# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# --- Multi-channel convolution function ---
def conv2d_multi(X, W, b):
    """
    X: (n_samples, in_channels, h, w)
    W: (out_channels, in_channels, f_h, f_w)
    b: (out_channels, 1)
    Returns: (n_samples, out_channels, out_h, out_w)
    """
    n_samples, in_channels, h, w = X.shape
    out_channels, _, f_h, f_w = W.shape
    out_h = h - f_h + 1
    out_w = w - f_w + 1
    out = np.zeros((n_samples, out_channels, out_h, out_w))

    for i in range(n_samples):
        for j in range(out_channels):
            for k in range(in_channels):
                for m in range(out_h):
                    for n in range(out_w):
                        region = X[i, k, m:m+f_h, n:n+f_w]
                        out[i, j, m, n] += np.sum(region * W[j, k])
            out[i, j] += b[j]
    return out

# --- Flatten function ---
def flatten(X):
    return X.reshape(X.shape[0], -1)

# --- CNN forward pass ---
def cnn_forward(X):
    # Reshape input to (n_samples, channels, height, width)
    X_reshaped = X.reshape(-1, 1, 128, 128)  # single-channel grayscale

    # Conv layer 1
    conv1 = relu(conv2d_multi(X_reshaped, W1, b1))

    # Conv layer 2
    conv2 = relu(conv2d_multi(conv1, W2, b2))

    # Flatten
    flat = flatten(conv2)

    # Fully connected output
    out = softmax(flat.dot(W_fc) + b_fc)
    return out

def cnn_forward_batched(X, batch_size=32):
    """
    Forward pass in batches to speed up computation.
    X: input data (num_samples, 128*128)
    batch_size: number of images to process at once
    Returns: softmax probabilities for all samples
    """
    n_samples = X.shape[0]
    all_preds = []

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        X_reshaped = X_batch.reshape(-1, 1, 128, 128)  # single-channel

        # Conv layer 1
        conv1 = relu(conv2d_multi(X_reshaped, W1, b1))

        # Conv layer 2
        conv2 = relu(conv2d_multi(conv1, W2, b2))

        # Flatten
        flat = flatten(conv2)

        # Fully connected + softmax
        out = softmax(flat.dot(W_fc) + b_fc)

        all_preds.append(out)

    return np.vstack(all_preds)

# --- Test forward pass ---
Y_pred = cnn_forward(X_train[:5])
print("CNN forward output shape:", Y_pred.shape)

# --- Training set ---
Y_train_pred_probs = cnn_forward_batched(X_train, batch_size=32)
y_train_pred = np.argmax(Y_train_pred_probs, axis=1)
train_accuracy = np.mean(y_train_pred == y_train)
print(f"Training accuracy: {train_accuracy * 100:.2f}%")

# --- Test set ---
Y_test_pred_probs = cnn_forward_batched(X_test, batch_size=32)
y_test_pred = np.argmax(Y_test_pred_probs, axis=1)
test_accuracy = np.mean(y_test_pred == y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


