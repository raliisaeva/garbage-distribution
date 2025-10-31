import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load data and save the images to pixels
df = pd.read_csv("Garbage_Dataset_Classification/metadata.csv")

folder = "Garbage_Dataset_Classification/images/"
pixel_data = []
for i, row in df.iterrows():
    image_path = os.path.join(folder, row['label'], row['filename'])
    if os.path.exists(image_path):
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        pixel_data.append(img_array)
    else:
        print(f"Image not found: {image_path}")

#Make an observation about the distribution of data in the dataset.
print("Before we start anything, we'll first analyze our dataset.")
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
