import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shutil

# Paths for train and validation sets
dataset_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\dataset"
train_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\train"
val_dir = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\split_dataset\validation"

# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Categories in the dataset
categories = ["diseased_acne", "diseased_dermatitis", "diseased_eczema", "healthy_skin", "diseased_melanoma", "diseased_psoriasis"]

# Create directories for train and validation
for split in ["train", "validation"]:
    for category in categories:
        os.makedirs(os.path.join(split_dir := train_dir if split == "train" else val_dir, category), exist_ok=True)

# Prepare data
data = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    images = os.listdir(category_path)

    for img_name in images:
        img_path = os.path.join(category_path, img_name)

        # Open, resize, and normalize the image
        img = Image.open(img_path)
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img = np.array(img) / 255.0  # Normalize pixel values to [0,1]

        data.append(img)
        labels.append(category)

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode the labels (categorical encoding)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into train and validation (80-20)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save the train and validation data in their respective directories
def save_data(data, labels, split_dir):
    for i, img in enumerate(data):
        img_category = label_encoder.inverse_transform([labels[i]])[0]
        img_path = os.path.join(split_dir, img_category, f"img_{i}.jpg")
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

save_data(X_train, y_train, train_dir)
save_data(X_val, y_val, val_dir)

print("Images are resized, normalized, and saved in train/validation directories!")
