import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/leapGestRecog"
IMG_SIZE = 64  # Resize to 64x64

def load_data():
    images, labels = [], []
    label_map = {}

    # Check if dataset path exists
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path {DATASET_PATH} not found.")
        return np.array([]), np.array([]), {}

    for label, folder in enumerate(sorted(os.listdir(DATASET_PATH))):
        folder_path = os.path.join(DATASET_PATH, folder)
        
        # Check if it's a valid folder
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder}")  # Debugging print
        label_map[label] = folder

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Check if it's an image file
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            images.append(img)
            labels.append(label)

    # Check if images were loaded
    if len(images) == 0 or len(labels) == 0:
        print("Error: No images found in dataset!")
        return np.array([]), np.array([]), {}

    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = to_categorical(labels, num_classes=len(label_map))

    return images, labels, label_map

X, y, label_map = load_data()

# Stop execution if data is empty
if X.size == 0 or y.size == 0:
    print("Error: No data available for training. Check dataset!")
    exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.savez("dataset/processed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, label_map=label_map)

print("✅ Data preprocessing completed successfully!")


