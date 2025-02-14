import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/hand_gesture_model.h5")

data = np.load("dataset/processed_data.npz", allow_pickle=True)
label_map = data["label_map"].item()

def predict_gesture(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    gesture = label_map[class_idx]

    return gesture

img_path = "test_image.jpg"  # Replace with actual image path
print(f"Predicted Gesture: {predict_gesture(img_path)}")
