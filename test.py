import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/hand_gesture_model.h5")

data = np.load("dataset/processed_data.npz", allow_pickle=True)
X_test, y_test = data["X_test"], data["y_test"]

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")
