import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/hand_gesture_model.h5")
data = np.load("dataset/processed_data.npz", allow_pickle=True)
label_map = data["label_map"].item()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)

    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    gesture = label_map[class_idx]

    cv2.putText(frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
