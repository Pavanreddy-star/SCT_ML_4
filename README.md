# Hand Gesture Recognition Using CNN

This project implements a hand gesture recognition system using deep learning (CNN) and OpenCV.

## 🚀 Features
✅ Trains a deep learning model using LeapGestRecog dataset  
✅ Recognizes hand gestures from images and live webcam feed  
✅ Uses OpenCV and TensorFlow for real-time detection  

## 📂 Folder Structure
- `scripts/` → Preprocessing & training scripts  
- `models/` → Saved trained models  
- `dataset/` → LeapGestRecog dataset (not included in GitHub)  
- `results/` → Screenshots of accuracy & predictions  

## 🛠 Setup & Run
```bash
git clone https://github.com/YourUsername/Hand-Gesture-Recognition.git
cd Hand-Gesture-Recognition
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
python realtime.py
