# Sign-Language-Detection
🚀 Project Overview

A real-time Sign Language Recognition web application that converts hand gestures into text using Computer Vision and Deep Learning.

🧠 Technologies Used

Python

Flask

TensorFlow / Keras

MediaPipe

OpenCV

Docker

⚙️ Features

Real-time hand gesture detection

CNN-based classification model

Web-based interface

Dockerized deployment

Automatic model training if model not found

📊 Model Details

Custom CNN architecture

Image augmentation

Multi-class classification (A–Z)

🐳 Docker Support
docker build -t sign-language-app .
docker run -p 5000:5000 sign-language-app
📁 Dataset


## 📁 Dataset

Dataset is not included due to size limitations.

To use this project:
1. Create a folder named `data`
2. Add subfolders for each class (A–Z)
3. Place training images inside each folder

🌐 How to Run

Install dependencies

Run python app.py
OR
Use Docker (recommended)
