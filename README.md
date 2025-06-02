# Document Classification using Deep Learning and Web Deployment

This project uses a Convolutional Neural Network (CNN) model trained on the RVL-CDIP dataset to classify scanned document images into 16 categories. It includes:

- Model training with TensorFlow
- Model conversion to `.tflite` for lightweight inference
- A Flask-based web application to upload and classify documents in real-time

## Tech Stack

- TensorFlow / Keras
- OpenCV
- Flask
- HTML/CSS/JS
- TFLite

## Instructions

1. Train the model: `python train_model.py`
2. Run the web app: `python app.py`
3. Visit: `http://localhost:5000` in your browser

