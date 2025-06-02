from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model/document_classifier_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = [
    "letter", "form", "email", "handwritten", "advertisement",
    "scientific_report", "scientific_publication", "specification",
    "file_folder", "news_article", "budget", "invoice", "presentation",
    "questionnaire", "resume", "memo"
]

IMG_SIZE = 224

def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", label=None)

        image = request.files["image"]
        if image.filename == "":
            return render_template("index.html", label=None)

        image_path = os.path.join("uploads", image.filename)
        os.makedirs("uploads", exist_ok=True)
        image.save(image_path)

        # Preprocess and predict
        input_data = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        predicted_index = np.argmax(output_data)
        confidence = float(output_data[0][predicted_index]) * 100

        label = CLASS_NAMES[predicted_index]
        return render_template("index.html", label=label, confidence=f"{confidence:.2f}")

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)