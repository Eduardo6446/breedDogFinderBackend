import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# Cargar variables de entorno
load_dotenv()

API_KEY = os.getenv("API_KEY")
UMBRAL_FIABILIDAD = 15

# Cargar etiquetas
with open("class_labels.json", "r") as f:
    labels = json.load(f)

# Cargar modelo
model = tf.keras.models.load_model("model/model.keras")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Verifica la API key
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    if not data or 'base64' not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decodificar la imagen desde base64
        image_data = base64.b64decode(data["base64"])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Hacer predicción
        prediction = model.predict(image_array)[0]
        accuracy = float(np.max(prediction)) * 100
        predicted_class = int(np.argmax(prediction))
        breed = labels[str(predicted_class)]

        response = {
            "breed_dog": breed,
            "accuracy": round(accuracy, 2),
            "unreliable": accuracy < UMBRAL_FIABILIDAD
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
