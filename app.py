from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

# Inisialisasi Flask app
app = Flask(__name__)

# Load model
model_path = "model/final_model_vgg.h5"
model = load_model(model_path)

# Load class labels (ubah sesuai dataset-mu)
class_labels = ['Botol Kaca', 'Kaleng', 'Kardus', 'Kertas', 'Plastik' ]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]  # Ambil file dari request
    
    try:
        # Convert file to BytesIO untuk kompatibilitas dengan load_img()
        file_bytes = BytesIO(file.read())  # Baca file sebagai byte stream
        
        # Preprocess image
        img = load_img(file_bytes, target_size=(224, 224))  # Load dari byte stream
        img_array = img_to_array(img) / 255.0  # Normalisasi
        img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimensi
        
        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
