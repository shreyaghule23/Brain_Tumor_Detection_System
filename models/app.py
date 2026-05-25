from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io, os
from tensorflow.keras.models import load_model

app     = Flask(__name__)
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Load model once at startup ─────────────────────────
MODEL_PATH = os.path.join("models", "vgg16_best.h5")
model = load_model(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

def preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)   # shape (1,224,224,3)

# ── Health check ──────────────────────────────────────
@app.route("/")
def health():
    return jsonify({"status": "ok", "model": "VGG16 brain tumor"})

# ── Prediction endpoint ───────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file_bytes = request.files["file"].read()
    inp        = preprocess(file_bytes)
    preds      = model.predict(inp, verbose=0)[0]
    idx        = int(np.argmax(preds))

    return jsonify({
        "prediction"   : CLASSES[idx],
        "confidence"   : round(float(preds[idx]) * 100, 2),
        "probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASSES, preds)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
