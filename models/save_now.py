import os
# Force modern Keras behavior
os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

# 1. Load the model (Ensuring we use the modern keras loader)
print("Loading model...")
model = keras.models.load_model("models/vgg16_best.h5")
print("✓ Model loaded successfully")

# 2. FILE 2 — Save as modern .keras file
# Note: Keras 3 prefers .keras over the old folder-based 'tf' format
model.save("models/vgg16_updated.keras")
print("✓ FILE 2 saved: models/vgg16_updated.keras")

# 3. FILE 3 — Convert to TFLite
print("Converting to TFLite...")
# FIX: Use 'from_keras_model' instead of 'from_saved_model' 
# This avoids the "Folder not found" OSError.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("models/vgg16_brain.tflite", "wb") as f:
    f.write(tflite_model)
print("✓ FILE 3 saved: models/vgg16_brain.tflite")

# 4. Confirm folder contents
print("\n── models/ folder ──")
for item in os.listdir("models"):
    path = f"models/{item}"
    if os.path.isfile(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  ✓  {item}  ({size:.1f} MB)")
    else:
        print(f"  ✓  {item}  (Folder)")
