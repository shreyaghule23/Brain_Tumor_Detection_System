import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator



# 1. Load the Model
# Use the .h5 file we fixed earlier or the new .keras file
model_path = "models/vgg16_best.h5" 
model = keras.models.load_model(model_path)
print(f" Model loaded from {model_path}")

# 2. Re-create the Test Dataset (Required for bulk evaluation)
TEST_DIR = "dataset/Testing"
IMG_SIZE = (224, 224)
BATCH = 32

test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_directory(
    TEST_DIR, 
    target_size=IMG_SIZE, 
    batch_size=BATCH,
    class_mode="categorical", 
    shuffle=False  # Crucial: Keep False to match predictions with labels
)

# Get class names (e.g., ['glioma', 'meningioma', 'no_tumor', 'pituitary'])
CLASSES = list(test_ds.class_indices.keys())

# --- PART A: Bulk Evaluation ---
print("\n--- Running Bulk Evaluation ---")
y_pred_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_ds.classes

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Confusion Matrix Visualization
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix - Brain Tumor Detection")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# --- PART B: Single Image Prediction Function ---
def predict_single_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0 # Normalize same as training

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100

    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"Result: {CLASSES[class_idx]} ({confidence:.2f}% confidence)")

# Example Usage:
# predict_single_image("dataset/Testing/glioma/Te-gl_0010.jpg")
