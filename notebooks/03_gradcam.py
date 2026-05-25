import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image
import os

# Use the fixed loader
model = keras.models.load_model("models/vgg16_best.h5")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def get_gradcam_heatmap(model, img_array, base_layer_name="vgg16", last_conv="block5_conv3"):
    # 1. Get the nested VGG16 base
    base_model = model.get_layer(base_layer_name)
    
    # 2. Build the grad model using ONLY the base model to ensure connectivity
    # We map base_model input -> (last conv output AND base model final output)
    grad_model = tf.keras.Model(
        inputs=[base_model.input],
        outputs=[base_model.get_layer(last_conv).output, base_model.output]
    )

    # 3. Use the tape
    with tf.GradientTape() as tape:
        # Crucial: cast to tensor and ensure tape is watching
        img_tensor = tf.cast(img_array, tf.float32)
        tape.watch(img_tensor)
        
        # Get internal features and the base model's last output
        conv_outputs, base_acts = grad_model(img_tensor)
        
        # Run the REST of the main model (the Dense layers) on the base output
        # This completes the path from Conv layers -> Final Prediction
        prediction = model.layers[2](base_acts) # GlobalAveragePooling
        prediction = model.layers[3](prediction) # Dense
        prediction = model.layers[4](prediction) # Dropout
        prediction = model.layers[5](prediction) # Final Dense
        
        pred_index = tf.argmax(prediction[0])
        class_channel = prediction[:, pred_index]

    # 4. Compute gradients - This will no longer be None
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError("Gradients are None! The connection between layers is broken.")

    # 5. Global Average Pooling of Gradients
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # 6. Weighted sum of convolutional output
    cam = conv_outputs[0] @ weights[..., tf.newaxis]
    cam = tf.squeeze(cam)

    # 7. ReLU and Normalization
    cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-10)
    
    return cam.numpy(), CLASSES[pred_index], float(tf.reduce_max(prediction))



def visualise_gradcam(img_path):
    # Load and preprocess
    img_orig = Image.open(img_path).convert("RGB")
    img_resized = img_orig.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    inp = np.expand_dims(arr, axis=0).astype("float32")

    try:
        cam, pred_label, confidence = get_gradcam_heatmap(model, inp)

        # Process heatmap for overlay
        heatmap = cv2.resize(cam, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Merge with original image (resized)
        overlay = cv2.addWeighted(np.uint8(255 * arr), 0.6, heatmap, 0.4, 0)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_resized); axes[0].set_title("Original MRI")
        axes[1].imshow(heatmap);     axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay);     axes[2].set_title(f"Overlay\n{pred_label} ({confidence:.1%})")
        
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/gradcam_result.png", dpi=120)
        plt.show()
    except Exception as e:
        print(f"Error during Grad-CAM: {e}")

# Example usage
visualise_gradcam("dataset/Testing/glioma/Te-gl_10.jpg")
