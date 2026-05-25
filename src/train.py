import os
import matplotlib.pyplot as pltd 
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
 
# ── Config ────────────────────────────────────────────
TRAIN_DIR = "dataset/Training"
TEST_DIR  = "dataset/Testing"
IMG_SIZE  = (224, 224)
BATCH     = 32
SEED      = 42
 
os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)
 
# ── Data ──────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255, validation_split=0.2,
    rotation_range=20, zoom_range=0.15,
    horizontal_flip=True, fill_mode="nearest"
)
val_gen  = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_gen = ImageDataGenerator(rescale=1./255)
 
train_ds = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", subset="training", seed=SEED
)
val_ds = val_gen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", subset="validation", seed=SEED
)
test_ds = test_gen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="categorical", shuffle=False
)
 
# ── Model ─────────────────────────────────────────────
base_model = VGG16(include_top=False, weights="imagenet",
                   input_shape=(224, 224, 3))
base_model.trainable = False
 
inputs  = Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(4, activation="softmax")(x)
 
model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
 
# ── Callbacks ─────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/vgg16_best.h5", monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-7, verbose=1),
]
 
# ── Phase 1 — train head only ─────────────────────────
print("\n=== PHASE 1: Training head only ===")
history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=25, callbacks=callbacks
)
 
# ── Phase 2 — fine-tune blocks 4 & 5 ─────────────────
print("\n=== PHASE 2: Fine-tuning top layers ===")
base_model.trainable = True
for layer in base_model.layers:
    if layer.name.startswith(("block1", "block2", "block3")):
        layer.trainable = False
 
model.compile(optimizer=Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
 
history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=20, callbacks=callbacks
)
 
# ── Plot ──────────────────────────────────────────────
def plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    val   = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]
    ep    = range(1, len(acc) + 1)
    ph2   = len(h1.history["accuracy"])
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ep, acc,  label="Train acc")
    ax1.plot(ep, val,  label="Val acc", linestyle="--")
    ax1.axvline(ph2, color="red", linestyle=":", label="Phase 2 start")
    ax1.set_title("Accuracy"); ax1.legend()
 
    ax2.plot(ep, loss,  label="Train loss")
    ax2.plot(ep, vloss, label="Val loss", linestyle="--")
    ax2.axvline(ph2, color="red", linestyle=":", label="Phase 2 start")
    ax2.set_title("Loss"); ax2.legend()
 
    plt.tight_layout()
    plt.savefig("plots/training_history.png", dpi=120)
    plt.show()
 
plot_history(history1, history2)
 
# ── Evaluate ──────────────────────────────────────────
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest accuracy : {test_acc*100:.1f}%")
print(f"Test loss     : {test_loss:.4f}")
 
# ── Save all 3 model files ────────────────────────────
print("\n── Saving models ─────────────────────────────────")
 
# FILE 1 — vgg16_best.h5
model.save("models/vgg16_best.h5")
print("✓ FILE 1 saved: models/vgg16_best.h5")
 
# FILE 2 — vgg16_savedmodel/
model.save("models/vgg16_savedmodel", save_format="tf")
print("✓ FILE 2 saved: models/vgg16_savedmodel/")
 
# FILE 3 — vgg16_brain.tflite
converter = tf.lite.TFLiteConverter.from_saved_model("models/vgg16_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("models/vgg16_brain.tflite", "wb") as f:
    f.write(tflite_model)
print("✓ FILE 3 saved: models/vgg16_brain.tflite")
 
# ── Confirm ───────────────────────────────────────────
print("\n── models/ folder contents ───────────────────────")
for item in os.listdir("models"):
    print(f"   ✓  {item}")
print("\nTraining complete! All model files are ready.")