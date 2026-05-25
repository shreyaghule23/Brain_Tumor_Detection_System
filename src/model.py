from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam

NUM_CLASSES = 4
IMG_SHAPE   = (224, 224, 3)

# ── Load VGG16 base (no top FC layers) ────────────────
base_model = VGG16(
    include_top = False,          # remove original FC head
    weights     = "imagenet",    # pretrained weights
    input_shape = IMG_SHAPE
)
base_model.trainable = False    # freeze all base layers

# ── Custom classification head ────────────────────────
inputs  = Input(shape=IMG_SHAPE)
x = base_model(inputs, training=False)


# Replace Flatten with GlobalAveragePooling (less overfitting)
x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs, name="VGG16_BrainTumor")

# ── Compile ───────────────────────────────────────────
model.compile(
    optimizer = Adam(learning_rate=1e-3),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

model.summary()
print(f"\nTrainable params: {sum(w.numpy().size for w in model.trainable_weights):,}")
