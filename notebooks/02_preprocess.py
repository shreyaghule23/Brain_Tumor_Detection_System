from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH    = 32
SEED     = 42
TRAIN_DIR = "dataset/Training"
TEST_DIR  = "dataset/Testing"

# ── Training generator (with augmentation) ────────────
train_gen = ImageDataGenerator(
    rescale          = 1./255,
    validation_split = 0.2,
    rotation_range   = 20,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    shear_range      = 0.1,
    zoom_range       = 0.15,
    horizontal_flip  = True,
    brightness_range = [0.8, 1.2],
    fill_mode        = "nearest"
)

# ── Validation generator (NO augmentation) ────────────
val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ── Test generator ─────────────────────────────────────
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

print("Train batches:", len(train_ds))   # ~72
print("Val batches  :", len(val_ds))     # ~18
print("Test batches :", len(test_ds))    # ~13
print("Class map    :", train_ds.class_indices)