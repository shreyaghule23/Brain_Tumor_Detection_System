import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Configuration ──────────────────────────────────────
TRAIN_DIR = "dataset/Training"
TEST_DIR  = "dataset/Testing"
CLASSES   = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ── Count images per class ─────────────────────────────
print("\n Class distribution:")
for cls in CLASSES:
    train_n = len(os.listdir(f"{TRAIN_DIR}/{cls}"))
    test_n  = len(os.listdir(f"{TEST_DIR}/{cls}"))
    print(f"  {cls:15s}  train={train_n}  test={test_n}")

# ── Check image sizes ──────────────────────────────────
sizes = []
for cls in CLASSES:
    folder = f"{TRAIN_DIR}/{cls}"
    for fname in os.listdir(folder)[:5]:
        img = Image.open(f"{folder}/{fname}")
        sizes.append(img.size)
print(f"\nSample image sizes: {set(sizes)}")

# ── Visualise sample images ────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Brain MRI — sample images per class", fontsize=14)

for i, cls in enumerate(CLASSES):
    folder = f"{TRAIN_DIR}/{cls}"
    imgs   = os.listdir(folder)[:2]
    for j, fname in enumerate(imgs):
        img = Image.open(f"{folder}/{fname}").convert("RGB")
        axes[j][i].imshow(img, cmap="gray")
        axes[j][i].set_title(cls if j == 0 else "")
        axes[j][i].axis("off")

plt.tight_layout()
plt.savefig("plots/sample_images.png", dpi=120)
plt.show()
