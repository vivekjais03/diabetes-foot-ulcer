import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = "dataset/Patches"
OUTPUT_DIR = "dataset/split_dataset"

# Classes
classes = ["Abnormal(Ulcer)", "Normal(Healthy skin)"]

# Create output folders
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# Split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Process each class
for cls in classes:
    cls_dir = os.path.join(DATASET_DIR, cls)
    images = os.listdir(cls_dir)

    # Train / temp split
    train_imgs, temp_imgs = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
    # Validation / test split
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUTPUT_DIR, "train", cls))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUTPUT_DIR, "val", cls))
    for img in test_imgs:
        shutil.copy(os.path.join(cls_dir, img), os.path.join(OUTPUT_DIR, "test", cls))

print("Dataset preprocessing complete!")
