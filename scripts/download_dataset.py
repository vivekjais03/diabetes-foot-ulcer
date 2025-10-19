import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download("laithjj/diabetic-foot-ulcer-dfu")
print("Dataset downloaded to:", path)

# Point to Patches folder (classification ready)
patches_path = os.path.join(path, "DFU", "Patches")
print("Patches folder path:", patches_path)
