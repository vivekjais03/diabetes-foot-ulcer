import os

dataset_path = "dataset/Patches"

# List the classes
classes = os.listdir(dataset_path)
print("Classes found:", classes)

# Check how many images in each
for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    print(f"{cls}: {len(os.listdir(cls_path))} images")
