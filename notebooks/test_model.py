import os
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input # type: ignore
from tensorflow.keras.utils import load_img, img_to_array

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the model
MODEL_PATH = os.path.join(BASE_DIR, "models", "foot_ulcer_model.h5")
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Path to validation data
VAL_DIR = os.path.join(BASE_DIR, "dataset", "split_dataset", "val")

# Automatically get the first image file it finds
sample_img_path = None
for root, dirs, files in os.walk(VAL_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_img_path = os.path.join(root, file)
            break
    if sample_img_path:
        break

if not sample_img_path:
    raise FileNotFoundError("No image files found in validation folder!")

print(f"Testing with sample image: {sample_img_path}")

# Load and preprocess the image
# img = image.load_img(sample_img_path, target_size=(128, 128))  # Match training size
# img_array = image.img_to_array(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)
img = load_img(sample_img_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
print("Raw prediction:", pred)

# Define class labels (update according to your dataset)
class_labels = ["Normal (Healthy skin)", "Abnormal (Ulcer)"]

# Handle binary vs multi-class
if pred.shape[1] == 1:
    prob = float(pred[0][0])
    label = class_labels[1] if prob > 0.5 else class_labels[0]
    confidence = prob if prob > 0.5 else 1 - prob
else:
    idx = np.argmax(pred, axis=1)[0]
    label = class_labels[idx]
    confidence = pred[0][idx]

print(f"Predicted class: {label} (Confidence: {confidence:.2f})")
