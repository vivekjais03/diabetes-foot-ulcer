import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "foot_ulcer_efficientnet_best.h5")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "split_dataset", "val")

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# Data generator for validation set
datagen = ImageDataGenerator(rescale=1./255)
val_generator = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",  # change to "categorical" if >2 classes
    shuffle=False
)

# Evaluate accuracy
loss, acc = model.evaluate(val_generator, verbose=1)
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# Predictions
y_pred = model.predict(val_generator)
y_pred_classes = (y_pred > 0.5).astype("int32")  # binary case

# True labels
y_true = val_generator.classes

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(val_generator.class_indices.keys())))

# Confusion matrix
print("🔎 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))
