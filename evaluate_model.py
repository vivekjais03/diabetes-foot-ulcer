import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input

def evaluate_model():
    # Load the trained model
    model_paths = [
        "models/foot_ulcer_efficientnet.h5",
        "models/foot_ulcer_efficientnet_best.h5",
        "models/foot_ulcer_model.h5"
    ]

    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = keras.models.load_model(path)
            print(f"Loaded model from: {path}")
            break

    if model is None:
        print("No trained model found!")
        return

    # Test data generator
    test_dir = "dataset/split_dataset/test"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return

    img_height, img_width = 224, 224  # EfficientNet default
    batch_size = 32

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False  # Important for evaluation
    )

    # Get predictions
    print("Evaluating model on test set...")
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Get true labels
    true_classes = test_generator.classes

    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    f1 = f1_score(true_classes, predicted_classes)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(".4f")
    print(".4f")
    print("\nDetailed Classification Report:")
    print("-"*50)
    print(classification_report(true_classes, predicted_classes,
                               target_names=['Normal (Healthy)', 'Abnormal (Ulcer)']))

    # Class distribution
    unique, counts = np.unique(true_classes, return_counts=True)
    print(f"\nTest set distribution:")
    print(f"Normal (Healthy): {counts[0]} samples")
    print(f"Abnormal (Ulcer): {counts[1]} samples")

    return accuracy, f1

if __name__ == "__main__":
    evaluate_model()
