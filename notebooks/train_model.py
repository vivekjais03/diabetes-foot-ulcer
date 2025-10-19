import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
import os

# Paths
base_dir = "dataset/split_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Check if directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found: {train_dir}")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory not found: {val_dir}")

print(f"Training directory: {train_dir}")
print(f"Validation directory: {val_dir}")

# Image settings (EfficientNetB0 default is 224x224)
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation for training (use EfficientNet preprocess_input)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# EfficientNet preprocessing for validation
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Build transfer learning model with EfficientNetB0
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
base_model.trainable = False  # freeze for initial training

inputs = layers.Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs, outputs)

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
checkpoint_path = os.path.join("models", "foot_ulcer_efficientnet_best.h5")
os.makedirs("models", exist_ok=True)
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# Train the top classifier
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=cb
)

# Fine-tune: unfreeze top layers of base model
base_model.trainable = True
# Unfreeze last 20 layers for fine-tuning
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=cb
)

# Save trained model
final_model_path = "models/foot_ulcer_efficientnet.h5"
model.save(final_model_path)

print(f"Model training complete! Saved to {final_model_path}")
