import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set the path to the training data
train_dir = 'train'  # Change this to the actual path to your 'train' folder

# Image size for model input
img_height = 224
img_width = 224

# Number of classes (diseases + healthy)
num_classes = 10  # There are 10 categories in total (9 diseases + 1 healthy)

# Create ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the training data using the flow_from_directory method
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical',  # Use categorical crossentropy loss since it's a multi-class problem
    shuffle=True
)

# Define a CNN model (using MobileNetV2 as the base model)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the pre-trained layers of the MobileNetV2 base model

# Build the model on top of MobileNetV2 base
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dense(num_classes, activation='softmax')  # Softmax for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)

# Save the trained model
model.save('tomato_leaf_disease_model.h5')

# Print model summary
model.summary()
