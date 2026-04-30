import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, Input, Model
from feature_extraction import preprocess_image, extract_features

data_dir = "./data"

# Load data
metadata_df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
X_images = []
X_features = []
y_labels = []

for _, row in metadata_df.iterrows():
    image_name = row['image_name']
    grade = row['grade']
    image_path = os.path.join(data_dir, image_name)
    
    # Preprocess image and extract features
    img = preprocess_image(image_path)
    features = extract_features(image_path)
    
    X_images.append(img)
    X_features.append(features)
    y_labels.append(grade)

# Prepare data arrays
X_images = np.array(X_images).astype(np.float32)
X_features = np.array(X_features).astype(np.float32)
y_labels = np.array([2 if label == 2 else 1 if label == 5 else 0 for label in y_labels])

# Build the model
image_input = Input(shape=(224, 224, 3), name="image_input")
x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)

feature_input = Input(shape=(3,), name="feature_input")
combined = layers.concatenate([x, feature_input])
x = layers.Dense(128, activation='relu')(combined)
output = layers.Dense(3, activation='softmax')(x)

model = Model(inputs=[image_input, feature_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X_images, X_features], y_labels, epochs=10, validation_split=0.2)
model.save("card_grading_model.h5")
