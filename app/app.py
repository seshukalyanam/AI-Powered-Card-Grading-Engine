import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os
from tensorflow.keras.models import load_model
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from feature_extraction import preprocess_image, extract_features
# Download and load the model from Hugging Face
hf_token = os.getenv("HF_TOKEN")  # Fetch the token from environment variables
model_path = hf_hub_download(repo_id="ncompashf/card_grading_model", filename="card_grading_model.h5", token=hf_token)
model = load_model(model_path)

# Define the grade mapping
grade_mapping = {0: 2, 1: 5, 2: 10}

# Set up the Streamlit GUI
st.title("Card Grading App")
st.write("Upload an image of a card, and the model will predict its grade.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Classifying...")

    # Preprocess and predict
    img = preprocess_image(image)  # Resize and normalize as needed
    features = extract_features(img)  # Extract custom features

    # Reshape for model prediction
    img = img.reshape(1, 224, 224, 3)
    features = np.array([features])

    # Predict the grade
    prediction = model.predict([img, features])
    predicted_grade = grade_mapping[np.argmax(prediction)]

    # Display the result
    st.success(f"Predicted Grade: {predicted_grade}")
