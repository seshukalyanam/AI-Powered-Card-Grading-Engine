# AI-Powered Card Grading Engine

## Overview

This project is an AI-powered card grading engine that predicts a card’s grade from an uploaded image. It combines image-based feature extraction with a custom neural network model to estimate one of three grades: 2, 5, or 10.

The repository contains:
- `app/app.py` — a Streamlit web app for uploading card images and showing predictions
- `scripts/feature_extraction.py` — image preprocessing and custom feature extraction logic
- `scripts/train.py` — training script for a hybrid image + feature model
- `scripts/preprocess.py` — helper script to download data from Azure Blob Storage
- `requirements.txt` — Python dependencies for local setup

---

## What This Project Does

The app accepts a card image, preprocesses it, extracts custom visual features, and uses a trained TensorFlow model to predict a grade.

Predicted grades are mapped from model output classes:
- `0` → `2`
- `1` → `5`
- `2` → `10`

---

## How the System Is Built

### 1. Image Preprocessing

The project uses `PIL` and `OpenCV` to:
- resize images to `224x224`
- normalize pixels to `[0, 1]`
- convert images for OpenCV feature extraction

### 2. Feature Extraction

`feature_extraction.py` extracts three custom features from each card image:
- `corners` — corner count estimated via Harris corner detection
- `centering` — distance between image center and the detected card bounding box center
- `edges` — normalized edge density using Canny edge detection

These features are supplied to the model alongside the image tensor.

### 3. Model Architecture

The training script builds a two-input Keras model:
- image input processed through convolutional layers
- custom feature input concatenated with image features
- final dense layers output a 3-class softmax prediction

The model is trained using `sparse_categorical_crossentropy` and saved as `card_grading_model.h5`.

### 4. Web App

The Streamlit app in `app/app.py`:
- loads the trained model from Hugging Face Hub
- reads the uploaded card image
- preprocesses the image and extracts features
- predicts the grade and displays it in the UI

---

## Setup

### Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

The app expects a Hugging Face token in `HF_TOKEN` to download the model from the Hub:

```bash
setx HF_TOKEN "your_hf_token"
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN = "your_hf_token"
```

### Azure Data Download

If you need to download the dataset from Azure Blob Storage, update `scripts/preprocess.py` with:
- `connection_string`
- `container_name`

Then run:

```bash
python scripts/preprocess.py
```

This will download `metadata.csv` and images into `./data`.

> Note: `scripts/preprocess.py` currently uses placeholder values. Do not commit real secrets to source control.

---

## Training

After preparing the data in `./data`, run the training script:

```bash
python scripts/train.py
```

This script expects:
- `./data/metadata.csv`
- image files listed in `metadata.csv`

It trains the model and saves `card_grading_model.h5` in the current directory.

---

## Running the App

Launch the Streamlit app with:

```bash
streamlit run app/app.py
```

Then open the URL shown in the terminal. Upload a card image and the app will display the predicted grade.

---

## File Structure

- `app/app.py` — Streamlit user interface and model inference
- `requirements.txt` — dependencies for the project
- `scripts/feature_extraction.py` — image processing and features
- `scripts/train.py` — model training pipeline
- `scripts/preprocess.py` — Azure blob downloader for data preparation
- `data/` — expected dataset folder containing images and `metadata.csv`
- `flagged/` — existing repository folder for flagged data or logs

---

## Notes and Improvements

This project is a lightweight proof of concept. Useful next steps include:
- fixing data-loading path handling in `scripts/train.py`
- using a properly configured Hugging Face repository ID in `app/app.py`
- expanding grade categories beyond 2, 5, and 10
- adding model evaluation and validation metrics
- packaging the app for production deployment

---

## Security

Sensitive values are not stored in the repository. The app and preprocessing scripts use placeholders for:
- Azure `connection_string`
- Azure `container_name`
- Hugging Face repo ID
- `HF_TOKEN`

Always keep these secrets out of Git history and use environment variables or secure secret storage.
