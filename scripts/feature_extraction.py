import numpy as np
from PIL import Image
import cv2

def preprocess_image(img, target_size=(224, 224)):
    # Resize and normalize the image
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize the image to the range [0, 1]
    return img_array

def ensure_uint8(img):
    # Ensure that image data is in uint8 format for OpenCV
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img

def extract_corners(img):
    img = ensure_uint8(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    return np.sum(corners > 0.01 * corners.max())

def extract_centering(img):
    img = ensure_uint8(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    img_center = (img.shape[1] // 2, img.shape[0] // 2)
    card_center = (x + w // 2, y + h // 2)
    return np.linalg.norm(np.array(img_center) - np.array(card_center))

def extract_edges(img):
    img = ensure_uint8(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    return np.sum(edges) / (img.shape[0] * img.shape[1])

def extract_features(image):
    # Extract features from the processed image
    corners = extract_corners(image)
    centering = extract_centering(image)
    edges = extract_edges(image)
    return [corners, centering, edges]
