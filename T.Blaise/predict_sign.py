import sys
import cv2
import numpy as np
import tensorflow as tf
import os

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # Adjust based on dataset

# Dictionary mapping category numbers to sign names (simplified example)
SIGN_NAMES = {
    0: "Stop",
    1: "Speed Limit",
    2: "Yield",
    3: "No Entry",
    4: "Traffic Light",
    # Add all NUM_CATEGORIES labels here
}

def load_model(model_path="model.h5"):
    """
    Load the trained model from the given file path.
    """
    return tf.keras.models.load_model(model_path)

def process_image(image_path):
    """
    Read and process an image for prediction.
    Resizes the image to (IMG_WIDTH, IMG_HEIGHT, 3).
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        numpy.ndarray: The preprocessed image ready for prediction.
    """
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize to match training data size
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

    # Normalize pixel values (scale to range 0-1)
    img = img / 255.0  

    # Expand dimensions to match model input shape (batch size of 1)
    return np.expand_dims(img, axis=0)

def predict(image_path, model):
    """
    Predicts the category of a traffic sign from an image.
    
    Args:
        image_path (str): Path to the image file.
        model (tf.keras.Model): The trained neural network model.
    
    Returns:
        tuple: (predicted_category, sign_name, probability)
    """
    # Preprocess the image
    image = process_image(image_path)

    # Get model predictions
    predictions = model.predict(image)

    # Extract the predicted category (index of max probability)
    predicted_category = np.argmax(predictions)

    # Extract probability of the predicted category
    probability = np.max(predictions)

    # Get sign name from dictionary (fallback to "Unknown" if not found)
    sign_name = SIGN_NAMES.get(predicted_category, "Unknown")

    return predicted_category, sign_name, probability

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_sign.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load trained model
    model = load_model()

    # Get prediction
    category, sign_name, confidence = predict(image_path, model)

    # Print result
    print(f"Predicted Category: {category}")
    print(f"Traffic Sign: {sign_name}")
    print(f"Confidence: {confidence:.3f}")
