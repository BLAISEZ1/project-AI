import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")
        



def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    
    The dataset is organized with one subdirectory per category, labeled from 0 to NUM_CATEGORIES-1.
    Each category folder contains images of traffic signs.

    Returns:
        A tuple (images, labels), where:
        - images is a list of numpy.ndarray, each representing an image resized to (IMG_WIDTH, IMG_HEIGHT, 3).
        - labels is a list of integers, representing the category of each image.
    """
    images = []
    labels = []

    # Iterate over each category (0 to NUM_CATEGORIES-1)
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        # Ensure the directory exists
        if not os.path.exists(category_path):
            continue
        
        # Iterate through each image in the category folder
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            
            # Read the image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                continue  # Skip files that are not images
            
            # Resize the image to (IMG_WIDTH, IMG_HEIGHT)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Append image and label
            images.append(img)
            labels.append(category)

    # Convert lists to numpy arrays for better performance in TensorFlow
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model for traffic sign classification.
    
    The model consists of:
    - Two convolutional layers followed by max pooling.
    - A fully connected layer with dropout to reduce overfitting.
    - A softmax output layer for classification into NUM_CATEGORIES.
    
    Returns:
        A compiled TensorFlow Keras model.
    """
    model = keras.Sequential([
        # Convolutional Layer 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flattening the layers to feed into Dense layers
        layers.Flatten(),

        # Fully connected layer
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # Dropout to reduce overfitting

        # Output layer with softmax activation for classification
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
