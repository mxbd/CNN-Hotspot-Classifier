import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_images(directory, label, img_size):
    """
    Load images from a directory, resize them, and label them.
    
    Parameters:
    directory (str): Path to the directory containing images.
    label (int): Label for the images (e.g., 0 for no hotspot, 1 for hotspot).
    img_size (tuple): Target size to resize the images (width, height).
    
    Returns:
    tuple: A tuple containing a list of images and a list of labels.
    """
    images, labels = [], []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            try:
                # Load image, convert to RGB, resize
                image = Image.open(filepath).convert('RGB')
                image = image.resize(img_size)
                images.append(np.array(image))
                labels.append(label)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    return images, labels


def prepare_data(hotspot_dir, no_hotspot_dir, img_size, validation_split, test_split, random_state):
    """
    Prepare the dataset by loading images, normalizing, encoding labels, and splitting into train, validation, and test sets.
    
    Parameters:
    hotspot_dir (str): Directory containing hotspot images.
    no_hotspot_dir (str): Directory containing no hotspot images.
    img_size (tuple): Target size to resize the images (width, height).
    validation_split (float): Fraction of the training data to use for validation.
    test_split (float): Fraction of the total data to reserve for testing.
    random_state (int): Seed for random number generator for reproducibility.
    
    Returns:
    tuple: Training, validation, and test datasets (features and labels).
    """
    # Load images and corresponding labels
    no_hotspot_images, no_hotspot_labels = load_images(no_hotspot_dir, 0, img_size)
    hotspot_images, hotspot_labels = load_images(hotspot_dir, 1, img_size)

    # Combine and preprocess
    images = np.array(hotspot_images + no_hotspot_images)
    labels = np.array(hotspot_labels + no_hotspot_labels)
    images = images.astype('float32') / 255.0  # # Normalize pixel values to [0, 1]
    labels = to_categorical(labels, num_classes=2)  # One-hot encode for binary classification

    # Split the data into training + validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=random_state
    )

    # Further split training + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_split, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
