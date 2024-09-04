import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import Config

def sort_images(model_path, image_dir, img_size):
    """
    Sort images into 'hotspots' and 'no_hotspots' directories based on model predictions.

    Parameters:
    model_path (str): Path to the pre-trained model.
    image_dir (str): Directory containing images to sort.
    img_size (tuple): Target size to resize images (width, height).

    Returns:
    None
    """
    # Load the pre-trained model
    model = load_model(model_path)

    # Walk through the image directory and its subdirectories
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith(".png"):
                filepath = os.path.join(root, filename)
                try:
                    # Load and preprocess the image
                    image = load_img(filepath, target_size=img_size)
                    image = img_to_array(image)
                    image = np.expand_dims(image, axis=0)
                    image = image.astype('float32') / 255.0

                    # Predict class using the model
                    prediction = model.predict(image)
                    predicted_class = np.argmax(prediction, axis=1)[0]

                    # Determine the output subfolder (hotspots or no_hotspots)
                    if predicted_class == 1:  
                        subfolder = 'hotspots'
                    else:
                        subfolder = 'no_hotspots'

                    # Replicate the subfolder structure in the output directory
                    relative_path = os.path.relpath(root, image_dir)
                    output_subdir = os.path.join(image_dir, 'sorted', subfolder, relative_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Move the file to the appropriate directory
                    shutil.move(filepath, os.path.join(output_subdir, filename))

                    print(f"Processed {filename}: {subfolder}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    

if __name__ == "__main__":
    """
    Main execution block to run the image sorting based on model predictions.
    """

    # User-defined paths and parameters
    model_path = 'Set model file path'  
    image_dir = 'Set the directory containing images' 

    # Run the image sorting
    sort_images(model_path, image_dir, Config.IMG_SIZE)
