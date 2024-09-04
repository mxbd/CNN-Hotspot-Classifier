
### Contents

- **`src/__init__.py`:** Initializes the `src` module, enabling relative imports within modules
  
- **`src/config.py`:** Contains all configuration settings for project (input directories, image size, batch size, number of epochs etc.)

- **`src/data_loader.py`:** Contains functions for loading and preprocessing data (includes reading images, resizing, normalizing, and splitting the data into training, val and test sets)

- **`src/model.py`:** Contains functions for defining, compiling, and training the model

- **`src/utils.py`:** Utility functions such as creating directories for storing results, saving models, and generating visualizations

- **`src/train.py`:** Script for running the training process

- **`src/tune.py`:** Script for running the hyperparameter tuning process

- **`src/sort_images.py`:** Script to define an input folder and model file to classify images with trained model

- **`runs/`:** Directory for saving the output from each training run, organized into subfolders (`run_01`, `run_02`, etc.)

### Environment Setup

- **Python Version:** The project was developed using Python 3.10.14

- **TensorFlow:** The project uses on TensorFlow for model creation and training.

  **Note:** 
  - **Anything above 2.10 is not supported on the GPU on Windows Native.**
  - Install TensorFlow with the following command:
    ```bash
    python -m pip install "tensorflow<2.11"
    ```

### How to Use

1. **Configure Settings:**
   - Open `src/config.py` and set the desired parameters such as directories, image size, batch size, and epochs.

2. **Run the Training Script:**
   - Execute the `train.py` script in the `src` directory to start the training process. The script will use the configurations set in `config.py`.

3. **OPTIONAL - Run the Tuning Script:**
   - Execute the `tune.py` script in the `src` directory to start the hyperparameter tuning process. Saves training run with "best" hyperparams in /runs folder.


