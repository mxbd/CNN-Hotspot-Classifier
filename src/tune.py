import keras_tuner as kt
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import create_run_folder, save_results, count_class_distribution
from data_loader import prepare_data
from config import Config

def build_model(hp):
    """
    Build a CNN model with hyperparameter tuning using Keras Tuner.

    Parameters:
    hp (HyperParameters): Hyperparameters object used for tuning.

    Returns:
    Sequential: A compiled Keras Sequential model.
    """

    # Use img_size tuple from Config
    img_size = Config.IMG_SIZE

    # Define hyperparameters for tuning
    filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    kernel_size = hp.Choice('kernel_size', values=[3, 5])
    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

    # Build model
    model = Sequential([
        Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units, activation='relu'),
        Dropout(dropout),
        Dense(2, activation='softmax')
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    """
    Main execution block for hyperparameter tuning, model training, and evaluation.
    """

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        Config.HOTSPOT_DIR, Config.NO_HOTSPOT_DIR,
        Config.IMG_SIZE, Config.VALIDATION_SPLIT,
        Config.TEST_SPLIT, Config.RANDOM_STATE
    )

    # Convert one-hot encoded labels to class labels
    y_train_classes = np.argmax(y_train, axis=1)

    # Initialize tuner for hyperparameter search
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory=Config.RUNS_DIR,
        project_name='cnn_tuning'
    )

    # Perform hyperparameter search
    tuner.search(X_train, y_train, epochs=25, validation_data=(X_val, y_val))

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Rebuild the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    # Start time for training
    start_time = time.time()

    # Train the model with the best hyperparameters
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=Config.EPOCHS)

    # End time for training
    end_time = time.time()

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Calculate class distributions
    train_class_dist = count_class_distribution(y_train)
    val_class_dist = count_class_distribution(y_val)
    test_class_dist = count_class_distribution(y_test)

    # Save the model and results
    run_folder = create_run_folder(Config.RUNS_DIR)
    
    save_results(
        run_folder, 
        model, 
        history, 
        y_train_classes,
        y_val_classes, 
        y_val_pred_classes, 
        y_test_classes, 
        y_test_pred_classes,
        start_time, 
        end_time, 
        Config,
        len(y_train_classes),
        train_class_dist,
        len(y_val_classes),
        val_class_dist,
        len(y_test_classes),
        test_class_dist,
    )

    print(f"Training complete. Results saved in {run_folder}.")
