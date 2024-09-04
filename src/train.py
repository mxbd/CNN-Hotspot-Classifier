import time
from config import Config
from data_loader import prepare_data
from model import build_model, train_model
from utils import create_run_folder, save_results, count_class_distribution
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

if __name__ == "__main__":
    """
    Main script to load data, build and train the model, evaluate the model, and save results.
    """

    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        Config.HOTSPOT_DIR, Config.NO_HOTSPOT_DIR,
        Config.IMG_SIZE, Config.VALIDATION_SPLIT,
        Config.TEST_SPLIT, Config.RANDOM_STATE
    )

    # Convert one-hot encoded labels to class labels
    y_train_classes = np.argmax(y_train, axis=1) # otherwise problem with Number of examples shown for training

    # Build and train the model
    start_time = time.time()
    model = build_model(Config.IMG_SIZE)
    history = train_model(model, X_train, y_train, Config.BATCH_SIZE, Config.EPOCHS, X_val, y_val)
    end_time = time.time()  

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    val_accuracy = accuracy_score(y_val_classes, y_val_pred_classes)
    val_classification_rep = classification_report(y_val_classes, y_val_pred_classes)

    print(f"Validation Accuracy: {val_accuracy}")
    print("Validation Classification Report:")
    print(val_classification_rep)

    # Evaluate the model on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.argmax(y_test_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_test_classes, y_test_pred_classes)
    test_classification_rep = classification_report(y_test_classes, y_test_pred_classes)

    print(f"Test Accuracy: {test_accuracy}")
    print("Test Classification Report:")
    print(test_classification_rep)

    # # Create a directory for saving the results and model
    run_folder = create_run_folder(Config.RUNS_DIR)

    # Calculate the number of examples and class distribution
    num_train, train_class_dist = len(y_train), count_class_distribution(y_train_classes)
    num_val, val_class_dist = len(y_val), count_class_distribution(y_val_classes)
    num_test, test_class_dist = len(y_test), count_class_distribution(y_test_classes)

    # Save the results, model, and training history
    save_results(
        run_folder, 
        model, 
        history, 
        y_train,
        y_val_classes, 
        y_val_pred_classes, 
        y_test_classes, 
        y_test_pred_classes,
        start_time, 
        end_time, 
        Config,
        num_train,
        train_class_dist,
        num_val,
        val_class_dist,
        num_test,
        test_class_dist,
    )

    print(f"Training complete. Results saved in {run_folder}.")
