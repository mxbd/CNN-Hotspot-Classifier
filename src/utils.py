import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def create_run_folder(base_path="runs"):
    """
    Creates a new run folder for saving model and evaluation results.
    """
    os.makedirs(base_path, exist_ok=True)
    run_id = 1
    while True:
        run_folder = os.path.join(base_path, f"run_{run_id:02d}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            os.makedirs(os.path.join(run_folder, "evaluation"))
            os.makedirs(os.path.join(run_folder, "model"))
            return run_folder
        run_id += 1


def count_class_distribution(y_classes):
    """
    Count the distribution of classes in a dataset.

    Parameters:
    y_classes (np.array): Array of class labels.

    Returns:
    tuple: Count of class 0 and class 1 examples.
    """
    count_class_0 = (y_classes == 0).sum()
    count_class_1 = (y_classes == 1).sum()
    return count_class_0, count_class_1


def save_results(run_folder, model, history, y_train, 
                 y_val_classes, y_val_pred_classes, 
                 y_test_classes, y_test_pred_classes,
                 start_time, end_time, config,
                 num_train, train_class_dist,
                 num_val, val_class_dist,
                 num_test, test_class_dist):
    """
    Save the model, training history, and evaluation metrics to the specified run folder.

    Parameters:
    run_folder (str): Path to the run folder.
    model (tf.keras.Model): The trained model.
    history (tf.keras.callbacks.History): Training history of the model.
    y_train (np.array): Training labels.
    y_val_classes (np.array): Actual validation labels.
    y_val_pred_classes (np.array): Predicted validation labels.
    y_test_classes (np.array): Actual test labels.
    y_test_pred_classes (np.array): Predicted test labels.
    start_time (float): Start time of the training process.
    end_time (float): End time of the training process.
    config (module): Configuration module with training parameters.
    num_train (int): Number of training examples.
    train_class_dist (tuple): Class distribution in training set.
    num_val (int): Number of validation examples.
    val_class_dist (tuple): Class distribution in validation set.
    num_test (int): Number of test examples.
    test_class_dist (tuple): Class distribution in test set.

    Returns:
    None
    """
    
    # Save the model
    model_folder = os.path.join(run_folder, "model")
    model.save(os.path.join(model_folder, 'cnn_hotspot_classifier.h5'))

    # Calculate accuracy and classification report for validation set
    val_accuracy = accuracy_score(y_val_classes, y_val_pred_classes)
    val_classification_rep = classification_report(y_val_classes, y_val_pred_classes)

    # Calculate accuracy and classification report for test set
    test_accuracy = accuracy_score(y_test_classes, y_test_pred_classes)
    test_classification_rep = classification_report(y_test_classes, y_test_pred_classes)

    # Calculate training time
    training_time = end_time - start_time

    # Get hardware info
    devices = tf.config.experimental.list_physical_devices()
    hardware_info = "\n".join([f"Device: {device.name}, Type: {device.device_type}" for device in devices])

    # Save the results to a text file in the "evaluation" subfolder
    evaluation_folder = os.path.join(run_folder, "evaluation")
    with open(os.path.join(evaluation_folder, 'results.txt'), 'w') as f:
        
        f.write("Validation and Test Results\n")
        f.write("==============================\n")
        f.write("\nValidation Results:\n")
        f.write(f"Accuracy: {val_accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(val_classification_rep)
        f.write("\nTest Results:\n")
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(test_classification_rep)

        f.write("\n\nTraining and Model Information\n")
        f.write("==============================\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Hardware Used:\n{hardware_info}\n")
        f.write(f"Epochs: {config.EPOCHS}\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Image Size: {config.IMG_SIZE}\n")
        f.write(f"Validation Split: {config.VALIDATION_SPLIT}\n")
        f.write(f"Test Split: {config.TEST_SPLIT}\n")
        f.write(f"Random State: {config.RANDOM_STATE}\n")

        # Include number of examples with class distribution
        f.write("\nNumber of examples:\n")
        f.write(f"Training: {num_train} ({train_class_dist[0]}/{train_class_dist[1]})\n")
        f.write(f"Validation: {num_val} ({val_class_dist[0]}/{val_class_dist[1]})\n")
        f.write(f"Test: {num_test} ({test_class_dist[0]}/{test_class_dist[1]})\n")

    # Create and save visualizations
    plot_accuracy_loss(history, evaluation_folder)
    plot_confusion_matrix(y_val_classes, y_val_pred_classes, evaluation_folder, 'val_confusion_matrix.png')
    plot_confusion_matrix(y_test_classes, y_test_pred_classes, evaluation_folder, 'test_confusion_matrix.png')


def plot_accuracy_loss(history, save_dir):
    """
    Plot and save accuracy and loss curves from the training history.

    Parameters:
    history (tf.keras.callbacks.History): Training history of the model.
    save_dir (str): Directory where the plots will be saved.

    Returns:
    None
    """

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))

    # Plotting model loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='orange')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))

    plt.close()

def plot_confusion_matrix(y_true_classes, y_pred_classes, save_dir, filename):
    """
    Plot and save a confusion matrix.

    Parameters:
    y_true_classes (np.array): True class labels.
    y_pred_classes (np.array): Predicted class labels.
    save_dir (str): Directory where the plot will be saved.
    filename (str): Filename for the saved confusion matrix plot.

    Returns:
    None
    """
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Hotspot', 'Hotspot'],
                yticklabels=['No Hotspot', 'Hotspot'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

