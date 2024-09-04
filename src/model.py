from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def build_model(img_size):
    """
    Build a CNN model with specified input image size.

    Parameters:
    img_size (tuple): The size of the input images (width, height).

    Returns:
    Sequential: A compiled model.
    """
    
    # Manually update with the best hyperparameters from the tuning process using tune.py
    filters = 96
    kernel_size = 3
    units = 64
    dropout = 0.3
    learning_rate = 0.00057

    model = Sequential([

        # First Conv2D layer, followed by MaxPooling
        Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),

         # Second Conv2D layer with double the filters, followed by MaxPooling
        Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten output from Conv2D layers to prepare for Dense layers 3D -> 1D
        Flatten(),

        # Dense layer with units num of neurons
        Dense(units, activation='relu'),

        # Dropout layer
        Dropout(dropout),

        # Output layer with 2 neurons for binary classification
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, batch_size, epochs, X_val, y_val):
    """
    Train the CNN model using data augmentation.

    Parameters:
    model (Sequential): The compiled Keras model to train.
    X_train (numpy.ndarray): Training data.
    y_train (numpy.ndarray): Training labels.
    batch_size (int): The size of the batches during training.
    epochs (int): The number of epochs to train the model.
    X_val (numpy.ndarray): Validation data.
    y_val (numpy.ndarray): Validation labels.

    Returns:
    History: A Keras History object containing the training history.
    """

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                                 width_shift_range=0.2, height_shift_range=0.2,
                                 horizontal_flip=True)
    datagen.fit(X_train)

    # Train model using augmented data
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        verbose=1)
    return history
