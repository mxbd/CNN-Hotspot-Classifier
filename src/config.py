import os

class Config:
    # Directories
    HOTSPOT_DIR = 'Path to hotspots dir'
    NO_HOTSPOT_DIR = 'Path to no hotspots dir'
    RUNS_DIR = RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs') 

    # Training parameters
    IMG_SIZE = (128, 128) # Image size for input  
    BATCH_SIZE = 32 # Batch size for training
    EPOCHS = 30 # Number of epochs for training
    VALIDATION_SPLIT = 0.2 # Fraction of data to use for validation
    TEST_SPLIT = 0.1 # Fraction of data to use for testing
    RANDOM_STATE = 30 # Seed for reproducibility
