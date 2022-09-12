import os

TRAIN_IMAGES_DIR = os.path.join(os.getcwd(), "src", "data", "BALANCED_FLAME_DATASET", "train")
TEST_IMAGES_DIR = os.path.join(os.getcwd(), "src", "data", "BALANCED_FLAME_DATASET", "test")
TRAINED_MODEL_SAVE_DIR = os.path.join(os.getcwd(), "src", "models", "trained_model")

INITIAL_LEARNING_RATE = 0.001
IMAGE_SIZE = (254, 254, 3)
COLOR_MODE = 'rgb'
BATCH_SIZE = 16
EPOCHS = 1
