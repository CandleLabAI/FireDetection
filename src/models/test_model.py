import sys
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data"))

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from data_generator import get_test_images_from_generator
from network import *


def main():
    test_images = get_test_images_from_generator(imageShape=IMAGE_SIZE, batchSize=BATCH_SIZE)
    model = tf.keras.models.load_model(TRAINED_MODEL_SAVE_DIR + "/xception_trained.h5")
    warnings.filterwarnings('ignore')
    pred_probabilities = model.predict_generator(test_images, verbose=1)
    predictions = pred_probabilities > 0
    print(classification_report(test_images.classes, predictions))
    print(confusion_matrix(test_images.classes, predictions))


if __name__ == '__main__':
    main()
