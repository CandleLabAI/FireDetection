import sys
import os
import warnings

sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "src", "data"))

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import *
from data_generator import getTestImagesFromGenerator
from network import *


def main():
    testImages = getTestImagesFromGenerator(imageShape=IMAGE_SIZE, batchSize=BATCH_SIZE)
    model = tf.keras.models.load_model(TRAINED_MODEL_SAVE_DIR + "/xception_trained.h5")
    warnings.filterwarnings('ignore')
    pred_probabilities = model.predict_generator(testImages, verbose=1)
    predictions = pred_probabilities > 0
    print(classification_report(testImages.classes, predictions))
    print(confusion_matrix(testImages.classes, predictions))


if __name__ == '__main__':
    main()
