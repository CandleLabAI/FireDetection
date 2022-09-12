import sys
import os
import warnings

sys.path.append(os.path.join(os.getcwd(), "src"))
sys.path.append(os.path.join(os.getcwd(), "src", "data"))

from config import *
from data_generator import getTrainImagesFromGenerator
from network import *

def main():
    trainImages = getTrainImagesFromGenerator(imageShape=IMAGE_SIZE, batchSize=BATCH_SIZE)
    xceptionModel = getXceptionModel()
    warnings.filterwarnings('ignore')
    results = xceptionModel.fit_generator(trainImages, epochs=EPOCHS)
    xceptionModel.save(TRAINED_MODEL_SAVE_DIR + "/xception_trained.h5")
    inceptionResnetModel = getInceptionResnetV2Model()
    results = inceptionResnetModel.fit_generator(trainImages, epochs=EPOCHS)
    inceptionResnetModel.save(TRAINED_MODEL_SAVE_DIR + "/inception_resnet_trained.h5")


if __name__ == '__main__':
    main()
