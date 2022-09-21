import sys
import os
import warnings

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data"))

from data_generator import get_train_images_from_generator
from network import *


def main():
    train_images = get_train_images_from_generator(imageShape=IMAGE_SIZE, batchSize=BATCH_SIZE)
    xception_model = get_xception_model()
    warnings.filterwarnings('ignore')
    results = xception_model.fit_generator(train_images, epochs=EPOCHS)
    xception_model.save(TRAINED_MODEL_SAVE_DIR + "/xception_trained.h5")
    inception_resnet_v2_model = get_inception_resnet_v2_model()
    results = inception_resnet_v2_model.fit_generator(train_images, epochs=EPOCHS)
    inception_resnet_v2_model.save(TRAINED_MODEL_SAVE_DIR + "/inception_resnet_trained.h5")


if __name__ == '__main__':
    main()
