import sys
import os

sys.path.append(os.path.join(os.getcwd(), "src"))

import tensorflow as tf
from config import *


def get_train_images_from_generator(imageShape, batchSize):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30,
                                                                      # rotate the image by a max of 30 degrees
                                                                      width_shift_range=0.20,
                                                                      # Shift the pic width by a max of 20%
                                                                      height_shift_range=0.20,
                                                                      # Shift the pic height by a max of 20%
                                                                      rescale=1 / 255,
                                                                      # Rescale the image by normalzing it.
                                                                      shear_range=0.2,
                                                                      # Shear means cutting away part of the image (
                                                                      # max 20%)
                                                                      zoom_range=0.2,
                                                                      # Zoom in by 20% max
                                                                      horizontal_flip=True,
                                                                      # Allow horizontal flipping
                                                                      fill_mode='nearest',
                                                                      # Fill in missing pixels with the nearest
                                                                      # filled value
                                                                      )
    train_images = image_generator.flow_from_directory(batch_size=batchSize,
                                                       directory=TRAIN_IMAGES_DIR,
                                                       color_mode=COLOR_MODE,
                                                       shuffle=True,
                                                       target_size=imageShape[:2],
                                                       class_mode='binary')
    return train_images


def get_test_images_from_generator(imageShape, batchSize):
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    test_images = image_generator.flow_from_directory(batch_size=batchSize,
                                                      directory=TEST_IMAGES_DIR,
                                                      color_mode=COLOR_MODE,
                                                      shuffle=True,
                                                      target_size=imageShape[:2],
                                                      class_mode='binary')
    return test_images
