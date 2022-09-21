import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from config import *

from tensorflow.keras import regularizers
from tensorflow.keras.applications import Xception, InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


def get_xception_model():
    model = Sequential()
    pretrained_model = Xception(include_top=False,
                                input_shape=IMAGE_SIZE,
                                pooling='avg',
                                classes='2',
                                weights='imagenet')
    for _, layers in enumerate(pretrained_model.layers):
        layers.trainable = True
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(units=512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_regularizer=regularizers.l2(1e-3), activation="linear"))
    model.compile(loss='hinge',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_inception_resnet_v2_model():
    model = Sequential()
    pretrained_model = InceptionResNetV2(include_top=False,
                                         input_shape=IMAGE_SIZE,
                                         pooling='avg',
                                         classes='2',
                                         weights='imagenet')
    for i, layers in enumerate(pretrained_model.layers):
        if i < 10:
            layers.trainable = False
        else:
            layers.trainable = True
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(units=512,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128,
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
