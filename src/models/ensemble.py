import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data"))

from config import *
from image_processing import classification_img_processing


def get_model_results(model, threshold):
    fire_test_dir = os.path.join(TEST_IMAGES_DIR, "Fire")
    no_fire_test_dir = os.path.join(TEST_IMAGES_DIR, "No_Fire")
    results = []
    for filename in tqdm(os.listdir(fire_test_dir)):
        img_path = os.path.join(fire_test_dir, filename)
        im = Image.open(img_path)
        im = im.resize((254, 254))
        im = np.asarray(im, dtype=np.float32)
        if im.shape[0] == 254 and im.shape[1] == 254 and im.shape[2] == 3:
            im = im / 255.0
            input_data = im.reshape(1, 254, 254, 3)
            predictions = model.predict(input_data)
            if predictions < threshold:
                results.append([filename, 0, 0])
            else:
                results.append([filename, 1, 0])

    for filename in tqdm(os.listdir(no_fire_test_dir)):
        img_path = no_fire_test_dir + '/' + filename
        im = Image.open(img_path)
        im = np.asarray(im, dtype=np.float32)
        im = im / 255.0
        input_data = im.reshape(1, 254, 254, 3)
        predictions = model.predict(input_data)
        if predictions < threshold:
            results.append([filename, 0, 1])
        else:
            results.append([filename, 1, 1])
    return results


def generate_image_processing_results(images_path, class_type):
    input_dir_path = os.path.join(images_path, class_type)
    results = []
    for image_filename in tqdm(os.listdir(input_dir_path)):
        image_path = os.path.join(input_dir_path, image_filename)
        img = np.array(Image.open(image_path))
        result = classification_img_processing(img)
        if class_type == 'Fire':
            if result == 1:
                results.append([image_filename, 0, 0])
            else:
                results.append([image_filename, 1, 0])
        else:
            if result == 1:
                results.append([image_filename, 0, 1])
            else:
                results.append([image_filename, 1, 1])
    return results


def get_image_processing_results():
    fire_results = generate_image_processing_results(TEST_IMAGES_DIR, "Fire")
    no_fire_results = generate_image_processing_results(TEST_IMAGES_DIR, "No_Fire")
    return np.concatenate([fire_results, no_fire_results])


def ensemble(sum_dict, original):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for key in sum_dict:
        if int(sum_dict[key]) <= 1:
            if int(original[key]) == 0:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if int(original[key]) == 0:
                fn = fn + 1
            else:
                tn = tn + 1
    print(f"TRUE POSITIVE: {tp}")
    print(f"TRUE NEGATIVE: {tn}")
    print(f"FALSE POSITIVE: {fp}")
    print(f"FALSE NEGATIVE: {fn}")


def main():
    trained_models_path = os.path.join(ROOT_DIR, "models", "trained_model")
    xception_trained_model = tf.keras.models.load_model(os.path.join(trained_models_path, "xception_trained.h5"))
    inception_resnet_v2_trained_model = tf.keras.models.load_model(
        os.path.join(trained_models_path, "inception_resnet_trained.h5"))
    xception_results = get_model_results(xception_trained_model, 0)
    inception_resnet_v2_results = get_model_results(inception_resnet_v2_trained_model, 0.5)
    cielab_results = get_image_processing_results()
    original = {}
    cielab_dict = {}
    for i in range(len(cielab_results)):
        original[cielab_results[i][0]] = cielab_results[i][2]
        cielab_dict[cielab_results[i][0]] = cielab_results[i][1]
    xception_dict = {}
    for i in range(len(xception_results)):
        xception_dict[xception_results[i][0]] = xception_results[i][1]
    resnet_dict = {}
    for i in range(len(inception_resnet_v2_results)):
        resnet_dict[inception_resnet_v2_results[i][0]] = inception_resnet_v2_results[i][1]
    sum_dict = {}
    for key in cielab_dict:
        sum_dict[key] = int(cielab_dict[key]) + int(xception_dict[key]) + int(resnet_dict[key])
    ensemble(sum_dict, original)


if __name__ == '__main__':
    main()
