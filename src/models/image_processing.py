import os
import sys
import numpy as np
from skimage import color
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data"))

from config import *


def classification_img_processing(img):
    lab = color.rgb2lab(img)
    h = lab.shape[0]  # height
    w = lab.shape[1]  # width
    for row in range(h):
        for col in range(w):
            # Applying limit of hue from 10◦ to 90◦ and chroma of 19.
            if lab[row][col][2] > 0 and lab[row][col][1] > 0 and (lab[row][col][2] * (1.0 / lab[row][col][1]) > 0.1763):
                if (lab[row][col][2] * lab[row][col][2] + lab[row][col][1] * lab[row][col][1]) >= 361:
                    return 1
    return 0


def print_count(images_path, class_type):
    input_dir_path = os.path.join(images_path, class_type)
    total = 0
    fire_count = 0
    for image_filename in tqdm(os.listdir(input_dir_path)):
        total += 1
        image_path = os.path.join(input_dir_path, image_filename)
        img = np.array(Image.open(image_path))
        result = classification_img_processing(img)
        fire_count += result

    print("Fire: ", fire_count)
    print("No Fire: ", total - fire_count)


def main():
    print("== TEST IMAGES == CLASS: FIRE")
    print_count(TEST_IMAGES_DIR, "Fire")
    print("== TEST IMAGES == CLASS: NO FIRE")
    print_count(TEST_IMAGES_DIR, "No_Fire")


if __name__ == '__main__':
    main()
