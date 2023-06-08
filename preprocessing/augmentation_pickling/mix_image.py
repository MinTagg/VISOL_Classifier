import cv2
import random
import numpy as np

def mix_image(image1, image2, target_size = (224, 224, 3), color = False):
    """
    image1 = light bulb
    image2 = grill
    """

    value = 0.7

    if image1.shape != target_size:
        image1 = cv2.resize(image1, target_size[:2])
    if image2.shape != target_size:
        image2 = cv2.resize(image2, target_size[:2])

    left_position = int((95/224) * target_size[0])
    right_position = int((190/224) * target_size[1])

    left = image1[:, :left_position, :]
    right = image1[:, right_position:, :]
    moddle = image2[:, left_position:right_position, :]

    mixture = np.concatenate((left, moddle, right), axis=1)
    
    if color:
        # make gray scale
        gray = cv2.cvtColor(mixture, cv2.COLOR_BGR2GRAY)
        # 
        r = (np.array(random.uniform(value, 1) * gray, dtype = np.int32))
        g = (np.array(random.uniform(value, 1) * gray, dtype = np.int32))
        b = (np.array(random.uniform(value, 1) * gray, dtype = np.int32))
        mixture = np.stack((r, g, b), axis=2)
    return mixture