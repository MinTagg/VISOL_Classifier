from mix_image import mix_image
import os
import pickle
import cv2
import numpy as np
import random

# PARAMETERS
source_path = 'TRAIN_DATA/'
save_path = 'AUGMENTATION_DATA/'
save_img_path = 'temp/'
NUMBER_OF_DATA = 1000

# load the pickle file in TRAIN_DATA
data_list = os.listdir(source_path)
data_list = [file for file in data_list if file.endswith(".pkl")]

count = 0
while count < NUMBER_OF_DATA:
    # select two random data
    data1, data2 = random.sample(data_list, 2)
    # load the pickle file
    with open(source_path + data1, 'rb') as f:
        data1 = pickle.load(f)
    with open(source_path + data2, 'rb') as f:
        data2 = pickle.load(f)
    # check the label is not same
    if data1['label'] == data2['label']:
        pass
    else:
        # mix two image
        img = mix_image(data1['image'], data2['image'])
        # save image in AUGMENTATION_DATA
        cv2.imwrite(save_img_path + str(count).zfill(5) + '.png', img)
        # pickle the image and label in AUGMENTATION_DATA
        data = {'image' : img, 'label' : 34.0}
        with open(save_path + str(count).zfill(5) + '.pkl', 'wb') as f:
            pickle.dump(data, f)
        count += 1
        if count % 100 == 0:
            print(count)