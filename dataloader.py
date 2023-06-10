# pytorch dataset
# load .pkl files in TRAIN_DATA
# it's dictionary type. key is 'image' and 'label'
# return torch type image and int type label
#
import torch
import pickle
from torch.utils.data import Dataset
import os
import numpy as np
from CONFIG.AUGMENTATION_CONFIG import *
import cv2

class MyDataset(Dataset):
    def __init__(self, augmentation, one_hot = True):
        self.one_hot = one_hot
        self.data_path = 'AUGMENTED_DATA/'
        self.data_list = os.listdir(self.data_path)
        # shuffle the data list
        np.random.shuffle(self.data_list)

        self.augmentation_bool = augmentation

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        with open(self.data_path + self.data_list[idx], 'rb') as f:
            data = pickle.load(f)
        image = data['image']
        label = int(data['label'])

        if self.augmentation_bool:
            image = self.augmentation(image)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)

        #print(self.data_list[idx])

        # one hot encoding label
        if self.one_hot:
            one_hot = torch.zeros(35)
        else:
            one_hot = torch.zeros(34)
        one_hot[label] = 1

        return image, one_hot
    
    def augmentation(self, image):
        image = self.augment_hsv(image, hgain = hsv_h, sgain = hsv_s, vgain = hsv_v)
        image = self.rotate(image, angle = degrees)
        image = self.translation(image, ratio = translate)
        image = self.shear(image, ratio = shear)
        image = self.flip(image, ud = flipud, lr = fliplr)
        image = self.mosaic(image, ratio = mosaic_ratio)
        image = self.noise(image, ratio = noise_ratio)
        return image
    
    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        return im
    
    def rotate(self, image, angle):
        # get random number -angle ~ angle
        angle = np.random.uniform(-angle, angle)
        # rotate image using cv2
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def translation(self, image, ratio):
        # get random number -ratio ~ ratio
        x = np.random.uniform(-ratio, ratio) * image.shape[1]
        y = np.random.uniform(-ratio, ratio) * image.shape[0]
        # translate image using cv2
        M = np.float32([[1, 0, x], [0, 1, y]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return image

    def shear(self, image, ratio):
        # get random number -ratio ~ ratio
        shear = np.random.uniform(-ratio, ratio)
        # shear image using cv2
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return image
    
    def flip(self, image, ud, lr):
        # check if the probabilty is ud
        if np.random.random() < ud:
            # flip up-down
            image = np.flipud(image)
        # check if the probabilty is lr
        if np.random.random() < lr:
            # flip left-right
            image = np.fliplr(image)
        return image
    
    def mosaic(self, image, ratio):
        ratio = int(image.shape[0] * ratio)
        original_shape = image.shape[0]
        image = cv2.resize(image, (ratio, ratio))
        image = cv2.resize(image, (original_shape, original_shape))
        return image
    
    def noise(self, image, ratio):
        # get random number -ratio ~ ratio
        noise = np.random.uniform(-ratio, ratio, image.shape)
        # add noise to image
        image = image + noise
        # clip to 0 ~ 255
        image = np.clip(image, 0, 255)
        return image
    
