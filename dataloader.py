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

class MyDataset(Dataset):
    def __init__(self):
        self.data_path = 'TRAIN_DATA/'
        self.data_list = os.listdir(self.data_path)
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        with open(self.data_path + self.data_list[idx], 'rb') as f:
            data = pickle.load(f)
        image = data['image']
        label = int(data['label'])
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = image / 255.0
        image = torch.from_numpy(image)

        #print(self.data_list[idx])

        # one hot encoding label
        one_hot = torch.zeros(34)
        one_hot[label] = 1

        return image, one_hot
# Compare this snippet from train.py: