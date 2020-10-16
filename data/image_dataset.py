import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import random
from os.path import join, dirname
from data.dataset_utils import *

class ImageDataset(data.Dataset):
    def __init__(self, name, split='train', val_size=0, img_transformer=None, mode='RGB'):
        if split == 'train':
            names, _, labels, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
        elif split =='val':
            _, names, _, labels = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % name), val_size)
        elif split == 'test':
            names, labels = get_dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % name))

        self.data_path = join(dirname(__file__), '..', 'datasets')
        self.names = names
        self.labels = labels
        self.mode = mode

        self._image_transformer = img_transformer

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert(self.mode)
        return img

    def __getitem__(self, index):
        img = self.get_image(index)
        sample = {'images': self._image_transformer(img),
                'class_labels': int(self.labels[index])}
        return sample

    def __len__(self):
        return len(self.names)

class ImageTestDataset(ImageDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert(self.mode)

        sample = {'images': self._image_transformer(img),
                'class_labels': int(self.labels[index])}
        return sample
