import os

import torch
import torchvision.transforms as transforms
from scipy.misc import imread, imsave
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        assert(mode == 'train' or mode == 'test')
        self.mode = mode
        self.img_dir = root_dir+'/'+self.mode
        self.attr_list = pd.read_csv(root_dir+'/'+mode+'.csv')
        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.attr_list.iloc[idx]['image_name']
        image = imread(os.path.join(self.img_dir, image_name))
        image = self.transform(image)
        attr = self.attr_list.iloc[idx].to_dict()
        sample = {'image': image, 'attr': attr, 'name': image_name}
        return sample


class DigitDataset(Dataset):
    def __init__(self, root_dir, subset, transform=None, mode='train', domain='source'):
        assert(mode == 'train' or mode == 'test' or mode == 'val')
        assert(domain == 'source' or domain == 'target')
        self.mode = mode
        self.domain = domain
        if mode == 'train':
            self.img_dir = root_dir+'/'+subset+'/train'
            self.attr_list = pd.read_csv(root_dir+'/'+subset+'/train.csv')
            self.attr_list = self.attr_list[:int(len(self.attr_list)*0.9)]
        elif mode == 'val':
            self.img_dir = root_dir+'/'+subset+'/train'
            self.attr_list = pd.read_csv(root_dir+'/'+subset+'/train.csv')
            self.attr_list = self.attr_list[int(len(self.attr_list)*0.9):]
        else:
            self.img_dir = root_dir+'/'+subset+'/test'
            self.attr_list = pd.read_csv(root_dir+'/'+subset+'/test.csv')

        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.attr_list.iloc[idx]['image_name']
        image = imread(os.path.join(self.img_dir, image_name), mode='RGB')
        image = self.transform(image)
        attr = self.attr_list.iloc[idx].to_dict()
        domain = 0. if self.domain == 'source' else 1.
        sample = {'image': image, 'name': image_name,
                  'label': self.attr_list.iloc[idx]['label'], 'domain': domain}
        return sample


if __name__ == '__main__':
    digit = DigitDataset(root_dir='hw3_data/digits', subset='usps', mode='val')
    print(len(digit))
    print(digit[0]['name'])
    print(digit[0]['image'].shape)
    print(digit[0]['label'])
