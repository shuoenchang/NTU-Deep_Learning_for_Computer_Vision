import os

import torch
import torchvision.transforms as transforms
from scipy.misc import imread, imsave
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', normalize=False):
        assert(mode == 'train' or mode == 'test')
        self.mode = mode
        self.img_dir = root_dir+'/'+self.mode
        self.attr_list = pd.read_csv(root_dir+'/'+mode+'.csv')
        if transform:
            self.transform = transforms
        elif normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
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
    def __init__(self, root_dir, subset, transform=None, mode='train', domain='source', normalize=False):
        assert(mode == 'train' or mode == 'test' or mode == 'val' or mode == 'feature')
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
        elif mode == 'feature':
            self.img_dir = root_dir+'/'+subset+'/test'
            self.attr_list = pd.read_csv(root_dir+'/'+subset+'/test.csv')
            self.attr_list = self.attr_list[int(len(self.attr_list)*0.7):]
        else:
            self.img_dir = root_dir
            self.attr_list = sorted(files for files in
                                    os.listdir(self.img_dir) if files.endswith('.png'))

        if transform:
            self.transform = transforms
        elif normalize:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.mode == 'test':
            image_name = self.attr_list[idx]
            image = imread(os.path.join(self.img_dir, image_name), mode='RGB')
            image = self.transform(image)
            domain = 0. if self.domain == 'source' else 1.
            sample = {'image': image, 'name': image_name, 'domain': domain}
            
        else:
            image_name = self.attr_list.iloc[idx]['image_name']
            image = imread(os.path.join(self.img_dir, image_name), mode='RGB')
            image = self.transform(image)
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
