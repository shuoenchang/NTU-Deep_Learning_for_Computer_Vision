import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image


def filenameToPILImage(x): return Image.open(x)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)


class GeneratorSampler(Sampler):
    def __init__(self, csv_path, n_batch=1, n_way=5, n_shot=1, n_query=15):
        csv_data = pd.read_csv(csv_path).set_index("id")
        self.n_batch = n_batch
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.labels = sorted(np.unique(csv_data['label']))
        self.n_class = len(self.labels)
        self.data_idx = []
        for label in self.labels:
            idx = np.argwhere(np.array(csv_data['label']) == label)
            self.data_idx.append(idx)
        self.data_idx = torch.LongTensor(self.data_idx).squeeze(2)

    def __iter__(self):
        for i in range(self.n_batch):
            support = []
            query = []
            rand_class = torch.randperm(self.n_class)[:self.n_way]
            for c in rand_class:
                rand_idx = torch.randperm(len(self.data_idx[c]))[:self.n_shot+self.n_query]  # noqa
                support.append(self.data_idx[c][rand_idx[:self.n_shot]])
                query.append(self.data_idx[c][rand_idx[self.n_shot:]])
            support = torch.stack(support).view(-1)
            query = torch.stack(query).view(-1)
            batch = torch.cat((support, query)).numpy()
            yield batch

    def __len__(self):
        return self.n_batch


class TestSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence)

    def __len__(self):
        return len(self.sampled_sequence)


if __name__ == '__main__':

    # test_dataset = MiniDataset('/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/val.csv',
    #                            '/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/val')
    # test_loader = DataLoader(
    #     test_dataset, batch_size=5 * (15 + 1),
    #     num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
    #     sampler=TestSampler('/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/val_testcase.csv'))

    train_dataset = MiniDataset('/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/train.csv',
                                '/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/train')
    train_loader = DataLoader(
        train_dataset, batch_sampler=GeneratorSampler(
            '/home/en/SSD/DLCV/hw4-shuoenchang/hw4_data/train.csv')
    )

    for i, data in enumerate(train_loader):
        image, label = data
        print(image.shape)
