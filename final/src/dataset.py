import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import os
from PIL import Image
import random

from torchvision.transforms.transforms import ToPILImage


class CockRoach_Dataset(Dataset):
    def __init__(self, root, mode='train', frame_per_dir=11, height=224, width=224, transform=None):
        # train, val, or test
        self.mode = mode

        # image root dir
        self.root = root
        self.root_depth = root.replace(mode, 'depth/{}'.format(mode))

        # default Transform
        if self.mode == 'train':
            self.transform = transforms.Compose([
                lambda x: Image.open(x),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomAffine(degrees=15, translate=(0, 0.2),
                #                         scale=(0.8, 1.2), shear=(-5, 5, -5, 5)),
                transforms.ToTensor(),
                transforms.RandomErasing(),
                # transforms.Resize((height, width)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        else:
            self.transform = transforms.Compose([
                lambda x: Image.open(x),
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))
            ])

        self.transform_to_gray = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Grayscale(),
            # transforms.Resize((32, 32))
        ])

        self.depth = transforms.Compose([
            lambda x: Image.open(x),
            transforms.ToTensor(),
            transforms.Grayscale(),
            # transforms.Resize((32, 32))
        ])

        if transform:
            self.transform = transform

        # 11 frames for oulu, 10 frames for SiW
        self.frame_per_dir = frame_per_dir
        self.height = height
        self.width = width
        self.video_paths = sorted(os.listdir(self.root))
        self.depth_paths = sorted(os.listdir(self.root_depth))
        if mode != 'test':
            self.video_paths = sorted(
                self.video_paths, key=lambda x: int(x.split('_')[-2]))
            self.depth_paths = sorted(
                self.depth_paths, key=lambda x: int(x.split('_')[-2]))

    def __getitem__(self, idx):
        video, video_binary, phone, session, humanID, access_type = self.read_video(
            self.video_paths[idx], self.depth_paths[idx])
        video_binary_ = video_binary
        if access_type == 1:
            label = 1
        elif access_type == -1:
            label = -1
        else:
            label = 0
            video_binary = torch.zeros_like(video_binary)
        return {
            'video': video,
            'binary': video_binary,
            'binary_': video_binary_,
            'phone': phone,
            'session': session,
            'humanID': humanID,
            'label': label,
            'video_id': self.video_paths[idx]
        }

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, video_dir, depth_dir):
        video_path = os.path.join(self.root, video_dir)
        img_paths = os.listdir(video_path)
        random.shuffle(img_paths)

        # Dir name : root/XXXX if test mode
        # Dir name : root/{phone}_{session}_{human ID}_{access type} if not test mode
        video_label = [-1]*4
        if self.mode != 'test':
            video_label = video_dir.split('_')

        video = torch.zeros((self.frame_per_dir, 3, self.height, self.width))
        video_binary = torch.zeros((self.frame_per_dir, 32, 32))
        for idx, img_path in enumerate(img_paths):
            path = os.path.join(self.root, os.path.join(video_dir, img_path))
            img = self.transform(path)
            path = os.path.join(self.root, os.path.join(depth_dir, img_path))
            depth_img = self.depth(path)
            if self.mode == 'train':
                # RandomHorizontalFlip
                if random.random() > 0.5:
                    img = TF.hflip(img)
                    depth_img = TF.hflip(depth_img)
                
                # RandomResizedCrop
                i, j, h, w = transforms.RandomResizedCrop.get_params(img, (0.85, 1), (3/4, 4/3))
                img = TF.resized_crop(img, i, j, h, w, (self.height, self.width))
                depth_img = TF.resized_crop(depth_img, i, j, h, w, (32, 32))
                
                # RandomRotation
                degree = random.uniform(-15, 15)
                img = TF.rotate(img, degree)
                depth_img = TF.rotate(depth_img, degree)

            video[idx] = img
            video_binary[idx] = depth_img
                
            if idx+1 == self.frame_per_dir:
                break
        # return ( torch tensor 10 * h * w , phone, session, humanID, access_type )
        return video, video_binary, int(video_label[0]), int(video_label[1]), int(video_label[2]), int(video_label[3])
