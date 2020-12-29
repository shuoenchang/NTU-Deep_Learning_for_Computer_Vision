import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image


class CockRoach_Dataset(Dataset):
    def __init__(self, root, mode='train', frame_per_dir=11, height=224, width=224, transform=None):
        # train, val, or test
        self.mode = mode

        # image root dir
        self.root = root

        # default Transform
        self.transform = transforms.Compose([
            lambda x: Image.open(x),
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.transform_to_gray = transforms.Compose([
            lambda x: Image.open(x),
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((32, 32))
        ])
        
        if transform:
            self.transform = transform

        # 11 frames for oulu, 10 frames for SiW
        self.frame_per_dir = frame_per_dir
        self.height = height
        self.width = width
        self.video_paths = sorted(os.listdir(self.root))
        if mode != 'test':
            self.video_paths = sorted(
                self.video_paths, key=lambda x: int(x.split('_')[-2]))

    def __getitem__(self, idx):
        video, video_binary, phone, session, humanID, access_type = self.read_video(
            self.video_paths[idx])
        if access_type == 1:
            label = 1
        elif access_type == -1:
            label = -1
        else:
            label = 0
            if self.mode == 'train':
                video_binary = torch.zeros_like(video_binary)
        return {
            'video': video,
            'binary': video_binary,
            'phone': phone,
            'session': session,
            'humanID': humanID,
            'label': label,
            'video_id': self.video_paths[idx]
        }

    def __len__(self):
        return len(self.video_paths)

    def read_video(self, video_dir):
        video_path = os.path.join(self.root, video_dir)
        img_paths = os.listdir(video_path)

        # Dir name : root/XXXX if test mode
        # Dir name : root/{phone}_{session}_{human ID}_{access type} if not test mode
        video_label = [-1]*4
        if self.mode != 'test':
            video_label = video_dir.split('_')

        video = torch.zeros((self.frame_per_dir, 3, self.height, self.width))
        video_binary = torch.zeros((self.frame_per_dir, 32, 32))
        for i, img_path in enumerate(img_paths):
            path = os.path.join(self.root, os.path.join(video_dir, img_path))
            img = self.transform(path)
            video[i] = img
            gray = self.transform_to_gray(path)
            gray = gray.squeeze(0)
            video_binary[i][gray > 0] = 1
        # return ( torch tensor 10 * h * w , phone, session, humanID, access_type )
        return video, video_binary, int(video_label[0]), int(video_label[1]), int(video_label[2]), int(video_label[3])
