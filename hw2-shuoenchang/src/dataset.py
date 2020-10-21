import os

import torch
import torchvision.transforms as transforms
from scipy.misc import imread
from torch.utils.data import Dataset
import numpy as np


class p1Dataset(Dataset):
    def __init__(self, root_dir, transform=None, has_gt=True):
        self.root_dir = root_dir
        self.file_list = sorted(files for files in
                                os.listdir(self.root_dir) if files.endswith('.png'))
        self.has_gt = has_gt
        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.file_list[idx]
        image = imread(os.path.join(self.root_dir, image_name))
        image = self.transform(image)
        if self.has_gt:
            target = int(image_name.split('_')[0])
            sample = {'image': image, 'target': target}
        else:
            sample = {'image': image, 'name': image_name}
        return sample


class p2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, has_gt=True):
        self.root_dir = root_dir
        self.file_list = sorted(files for files in
                                os.listdir(self.root_dir) if files.endswith('.jpg'))
        self.has_gt = has_gt
        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.file_list[idx]
        image = imread(os.path.join(self.root_dir, image_name))
        image = self.transform(image)
        if self.has_gt:
            target_name = image_name.split('_')[0]+'_mask.png'
            mask = imread(os.path.join(self.root_dir, target_name))
            target = self.mask_to_target(mask)
            sample = {'image': image, 'target': target}
        else:
            sample = {'image': image, 'name': image_name}
        return sample

    def mask_to_target(self, mask):
        target = np.empty((512, 512), dtype=np.int)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        target[mask == 3] = 0  # (Cyan: 011) Urban land
        target[mask == 6] = 1  # (Yellow: 110) Agriculture land
        target[mask == 5] = 2  # (Purple: 101) Rangeland
        target[mask == 2] = 3  # (Green: 010) Forest land
        target[mask == 1] = 4  # (Blue: 001) Water
        target[mask == 7] = 5  # (White: 111) Barren land
        target[mask == 0] = 6  # (Black: 000) Unknown
        target[mask == 4] = 6  # (Red: 100) Unknown
        return target


if __name__ == '__main__':
    p2 = p2Dataset(root_dir='hw2_data/p2_data/train')
    print(p2[26]['target'])
    print(p2[26]['image'].shape)
    print(type(p2[26]['target']))
