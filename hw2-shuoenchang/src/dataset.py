import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from scipy.misc import imread


class p1Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = sorted(files for files in
                                os.listdir(self.root_dir) if files.endswith('.png'))
        if transform:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.file_list[idx]
        image = imread(os.path.join(self.root_dir, image_name))
        image = self.transform(image)
        target = image_name.split('_')[0]
        sample = {'image': image, 'target': target}
        return sample


if __name__ == '__main__':
    p1 = p1Dataset('hw2_data/p1_data/train_50')
    print(p1[0]['image'])
    print(p1[0]['target'])
    print(p1[0]['image'].shape)
    print(type(p1[0]['image']))
