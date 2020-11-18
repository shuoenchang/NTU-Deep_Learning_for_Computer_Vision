import argparse

import numpy as np
import torch
import torch.optim as optim
import torchvision
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.misc import imsave
from src.dataset import FaceDataset
from models.GAN import Generator


def generate(G, device, output_path, seed):
    latent = torch.randn((32, 100, 1, 1)).to(device)
    fake_image = G(latent)
    torchvision.utils.save_image(
        fake_image.cpu().data, 'outputs/q2/{}.png'.format(seed), nrow=8, normalize=True, range=(-1, 1))
    
def main(args):
    print('seed: {}'.format(args.seed))
    torch.manual_seed(args.seed)
    
    device = 'cuda'
    G = Generator(100).to(device)
    G.load_state_dict(torch.load(
        '/home/en/SSD/DLCV/hw3-shuoenchang/weights/q2.pth', map_location=device))
    
    generate(G, device, output_path=args.save_folder, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw3-2 Testing Script')
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--seed', type=int, default=19)
    args = parser.parse_args()
    main(args)
