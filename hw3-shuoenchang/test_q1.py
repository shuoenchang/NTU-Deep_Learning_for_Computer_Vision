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
from models.VAE import VAE, VAE2


def test(dataset, model, device, output_path):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            output, _, _ = model(image)
            loss = F.mse_loss(image, output)
            output = output.permute(0, 2, 3, 1)
            output = output.cpu()
            # output = (output*255).type(torch.uint8)
            for i in range(len(output)):
                imsave(output_path+'/'+name[i], output[i])
            print('test loss: {}'.format(loss))
            break


def generate(model, device, output_path, seed):
    
    model.eval()
    with torch.no_grad():
        output = model.construct(num_image=32)
        torchvision.utils.save_image(output.cpu().data, output_path+'/{}.png'.format(args.seed), nrow=8)
    
def main(args):
    print('seed: {}'.format(args.seed))
    torch.manual_seed(args.seed)
    test_set = DataLoader(dataset=FaceDataset(args.image_folder, mode='test'),
                          batch_size=10, shuffle=True)
    device = 'cuda'
    model = VAE2(1024).to(device)
    model.load_state_dict(torch.load(
        '/home/en/SSD/DLCV/hw3-shuoenchang/weights/q1_/lamda_0.0001904-dim_1024-VAE2.pth', map_location=device))
    
    import os
    # args.save_folder = args.save_folder+'/{:03d}/'.format(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    
    # test(test_set, model, device, output_path=args.save_folder)
    generate(model, device, output_path=args.save_folder, seed=args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw3-1 Testing Script')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--seed', type=int, default=19)
    args = parser.parse_args()
    main(args)
