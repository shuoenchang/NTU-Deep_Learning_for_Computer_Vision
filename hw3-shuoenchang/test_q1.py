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
        for j, data in enumerate(dataset):
            image = data['image'].to(device)
            name = data['name']
            output, _, _ = model(image)
            loss = F.mse_loss(image, output)
            output = output.permute(0, 2, 3, 1)
            output = output.cpu()
            # output = (output*255).type(torch.uint8)
            for i in range(len(output)):
                imsave(output_path+'/'+name[i], image[i].permute(1, 2, 0).cpu())
                imsave(output_path+'/{:.3f}_'.format(loss.item())+name[i], output[i])
            print('test loss: {}'.format(loss))
            
            if j==9: break


def generate(model, device, output_path, seed):
    
    model.eval()
    with torch.no_grad():
        output = model.construct(num_image=32)
        torchvision.utils.save_image(output.cpu().data, output_path, nrow=8)
    
def main(args):
    torch.manual_seed(args.seed)
    device = 'cuda'
    model = VAE2(1024).to(device)
    model.load_state_dict(torch.load(
        'weights/q1.pth', map_location=device))
    if args.generate:
        generate(model, device, output_path=args.save_folder, seed=args.seed)
    else:
        test_set = DataLoader(dataset=FaceDataset(args.image_folder, mode='test'),
                            batch_size=1, shuffle=True)
        test(test_set, model, device, output_path=args.save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw3-1 Testing Script')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--seed', type=int, default=594)
    parser.add_argument('--generate', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
