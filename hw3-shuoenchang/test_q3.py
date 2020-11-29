import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.DANN import DANN
from src.dataset import DigitDataset
from src.utils import draw_features


def testing(dataset, model, device, output_path):
    model.eval()
    with open(output_path, 'w') as f:
        f.write('image_name, label\n')
    with torch.no_grad():
        corrects = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            label, domain = model(image, 0)

            _, predicts = torch.max(label, 1)

            for image_name, label in zip(name, predicts):
                with open(output_path, 'a') as f:
                    f.write(f'{image_name},{label}\n')


def feature(sourceset, targetset, model, device):
    model.eval()
    features = np.empty((0, 1024))
    targets = np.empty((0,), dtype=np.int8)
    domains = np.empty((0,), dtype=np.int8)
    with torch.no_grad():
        for data in sourceset:
            image = data['image'].to(device)
            target = data['label'].to(device)
            domain = data['domain'].to(device)
            latent = model(image, 0, True)
            features = np.concatenate(
                (features, latent.cpu().numpy()), axis=0)
            targets = np.concatenate((targets, target), axis=0)
            domains = np.concatenate((domains, domain), axis=0)
        for data in targetset:
            image = data['image'].to(device)
            target = data['label'].to(device)
            domain = data['domain'].to(device)
            latent = model(image, 0, True)
            features = np.concatenate(
                (features, latent.cpu().numpy()), axis=0)
            targets = np.concatenate((targets, target), axis=0)
            domains = np.concatenate((domains, domain), axis=0)
    print(features.shape, targets.shape, domains.shape)
    draw_features(features, targets, 10, domains)


def main(args):

    if args.dataset == 'svhn' and args.model_name == '':
        args.model_name = 'mnistm-svhn'
    elif args.dataset == 'usps' and args.model_name == '':
        args.model_name = 'svhn-usps'
    elif args.dataset == "mnistm" and args.model_name == '':
        args.model_name = 'usps-mnistm'

    model = DANN().to(args.device)
    model.load_state_dict(torch.load(
        'weights/q3/{}.pth'.format(args.model_name), map_location=args.device))
    
    # print(args.dataset, args.model_name)
    if args.feature:
        sourceset = DataLoader(dataset=DigitDataset(args.input_path, subset=args.source, mode='feature', domain='source'),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        targetset = DataLoader(dataset=DigitDataset(args.input_path, subset=args.dataset, mode='feature', domain='target'),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        feature(sourceset, targetset, model, args.device)
    else:
        testset = DataLoader(dataset=DigitDataset(args.input_path, subset=args.dataset, mode='test', domain='source'),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        testing(testset, model, args.device, args.output_path)  # noqa


if __name__ == '__main__':
    torch.manual_seed(422)
    parser = argparse.ArgumentParser(description='DLCV hw3-3 Testing Script')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--feature', action='store_true', default=False)
    parser.add_argument('--source', type=str, default='')
    args = parser.parse_args()
    main(args)
