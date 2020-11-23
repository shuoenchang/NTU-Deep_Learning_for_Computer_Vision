import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import models.MCD as MCD
from src.dataset import DigitDataset


def testing(dataset, F, C, device, output_path):
    F.eval()
    C.eval()
    with open(output_path, 'w') as f:
        f.write('image_name, label\n')
    with torch.no_grad():
        corrects = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            feature = F(image)
            label = C(feature)
            _, predicts = torch.max(label, 1)

            for image_name, label in zip(name, predicts):
                with open(output_path, 'a') as f:
                    f.write(f'{image_name}, {label}\n')


def main(args):
    testset = DataLoader(dataset=DigitDataset(args.input_path, subset=args.dataset, mode='test', domain='source', normalize=True),
                         batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.dataset == 'svhn' and args.model_name == '':
        args.model_name = 'mnistm-svhn'
    elif args.dataset == 'usps' and args.model_name == '':
        args.model_name = 'svhn-usps'
    elif args.dataset == "mnistm" and args.model_name == '':
        args.model_name = 'usps-mnistm'

    F = MCD.Feature(256).to(args.device)
    C = MCD.Classifier(256).to(args.device)
    F.load_state_dict(torch.load(
        'weights/q4/{}_F.pth'.format(args.model_name), map_location=args.device))
    C.load_state_dict(torch.load(
        'weights/q4/{}_C.pth'.format(args.model_name), map_location=args.device))

    print(args.dataset, args.model_name)
    testing(testset, F, C, args.device, args.output_path)  # noqa


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
    args = parser.parse_args()
    main(args)
