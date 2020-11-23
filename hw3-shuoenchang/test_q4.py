import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import models.MCD as MCD
import models.GTA as GTA
from src.dataset import DigitDataset


def testing(dataset, F, C, device):
    F.eval()
    C.eval()
    total_label = []
    with torch.no_grad():
        corrects = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            gt_label = data['label'].to(device)
            feature = F(image)
            label = C(feature)
            _, preds = torch.max(label, 1)
            corrects += (preds == gt_label).sum()
    acc = corrects.item() / len(dataset.dataset)

    print('acc {:.3f}'.format(acc))
    return acc


def main(args):

    testset = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.dataset, mode='test', domain='source', normalize=True),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # if args.dataset=='svhn':
    F = MCD.Feature(256).to(args.device)
    C = MCD.Classifier(256).to(args.device)
    # elif args.dataset=='usps':
    #     F = GTA.Feature(64).to(args.device)
    #     C = GTA.Classifier(64).to(args.device)
    # elif args.dataset=='mnistm':
    #     F = GTA.Feature(384).to(args.device)
    #     C = GTA.Classifier(384).to(args.device)
    F.load_state_dict(torch.load(
        'weights/q4/{}_F.pth'.format(args.model_name), map_location=args.device))
    C.load_state_dict(torch.load(
        'weights/q4/{}_C.pth'.format(args.model_name), map_location=args.device))
    print(args.dataset, args.model_name)
    acc = testing(testset, F, C, args.device)  # noqa


if __name__ == '__main__':
    torch.manual_seed(422)
    parser = argparse.ArgumentParser(description='DLCV hw3-4 Testing Script')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    main(args)
