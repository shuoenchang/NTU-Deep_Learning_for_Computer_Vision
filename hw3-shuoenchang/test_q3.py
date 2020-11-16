import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.DANN import DANN
from src.dataset import DigitDataset
from src.loss import DANN_loss
from src.utils import get_lambda


def testing(dataset, model, device, criterion, lambda_domain):
    model.eval()
    total_loss = []
    total_domain = []
    total_label = []
    with torch.no_grad():
        corrects = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            gt_domain = data['domain'].to(device)
            gt_label = data['label'].to(device)
            label, domain = model(image, lambda_domain)

            _, preds = torch.max(label, 1)
            corrects += (preds == gt_label).sum()

            loss_s_domain, loss_s_label = criterion(
                domain, gt_domain, label, gt_label)
            loss = loss_s_domain+loss_s_label
            total_loss.append(loss.item())
            total_domain.append(loss_s_domain.item())
            total_label.append(loss_s_label.item())
    acc = corrects.item() / len(dataset.dataset)

    print('   acc {:.3f}, d {:.3f}, l {:.3f}, loss {:.3f}'.format(
        acc, np.mean(total_domain), np.mean(total_label), np.mean(total_loss)))
    return np.mean(total_label), acc


def main(args):

    SourceVal = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.dataset, mode='val', domain='source'),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DANN().to(args.device)
    model.load_state_dict(torch.load(
        'weights/q3/{}.pth'.format(args.model_name), map_location=args.device))
    criterion = DANN_loss()

    loss, acc = testing(SourceVal, model, args.device, criterion, 0)  # noqa


if __name__ == '__main__':
    torch.manual_seed(422)
    parser = argparse.ArgumentParser(description='DLCV hw3-1 Training Script')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    main(args)
