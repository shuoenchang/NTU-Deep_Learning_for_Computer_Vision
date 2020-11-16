import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from models.DANN import DANN
from src.dataset import DigitDataset
from src.loss import DANN_loss
from src.utils import get_lambda

wandb.init(project="dlcv_3-3")


def train(source, target, model, optimzer, device, criterion, lambda_domain):
    model.to(device)
    model.train()
    total_loss = []
    smaller_step = len(source) if len(
        source) < len(target) else len(target)
    source = iter(source)
    target = iter(target)
    for step in range(smaller_step):
        source_data = next(source)
        optimzer.zero_grad()
        image = source_data['image'].to(device)
        gt_domain = source_data['domain'].to(device)
        gt_label = source_data['label'].to(device)
        label, domain = model(image, lambda_domain)
        loss_s_domain, loss_s_label = criterion(
            domain, gt_domain, label, gt_label)

        target_data = next(target)
        image = target_data['image'].to(device)
        gt_domain = target_data['domain'].to(device)
        label, domain = model(image, lambda_domain)
        loss_t_domain = criterion(domain, gt_domain)

        loss = loss_s_domain+loss_s_label+loss_t_domain
        assert (np.isnan(loss.item()) == False)

        wandb.log({"lambda_domain": lambda_domain}, commit=False)
        wandb.log({"loss_s_domain": loss_s_domain}, commit=False)
        wandb.log({"loss_s_label": loss_s_label}, commit=False)
        wandb.log({"loss_t_domain": loss_t_domain}, commit=False)
        wandb.log({"loss": loss})

        total_loss.append(loss.item())
        loss.backward()
        optimzer.step()
        if step % 50 == 0:
            print('step {}: s_d {:.3f}, s_l {:.3f}, t_d {:.3f}, loss {:.3f}'.format(
                step, loss_s_domain, loss_s_label, loss_t_domain, loss), end='\r')
    print('train:       s_d {:.3f}, s_l {:.3f}, t_d {:.3f}, loss {:.3f}'.format(
        loss_s_domain, loss_s_label, loss_t_domain, np.mean(total_loss)))
    return np.mean(total_loss)


def train_without_target(source, model, optimzer, device, criterion, lambda_domain):
    model.to(device)
    model.train()
    total_loss = []
    for step, source_data in enumerate(source):
        optimzer.zero_grad()
        image = source_data['image'].to(device)
        gt_domain = source_data['domain'].to(device)
        gt_label = source_data['label'].to(device)
        label, domain = model(image, lambda_domain)
        loss_s_domain, loss_s_label = criterion(
            domain, gt_domain, label, gt_label)

        loss = loss_s_domain+loss_s_label
        assert (np.isnan(loss.item()) == False)

        wandb.log({"lambda_domain": lambda_domain}, commit=False)
        wandb.log({"loss_s_domain": loss_s_domain}, commit=False)
        wandb.log({"loss_s_label": loss_s_label}, commit=False)
        wandb.log({"loss": loss})

        total_loss.append(loss.item())
        loss.backward()
        optimzer.step()
        if step % 50 == 0:
            print('step {}: s_d {:.3f}, s_l {:.3f}, loss {:.3f}'.format(
                step, loss_s_domain, loss_s_label, loss), end='\r')
    print('train:       s_d {:.3f}, s_l {:.3f}, loss {:.3f}'.format(
        loss_s_domain, loss_s_label, np.mean(total_loss)))
    return np.mean(total_loss)


def validation(dataset, model, device, criterion, lambda_domain):
    model.to(device)
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

    SourceTrain = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.source, mode='train', domain='source'),
                             batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    SourceVal = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.source, mode='val', domain='source'),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.target:
        TargetTrain = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.target, mode='train', domain='target'),
                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        TargetVal = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.target, mode='val', domain='target'),
                               batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        TargetTrain = None
        TargetVal = None

    model = DANN()

    criterion = DANN_loss()
    optimzer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = args.device
    min_loss = 4
    for epoch in range(args.max_epoch):

        print('\nepoch: {}'.format(epoch))
        lambda_domain = get_lambda(epoch, args.max_epoch)
        if TargetTrain:
            loss = train(SourceTrain, TargetTrain, model,
                         optimzer, device, criterion, lambda_domain)
        else:
            loss = train_without_target(SourceTrain, model,
                                        optimzer, device, criterion, lambda_domain)
        print('val_source')
        loss, acc = validation(SourceVal, model, device, criterion, lambda_domain)  # noqa
        if TargetVal:
            print('val_target')
            loss, acc = validation(TargetVal, model, device, criterion, lambda_domain)  # noqa
        wandb.log({"val_loss": loss, 'acc': acc})
        if loss < min_loss and acc>0.4:
            torch.save(model.state_dict(),
                       '{}/{}-{}.pth'.format(args.save_folder, args.source, args.target))
            min_loss = loss
            print('Best epoch: {}'.format(epoch))


if __name__ == '__main__':
    torch.manual_seed(422)
    parser = argparse.ArgumentParser(description='DLCV hw3-3 Training Script')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q3',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate for training')
    parser.add_argument('--max_epoch', default=20, type=int,
                        help='epoch for training')
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()
    wandb.config.update(args)
    main(args)
