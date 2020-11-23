import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from models.MCD import Feature, Classifier
from src.dataset import DigitDataset

wandb.init(project="dlcv_3-4_MCD")


def train(source, target, model, optimzer, device, opt):
    for m in model.values():
        m.to(device)
        m.train()

    CE = nn.CrossEntropyLoss()
    total_loss = []
    smaller_step = len(source) if len(source) < len(target) else len(target)  # noqa
    source = iter(source)
    target = iter(target)

    for step in range(smaller_step):

        source_data = next(source)
        source_image = source_data['image'].to(device)
        source_label = source_data['label'].to(device)

        target_data = next(target)
        target_image = target_data['image'].to(device)

        # Step 1
        optimzer['F'].zero_grad()
        optimzer['C_1'].zero_grad()
        optimzer['C_2'].zero_grad()

        feature_s = model['F'](source_image)
        pred_s_1 = model['C_1'](feature_s)
        pred_s_2 = model['C_2'](feature_s)
        loss_s = CE(pred_s_1, source_label)+CE(pred_s_2, source_label)

        loss_s.backward()
        optimzer['F'].step()
        optimzer['C_1'].step()
        optimzer['C_2'].step()
        wandb.log({"loss_s": loss_s}, commit=False)

        # Step 2
        optimzer['F'].zero_grad()
        optimzer['C_1'].zero_grad()
        optimzer['C_2'].zero_grad()

        feature_s = model['F'](source_image)
        pred_s_1 = model['C_1'](feature_s)
        pred_s_2 = model['C_2'](feature_s)

        feature_t = model['F'](target_image)
        pred_t_1 = model['C_1'](feature_t)
        pred_t_2 = model['C_2'](feature_t)

        loss_src = CE(pred_s_1, source_label)+CE(pred_s_2, source_label)
        loss_discrepancy = torch.mean(
            torch.abs(F.softmax(pred_t_1, dim=1) - F.softmax(pred_t_2, dim=1)))

        loss_d = loss_src - loss_discrepancy

        loss_d.backward()
        optimzer['C_1'].step()
        optimzer['C_2'].step()
        wandb.log({"loss_d": loss_d}, commit=False)

        # Step 3
        for i in range(opt.num_k):
            optimzer['F'].zero_grad()
            optimzer['C_1'].zero_grad()
            optimzer['C_2'].zero_grad()

            feature_t = model['F'](target_image)
            pred_t_1 = model['C_1'](feature_t)
            pred_t_2 = model['C_2'](feature_t)

            loss_discrepancy = torch.mean(
                torch.abs(F.softmax(pred_t_1, dim=1) - F.softmax(pred_t_2, dim=1)))
            loss_discrepancy.backward()
            optimzer['F'].step()

        wandb.log({"loss_t": loss_discrepancy})

        print('step {}: s {:.3f}, d {:.3f}, t {:.3f}'.format(
            step, loss_s, loss_d, loss_discrepancy), end='\r')
    print('\n')


def validation(dataset, model, device):
    for m in model.values():
        m.to(device)
        m.eval()
    with torch.no_grad():
        correct_1 = torch.zeros(1).to(device)
        correct_2 = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            gt_label = data['label'].to(device)
            feature = model['F'](image)
            predict = model['C_1'](feature)
            _, preds = torch.max(predict, 1)
            correct_1 += (preds == gt_label).sum()
        for data in dataset:
            image = data['image'].to(device)
            gt_label = data['label'].to(device)
            feature = model['F'](image)
            predict = model['C_2'](feature)
            _, preds = torch.max(predict, 1)
            correct_2 += (preds == gt_label).sum()
    corrects = correct_1 if correct_1.item() > correct_2.item() else correct_2
    acc = corrects.item() / len(dataset.dataset)

    print('acc {:.3f}'.format(acc))
    return acc


def main(args):

    SourceTrain = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.source, mode='train', domain='source', normalize=True),
                             batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    SourceVal = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.source, mode='val', domain='source', normalize=True),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.target:
        TargetTrain = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.target, mode='train', domain='target', normalize=True),
                                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        TargetVal = DataLoader(dataset=DigitDataset('hw3_data/digits', subset=args.target, mode='val', domain='target', normalize=True),
                               batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        TargetTrain = SourceTrain
        TargetVal = SourceVal

    model = {
        'F': Feature(args.feature_dim),
        'C_1': Classifier(args.feature_dim),
        'C_2': Classifier(args.feature_dim)
    }

    optimzer = {
        'F': optim.RMSprop(model['F'].parameters(), lr=args.learning_rate),
        'C_1': optim.RMSprop(model['C_1'].parameters(), lr=args.learning_rate),
        'C_2': optim.RMSprop(model['C_2'].parameters(), lr=args.learning_rate)
    }

    device = args.device
    for epoch in range(args.max_epoch):

        print('\nepoch: {}'.format(epoch))

        if TargetTrain:
            train(SourceTrain, TargetTrain, model,
                  optimzer, device, args)
        else:
            train_without_target(SourceTrain, model,
                                 optimzer, device)

        print('val_source')
        acc = validation(SourceVal, model, device)  # noqa
        if TargetVal:
            print('val_target')
            acc = validation(TargetVal, model, device)  # noqa
        wandb.log({'acc': acc})
        with open('{}/{}-{}.txt'.format(args.save_folder, args.source, args.target), 'r') as f:
            max_acc = float(f.readline())
        if acc > max_acc:
            torch.save(model['F'].state_dict(),
                       '{}/{}-{}_F.pth'.format(args.save_folder, args.source, args.target))
            torch.save(model['C_1'].state_dict(),
                       '{}/{}-{}_C1.pth'.format(args.save_folder, args.source, args.target))
            torch.save(model['C_2'].state_dict(),
                       '{}/{}-{}_C2.pth'.format(args.save_folder, args.source, args.target))
            print('Best epoch: {}'.format(epoch))
            with open('{}/{}-{}.txt'.format(args.save_folder, args.source, args.target), 'w') as f:
                f.write(str(acc))
                f.write('\n')
                f.write(str(args))
                f.write('\n')


if __name__ == '__main__':
    torch.manual_seed(422)
    parser = argparse.ArgumentParser(description='DLCV hw3-3 Training Script')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q4',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate for training')
    parser.add_argument('--max_epoch', default=30, type=int,
                        help='epoch for training')
    parser.add_argument('--feature_dim', default=256, type=int,
                        help='feature dimension after F network')
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--num_k', type=int, default=3,
                        help='hyper paremeter for generator update')
    args = parser.parse_args()
    wandb.config.update(args)
    main(args)
