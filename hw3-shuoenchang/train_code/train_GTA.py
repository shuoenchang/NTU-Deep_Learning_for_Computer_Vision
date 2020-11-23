import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from models.GTA import Generator, Discriminator, Feature, Classifier
from src.dataset import DigitDataset
from src.utils import class_to_onehot

wandb.init(project="dlcv_3-4")


def train(source, target, model, optimzer, device, opt):
    for m in model.values():
        m.to(device)
        m.train()

    CE = nn.CrossEntropyLoss()
    BCE = nn.BCEWithLogitsLoss()
    total_loss = []
    smaller_step = len(source) if len(source) < len(target) else len(target)  # noqa
    source = iter(source)
    target = iter(target)

    for step in range(smaller_step):

        source_data = next(source)
        source_image = source_data['image'].to(device)
        source_label = source_data['label'].to(device)
        source_label_onehot = class_to_onehot(source_label, device)

        target_data = next(target)
        target_image = target_data['image'].to(device)
        target_label = torch.zeros_like(source_label).to(device)+10
        target_label_onehot = class_to_onehot(target_label, device)

        zeros_label = torch.zeros(len(source_image), 1).to(device)
        ones_label = torch.ones(len(source_image), 1).to(device)

        # Update D
        optimzer['D'].zero_grad()

        source_feature = model['F'](source_image)
        target_feature = model['F'](target_image)

        latent = torch.randn(len(source_feature), opt.latent_dim).to(device)
        source_z = torch.cat((source_feature, source_label_onehot, latent), dim=1)  # noqa
        source_fake_image = model['G'](source_z)
        source_real, source_image_predict = model['D'](source_image)
        source_fake, _ = model['D'](source_fake_image)
        loss_src_data = BCE(source_real, ones_label) + BCE(source_fake, zeros_label)  # noqa
        loss_src_cls = CE(source_image_predict, source_label)

        latent = torch.randn(len(target_feature), opt.latent_dim).to(device)
        target_z = torch.cat((target_feature, target_label_onehot, latent), dim=1)  # noqa
        target_fake_image = model['G'](target_z)
        target_fake, _ = model['D'](target_fake_image)
        loss_tgt_adv = BCE(target_fake, zeros_label)

        loss_D = loss_src_data+loss_src_cls+loss_tgt_adv
        loss_D.backward(retain_graph=True)
        optimzer['D'].step()
        wandb.log({"loss_D": loss_D}, commit=False)

        # Update G
        optimzer['G'].zero_grad()
        source_fake, source_fake_predict = model['D'](source_fake_image)
        loss_src_data = BCE(source_fake, ones_label)
        loss_src_fake_predict = CE(source_fake_predict, source_label)  # noqa
        loss_G = loss_src_data+loss_src_fake_predict
        loss_G.backward(retain_graph=True)
        optimzer['G'].step()
        wandb.log({"loss_G": loss_G}, commit=False)

        # Update C
        optimzer['C'].zero_grad()
        source_predict = model['C'](source_feature)
        loss_C = CE(source_predict, source_label)
        loss_C.backward(retain_graph=True)
        optimzer['C'].step()
        wandb.log({"loss_C": loss_C}, commit=False)

        # Update F
        optimzer['F'].zero_grad()
        source_predict = model['C'](source_feature)  # from C loss
        loss_from_C = CE(source_predict, source_label)

        _, source_fake_predict = model['D'](source_fake_image)  # from D loss
        loss_D_src_cls = CE(source_fake_predict, source_label)*opt.adv_weight

        target_fake, _ = model['D'](target_fake_image)
        loss_D_tgt_adv = BCE(target_fake, ones_label)*(opt.alpha*opt.adv_weight)

        loss_from_D = loss_D_src_cls + loss_D_tgt_adv

        loss_F = loss_from_C+loss_from_D
        loss_F.backward()
        optimzer['F'].step()
        wandb.log({"loss_F": loss_F}, commit=False)

        loss = loss_D+loss_G+loss_C+loss_F
        assert (np.isnan(loss.item()) == False)

        wandb.log({"loss": loss})
        total_loss.append(loss.item())
        print('step {}: D {:.3f}, G {:.3f}, C {:.3f}, F {:.3f}'.format(step, loss_D, loss_G, loss_C, loss_F), end='\r')
    print('\n')
    return np.mean(total_loss)


def validation(dataset, model, device):
    for m in model.values():
        m.to(device)
        m.eval()
    with torch.no_grad():
        corrects = torch.zeros(1).to(device)
        for data in dataset:
            image = data['image'].to(device)
            gt_label = data['label'].to(device)
            feature = model['F'](image)
            predict = model['C'](feature)
            _, preds = torch.max(predict, 1)
            corrects += (preds == gt_label).sum()

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
        'G': Generator(args.latent_dim, args.feature_dim),
        'D': Discriminator(args.feature_dim),
        'C': Classifier(args.feature_dim)
    }

    optimzer = {
        'F': optim.RMSprop(model['F'].parameters(), lr=args.learning_rate),
        'G': optim.RMSprop(model['G'].parameters(), lr=args.learning_rate),
        'D': optim.RMSprop(model['D'].parameters(), lr=args.learning_rate),
        'C': optim.RMSprop(model['C'].parameters(), lr=args.learning_rate)
    }

    device = args.device
    for epoch in range(args.max_epoch):

        print('\nepoch: {}'.format(epoch))

        if TargetTrain:
            loss = train(SourceTrain, TargetTrain, model,
                         optimzer, device, args)
        else:
            loss = train_without_target(SourceTrain, model,
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
            torch.save(model['C'].state_dict(),
                       '{}/{}-{}_C.pth'.format(args.save_folder, args.source, args.target))
            print('Best epoch: {}'.format(epoch))
            with open('{}/{}-{}.txt'.format(args.save_folder, args.source, args.target), 'w') as f:
                f.write(str(acc))
                f.write('\n')
                f.write(str(args))


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
    parser.add_argument('--feature_dim', default=128, type=int,
                        help='feature dimension after F network')
    parser.add_argument('--latent_dim', default=128, type=int,
                        help='latent dimension for G network')
    parser.add_argument('--adv_weight', type=float,
                        default=0.1, help='weight for adv loss')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='multiplicative factor for target adv. loss')
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()
    wandb.config.update(args)
    main(args)
