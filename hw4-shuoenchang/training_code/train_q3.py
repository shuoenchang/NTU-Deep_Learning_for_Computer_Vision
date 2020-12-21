import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from numpy.core.defchararray import mod
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.prototypical import Convnet4, Hallucination, Discriminator
from src.dataset import GeneratorSampler, MiniDataset
from src.utils import Distance, calculate_acc, set_seed, worker_init_fn


def train(data_loader, model, hall, disc, criterion, optimzer, args):
    model.train()
    total_loss = []
    total_loss_d = []
    total_loss_g = []
    total_acc = []
    for step, data in enumerate(data_loader):
        image, _ = data
        support = {'image': image[:args.n_way*args.n_shot].to(args.device),
                   'label': torch.LongTensor([i//args.n_shot for i in range(args.n_way*args.n_shot)])}
        query = {'image': image[args.n_way*args.n_shot:].to(args.device),
                 'label': torch.LongTensor([i//args.n_query for i in range(args.n_way*args.n_query)]).to(args.device)}
        proto = model(support['image']).view(args.n_way, args.n_shot, -1)

        # Train Model
        new_proto = torch.empty(
            [args.n_way, args.n_shot+args.n_aug, args.dim]).to(args.device)
        for c in range(args.n_way):
            fake = hall(proto[c][0])
            new_proto[c] = torch.cat([proto[c], fake], dim=0)
        new_proto = new_proto.mean(1)

        feature = model(query['image'])
        distance = Distance(args)
        logits = distance(new_proto, feature)
        loss = criterion(logits, query['label'])
        total_loss.append(loss.item())
        accuarcy = calculate_acc(logits, query['label'])
        total_acc.append(accuarcy)

        optimzer['m'].zero_grad()
        optimzer['h'].zero_grad()
        loss.backward(retain_graph=True)
        optimzer['m'].step()
        optimzer['h'].step()

        # Train Discriminator
        loss_real = 0
        loss_fake = 0
        for c in range(args.n_way):
            fake = hall(proto[c][0])
            loss_real += torch.mean(disc(proto[c]))
            loss_fake += torch.mean(disc(fake))
        loss_d = -loss_real+loss_fake
        total_loss_d.append(loss_d.item())

        optimzer['d'].zero_grad()
        loss_d.backward(retain_graph=True)
        optimzer['d'].step()
        disc.weight_cliping()

        # Train Generator
        if step % 5 == 0:
            loss_g = 0
            for c in range(args.n_way):
                fake = hall(proto[c][0])
                loss_g += -torch.mean(disc(fake))
            total_loss_g.append(loss_g.item())

            optimzer['h'].zero_grad()
            loss_g.backward()
            optimzer['h'].step()

        if step % 50 == 0:
            print('step {}: loss_D {:.3f}, loss_G {:.3f}, loss {:.3f}, acc {:.3f}'.format(
                step, np.mean(total_loss_d), np.mean(total_loss_g), np.mean(total_loss), np.mean(total_acc)), end='\r')
    print('Training: loss_D {:.3f}, loss_G {:.3f}, loss {:.3f}, acc {:.3f}'.format(
        np.mean(total_loss_d), np.mean(total_loss_g), np.mean(total_loss), np.mean(total_acc)), end='\n')


def val(data_loader, model, hall, criterion, args):
    with torch.no_grad():
        model.eval()
        total_loss = []
        total_acc = []
        for _, data in enumerate(data_loader):
            image, _ = data
            support = {'image': image[:args.n_way*args.n_shot].to(args.device),
                       'label': torch.LongTensor([i//args.n_shot for i in range(args.n_way*args.n_shot)])}
            query = {'image': image[args.n_way*args.n_shot:].to(args.device),
                     'label': torch.LongTensor([i//args.n_query for i in range(args.n_way*args.n_query)]).to(args.device)}
            proto = model(support['image']).view(args.n_way, args.n_shot, -1)
            new_proto = torch.empty(
                [args.n_way, args.n_shot+args.n_aug, args.dim]).to(args.device)
            for c in range(args.n_way):
                fake = hall(proto[c][0])
                new_proto[c] = torch.cat([proto[c], fake], dim=0)
            new_proto = new_proto.mean(1)

            feature = model(query['image'])
            distance = Distance(args)
            logits = distance(new_proto, feature)
            loss = criterion(logits, query['label'])
            total_loss.append(loss.item())
            accuracy = calculate_acc(logits, query['label'])
            total_acc.append(accuracy)

    print('Validation: loss {:.3f}, acc {:.3f}\n'.format(
        np.mean(total_loss), np.mean(total_acc)))
    return np.mean(total_loss), np.mean(total_acc)


def main(args):

    train_dataset = MiniDataset(args.train_csv, args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_sampler=GeneratorSampler(
        args.train_csv, args.batch_size, args.n_way, args.n_shot, args.n_query),
        num_workers=3, worker_init_fn=worker_init_fn)
    val_dataset = MiniDataset(args.val_csv, args.val_data_dir)
    val_loader = DataLoader(val_dataset, batch_sampler=GeneratorSampler(
        args.val_csv, 600, args.n_way, args.n_shot, args.n_query),
        num_workers=3, worker_init_fn=worker_init_fn)

    model = Convnet4(out_channels=args.dim).to(args.device)
    hall = Hallucination(args.dim, args.n_aug).to(args.device)
    disc = Discriminator(args.dim, args.weight_clip).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = {'m': optim.Adam(model.parameters(), lr=args.learning_rate),
                 'h': optim.Adam(hall.parameters(), lr=args.learning_rate),
                 'd': optim.Adam(disc.parameters(), lr=args.learning_rate)}
    lr_scheduler = {'m': StepLR(optimizer['m'], step_size=5, gamma=0.5),
                    'h': StepLR(optimizer['h'], step_size=5, gamma=0.5),
                    'd': StepLR(optimizer['d'], step_size=5, gamma=0.5)}
    min_accuracy = 0.46

    for epoch in range(args.max_epoch):
        print('epoch: ', epoch)
        train(train_loader, model, hall, disc, criterion, optimizer, args)
        loss, accuracy = val(val_loader, model, hall, criterion, args)
        lr_scheduler['m'].step()
        lr_scheduler['h'].step()
        lr_scheduler['d'].step()
        if accuracy > min_accuracy:
            torch.save(model.state_dict(),
                       '{}/epoch_{:02d}-{:.3f}_m.pth'.format(args.save_folder, epoch, accuracy))
            torch.save(hall.state_dict(),
                       '{}/epoch_{:02d}-{:.3f}_h.pth'.format(args.save_folder, epoch, accuracy))
            min_accuracy = accuracy
            print('\rBest Epoch: {}\n'.format(epoch))


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_way', default=5, type=int,
                        help='N_way (default: 5)')
    parser.add_argument('--n_shot', default=1, type=int,
                        help='N_shot (default: 1)')
    parser.add_argument('--n_query', default=15, type=int,
                        help='N_query (default: 15)')
    parser.add_argument('--n_aug', default=10, type=int,
                        help='M use in Hallucination (default: 10)')
    parser.add_argument('--dim', default=100, type=int,
                        help='dim for feature size (default: 100)')
    parser.add_argument('--learning_rate', default=5e-4,
                        type=float, help='learning rate for training')
    parser.add_argument('--weight_clip', default=0.001,
                        type=float, help='learning rate for training')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help='Batch size for training')
    parser.add_argument('--max_epoch', default=50, type=int,
                        help='Epoch for training')
    parser.add_argument('--distance_type', default='euclidean', type=str,
                        help='Distance type for training')
    parser.add_argument('--save_folder', default='weights/q3',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--train_csv', default='hw4_data/train.csv', type=str,
                        help="Training images csv file")
    parser.add_argument('--train_data_dir', default='hw4_data/train', type=str,
                        help="Training images directory")
    parser.add_argument('--val_csv', default='hw4_data/val.csv', type=str,
                        help="Val images csv file")
    parser.add_argument('--val_data_dir', default='hw4_data/val', type=str,
                        help="Val images directory")
    parser.add_argument('--seed', default=877, type=int,
                        help="random seed")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    main(args)
