import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.prototypical import Convnet4
from src.dataset import GeneratorSampler, MiniDataset
from src.utils import Distance, calculate_acc, set_seed, worker_init_fn


def train(data_loader, model, distance, criterion, optimzer, args, optimzer_D=None):
    model.train()
    total_loss = []
    total_acc = []
    for step, data in enumerate(data_loader):
        image, _ = data
        support = {'image': image[:args.n_way*args.n_shot].to(args.device),
                   'label': torch.LongTensor([i//args.n_shot for i in range(args.n_way*args.n_shot)])}
        query = {'image': image[args.n_way*args.n_shot:].to(args.device),
                 'label': torch.LongTensor([i//args.n_query for i in range(args.n_way*args.n_query)]).to(args.device)}
        proto = model(support['image']).view(args.n_way, args.n_shot, -1)
        proto = proto.mean(1)

        feature = model(query['image'])
        logits = distance(proto, feature)
        loss = criterion(logits, query['label'])
        total_loss.append(loss.item())
        accuarcy = calculate_acc(logits, query['label'])
        total_acc.append(accuarcy)

        optimzer.zero_grad()
        if optimzer_D:
            optimzer_D.zero_grad()
        loss.backward()
        optimzer.step()
        if optimzer_D:
            optimzer_D.step()
        if step % 50 == 0:
            print('step {}: loss {:.3f}, acc {:.3f}'.format(
                step, np.mean(total_loss), np.mean(total_acc)), end='\r')
    print('Training: loss {:.3f}, acc {:.3f}'.format(
        np.mean(total_loss), np.mean(total_acc)), end='\n')


def val(data_loader, model, distance, criterion, args):
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
            proto = proto.mean(1)

            feature = model(query['image'])
            logits = distance(proto, feature)
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
    distance = Distance(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.distance_type == 'param':
        optimzer_D = optim.Adam(
            distance.param_model.parameters(), lr=args.learning_rate)
    else:
        optimzer_D = None
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.75)
    min_accuracy = 0.44

    for epoch in range(args.max_epoch):
        print('epoch: ', epoch)
        train(train_loader, model, distance,
              criterion, optimizer, args, optimzer_D)
        loss, accuracy = val(val_loader, model, distance, criterion, args)
        lr_scheduler.step()
        if accuracy > min_accuracy:
            torch.save(model.state_dict(),
                       '{}/epoch_{:02d}-{:.3f}.pth'.format(args.save_folder, epoch, accuracy))
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
    parser.add_argument('--dim', default=100, type=int,
                        help='dim for feature size (default: 100)')
    parser.add_argument('--learning_rate', default=1e-4,
                        type=float, help='learning rate for training')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help='Batch size for training')
    parser.add_argument('--max_epoch', default=30, type=int,
                        help='Epoch for training')
    parser.add_argument('--distance_type', default='euclidean', type=str,
                        help='Distance type for training')
    parser.add_argument('--save_folder', default='weights/q1',
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
