import argparse
import os

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from models.CDCN import CDCNpp
from src.dataset import CockRoach_Dataset
from src.loss import Loss
from src.optimizer import Optimizer
from src.utils import set_seed, worker_init_fn


def train(data_loader, model, criterion, optimizer, args):
    model.train()
    total_loss = []
    optimizer['m'].zero_grad()
    for step, data in enumerate(data_loader):
        video, binary, label = data['video'].to(args.device), data['binary'].to(
            args.device), data['label'].to(args.device)
        bs, t, c, w, h = video.shape
        video = video.view(-1, c, w, h)
        binary = binary.view(-1, 32, 32)
        map_x, x_concat, attention1, attention2, attention3, x_input = model(video) # noqa
        absolute_loss = criterion['absolute'](map_x, binary)
        contrastive_loss = criterion['contrastive'](map_x, binary)

        loss = absolute_loss+contrastive_loss
        total_loss.append(loss.item())
        loss = loss/args.accumulation_steps
        loss.backward()

        if (step+1) % args.accumulation_steps == 0:
            optimizer['m'].step()
            optimizer['m'].zero_grad()
        if step % 5 == 0: 
            print('step {}: loss {:.3f}'.format(step, np.mean(total_loss)), end='\r')
            
    print('Training:   loss {:.3f}'.format(
        np.mean(total_loss)), end='\n')


def val(data_loader, model, criterion, args):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            video, binary, label = data['video'].to(args.device), data['binary'].to(
                args.device), data['label'].to(args.device)
            bs, t, c, w, h = video.shape
            video = video.view(-1, c, w, h)
            map_x, x_concat, attention1, attention2, attention3, x_input = model(video) # noqa
            map_x = map_x.view(bs, t, 32, 32)
            for video_id in range(bs):
                map_score = 0.0
                for frame_id in range(t):
                    score_norm = torch.sum(map_x[video_id, frame_id, :, :]) \
                        / torch.sum(binary[video_id, frame_id, :, :])
                    # print(score_norm)
                    map_score += score_norm
                map_score = map_score/t
                print(label[video_id])
                print(map_score)


def main(args):

    train_dataset = CockRoach_Dataset(args.train_data_dir, 'train')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True)
    val_dataset = CockRoach_Dataset(args.val_data_dir, 'val')
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=3)

    model = CDCNpp().to(args.device)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('weights/epoch_04.pth'))
    criterion = {'absolute': Loss('MSE', args),
                 'contrastive': Loss('CD', args)}
    optimizer = {'m': Optimizer(
        model.parameters(), 'Adam', lr=args.learning_rate)}
    lr_scheduler = {'m': StepLR(optimizer['m'], step_size=args.step_size, gamma=args.gamma)}
    min_loss = 1

    for epoch in range(args.max_epoch):
        print('epoch: ', epoch)
        train(train_loader, model, criterion, optimizer, args)
        # val(val_loader, model, criterion, args)
        lr_scheduler['m'].step()
        # if loss < min_loss:
        torch.save(model.state_dict(),
                   '{}/epoch_{:02d}.pth'.format(args.save_folder, epoch))
        # min_loss = loss
        # print('\rBest Epoch: {}\n'.format(epoch))


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--learning_rate', default=5e-4,
                        type=float, help='learning rate for training')
    parser.add_argument('--step_size', default=10,
                        type=int, help='step_size for scheduler')
    parser.add_argument('--gamma', default=0.7,
                        type=float, help='gamma for scheduler')
    parser.add_argument('--accumulation_steps', default=5,
                        type=int, help='accumulation steps for gradients')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--max_epoch', default=100, type=int,
                        help='Epoch for training')
    parser.add_argument('--save_folder', default='weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--train_data_dir', default='data/oulu/train', type=str,
                        help="Training images directory")
    parser.add_argument('--val_data_dir', default='data/oulu/val', type=str,
                        help="Val images directory")
    parser.add_argument('--seed', default=877, type=int,
                        help="random seed")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    main(args)
