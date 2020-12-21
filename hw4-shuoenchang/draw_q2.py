import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.prototypical import Convnet4, Hallucination
from src.dataset import MiniDataset, GeneratorSampler
from src.utils import Distance, worker_init_fn, set_seed, draw_features


def predict(args, model, data_loader, distance, hall):

    with torch.no_grad():
        model.eval()
        hall.eval()
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            # split data into support and query data
            support_input = data[:args.n_way * args.n_shot].to(args.device)

            proto = model(support_input).view(args.n_way, args.n_shot, -1)
            feature = proto.view(-1, args.dim).to('cpu')
            target = np.arange(args.n_way).repeat(args.n_shot)
            print(feature.shape, target.shape)
            
            for c in range(args.n_way):
                fake = hall(proto[c][0]).to('cpu')
                feature = torch.cat([feature, fake], dim=0)
                target = np.concatenate([target, [c+5 for _ in range(len(fake))]])
            draw_features(feature, target, 5, args.save_name)


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_way', default=5, type=int,
                        help='N_way (default: 5)')
    parser.add_argument('--n_shot', default=200, type=int,
                        help='N_shot (default: 1)')
    parser.add_argument('--n_query', default=15, type=int,
                        help='N_query (default: 15)')
    parser.add_argument('--n_aug', default=20, type=int,
                        help='M use in Hallucination (default: 10)')
    parser.add_argument('--dim', default=100, type=int,
                        help='dim for feature size (default: 100)')
    parser.add_argument('--learning_rate', default=5e-4,
                        type=float, help='learning rate for training')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--max_epoch', default=50, type=int,
                        help='Epoch for training')
    parser.add_argument('--distance_type', default='euclidean', type=str,
                        help='Distance type for training')
    parser.add_argument('--save_folder', default='weights/q2',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--train_csv', default='hw4_data/train.csv', type=str,
                        help="Training images csv file")
    parser.add_argument('--train_data_dir', default='hw4_data/train', type=str,
                        help="Training images directory")
    parser.add_argument('--model', type=str, help="Model checkpoint path")
    parser.add_argument('--hall', type=str,
                        help="Hallucination checkpoint path")
    parser.add_argument('--seed', default=877, type=int,
                        help="random seed")
    parser.add_argument('--save_name', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    train_dataset = MiniDataset(args.train_csv, args.train_data_dir)
    train_loader = DataLoader(train_dataset, batch_sampler=GeneratorSampler(
        args.train_csv, args.batch_size, args.n_way, args.n_shot, args.n_query),
        num_workers=3, worker_init_fn=worker_init_fn)

    model = Convnet4(out_channels=args.dim).to(args.device)
    model.load_state_dict(torch.load(args.model))
    hall = Hallucination(args.dim, args.n_aug).to(args.device)
    hall.load_state_dict(torch.load(args.hall))

    distance = Distance(args)
    predict(args, model, train_loader, distance, hall)
