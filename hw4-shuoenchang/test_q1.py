import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.prototypical import Convnet4
from src.dataset import MiniDataset, TestSampler
from src.utils import Distance, worker_init_fn, set_seed


def predict(args, model, data_loader, distance):

    with torch.no_grad():
        model.eval()
        # print(model)
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            with open(args.output_csv, 'a') as f:
                f.write("{}".format(i))
            # split data into support and query data
            support_input = data[:args.n_way * args.n_shot].to(args.device)
            query_input = data[args.n_way * args.n_shot:].to(args.device)

            proto = model(support_input).view(args.n_way, args.n_shot, -1)
            proto = proto.mean(1)

            feature = model(query_input)
            logits = distance(proto, feature)
            preds = torch.argmax(logits, dim=1)
            with open(args.output_csv, 'a') as f:
                for pred in preds:
                    f.write(",{}".format(pred))
                f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_way', default=5, type=int,
                        help='N_way (default: 5)')
    parser.add_argument('--n_shot', default=1, type=int,
                        help='N_shot (default: 1)')
    parser.add_argument('--n_query', default=15, type=int,
                        help='N_query (default: 15)')
    parser.add_argument('--model', type=str, help="Model checkpoint path")
    parser.add_argument('--param', type=str, help="Distance checkpoint path")
    parser.add_argument('--test_csv', default='hw4_data/val.csv',
                        type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', default='hw4_data/val',
                        type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', default='hw4_data/val_testcase.csv',
                        type=str, help="Test case csv")
    parser.add_argument('--output_csv', default='outputs/output.csv',
                        type=str, help="Output filename")
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for testing')
    parser.add_argument('--dim', default=100, type=int,
                        help='dim for feature size (default: 100)')
    parser.add_argument('--distance_type', default='euclidean', type=str,
                        help='Distance type for testing')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(123)
    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.n_way * (args.n_query + args.n_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=TestSampler(args.testcase_csv))

    # TODO: load your model
    model = Convnet4(out_channels=args.dim).to(args.device)
    model.load_state_dict(torch.load(args.model))
    distance = Distance(args)
    if args.distance_type == 'param':
        assert args.param
        distance.param.load_state_dict(torch.load(args.param))
    with open(args.output_csv, 'w') as f:
        f.write('episode_id')
        for i in range(75):
            f.write(',query{}'.format(i))
        f.write('\n')
    predict(args, model, test_loader, distance)
