import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.ResNext import resnext101
from models.VGG import VGG16
from src.dataset import p1Dataset


def train(dataset, model, optimzer, criterion, device):
    model.to(device)
    model.train()
    total_loss = []
    for step, data in enumerate(dataset):
        optimzer.zero_grad()
        image = data['image'].to(device)
        target = data['target'].to(device)
        output = model(image)
        loss = criterion(output, target)
        total_loss.append(loss.item())
        loss.backward()
        optimzer.step()
        
    print('train: {}'.format(np.mean(total_loss)))
    
    return np.mean(total_loss)


def validation(dataset, model, criterion, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = []
    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            target = data['target'].to(device)
            output = model(image)
            loss = criterion(output, target)
            total_loss.append(loss.item())
            _, predict = output.topk(1)
            predict = predict.squeeze(1)
            correct += int(sum(predict == target))
            total += len(predict)
            accuracy = correct/total

    print('validation: {}, accuracy:{:.3f}'.format(
        np.mean(total_loss), accuracy))

    return np.mean(total_loss), accuracy


def main(args):
    with open('{}/log.txt'.format(args.save_folder), 'w') as f:
        pass

    train_set = DataLoader(dataset=p1Dataset('hw2_data/p1_data/train_50'),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_set = DataLoader(dataset=p1Dataset('hw2_data/p1_data/val_50'),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = resnext101(pretrained=True, n_classes=50)
    optimzer = optim.Adam(model.parameters(), lr=2e-6)
    criterion = CrossEntropyLoss(ignore_index=6)
    device = 'cuda:1'
    min_loss = 2
    for epoch in range(20):

        print('\nepoch: {}'.format(epoch))
        with open('{}/log.txt'.format(args.save_folder), 'a') as f:
            f.write('\nepoch: {}\n'.format(epoch))

        loss = train(train_set, model, optimzer, criterion, device)
        with open('{}/log.txt'.format(args.save_folder), 'a') as f:
            f.write('train: {}\n'.format(loss))

        loss, accuracy = validation(valid_set, model, criterion, device)
        with open('{}/log.txt'.format(args.save_folder), 'a') as f:
            f.write('validation: {}, accuracy:{:.3f}\n'.format(loss, accuracy))

        if accuracy > 0.7 and loss < min_loss:
            torch.save(model.state_dict(),
                       '{}/epoch_{:02d}-{:.3f}.pth'.format(args.save_folder, epoch, accuracy))
            min_loss = loss
            print('Best epoch: {}'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw2-1 Training Script')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q1',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    main(args)
