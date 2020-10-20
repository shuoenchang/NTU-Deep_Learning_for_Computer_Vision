import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

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
        if (step+1) % 50 == 0:
            print('train: {}'.format(np.mean(total_loss)))


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
    return accuracy


if __name__ == '__main__':
    train_set = DataLoader(dataset=p1Dataset('hw2_data/p1_data/train_50'),
                           batch_size=64, shuffle=True)
    valid_set = DataLoader(dataset=p1Dataset('hw2_data/p1_data/val_50'),
                           batch_size=64, shuffle=True)
    model = VGG16(pretrained=True, n_classes=50)
    optimzer = optim.SGD(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    device = 'cuda:0'
    max_acc = 0.7
    for epoch in range(150):
        print('\nepoch: {}'.format(epoch))
        train(train_set, model, optimzer, criterion, device)
        accuracy = validation(valid_set, model, criterion, device)
        if max_acc < accuracy:
            torch.save(
                model, 'weights/epoch_{:02d}-{:.3f}.pth'.format(epoch, accuracy))
            max_acc = accuracy
            print('Best epoch: {}'.format(epoch))
