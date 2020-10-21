import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.FCN import FCN32s
from src.dataset import p2Dataset
from src.mean_iou_evaluate import mean_iou_score


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
    with open('weights/q2/log.txt', 'a') as f:
        f.write('train: {}\n'.format(np.mean(total_loss)))


def validation(dataset, model, criterion, device):
    model.to(device)
    model.eval()
    total_loss = []
    total_predict = np.empty((0, 512, 512))
    total_target = np.empty((0, 512, 512))
    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            target = data['target'].to(device)
            output = model(image)

            loss = criterion(output, target)
            total_loss.append(loss.item())
            _, predict = output.topk(1, dim=1)
            predict = predict.squeeze(1)

            total_predict = np.concatenate(
                (total_predict, np.array(predict.cpu())))
            total_target = np.concatenate(
                (total_target, np.array(target.cpu())))
    print('validation: {}'.format(np.mean(total_loss)))
    iou = mean_iou_score(total_predict, total_target)
    with open('weights/q2/log.txt', 'a') as f:
        f.write('validation: {}, mIoU:{:.3f}\n'.format(
            np.mean(total_loss), iou))

    return iou, np.mean(total_loss)


if __name__ == '__main__':
    with open('weights/q2/log.txt', 'w') as f:
        pass

    train_set = DataLoader(dataset=p2Dataset(root_dir='hw2_data/p2_data/train'),
                           batch_size=16, shuffle=True)
    valid_set = DataLoader(dataset=p2Dataset(root_dir='hw2_data/p2_data/validation'),
                           batch_size=16, shuffle=True)
    model = FCN32s(pretrained=True, n_classes=7)
    optimzer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    device = 'cuda:1'
    min_loss = 2
    for epoch in range(20):
        if epoch == 5:
            optimzer = optim.Adam(model.parameters(), lr=3e-5)
        if epoch == 10:
            optimzer = optim.Adam(model.parameters(), lr=1e-5)

        print('\nepoch: {}'.format(epoch))
        with open('weights/q2/log.txt', 'a') as f:
            f.write('\nepoch: {}\n'.format(epoch))

        train(train_set, model, optimzer, criterion, device)
        iou, loss = validation(valid_set, model, criterion, device)
        if iou > 0.6 and loss < min_loss:
            torch.save(model.state_dict(),
                       'weights/q2/epoch_{:02d}-{:.3f}.pth'.format(epoch, iou))
            min_loss = loss
            print('Best epoch: {}'.format(epoch))
