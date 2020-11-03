import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.segmentation import VGGFCN8s, VGGFCN16s, VGGFCN32s, Res101FCN, DeepLabV3, EnSegNet8, EnSegNet32, EnVGGFCN32s
from src.dataset import p2Dataset
from src.mean_iou_evaluate import mean_iou_score
from src.utils import logging


def train(dataset, model, optimzer, criterion, device):
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
    return np.mean(total_loss)


def validation(dataset, model, criterion, device):
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
    iou = mean_iou_score(total_predict, total_target)

    return np.mean(total_loss), iou


def main(args):
    torch.manual_seed(422)
    log = logging(args.save_folder)

    train_set = DataLoader(dataset=p2Dataset(root_dir='hw2_data/p2_data/train'),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_set = DataLoader(dataset=p2Dataset(root_dir='hw2_data/p2_data/validation'),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = args.device
    model = VGGFCN8s(pretrained=True, n_classes=7)
    
    # if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    # model.load_state_dict(torch.load(
    #     'weights/q2/fcn8s/epoch_149-0.660.pth', map_location=args.device))
    optimzer = optim.Adam(model.parameters(), lr=7e-5)
    criterion = CrossEntropyLoss(ignore_index=6)
    min_loss = 2
    
    for epoch in range(50):
        log.write('\nepoch: {}\n'.format(epoch))
        loss = train(train_set, model, optimzer, criterion, device)
        log.write('train: {}\n'.format(loss))

        loss, iou = validation(valid_set, model, criterion, device)
        log.write('validation: {}, mIoU:{:.3f}\n'.format(loss, iou))
        if iou > 0.6 and loss < min_loss:
            torch.save(model.state_dict(),
                       '{}/epoch_{:02d}-{:.3f}.pth'.format(args.save_folder, epoch, iou))
            min_loss = loss
            log.write('Best epoch: {}\n'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw2-2 Training Script')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q2/fcn8s',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    main(args)
