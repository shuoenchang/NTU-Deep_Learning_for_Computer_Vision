import argparse
import os

import numpy as np
from numpy.core.fromnumeric import sort
import torch
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn import metrics

from models.CDCN import CDCNpp
from src.dataset import CockRoach_Dataset
from src.loss import Loss
from src.optimizer import Optimizer
from src.utils import calculate_acc, set_seed, worker_init_fn


def train(data_loader, model, criterion, optimizer, args):
    model.train()
    total_loss = []
    total_absolute = []
    total_contrastive = []
    total_classify = []
    total_label = []
    total_pred = []
    optimizer['m'].zero_grad()
    for step, data in enumerate(data_loader):
        video, binary, label, binary_ = data['video'].to(args.device), data['binary'].to(
            args.device), data['label'].to(args.device), data['binary_'].to(args.device)
        bs, t, c, w, h = video.shape
        video = video.view(-1, c, w, h)
        binary = binary.view(-1, 32, 32)
        map_x = model(video)  # noqa

        absolute_loss = criterion['absolute'](map_x, binary)
        total_absolute.append(absolute_loss.item())
        contrastive_loss = criterion['contrastive'](map_x, binary)
        total_contrastive.append(contrastive_loss.item())

        score_norm = torch.mean(map_x.view(bs, -1), dim=1) \
            # / torch.sum(binary_.view(bs, -1), dim=1)
        score_norm[score_norm > 1] = 1
        classify_loss = criterion['classify'](
            score_norm, label.type_as(score_norm))
        total_classify.append(classify_loss.item())
        for i in range(len(data['label'])):
            total_label.append(data['label'][i].item())
            total_pred.append(score_norm[i].item())

        loss = absolute_loss+contrastive_loss # +classify_loss
        total_loss.append(loss.item())

        loss = loss/args.accumulation_steps
        loss.backward()

        if (step+1) % args.accumulation_steps == 0:
            optimizer['m'].step()
            optimizer['m'].zero_grad()
        if step % 5 == 0:
            print('\rstep {}: loss {:.3f}, a {:.3f}, c {:.3f}, class {:.3f}'.format(step, np.mean(total_loss), np.mean(total_absolute),
                                                                                  np.mean(total_contrastive), np.mean(total_classify)), end='')
    fpr, tpr, thresholds = metrics.roc_curve(total_label, total_pred, pos_label=1)
    
    print('\rTraining: AUC {:.3f} loss {:.3f}, a {:.3f}, c {:.3f}, class {:.3f}'.format(metrics.auc(fpr, tpr), np.mean(total_loss), np.mean(total_absolute),
                                                                                      np.mean(total_contrastive), np.mean(total_classify)))


def val(data_loader, model, criterion, args):
    model.eval()
    with open('outputs/val.csv', 'w') as f:
        f.write('labels,preds\n')
    labels = []
    preds = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            video, binary_, label = data['video'].to(args.device), data['binary_'].to(
                args.device), data['label']#.to(args.device)
            bs, t, c, w, h = video.shape
            video = video.view(-1, c, w, h)
            map_x = model(video)  # noqa
            map_x = map_x.view(bs, t, 32, 32)
            for video_i in range(bs):
                map_score = []
                for frame_i in range(t):
                    score_norm = torch.mean(map_x[video_i, frame_i, :, :]) \
                        # / torch.sum(binary_[video_i, frame_i, :, :])
                    map_score.append(score_norm.item())
                map_score = np.array(sorted(map_score))
                # pos = sum(map_score[2:-2]>0.5)
                # neg = sum(map_score[2:-2]<=0.5)
                # if pos>neg:
                #     map_score = map_score[-1]
                # elif pos<neg:
                #     map_score = map_score[0]
                # else:
                #     if np.mean(map_score[2:-2])>0.5:
                #         map_score = map_score[-1]
                #     else:
                #         map_score = map_score[0]
                map_score = np.mean(map_score)
                map_score = 1 if map_score>1 else map_score
                with open('outputs/val.csv', 'a') as f:
                    f.write('{},{}\n'.format(
                        label[video_i].item(), map_score))
                labels.append(label[video_i].item())
                preds.append(map_score)
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    loss = criterion['classify'](torch.Tensor(preds), torch.Tensor(labels))
    acc = calculate_acc(torch.Tensor(preds), torch.Tensor(labels))
    print('\rValidation: AUC {:.7f}, loss {:.3f}, acc {:.3f}'.format(metrics.auc(fpr, tpr), loss, acc))


def test(data_loader, model, criterion, args):
    model.eval()
    total_loss = []
    with open(args.output_csv, 'w') as f:
        f.write('video_id,label\n')
    with torch.no_grad():
        for step, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            video, binary_, video_id = data['video'].to(args.device), data['binary_'].to(
                args.device), data['video_id']
            bs, t, c, w, h = video.shape
            video = video.view(-1, c, w, h)
            map_x = model(video)  # noqa
            map_x = map_x.view(bs, t, 32, 32)
            for video_i in range(bs):
                map_score = []
                for frame_i in range(t):
                    score_norm = torch.mean(map_x[video_i, frame_i, :, :]) \
                        # / torch.sum(binary_[video_i, frame_i, :, :])
                    map_score.append(score_norm.item())
                map_score = np.array(sorted(map_score))
                # pos = sum(map_score[2:-2]>0.5)
                # neg = sum(map_score[2:-2]<=0.5)
                # if pos>neg:
                #     map_score = map_score[-1]
                # elif pos<neg:
                #     map_score = map_score[0]
                # else:
                #     if np.mean(map_score[2:-2])>0.5:
                #         map_score = map_score[-1]
                #     else:
                #         map_score = map_score[0]
                map_score = np.mean(map_score)
                map_score = 1 if map_score>1 else map_score
                with open(args.output_csv, 'a') as f:
                    f.write('{},{}\n'.format(
                        str(video_id[video_i]), map_score))


def main(args):

    train_dataset = CockRoach_Dataset(
        args.train_data_dir, 'train', args.train_frame_per_video)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=3, shuffle=True)
    val_dataset = CockRoach_Dataset(
        args.val_data_dir, 'val', args.val_frame_per_video)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=3)
    test_dataset = CockRoach_Dataset(
        args.test_data_dir, 'test', args.test_frame_per_video)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=3)

    model = CDCNpp().to(args.device)
    if args.val or args.test:
        print('load model')
        model.load_state_dict(torch.load('weights/depth_flip_noacc/epoch_286.pth', map_location=args.device))
    model = nn.DataParallel(model)
    criterion = {'absolute': Loss('MSE', args),
                 'contrastive': Loss('CD', args),
                 'classify': Loss('BCELOSS', args)}
    optimizer = {'m': Optimizer(
        model.parameters(), 'Adam', lr=args.learning_rate)}
    lr_scheduler = {'m': StepLR(
        optimizer['m'], step_size=args.step_size, gamma=args.gamma)}
    min_loss = 1
    if args.val:
        val(val_loader, model, criterion, args)
    elif args.test:
        test(test_loader, model, criterion, args)
    else:
        for epoch in range(args.max_epoch):
            print('epoch: ', epoch)
            train(train_loader, model, criterion, optimizer, args)
            lr_scheduler['m'].step()
            # if loss < min_loss:
            torch.save(model.module.state_dict(),
                       '{}/epoch_{:02d}.pth'.format(args.save_folder, epoch))


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--learning_rate', default=1e-4,
                        type=float, help='learning rate for training')
    parser.add_argument('--step_size', default=20,
                        type=int, help='step_size for scheduler')
    parser.add_argument('--gamma', default=0.5,
                        type=float, help='gamma for scheduler')
    parser.add_argument('--BCE_weight', default=5,
                        type=float, help='pos weight in BCELoss')
    parser.add_argument('--accumulation_steps', default=5,
                        type=int, help='accumulation steps for gradients')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--max_epoch', default=150, type=int,
                        help='Epoch for training')
    parser.add_argument('--save_folder', default='weights',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--train_data_dir', default='data/oulu/train', type=str,
                        help="Training images directory")
    parser.add_argument('--train_frame_per_video', default=1, type=int,
                        help="Training frame per video")
    parser.add_argument('--val_data_dir', default='data/oulu/val', type=str,
                        help="Val images directory")
    parser.add_argument('--val_frame_per_video', default=10, type=int,
                        help="Val frame per video")
    parser.add_argument('--test_data_dir', default='data/oulu/test', type=str,
                        help="Test images directory")
    parser.add_argument('--output_csv', default='outputs/pred_oulu.csv', type=str,
                        help="Test images directory")
    parser.add_argument('--test_frame_per_video', default=10, type=int,
                        help="Test frame per video")
    parser.add_argument('--seed', default=877, type=int,
                        help="random seed")
    parser.add_argument('--val', action='store_true',
                        default=False, help='whether validation')
    parser.add_argument('--test', action='store_true',
                        default=False, help='whether validation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    main(args)
