import argparse

import numpy as np
import torch
import torch.optim as optim
from sklearn.manifold import TSNE
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.ResNext import resnext101
from models.VGG import VGG16
from src.dataset import p1Dataset
from src.utils import draw_features


def feature(dataset, model, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        features = np.empty((0, 2048*7*7))
        targets = np.empty((0,), dtype=np.int8)
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            target = data['target']
            output, feature = model(image)
            features = np.concatenate(
                (features, feature.cpu().numpy()), axis=0)
            targets = np.concatenate((targets, target), axis=0)
    print(features.shape, targets.shape)
    draw_features(features, targets)


def test(dataset, model, device, output_path):
    model.to(device)
    model.eval()
    with open(output_path, 'w') as f:
        f.write('image_id, label\n')

    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            output = model(image)
            _, predict = output.topk(1)
            predict = predict.squeeze(1)
            for image_id, label in zip(name, predict):
                with open(output_path, 'a') as f:
                    f.write(f'{image_id}, {label}\n')


def main(args):
    test_set = DataLoader(dataset=p1Dataset(args.image_folder, has_gt=False),
                          batch_size=200, shuffle=False)
    device = 'cuda:0'
    model = resnext101(pretrained=True, n_classes=50).to(device)
    model.load_state_dict(torch.load(
        '2-1_0.878.pth', map_location=device))
    test(test_set, model, device, output_path=args.save_folder+'/test_pred.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw2-1 Testing Script')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()
    main(args)
