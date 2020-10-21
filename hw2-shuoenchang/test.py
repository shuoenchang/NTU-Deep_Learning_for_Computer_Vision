import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.VGG import VGG16
from src.dataset import p1Dataset


def test(dataset, model, device):
    model.to(device)
    model.eval()
    with open('output/p1.csv', 'w') as f:
        f.write('image_id, label\n')

    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            output = model(image)
            _, predict = output.topk(1)
            predict = predict.squeeze(1)
            for image_id, label in zip(name, predict):
                with open('output/p1.csv', 'a') as f:
                    f.write(f'{image_id}, {label}\n')


if __name__ == '__main__':
    test_set = DataLoader(dataset=p1Dataset('hw2_data/p1_data/val_50', has_gt=False),
                          batch_size=256, shuffle=False)
    model = VGG16(pretrained=True, n_classes=50)
    model.load_state_dict(torch.load('weights/0.756.pth'))
    device = 'cuda:1'
    max_acc = 0.7
    test(test_set, model, device)
