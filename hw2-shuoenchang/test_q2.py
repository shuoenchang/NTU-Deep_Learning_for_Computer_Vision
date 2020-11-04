import argparse

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.segmentation import (DeepLabV3, Res101FCN, VGGFCN8s, VGGFCN16s,
                                 VGGFCN32s, EnSegNet8)
from src.dataset import p2Dataset
from src.mean_iou_evaluate import mean_iou_score
from src.utils import draw_masks
from scipy import misc
from imageio import imwrite


def test(dataset, model, device, output_path):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            name = data['name']
            output = model(image)
            _, predict = output.topk(1, dim=1)
            predict = predict.squeeze(1).cpu()

            for i in range(len(output)):
                _, predict = output[i].topk(1, dim=0)
                predict = predict.squeeze(0).cpu()
                mask = draw_masks(predict)
                imwrite(output_path+'/'+name[i]+'_mask.png', mask)


def main(args):
    test_set = DataLoader(dataset=p2Dataset(args.image_folder, has_gt=False),
                          batch_size=16, shuffle=False)
    device = 'cuda:0'
    if args.best:
        model = EnSegNet8(n_classes=7, pretrained=True).to(device)
        model.load_state_dict(torch.load('weights/2-2_best_0.700.pth', map_location=device))
    else:
        model = VGGFCN32s(n_classes=7, pretrained=True).to(device)
        model.load_state_dict(torch.load('weights/2-2_0.689.pth', map_location=device))
    test(test_set, model, device, output_path=args.save_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw2-1 Testing Script')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--best', dest='best', action='store_true')
    args = parser.parse_args()
    main(args)
