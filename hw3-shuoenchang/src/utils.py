import numpy as np
import torch


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def class_to_onehot(y, device):
    y = y.unsqueeze(1)
    onehot = torch.zeros(len(y), 11).to(device)
    return onehot.scatter_(1, y, 1)