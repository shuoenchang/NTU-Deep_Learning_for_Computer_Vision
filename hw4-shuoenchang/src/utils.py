import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def calculate_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return torch.mean((pred==label).type(torch.FloatTensor))


class Distance(object):
    def __init__(self, distance_type):
        self.type = distance_type

    def __call__(self, support, query):
        if self.type == 'euclidean':
            return self.euclidean_dist(support, query)

    def euclidean_dist(self, x, y):
        n_class = x.size(0)
        n_query = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(0).expand(n_query, n_class, d)
        y = y.unsqueeze(1).expand(n_query, n_class, d)

        return -(torch.pow(x - y, 2).sum(dim=2))
