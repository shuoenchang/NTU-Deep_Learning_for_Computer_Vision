import numpy as np
import torch
from torch import nn
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
    return torch.mean((pred == label).type(torch.FloatTensor))


class Distance(object):
    def __init__(self, args):
        self.type = args.distance_type
        self.param_model = self.Parametric(
            args.dim, args.n_way).to(args.device)

    def __call__(self, support, query):
        if self.type == 'euclidean':
            return self.euclidean_dist(support, query)
        elif self.type == 'cosine':
            return self.cosine_dist(support, query)
        elif self.type == 'param':
            return self.param_model(support, query)

    def euclidean_dist(self, x, y):
        n_class = x.size(0)
        n_query = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(0).expand(n_query, n_class, d)
        y = y.unsqueeze(1).expand(n_query, n_class, d)

        return -(torch.pow(x - y, 2).sum(dim=2))

    def cosine_dist(self, x, y):
        n_class = x.size(0)
        n_query = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(0).expand(n_query, n_class, d)
        y = y.unsqueeze(1).expand(n_query, n_class, d)

        cos = nn.CosineSimilarity(dim=2)
        return cos(x, y)

    class Parametric(nn.Module):
        def __init__(self, f_dim, n_class):
            super().__init__()
            self.weight_cliping_limit = 0.01
            self.net = nn.Sequential(
                nn.Linear(f_dim*(1+n_class), f_dim),
                nn.Tanh(),
                nn.Linear(f_dim, f_dim),
                nn.Tanh(),
                nn.Linear(f_dim, n_class),
            )

        def forward(self, x, y):
            n_class = x.size(0)
            n_query = y.size(0)
            d = x.size(1)
            assert d == y.size(1)

            x = x.unsqueeze(0).expand(n_query, n_class, d)
            x = x.view(n_query, -1)
            inputs = torch.cat((x, y), dim=1)
            return self.net(inputs)

        def weight_cliping(self):
            for p in self.parameters():
                p.data.clamp_(-self.weight_cliping_limit,
                              self.weight_cliping_limit)
