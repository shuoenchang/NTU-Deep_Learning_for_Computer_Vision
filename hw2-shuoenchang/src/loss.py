# code from: https://gist.github.com/f1recracker/0f564fd48f15a58f4b92b3eb3879149b
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='mean')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        CE_loss = super().forward(input_, target)
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * CE_loss
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else: 
            return loss


if __name__ == '__main__':
    loss = FocalLoss(gamma=2)
    predict = torch.rand(16, 7, 512, 512)
    target = torch.randint(6, (16, 512, 512), dtype=torch.int64)
    l = loss(predict, target)
    print(l)
