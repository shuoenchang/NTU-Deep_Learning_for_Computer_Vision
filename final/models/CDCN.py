import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride,
                                padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel,
                               padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CDCNpp(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7, hidden_size=256):
        super(CDCNpp, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 80, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(80),
            nn.ReLU(),

        )

        self.Block1 = nn.Sequential(
            basic_conv(80, 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            basic_conv(160, int(160*1.6), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),
            basic_conv(int(160*1.6), 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, int(160*1.4), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),
            basic_conv(int(160*1.4), 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(160, 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, int(160*1.2), kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Original

        self.lastconv1 = nn.Sequential(
            basic_conv(160*3, 160, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, 1, kernel_size=3, stride=1,
                       padding=1, bias=False, theta=theta),
            nn.ReLU(),
        )

        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(
            size=(32, 32), mode='bilinear', align_corners=False)

        # new module
        self.gru = nn.GRU(1024, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, return_feature=False):	    	# x [3, 256, 256]
        
        bs, t, c, w, h = x.shape
        x = x.view(-1, c, w, h)

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat(
            (x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)
        # pdb.set_trace()

        map_x = self.lastconv1(x_concat)

        map_x = map_x.squeeze(1)
        feature = map_x.view(bs, t, -1)
        self.gru.flatten_parameters()
        gru_out, hidden = self.gru(feature)
        gru_out = torch.mean(gru_out, dim=1)
        predict = self.classifier(gru_out)
        predict = predict.squeeze(1)
        if return_feature:
            return map_x, x_concat, attention1, attention2, attention3, x_input
        else:
            # return map_x
            return map_x, predict


if __name__ == '__main__':
    model = CDCNpp()
    x = torch.randn(16, 3, 256, 256)
    y = model(x)
    print(y[0].shape)
