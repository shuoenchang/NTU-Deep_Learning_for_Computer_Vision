import torch
import torch.nn as nn


class Convnet4(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, out_channels),
        )
        # self.fc = nn.Linear(1600, 64)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


        
