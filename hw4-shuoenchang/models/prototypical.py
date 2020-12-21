import torch
import torch.nn as nn
import torch.functional as F


class Convnet4(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=100):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(in_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
            self.conv_block(hid_channels, hid_channels),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 500),
            nn.ReLU(True),

            nn.Linear(500, out_channels),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x.view(x.size(0), -1)


class Hallucination(nn.Module):
    def __init__(self, f_dim=100, n_aug=10):
        super().__init__()
        self.f_dim = f_dim
        self.M = n_aug
        self.net = nn.Sequential(
            nn.Linear(f_dim*2, f_dim),
            nn.ReLU(True),
            nn.Linear(f_dim, f_dim),
            nn.ReLU(True),
            nn.Linear(f_dim, f_dim),
        )

    def forward(self, x, return_noise=False):
        noise = torch.randn(self.M, self.f_dim).to(x.device)
        x = x.expand(noise.shape)
        x = torch.cat([x, noise], dim=1)
        x = self.net(x)
        if return_noise:
            return x, noise
        return x


class Discriminator(nn.Module):
    def __init__(self, f_dim, weight_cliping_limit):
        super().__init__()
        self.weight_cliping_limit = weight_cliping_limit
        self.net = nn.Sequential(
            nn.Linear(f_dim, f_dim//2),
            nn.LeakyReLU(),
            nn.Linear(f_dim//2, f_dim//2),
            nn.LeakyReLU(),
            nn.Linear(f_dim//2, 1),
        )

    def forward(self, x):
        outputs = self.net(x)
        return outputs

    def weight_cliping(self):
        for p in self.parameters():
            p.data.clamp_(-self.weight_cliping_limit,
                          self.weight_cliping_limit)


if __name__ == '__main__':
    model = Discriminator(f_dim=100, weight_cliping_limit=0.005)
    print(model)
