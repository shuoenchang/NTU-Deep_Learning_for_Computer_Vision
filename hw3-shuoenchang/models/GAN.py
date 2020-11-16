import torch
import torch.nn as nn
import torchvision.models as models


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            # (batch, latent, 1, 1) => (batch, 1024, 4, 4)
            nn.ConvTranspose2d(latent_dim, 1024, kernel_size=4),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, x):
        fake = self.generator(x)
        return fake


class Discriminator(nn.Module):
    def __init__(self, weight_cliping_limit):
        super().__init__()
        self.weight_cliping_limit = weight_cliping_limit
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, 5, stride=2, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 1024, 5, stride=2, padding=2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(1024, 1, 4),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.discriminator(x)
        return x

    def weight_cliping(self):
        for p in self.parameters():
            p.data.clamp_(-self.weight_cliping_limit,
                          self.weight_cliping_limit)


if __name__ == '__main__':
    model = Generator(100, 0.001).cuda()
    model.weight_cliping()
    print(model)
    x = torch.rand((5, 100, 1, 1)).cuda()
    y = model(x)
    print(y.shape)
