import torch
import torch.nn as nn
import torchvision.models as models


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            # (batch, latent, 1, 1) => (batch, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=5,
                               stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.init_weights()

    def forward(self, x):
        fake = self.generator(x)
        return fake
    
    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, weight_cliping_limit):
        super().__init__()
        self.weight_cliping_limit = weight_cliping_limit
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 256, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 512, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(512, 1, 4),
            nn.Flatten(),
        )
        self.init_weights()

    def forward(self, x):
        x = self.discriminator(x)
        return x

    def weight_cliping(self):
        for p in self.parameters():
            p.data.clamp_(-self.weight_cliping_limit,
                          self.weight_cliping_limit)

    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    model = Discriminator(0).cuda()
    print(model)
    # x = torch.rand((5, 100, 1, 1)).cuda()
    # y = model(x)
    # print(y.shape)
