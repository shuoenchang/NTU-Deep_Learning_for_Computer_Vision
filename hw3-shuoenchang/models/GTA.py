import torch
import torch.nn as nn
import torchvision.models as models


class Generator(nn.Module):
    def __init__(self, latent_dim, feature_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.G = nn.Sequential(
            # (batch, latent, 1, 1) => (batch, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim+feature_dim+10+1,
                               512, kernel_size=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4,
                               stride=2, padding=3, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.init_weights()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        fake = self.G(x)
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
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
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
        )

        self.realfake = nn.Sequential(
            nn.Conv2d(512, 1, 2),
            nn.Flatten(),
        )

        self.classes = nn.Sequential(
            nn.Conv2d(512, 10, 2),
            nn.Flatten(),
        )
        self.init_weights()

    def forward(self, x):
        x = self.feature(x)
        realfake = self.realfake(x)
        classes = self.classes(x)
        return realfake, classes

    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Feature(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, feature_dim, 4),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
        )

    def forward(self, x):
        return self.feature(x)


class Classifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    f = F().cuda()
    c = C().cuda()
    print(f)
    print(c)
    x = torch.rand((5, 3, 28, 28)).cuda()
    feature = f(x)
    classes = c(feature)

    print(feature.shape, classes.shape)
