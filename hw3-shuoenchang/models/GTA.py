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
            nn.ConvTranspose2d(latent_dim+feature_dim*2+10+1,
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
                               stride=2, padding=1),
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
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feature_dim, feature_dim*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feature_dim*2, feature_dim*4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*4),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feature_dim*4, feature_dim*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        self.realfake = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim*2, 1),
            # nn.Conv2d(128, 1, 3),
        )

        self.classes = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim*2, 10),
            # nn.Conv2d(128, 10, 3),
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
            elif classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.1)


class Feature(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature = nn.Sequential(
            nn.Conv2d(3, feature_dim, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feature_dim, feature_dim, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(feature_dim, feature_dim*2, 5),
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )
        self.init_weights()

    def forward(self, x):
        x = self.feature(x)
        return x

    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)

class Classifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim*2, 10),
        )
        self.init_weights()

    def forward(self, x):
        return self.classifier(x)
    
    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.1)


if __name__ == '__main__':
    # f = F().cuda()
    # c = C().cuda()
    d = Discriminator().cuda()
    # print(f)
    # print(c)
    x = torch.rand((5, 3, 28, 28)).cuda()
    realfake, classes = d(x)

    print(realfake.shape, classes.shape)
