import torch
import torch.nn as nn
import torchvision.models as models


class Feature(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 5, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Flatten(),

            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
			nn.ReLU(True),
            nn.Dropout(0.5),
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
            elif classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.1)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


class Classifier(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 10),
        )
        self.init_weights()

    def forward(self, x):
        return self.classifier(x)

    def init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.1)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    # f = F().cuda()
    # c = C().cuda()
    d = Discriminator().cuda()
    # print(f)
    # print(c)
    x = torch.rand((5, 3, 28, 28)).cuda()
    realfake, classes = d(x)

    print(realfake.shape, classes.shape)
