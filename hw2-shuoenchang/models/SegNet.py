import torch
import torch.nn as nn
import torchvision.models as models


class FCN32s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained).features
        self.fcn = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcn(x)
        x = self.classifier(x)
        return x


class FCN8s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((224, 224))
        self.block1 = models.vgg16(pretrained=pretrained).features[0:5]
        self.block2 = models.vgg16(pretrained=pretrained).features[5:10]
        self.block3 = models.vgg16(pretrained=pretrained).features[10:17]
        self.block4 = models.vgg16(pretrained=pretrained).features[17:24]
        self.block5 = models.vgg16(pretrained=pretrained).features[24:]
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x3 = x
        x = self.block4(x)
        x4 = x
        x = self.block5(x)
        x = self.deconv1(x)
        x = self.deconv2(x+x4)
        x = self.deconv3(x+x3)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = FCN8s(pretrained=False, n_classes=7)
    print(model)
    x = torch.rand((1, 3, 512, 512))
    y = model(x)
    print(y.shape)
