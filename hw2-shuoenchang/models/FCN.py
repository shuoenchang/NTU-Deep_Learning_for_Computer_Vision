import torch
import torch.nn as nn
import torchvision.models as models


class FCN32s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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
        # x = self.avgpool(x)
        x = self.fcn(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = FCN32s(pretrained=False, n_classes=7)
    print(model)
    x = torch.rand((1, 3, 512, 512))
    y = model(x)
    print(y.shape)
