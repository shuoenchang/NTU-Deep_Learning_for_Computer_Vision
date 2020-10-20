import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGG16(pretrained=False, n_classes=5)
    x = torch.rand((1, 3, 32, 32))
    y = model(x)
    print(y)
