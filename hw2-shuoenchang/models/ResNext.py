import torch
import torch.nn as nn
import torchvision.models as models


class resnext101(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.resnext101_32x8d(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    model = resnext101(pretrained=True, n_classes=5)
    print(model.backbone)
    x = torch.rand((1, 3, 224, 224))
    y = model(x)
    print(y)
