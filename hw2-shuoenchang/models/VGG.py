import torch
import torch.nn as nn
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained)
        self.backbone.classifier._modules['6'] = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    model = VGG16(pretrained=True, n_classes=5)
    print(model)
    x = torch.rand((1, 3, 32, 32))
    y = model(x)
    print(y)
