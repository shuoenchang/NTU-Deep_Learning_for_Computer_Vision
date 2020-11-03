import torch
import torch.nn as nn
import torchvision.models as models


class resnext101(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000, feature=False):
        super().__init__()
        self.backbone = models.resnext101_32x8d(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, n_classes)
        self.feature = feature

    def forward(self, x):
        if self.feature:
            model_children = [x for x in self.backbone.children()]
            for module in model_children[:-2]:
                x = module(x)
            feature = torch.flatten(x, 1)
            x = model_children[-2](x)
            x = torch.flatten(x, 1)
            y = model_children[-1](x)
            return y, feature
        else:
            y = self.backbone(x)
            return y


if __name__ == '__main__':
    # from torchsummary import summary
    model = resnext101(pretrained=True, n_classes=50, feature=True)
    model.cuda()
    # summary(model, (3, 224, 224))
    print(model)
    x = torch.rand((1, 3, 224, 224)).cuda()
    y, feature = model(x)
    print(y.shape, feature.shape)
