import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function


class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

        )
        self.label_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 10),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.GRL_layer = GRL.apply

    def forward(self, x, lambda_domain, draw=False):
        x = self.feature_extractor(x)
        feature = torch.flatten(x, 1)
        label = self.label_classifier(feature)
        domain = self.GRL_layer(feature, lambda_domain)
        domain = self.domain_classifier(domain)
        if draw:
            return feature
        else:
            return label, domain


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lambda_domain):
        ctx.lambda_domain = lambda_domain
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -(ctx.lambda_domain*grad_output), None


if __name__ == '__main__':
    model = DANN()
    print(model)
    x = torch.rand(5, 3, 28, 28)
    label, domain = model(x, lambda_domain=0.3)
    print(label.shape, domain.shape)
