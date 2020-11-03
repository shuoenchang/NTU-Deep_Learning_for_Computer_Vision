import torch
import torch.nn as nn
import torchvision.models as models


class VGGFCN32s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.backbone = models.vgg16(pretrained=pretrained).features
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier = nn.Conv2d(4096, n_classes, kernel_size=1)
        self.upsampled = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=32, stride=32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcn(x)
        x = self.classifier(x)
        x = self.upsampled(x)
        return x


class VGGFCN16s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((224, 224))
        self.block1 = models.vgg16(pretrained=pretrained).features[0:5]
        self.block2 = models.vgg16(pretrained=pretrained).features[5:10]
        self.block3 = models.vgg16(pretrained=pretrained).features[10:17]
        self.block4 = models.vgg16(pretrained=pretrained).features[17:24]
        self.block5 = models.vgg16(pretrained=pretrained).features[24:]
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.classifier_block4 = nn.Conv2d(512, n_classes, kernel_size=1)
        self.classifier_block5 = nn.Conv2d(4096, n_classes, kernel_size=1)
        self.upsampled_2 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=2, stride=2)
        self.upsampled_16 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=16, stride=16)

    def forward(self, x):
        x = self.block1(x)  # 256
        x = self.block2(x)  # 128
        x = self.block3(x)  # 64
        x = self.block4(x)  # 32
        x4 = self.classifier_block4(x)
        x = self.block5(x)
        x = self.fcn(x)
        x5 = self.classifier_block5(x)
        x5 = self.upsampled_2(x5)
        x = self.upsampled_16(x4+x5)
        return x


class VGGFCN8s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((224, 224))
        self.block1 = models.vgg16(pretrained=pretrained).features[0:5]
        self.block2 = models.vgg16(pretrained=pretrained).features[5:10]
        self.block3 = models.vgg16(pretrained=pretrained).features[10:17]
        self.block4 = models.vgg16(pretrained=pretrained).features[17:24]
        self.block5 = models.vgg16(pretrained=pretrained).features[24:]
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )
        self.classifier_block3 = nn.Sequential(
            nn.Conv2d(256, n_classes, kernel_size=1),
            # nn.ReLU(inplace=True)
        )
        self.classifier_block4 = nn.Sequential(
            nn.Conv2d(512, n_classes, kernel_size=1),
            # nn.ReLU(inplace=True)
        )
        self.classifier_block5 = nn.Sequential(
            nn.Conv2d(4096, n_classes, kernel_size=1),
            # nn.ReLU(inplace=True)
        )
        self.upsampled_2_5 = nn.Sequential(
            nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2),
            # nn.ReLU(inplace=True)
        )
        self.upsampled_2_4 = nn.Sequential(
            nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2),
            # nn.ReLU(inplace=True)
        )
        self.upsampled_8 = nn.ConvTranspose2d(
            n_classes, n_classes, kernel_size=8, stride=8)

    def forward(self, x):
        x = self.block1(x)  # 256
        x = self.block2(x)  # 128
        x = self.block3(x)  # 64
        x3 = self.classifier_block3(x)
        x = self.block4(x)  # 32
        x4 = self.classifier_block4(x)
        x = self.block5(x)
        x = self.fcn(x)
        x5 = self.classifier_block5(x)
        x5 = self.upsampled_2_5(x5)
        x4 = self.upsampled_2_4(x4+x5)
        x = self.upsampled_8(x3+x4)
        return x


class Res101FCN(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.model = models.segmentation.fcn_resnet101(pretrained=pretrained)
        self.model.classifier[-1] = nn.Conv2d(512, n_classes, 1, 1)

    def forward(self, x):
        x = self.model(x)['out']
        return x


class DeepLabV3(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet101(
            pretrained=pretrained)
        self.model.classifier[-1] = nn.Conv2d(256, n_classes, 1, 1)

    def forward(self, x):
        x = self.model(x)['out']
        return x


class EnSegNet8(nn.Module):
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


class EnSegNet32(nn.Module):
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
        x = self.block4(x)
        x = self.block5(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.classifier(x)
        return x
    
    
class EnVGGFCN32s(nn.Module):
    def __init__(self, pretrained=False, n_classes=1000):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool2d((224, 224))
        self.block1 = models.vgg16(pretrained=pretrained).features[0:5]
        self.block2 = models.vgg16(pretrained=pretrained).features[5:10]
        self.block3 = models.vgg16(pretrained=pretrained).features[10:17]
        self.block4 = models.vgg16(pretrained=pretrained).features[17:24]
        self.block5 = models.vgg16(pretrained=pretrained).features[24:]
        self.fcn = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=3,
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
        x = self.block4(x)
        x = self.block5(x)
        x = self.fcn(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.classifier(x)
        return x
    
    
if __name__ == '__main__':
    model = VGGFCN8s(n_classes=7, pretrained=True)
    print(model)
    x = torch.rand((4, 3, 512, 512))
    y = model(x)
    print(y.shape)
