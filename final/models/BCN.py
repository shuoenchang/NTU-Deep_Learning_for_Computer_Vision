import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCN(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 50, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1 = ConvBlock(50, 100)
        self.bconv1 = BilateralConvBlock(50, 100, sigma)
        
        self.conv2 = ConvBlock(100, 100)
        self.bconv2 = BilateralConvBlock(100, 100, sigma)

        self.conv3 = ConvBlock(100, 100)
        self.bconv3 = BilateralConvBlock(100, 100, sigma)

        self.refine_low = MFRM(c_in=100)
        self.refine_med = MFRM(c_in=100)
        self.refine_high = MFRM(c_in=100)

    def forward(self, x):
        ''' x: (3, H, W) '''
        x = self.conv0(x)
        
        x1_c = self.conv1(x)
        x1_b = self.bconv1(x)
        low_feature = F.max_pool2d(x1_b + x1_c, kernel_size=3, stride=2, padding=1)        

        x = F.max_pool2d(x1_c, kernel_size=3, stride=2, padding=1)
        x2_c = self.conv2(x)
        x2_b = self.bconv2(x)
        med_feature = F.max_pool2d(x2_b + x2_c, kernel_size=3, stride=2, padding=1)        

        x = F.max_pool2d(x2_c, kernel_size=3, stride=2, padding=1)
        x3_c = self.conv3(x)
        x3_b = self.bconv3(x)
        high_feature = F.max_pool2d(x3_b + x3_c, kernel_size=3, stride=2, padding=1)

        #print(low_feature.shape, med_feature.shape, high_feature.shape)
        refine_low_feature  =  self.refine_low(low_feature).repeat(1, 1, 1, 1) 
        refine_med_feature  =  self.refine_med(med_feature).repeat(1, 1, 2, 2) 
        refine_high_feature = self.refine_high(high_feature).repeat(1, 1, 4, 4) 
        
        #print(refine_low_feature.shape, refine_med_feature.shape, refine_high_feature.shape)
        preds = torch.cat((refine_low_feature, refine_med_feature, 
                        refine_high_feature), dim=-3)  # -3: channel
        print(preds.shape)
        return preds

def ConvBlock(c_in=100, c_out=100):
    return nn.Sequential(
        nn.Conv2d(c_in, 100, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.Conv2d(150, c_out, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
    )

def BilateralConvBlock(c_in=100, c_out=100, sigma=0.1):
    return nn.Sequential(
        BilateralOperator(sigma),
        nn.Conv2d(c_in, 100, kernel_size=3, stride=1, padding=1),
        nn.ReLU(), 
        nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
        nn.Conv2d(150, c_out, kernel_size=3, stride=1, padding=1), 
        nn.ReLU(),
    )

class BilateralOperator(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        ''' x: (C, H, W) '''
        B, C, H, W = x.shape
        DBO = torch.zeros_like(x)
        for i in range(H):
            for j in range(W):
                I_curr = x[:, :, i, j].unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
                DBO_kernel = self.gaussian(x-I_curr) # (B, C, H, W)
                DBO_norm = (DBO_kernel * x).view(B, C, -1).sum(-1) \
                                / DBO_kernel.view(B, C, -1).sum(-1)
                DBO[:, :, i, j] = DBO_norm.view(B, C)

        return DBO
        
    def gaussian(self, x):
        return torch.exp(-torch.pow(x/self.sigma, 2))

class MFRM(nn.Module):
    def __init__(self, c_in=3, c_hid=20, k=5):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(c_hid, k**2, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.Conv2d(k**2, 1, kernel_size=(3, 3), padding=1),
            nn.Softmax()
        )

    def forward(self, x):
        ''' x: (3, H, W) '''
        score = self.score(x)
        x = x * score
        return x



if __name__ == "__main__":
    from torchsummary import summary
    model = BCN()
    summary(model, (3, 64, 64), device='cpu')
    
    # a = torch.rand((2, 3, 64, 64))
    # out, _, _ = model(a)
    # print(out.shape)
