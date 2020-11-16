import torch.nn.functional as F
from torch import nn
import torch


class VAE_loss(nn.Module):
    def __init__(self, lambda_KL):
        super().__init__()
        self.lambda_KL = lambda_KL

    def forward(self, output, image, logvar, mu):

        loss_reconstruct = F.mse_loss(output, image)
        loss_KL = torch.mean(
            torch.sum(-0.5*(1+logvar-mu**2-torch.exp(logvar)), dim=1))
        loss = loss_reconstruct+self.lambda_KL*loss_KL

        return loss_reconstruct, loss_KL, loss


class DANN_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, domain, gt_domain, label=None, gt_label=None):
        domain = domain.view_as(gt_domain)
        loss_domain = F.binary_cross_entropy_with_logits(domain, gt_domain)
        if gt_label is not None:
            loss_label = F.cross_entropy(label, gt_label)
            return loss_domain, loss_label
        else:
            return loss_domain


if __name__ == '__main__':
    loss = VAE_loss(0.01)
    output = torch.rand(3, 3, 16, 16)
    image = torch.rand(3, 3, 16, 16)
    logvar = torch.rand(3, 512)
    mu = torch.rand(3, 512)
    loss(output, image, logvar, mu)
