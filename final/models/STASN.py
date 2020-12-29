import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from src.optimizer import Optimizer
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class STASN(nn.Module):
    def __init__(self, args, hidden_size=256, device=device):
        super().__init__()
        self.args = args
        self.tasm = TASM(hidden_size=hidden_size).to(device)
        self.ram = RAM(s_h=args.s_h, s_w=args.s_w, K=args.K).to(device)
        self.sasm = SASM().to(device)
        self.model_dic = {'TASM': self.tasm,
                          'RAM': self.ram, 'SASM': self.sasm}
        self.optimizer = None
        self.scheduler = None
        self.stage = args.stage
        if self.stage != 'test':
            self.optimizer = {}
            self.scheduler = {}
            optimizers = args.optimizer
            for name, opt in zip(self.model_dic.keys(), optimizers):
                self.optimizer[name] = Optimizer(
                    self.model_dic[name].parameters(), opt, args.lr)
                self.scheduler[name] = torch.optim.lr_scheduler.StepLR(
                    self.optimizer[name], step_size=5, gamma=0.1)
        self.K = args.K

    def forward(self, x, label, criterion=None, update=False, epoch=0):
        #bs, t, c, h, w = x.size()
        predict, feature_map = self.tasm(x, label, update=update)
        score_map = None
        location_label = None
        tasm_loss = 0

        if update:
            tasm_loss = self.update(
                predict, label.unsqueeze(-1).float(), 'TASM', criterion['BCE']).item()
        else:
            if self.stage == 'train':
                tasm_loss = self.compute_loss(
                    predict.squeeze(-1), label.float(), criterion['BCE']).item()

        if update:
            if epoch > 10:
                location_label = self.getLocation(feature_map)

        #loc = self.ram(feature_map)

        return predict, tasm_loss

    def step(self):
        for model_name in self.scheduler.keys():
            self.scheduler[model_name].step()

    def update(self, x, label, model_name, criterion):
        self.optimizer[model_name].zero_grad()
        loss = self.compute_loss(x, label, criterion)
        loss.backward()
        self.optimizer[model_name].step()
        return loss

    def load(self):
        self.tasm.load_state_dict(torch.load(os.path.join(
            self.args.save_dir, f'TASM_seed_{self.args.seed}_opt_{self.args.optimizer[0]}_lr_{self.args.lr}')))

    def save(self):
        torch.save(self.tasm.state_dict(), os.path.join(self.args.save_dir,
                                                        f'TASM_seed_{self.args.seed}_opt_{self.args.optimizer[0]}_lr_{self.args.lr}'))

    def compute_loss(self, x, label, criterion):
        loss = criterion(x, label)
        return loss

    def getLocation(self, feature_map):
        bs, t, f_c, f_h, f_w = feature_map.size()
        gradient = feature_map.grad.view(bs, t, f_c, -1)
        alpha = torch.sum(gradient, dim=-1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand_as(feature_map)
        score_map = alpha * feature_map
        score_map = torch.relu(torch.sum(score_map, dim=2))
        score_map = F.adaptive_avg_pool2d(score_map, (4, 4))
        score_map = score_map.view(bs, t, -1)
        _, topk_idx = score_map.topk(self.K)
        topk_idx = topk_idx.float()
        a_x, a_y = topk_idx % 4, topk_idx//4
        location_label = torch.cat((a_x, a_y), dim=-1).squeeze(0).float()/4
        return location_label

    def getRegions(self, x, locations):
        pass


class TASM(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        self.avg_pool = self.resnet50.avgpool

        # remove avg_pool and fc in resnet50
        self.resnet50 = torch.nn.Sequential(
            *(list(self.resnet50.children())[:-2]))
        self.gru = nn.GRU(2048, hidden_size, num_layers=2, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, label, criterion=None, optimizer=None, update=False):
        # x shape (batch_size, frame, channel, h, w)
        # (bs, frame, channel, h, w) -> (bs * frame, channel, h, w)
        bs, t, c, h, w = x.size()
        x = x.view(bs*t, c, h, w)
        feature_map = self.resnet50(x)
        _, f_c, f_h, f_w = feature_map.size()
        feature_map = feature_map.view(bs, t, f_c, f_h, f_w)
        if update:
            feature_map.retain_grad()
        x = self.avg_pool(feature_map)
        x = x.view(x.size(0), x.size(1), -1)
        x, hidden = self.gru(x)
        x = self.classifier(x[:, -1])

        return x, feature_map


class RAM(nn.Module):
    def __init__(self, s_h, s_w, K):
        super().__init__()
        # groups can implement depth-wise convolution, default is 1.
        self.depth_wise_conv = nn.Conv2d(
            2048, 2048*1, kernel_size=7, groups=2048)
        self.conv = nn.Conv2d(2048, 2*K, kernel_size=1)

    def forward(self, x):
        bs, t, c, h, w = x.size()

        x = x.view(bs*t, c, h, w)
        x = self.depth_wise_conv(x)
        x = self.conv(x)
        return x.view(bs, t, -1)


class SASM(nn.Module):
    def __init__(self,):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)

    def forward(self, x):
        pass
