import torch
import torch.nn as nn
import torchvision.models as models


class VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 32*32
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 16*16
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 8*8
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),  # 4*4
        )
        self.logvar = nn.Linear(128*16, latent_dim)
        self.mu = nn.Linear(128*16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        logvar = self.logvar(x)
        mu = self.mu(x)

        x = self.sampled(logvar, mu)
        x = self.decoder_input(x)
        x = x.reshape(-1, 128, 4, 4)
        x = self.decoder(x)
        x = self.tanh(x)/2 + 0.5
        return x, logvar, mu

    def sampled(self, logvar, mu):
        std = torch.exp(logvar)**0.5
        e = torch.randn(std.size())
        if torch.cuda.is_available():
            e = e.to('cuda')
        return std*e + mu


class VAE2(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.logvar = nn.Linear(128*16, latent_dim)
        self.mu = nn.Linear(128*16, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 128*16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
        )
        self.tanh = nn.Tanh()
        self.latent_dim = latent_dim

    def forward(self, x, draw=False):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        logvar = self.logvar(x)
        mu = self.mu(x)

        x = self.sampled(logvar, mu)
        latent = torch.cat((logvar, mu),1)
        x = self.decoder_input(x)
        x = x.reshape(-1, 128, 4, 4)
        x = self.decoder(x)
        x = self.tanh(x)/2 + 0.5
        if draw:
            return latent
        else:
            return x, logvar, mu
        
    def sampled(self, logvar, mu):
        std = torch.exp(logvar)**0.5
        e = torch.randn(std.size())
        if torch.cuda.is_available():
            e = e.to('cuda')
        return std*e + mu
    
    def construct(self, num_image):
        logvar = torch.zeros((num_image, self.latent_dim))
        mu = torch.zeros((num_image, self.latent_dim))
        if torch.cuda.is_available():
            logvar = logvar.to('cuda')
            mu = mu.to('cuda')
        x = self.sampled(logvar, mu)
        x = self.decoder_input(x)
        x = x.reshape(-1, 128, 4, 4)
        x = self.decoder(x)
        x = self.tanh(x)/2 + 0.5
        return x
    
    
if __name__ == '__main__':
    model = VAE2().cuda()
    print(model)
    x = torch.rand((1, 3, 64, 64)).cuda()
    y, logvar, mu = model(x)
    print(y.shape)
