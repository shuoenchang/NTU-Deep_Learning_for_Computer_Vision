import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

import wandb
from models.GAN import Discriminator, Generator
from src.dataset import FaceDataset

wandb.init(project="dlcv_3-2")


def train(dataset, G, D, optimzer_G, optimzer_D, device, latent_dim, n_critic):
    G.train()
    D.train()

    for step, data in enumerate(dataset):

        # Train Discriminator
        optimzer_G.zero_grad()
        optimzer_D.zero_grad()

        real_image = data['image'].to(device)
        latent = torch.randn((len(real_image), latent_dim, 1, 1)).to(device)
        fake_image = G(latent).detach()

        loss_real = torch.mean(D(real_image))
        loss_fake = torch.mean(D(fake_image))
        loss_D = -loss_real+loss_fake
        loss_D.backward()
        optimzer_D.step()
        D.weight_cliping()

        # Train Generator
        if step % n_critic == 0:
            optimzer_G.zero_grad()
            optimzer_D.zero_grad()

            latent = torch.randn(
                (len(real_image), latent_dim, 1, 1)).to(device)
            fake_image = G(latent)
            loss_G = -torch.mean(D(fake_image))
            loss_G.backward()
            optimzer_G.step()

            wandb.log({"loss_real": loss_real}, commit=False)
            wandb.log({"loss_fake": loss_fake}, commit=False)
            wandb.log({"loss_D": loss_D}, commit=False)
            wandb.log({"loss_G": loss_G})
            torchvision.utils.save_image(
                fake_image.cpu().data, 'outputs/q2.png', nrow=8)
            print('step {}: loss_real {:.3f}, loss_fake {:.3f}, loss_D {:.3f}, loss_G {:.3f}'.format(
                step, loss_real, loss_fake, loss_D, loss_G), end='\r')
    print()


def val(G, device, epoch):
    latent = torch.randn((32, latent_dim, 1, 1)).to(device)
    fake_image = G(latent)
    torchvision.utils.save_image(
        fake_image.cpu().data, 'outputs/q2/{:02d}.png'.format(epoch), nrow=8)


def main(args):

    train_set = DataLoader(dataset=FaceDataset('hw3_data/face', mode='train'),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_set = DataLoader(dataset=FaceDataset('hw3_data/face', mode='test'),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    G = Generator(args.latent_dim).to(args.device)
    D = Discriminator(args.weight_cliping_limit).to(args.device)
    optimzer_G = optim.RMSprop(G.parameters(), lr=args.learning_rate)
    optimzer_D = optim.RMSprop(D.parameters(), lr=args.learning_rate)

    min_loss = 2
    for epoch in range(100):

        print('\nepoch: {}'.format(epoch))
        train(train_set, G, D, optimzer_G, optimzer_D,
              args.device, args.latent_dim, args.n_critic)
        val(G, args.device, epoch)

        torch.save(G.state_dict(), '{}/G_{:02d}.pth'.format(args.save_folder, epoch))  # noqa
        torch.save(D.state_dict(), '{}/D_{:02d}.pth'.format(args.save_folder, epoch))  # noqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw3-2 Training Script')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q2',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='learning rate for training')
    parser.add_argument('--latent_dim', default=1024, type=int,
                        help='latent dim for GAN')
    parser.add_argument('--weight_cliping_limit', default=0.01, type=float)
    parser.add_argument('--n_critic', default=5, type=int)
    args = parser.parse_args()
    wandb.config.update(args)
    main(args)
