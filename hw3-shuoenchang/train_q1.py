import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.VAE import VAE, VAE2
from src.dataset import FaceDataset
from src.loss import VAE_loss
import os


import wandb
wandb.init(project="dlcv_3-1")


def train(dataset, model, optimzer, device, criterion):
    model.to(device)
    model.train()
    total_loss = []
    for step, data in enumerate(dataset):
        optimzer.zero_grad()
        image = data['image'].to(device)
        
        output, logvar, mu = model(image)
        loss_reconstruct, loss_KL, loss = criterion(output, image, logvar, mu)
        assert (np.isnan(loss.item()) == False)
        
        wandb.log({"loss_reconstruct": loss_reconstruct}, commit=False)
        wandb.log({"loss_KL": loss_KL}, commit=False)
        wandb.log({"loss": loss})
        total_loss.append(loss.item())
        loss.backward()
        optimzer.step()
        if step%50==0:
            print('step {}: reconstruct {:.3f}, KL {:.3f}, loss {:.3f}'.format(step, loss_reconstruct, loss_KL, loss), end='\r')
    print('train: reconstruct {:.3f}, KL {:.3f}, loss {:.3f}'.format(loss_reconstruct, loss_KL, np.mean(total_loss)))
    return np.mean(total_loss)


def validation(dataset, model, device, criterion):
    model.to(device)
    model.eval()
    total_loss = []
    with torch.no_grad():
        for data in dataset:
            image = data['image'].to(device)
            output, logvar, mu = model(image)
            loss_reconstruct, loss_KL, loss = criterion(output, image, logvar, mu)
            total_loss.append(loss.item())

    print('validation: {}'.format(np.mean(total_loss)))

    return np.mean(total_loss)


def main(args):

    train_set = DataLoader(dataset=FaceDataset('hw3_data/face', mode='train'),
                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_set = DataLoader(dataset=FaceDataset('hw3_data/face', mode='test'),
                           batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VAE(latent_dim=args.latent_dim)
    if args.model=='VAE2':
        model = VAE2(latent_dim=args.latent_dim)
    criterion = VAE_loss(lambda_KL=args.lambda_KL)
    optimzer = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = 'cuda'
    min_loss = 2
    for epoch in range(50):

        print('\nepoch: {}'.format(epoch))

        loss = train(train_set, model, optimzer, device, criterion)
        loss = validation(valid_set, model, device, criterion)
        if loss < min_loss:
            torch.save(model.state_dict(),
                       '{}/lamda_{:.7f}-dim_{}-{}.pth'.format(args.save_folder, args.lambda_KL, args.latent_dim, args.model))
            min_loss = loss
            print('Best epoch: {}'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLCV hw3-1 Training Script')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Choose the device for training')
    parser.add_argument('--save_folder', default='weights/q1',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='learning rate for training')
    parser.add_argument('--latent_dim', default=1024, type=int,
                        help='latent dim for VAE')
    parser.add_argument('--lambda_KL', default=1e-5, type=float)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    wandb.config.update(args) 
    main(args)
