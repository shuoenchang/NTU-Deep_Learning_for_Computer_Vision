import argparse
import os

import numpy as np
import scipy.misc
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class logging():
    def __init__(self, save_folder):
        self.log_path = save_folder+'/log.txt'
        # with open(self.log_path, 'w') as f:
        # pass

    def write(self, text):
        with open(self.log_path, 'a') as f:
            f.write(text)
        print(text)


def draw_masks(mask, filepath=''):
    output = np.zeros((512, 512, 3), dtype=np.uint8)
    output[mask == 0] = (0, 255, 255)  # (Cyan: 011) Urban land
    output[mask == 1] = (255, 255, 0)  # (Yellow: 110) Agriculture land
    output[mask == 2] = (255, 0, 255)  # (Purple: 101) Rangeland
    output[mask == 3] = (0, 255, 0)  # (Green: 010) Forest land
    output[mask == 4] = (0, 0, 255)  # (Blue: 001) Water
    output[mask == 5] = (255, 255, 255)  # (White: 111) Barren land
    output[mask == 6] = (0, 0, 0)  # (Black: 000) Unknown
    return output


def draw_features(features, targets):
    # ref: https://mortis.tech/2019/11/program_note/664/
    tsne = TSNE(n_components=2, init='pca', verbose=1).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = (tx-min(tx))/(max(tx)-min(tx))
    ty = (ty-min(ty))/(max(ty)-min(ty))

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    color = np.random.rand(50, 3)

    for i in range(50):
        plt.plot(tx[targets==i], ty[targets==i], '.', label=str(i), c=color[i])
    plt.legend(loc='best', ncol=3)
    plt.show()


if __name__ == '__main__':
    features = np.random.randn(2500, 2048*7*7)
    targets = np.random.randint(50, size=2500)
    draw_features(features, targets)
