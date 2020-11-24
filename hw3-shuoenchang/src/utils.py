import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1+np.exp(-10.*p)) - 1.

def class_to_onehot(y, device):
    y = y.unsqueeze(1)
    onehot = torch.zeros(len(y), 11).to(device)
    return onehot.scatter_(1, y, 1)

def draw_features(features, targets, n_classes, domains=None):
    tsne = TSNE(n_components=2, init='pca', verbose=1).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = (tx-min(tx))/(max(tx)-min(tx))
    ty = (ty-min(ty))/(max(ty)-min(ty))

    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # color = np.random.rand(n_classes, 3)
    color = ['r', 'g', 'b', 'k', 'gold', 'm', 'c', 'orange', 'cyan', 'pink']
    domain_label = ['source', 'target']
    for i in range(n_classes):
        plt.plot(tx[targets==i], ty[targets==i], '.', label=str(i), c=color[i])
    plt.legend(loc='best', ncol=2)
    plt.savefig("/home/en/SSD/DLCV/hw3-shuoenchang/figures/fig4_2_usps_label_source.png")
    plt.show()
    
    for i in range(2):
        plt.plot(tx[domains==i], ty[domains==i], '.', label=domain_label[i], c=color[i])
    plt.legend(loc='best', ncol=2)
    plt.savefig("/home/en/SSD/DLCV/hw3-shuoenchang/figures/fig4_2_usps_domain_source.png")
    plt.show()