import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST('../../data', train=is_train, transform=transforms.Compose([transforms.ToTensor()]))

    X = dataset.data.float() / 255
    y = dataset.targets

    if flatten:
        X = X.reshape(X.size(0), -1)
    
    return X, y

def split_data(X, y, train_ratio=0.8):
    train_cnt = int(X.size(0) * train_ratio)
    valid_cnt = X.size(0) - train_cnt

    indices = torch.randperm(X.size(0))
    X = torch.index_select(X, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)

    return X, y