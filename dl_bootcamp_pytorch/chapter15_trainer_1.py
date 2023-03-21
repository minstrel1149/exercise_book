import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from torchvision import datasets, transforms

def train(self, train_data, valid_data, config):
    lowest_loss = np.inf
    best_model = None

    for epoch_index in range(config.n_epochs):
        train_loss = self._train(train_data[0], train_data[1], config)
        valid_loss = self._validate(valid_data[0], valid_data[1], config)

        if valid_loss <= lowest_loss:
            lowest_loss = valid_loss
            best_model = deepcopy(self.model.state_dict())
        
        print(f'Epoch {epoch_index + 1}/{config.n_epochs}: train loss={train_loss:.4e}, valid loss={valid_loss:.4e}, lowest loss={lowest_loss:.4e}')
    
    self.model.load_state_dict(best_model)