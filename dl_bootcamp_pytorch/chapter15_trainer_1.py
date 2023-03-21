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

def _train(self, X, y, config):
    self.model.train()

    X, y = self._batchify(X, y, config.batch_size)
    total_loss = 0

    for i, (X_i, y_i) in enumerate(zip(X, y)):
        y_hat_i = self.model(X_i)
        loss_i = self.crit(y_hat_i, y_i.squeeze())

        self.optimizer.zero_grad()
        loss_i.backward()
        self.optimizer.step()

        if config.verbose >= 0.2:
            print(f'Train iteration({i + 1}{len(X)}: loss={float(loss_i):.4e})')
        
        total_loss += float(loss_i)
    
    return total_loss / len(X)

def _batchify(self, X, y, batch_size, random_split=True):
    if random_split:
        indices = torch.randperm(X.size(0), device=X.device)
        X = torch.index_select(X, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)
    
    X = X.split(batch_size, dim=0)
    y = y.split(batch_size, dim=0)

    return X, y

def _validate(self, X, y, config):
    self.model.eval()

    with torch.no_grad():
        X, y = self.batchify(X, y, config.batch_size, random_split=False)
        total_loss = 0

        for i, (X_i, y_i) in enumerate(zip(X, y)):
            y_hat_i = self.model(X_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            if config.verbose >= 2:
                print(f'Valid iteration({i + 1}{len(X)}: loss={float(loss_i):.4e})')
            
            total_loss += float(loss_i)
        
    return total_loss / len(X)
