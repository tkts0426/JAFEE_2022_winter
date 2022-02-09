import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn import utils

class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, input_dim_target,  n_latent, n_hidden=50, alpha=0.2):

        super().__init__()
        
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.input_dim_target = input_dim_target
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.alpha = alpha

        # making encode layers
        self.dense_enc0 = nn.Flatten()
        self.dense_enc1 = nn.Linear(self.cond_dim + self.input_dim, self.n_hidden)
        self.dense_enc_mn = nn.Linear(self.n_hidden, self.n_latent)
        self.dense_enc_sd = nn.Linear(self.n_hidden, self.n_latent)

        # making decode layers
        self.dense_dec0 = nn.Linear(self.cond_dim + self.n_latent, self.n_hidden, bias = False)
        self.dense_dec1 = nn.Linear(self.n_hidden, self.input_dim_target, bias = False)

    def lrelu(self, x, alpha=0.3):
        return torch.max(x, torch.mul(x, alpha))

    def _decoded_loss(self, unreshaped, y_flat):
        return torch.sum(torch.pow(torch.sub(unreshaped, y_flat), 2), dim = 1)

    def _latent_loss(self, mn, sd):
        return -0.5 * torch.sum(1. + sd - torch.square(mn) - torch.exp(sd), dim = 1)

    def _loss(self, decoded_loss, latent_loss):
        return torch.mean((1 - self.alpha) * decoded_loss + self.alpha * latent_loss)

    def encoder(self, X_in, cond):
        
        x = torch.cat([X_in, cond], 1)
        x = self.dense_enc0(x).float()
        x = self.lrelu(self.dense_enc1(x))

        mn = self.lrelu(self.dense_enc_mn(x))
        sd = self.lrelu(self.dense_enc_sd(x))
        
        epsilon = torch.randn(x.shape[0], self.n_latent, dtype = torch.float64 )
        # reparametrize
        sample_z = mn + torch.mul(epsilon, torch.exp(sd / 2.))
        return sample_z, mn, sd

    def decoder(self, sample_z, cond):

        x = torch.cat([sample_z, cond], 1).float()
        x = self.lrelu(self.dense_dec0(x))
        x = torch.sigmoid(self.dense_dec1(x))

        return x
    
    def forward(self, X_in, cond):
        
        sample_z, mn, sd = self.encoder(X_in, cond)
        decode = self.decoder(sample_z, cond)

        return mn, sd, sample_z, decode


    def generate(self, cond, n_samples=None):
        if n_samples is not None:
            randoms = []
            for i in range(n_samples):
                random = np.random.normal(0, 1, size=(1, self.n_latent))
                randoms.append(random)
            randoms = torch.Tensor(randoms).reshape(-1, self.n_latent)
            cond = torch.Tensor(cond.float())            
        else:
            randoms = np.random.normal(0, 1, size=(1, self.n_latent))
            cond = torch.Tensor(cond.float())
        
        randoms = torch.from_numpy(randoms.astype(np.float64)).clone()
        samples = self.decoder(randoms, cond)

        return samples

    # ここは直接セルに打ち込んでいってもいいかもしれない
    # def train(self, data, data_cond, n_epochs=10000, learning_rate=0.005, show_progress=False):
        
        
    #     data = utils.as_float_array(data)
    #     data_cond = utils.as_float_array(data_cond)

    #     if len(data_cond.shape) == 1:
    #         data_cond = data_cond.reshape(-1, 1)

    #     assert data.max() <= 1. and data.min() >= 0., "All features of the dataset must be between 0 and 1."

    #     input_dim = data.shape[1]
    #     dim_cond = data_cond.shape[1]

    #     # device = torch.device("cuda:0") // GPU使う際に必要
    #     net = self.
    #     # net = net.to(device) // GPU使う際に必要
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

    #     for i in tqdm(range(n_epochs), desc="Training"):
    #         for input_, target in dataloader 


        
        
        
        