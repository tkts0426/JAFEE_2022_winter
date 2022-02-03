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

class CustomLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        # initialization
        

    def _decoded_loss(self, unreshaped, y_flat):
        return torch.sum(torch.pow(torch.sub(unreshaped, y_flat), 2), dim = 1)

    def _latent_loss(self, mn, sd):
        return -0.5 * torch.sum(1. + sd - torch.square(mn) - torch.exp(sd), dim = 1)

    def _loss(self, decoded_loss, latent_loss, alpha):
        return torch.mean((1 - alpha) * decoded_loss + alpha * latent_loss)

    def forward(self, mn, sd, outputs, targets, alpha):
        '''
        outputs: predictions
         targets: true label or values
        '''
        
        decoded_loss = self._decoded_loss(outputs, targets)
        latent_loss = self._latent_loss(mn, sd)

        loss = self._loss(decoded_loss, latent_loss, alpha)

        return loss