#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday January 24 13:33:04 2025

@author: Pratik
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist



def exponential_cov(x1,x2,params,batch_size, var, is_cuda = True):
    N = x1.shape[0]
    dist = torch.cdist(x1,x2).unsqueeze(-1)
    dist = dist.expand(-1, -1, batch_size)
    cov_mat = torch.exp((-dist)/params[0])
    cov_mat = cov_mat.permute(2,0,1)
    
    cov_mat = cov_mat * var
    if is_cuda:
        nugget = params[1] * torch.eye(N).unsqueeze(-1).expand(-1, -1, batch_size).permute(2,0,1).cuda()
    else:
        nugget = params[1] * torch.eye(N).unsqueeze(-1).expand(-1, -1, batch_size).permute(2,0,1)
    cov_mat = cov_mat + nugget
    cov_mat_inv = torch.linalg.inv(cov_mat)
    return cov_mat, cov_mat_inv

def mult_normal(N,y,m_v, logdet, cov_mat_inv, batch_size, eps=0.00001):
#     y = torch.tensor(y)
    
    c = - 0.5 * np.log(2*np.pi)
    y = y - torch.ones(y.shape).cuda()* m_v
    tr_y = torch.transpose(y, 1, 2)
#     print("######### Y TRANSPOSE "+str(tr_y[0]))
    term1 = torch.matmul(tr_y, cov_mat_inv)
    
    term2 = torch.matmul(term1,y).reshape(batch_size,1)
    # det = torch.det(cov_mat)
    # print(det)
    w = N*c  - 0.5 * term2 - (0.5 * logdet)
#     print("########### "+str(w))
    return torch.sum(w)
    
def ind_normal(mean,std,v1,v2):
    dist1 = torch.distributions.normal.Normal(mean[0], torch.exp(std[0]))
    dist2 = torch.distributions.normal.Normal(mean[1], torch.exp(std[1]))
    return dist1.log_prob(v1) + dist2.log_prob(v2)


class CovarianceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CovarianceModel, self).__init__()
        self.output_dim = output_dim
        # self.x_ = nn.Parameter(torch.tensor(1e-2))

        # Neural net to output only the required banded elements
        self.var_net = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=7, padding=3),  # padding = (kernel_size - 1) // 2 to preserve length
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1), # same here
            nn.Softplus()  # ensures positivity for variance
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        var = self.var_net(x).squeeze(1) # shape: [B, num_band_elems]
        # var += 1e-4 * torch.ones(self.output_dim, device=x.device).unsqueeze(0)
        # print(var.shape)
        return var
    

class CovarianceModel2D(nn.Module):
    def __init__(self):
        super(CovarianceModel2D, self).__init__()
        # self.output_dim = output_dim

        # Neural net to output variance map
        self.var_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),  # preserves spatial size
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # preserves spatial size
            nn.Softplus()  # ensures positive variance
        )

    def forward(self, x):
        # x shape: [B, 3, 64, 64]
        var = self.var_net(x)  # shape: [B, 1, 64, 64]
        var = var.view(x.size(0), -1)  # reshape to [B, 64*64]
        return var
    

def gaussian_nll(y_true, mu, cov):
    dist = MultivariateNormal(mu, covariance_matrix=cov)
    return -dist.log_prob(y_true).mean()


def max_diag_and_offdiag(cov):
    B, d, _ = cov.shape

    # Extract diagonals
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)  # shape: [B, d]
    max_diag = diag.max()

    # Create mask for off-diagonals
    eye = torch.eye(d, device=cov.device).unsqueeze(0)  # shape: [1, d, d]
    off_diag_mask = (1 - eye).bool()

    # Extract off-diagonal values using mask
    off_diag = cov.masked_select(off_diag_mask.expand_as(cov))
    max_off_diag = off_diag.abs().max()  # optionally abs()
      # optionally abs()

    return max_diag.item(), max_off_diag.item()


def minmax_normalize(x, eps=1e-8):
    min_val = x.min(dim=-1, keepdim=True)[0]
    max_val = x.max(dim=-1, keepdim=True)[0]
    return (x - min_val) / (max_val - min_val + eps)

def rbf_kernel(x1, x2, lengthscale=0.01, variance=0.8):
    sqdist = (x1 - x2.T)**2
    return variance * torch.exp(-0.5 * sqdist / lengthscale**2)
    
def rbf_kernel2D(x1, x2, lengthscale=0.05, variance=0.8):
    sqdist = cdist(x1,x2)
    return variance * np.exp(-0.5 * sqdist / lengthscale**2)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

class DeepKrigingModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepKrigingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 50)
        self.fc7 = nn.Linear(50, 50)
        self.fc8 = nn.Linear(50, 50)
        self.fc9 = nn.Linear(50, 1)  # Output layer
        
         # Trainable parameter for x_
        self.x_ = nn.Parameter(torch.tensor(0.5))  # initialize at some value (e.g., 0.5)
        
        # Final output layer for x_
        self.fc_x_ = nn.Linear(50, 1)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x1 = self.fc9(x)
        
        # x_ output (with trainable parameter, transformed via softplus)
        x_ = torch.nn.functional.softplus(self.fc_x_(x)) * self.x_
        
        # Multiply x by 10
        #x = x * 10
        
        # x2 = x1 + x_
        x2 = x1 + x_
        
        # x3 = x1 - x_
        x3 = x1 - x_
        
        return x1,x2,x3 

def tilted_loss1(y, f):
    q=0.5
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum
    
def tilted_loss2(y, f):
    q=0.975
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum
    
def tilted_loss3(y, f):
    q=0.025
    e1 = (y - f)
    the_sum = torch.mean(torch.maximum(q * e1, (q - 1) * e1), dim=-1)
    return the_sum
