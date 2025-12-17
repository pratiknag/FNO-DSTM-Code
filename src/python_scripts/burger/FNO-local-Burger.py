#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday January 24 13:33:04 2025

@author: Pratik
"""



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
torch.autograd.set_detect_anomaly(True)
import math
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utilities3 import *
from utils.FNO import *
from utils.utils import exponential_cov,mult_normal

torch.manual_seed(0)
np.random.seed(0)


# class CovarianceModel(nn.Module):
#     def __init__(self, input_dim, output_dim, bandwidth=50):
#         super(CovarianceModel, self).__init__()
#         self.output_dim = output_dim
#         self.bandwidth = bandwidth
#         self.x_ = nn.Parameter(torch.tensor(1e-2))
#         # self.x__ = nn.Parameter(torch.tensor(1e-2))

#         # Precompute banded lower-triangular indices
#         row_idx, col_idx = torch.tril_indices(output_dim, output_dim)
#         band_mask = (row_idx - col_idx) < bandwidth
#         self.row_idx_band = row_idx[band_mask]
#         self.col_idx_band = col_idx[band_mask]
#         self.num_band_elems = len(self.row_idx_band)

#         # Neural net to output only the required banded elements
#         self.cov_factor_net = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.ReLU(),
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.num_band_elems),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         B = x.shape[0]
#         d = self.output_dim

#         tril_elements = torch.exp(self.cov_factor_net(x))  # shape: [B, num_band_elems]
#         L = torch.zeros(B, d, d, device=x.device)
#         det = torch.ones(B, device=x.device)

#         diag_indices = torch.arange(d, device=x.device)

#         for b in range(B):
#             L_b = L[b]
#             tril_flat = tril_elements[b]
#             L_b[self.row_idx_band, self.col_idx_band] = tril_flat
#             # Make diagonals positive
#             L_b[diag_indices, diag_indices] = L_b[diag_indices, diag_indices] + 1e-3
#             det[b] = torch.prod(L_b[diag_indices, diag_indices])**2

#         if torch.isinf(torch.sum(det)):
#             det = 1e8 * torch.ones(B, device=x.device)

#         cov = self.x_ * torch.bmm(L, L.transpose(1, 2))
#         cov += 1e-4 * torch.eye(d, device=x.device).unsqueeze(0)
#         cov_inv = torch.linalg.inv(cov)

#         return cov_inv, det, cov

class CovarianceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CovarianceModel, self).__init__()
        self.output_dim = output_dim
        # self.x_ = nn.Parameter(torch.tensor(1e-2))

        # Neural net to output only the required banded elements
        self.var_net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),  # padding = (kernel_size - 1) // 2 to preserve length
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1), # same here
            nn.Softplus()  # ensures positivity for variance
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        var = self.var_net(x).squeeze(1) # shape: [B, num_band_elems]
        # var += 1e-4 * torch.ones(self.output_dim, device=x.device).unsqueeze(0)
        # print(var.shape)
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

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def main():
    ################################################################
    #  configurations
    ################################################################
    ntrain = 950
    ntest = 50
    train = False
    sub = 1 #subsampling rate

    s = 2**11

    batch_size = 16
    learning_rate = 0.001

    epochs = 70
    # epochs_cov = 50
    step_size = 10
    gamma = 0.5

    modes = 16
    width = 64


    ################################################################
    # Burger data for varying \beta
    ################################################################
    
    data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")

    n = 2048

    # Generate positions (can be 1D spatial points)
    x = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x) # shape: (n,)
    # std = 0.2 * x + 1e-3 * torch.ones_like(x)
    print(std)
    D = torch.diag(std)
    print(D.shape)
    # Construct covariance matrix (2048x2048)
    c = rbf_kernel(x.unsqueeze(1), x.unsqueeze(1))
    # c_inv = torch.linalg.inv(c).cuda()
    # print(c_inv)
    # log_det_c = torch.logdet(c).cuda()
    
    Sigma = D @ c @ D
    # print(Sigma.shape)
    # Add jitter to ensure numerical stability
    Sigma += 1e-4 * torch.eye(n)

    # Sample from multivariate normal
    mean = torch.zeros(n)
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma)
    # samples = mvn.sample((1000,)) 
    # noise_std = 0.01  # you can change the std dev
    x_data= torch.tensor(data[:,:,4],dtype=torch.float)
    # noise = torch.randn_like(x_data) * noise_std
    # x_data = x_data + samples
    y_data = torch.tensor(data[:,:,9],dtype=torch.float)
    # noise = torch.randn_like(y_data) * noise_std
    # torch.manual_seed(23)
    # samples = mvn.sample((1000,)) 
    # y_data = y_data + samples
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    x_train1 = minmax_normalize(x_train.clone())
    x_test1 = minmax_normalize(x_test.clone())
    
    # cat the locations information
    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train1,x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1,x_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    model = Net1d(modes, width).cuda()
    cov_model = CovarianceModel(s,s).cuda()
    print(model.count_params())
    # GP_params = nn.Parameter(torch.exp(torch.randn(2, dtype=torch.float)), requires_grad=True)
    combined_params = [p for p in model.parameters() if p.requires_grad
                       ] + [p for p in cov_model.parameters() if p.requires_grad]
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)
    # coords = torch.linspace(0, 1, s).view(s,1)
    # coords = coords.cuda()
    # myloss = LpLoss(size_average=False)
    print("training started for varying beta --- FNO training phase")
    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        for ep in range(epochs+ 50):
            # if ep > 30:
            #     freeze(model)
            #     model.eval()
            #     cov_model.train()
                # if ep % interval == 0:
                #     unfreeze(model)
                #     model.train()
                # else:
                #     freeze(model)
                #     model.eval()
            # elif ep > 0:
            #     freeze(cov_model)
            #     model.train()
            #     cov_model.eval()
            # else:
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                # mse = 0
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                # print(x1.shape)
                bs = x.shape[0]
                optimizer2.zero_grad()
                out = model(x)
                # print(out.shape)
                # print(y.shape)
                # var = cov_model(x1)
                # var = std
                # D = torch.diag_embed(var)

                # Construct covariance matrix (2048x2048)
                # c = rbf_kernel(x, x)
                # c_inv = torch.linalg.inv(c).cuda()
                # print(c_inv)
                # log_det_c = torch.logdet(c).cuda()
                
                # Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                # Sigma += 1e-4 * torch.eye(n).cuda()

                # inv_var = 1.0 / var                              # Shape: (B, N)
                # D_inv = torch.diag_embed(inv_var)               # Shape: (B, N, N)
                # # print(D_inv)
                # # cov_inv = D_inv @ c_inv @ D_inv
                # # cov_inv = D_inv @ c_inv @ D_inv                 # Shape: (B, N, N)
                # cov_inv = c_inv
                # # print(cov_inv)
                # # log determinant: log|D| = sum(log(diag)) for each batch
                # log_det_D = torch.sum(torch.log(var), dim=1)    # Shape: (B,)
                # det = 2 * log_det_D + log_det_c  
                # # print(det)
                # # Check for inf or nan in cov_inv
                # has_inf_cov = torch.isinf(cov_inv).any()
                # has_nan_cov = torch.isnan(cov_inv).any()

                # # Check for inf or nan in det
                # has_inf_det = torch.isinf(det).any()
                # has_nan_det = torch.isnan(det).any()

                # print(f"cov_inv has inf: {has_inf_cov.item()}, nan: {has_nan_cov.item()}")
                # print(f"det has inf: {has_inf_det.item()}, nan: {has_nan_det.item()}")               # Shape: (B,)
                mse = F.mse_loss(out.view(-1), y.view(-1), reduction='mean')

                # loss = -mult_normal(s, y.reshape(bs,s,1), out.reshape(bs,s,1), det, cov_inv, bs)
                # loss = -log_prob_mult_normal
                # l2.backward()
                # loss = gaussian_nll(y, out, Sigma) 
                # + torch.mean(var)*1e3
                # loss.backward()
                mse.backward()
                optimizer2.step()
                train_mse += mse.item()
                # train_l2 += loss.item()
            
            scheduler2.step()
            model.eval()
            # cov_model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x1, x, y in test_loader:
                    x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                    out = model(x)
                    test_l2 +=  F.mse_loss(out, y, reduction='mean').item()

            train_mse /= len(train_loader)
            test_l2 /= ntest

            t2 = default_timer()
            print(ep, t2-t1, train_mse, test_l2)
            # Early stopping
            if train_mse < best_val_loss:
                best_val_loss = train_mse
                epochs_without_improvement = 0
                # Save the model
                torch.save(model.state_dict(), 'models/burger-FNO-lcl_dat-nonlcl.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 12:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-nonlcl.pth', weights_only=True))
        # freeze(model)
        # model.eval()
        interval = 5
        for ep in range(epochs ):
            if ep % interval == 0:
                    unfreeze(model)
                    model.train()
            else:
                freeze(model)
                model.eval()
            # model.train()
            cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                # mse = 0
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                # print(x1.shape)
                bs = x.shape[0]
                optimizer.zero_grad()
                out = model(x)
                var = cov_model(x1)
                # var = var * torch.ones_like(x1)
                # print(var.shape)
                # var = std
                D = torch.diag_embed(var)
                # print(D)
                # Construct covariance matrix (2048x2048)
                # c = rbf_kernel(x, x)
                # c_inv = torch.linalg.inv(c).cuda()
                # print(c_inv)
                # log_det_c = torch.logdet(c).cuda()
                # print(c)
                Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                Sigma += 1e-4 * torch.eye(n).cuda()
                # print(out.shape)
                # print(y.shape)
                loss = gaussian_nll(y, out, Sigma)
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
            
            scheduler.step()
            # cov_model.eval()
            # test_l2 = 0.0
            # with torch.no_grad():
            #     for x1, x, y in test_loader:
            #         x1, x, y = x1.cuda(), x.cuda(), y.cuda()

            #         out = model(x)
            #         test_l2 +=  F.mse_loss(out, y, reduction='mean').item()

            # train_mse /= len(train_loader)
            # test_l2 /= ntest
            print("Gaussian kernel covariance estimates : {:.6f} {:.6f}".format(
                                        var[0].min().item(), var[0].max().item()))
            t2 = default_timer()
            print(ep, t2-t1, train_l2)
            # Early stopping
            if train_l2 < best_val_loss:
                best_val_loss = train_l2
                epochs_without_improvement = 0
                # Save the model
                torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
                torch.save(model.state_dict(), 'models/burger-FNO-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 30:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        # unfreeze(model)
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-nonlcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-lcl_dat-nonlcl.pth', weights_only=True))
        pred = torch.zeros_like(y_test)
        cov = torch.zeros(ntest,s)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=1, shuffle=False)
        with torch.no_grad():
            for x1, x, y in test_loader:
                
                x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                out = model(x)
                pred[index] = out
                var = cov_model(x1)
                # cov_mat, cov_mat_inv = exponential_cov(coords,
                #                                         coords, GP_params, 1, var)
                cov[index] = var[0]
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index = index + 1
        print("average loss is : {}".format(test_l2/index))
        print("Gaussian kernel covarinace estimates : {}".format(var[0,:10].cpu().detach().numpy()))
        pred = pred.numpy()
        cov = cov.numpy()
        np.save("pred/burger-FNO-lcl_dat-nonlcl.npy",pred)
        np.save("pred/burger-FNO-lcl_dat-nonlcl_cov.npy",cov)
        np.save("pred/burger_test_nonlcl.npy", y_test)

    
        
    else:
    #### Testing phase
        model = Net1d(modes, width).cuda()
        cov_model = CovarianceModel(s,s).cuda()
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-nonlcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-lcl_dat-nonlcl.pth', weights_only=True))
        pred = torch.zeros_like(y_test)
        cov = torch.zeros(ntest,s)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=1, shuffle=False)
        with torch.no_grad():
            for x1, x, y in test_loader:
                
                x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                out = model(x)
                pred[index] = out
                var = cov_model(x1)
                # cov_mat, cov_mat_inv = exponential_cov(coords,
                #                                         coords, GP_params, 1, var)
                cov[index] = var[0]
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index = index + 1
        print("average loss is : {}".format(test_l2/index))
        print("Gaussian kernel covarinace estimates : {}".format(var[0,:10].cpu().detach().numpy()))
        pred = pred.numpy()
        cov = cov.numpy()
        np.save("pred/burger-FNO-lcl_dat-nonlcl.npy",pred)
        np.save("pred/burger-FNO-lcl_dat-nonlcl_cov.npy",cov)
        np.save("pred/burger_test_nonlcl.npy", y_test)
    ################################################################
    # Burger data for fixed \beta
    ################################################################
    
    data = np.load("datasets/generated_1d_data_Burger-04_matern.npy")
    train = True
    n = 2048

    # Generate positions (can be 1D spatial points)
    x = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x) # shape: (n,)
    # std = 0.2 * x + 1e-3 * torch.ones_like(x)
    print(std)
    D = torch.diag(std)
    print(D.shape)
    # Construct covariance matrix (2048x2048)
    c = rbf_kernel(x.unsqueeze(1), x.unsqueeze(1))
    # c_inv = torch.linalg.inv(c).cuda()
    # print(c_inv)
    # log_det_c = torch.logdet(c).cuda()
    
    Sigma = D @ c @ D
    # print(Sigma.shape)
    # Add jitter to ensure numerical stability
    Sigma += 1e-4 * torch.eye(n)

    # Sample from multivariate normal
    mean = torch.zeros(n)
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=Sigma)
    # samples = mvn.sample((1000,)) 
    # noise_std = 0.01  # you can change the std dev
    x_data= torch.tensor(data[:,:,4],dtype=torch.float)
    # noise = torch.randn_like(x_data) * noise_std
    # x_data = x_data + samples
    y_data = torch.tensor(data[:,:,9],dtype=torch.float)
    # noise = torch.randn_like(y_data) * noise_std
    # torch.manual_seed(23)
    # samples = mvn.sample((1000,)) 
    # y_data = y_data + samples
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    x_train1 = minmax_normalize(x_train.clone())
    x_test1 = minmax_normalize(x_test.clone())
    
    # cat the locations information
    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train1,x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1,x_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    model = Net1d(modes, width).cuda()
    cov_model = CovarianceModel(s,s).cuda()
    print(model.count_params())
    # GP_params = nn.Parameter(torch.exp(torch.randn(2, dtype=torch.float)), requires_grad=True)
    combined_params = [p for p in model.parameters() if p.requires_grad
                       ] + [p for p in cov_model.parameters() if p.requires_grad]
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)
    # coords = torch.linspace(0, 1, s).view(s,1)
    # coords = coords.cuda()
    # myloss = LpLoss(size_average=False)
    print("training started for varying beta --- FNO training phase")
    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        for ep in range(epochs+ 50):
            # if ep > 30:
            #     freeze(model)
            #     model.eval()
            #     cov_model.train()
                # if ep % interval == 0:
                #     unfreeze(model)
                #     model.train()
                # else:
                #     freeze(model)
                #     model.eval()
            # elif ep > 0:
            #     freeze(cov_model)
            #     model.train()
            #     cov_model.eval()
            # else:
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                # mse = 0
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                # print(x1.shape)
                bs = x.shape[0]
                optimizer2.zero_grad()
                out = model(x)
                # print(out.shape)
                # print(y.shape)
                # var = cov_model(x1)
                # var = std
                # D = torch.diag_embed(var)

                # Construct covariance matrix (2048x2048)
                # c = rbf_kernel(x, x)
                # c_inv = torch.linalg.inv(c).cuda()
                # print(c_inv)
                # log_det_c = torch.logdet(c).cuda()
                
                # Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                # Sigma += 1e-4 * torch.eye(n).cuda()

                # inv_var = 1.0 / var                              # Shape: (B, N)
                # D_inv = torch.diag_embed(inv_var)               # Shape: (B, N, N)
                # # print(D_inv)
                # # cov_inv = D_inv @ c_inv @ D_inv
                # # cov_inv = D_inv @ c_inv @ D_inv                 # Shape: (B, N, N)
                # cov_inv = c_inv
                # # print(cov_inv)
                # # log determinant: log|D| = sum(log(diag)) for each batch
                # log_det_D = torch.sum(torch.log(var), dim=1)    # Shape: (B,)
                # det = 2 * log_det_D + log_det_c  
                # # print(det)
                # # Check for inf or nan in cov_inv
                # has_inf_cov = torch.isinf(cov_inv).any()
                # has_nan_cov = torch.isnan(cov_inv).any()

                # # Check for inf or nan in det
                # has_inf_det = torch.isinf(det).any()
                # has_nan_det = torch.isnan(det).any()

                # print(f"cov_inv has inf: {has_inf_cov.item()}, nan: {has_nan_cov.item()}")
                # print(f"det has inf: {has_inf_det.item()}, nan: {has_nan_det.item()}")               # Shape: (B,)
                mse = F.mse_loss(out.view(-1), y.view(-1), reduction='mean')

                # loss = -mult_normal(s, y.reshape(bs,s,1), out.reshape(bs,s,1), det, cov_inv, bs)
                # loss = -log_prob_mult_normal
                # l2.backward()
                # loss = gaussian_nll(y, out, Sigma) 
                # + torch.mean(var)*1e3
                # loss.backward()
                mse.backward()
                optimizer2.step()
                train_mse += mse.item()
                # train_l2 += loss.item()
            
            scheduler2.step()
            model.eval()
            # cov_model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for x1, x, y in test_loader:
                    x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                    out = model(x)
                    test_l2 +=  F.mse_loss(out, y, reduction='mean').item()

            train_mse /= len(train_loader)
            test_l2 /= ntest

            t2 = default_timer()
            print(ep, t2-t1, train_mse, test_l2)
            # Early stopping
            if train_mse < best_val_loss:
                best_val_loss = train_mse
                epochs_without_improvement = 0
                # Save the model
                torch.save(model.state_dict(), 'models/burger-FNO-lcl_dat-lcl.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 12:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-lcl.pth', weights_only=True))
        # freeze(model)
        # model.eval()
        interval = 5
        for ep in range(epochs ):
            if ep % interval == 0:
                    unfreeze(model)
                    model.train()
            else:
                freeze(model)
                model.eval()
            # model.train()
            cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                # mse = 0
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                # print(x1.shape)
                bs = x.shape[0]
                optimizer.zero_grad()
                out = model(x)
                var = cov_model(x1)
                # print(var.shape)
                # var = std
                D = torch.diag_embed(var)
                # print(D)
                # Construct covariance matrix (2048x2048)
                # c = rbf_kernel(x, x)
                # c_inv = torch.linalg.inv(c).cuda()
                # print(c_inv)
                # log_det_c = torch.logdet(c).cuda()
                # print(c)
                Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                Sigma += 1e-4 * torch.eye(n).cuda()
                # print(out.shape)
                # print(y.shape)
                loss = gaussian_nll(y, out, Sigma)
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
            
            scheduler.step()
            # cov_model.eval()
            # test_l2 = 0.0
            # with torch.no_grad():
            #     for x1, x, y in test_loader:
            #         x1, x, y = x1.cuda(), x.cuda(), y.cuda()

            #         out = model(x)
            #         test_l2 +=  F.mse_loss(out, y, reduction='mean').item()

            # train_mse /= len(train_loader)
            # test_l2 /= ntest
            print("Gaussian kernel covariance estimates : {:.6f} {:.6f}".format(
                                        var[0].min().item(), var[0].max().item()))
            t2 = default_timer()
            print(ep, t2-t1, train_l2)
            # Early stopping
            if train_l2 < best_val_loss:
                best_val_loss = train_l2
                epochs_without_improvement = 0
                # Save the model
                torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-lcl.pth')
                torch.save(model.state_dict(), 'models/burger-FNO-lcl_dat-lcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 30:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        # unfreeze(model)
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-lcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-lcl_dat-lcl.pth', weights_only=True))
        pred = torch.zeros_like(y_test)
        cov = torch.zeros(ntest,s)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=1, shuffle=False)
        with torch.no_grad():
            for x1, x, y in test_loader:
                
                x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                out = model(x)
                pred[index] = out
                var = cov_model(x1)
                # cov_mat, cov_mat_inv = exponential_cov(coords,
                #                                         coords, GP_params, 1, var)
                cov[index] = var[0]
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index = index + 1
        print("average loss is : {}".format(test_l2/index))
        print("Gaussian kernel covarinace estimates : {}".format(var[0,:10].cpu().detach().numpy()))
        pred = pred.numpy()
        cov = cov.numpy()
        np.save("pred/burger-FNO-lcl_dat-lcl.npy",pred)
        np.save("pred/burger-FNO-lcl_dat-lcl_cov.npy",cov)
        np.save("pred/burger_test_lcl.npy", y_test)


    
        
    else:
    #### Testing phase
        model = Net1d(modes, width).cuda()
        cov_model = CovarianceModel(s,s).cuda()
        model.load_state_dict(torch.load('models/burger-FNO-lcl_dat-lcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-lcl_dat-lcl.pth', weights_only=True))
        pred = torch.zeros_like(y_test)
        cov = torch.zeros(ntest,s,s)
        index = 0
        test_l2 = 0
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=1, shuffle=False)
        with torch.no_grad():
            for x1, x, y in test_loader:
                
                x1, x, y = x1.cuda(), x.cuda(), y.cuda()

                out = model(x)
                pred[index] = out
                cov_inv,det,cov_mat = cov_model(x1)
                # cov_mat, cov_mat_inv = exponential_cov(coords,
                #                                         coords, GP_params, 1, var)
                cov[index] = cov_mat[0]
                test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
                index = index + 1
        print("average loss is : {}".format(test_l2/index))
        print("Gaussian kernel covarinace estimates : {}".format(cov_mat[0,:10,:10].cpu().detach().numpy()))
        pred = pred.numpy()
        cov = cov.numpy()
        np.save("pred/burger-FNO-lcl_dat-lcl.npy",pred)
        np.save("pred/burger-FNO-lcl_dat-lcl_cov.npy",cov)
        np.save("pred/burger_test_lcl.npy", y_test)

if __name__ == '__main__':
    main()

