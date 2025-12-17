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
from utils.utils import *

torch.manual_seed(0)
np.random.seed(0)


def main():
    ################################################################
    #  configurations
    ################################################################

    ntrain = 950
    s = 2**11
    T = 5
    coords = torch.linspace(0, 1, s).view(s,1)
    coords = coords.cuda()
    train = True
    ntest = 50

    batch_size = 20
    learning_rate = 0.001

    epochs = 50
    step_size = 10
    gamma = 0.5

    modes1 = 16
    modes2 = 3
    width = 32
    
    ################################################################
    # Burger data for varying \beta
    ################################################################
    data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
    data = torch.tensor(data ,dtype=torch.float)
    n = 2048

    # Generate positions (can be 1D spatial points)
    x = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x) # shape: (n,)
    # std = 0.25 * x + 1e-2 * torch.ones_like(x)
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
    # samples = mvn.sample((1000,11)).permute(0,2,1)
    
    # data = data + samples
    
    x_train = data[:ntrain,:,0:T].reshape(ntrain, s, T,1)
    #y_train = data[:ntrain,:,T+1:]
    y_train = data[:ntrain,:,9]

    #x_test = data[ntrain:,:,0:T].reshape(ntest, s, 1,1)
    x_test = data[ntrain:,:,0:T].reshape(ntest, s, T,1)
    #y_test = data[ntrain:,:,T+1:]
    y_test = data[ntrain:,:,9]
    
    x_train1 = data[:ntrain,:,0:T].permute(0,2,1)
    x_test1 = data[ntrain:,:,0:T].permute(0,2,1)
    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, s), dtype=torch.float)
    gridx = gridx.reshape(1, s, 1, 1).repeat([1, 1, T,1])
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, T, 1).repeat([1, s, 1, 1])
    x_train = torch.cat([x_train, gridx.repeat([ntrain,1,1,1]), 
                           gridt.repeat([ntrain,1,1,1]) ], dim=-1)
    x_test = torch.cat([x_test, gridx.repeat([ntest,1,1,1]), 
                           gridt.repeat([ntest,1,1,1])], dim=-1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train1,x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=batch_size, shuffle=False)

    
    model = Net2d(modes1,modes2, width).cuda()
    cov_model = CovarianceModel(s,s).cuda()
    print(model.count_params())
    # GP_params = nn.Parameter(torch.tensor([0.4, 0.5, 0.000001], dtype=torch.float), requires_grad=True)
        
    combined_params = [p for p in model.parameters() if p.requires_grad
                       ] + [p for p in cov_model.parameters() if p.requires_grad]
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

    print("training started for varying beta --- FNO training phase")
    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        for ep in range(epochs + 50):
            
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                bs = x.shape[0]
                optimizer2.zero_grad()
                out = model(x)
                
                mse = F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
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
                torch.save(model.state_dict(), 'models/burger-FNO-nonlcl_dat-nonlcl.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 12:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/burger-FNO-nonlcl_dat-nonlcl.pth', weights_only=True))
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
                D = torch.diag_embed(var)
                Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                Sigma += 1e-4 * torch.eye(n).cuda()

                # std1 = torch.std(x1, dim=1, unbiased=False)
                # print(std1.shape)
                loss = gaussian_nll(y, out, Sigma) 
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
            
            scheduler.step()
            print("Gaussian kernel covariance estimates : {:.6f} {:.6f}".format(
                                        var[0].min().item(), var[0].max().item()))
            t2 = default_timer()
            print(ep, t2-t1, train_l2)
            # Early stopping
            if train_l2 < best_val_loss:
                best_val_loss = train_l2
                epochs_without_improvement = 0
                # Save the model
                torch.save(cov_model.state_dict(), 'models/burger-cov-nonlcl_dat-nonlcl.pth')
                torch.save(model.state_dict(), 'models/burger-FNO-nonlcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 30:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        # unfreeze(model)
        model.load_state_dict(torch.load('models/burger-FNO-nonlcl_dat-nonlcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-nonlcl_dat-nonlcl.pth', weights_only=True))
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
        np.save("pred/burger-FNO-nonlcl_dat-nonlcl.npy",pred)
        np.save("pred/burger-FNO-nonlcl_dat-nonlcl_cov.npy",cov)
        np.save("pred/burger_test_nonlcl-FNO-nonlcl.npy", y_test)
    ################################################################
    # Burger data for fixed \beta
    ################################################################
    
    data = np.load("datasets/generated_1d_data_Burger-04_matern.npy")
    data = torch.tensor(data ,dtype=torch.float)
    n = 2048

    # Generate positions (can be 1D spatial points)
    x = torch.linspace(0, 1, n)

    # RBF Kernel function
    std = 0.08 * torch.ones_like(x) # shape: (n,)
    # std = 0.25 * x + 1e-2 * torch.ones_like(x)
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
    # samples = mvn.sample((1000,11)).permute(0,2,1)
    # print(data.shape)
    # print(samples.shape)
    # data = data + samples
    
    x_train = data[:ntrain,:,0:T].reshape(ntrain, s, T,1)
    #y_train = data[:ntrain,:,T+1:]
    y_train = data[:ntrain,:,9]

    #x_test = data[ntrain:,:,0:T].reshape(ntest, s, 1,1)
    x_test = data[ntrain:,:,0:T].reshape(ntest, s, T,1)
    #y_test = data[ntrain:,:,T+1:]
    y_test = data[ntrain:,:,9]
    
    x_train1 = data[:ntrain,:,0:T].permute(0,2,1)
    x_test1 = data[ntrain:,:,0:T].permute(0,2,1)
    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, s), dtype=torch.float)
    gridx = gridx.reshape(1, s, 1, 1).repeat([1, 1, T,1])
    gridt = torch.tensor(np.linspace(0, 1, T), dtype=torch.float)
    gridt = gridt.reshape(1, 1, T, 1).repeat([1, s, 1, 1])
    x_train = torch.cat([x_train, gridx.repeat([ntrain,1,1,1]), 
                           gridt.repeat([ntrain,1,1,1]) ], dim=-1)
    x_test = torch.cat([x_test, gridx.repeat([ntest,1,1,1]), 
                           gridt.repeat([ntest,1,1,1])], dim=-1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train1,x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test1, x_test, y_test), batch_size=batch_size, shuffle=False)

    
    model = Net2d(modes1,modes2, width).cuda()
    cov_model = CovarianceModel(s,s).cuda()
    print(model.count_params())
    # GP_params = nn.Parameter(torch.tensor([0.4, 0.5, 0.000001], dtype=torch.float), requires_grad=True)
        
    combined_params = [p for p in model.parameters() if p.requires_grad
                       ] + [p for p in cov_model.parameters() if p.requires_grad]
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(combined_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

    print("training started for fixed beta --- FNO training phase")
    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        for ep in range(epochs + 50):
            
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x1,x, y in train_loader:
                x1,x, y =x1.cuda(), x.cuda(), y.cuda()
                bs = x.shape[0]
                optimizer2.zero_grad()
                out = model(x)
                
                mse = F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
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
                torch.save(model.state_dict(), 'models/burger-FNO-nonlcl_dat-lcl.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 12:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/burger-FNO-nonlcl_dat-lcl.pth', weights_only=True))
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
                D = torch.diag_embed(var)
                Sigma = D @ c.cuda() @ D
                # Add jitter to ensure numerical stability
                Sigma += 1e-4 * torch.eye(n).cuda()

                # std1 = torch.std(x1, dim=1, unbiased=False)
                # print(std1.shape)
                loss = gaussian_nll(y, out, Sigma) 
                loss.backward()
                optimizer.step()
                train_l2 += loss.item()
            
            scheduler.step()
            print("Gaussian kernel covariance estimates : {:.6f} {:.6f}".format(
                                        var[0].min().item(), var[0].max().item()))
            t2 = default_timer()
            print(ep, t2-t1, train_l2)
            # Early stopping
            if train_l2 < best_val_loss:
                best_val_loss = train_l2
                epochs_without_improvement = 0
                # Save the model
                torch.save(cov_model.state_dict(), 'models/burger-cov-nonlcl_dat-lcl.pth')
                torch.save(model.state_dict(), 'models/burger-FNO-nonlcl_dat-lcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 30:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        # unfreeze(model)
        model.load_state_dict(torch.load('models/burger-FNO-nonlcl_dat-lcl.pth', weights_only=True))
        cov_model.load_state_dict(torch.load('models/burger-cov-nonlcl_dat-lcl.pth', weights_only=True))
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
        np.save("pred/burger-FNO-nonlcl_dat-lcl.npy",pred)
        np.save("pred/burger-FNO-nonlcl_dat-lcl_cov.npy",cov)
        np.save("pred/burger_test_nonlcl-FNO-lcl.npy", y_test)
    
    
    
if __name__ == '__main__':
    main()
