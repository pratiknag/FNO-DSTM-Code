#!/usr/bin/env python
# coding: utf-8

"""
Created on Friday May 27 2025

@author: Pratik

ConvLSTM Training Script for SST or Precipitation Data
-------------------------------------------------------

This script trains a convolutional LSTM (ConvLSTM) model on spatio-temporal datasets such as 
Sea Surface Temperature (SST) or Precipitation data.

Command-Line Arguments:
-----------------------
--train       : bool (default=False)
    Specify "--train" to train the model.
    
--epochs      : int (default=20)   
    Number of training epochs to run.

--lr          : float (default=0.0025)   
    Learning rate for the optimizer.

--batch_size  : int (default=20)   
    Batch size for training.

--step_size   : int (default=3)   
    Step size for learning rate scheduler (after how many epochs to decay the LR).

--gamma       : float (default=0.5)   
    Multiplicative factor for learning rate decay in the scheduler.

--data        : str (default="sst")   
    Dataset name. Valid options are "sst" or "precip".
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# torch.autograd.set_detect_anomaly(True)
from timeit import default_timer
import sys
import os
import argparse
# Add the utils directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utilities3 import *
from utils.ConvLSTM import *
from utils.utils import *

torch.manual_seed(0)
np.random.seed(0)

def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM on SST data')
    parser.add_argument('--train', action='store_true', help='Train the model (default: False if not specified)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0025, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--step_size', type=int, default=3, help='Step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='Gamma value.')
    parser.add_argument('--data', type=str, default="sst", help='Dataset name, valid options are sst, precip.')

    args = parser.parse_args()

    # Use the arguments
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    step_size = args.step_size
    gamma = args.gamma
    dat_name = args.data
    S = 64
    train = args.train
    # print(epochs, learning_rate, step_size, gamma)

    if dat_name == "sst":
        t1 = default_timer()
        T_in = 3
        T = 6
        
        ### location vectors 

        x = np.linspace(0,1,S)
        y = np.linspace(0,1,S)
        xv,yv = np.meshgrid(x,y)
        coords = np.hstack((xv.reshape(S*S,1),yv.reshape(S*S,1)))
        # coords = torch.tensor(coords, dtype=torch.float)
        cov = rbf_kernel2D(coords, coords)
        cov = torch.tensor(cov, dtype = torch.float)


        ################################################################
        # load data
        ################################################################

        data_train = np.load("datasets/sst_data-6t-sample.npy")
        data_test = np.load("datasets/sst_data-test-6t.npy")
        #data = np.load("generated_1d_data_Burger_FNO.npy")
        data_train, _ = train_test_split(data_train, train_size=0.99, random_state=42)
        print(data_train.shape)
        ntrain = data_train.shape[0]
        ntest = data_test.shape[0]
        train_a = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float)
        # train_a = train_a.repeat([1,1,1,T_in,1])
        test_a = torch.tensor(data_test[:,:,:,:T_in],dtype=torch.float)
        train_u = torch.tensor(data_train[:,:,:,5],dtype=torch.float)
        test_u = torch.tensor(data_test[:,:,:,5],dtype=torch.float)

        x_train = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
        x_test = torch.tensor(data_test[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
        print(x_train.shape)
        print(test_u.shape)
        

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=batch_size, shuffle=False)

        t2 = default_timer()
    elif dat_name == "precip":
        t1 = default_timer()
        T_in = 3
        T = 6
        
        ### location vectors 

        x = np.linspace(0,1,S)
        y = np.linspace(0,1,S)
        xv,yv = np.meshgrid(x,y)
        coords = np.hstack((xv.reshape(S*S,1),yv.reshape(S*S,1)))
        # coords = torch.tensor(coords, dtype=torch.float)
        cov = rbf_kernel2D(coords, coords)
        cov = torch.tensor(cov, dtype = torch.float)


        ################################################################
        # load data
        ################################################################

        data = np.load("datasets/precipitation_interpolated_data-sample.npy")
        
        data_train, data_test = train_test_split(data, train_size=0.99, random_state=42)
        # ntrain = data_train.shape[0]
        ntest = data_test.shape[0]
        # train_a = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float)
        # train_a = train_a.repeat([1,1,1,T_in,1])
        test_a = torch.tensor(data_test[:,:,:,:T_in],dtype=torch.float)
        train_u = torch.tensor(data_train[:,:,:,5],dtype=torch.float)
        test_u = torch.tensor(data_test[:,:,:,5],dtype=torch.float)

        x_train = torch.tensor(data_train[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
        x_test = torch.tensor(data_test[:,:,:,:T_in],dtype=torch.float).permute(0,3,1,2)
        print(x_train.shape)
        print(test_u.shape)
        

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, train_u), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=batch_size, shuffle=False)

        t2 = default_timer()
    else:
        print("Give valid dataset name !!!!")
        exit(0);
    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    ################################################################
    # training and evaluation
    ################################################################
    model = ConvLSTM2D_Model(T_in,1).cuda()
    cov_model = CovarianceModel2D().cuda()
    # print(model.count_params())
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

    
    best_val_loss = float('inf')
    interval = 5
    torch.manual_seed(329)
    if train:
        print("training started --- ConvLSTM training phase")
        for ep in range(epochs):
            
            model.train()
            # cov_model.train()
            t1 = default_timer()
            train_mse = 0
            train_l2 = 0
            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
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
                for x, y in test_loader:
                    x, y = x.cuda(), y.cuda()

                    out = model(x)
                    test_l2 +=  F.mse_loss(out.view(-1), y.view(-1), reduction='mean').item()

            train_mse /= len(train_loader)
            test_l2 /= ntest

            t2 = default_timer()
            print(ep, t2-t1, train_mse, test_l2)
            # Early stopping
            if train_mse < best_val_loss:
                best_val_loss = train_mse
                epochs_without_improvement = 0
                # Save the model
                torch.save(model.state_dict(), 'models/'+str(dat_name)+'-ConvLSTM.pth')
                # torch.save(cov_model.state_dict(), 'models/burger-cov-lcl_dat-nonlcl.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 6:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        model.load_state_dict(torch.load('models/'+str(dat_name)+'-ConvLSTM.pth', weights_only=True))
        # freeze(model)
        # model.eval()
        interval = 3
        best_val_loss = float('inf')
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
            for x, y in train_loader:
                # mse = 0
                x, y = x.cuda(), y.cuda()
                # print(x1.shape)
                bs = x.shape[0]
                optimizer.zero_grad()
                out = model(x)
                var = cov_model(x)
                D = torch.diag_embed(var)
                Sigma = D @ cov.cuda() @ D
                # Add jitter to ensure numerical stability
                Sigma += 1e-4 * torch.eye(S*S).cuda()

                # std1 = torch.std(x1, dim=1, unbiased=False)
                # print(std1.shape)
                # print(out.shape)
                # print(Sigma.shape)
                loss = gaussian_nll(y.reshape(bs, -1), out.reshape(bs, -1), Sigma) 
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
                torch.save(cov_model.state_dict(), 'models/'+str(dat_name)+'-ConvLSTM-cov.pth')
                torch.save(model.state_dict(), 'models/'+str(dat_name)+'-ConvLSTM.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 10:
                    print(f"Early stopping at epoch {ep+1}")
                    break
        # unfreeze(model)
    else:
        print("Skipped training ... Loading model weights from previously saved model ... ")
        model_path = 'models/'+str(dat_name)+'-ConvLSTM.pth'
        
        if not os.path.exists(model_path):
            print(f"Model file not found at '{model_path}'.")
            print("Please train the model first by running the script with the '--train' flag.")
            sys.exit(1)
    print("Generating predictions ...")
    model.load_state_dict(torch.load('models/'+str(dat_name)+'-ConvLSTM.pth', weights_only=True))
    cov_model.load_state_dict(torch.load('models/'+str(dat_name)+'-ConvLSTM-cov.pth', weights_only=True))
    pred = torch.zeros_like(test_u)
    cov1 = torch.zeros(ntest,S*S)
    index = 0
    test_l2 = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, test_u), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred[index] = out
            var = cov_model(x)
            # cov_mat, cov_mat_inv = exponential_cov(coords,
            #                                         coords, GP_params, 1, var)
            cov1[index] = var[0]
            test_l2 += F.mse_loss(out.view(-1), y.view(-1), reduction='mean')
            index = index + 1
    print("average loss is : {}".format(test_l2/index))
    print("Gaussian kernel covarinace estimates : {}".format(var[0,:10].cpu().detach().numpy()))
    pred = pred.numpy()
    cov1 = cov1.numpy()
    np.save("pred/"+str(dat_name)+"-ConvLSTM-pred.npy",pred)
    np.save("pred/"+str(dat_name)+"-ConvLSTM-cov.npy",cov1)
    np.save("pred/"+str(dat_name)+"-ConvLSTM-test.npy", test_u)


if __name__ == '__main__':
    main()