#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday May  30 2025

@author: Pratik

STDK Training Script for SST or Precipitation Data
--------------------------------------------------

This script trains a Space-Time DeepKriging (STDK) model on spatio-temporal datasets such as 
Sea Surface Temperature (SST) or Precipitation data.

Command-Line Arguments:
-----------------------
--epochs      : int (default=100)   
    Number of training epochs to run.

--lr          : float (default=0.0025)   
    Learning rate for the optimizer.

--batch_size  : int (default=20)   
    Batch size for training.

--step_size   : int (default=3)   
    Step size for the learning rate scheduler, indicating how many epochs between each decay.

--gamma       : float (default=0.5)   
    Multiplicative factor of learning rate decay applied at each step.

--data        : str (default="sst")   
    Dataset name to be used. Valid options are "sst" (Sea Surface Temperature) and "precip" (Precipitation).
"""

import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import *


def main():

    ################################################################
    #  configurations
    ################################################################
    parser = argparse.ArgumentParser(description='Train STDK on SST data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
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

    T = 3

    num_epochs = 200
    patience = 10
    step_size = 50
    gamma = 0.5

    ################################################################
    # Load datasets #######################
    ################################################################
    if dat_name == "sst":
        data = np.load("datasets/sst_data-6t-sample.npy")
        # phi_train = np.load("datasets/phi_float16_train-burger.npy")
        # phi_test  = np.load("datasets/phi_float16_test-burger.npy")
        data = torch.tensor(data ,dtype=torch.float)
        # phi_train = torch.tensor(phi_train, dtype=torch.float)
        # phi_test = torch.tensor(phi_test, dtype=torch.float)
        
        ntest = data.shape[0]
        print(ntest)
        #### coordinates 

        time_points = 3
        space_points = 64
        time_pts_test = 1.0
        x1 = np.linspace(0, 1, space_points)    # shape: (64,)
        x2 = np.linspace(0, 1, space_points)    # shape: (64,)
        t = np.linspace(0, 1, time_points)     # shape: (time_points,)

        X1,X2, Ti = np.meshgrid(x1,x2, t, indexing='ij')
        coords = np.stack([X1.ravel(),X2.ravel(), Ti.ravel()], axis=-1)  # shape: (20480, 2)
        t_test = np.array([time_pts_test])
        X1_test, X2_test, T_test = np.meshgrid(x1, x2, t_test, indexing='ij')  # shape: (2048, 1)
        coords_test = np.stack([X1_test.ravel(), X2_test.ravel(), T_test.ravel()], axis=-1)  # shape: (2048, 2)
        phi_train = torch.tensor(coords,dtype=torch.float)
        phi_test = torch.tensor(coords_test,dtype=torch.float)
        y_train = data[:,:,:,0:T].reshape(ntest, -1)
        #y_train = data[:ntrain,:,T+1:]
        y_test = data[:,:,:,5].reshape(ntest, -1)
    elif dat_name == "precip":
        data = np.load("datasets/precipitation_interpolated_data-sample.npy")
        _, data = train_test_split(data, train_size=0.01, random_state=42)
        # phi_train = np.load("datasets/phi_float16_train-burger.npy")
        # phi_test  = np.load("datasets/phi_float16_test-burger.npy")
        data = torch.tensor(data ,dtype=torch.float)
        # phi_train = torch.tensor(phi_train, dtype=torch.float)
        # phi_test = torch.tensor(phi_test, dtype=torch.float)
        
        ntest = data.shape[0]
        print(ntest)
        #### coordinates 

        time_points = 3
        space_points = 64
        time_pts_test = 1.0
        x1 = np.linspace(0, 1, space_points)    # shape: (64,)
        x2 = np.linspace(0, 1, space_points)    # shape: (64,)
        t = np.linspace(0, 1, time_points)     # shape: (time_points,)

        X1,X2, Ti = np.meshgrid(x1,x2, t, indexing='ij')
        coords = np.stack([X1.ravel(),X2.ravel(), Ti.ravel()], axis=-1)  # shape: (20480, 2)
        t_test = np.array([time_pts_test])
        X1_test, X2_test, T_test = np.meshgrid(x1, x2, t_test, indexing='ij')  # shape: (2048, 1)
        coords_test = np.stack([X1_test.ravel(), X2_test.ravel(), T_test.ravel()], axis=-1)  # shape: (2048, 2)
        phi_train = torch.tensor(coords,dtype=torch.float)
        phi_test = torch.tensor(coords_test,dtype=torch.float)
        y_train = data[:,:,:,0:T].reshape(ntest, -1)
        #y_train = data[:ntrain,:,T+1:]
        y_test = data[:,:,:,5].reshape(ntest, -1)
    else:
        print("Give valid dataset name !!!!")
        exit(0);
    # exit(0);

    pred_all = torch.zeros_like(y_test)
    lb_all = torch.zeros_like(y_test)
    ub_all = torch.zeros_like(y_test)
    np.save("pred/"+str(dat_name)+"-STDK_testy.npy", y_test)
    # exit(0);
    mse = 0
    for inst in range(ntest):

        print(f"!!!!!!!!!!!!!!!!!Training for Instance {inst}!!!!!!!!!!!!!!!!!!!!!!")
        # Initialize model
        model = DeepKrigingModel(input_dim=phi_train.shape[1])
        
        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # Create data loaders for training and testing (batching)
        
        train_dataset = TensorDataset(phi_train, y_train[inst,:])
        test_dataset = TensorDataset(phi_test, y_test[inst,:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Training loop
        start_time = time.time()
          # Early stopping patience
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        

        directory = 'models/'

        # Check if directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        for epoch in range(epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                x1, x2, x3 = model(inputs)
                # print(x1.shape)
                # print(targets.shape)
                # Compute loss
                loss = tilted_loss1(targets, x1.view(-1)) + tilted_loss2(targets,x2.view(-1)) + tilted_loss3(targets,x3.view(-1))
                #print(loss.shape)
                # Backward pass and optimize
                loss.sum().backward()
                optimizer.step()
                
                running_loss += loss.sum().item()
            scheduler.step()
            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    x1, x2, x3 = model(inputs)
                    loss = tilted_loss1(targets, x1.view(-1)) + tilted_loss2(targets,x2.view(-1)) + tilted_loss3(targets,x3.view(-1))
                    val_loss += loss.sum().item()

            # Print training and validation loss
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(test_loader)}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the model
                torch.save(model.state_dict(), 'models/model_interpolation-STDK-'+str(dat_name)+'.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            

        end_time = time.time()
        print(f"Training completed in {end_time - start_time} seconds for instance {inst}")

        # Loading the best model
        model.load_state_dict(torch.load('models/model_interpolation-STDK-'+str(dat_name)+'.pth'))

        # Make predictions
        model.eval()
        y_pred = torch.zeros_like(y_test[inst,:])
        lb_pred = torch.zeros_like(y_test[inst,:])
        ub_pred = torch.zeros_like(y_test[inst,:])
        index = 0
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for inputs, targets in test_loader:
                x1,x2,x3 = model(inputs)
                y_pred[index] = x1[0]
                ub_pred[index] = x2[0]
                lb_pred[index] = x3[0]
                index += 1

        pred_all[inst,:] = y_pred
        ub_all[inst,:] = ub_pred
        lb_all[inst,:] = lb_pred
        # Evaluate model performance (e.g., Mean Squared Error)
        mse += F.mse_loss(y_pred, y_test[inst,:], reduction='mean')
        
    print(f"Mean Squared Error: {mse/ntest}")
    np.save("pred/"+str(dat_name)+"-STDK_med.npy",pred_all)
    np.save("pred/"+str(dat_name)+"-STDK_lb.npy",lb_all)
    np.save("pred/"+str(dat_name)+"-STDK_ub.npy", ub_all)
    np.save("pred/"+str(dat_name)+"-coords-STDK.npy", coords_test)
    


if __name__ == '__main__':
   main()
    