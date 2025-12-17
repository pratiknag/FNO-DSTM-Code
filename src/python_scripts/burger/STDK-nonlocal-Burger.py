#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd

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

def rbf_kernel(x1, x2, lengthscale=0.01, variance=0.8):
    sqdist = (x1 - x2.T)**2
    return variance * torch.exp(-0.5 * sqdist / lengthscale**2)    

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

    num_epochs = 200
    patience = 10
    step_size = 50
    gamma = 0.5

    ################################################################
    # Burger data for varying \beta
    ################################################################

    data = np.load("datasets/generated_1d_data_Burger-beta~U<0.05-0.7>_matern.npy")
    phi_train = np.load("datasets/phi_float16_train-burger.npy")
    phi_test  = np.load("datasets/phi_float16_test-burger.npy")
    data = torch.tensor(data ,dtype=torch.float)
    phi_train = torch.tensor(phi_train, dtype=torch.float)
    phi_test = torch.tensor(phi_test, dtype=torch.float)
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
    

    #### coordinates 

    time_points = 5
    space_points = 2**11
    time_pts_test = 1.0
    x = np.linspace(0, 1, space_points)    # shape: (2048,)
    t = np.linspace(0, 1, time_points)     # shape: (time_points,)

    X, Ti = np.meshgrid(x, t, indexing='ij')
    coords = np.stack([X.ravel(), Ti.ravel()], axis=-1)  # shape: (20480, 2)
    t_test = np.array([time_pts_test])
    X_test, T_test = np.meshgrid(x, t_test, indexing='ij')  # shape: (2048, 1)
    coords_test = np.stack([X_test.ravel(), T_test.ravel()], axis=-1)  # shape: (2048, 2)
    phi_train = torch.tensor(coords,dtype=torch.float)
    phi_test = torch.tensor(coords_test,dtype=torch.float)
    y_train = data[ntrain:,:,0:T].reshape(ntest, -1)
    #y_train = data[:ntrain,:,T+1:]
    y_test = data[ntrain:,:,9].reshape(ntest, -1)
    # exit(0);

    pred_all = torch.zeros_like(y_test)
    lb_all = torch.zeros_like(y_test)
    ub_all = torch.zeros_like(y_test)
    mse = 0
    for inst in range(ntest):

        print(f"!!!!!!!!!!!!!!!!!Training for Instance {inst}!!!!!!!!!!!!!!!!!!!!!!")
        # Initialize model
        model = DeepKrigingModel(input_dim=phi_train.shape[1])
        
        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        # Create data loaders for training and testing (batching)
        
        train_dataset = TensorDataset(phi_train, y_train[inst,:])
        test_dataset = TensorDataset(phi_test, y_test[inst,:])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Training loop
        start_time = time.time()
          # Early stopping patience
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        

        directory = 'models/'

        # Check if directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        for epoch in range(num_epochs):
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
                torch.save(model.state_dict(), 'models/model_interpolation-STDK.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            

        end_time = time.time()
        print(f"Training completed in {end_time - start_time} seconds for instance {inst}")

        # Loading the best model
        model.load_state_dict(torch.load('models/model_interpolation-STDK.pth'))

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
    np.save("pred/burger-STDK_med-nonlcl_dat-nonlcl.npy",pred_all)
    np.save("pred/burger-STDK_lb-nonlcl_dat-nonlcl.npy",lb_all)
    np.save("pred/burger-STDK_ub-nonlcl_dat-nonlcl.npy", ub_all)

if __name__ == '__main__':
   main()
    


